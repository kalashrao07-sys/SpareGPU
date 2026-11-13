"""SpareGPU / two-portal tests: shared rules, E1 constraints HC5-HC7, E2 ledger and commit re-checks, exchange pages."""
import os, sys, tempfile, time, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
config.JOBLOG_DIR = tempfile.mkdtemp()
import rules
from e1_scheduling.selection import select_gpu
from e1_scheduling.scheduler import run_pass
from e2_persistence import AssignmentRejected, Repository
from e4_web import auth, create_app

H = 3600
NOW = time.mktime(time.strptime("2026-11-06 23:00", "%Y-%m-%d %H:%M"))


class RulesTests(unittest.TestCase):
    def test_window(self):
        self.assertTrue(rules.fits_window(None, None, NOW, NOW + 10 * H))
        self.assertTrue(rules.fits_window(22 * 60, 7 * 60, NOW, NOW + 6 * H))      # crosses midnight
        self.assertFalse(rules.fits_window(22 * 60, 7 * 60, NOW, NOW + 10 * H))
        self.assertFalse(rules.fits_window(9 * 60, 17 * 60, NOW, NOW + H))

    def test_rate_and_privacy(self):
        gpu = {"provider_id": 1, "owner_user_id": 5, "price_per_hour": 4, "private_only": 1}
        self.assertEqual(rules.effective_rate(gpu, {"user_id": 5, "trusted_providers": set()}), 0)
        self.assertEqual(rules.effective_rate(gpu, {"user_id": 6, "trusted_providers": {1}}), 0)
        self.assertEqual(rules.effective_rate(gpu, {"user_id": 7, "trusted_providers": set()}), 4)
        self.assertFalse(rules.may_use(gpu, {"user_id": 7, "trusted_providers": set()}))
        self.assertTrue(rules.may_use(gpu, {"user_id": 6, "trusted_providers": {1}}))


def g(i, price=0, **kw):
    d = {"gpu_id": i, "gpu_type": "RTX", "memory_gb": 8, "current_job_id": None, "free_since": NOW - 100,
         "provider_id": i, "owner_user_id": 99, "price_per_hour": price, "accepts_scripts": 1}
    d.update(kw)
    return d


JOB = {"user_id": 1, "memory_gb": 2, "duration_min": 60, "kind": "template", "trusted_providers": set(), "available_credits": 100}


class SelectionTests(unittest.TestCase):
    def test_cheapest_wins_before_idle_time(self):
        d = select_gpu(JOB, [g(1, 5, free_since=NOW - 1000), g(2, 2, free_since=NOW - 1)], NOW)
        self.assertEqual((d.gpu_id, d.rule), (2, "lowest price"))

    def test_equal_price_falls_back_to_sp2(self):
        d = select_gpu(JOB, [g(1, free_since=NOW - 1000), g(2, free_since=NOW - 1)], NOW)
        self.assertEqual((d.gpu_id, d.rule), (2, "SP2 idle time"))

    def test_holds(self):
        config.CREDITS_ENABLED = True
        self.addCleanup(setattr, config, "CREDITS_ENABLED", False)
        self.assertIn("private", select_gpu(JOB, [g(1, private_only=1)], NOW).reason)
        self.assertIn("hours", select_gpu(JOB, [g(1, avail_from=9 * 60, avail_to=17 * 60)], NOW).reason)
        self.assertIn("maximum price", select_gpu({**JOB, "max_price_per_hour": 1}, [g(1, 3)], NOW).reason)
        self.assertIn("credits", select_gpu({**JOB, "available_credits": 1}, [g(1, 3)], NOW).reason)
        self.assertEqual(select_gpu({**JOB, "trusted_providers": {1}, "available_credits": 0}, [g(1, 3, private_only=1)], NOW).gpu_id, 1)


class EconomyTests(unittest.TestCase):
    def setUp(self):
        config.CREDITS_ENABLED = True
        self.addCleanup(setattr, config, "CREDITS_ENABLED", False)
        self.tmp = tempfile.mkdtemp()
        config.JOBLOG_DIR = os.path.join(self.tmp, "logs")
        self.repo = Repository(os.path.join(self.tmp, "t.db")); self.repo.init()
        self.lender = self.repo.register_user("lena", "password1", 0)
        self.renter = self.repo.register_user("rex", "password1", 100)
        self.pid, _ = self.repo.create_provider(self.lender, "lena-pc", approved=1, price_per_hour=6, accepts_scripts=1)
        self.repo.sync_provider_gpus(self.pid, [{"device_index": 0, "name": "RTX 3050", "gpu_type": "RTX3050", "memory_gb": 6}], now=NOW)
        self.repo.touch_provider(self.pid, now=NOW)

    def job(self, **kw):
        base = dict(template="matmul", memory_gb=1, duration_min=60, priority=3)
        base.update(kw)
        return self.repo.submit_job(self.renter, now=NOW, **base)

    def test_run_and_settlement(self):
        j = self.job()
        run_pass(self.repo, NOW)
        a = self.repo.get_job(j)
        self.assertEqual(a["state"], "running")
        self.assertEqual(self.repo.reserved(self.renter), 6)
        self.repo.complete_job(j, "done", now=a["start_time"] + 1800)       # half the booked hour used
        self.assertEqual(self.repo.balance(self.renter), 97)
        self.assertEqual(self.repo.balance(self.lender), 3)
        self.assertEqual(self.repo.get_job(j)["cost"], 3)

    def test_too_expensive_and_poor_jobs_are_held(self):
        j = self.job(max_price_per_hour=2)
        run_pass(self.repo, NOW)
        self.assertEqual(self.repo.get_job(j)["state"], "held")
        self.repo.cancel_job(j, self.renter) if hasattr(self.repo, "cancel_job") else None
        self.repo.credit(self.renter, -99, "spend", now=NOW)
        j2 = self.job()
        run_pass(self.repo, NOW)
        self.assertIn("credits", self.repo.get_job(j2)["held_reason"])

    def test_commit_rechecks_rules(self):
        j = self.job(max_price_per_hour=2)
        gid = self.repo.list_active_gpus(NOW)[0]["gpu_id"]
        with self.assertRaises(AssignmentRejected):
            self.repo.commit_assignment(j, gid, NOW, NOW + H, "x", now=NOW)
        self.repo.update_provider_terms(self.pid, 6, 9 * 60, 17 * 60, 0, None)
        j2 = self.job()
        with self.assertRaises(AssignmentRejected):
            self.repo.commit_assignment(j2, gid, NOW, NOW + H, "x", now=NOW)

    def test_pause_hides_gpu_and_owner_stop_marks_jobs(self):
        j = self.job()
        run_pass(self.repo, NOW)
        self.repo.owner_stop_all(self.pid)
        self.assertEqual(self.repo.provider_stop_requests(self.pid), {j: "owner"})
        self.assertEqual(self.repo.list_active_gpus(NOW), [])

    def test_trusted_rent_free(self):
        self.repo.trust_user(self.pid, self.renter)
        j = self.job()
        run_pass(self.repo, NOW)
        a = self.repo.get_job(j)
        self.repo.complete_job(j, "ok", now=a["start_time"] + 600)
        self.assertEqual(self.repo.balance(self.renter), 100)


class PortalWebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        config.JOBLOG_DIR = os.path.join(self.tmp, "logs")
        self.db = os.path.join(self.tmp, "t.db")
        self.repo = Repository(self.db); self.repo.init()
        self.repo.create_user("root", "pw-root", "admin")
        self.repo.create_user("stu", "pw-stu", "submitter")
        auth._FAILS.clear()
        self.ex = create_app(self.db, portal="exchange")
        self.col = create_app(self.db, portal="college")

    def login(self, app, u):
        c = app.test_client()
        self.assertEqual(c.post("/login", json={"username": u, "password": "pw-" + u}).status_code, 200)
        return c

    def test_college_hides_exchange(self):
        c = self.login(self.col, "stu")
        for url in ("/market", "/share", "/wallet", "/register", "/providers", "/connect"):
            self.assertEqual(c.get(url).status_code, 404, url)
        html = c.get("/home").get_data(as_text=True)
        self.assertNotIn("Providers", html); self.assertNotIn("SpareGPU", html); self.assertIn("kletech", html)

    def test_editor_files_rename_delete_and_gpu_status(self):
        c = self.login(self.col, "stu")
        for path in ("a/x.py", "a/y.py", "b.py"):
            self.assertEqual(c.put("/api/files", json={"path": path, "content": "1"}).status_code, 200)
        r = c.post("/api/files/rename", json={"path": "a", "new_path": "c"})
        self.assertEqual((r.status_code, r.get_json()["moved"]), (200, 2))
        self.assertEqual(c.post("/api/files/rename", json={"path": "c/x.py", "new_path": "b.py"}).status_code, 400)
        self.assertEqual(c.open("/api/files", method="DELETE", json={"path": "c", "folder": True}).status_code, 200)
        left = [f["path"] for f in c.get("/api/files").get_json()["files"]]
        self.assertEqual(left, ["b.py"])
        st = c.get("/api/gpu-status").get_json()           # no worker online: must say so honestly
        self.assertEqual(st["online"], 0); self.assertFalse(st["scripts_ok"])

    def test_admin_gpu_management(self):
        c = self.login(self.col, "root")
        r = c.post("/admin/servers", json={"name": "omkar-laptop", "gpu_name": "RTX 2050", "memory_gb": "4"})
        self.assertEqual(r.status_code, 201); self.assertTrue(r.get_json()["token"])
        self.assertEqual(c.post("/admin/servers", json={"name": "omkar-laptop", "gpu_name": "x", "memory_gb": "4"}).status_code, 409)
        self.assertEqual(c.post("/admin/servers", json={"name": "bad", "gpu_name": "x", "memory_gb": "0"}).status_code, 400)
        gid = self.repo.list_all_gpus()[0]["gpu_id"]
        self.assertEqual(self.repo.list_active_gpus(), [])        # declared but no worker connected: never available
        html = c.get("/admin/devices").get_data(as_text=True)
        self.assertIn("RTX 2050", html); self.assertIn("offline (no worker)", html)
        self.assertEqual(c.post(f"/admin/gpus/{gid}/update", json={"name": "RTX 2050 Ti", "memory_gb": "6"}).status_code, 200)
        self.assertEqual(self.repo.list_all_gpus()[0]["memory_gb"], 6)
        self.assertEqual(c.post(f"/admin/gpus/{gid}/remove", json={}).status_code, 200)
        self.assertIn("disabled", c.get("/admin/devices").get_data(as_text=True))
        s = self.login(self.col, "stu")
        self.assertIn(s.get("/admin/devices").status_code, (302, 403, 404))
        self.assertIn(s.post("/admin/servers", json={"name": "z", "gpu_name": "x", "memory_gb": "4"}).status_code, (302, 403, 404))

    def test_college_admin_registers_auto_approved_node(self):
        c = self.login(self.col, "root")
        r = c.post("/providers", json={"name": "demo-node"})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(self.repo.list_providers()[0]["approved"], 1)

    def test_exchange_signup_and_share_flow(self):
        config.CREDITS_ENABLED = True
        self.addCleanup(setattr, config, "CREDITS_ENABLED", False)
        c = self.ex.test_client()
        r = c.post("/register", json={"username": "omkar", "password": "longpassword"})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(c.get("/wallet", headers={"Accept": "application/json"}).get_json()["balance"], 100)
        self.assertEqual(c.post("/register", json={"username": "omkar", "password": "longpassword"}).status_code, 400)
        r = c.post("/share", json={"name": "omkar-pc", "price_per_hour": 3, "avail_from": "22:00", "avail_to": "07:00"})
        self.assertEqual(r.status_code, 201)
        pid = r.get_json()["provider_id"]
        self.assertEqual(c.post("/share", json={"name": "bad", "price_per_hour": -1}).status_code, 400)
        m = c.get("/market", headers={"Accept": "application/json"}).get_json()
        self.assertEqual(m[0]["price_per_hour"], 3)
        stranger = self.login(self.ex, "stu")
        self.assertEqual(stranger.post(f"/share/{pid}/stop", json={}).status_code, 404)    # only the owner may stop it
        self.assertEqual(c.post(f"/share/{pid}/pause", json={"enabled": 0}).status_code, 200)
        self.assertEqual(self.repo.get_provider(pid)["sharing_enabled"], 0)
        self.assertIn(b"Market", c.get("/market").data)

    def test_admin_grants_credits_only_admin(self):
        config.CREDITS_ENABLED = True
        self.addCleanup(setattr, config, "CREDITS_ENABLED", False)
        s = self.login(self.ex, "stu")
        self.assertEqual(s.post("/admin/credits", json={"username": "stu", "amount": 5}).status_code, 403)
        a = self.login(self.ex, "root")
        self.assertEqual(a.post("/admin/credits", json={"username": "stu", "amount": 25}).status_code, 200)
        self.assertEqual(self.repo.balance(self.repo.get_user_by_name("stu")["user_id"]), 25)


if __name__ == "__main__":
    unittest.main()


class NoWalletTests(unittest.TestCase):
    def test_wallet_gone_and_editor_needs_allocation(self):
        tmp = tempfile.mkdtemp(); config.JOBLOG_DIR = os.path.join(tmp, "l")
        db = os.path.join(tmp, "t.db"); r = Repository(db); r.init(); r.create_user("stu", "pw-stu", "submitter")
        auth._FAILS.clear()
        app = create_app(db, portal="exchange"); c = app.test_client()
        c.post("/login", json={"username": "stu", "password": "pw-stu"})
        self.assertEqual(c.get("/wallet").status_code, 404)
        self.assertEqual(c.get("/editor").status_code, 302)           # no GPU allocated yet
        html = c.get("/market").get_data(as_text=True)
        self.assertNotIn("credits", html); self.assertNotIn(">Editor<", html); self.assertNotIn("Priority", c.get("/jobs/new").get_data(as_text=True))
