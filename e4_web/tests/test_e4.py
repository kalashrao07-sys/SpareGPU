import os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config, tempfile as _tf
config.JOBLOG_DIR = _tf.mkdtemp()
from e2_persistence import Repository
from e4_web import auth, create_app


class E4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        config.JOBLOG_DIR = os.path.join(self.tmp, "joblogs")
        self.db = os.path.join(self.tmp, "t.db")
        self.repo = Repository(self.db)
        self.repo.init()
        self.repo.create_user("alice", "pw-alice", "submitter")
        self.repo.create_user("bob", "pw-bob", "submitter")
        self.repo.create_user("root", "pw-root", "admin")
        self.gpu = self.repo.add_gpu("RTX", "RTX", 4)
        auth._FAILS.clear()
        self.app = create_app(self.db)

    def tearDown(self):
        self.repo.close()

    def client(self, user=None):
        c = self.app.test_client()
        if user:
            r = c.post("/login", json={"username": user, "password": "pw-" + user})
            self.assertEqual(r.status_code, 200)
        return c

    GOOD = {"project_name": "demo", "template": "matmul", "memory_gb": 1, "duration_min": 1, "priority": 3}

    def test_login_required(self):
        c = self.app.test_client()
        self.assertEqual(c.get("/pool", headers={"Accept": "application/json"}).status_code, 401)
        self.assertEqual(c.get("/jobs/new").status_code, 302)

    def test_wrong_password_and_lockout(self):
        c = self.app.test_client()
        for _ in range(5):
            self.assertEqual(c.post("/login", json={"username": "alice", "password": "x"}).status_code, 401)
        self.assertEqual(c.post("/login", json={"username": "alice", "password": "pw-alice"}).status_code, 429)

    def test_valid_submission_fr1(self):
        r = self.client("alice").post("/jobs", json=self.GOOD)
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.get_json()["state"], "queued")

    def test_invalid_input_rejected_and_never_stored_fr3(self):
        c = self.client("alice")
        bad = [dict(self.GOOD, priority=9), dict(self.GOOD, priority=0), dict(self.GOOD, priority=2.5),
               dict(self.GOOD, memory_gb=0), dict(self.GOOD, memory_gb=-1), dict(self.GOOD, memory_gb="abc"),
               dict(self.GOOD, duration_min=0), dict(self.GOOD, duration_min=10 ** 6),
               dict(self.GOOD, template="rm -rf /"), dict(self.GOOD, deadline="tomorrow-ish"),
               {"template": "matmul"}, dict(self.GOOD, memory_gb=float("nan"))]
        for payload in bad:
            r = c.post("/jobs", json=payload)
            self.assertEqual(r.status_code, 400, payload)
            self.assertIn("field", r.get_json())
        self.assertEqual(self.repo.list_jobs(), [])

    def test_deadline_is_optional_metadata_fr2(self):
        r = self.client("alice").post("/jobs", json=dict(self.GOOD, deadline="2026-11-05T18:30"))
        self.assertEqual(r.status_code, 201)
        self.assertEqual(self.repo.get_job(r.get_json()["job_id"])["deadline"], "2026-11-05T18:30")

    def test_own_job_only_fr18(self):
        jid = self.client("alice").post("/jobs", json=self.GOOD).get_json()["job_id"]
        j = {"Accept": "application/json"}
        self.assertEqual(self.client("alice").get(f"/jobs/{jid}", headers=j).status_code, 200)
        self.assertEqual(self.client("bob").get(f"/jobs/{jid}", headers=j).status_code, 404)
        self.assertEqual(self.client("root").get(f"/jobs/{jid}", headers=j).status_code, 200)
        self.assertEqual(self.client("bob").get("/jobs", headers=j).get_json(), [])

    def test_admin_pages_are_server_side_guarded_fr19_fr20(self):
        c = self.client("alice")
        self.assertEqual(c.get("/admin").status_code, 403)
        self.assertEqual(c.get("/admin/held").status_code, 403)
        self.assertEqual(c.post("/admin/gpus", json={"name": "x", "gpu_type": "y", "memory_gb": 8}).status_code, 403)
        self.assertEqual(c.post(f"/admin/gpus/{self.gpu}/remove", json={}).status_code, 403)
        self.assertEqual(self.client("root").get("/admin").status_code, 200)

    def test_simulated_gpus_are_refused_until_approved(self):
        r = self.client("root").post("/admin/gpus", json={"name": "x", "gpu_type": "y", "memory_gb": 8})
        self.assertEqual(r.status_code, 403)
        self.assertIn("approves", r.get_json()["error"])

    def test_remove_gpu_with_running_job_blocked_uc6(self):
        jid = self.client("alice").post("/jobs", json=self.GOOD).get_json()["job_id"]
        self.repo.commit_assignment(jid, self.gpu, 1, 61, "x")
        r = self.client("root").post(f"/admin/gpus/{self.gpu}/remove", json={})
        self.assertEqual(r.status_code, 409)
        self.repo.complete_job(jid, "done")
        self.assertEqual(self.client("root").post(f"/admin/gpus/{self.gpu}/remove", json={}).status_code, 200)

    def test_pool_hides_other_users_names(self):
        jid = self.client("alice").post("/jobs", json=self.GOOD).get_json()["job_id"]
        self.repo.commit_assignment(jid, self.gpu, 1, 61, "x")
        j = {"Accept": "application/json"}
        bob = self.client("bob").get("/pool", headers=j).get_json()["gpus"][0]
        self.assertEqual((bob["busy"], bob["who"], bob["job_id"]), (True, "another user", None))
        me = self.client("alice").get("/pool", headers=j).get_json()["gpus"][0]
        self.assertEqual((me["who"], me["job_id"]), ("you", jid))
        adm = self.client("root").get("/pool", headers=j).get_json()["gpus"][0]
        self.assertEqual(adm["who"], "alice")

    def test_form_post_without_csrf_token_is_rejected(self):
        c = self.client("alice")
        r = c.post("/jobs", data={"project_name": "demo", "template": "matmul", "memory_gb": "1", "duration_min": "1", "priority": "3"})
        self.assertEqual(r.status_code, 400)

    def test_form_post_with_token_works_and_pages_render(self):
        c = self.app.test_client()
        page = c.get("/login").get_data(as_text=True)
        token = page.split('name="csrf_token" value="')[1].split('"')[0]
        r = c.post("/login", data={"username": "alice", "password": "pw-alice", "csrf_token": token})
        self.assertEqual(r.status_code, 302)
        token = c.get("/jobs/new").get_data(as_text=True).split('name="csrf_token" value="')[1].split('"')[0]
        r = c.post("/jobs", data={"project_name": "demo", "template": "train", "memory_gb": "1", "duration_min": "0.5", "priority": "2",
                                  "csrf_token": token})
        self.assertEqual(r.status_code, 302)
        for path in ("/jobs", "/pool", r.headers["Location"]):
            self.assertEqual(c.get(path).status_code, 200, path)
    def test_provider_registration_by_admin_only_and_auto_approved(self):
        self.assertEqual(self.client("alice").post("/providers", json={"name": "alice-pc"}).status_code, 404)   # students never register GPUs
        r = self.client("root").post("/providers", json={"name": "lab-node"})
        self.assertEqual(r.status_code, 201)
        tok, pid = r.get_json()["token"], r.get_json()["provider_id"]
        self.assertTrue(self.repo.get_provider(pid)["approved"])
        self.assertEqual(self.client("root").post("/providers", json={"name": "lab-node"}).status_code, 409)
        self.assertEqual(self.client("root").post("/providers", json={"name": "bad;name"}).status_code, 400)
        self.assertEqual(self.client("alice").post(f"/providers/{pid}/approve", json={"approved": 0}).status_code, 403)
        self.assertEqual(self.client("root").post(f"/providers/{pid}/approve", json={"approved": 0}).status_code, 200)
        self.assertNotIn(tok, str(self.repo.get_provider(pid)))            # only a hash is stored
        self.assertEqual(self.client("root").get("/providers").status_code, 200)

    def test_script_job_validation_and_live_log_access(self):
        c = self.client("alice")
        bad = dict(self.GOOD, kind="script", script_path="../x.py")
        self.assertEqual(c.post("/jobs", json=bad).status_code, 400)
        ok = dict(self.GOOD, kind="script", script_path="proj/train.py", framework="PyTorch")
        self.assertEqual(c.post("/jobs", json=ok).status_code, 400)             # file not saved in the editor yet
        self.assertEqual(c.put("/api/files", json={"path": "proj/train.py", "content": "print(1)"}).status_code, 200)
        jid = c.post("/jobs", json=ok).get_json()["job_id"]
        self.repo.append_log(jid, "epoch 1\n")
        d = c.get(f"/jobs/{jid}/log?offset=0").get_json()
        self.assertEqual((d["text"], d["offset"]), ("epoch 1\n", 8))
        self.assertEqual(c.get(f"/jobs/{jid}/log?offset=8").get_json()["text"], "")
        self.assertEqual(self.client("bob").get(f"/jobs/{jid}/log").status_code, 404)   # own logs only
    def test_editor_file_api_is_private_validated_and_exportable(self):
        import io, zipfile
        a, b = self.client("alice"), self.client("bob")
        self.assertEqual(a.put("/api/files", json={"path": "p/a.py", "content": "x=1\n"}).status_code, 200)
        self.assertEqual(a.put("/api/files", json={"path": "../evil.py", "content": "x"}).status_code, 400)
        self.assertEqual(a.put("/api/files", json={"path": "p/b.py"}).status_code, 400)
        self.assertEqual(b.get("/api/files?path=p/a.py").status_code, 404)           # bob cannot read alice's file
        self.assertEqual(b.get("/api/files").get_json()["files"], [])
        self.assertEqual(self.app.test_client().get("/api/files").status_code, 302)
        z = zipfile.ZipFile(io.BytesIO(a.get("/api/files/backup.zip").data))
        self.assertEqual(z.read("p/a.py"), b"x=1\n")
        self.assertEqual(a.delete("/api/files", json={"path": "p/a.py"}).status_code, 200)
        self.assertEqual(self.app.test_client().put("/api/files", json={"path": "x.py", "content": ""}).status_code, 401)
        # a form post (cross-site style) cannot write files
        self.assertEqual(a.put("/api/files", data={"path": "x.py", "content": "x"}).status_code, 400)

    def test_terminal_only_for_owner_of_running_user_code(self):
        a, b = self.client("alice"), self.client("bob")
        owner = self.repo.verify_login("root", "pw-root")["user_id"]
        pid, _ = self.repo.create_provider(owner, "box", accepts_scripts=1)
        self.repo.set_provider_approved(pid, 1); self.repo.touch_provider(pid)
        pgpu = self.repo.sync_provider_gpus(pid, [dict(device_index=0, name="G", gpu_type="G", memory_gb=4)])[0]
        jid = a.post("/jobs", json={"project_name": "p", "kind": "session", "memory_gb": 1, "duration_min": 1, "priority": 3}).get_json()["job_id"]
        self.assertEqual(a.get(f"/jobs/{jid}/term").status_code, 409)               # not running yet
        self.repo.commit_assignment(jid, pgpu, 0, 100, "x")
        self.assertEqual(b.get(f"/jobs/{jid}/term").status_code, 404)               # other user
        self.assertEqual(self.client("root").get(f"/jobs/{jid}/term").status_code, 404)   # not even admin
        self.assertEqual(a.get(f"/jobs/{jid}/term").status_code, 200)
        self.assertEqual(a.post(f"/jobs/{jid}/term", json={"data": "ls\n"}).status_code, 200)
        self.assertEqual(a.post(f"/jobs/{jid}/term", json={"data": "x" * 5000}).status_code, 400)
        tpl = a.post("/jobs", json=self.GOOD).get_json()["job_id"]
        self.assertEqual(a.get(f"/jobs/{tpl}/term").status_code, 400)

    def test_stop_and_cancel(self):
        a, b = self.client("alice"), self.client("bob")
        jid = a.post("/jobs", json=self.GOOD).get_json()["job_id"]
        self.assertEqual(b.post(f"/jobs/{jid}/stop", json={}).status_code, 404)     # not bob's job
        self.assertEqual(a.post(f"/jobs/{jid}/stop", json={}).get_json()["result"], "cancelled")
        self.assertEqual(self.repo.get_job(jid)["state"], "completed")
        j2 = a.post("/jobs", json=self.GOOD).get_json()["job_id"]
        self.repo.commit_assignment(j2, self.gpu, 0, 100, "x")
        self.assertEqual(a.post(f"/jobs/{j2}/stop", json={}).get_json()["result"], "stopping")
        self.assertEqual(self.repo.get_job(j2)["stop_requested"], 1)

    def test_connect_page_renders(self):
        self.assertEqual(self.client("alice").get("/connect").status_code, 404)    # admin-only in the college portal
        self.assertEqual(self.client("root").get("/connect").status_code, 200)
        self.assertEqual(self.client("alice").get("/editor").status_code, 200)

    def test_new_admin_pages_render_and_are_admin_only(self):
        for path in ("/admin", "/admin/requests", "/admin/fleet", "/admin/devices"):
            self.assertEqual(self.client("alice").get(path).status_code, 403, path)
            self.assertEqual(self.app.test_client().get(path, headers={"Accept": "application/json"}).status_code, 401, path)
            self.assertEqual(self.client("root").get(path).status_code, 200, path)
        self.assertEqual(self.client("alice").get("/home").status_code, 200)
        self.assertEqual(self.client("root").get("/home").status_code, 302)          # admin lands on the dashboard
        self.assertEqual(self.client("root").get("/admin", headers={"Accept": "application/json"}).get_json().keys(), {"gpus", "held"})

    def test_decline_provider_request(self):
        pid, _ = self.repo.create_provider(self.repo.verify_login("alice", "pw-alice")["user_id"], "alice-pc")
        self.assertEqual(self.client("alice").post(f"/providers/{pid}/decline", json={}).status_code, 403)
        self.assertEqual(self.client("root").post(f"/providers/{pid}/decline", json={}).status_code, 200)
        self.assertIsNone(self.repo.get_provider(pid))
        self.assertEqual(self.client("root").post(f"/providers/{pid}/decline", json={}).status_code, 404)
        # a provider that has already run jobs cannot be deleted, only suspended
        pid2, _ = self.repo.create_provider(self.repo.verify_login("alice", "pw-alice")["user_id"], "used-pc", accepts_scripts=1)
        self.repo.set_provider_approved(pid2, 1); self.repo.touch_provider(pid2)
        gid = self.repo.sync_provider_gpus(pid2, [dict(device_index=0, name="G", gpu_type="G", memory_gb=4)])[0]
        j = self.repo.submit_job(self.repo.verify_login("bob", "pw-bob")["user_id"], "matmul", 1, 1, 3)
        self.repo.commit_assignment(j, gid, 0, 100, "x")
        self.assertEqual(self.client("root").post(f"/providers/{pid2}/decline", json={}).status_code, 409)

    def test_admin_stats_and_fleet_view(self):
        owner = self.repo.verify_login("alice", "pw-alice")["user_id"]
        pid, _ = self.repo.create_provider(owner, "lab")
        self.repo.sync_provider_gpus(pid, [dict(device_index=i, name=f"G{i}", gpu_type="G", memory_gb=8) for i in range(2)])
        self.assertEqual(self.repo.admin_stats()["pending_providers"], 1)
        self.repo.set_provider_approved(pid, 1); self.repo.touch_provider(pid)
        fleet = {f["name"]: f for f in self.repo.fleet_view()}
        self.assertEqual((fleet["lab"]["total"], fleet["lab"]["allocated"], fleet["lab"]["available"], fleet["lab"]["percent"]), (2, 0, 2, 0))
        j = self.repo.submit_job(owner, "matmul", 1, 1, 3)
        self.repo.commit_assignment(j, fleet["lab"]["gpus"][0]["gpu_id"], 0, 100, "x")
        f = {f["name"]: f for f in self.repo.fleet_view()}["lab"]
        self.assertEqual((f["allocated"], f["available"], f["percent"]), (1, 1, 50))
        self.assertEqual(self.repo.admin_stats()["running_jobs"], 1)

    def test_html_is_escaped(self):
        self.repo.rename_gpu(self.gpu, "<script>alert(1)</script>")
        html = self.client("alice").get("/pool").get_data(as_text=True)
        self.assertNotIn("<script>alert(1)</script>", html)


if __name__ == "__main__":
    unittest.main()
