"""End-to-end scenarios (the 12 listed by the guide) driven through the real web app, scheduler pass,
heartbeat handler and watchdog. No Docker or GPU is needed: the worker is simulated by calling the same
heartbeat/complete functions the real agent calls, so every server-side rule is exercised for real."""
import os, sys, tempfile, threading, time, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from e1_scheduling.scheduler import run_pass
from e2_persistence import Repository
from e3_runtime.provider_monitor import handle_heartbeat, watchdog_pass
from e4_web import auth
from e4_web.app import create_app


JSON = {"Accept": "application/json"}


class Scenarios(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        config.JOBLOG_DIR = os.path.join(self.tmp, "logs")
        self.db = os.path.join(self.tmp, "t.db")
        self.repo = Repository(self.db); self.repo.init()
        self.admin = self.repo.create_user("root", "pw-root", "admin")
        self.u1 = self.repo.create_user("ann", "pw-ann", "submitter")
        self.u2 = self.repo.create_user("bob", "pw-bob", "submitter")
        auth._FAILS.clear()
        self.app = create_app(self.db, portal="college")

    def login(self, u):
        c = self.app.test_client()
        self.assertEqual(c.post("/login", json={"username": u, "password": "pw-" + u}).status_code, 200)
        return c

    def worker(self, name="omkar", mem=4.0, now=None):
        """Register a worker with one GPU and send its first heartbeat (= it is online)."""
        pid, tok = self.repo.add_server(self.admin, name, "RTX 2050", "RTX 2050", mem)
        prov = self.repo.get_provider(pid)
        handle_heartbeat(self.repo, prov, {"accepts_scripts": True, "gpus": [
            {"device_index": 0, "name": "RTX 2050", "gpu_type": "RTX 2050", "memory_gb": mem}]}, now)
        return pid, tok

    def submit(self, c, mem=1, mins=5):
        r = c.post("/jobs", json={"kind": "script", "script_path": "a.py", "memory_gb": mem,
                                  "duration_min": mins, "priority": 3, "project_name": "a"})
        self.assertEqual(r.status_code, 201, r.get_data(as_text=True))
        return r.get_json()["job_id"]

    def put_script(self, uid):
        self.repo.save_file(uid, "a.py", "print(1)")

    # 1. many users submit at once
    def test_01_concurrent_submissions(self):
        self.put_script(self.u1)
        ids, errs = [], []
        def go():
            try:
                ids.append(self.submit(self.login("ann")))
            except Exception as e:
                errs.append(e)
        ts = [threading.Thread(target=go) for _ in range(6)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(errs, []); self.assertEqual(len(set(ids)), 6)

    # 2 + 3 + 4. two jobs compete for one GPU, loser queues, takes the GPU when it is freed
    def test_02_03_04_contention_queue_then_assign(self):
        self.put_script(self.u1); self.put_script(self.u2)
        self.worker()
        a, b = self.submit(self.login("ann")), self.submit(self.login("bob"))
        run_pass(self.repo)
        states = sorted(self.repo.get_job(j)["state"] for j in (a, b))
        self.assertEqual(states, ["held", "running"])                       # never two on one GPU (HC2)
        first = a if self.repo.get_job(a)["state"] == "running" else b
        second = b if first == a else a
        self.assertEqual(self.repo.get_job(second)["held_reason"], "a suitable GPU exists but none is currently free")
        self.assertEqual(self.repo.check_consistency(), [])
        self.repo.complete_job(first, "done")
        run_pass(self.repo)
        self.assertEqual(self.repo.get_job(second)["state"], "running")     # assigned as soon as the GPU is free

    # 5. execution attempt with no GPU: held with an honest reason, never "running"
    def test_05_no_gpu_never_runs(self):
        self.put_script(self.u1)
        c = self.login("ann")
        self.assertEqual(c.get("/api/gpu-status").get_json()["online"], 0)
        j = self.submit(c)
        run_pass(self.repo)
        job = self.repo.get_job(j)
        self.assertEqual(job["state"], "held"); self.assertIn("no", (job["held_reason"] or "").lower())
        self.assertEqual(c.get(f"/jobs/{j}", headers=JSON).get_json()["state"], "held")

    # 6. worker disconnects mid-run: job is re-queued, not left "running", not lost
    def test_06_worker_disconnect_requeues(self):
        self.put_script(self.u1)
        pid, _ = self.worker(now=1000.0)
        j = self.submit(self.login("ann"))
        run_pass(self.repo, now=1001.0)
        self.assertEqual(self.repo.get_job(j)["state"], "running")
        watchdog_pass(self.repo, now=1001.0 + config.PROVIDER_OFFLINE_S + 5)
        job = self.repo.get_job(j)
        self.assertEqual(job["state"], "queued"); self.assertIn("offline", job["result_note"] or "")
        run_pass(self.repo, now=1100.0)                                     # offline worker is not scheduled again
        self.assertNotEqual(self.repo.get_job(j)["state"], "running")
        self.assertEqual(self.repo.check_consistency(), [])

    # 7. cancel (queued and running)
    def test_07_cancel(self):
        self.put_script(self.u1)
        c = self.login("ann")
        q = self.submit(c)                                                  # no GPU: stays waiting
        self.assertEqual(c.post(f"/jobs/{q}/stop", json={}).status_code, 200)
        self.assertEqual(self.repo.get_job(q)["state"], "completed")
        self.worker()
        r = self.submit(c); run_pass(self.repo)
        self.assertEqual(self.repo.get_job(r)["state"], "running")
        self.assertEqual(c.post(f"/jobs/{r}/stop", json={}).status_code, 200)
        self.assertEqual(self.repo.get_job(r)["stop_requested"], 1)         # the worker kills it on its next poll

    # 7b. timeout safety net: a job far past its declared end is closed by the server
    def test_07b_timeout_closes_overdue_job(self):
        self.put_script(self.u1); self.worker(now=1000.0)
        j = self.submit(self.login("ann"), mins=1); run_pass(self.repo, now=1001.0)
        end = self.repo.get_job(j)["end_time"]
        late = end + 3600
        handle_heartbeat(self.repo, self.repo.list_providers()[0], {"gpus": [], "running": [
            {"assignment_id": self.repo.running_assignments()[0]["assignment_id"], "pid": 1}]}, late)   # worker alive, job not stopped
        watchdog_pass(self.repo, now=late)
        self.assertEqual(self.repo.get_job(j)["state"], "completed")

    # 8. admin adds / disables GPUs, effect on scheduling
    def test_08_admin_gpu_changes_affect_scheduling(self):
        self.put_script(self.u1)
        a = self.login("root")
        self.assertEqual(a.post("/admin/servers", json={"name": "s1", "gpu_name": "RTX 2050", "memory_gb": "4"}).status_code, 201)
        prov = self.repo.list_providers()[0]
        handle_heartbeat(self.repo, prov, {"accepts_scripts": True, "gpus": [
            {"device_index": 0, "name": "RTX 2050", "gpu_type": "RTX 2050", "memory_gb": 4}]})
        gid = self.repo.list_all_gpus()[0]["gpu_id"]
        j = self.submit(self.login("ann")); run_pass(self.repo)
        self.assertEqual(self.repo.get_job(j)["state"], "running")
        self.assertEqual(a.post(f"/admin/gpus/{gid}/remove", json={}).status_code, 409)   # busy: refused
        self.repo.complete_job(j, "ok")
        self.assertEqual(a.post(f"/admin/gpus/{gid}/remove", json={}).status_code, 200)
        j2 = self.submit(self.login("ann")); run_pass(self.repo)
        self.assertEqual(self.repo.get_job(j2)["state"], "held")            # disabled GPU gets nothing
        self.assertEqual(a.post(f"/admin/gpus/{gid}/restore", json={}).status_code, 200)
        run_pass(self.repo)
        self.assertEqual(self.repo.get_job(j2)["state"], "running")

    # 9. editor file operations
    def test_09_editor_file_operations(self):
        c = self.login("ann")
        self.assertEqual(c.put("/api/files", json={"path": "src/m.py", "content": "x=1"}).status_code, 200)
        self.assertEqual(c.get("/api/files?path=src/m.py").get_json()["content"], "x=1")
        self.assertEqual(c.post("/api/files/rename", json={"path": "src/m.py", "new_path": "src/n.py"}).status_code, 200)
        self.assertEqual(c.put("/api/files", json={"path": "../evil.py", "content": ""}).status_code, 400)
        self.assertEqual(c.delete("/api/files", json={"path": "src/n.py"}).status_code, 200)
        self.assertEqual(c.get("/api/files").get_json()["files"], [])

    # 10. successful run: output and final status reach the user
    def test_10_successful_execution_output(self):
        self.put_script(self.u1); self.worker()
        c = self.login("ann"); j = self.submit(c); run_pass(self.repo)
        self.repo.append_log(j, "hello from the GPU\n")
        self.repo.complete_job(j, "finished: exit code 0")
        self.assertIn("hello from the GPU", c.get(f"/jobs/{j}/log?offset=0").get_json()["text"])
        v = c.get(f"/jobs/{j}", headers=JSON).get_json()
        self.assertEqual((v["state"], v["result_note"]), ("completed", "finished: exit code 0"))

    # 11. authentication and isolation
    def test_11_auth_restrictions(self):
        self.put_script(self.u1)
        anon = self.app.test_client()
        self.assertIn(anon.post("/jobs", json={"kind": "script"}).status_code, (302, 401))
        self.assertIn(anon.get("/api/files").status_code, (302, 401))
        j = self.submit(self.login("ann"))
        bob = self.login("bob")
        self.assertEqual(bob.get(f"/jobs/{j}").status_code, 404)           # other users' jobs are invisible
        self.assertEqual(bob.get(f"/jobs/{j}/log").status_code, 404)
        self.assertEqual(bob.post(f"/jobs/{j}/stop", json={}).status_code, 404)
        self.assertEqual(bob.get("/api/files").get_json()["files"], [])    # separate workspaces
        self.assertIn(bob.get("/admin/devices").status_code, (302, 403, 404))
        self.assertEqual(anon.post("/agent/heartbeat", json={}, headers={"Authorization": "Bearer nope"}).status_code, 401)

    # 12. database failure mid-commit: nothing half-done, and the system recovers
    def test_12_database_failure_and_recovery(self):
        self.put_script(self.u1); self.worker()
        j = self.submit(self.login("ann"))
        gpu = self.repo.list_active_gpus()[0]
        with self.assertRaises(Exception):
            self.repo.commit_assignment(j, gpu["gpu_id"], 1.0, 61.0, "only valid GPU", _fail_before_commit=True)
        self.assertEqual(self.repo.get_job(j)["state"], "queued")           # rolled back completely
        self.assertIsNone(self.repo.list_active_gpus()[0]["current_job_id"])
        run_pass(self.repo)                                                 # the next pass simply succeeds
        self.assertEqual(self.repo.get_job(j)["state"], "running")
        self.assertEqual(self.repo.check_consistency(), [])


if __name__ == "__main__":
    unittest.main()
