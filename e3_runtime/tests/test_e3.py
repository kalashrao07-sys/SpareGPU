"""E3 tests: provider agent, heartbeat protocol, offline/restart recovery, and the NFR4 crash test
(scheduler force-killed INSIDE the commit transaction, 5 times).
The agent talks to the REAL Flask routes through the test client, so these tests cover the full
server <-> agent path without needing a network."""
import os, subprocess, sys, tempfile, time, unittest, types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import psutil
import config
import tempfile as _tf
config.JOBLOG_DIR = _tf.mkdtemp()
from e1_scheduling.scheduler import run_pass
from e2_persistence import Repository
from e3_runtime import agent as agent_mod, gpu_probe, recovery
from e3_runtime.provider_monitor import watchdog_pass
from e4_web.app import create_app

CRASH_RUNS = 5


class ClientApi:
    """Stands in for agent.Api: same post(), but goes straight into the Flask app."""
    def __init__(self, client, token):
        self.client, self.token = client, token

    def post(self, path, payload, timeout=10):
        r = self.client.post(path, json=payload, headers={"Authorization": "Bearer " + self.token})
        if r.status_code != 200:
            raise ValueError(f"{path} -> {r.status_code}")
        return r.get_json()


class Env:
    """A server database + app + one approved provider with a dev-mode agent."""
    def __init__(self, scripts=False):
        self.tmp = tempfile.mkdtemp()
        config.JOBLOG_DIR = os.path.join(self.tmp, "joblogs")
        self.db = os.path.join(self.tmp, "t.db")
        self.repo = Repository(self.db)
        self.repo.init()
        self.u1 = self.repo.create_user("kalash", "pw", "submitter")
        self.u2 = self.repo.create_user("komal", "pw", "submitter")
        self.owner = self.repo.create_user("omkar", "pw", "submitter")
        self.pid, self.token = self.repo.create_provider(self.owner, "omkar-laptop")
        self.repo.set_provider_approved(self.pid, 1)
        self.app = create_app(self.db)
        self.api = ClientApi(self.app.test_client(), self.token)
        self.agent = self.make_agent(scripts)
        self.agent.step()                                    # first heartbeat: registers the GPU

    def make_agent(self, scripts=False):
        return agent_mod.Agent(self.api, os.path.join(self.tmp, "agent"), dev=True, native_scripts=scripts)

    def run_until(self, cond, tries=120, agent=None):
        agent = agent or self.agent
        for _ in range(tries):
            agent.step()
            run_pass(self.repo)
            if cond():
                return True
            time.sleep(0.25)
        return False


class AgentSafetyTests(unittest.TestCase):
    def test_project_files_cannot_escape_the_job_folder(self):
        jobdir = os.path.join(tempfile.mkdtemp(), "job")
        n = agent_mod.write_project(jobdir, {"ok.py": "print(1)", "sub/dir/b.py": "x", "../evil.py": "x",
                                             "/abs.py": "x", "a/../../evil2.py": "x"})
        self.assertEqual(n, 2)
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(jobdir), "evil.py")))
        for bad in ("../x", "/etc/passwd", "a/../../x", "", None):
            with self.assertRaises(agent_mod.ScriptRejected, msg=bad):
                agent_mod.safe_join(jobdir, bad)

    def test_symlink_escape_is_rejected(self):
        base = tempfile.mkdtemp()
        os.makedirs(os.path.join(base, "job"))
        outside = os.path.join(base, "outside.py")
        open(outside, "w").write("print(1)")
        try:
            os.symlink(outside, os.path.join(base, "job", "link.py"))
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available")
        with self.assertRaises(agent_mod.ScriptRejected):
            agent_mod.safe_join(os.path.join(base, "job"), "link.py")

    def test_user_code_refused_unless_sandbox_enabled(self):
        e = Env(scripts=False)
        a = {"kind": "script", "script_path": "proj/a.py", "username": "kalash", "memory_gb": 1, "end_time": 1,
             "job_id": 1, "assignment_id": 1}
        with self.assertRaises(agent_mod.ScriptRejected):
            e.agent.build(a)
        with self.assertRaises(agent_mod.ScriptRejected):
            e.agent.build(dict(a, kind="template", template="rm -rf /"))
        with self.assertRaises(agent_mod.ScriptRejected):
            Env(scripts=True).agent.build(dict(a, kind="session"))        # sessions need real Docker

    def test_docker_command_is_locked_down(self):
        opts = {"image": "img:1", "network": "none", "cpus": 2, "ram_gb": 8, "dev": False}
        a = {"job_id": 7, "kind": "script", "script_path": "proj/train.py", "memory_gb": 3.5, "end_time": 1234.5,
             "device_index": 1}
        cmd = agent_mod.build_docker_command(a, "C:\\jobs\\job_7", opts)
        j = " ".join(cmd)
        for must in ("--rm", "--network none", "--cap-drop ALL", "--security-opt no-new-privileges", "--read-only",
                     "--pids-limit 256", "--memory 8g", "--memory-swap 8g", "--cpus 2", "-u 1000:1000",
                     "--gpus device=1", "--name gpusched_job_7", "label gpusched=1", "GPUSCHED_MEMORY_GB=3.5"):
            self.assertIn(must, j)
        self.assertNotIn("--privileged", j)
        self.assertNotIn("docker.sock", j)
        self.assertEqual(cmd[-4:-1], ["python", "-u", "/gpusched/sandbox_wrapper.py"])
        self.assertEqual(cmd[-1], "proj/train.py")                           # the only user-chosen argv item
        self.assertEqual(sum(1 for c in cmd if c.startswith("type=bind")), 2)  # tool dir (read-only) + the job copy only
        sess = agent_mod.build_docker_command(dict(a, kind="session", script_path=None), "d", opts)
        self.assertEqual(sess[-2:], ["sleep", "infinity"])

    def test_orphan_containers_are_removed_after_agent_restart(self):
        e = Env()
        e.agent.docker = True
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            return types.SimpleNamespace(stdout="abc123 def456\n", returncode=0)
        real = agent_mod.subprocess.run
        agent_mod.subprocess.run = fake_run
        try:
            self.assertEqual(e.agent.kill_orphans(), 2)
        finally:
            agent_mod.subprocess.run = real
        self.assertIn(["docker", "rm", "-f", "abc123", "def456"], calls)

    def test_nvidia_smi_output_is_parsed(self):
        """The exact format nvidia-smi prints on the target laptop (RTX 2050, 4096 MiB)."""
        fake = lambda cmd, **kw: types.SimpleNamespace(stdout="0, NVIDIA GeForce RTX 2050, 4096\n" if "memory.total" in cmd[1]
                                                       else "0, 123, 45\n")
        real = gpu_probe.subprocess.run
        gpu_probe.subprocess.run = fake
        try:
            self.assertEqual(gpu_probe.probe_gpus(),
                             [{"device_index": 0, "name": "NVIDIA GeForce RTX 2050", "gpu_type": "RTX 2050", "memory_gb": 4.0}])
            self.assertEqual(gpu_probe.read_stats(), {0: (123.0, 45.0)})
        finally:
            gpu_probe.subprocess.run = real


class PlatformFlowTests(unittest.TestCase):
    def test_job_runs_end_to_end_with_live_log(self):
        e = Env()
        j = e.repo.submit_job(e.u1, "matmul", 0.5, 0.04, 3, project_name="p")
        self.assertTrue(e.run_until(lambda: e.repo.get_job(j)["state"] == "completed"))
        job = e.repo.get_job(j)
        self.assertTrue(job["result_note"].startswith("finished"), job["result_note"])
        self.assertIn("RESULT", e.repo.read_log(j)[0])             # output reached the server
        self.assertEqual(job["provider_name"], "omkar-laptop")
        self.assertIsNone(e.repo.list_active_gpus()[0]["current_job_id"])
        self.assertEqual(e.repo.check_consistency(), [])

    def test_simultaneous_requests_run_by_priority_one_at_a_time(self):
        e = Env()
        a = e.repo.submit_job(e.u1, "matmul", 0.5, 0.04, 2, now=1)
        b = e.repo.submit_job(e.u2, "train", 0.5, 0.04, 4, now=2)
        run_pass(e.repo)
        self.assertEqual(e.repo.get_job(b)["state"], "running")
        self.assertEqual(e.repo.get_job(a)["state"], "held")
        self.assertTrue(e.run_until(lambda: e.repo.get_job(a)["state"] == "completed", tries=200))
        self.assertEqual(e.repo.get_job(b)["state"], "completed")
        self.assertLessEqual(e.repo.get_job(b)["end_time"], e.repo.get_job(a)["start_time"] + 1)
        self.assertEqual(e.repo.check_consistency(), [])

    def test_user_script_runs_from_editor_project_and_streams_output(self):
        e = Env(scripts=True)
        e.repo.save_file(e.u1, "proj/hello.py", "import os\nprint('hello from user script')\nprint('mem', os.environ['GPUSCHED_MEMORY_GB'])\n")
        e.repo.save_file(e.u2, "proj/hello.py", "print('KOMAL SECRET')\n")        # another user's file must never be used
        e.agent.step()                                         # tell the server user code is allowed
        j = e.repo.submit_job(e.u1, None, 1, 0.5, 3, kind="script", script_path="proj/hello.py", project_name="p")
        self.assertTrue(e.run_until(lambda: e.repo.get_job(j)["state"] == "completed"))
        log = e.repo.read_log(j)[0]
        self.assertIn("hello from user script", log)
        self.assertNotIn("KOMAL", log)
        self.assertFalse(os.path.exists(os.path.join(e.tmp, "agent", "jobs", f"job_{j}")))   # job copy removed afterwards

    def test_stop_button_kills_a_running_job(self):
        e = Env(scripts=True)
        e.repo.save_file(e.u1, "long.py", "import time\nprint('started', flush=True)\ntime.sleep(60)\n")
        e.agent.step()
        j = e.repo.submit_job(e.u1, None, 1, 5, 3, kind="script", script_path="long.py", project_name="p")
        self.assertTrue(e.run_until(lambda: "started" in e.repo.read_log(j)[0], tries=60))
        self.assertEqual(e.repo.cancel_job(j, e.u1), "stopping")
        self.assertTrue(e.run_until(lambda: e.repo.get_job(j)["state"] == "completed", tries=40))
        self.assertEqual(e.repo.get_job(j)["result_note"], "stopped by user")
        self.assertEqual(e.repo.check_consistency(), [])

    def test_terminal_relay_between_browser_and_job_sandbox(self):
        e = Env(scripts=True)
        e.agent.exec_factory = lambda entry: ["sh"]            # stands in for `docker exec -i <container> bash`
        e.repo.save_file(e.u1, "long.py", "import time\ntime.sleep(60)\n")
        e.agent.step()
        j = e.repo.submit_job(e.u1, None, 1, 5, 3, kind="script", script_path="long.py", project_name="p")
        self.assertTrue(e.run_until(lambda: e.repo.get_job(j)["state"] == "running" and e.agent.live, tries=40))
        e.repo.term_browser_seen(j)                            # the user opens the terminal
        e.agent.step()                                         # heartbeat learns the terminal is open, starts the shell
        e.repo.term_send(j, "echo relay-works-$((20+22))\n")
        out = ""
        for _ in range(40):
            e.agent.pump_terminals()
            time.sleep(0.15)
            out += e.repo.term_read_output(j, 0)[0]
            if "relay-works-42" in out:
                break
        self.assertIn("relay-works-42", out)
        e.repo.term_clear(j)
        e.repo.cancel_job(j, e.u1)
        e.run_until(lambda: e.repo.get_job(j)["state"] == "completed", tries=40)
        self.assertEqual(e.agent.terms, {})                    # shell closed with the job

    def test_script_job_waits_when_provider_has_scripts_off(self):
        e = Env(scripts=False)
        j = e.repo.submit_job(e.u1, None, 1, 0.5, 3, kind="script", script_path="proj/hello.py", project_name="p")
        run_pass(e.repo)
        job = e.repo.get_job(j)
        self.assertEqual(job["state"], "held")
        self.assertIn("accepts your own", job["held_reason"])

    def test_job_that_ignores_its_duration_is_killed(self):
        e = Env()
        sleeper = os.path.join(e.tmp, "sleeper.py")
        open(sleeper, "w").write("import time\ntime.sleep(60)\n")
        old = dict(agent_mod.JOB_SCRIPTS), config.KILL_GRACE_S
        agent_mod.JOB_SCRIPTS["matmul"] = sleeper
        config.KILL_GRACE_S = 0
        try:
            j = e.repo.submit_job(e.u1, "matmul", 0.5, 0.02, 3)
            self.assertTrue(e.run_until(lambda: e.repo.get_job(j)["state"] == "completed", tries=40))
            self.assertIn("exceeded the declared duration", e.repo.get_job(j)["result_note"])
        finally:
            agent_mod.JOB_SCRIPTS.update(old[0])
            config.KILL_GRACE_S = old[1]


class ProviderFailureTests(unittest.TestCase):
    def test_offline_provider_jobs_are_requeued_and_not_scheduled(self):
        e = Env()
        j = e.repo.submit_job(e.u1, "matmul", 0.5, 5, 3)
        run_pass(e.repo)
        e.agent.step()                                         # agent starts it
        self.assertEqual(e.repo.get_job(j)["state"], "running")
        later = time.time() + config.PROVIDER_OFFLINE_S + 5    # laptop closed: no more heartbeats
        self.assertEqual(watchdog_pass(e.repo, later), [(j, "requeued: provider offline")])
        self.assertEqual(e.repo.get_job(j)["state"], "queued")
        run_pass(e.repo, later)
        self.assertEqual(e.repo.get_job(j)["state"], "held")   # nowhere to run until it returns
        self.assertEqual(e.repo.check_consistency(), [])
        for entry in list(e.agent.live.values()):              # clean up the real process
            recovery.kill_tree(entry["proc"])

    def test_agent_restart_kills_orphans_and_server_requeues(self):
        e = Env()
        j = e.repo.submit_job(e.u1, "matmul", 0.5, 5, 3)
        run_pass(e.repo)
        e.agent.step(); e.agent.step()
        old_proc = list(e.agent.live.values())[0]["proc"]
        self.assertTrue(old_proc.is_running())
        new_agent = e.make_agent()                             # same work folder = same laptop restarted
        self.assertEqual(new_agent.kill_orphans(), 1)
        list(e.agent.live.values())[0]["popen"].wait(timeout=5)   # in this test the old agent is our own process: reap it
        self.assertFalse(old_proc.is_running())
        new_agent.step()                                       # heartbeat reports nothing running
        self.assertEqual(e.repo.get_job(j)["state"], "queued")
        self.assertEqual(e.repo.check_consistency(), [])

    def test_server_restart_does_not_disturb_running_job(self):
        e = Env()
        j = e.repo.submit_job(e.u1, "matmul", 0.5, 0.05, 3)
        run_pass(e.repo)
        e.agent.step()
        e.app = create_app(e.db)                               # "restart" the web server
        e.agent.api = ClientApi(e.app.test_client(), e.token)
        self.assertTrue(e.run_until(lambda: e.repo.get_job(j)["state"] == "completed"))

    def test_bad_token_and_foreign_job_are_refused(self):
        e = Env()
        c = e.app.test_client()
        self.assertEqual(c.post("/agent/heartbeat", json={}, headers={"Authorization": "Bearer nope"}).status_code, 401)
        self.assertEqual(c.post("/agent/heartbeat", json={}).status_code, 401)
        other_pid, other_tok = e.repo.create_provider(e.owner, "other")
        e.repo.set_provider_approved(other_pid, 1)
        j = e.repo.submit_job(e.u1, "matmul", 0.5, 5, 3)
        run_pass(e.repo)
        e.agent.step()
        hdr = {"Authorization": "Bearer " + other_tok}
        self.assertEqual(c.post("/agent/log", json={"job_id": j, "text": "evil"}, headers=hdr).status_code, 403)
        c.post("/agent/finish", json={"job_id": j, "note": "hijack"}, headers=hdr)
        self.assertEqual(e.repo.get_job(j)["state"], "running")
        for entry in list(e.agent.live.values()):
            recovery.kill_tree(entry["proc"])

    def test_unapproved_provider_receives_no_jobs(self):
        e = Env()
        e.repo.set_provider_approved(e.pid, 0)
        e.repo.submit_job(e.u1, "matmul", 0.5, 5, 3)
        run_pass(e.repo)
        reply = e.agent.heartbeat()
        self.assertEqual(reply["launches"], [])


class CrashTests(unittest.TestCase):
    def start_scheduler(self, db, env_extra=None):
        env = os.environ.copy()
        env.update({"GPUSCHED_DB": db, "GPUSCHED_MODE": "dev"})
        env.update(env_extra or {})
        return subprocess.Popen([sys.executable, "-u", "-m", "e1_scheduling.scheduler"], cwd=ROOT, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_nfr4_kill_scheduler_inside_commit_5_times(self):
        for run in range(1, CRASH_RUNS + 1):
            with self.subTest(run=run):
                tmp = tempfile.mkdtemp()
                db, mark = os.path.join(tmp, "t.db"), os.path.join(tmp, "mark")
                repo = Repository(db)
                repo.init()
                uid = repo.create_user("u", "pw", "submitter")
                repo.add_gpu("DEV", "DEV", 4, device_index=0)
                j = repo.submit_job(uid, "matmul", 1, 1, 3)
                p = self.start_scheduler(db, {"GPUSCHED_FAULT": "sleep_in_commit", "GPUSCHED_FAULT_SLEEP": "30",
                                              "GPUSCHED_FAULT_MARK": mark})
                try:
                    for _ in range(100):                             # wait until it is inside the transaction
                        if os.path.exists(mark):
                            break
                        time.sleep(0.2)
                    self.assertTrue(os.path.exists(mark), "scheduler never reached the commit")
                    psutil.Process(p.pid).kill()                     # force kill (TerminateProcess on Windows / SIGKILL)
                    p.wait()
                    # the half-written transaction must be gone
                    self.assertEqual(repo.check_consistency(), [])
                    self.assertEqual(repo.get_job(j)["state"], "queued")
                    self.assertIsNone(repo.get_job(j)["gpu_id"])
                    self.assertIsNone(repo.list_active_gpus()[0]["current_job_id"])
                    # restart without the fault: the job is scheduled normally
                    p2 = self.start_scheduler(db)
                    try:
                        for _ in range(60):
                            if repo.get_job(j)["state"] == "running":
                                break
                            time.sleep(0.25)
                        self.assertEqual(repo.get_job(j)["state"], "running")
                        self.assertEqual(repo.check_consistency(), [])
                    finally:
                        p2.kill()
                        p2.wait()
                finally:
                    if p.poll() is None:
                        p.kill()
                    repo.close()


if __name__ == "__main__":
    unittest.main()
