import os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config, tempfile as _tf
config.JOBLOG_DIR = _tf.mkdtemp()
from e1_scheduling.heap_queue import JobQueue
from e1_scheduling.scheduler import run_pass
from e1_scheduling.selection import (REASON_NO_MEMORY, REASON_NONE_FREE, filter_gpus, select_gpu)
from e2_persistence import Repository


def gpu(i, mem=8.0, typ="A", busy=False, free_since=0.0):
    return {"gpu_id": i, "gpu_type": typ, "memory_gb": mem, "current_job_id": 99 if busy else None, "free_since": free_since}


class SelectionTests(unittest.TestCase):
    def test_heap_order_priority_then_submission(self):
        jobs = [{"job_id": 1, "priority": 2, "submitted_at": 1}, {"job_id": 2, "priority": 5, "submitted_at": 9},
                {"job_id": 3, "priority": 5, "submitted_at": 4}, {"job_id": 4, "priority": 1, "submitted_at": 0}]
        q, order = JobQueue(jobs), []
        while len(q):
            order.append(q.pop()["job_id"])
        self.assertEqual(order, [3, 2, 1, 4])

    def test_hc1_hold_reason(self):
        d = select_gpu({"memory_gb": 16}, [gpu(1, 8), gpu(2, 12)], 10)
        self.assertEqual((d.kind, d.reason), ("hold", REASON_NO_MEMORY))

    def test_hc2_hold_reason(self):
        d = select_gpu({"memory_gb": 4}, [gpu(1, 8, busy=True)], 10)
        self.assertEqual((d.kind, d.reason), ("hold", REASON_NONE_FREE))

    def test_every_gpu_is_evaluated_and_busy_ones_rejected(self):
        passed, cands = filter_gpus({"memory_gb": 6}, [gpu(1, 4), gpu(2, 8, busy=True), gpu(3, 8)])
        self.assertEqual([g["gpu_id"] for g in passed], [2, 3])
        self.assertEqual([g["gpu_id"] for g in cands], [3])

    def test_sp2_smaller_idle_gap_wins(self):
        d = select_gpu({"memory_gb": 4}, [gpu(1, free_since=0), gpu(2, free_since=90)], now=100)
        self.assertEqual((d.gpu_id, d.rule), (2, "SP2 idle time"))

    def test_sp3_used_only_when_sp2_ties(self):
        d = select_gpu({"memory_gb": 4, "pref_gpu_type": "B"}, [gpu(1, typ="A", free_since=50), gpu(2, typ="B", free_since=50)], 100)
        self.assertEqual((d.gpu_id, d.rule), (2, "SP3 GPU-type continuity"))
        # a better SP2 value beats SP3 (ranked, not blended - FR8)
        d = select_gpu({"memory_gb": 4, "pref_gpu_type": "B"}, [gpu(1, typ="A", free_since=99), gpu(2, typ="B", free_since=10)], 100)
        self.assertEqual(d.gpu_id, 1)

    def test_single_candidate_rule(self):
        d = select_gpu({"memory_gb": 4}, [gpu(1)], 5)
        self.assertEqual(d.rule, "only valid GPU")


class PassTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = Repository(os.path.join(self.tmp, "t.db"))
        self.repo.init()
        self.uid = self.repo.create_user("u", "pw", "submitter")

    def tearDown(self):
        self.repo.close()

    def job(self, mem, prio, t, dur=1):
        return self.repo.submit_job(self.uid, "matmul", mem, dur, prio, now=t)

    def test_higher_priority_gets_the_only_gpu(self):
        self.repo.add_gpu("G1", "A", 8, now=0)
        low, high = self.job(2, 1, 1), self.job(2, 5, 2)
        run_pass(self.repo, now=10)
        self.assertEqual(self.repo.get_job(high)["state"], "running")
        j = self.repo.get_job(low)
        self.assertEqual((j["state"], j["held_reason"]), ("held", REASON_NONE_FREE))

    def test_held_job_is_re_evaluated_when_a_gpu_appears_fr13(self):
        self.repo.add_gpu("small", "A", 4, now=0)
        j = self.job(8, 3, 1)
        run_pass(self.repo, now=5)
        self.assertEqual(self.repo.get_job(j)["held_reason"], REASON_NO_MEMORY)
        self.repo.add_gpu("big", "B", 16, now=6)
        run_pass(self.repo, now=7)
        self.assertEqual(self.repo.get_job(j)["state"], "running")

    def test_nfr8_nfr9_6_gpus_20_jobs_zero_violations(self):
        """Test fixture only: 6 GPU rows and 20 jobs in a throw-away database."""
        mems = [4, 8, 8, 16, 24, 32]
        for i, m in enumerate(mems):
            self.repo.add_gpu(f"T{i}", "A" if i % 2 else "B", m, is_simulated=1, now=0)
        for k in range(20):
            self.job(mem=[2, 6, 10, 20, 30, 40][k % 6], prio=1 + k % 5, t=k, dur=1)
        done, now = set(), 100.0
        for step in range(60):
            run_pass(self.repo, now=now)
            self.assertEqual(self.repo.check_consistency(), [])           # automatic post-run check (NFR9)
            for a in self.repo.running_assignments():
                self.repo.complete_job(a["job_id"], "ok", now=now)
                done.add(a["job_id"])
            now += 60
        states = [j["state"] for j in self.repo.list_jobs()]
        self.assertEqual(states.count("completed") + states.count("held"), 20)
        self.assertEqual(states.count("held"), 3)        # the three 40 GB jobs: larger than the biggest (32 GB) GPU
        self.assertEqual(states.count("completed"), 17)
        for j in self.repo.list_jobs():
            if j["state"] == "held":
                self.assertEqual(j["held_reason"], REASON_NO_MEMORY)


if __name__ == "__main__":
    unittest.main()


class V2Tests(unittest.TestCase):
    """Provider-aware behaviour: online check, HC4 (script mode), new hold reasons."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = Repository(os.path.join(self.tmp, "t.db"))
        self.repo.init()
        self.owner = self.repo.create_user("omkar", "pw", "submitter")
        self.user = self.repo.create_user("kalash", "pw", "submitter")
        self.pid, _ = self.repo.create_provider(self.owner, "omkar-laptop", accepts_scripts=0)
        self.repo.sync_provider_gpus(self.pid, [dict(device_index=0, name="RTX 2050", gpu_type="RTX 2050", memory_gb=4)])

    def tearDown(self):
        self.repo.close()

    def reason(self, job_id):
        return self.repo.get_job(job_id)["held_reason"]

    def test_unapproved_or_offline_provider_is_never_used(self):
        from e1_scheduling.selection import REASON_NO_PROVIDER
        j = self.repo.submit_job(self.user, "matmul", 2, 5, 3)
        run_pass(self.repo)                                   # not approved
        self.assertEqual(self.reason(j), REASON_NO_PROVIDER)
        self.repo.set_provider_approved(self.pid, 1)
        run_pass(self.repo)                                   # approved but never reported
        self.assertEqual(self.repo.get_job(j)["state"], "held")
        self.repo.touch_provider(self.pid)
        run_pass(self.repo)
        self.assertEqual(self.repo.get_job(j)["state"], "running")

    def test_script_job_needs_script_mode(self):
        from e1_scheduling.selection import REASON_NO_SCRIPTS
        self.repo.set_provider_approved(self.pid, 1); self.repo.touch_provider(self.pid)
        j = self.repo.submit_job(self.user, None, 2, 5, 3, kind="script", script_path="p/train.py")
        run_pass(self.repo)
        self.assertEqual(self.reason(j), REASON_NO_SCRIPTS)
        self.repo.set_provider_scripts(self.pid, 1)
        run_pass(self.repo)
        self.assertEqual(self.repo.get_job(j)["state"], "running")

    def test_priority_order_between_simultaneous_users(self):
        self.repo.set_provider_approved(self.pid, 1); self.repo.touch_provider(self.pid)
        low = self.repo.submit_job(self.user, "matmul", 2, 5, 1, now=1)
        high = self.repo.submit_job(self.owner, "matmul", 2, 5, 5, now=2)
        run_pass(self.repo)
        self.assertEqual(self.repo.get_job(high)["state"], "running")
        self.assertEqual(self.repo.get_job(low)["state"], "held")

    def test_commit_rechecks_hc4_even_if_scheduler_is_bypassed(self):
        from e2_persistence import AssignmentRejected
        self.repo.set_provider_approved(self.pid, 1); self.repo.touch_provider(self.pid)
        j = self.repo.submit_job(self.user, None, 2, 5, 3, kind="script", script_path="p/a.py")
        g = self.repo.list_active_gpus()[0]["gpu_id"]
        with self.assertRaises(AssignmentRejected):
            self.repo.commit_assignment(j, g, 0, 60, "x")
        self.assertEqual(self.repo.check_consistency(), [])
