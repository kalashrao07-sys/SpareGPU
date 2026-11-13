import os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config, tempfile as _tf
config.JOBLOG_DIR = _tf.mkdtemp()
from e2_persistence import AssignmentRejected, BadState, GpuBusy, Repository


class E2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = Repository(os.path.join(self.tmp, "t.db"))
        self.repo.init()
        self.uid = self.repo.create_user("u", "secret", "submitter")
        self.gpu = self.repo.add_gpu("G", "A", 8, now=0)
        self.job = self.repo.submit_job(self.uid, "matmul", 4, 1, 3, now=1)

    def tearDown(self):
        self.repo.close()

    def test_atomic_commit_success(self):
        self.repo.commit_assignment(self.job, self.gpu, 10, 70, "only valid GPU", now=10)
        self.assertEqual(self.repo.get_job(self.job)["state"], "running")
        self.assertEqual(self.repo.list_active_gpus()[0]["current_job_id"], self.job)
        self.assertEqual(self.repo.check_consistency(), [])

    def test_nfr1_failure_before_commit_leaves_nothing_behind(self):
        with self.assertRaises(RuntimeError):
            self.repo.commit_assignment(self.job, self.gpu, 10, 70, "x", _fail_before_commit=True)
        j = self.repo.get_job(self.job)
        self.assertEqual(j["state"], "queued")
        self.assertIsNone(j["gpu_id"])                                  # no assignment row
        self.assertIsNone(self.repo.list_active_gpus()[0]["current_job_id"])
        self.assertEqual(self.repo.check_consistency(), [])

    def test_hc2_enforced_inside_the_database(self):
        self.repo.commit_assignment(self.job, self.gpu, 10, 70, "x")
        j2 = self.repo.submit_job(self.uid, "train", 1, 1, 5)
        with self.assertRaises(AssignmentRejected):
            self.repo.commit_assignment(j2, self.gpu, 11, 71, "x")
        self.assertEqual(self.repo.get_job(j2)["state"], "queued")

    def test_hc1_enforced_inside_the_database(self):
        big = self.repo.submit_job(self.uid, "train", 16, 1, 5)
        with self.assertRaises(AssignmentRejected):
            self.repo.commit_assignment(big, self.gpu, 10, 70, "x")

    def test_cannot_assign_a_running_job_twice(self):
        self.repo.commit_assignment(self.job, self.gpu, 10, 70, "x")
        g2 = self.repo.add_gpu("G2", "A", 8)
        with self.assertRaises(BadState):
            self.repo.commit_assignment(self.job, g2, 10, 70, "x")

    def test_remove_gpu_blocked_while_running_uc6(self):
        self.repo.commit_assignment(self.job, self.gpu, 10, 70, "x")
        with self.assertRaises(GpuBusy):
            self.repo.remove_gpu(self.gpu)
        self.repo.complete_job(self.job, "done")
        self.repo.remove_gpu(self.gpu)
        self.assertEqual(self.repo.list_active_gpus(), [])

    def test_complete_releases_gpu_fr16(self):
        self.repo.commit_assignment(self.job, self.gpu, 10, 70, "x")
        self.assertTrue(self.repo.complete_job(self.job, "ok", now=80))
        g = self.repo.list_active_gpus()[0]
        self.assertEqual((g["current_job_id"], g["free_since"]), (None, 80))
        self.assertFalse(self.repo.complete_job(self.job, "again"))     # idempotent

    def test_requeue_frees_gpu_and_clears_assignment(self):
        self.repo.commit_assignment(self.job, self.gpu, 10, 70, "x")
        self.assertTrue(self.repo.requeue_job(self.job, "crash"))
        self.assertEqual(self.repo.get_job(self.job)["state"], "queued")
        self.assertEqual(self.repo.check_consistency(), [])

    def test_database_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            self.repo.submit_job(self.uid, "matmul", 4, 1, 9)           # priority 9
        with self.assertRaises(ValueError):
            self.repo.submit_job(self.uid, "rm -rf", 4, 1, 3)           # unknown workload
        with self.assertRaises(ValueError):
            self.repo.submit_job(self.uid, "matmul", 0, 1, 3)

    def test_login(self):
        self.assertEqual(self.repo.verify_login("u", "secret")["role"], "submitter")
        self.assertIsNone(self.repo.verify_login("u", "wrong"))
        self.assertIsNone(self.repo.verify_login("nobody", "secret"))

    def test_consistency_check_detects_damage(self):
        self.repo.conn.execute("UPDATE job SET state='running' WHERE job_id=?", (self.job,))   # running, no assignment
        self.assertTrue(any("HC3" in p for p in self.repo.check_consistency()))


if __name__ == "__main__":
    unittest.main()


class V3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = Repository(os.path.join(self.tmp, "t.db"))
        self.repo.init()
        self.u = self.repo.create_user("kalash", "pw", "submitter")
        self.v = self.repo.create_user("komal", "pw", "submitter")

    def tearDown(self):
        self.repo.close()

    def test_files_are_saved_per_user_and_validated(self):
        self.repo.save_file(self.u, "proj/train.py", "print(1)\n")
        self.repo.save_file(self.u, "proj/train.py", "print(2)\n")             # overwrite
        self.assertEqual(self.repo.get_file(self.u, "proj/train.py"), "print(2)\n")
        self.assertIsNone(self.repo.get_file(self.v, "proj/train.py"))          # other user cannot see it
        for bad in ("../x.py", "/etc/passwd", "a//b.py", "a/./b.py", "dir/", "a\\b.py", "", ".hidden/..", "x;y.py"):
            with self.assertRaises(ValueError, msg=bad):
                self.repo.save_file(self.u, bad, "x")
        with self.assertRaises(ValueError):
            self.repo.save_file(self.u, "big.txt", "x" * (self.repo.MAX_FILE_BYTES + 1))
        self.assertEqual([f["path"] for f in self.repo.list_files(self.u)], ["proj/train.py"])
        self.assertTrue(self.repo.delete_file(self.u, "proj/train.py"))
        self.assertEqual(self.repo.get_project(self.u), {})

    def test_file_count_and_total_limits(self):
        self.repo.MAX_FILES = 3
        for i in range(3):
            self.repo.save_file(self.u, f"f{i}.py", "x")
        with self.assertRaises(ValueError):
            self.repo.save_file(self.u, "f9.py", "x")
        self.repo.save_file(self.u, "f0.py", "y")                                # overwriting is still allowed

    def test_session_job_and_terminal_relay(self):
        j = self.repo.submit_job(self.u, None, 1, 5, 3, kind="session", project_name="shell")
        self.assertEqual(self.repo.get_job(j)["kind"], "session")
        self.assertFalse(self.repo.term_active(j))
        self.repo.term_browser_seen(j, now=100)
        self.assertTrue(self.repo.term_active(j, now=105))
        self.assertFalse(self.repo.term_active(j, now=120))
        self.repo.term_send(j, "ls\n"); self.repo.term_send(j, "pwd\n")
        self.assertEqual(self.repo.term_take_input(j), ["ls\n", "pwd\n"])
        self.assertEqual(self.repo.term_take_input(j), [])                       # delivered once only
        self.repo.term_append_output(j, "a\n"); self.repo.term_append_output(j, "b\n")
        text, last = self.repo.term_read_output(j, 0)
        self.assertEqual(text, "a\nb\n")
        self.assertEqual(self.repo.term_read_output(j, last), ("", last))
        with self.assertRaises(ValueError):
            self.repo.term_send(j, "x" * 5000)
        self.repo.term_clear(j)
        self.assertEqual(self.repo.term_read_output(j, 0)[0], "")

    def test_terminal_state_is_cleared_when_job_ends(self):
        self.repo.add_gpu("G", "G", 4, device_index=0)
        j = self.repo.submit_job(self.u, "matmul", 1, 5, 3)
        g = self.repo.list_active_gpus()[0]["gpu_id"]
        self.repo.commit_assignment(j, g, 0, 100, "x")
        self.repo.term_append_output(j, "hi")
        self.repo.complete_job(j, "finished")
        self.assertEqual(self.repo.term_read_output(j, 0)[0], "")
