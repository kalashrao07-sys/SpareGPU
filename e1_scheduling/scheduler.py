"""E1 Scheduling & Constraint Engine: one scheduling pass, and the loop that repeats it.

Run as its own process (started and restarted by E3):  python -m e1_scheduling.scheduler

Role of this file: it is the only place that DECIDES. It reads waiting jobs and free GPUs from E2, asks selection.py
which GPU is best, and asks E2 to commit the choice. E2 re-checks the hard constraints inside one transaction, so even
if the data changed since we read it (a race), a wrong assignment can never be written.
"""
import sys
import time

import config
from e2_persistence import AssignmentRejected, BadState, NotFound, Repository

from .heap_queue import JobQueue
from .selection import select_gpu


def run_pass(repo, now=None):
    """Look at every queued and held job once, best first. Returns a list of (job_id, outcome).
    Held jobs are re-evaluated on every pass, so they are placed as soon as a GPU frees up (FR13)."""
    now = time.time() if now is None else now        # tests pass a fixed "now" so results are repeatable
    results = []
    # Build a priority queue (heap) of every waiting job: highest priority first, earliest submission breaks ties (SP1).
    queue = JobQueue(repo.list_open_jobs())
    while len(queue):
        job = queue.pop()
        gpus = repo.list_active_gpus(now)        # re-read: an earlier commit in this pass may have taken a GPU
        d = select_gpu(job, gpus, now)                   # pure decision: no database writes here
        if d.kind == "assign":
            try:
                # The commit is atomic in E2: assignment row + job "running" + GPU "occupied" all happen or none do.
                repo.commit_assignment(job["job_id"], d.gpu_id, now,
                                       now + job["duration_min"] * 60.0, d.rule, now=now)
                results.append((job["job_id"], f"assigned to GPU {d.gpu_id} ({d.rule})"))
            except (AssignmentRejected, BadState, NotFound) as e:
                results.append((job["job_id"], f"commit rejected: {e}"))   # stays queued, retried next pass
        else:
            # No suitable GPU yet: record WHY on the job so the user sees it. It is retried on every pass.
            repo.set_held(job["job_id"], d.reason)
            results.append((job["job_id"], f"held: {d.reason}"))
    return results


def main():
    """Process entry point: run a pass, sleep SCHEDULER_POLL_S, repeat forever."""
    repo = Repository()
    repo.init()                                      # creates / migrates the database if needed
    print("[E1] scheduler started", flush=True)
    while True:
        try:
            for job_id, outcome in run_pass(repo):
                if not outcome.startswith("held"):
                    print(f"[E1] job {job_id}: {outcome}", flush=True)
        except Exception as e:      # keep the loop alive; E3 restarts us if the process itself dies
            print(f"[E1] pass failed: {e}", file=sys.stderr, flush=True)
        time.sleep(config.SCHEDULER_POLL_S)


if __name__ == "__main__":
    main()
