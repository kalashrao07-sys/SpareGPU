"""E3: process helpers and the post-restart consistency check.

The database already guarantees that a transaction cut off by a crash is never half-stored
(E2 / NFR1). Job processes now live on the provider laptops, so crash handling is split:
  * server restart  -> nothing to do for running jobs; agents keep running them and keep reporting.
  * agent restart   -> the agent kills job processes left from its previous run (agent.kill_orphans);
                       the server sees they are no longer reported and re-queues them (provider_monitor).
  * provider lost   -> the watchdog re-queues its running jobs after PROVIDER_OFFLINE_S.
"""
import psutil


def kill_tree(proc):
    """Force-kill a process and its children (the closest Windows equivalent of kill -9)."""
    try:
        # List children first: once the parent dies we could no longer find them.
        kids = proc.children(recursive=True)
    except psutil.Error:
        kids = []
    # Kill children, then the parent. A process that already exited raises psutil.Error; ignore it.
    for p in kids + [proc]:
        try:
            p.kill()
        except psutil.Error:
            pass


def verify(repo):
    """Consistency check after any restart (NFR2-NFR4). Empty list = consistent.

    The supervisor runs this at startup and before each child restart, e.g. to find a job marked running
    with no assignment. The real checks live in E2's Repository.check_consistency()."""
    return repo.check_consistency()
