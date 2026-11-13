"""E3 (server side): provider heartbeats and the watchdog that re-queues work from lost providers.

handle_heartbeat() is called by E4's /agent/heartbeat route after it has authenticated the provider.
watchdog_pass() runs in its own process (python -m e3_runtime.provider_monitor).

E3 idea: the heartbeat arrives every 2 s and each one refreshes the provider's "last seen" time. If nothing
arrives for PROVIDER_OFFLINE_S (20 s) the provider counts as offline and the watchdog re-queues its running
jobs so the scheduler can give them to another GPU. The monitor is its own process, so it keeps working even
if the web server or the scheduler crashes.
"""
import sys
import time

import config
from e2_persistence import Repository


def handle_heartbeat(repo, provider, payload, now=None):
    """Record a provider's report and answer with the jobs it should start.
    payload: {gpus:[...], running:[{assignment_id,pid}], accepts_scripts:bool (= sandbox available)}"""
    now = time.time() if now is None else now
    pid = provider["provider_id"]
    # Update 'last seen' = this is the heartbeat itself.
    repo.touch_provider(pid, now)
    if "accepts_scripts" in payload:
        # Remember whether this provider has a sandbox (Docker) and so accepts users' own code.
        repo.set_provider_scripts(pid, bool(payload["accepts_scripts"]), payload.get("workspace_root"))
    # Sizes are clipped and values cast (str/int/float) because the payload comes from the network.
    gpus = [g for g in payload.get("gpus", []) if isinstance(g, dict)]
    ids = repo.sync_provider_gpus(pid, [{"device_index": int(g["device_index"]), "name": str(g["name"])[:80],
                                         "gpu_type": str(g["gpu_type"])[:40], "memory_gb": float(g["memory_gb"])}
                                        for g in gpus], now) if gpus else []
    by_index = {int(g["device_index"]): gid for g, gid in zip(gpus, ids)}
    for g in gpus:
        if "mem_used_mb" in g and int(g["device_index"]) in by_index:
            repo.update_gpu_stats(by_index[int(g["device_index"])], float(g["mem_used_mb"]),
                                  float(g.get("util_pct", 0)), now)
    # assignment_id -> pid for every job the agent says it is running right now.
    reported = {int(r["assignment_id"]): r.get("pid") for r in payload.get("running", []) if isinstance(r, dict)}
    # Lost job (e.g. agent restarted and killed it): the server expects it but the agent does not report
    # it.
    for a in repo.provider_active(pid):                   # server thinks it runs, agent does not: it was lost
        if a["assignment_id"] not in reported:
            repo.requeue_job(a["job_id"], "provider restarted and lost the job; re-queued", now=now)
    # First time the agent reports a job: mark it as really launched.
    for a in repo.running_assignments(pid):               # agent has started it: remember that
        if not a["launched"] and a["assignment_id"] in reported:
            repo.mark_launched(a["assignment_id"], reported[a["assignment_id"]], now)
    # Jobs assigned to this provider that it has not started yet; they go back in the reply.
    launches = [{"assignment_id": a["assignment_id"], "job_id": a["job_id"], "kind": a["kind"],
                 "template": a["template"], "script_path": a["script_path"], "username": a["username"],
                 "memory_gb": a["memory_gb"], "end_time": a["end_time"], "device_index": a["device_index"]}
                for a in repo.provider_launches(pid) if a["assignment_id"] not in reported]
    # Jobs the user/owner asked to stop; the agent kills them. Also returns which jobs have a terminal
    # open.
    stops = repo.provider_stop_requests(pid)
    return {"launches": launches, "server_time": now,
            "stop_jobs": list(stops), "stop_reasons": {str(k): v for k, v in stops.items()}, "term_jobs": repo.provider_terminal_jobs(pid, now)}


def watchdog_pass(repo, now=None):
    """Re-queue jobs of providers that stopped reporting; stop jobs far past their planned end."""
    now = time.time() if now is None else now
    done = []
    # Providers whose last heartbeat is older than PROVIDER_OFFLINE_S.
    for prov_id in repo.offline_provider_ids(now):
        for a in repo.running_assignments(prov_id):
            if repo.requeue_job(a["job_id"], "provider went offline; re-queued", now=now):
                done.append((a["job_id"], "requeued: provider offline"))
    # Safety net: a job far past its end time (agent did not stop it) is closed on the server side.
    for a in repo.overdue_assignments(now):
        if repo.complete_job(a["job_id"], "stopped: exceeded the declared duration", now=now):
            done.append((a["job_id"], "completed: overdue"))
    return done


def main():
    """Monitor process: run watchdog_pass() every AGENT_POLL_S forever; one failed pass must not kill the loop."""
    repo = Repository()
    repo.init()
    print("[E3] provider monitor started", flush=True)
    while True:
        try:
            for job_id, what in watchdog_pass(repo):
                print(f"[E3] job {job_id} {what}", flush=True)
        except Exception as e:
            print(f"[E3] monitor pass failed: {e}", file=sys.stderr, flush=True)
        time.sleep(config.AGENT_POLL_S)


if __name__ == "__main__":
    main()
