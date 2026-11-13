"""E3: GPU discovery and the job runner.

Run as its own process (started and restarted by the supervisor):  python -m e3_runtime.gpu_runner
It starts jobs whose assignment E1 has committed, watches them, enforces the declared duration
(a job is force-killed KILL_GRACE_S after its end time), frees the GPU when a job ends, and writes
live GPU memory / utilisation into E2 for the pool page.

Safety: only the two whitelisted scripts in e3_runtime/jobs can ever run, and the only arguments
passed are numbers produced by the validated database record. No user text reaches a command line.
"""
import json
import os
import subprocess
import sys
import time

import psutil

import config
from e2_persistence import Repository

from . import recovery

JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
JOB_SCRIPTS = {                                   # the whitelist
    "matmul": os.path.join(JOBS_DIR, "matmul_benchmark.py"),
    "train": os.path.join(JOBS_DIR, "mlp_training.py"),
}


# ---------------------------------------------------------------------- GPU discovery
def _smi(query):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError("nvidia-smi not available - is the NVIDIA driver installed? "
                           "(use GPUSCHED_MODE=dev only on a machine without an NVIDIA GPU)") from e
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def probe_gpus():
    """Return [{'index','name','type','memory_gb'}] for the pool."""
    if config.GPU_MODE == "dev":
        return [{"index": 0, "name": "DEV (no real GPU)", "type": "DEV",
                 "memory_gb": float(os.environ.get("GPUSCHED_DEV_MEM_GB", "4"))}]
    gpus = []
    for idx, name, mem_mib in _smi("index,name,memory.total"):
        gpus.append({"index": int(idx), "name": name,
                     "type": name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").strip(),
                     "memory_gb": round(float(mem_mib) / 1024.0, 2)})
    return gpus


def register_gpus(repo):
    """Put the discovered GPU(s) in the pool (E2). Safe to call repeatedly."""
    return [repo.upsert_real_gpu(g["index"], g["name"], g["type"], g["memory_gb"]) for g in probe_gpus()]


def read_stats():
    """{device_index: (mem_used_mb, util_pct)}"""
    if config.GPU_MODE == "dev":
        return {0: (0.0, 0.0)}
    try:
        return {int(i): (float(m), float(u)) for i, m, u in _smi("index,memory.used,utilization.gpu")}
    except (RuntimeError, ValueError):
        return {}


# ---------------------------------------------------------------------- running jobs
def build_command(a):
    """Command for an assignment. Whitelisted script + numbers only."""
    script = JOB_SCRIPTS[a["template"]]
    return [sys.executable, "-u", script,
            "--memory-gb", f"{float(a['memory_gb']):.3f}",
            "--deadline-epoch", f"{float(a['end_time']):.3f}",
            "--overhead-gb", str(config.CUDA_OVERHEAD_GB),
            "--device", "cpu" if config.GPU_MODE == "dev" else "cuda"]


def _summarise(log_path, rc):
    last, result = "", None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("RESULT "):
                    result = line[7:]
                elif line:
                    last = line
    except OSError:
        pass
    if rc == 0 and result:
        try:
            r = json.loads(result)
            return "finished: " + ", ".join(f"{k}={v}" for k, v in r.items())
        except ValueError:
            return "finished: " + result
    if rc == 0:
        return "finished"
    return f"failed (exit code {rc}): {last[:200]}" if last else f"failed (exit code {rc})"


class Runner:
    def __init__(self, repo):
        self.repo = repo
        self.live = {}                                  # assignment_id -> dict(proc, job_id, end, log)

    def launch(self, a):
        config.ensure_dirs()
        log_path = os.path.join(config.LOG_DIR, f"job_{a['job_id']}.log")
        env = os.environ.copy()
        if config.GPU_MODE != "dev" and a.get("device_index") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a["device_index"])
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                p = subprocess.Popen(build_command(a), stdout=logf, stderr=subprocess.STDOUT,
                                     cwd=config.BASE_DIR, env=env, creationflags=flags)
        except Exception as e:                           # could not even start: release the GPU
            self.repo.complete_job(a["job_id"], f"failed to start: {e}")
            print(f"[E3] job {a['job_id']} failed to start: {e}", flush=True)
            return
        self.repo.mark_launched(a["assignment_id"], p.pid)
        self.live[a["assignment_id"]] = {"proc": psutil.Process(p.pid), "popen": p, "job_id": a["job_id"],
                                         "end": a["end_time"], "log": log_path}
        print(f"[E3] job {a['job_id']} started (pid {p.pid})", flush=True)

    def adopt(self, assignment, proc):
        self.live[assignment["assignment_id"]] = {
            "proc": proc, "popen": None, "job_id": assignment["job_id"], "end": assignment["end_time"],
            "log": os.path.join(config.LOG_DIR, f"job_{assignment['job_id']}.log")}
        print(f"[E3] adopted running job {assignment['job_id']} (pid {proc.pid}) after restart", flush=True)

    def poll(self, now=None):
        now = time.time() if now is None else now
        for aid in list(self.live):
            e = self.live[aid]
            rc = e["popen"].poll() if e["popen"] is not None else (None if e["proc"].is_running() else 0)
            if rc is None and now > e["end"] + config.KILL_GRACE_S:
                recovery.kill_tree(e["proc"])
                if e["popen"] is not None:
                    try:
                        e["popen"].wait(timeout=5)          # reap the killed process
                    except subprocess.TimeoutExpired:
                        pass
                note, rc = "stopped: exceeded the declared duration", "killed"
            elif rc is None:
                continue
            else:
                note = _summarise(e["log"], rc) if e["popen"] is not None else "finished (adopted after restart)"
            self.repo.complete_job(e["job_id"], note)
            print(f"[E3] job {e['job_id']} {note}", flush=True)
            del self.live[aid]

    def step(self):
        for a in self.repo.unlaunched_assignments():
            self.launch(a)
        self.poll()


def main():
    repo = Repository()
    repo.init()
    runner = Runner(repo)
    rep = recovery.reconcile(repo)
    for a, proc in rep["adopted"]:
        runner.adopt(a, proc)
    if rep["requeued"]:
        print(f"[E3] recovery re-queued jobs {rep['requeued']}", flush=True)
    print("[E3] runner started", flush=True)
    last_stats = 0.0
    gpu_by_index = {}
    while True:
        try:
            runner.step()
            if time.time() - last_stats >= 2.0:
                last_stats = time.time()
                gpu_by_index = {g["device_index"]: g["gpu_id"] for g in repo.list_all_gpus()
                                if not g["is_simulated"]}
                for idx, (mem, util) in read_stats().items():
                    if idx in gpu_by_index:
                        repo.update_gpu_stats(gpu_by_index[idx], mem, util)
        except Exception as e:
            print(f"[E3] runner step failed: {e}", file=sys.stderr, flush=True)
        time.sleep(config.RUNNER_POLL_S)


if __name__ == "__main__":
    main()
