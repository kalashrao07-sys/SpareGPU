"""E3 provider agent: runs on the GPU owner's laptop (e.g. Omkar's) and is the ONLY part that touches the GPU.

    python -m e3_runtime.agent --server http://SERVER_IP:5000 --token <provider token> [--docker]

E3 (Process Isolation & Recovery) - this file is the "lender laptop" half of E3.
  * Heartbeat: every AGENT_POLL_S (2 s) the agent POSTs to the server with an "Authorization: Bearer <token>"
    header (the token identifies the provider). The server answers with new jobs to launch, jobs to stop and
    which jobs have a terminal open. If no heartbeat arrives for PROVIDER_OFFLINE_S (20 s) the server's
    watchdog (provider_monitor.py) marks the provider offline and re-queues its running jobs.
  * Agent restart: kill_orphans() kills every job process/container the previous run left behind. The server
    then sees the jobs missing from the heartbeat and re-queues them. No ghost job keeps using the GPU.
  * Server restart: the agent only retries on network errors (see main()); jobs keep running on the laptop.
  * Stop requests carry a reason ("owner" = the GPU owner pressed stop, otherwise the user who submitted it).
  * Why Docker (HC4): code written by a stranger must never run directly on the lender's laptop, so user
    code runs only inside a locked-down container (see build_docker_command).

Every 2 s it sends a heartbeat (GPUs, live memory/utilisation, running jobs) and receives the jobs the scheduler
assigned to this laptop. It starts them, streams their output to the server as live logs, force-kills any job that
overruns its time (or that the user stopped), relays the interactive terminal, and reports when each job ends.

SAFETY MODEL
  * Prepared workloads (matmul, train) come from the whitelist in e3_runtime/jobs.
  * A user's own code (scripts and interactive sessions) is accepted ONLY with --docker. It then runs inside a
    container: no network, read-only root file system, all Linux capabilities dropped, non-root user, memory / CPU /
    process limits, one GPU only, and only a copy of the user's own project files mounted. It cannot see this laptop.
  * --unsafe-native-scripts runs scripts directly on this laptop. For development on a machine without Docker only.
  * No text typed on the website is ever placed on a command line: the container command is built from fixed
    pieces plus numbers and a validated path.
"""
import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import psutil

import config
from . import gpu_probe, recovery

# Folder with the whitelisted workload scripts; it is mounted read-only into containers.
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
# Whitelist of prepared workloads: the website can only choose a KEY here, never a file name.
JOB_SCRIPTS = {"matmul": os.path.join(JOBS_DIR, "matmul_benchmark.py"),
               "train": os.path.join(JOBS_DIR, "mlp_training.py")}
# Docker image with PyTorch + CUDA; used for every job unless --image overrides it.
DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
# Max bytes of log sent to the server per request (keeps each POST small).
CHUNK = 16000


class ScriptRejected(Exception):
    """Raised when a job must not be started (bad path, no sandbox, unknown kind). Shown to the user as a failure."""
    pass


def safe_join(base, rel):
    """base/rel with the guarantee that the result stays inside base (no '..', no absolute path, no symlink escape)."""
    # Reject non-strings, empty names, absolute paths and any '..' part (path traversal attack).
    if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")) or ".." in rel.replace("\\", "/").split("/"):
        raise ScriptRejected("invalid path")
    # realpath resolves symlinks, so a link pointing outside the job folder is caught below.
    full = os.path.realpath(os.path.join(base, rel))
    root = os.path.realpath(base)
    # Final check: the real path must still be inside the job folder.
    if os.path.commonpath([root, full]) != root:
        raise ScriptRejected("path escapes the job folder")
    return full


def write_project(jobdir, files):
    """Replace jobdir with a fresh copy of the user's project files. Returns the number of files written."""
    # Start from an empty folder so files of an older run can never leak into this job.
    shutil.rmtree(jobdir, ignore_errors=True)
    os.makedirs(jobdir)
    n = 0
    for rel, content in files.items():
        try:
            # Every file name comes from the user, so each one is validated; unsafe names are silently
            # skipped.
            full = safe_join(jobdir, rel)
        except ScriptRejected:
            continue
        os.makedirs(os.path.dirname(full), exist_ok=True)
        # newline='' keeps the user's line endings unchanged.
        with open(full, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        n += 1
    return n


def container_name(job_id):
    """Deterministic container name per job so we can later 'docker kill' it. int() blocks any injected text."""
    return f"gpusched_job_{int(job_id)}"


def build_docker_command(a, jobdir, opts):
    """The `docker run` command for an assignment. Pure function (unit-tested): only fixed flags, numbers, and
    names derived from numbers; the one user-chosen value (script path) was validated and is a single argv item.

    Flag by flag (HC4: user code only ever runs inside this sandbox):
      --rm                          delete the container automatically when it exits (no leftovers).
      --name gpusched_job_<id>      known name, so the agent can kill / exec into it.
      --label gpusched=1            tag used by kill_orphans() to find our containers after an agent restart.
      --memory / --memory-swap      hard RAM limit; swap = same value means no swap, so the limit is real.
      --cpus                        CPU share the job may use (default 2 cores).
      --pids-limit 256              caps processes/threads, stops fork bombs.
      --network none                no network at all: code cannot download malware or leak data
                                    (only 'bridge' if the lender explicitly chose --docker-network bridge).
      --cap-drop ALL                drop every Linux capability (no mount, no raw sockets, ...).
      --security-opt no-new-privileges   setuid programs inside cannot gain extra rights.
      --read-only                   the container's root file system cannot be modified.
      --tmpfs /tmp:rw,size=512m     the only writable scratch space: in RAM, max 512 MB, wiped on exit.
      -u 1000:1000                  run as an ordinary non-root user instead of root.
      -e ...                        environment: HOME in the writable /tmp, unbuffered Python output (live logs),
                                    the declared GPU memory and the deadline for the wrapper/workloads.
      --mount ... /gpusched         our whitelisted job scripts, mounted READ-ONLY.
      --gpus device=N               expose exactly ONE GPU (the one assigned) and not the others.
      --mount ... /work             a bind mount of this job's own copy of the user's files (the only
                                    laptop folder the container can see); also the working directory.
    """
    jid = int(a["job_id"])
    # At least 1 GB of RAM; float() makes sure only a number ends up on the command line.
    ram = max(1.0, float(opts.get("ram_gb", 8)))
    cmd = ["docker", "run", "--rm", "--name", container_name(jid), "--label", "gpusched=1",
           "--memory", f"{ram:g}g", "--memory-swap", f"{ram:g}g", "--cpus", f"{float(opts.get('cpus', 2)):g}",
           "--pids-limit", "256", "--network", opts.get("network", "none"),
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
           "--tmpfs", "/tmp:rw,size=512m", "-u", "1000:1000",
           "-e", "HOME=/tmp", "-e", "PYTHONUNBUFFERED=1",
           "-e", f"GPUSCHED_MEMORY_GB={float(a['memory_gb']):g}",
           "-e", f"GPUSCHED_DEADLINE_EPOCH={float(a['end_time']):.3f}",
           "--mount", f"type=bind,source={JOBS_DIR},target=/gpusched,readonly"]
    # In --dev mode there is no NVIDIA GPU, so the --gpus flag would make Docker fail.
    if not opts.get("dev"):
        cmd += ["--gpus", f"device={int(a.get('device_index') or 0)}"]
    # Three kinds of job: prepared 'template' workload, user 'script', or interactive 'session'.
    if a["kind"] == "template":
        cmd += ["--mount", f"type=bind,source={jobdir},target=/work", "-w", "/work", opts["image"],
                # Template: run one of our own whitelisted scripts, passing the numbers as arguments.
                "python", "-u", "/gpusched/" + os.path.basename(JOB_SCRIPTS[a["template"]]),
                "--memory-gb", f"{float(a['memory_gb']):.3f}", "--deadline-epoch", f"{float(a['end_time']):.3f}",
                "--overhead-gb", str(config.CUDA_OVERHEAD_GB), "--device", "cpu" if opts.get("dev") else "cuda"]
    elif a["kind"] == "script":
        cmd += ["--mount", f"type=bind,source={jobdir},target=/work", "-w", "/work", opts["image"],
                # Script: sandbox_wrapper.py caps GPU memory, then runs the user's file (one argv item, no
                # shell).
                "python", "-u", "/gpusched/sandbox_wrapper.py", a["script_path"]]
    else:                                                             # interactive session: just keep the box alive
        # 'sleep infinity' does nothing; the terminal later uses 'docker exec' to open a shell in this container
        cmd += ["--mount", f"type=bind,source={jobdir},target=/work", "-w", "/work", opts["image"], "sleep", "infinity"]
    return cmd


def summarise(log_path, rc):
    """Turn the job's log and exit code into the one-line note shown on the website."""
    # Scan the log: remember the last non-empty line and the line starting with 'RESULT ' (printed by
    # workloads).
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
    # Exit code 0 = success; otherwise show the last log line as a hint about what went wrong.
    if rc == 0 and result:
        try:
            return "finished: " + ", ".join(f"{k}={v}" for k, v in json.loads(result).items())
        except ValueError:
            return "finished: " + result
    if rc == 0:
        return "finished"
    return f"failed (exit code {rc}): {last[:200]}" if last else f"failed (exit code {rc})"


class Api:
    """Tiny HTTP client for the server: JSON POST + Bearer token (the provider's password, sent on every call)."""
    def __init__(self, server, token):
        self.server, self.token = server.rstrip("/"), token

    def post(self, path, payload, timeout=10):
        # The Authorization header is how the server knows which provider is calling.
        req = urllib.request.Request(self.server + path, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.token})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")


class TermSession:
    """One interactive shell inside a job's container. A reader thread collects its output."""
    def __init__(self, cmd):
        # stderr merged into stdout so the user sees both; CREATE_NO_WINDOW hides a console window on
        # Windows.
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.out = queue.Queue()
        # A background thread reads output so the agent loop never blocks waiting for the shell.
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        # read1 returns whatever is available right now (up to 2048 bytes); b'' means the shell exited.
        for chunk in iter(lambda: self.proc.stdout.read1(2048), b""):
            self.out.put(chunk.decode("utf-8", "replace"))

    def write(self, text):
        try:
            self.proc.stdin.write(text.encode("utf-8")); self.proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def drain(self):
        # Take everything the reader thread has queued so far (non-blocking).
        parts = []
        while not self.out.empty():
            parts.append(self.out.get_nowait())
        return "".join(parts)

    def close(self):
        try:
            self.proc.kill(); self.proc.wait(timeout=3)
        except Exception:
            pass


class Agent:
    """The provider agent: owns the running jobs (self.live) and the per-job terminals."""
    def __init__(self, api, workdir, dev=False, dev_mem_gb=4.0, docker=False, native_scripts=False,
                 image=DEFAULT_IMAGE, network="none", cpus=2.0, ram_gb=8.0):
        self.api, self.dev, self.dev_mem_gb = api, dev, dev_mem_gb
        self.docker, self.native_scripts = docker, native_scripts
        self.opts = {"image": image, "network": network, "cpus": cpus, "ram_gb": ram_gb, "dev": dev}
        self.workdir = workdir
        self.logdir = os.path.join(workdir, "joblogs")
        self.jobsdir = os.path.join(workdir, "jobs")
        # Remembers the pid of each running job on disk, so kill_orphans() works after a crash.
        self.state_file = os.path.join(workdir, "agent_state.json")
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.jobsdir, exist_ok=True)
        self.live = {}            # assignment_id -> dict
        self.terms = {}           # job_id -> TermSession
        self.term_jobs = set()    # jobs whose owner has the terminal open (from the last heartbeat)
        # Builds the command that opens a shell; a test can swap it for a fake.
        self.exec_factory = self._docker_exec          # tests replace this
        self.last_stop = set()
        self.stop_reasons = {}

    @property
    # True when users' own code may be run: either in Docker or (dev only) natively.
    def sandbox_ok(self):
        """Reported to the server as 'accepts_scripts' so it only sends user code to capable providers."""
        return self.docker or self.native_scripts

    # ----------------------------------------------------------- crash safety on the laptop
    def _save_state(self):
        # Save pid + process start time per running job. The start time lets us tell the real job from a
        # reused pid.
        data = {str(aid): {"pid": e["proc"].pid, "created": e["proc"].create_time()} for aid, e in self.live.items()
                if e["rc"] is None and e["proc"] is not None}
        with open(self.state_file, "w") as f:
            json.dump(data, f)

    def kill_orphans(self):
        """After an agent restart: kill job processes and containers left by the previous run (server re-queues)."""
        n = 0
        try:
            # Read the pids saved by the previous run (missing/corrupt file = nothing to clean).
            with open(self.state_file) as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = {}
        for rec in old.values():
            try:
                p = psutil.Process(rec["pid"])
                # Same pid AND same start time = it really is our old job, not an unrelated process.
                if abs(p.create_time() - rec["created"]) < 1.0:
                    recovery.kill_tree(p); n += 1
            except psutil.Error:
                pass
        with open(self.state_file, "w") as f:
            json.dump({}, f)
        if self.docker:
            try:
                # Find every container carrying our label (running or stopped) and force-remove it.
                ids = subprocess.run(["docker", "ps", "-aq", "--filter", "label=gpusched=1"], capture_output=True,
                                     text=True, timeout=20).stdout.split()
                if ids:
                    subprocess.run(["docker", "rm", "-f"] + ids, capture_output=True, timeout=60)
                    n += len(ids)
            except (OSError, subprocess.SubprocessError):
                pass
        return n

    # ----------------------------------------------------------- launching
    def fetch_project(self, a):
        """Copy the user's project (kept on the server) into a fresh job folder on this laptop."""
        jobdir = os.path.join(self.jobsdir, f"job_{a['job_id']}")
        # The project files are stored on the server, so the laptop downloads them for each job.
        files = self.api.post("/agent/project", {"job_id": a["job_id"]}).get("files", {})
        write_project(jobdir, files)
        return jobdir, files

    def build(self, a):
        """Return (command, cwd, container_name_or_None, jobdir). Raises ScriptRejected."""
        kind = a["kind"]
        if kind == "template":
            # Only whitelisted workloads may run.
            if a["template"] not in JOB_SCRIPTS:
                raise ScriptRejected("unknown prepared workload")
            jobdir = os.path.join(self.jobsdir, f"job_{a['job_id']}")
            os.makedirs(jobdir, exist_ok=True)
            if self.docker:
                return build_docker_command(a, jobdir, self.opts), self.workdir, container_name(a["job_id"]), jobdir
            # Without Docker: run the whitelisted script directly with this Python (no user code involved).
            common = ["--memory-gb", f"{float(a['memory_gb']):.3f}", "--deadline-epoch", f"{float(a['end_time']):.3f}",
                      "--overhead-gb", str(config.CUDA_OVERHEAD_GB), "--device", "cpu" if self.dev else "cuda"]
            return [sys.executable, "-u", JOB_SCRIPTS[a["template"]]] + common, self.workdir, None, jobdir
        # From here on the job contains the USER'S OWN code: refuse unless a sandbox is enabled.
        if not self.sandbox_ok:
            raise ScriptRejected("this provider does not run users' own code (start the agent with --docker)")
        if kind == "session" and not self.docker:
            raise ScriptRejected("interactive sessions need Docker")
        jobdir, files = self.fetch_project(a)
        if kind == "script":
            # The script must be one of the uploaded files, and its path must stay inside the job folder.
            if a["script_path"] not in files:
                raise ScriptRejected(f"script not found in your project files: {a['script_path']}")
            full = safe_join(jobdir, a["script_path"])
            if self.docker:
                return build_docker_command(a, jobdir, self.opts), self.workdir, container_name(a["job_id"]), jobdir
            # Unsafe native mode (--unsafe-native-scripts): no container, runs directly on the laptop. Dev
            # only.
            return [sys.executable, "-u", full], os.path.dirname(full), None, jobdir
        if kind == "session":
            if not self.docker:
                raise ScriptRejected("interactive sessions need Docker")
            return build_docker_command(a, jobdir, self.opts), self.workdir, container_name(a["job_id"]), jobdir
        raise ScriptRejected("unknown job kind")

    def launch(self, a):
        """Start one assignment as a child process, log its output to a file and record it in self.live."""
        aid, jid = a["assignment_id"], a["job_id"]
        log = os.path.join(self.logdir, f"job_{jid}.log")
        try:
            cmd, cwd, cname, jobdir = self.build(a)
            # The job gets the declared memory and the deadline as environment variables.
            env = os.environ.copy()
            env.update({"PYTHONUNBUFFERED": "1", "GPUSCHED_MEMORY_GB": str(a["memory_gb"]),
                        "GPUSCHED_DEADLINE_EPOCH": f"{float(a['end_time']):.3f}"})
            if not self.dev and a.get("device_index") is not None:
                # Only matters for native runs: limits which GPU the process can see (Docker uses --gpus).
                env["CUDA_VISIBLE_DEVICES"] = str(a["device_index"])
            with open(log, "w", encoding="utf-8") as lf:
                # stdout+stderr go to a log file; ship_logs() sends it to the server in pieces.
                p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            # popen = process handle, proc = psutil handle (to kill children), offset = bytes of log
            # already sent.
            self.live[aid] = {"popen": p, "proc": psutil.Process(p.pid), "job_id": jid, "end": a["end_time"],
                              "log": log, "offset": 0, "rc": None, "note": None, "assignment_id": aid,
                              "container": cname, "kind": a["kind"], "stopped": False}
            self._save_state()
            print(f"[agent] job {jid} ({a['kind']}) started (pid {p.pid}{', docker' if cname else ''})", flush=True)
        except Exception as e:
            # Could not start (rejected or crashed): record it as already failed so finish_ready() reports it.
            with open(log, "w", encoding="utf-8") as lf:
                lf.write(f"could not start: {e}\n")
            self.live[aid] = {"popen": None, "proc": None, "job_id": jid, "end": a["end_time"], "log": log,
                              "offset": 0, "rc": "failed", "note": f"failed to start: {e}", "assignment_id": aid,
                              "container": None, "kind": a["kind"], "stopped": False}
            print(f"[agent] job {jid} could not start: {e}", flush=True)

    def kill_job(self, e):
        """Force-stop a job: kill its container, then its process tree, then close any terminal."""
        if e["container"]:
            try:
                # Killing only the 'docker run' client would leave the container alive, so kill the
                # container itself.
                subprocess.run(["docker", "kill", e["container"]], capture_output=True, timeout=20)
            except (OSError, subprocess.SubprocessError):
                pass
        if e["proc"] is not None:
            # Also kill the process and all its children (like kill -9).
            recovery.kill_tree(e["proc"])
        try:
            e["popen"].wait(timeout=5)
        except (subprocess.TimeoutExpired, AttributeError):
            pass
        t = self.terms.pop(e["job_id"], None)
        if t:
            t.close()

    # ----------------------------------------------------------- one cycle
    def heartbeat(self):
        """One heartbeat: report GPUs and running jobs; read launches, stop requests and terminal requests."""
        # Static info (name, memory) and live numbers (used memory, utilisation) from nvidia-smi.
        gpus = gpu_probe.probe_gpus(self.dev, self.dev_mem_gb)
        stats = gpu_probe.read_stats(self.dev)
        for g in gpus:
            if g["device_index"] in stats:
                g["mem_used_mb"], g["util_pct"] = stats[g["device_index"]]
        # Telling the server which jobs we run lets it detect lost jobs (known to server but not reported).
        running = [{"assignment_id": aid, "pid": e["proc"].pid if e["proc"] else None} for aid, e in self.live.items()]
        reply = self.api.post("/agent/heartbeat", {"gpus": gpus, "running": running, "accepts_scripts": self.sandbox_ok})
        self.term_jobs = set(reply.get("term_jobs", []))
        self.last_stop = set(reply.get("stop_jobs", []))
        # JSON keys are strings, so convert them back to int job ids.
        self.stop_reasons = {int(k): v for k, v in reply.get("stop_reasons", {}).items()}
        return reply

    def ship_logs(self, e):
        """Send the next unsent piece of the job's log file to the server (live logs on the website)."""
        try:
            with open(e["log"], "rb") as f:
                # Continue exactly where the last upload ended.
                f.seek(e["offset"])
                data = f.read(CHUNK)
        except OSError:
            return
        if data:
            self.api.post("/agent/log", {"job_id": e["job_id"], "text": data.decode("utf-8", "replace")})
            e["offset"] += len(data)

    def poll_jobs(self, now):
        """Check every running job: has it exited, was it stopped, or did it run past its time limit?"""
        for e in self.live.values():
            if e["rc"] is not None or e["popen"] is None:
                continue
            # poll() returns None while the process is still running, else its exit code.
            rc = e["popen"].poll()
            # Stop requested via the website (user, or the GPU owner). The reason picks the message shown.
            if rc is None and e["job_id"] in self.last_stop:
                self.kill_job(e)
                e["rc"] = "killed"
                e["note"] = ("stopped by the GPU owner" if self.stop_reasons.get(e["job_id"]) == "owner" else "stopped by user")
            # Timeout: past the declared end time plus a small grace period (KILL_GRACE_S) -> force kill.
            elif rc is None and now > e["end"] + config.KILL_GRACE_S:
                self.kill_job(e)
                e["rc"] = "killed"
                e["note"] = ("session ended: time limit reached" if e["kind"] == "session"
                             else "stopped: exceeded the declared duration")
            elif rc is not None:
                e["rc"], e["note"] = rc, summarise(e["log"], rc)

    def finish_ready(self):
        """For jobs that ended: upload the remaining log, tell the server, and clean up the job folder."""
        for aid in list(self.live):
            e = self.live[aid]
            if e["rc"] is None:
                continue
            while True:
                # Keep shipping until a call sends nothing new, so no log tail is lost.
                before = e["offset"]
                self.ship_logs(e)
                if e["offset"] == before:
                    break
            # Tell the server the job is over; the note becomes its final status text.
            self.api.post("/agent/finish", {"job_id": e["job_id"], "assignment_id": aid, "note": e["note"]})
            print(f"[agent] job {e['job_id']} {e['note']}", flush=True)
            # Delete the user's files from the lender's laptop once the job is done.
            shutil.rmtree(os.path.join(self.jobsdir, f"job_{e['job_id']}"), ignore_errors=True)
            t = self.terms.pop(e["job_id"], None)
            if t:
                t.close()
            del self.live[aid]
            self._save_state()

    # ----------------------------------------------------------- interactive terminal
    # Interactive terminal: the container runs 'sleep infinity'; each terminal is a 'docker exec -i ...
    # bash' inside it.
    def _docker_exec(self, e):
        name = e["container"]
        # -i keeps stdin open so keystrokes can be piped in; same non-root user and /work folder as the
        # job.
        return ["docker", "exec", "-i", "-u", "1000:1000", "-w", "/work", name, "bash"]

    def pump_terminals(self):
        """Called often (every ~0.3 s) while someone has the terminal open: ship output, collect input.

        This is a relay: shell output goes up to the server, typed input comes back in the reply, and the agent
        writes it to the shell's stdin. The browser never connects to the laptop directly (polling, no open port).
        """
        # Only jobs that are still running can have a terminal.
        by_job = {e["job_id"]: e for e in self.live.values() if e["rc"] is None}
        for jid in list(self.terms):                                   # browser closed the terminal
            if jid not in self.term_jobs or jid not in by_job:
                self.terms.pop(jid).close()
        for jid in self.term_jobs:
            e = by_job.get(jid)
            if e is None or not (e["container"] or self.exec_factory is not self._docker_exec):
                continue                                               # terminals only exist inside the sandbox
            if jid not in self.terms:
                try:
                    # Open the shell lazily on first use.
                    self.terms[jid] = TermSession(self.exec_factory(e))
                except OSError as ex:
                    self.api.post("/agent/term", {"job_id": jid, "output": f"[terminal could not start: {ex}]\n"})
                    continue
            t = self.terms[jid]
            # Upload new output; the reply holds whatever the user typed in the browser since the last
            # call.
            reply = self.api.post("/agent/term", {"job_id": jid, "output": t.drain()})
            for chunk in reply.get("input", []):
                t.write(chunk)

    def step(self, now=None):
        """One full cycle, run every AGENT_POLL_S: heartbeat, launch, check jobs, ship logs, finish, terminals."""
        now = time.time() if now is None else now
        reply = self.heartbeat()
        for a in reply.get("launches", []):
            # The server repeats a launch until we report the job as running, so skip ones already started.
            if a["assignment_id"] not in self.live:
                self.launch(a)
        self.poll_jobs(now)
        for e in list(self.live.values()):
            self.ship_logs(e)
        self.finish_ready()
        self.pump_terminals()


def main(argv=None):
    """Command-line entry point: check the setup, clean up orphans, then loop forever polling the server."""
    ap = argparse.ArgumentParser(description="GPU provider agent")
    ap.add_argument("--server", required=True)
    # Token from the website (Providers > Register); can also come from the GPUSCHED_TOKEN variable.
    ap.add_argument("--token", default=os.environ.get("GPUSCHED_TOKEN"))
    ap.add_argument("--dev", action="store_true", help="no NVIDIA GPU: advertise a DEV GPU and run jobs on CPU")
    ap.add_argument("--dev-mem-gb", type=float, default=4.0)
    ap.add_argument("--docker", action="store_true", help="run jobs in Docker containers; REQUIRED to accept users' own code")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="container image with PyTorch (pull it first)")
    ap.add_argument("--docker-network", choices=["none", "bridge"], default="none",
                    help="'none' (default) = no internet inside jobs; 'bridge' lets jobs pip install / download data")
    ap.add_argument("--cpus", type=float, default=2.0)
    ap.add_argument("--ram-gb", type=float, default=8.0, help="CPU RAM limit per container")
    ap.add_argument("--unsafe-native-scripts", action="store_true",
                    help="DEVELOPMENT ONLY: run users' scripts directly on this laptop without Docker")
    ap.add_argument("--workdir", default=os.path.join(os.path.expanduser("~"), ".gpu_agent"))
    args = ap.parse_args(argv)
    if not args.token:
        sys.exit("token missing: pass --token or set GPUSCHED_TOKEN (create it on the website: Providers > Register)")
    if args.docker:
        try:
            # Fail early with a clear message if the Docker daemon is not running or the image is missing.
            subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, check=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            sys.exit("Docker is not running. Start Docker Desktop and try again.")
        img = subprocess.run(["docker", "image", "inspect", args.image], capture_output=True)
        if img.returncode != 0:
            sys.exit(f"Image missing. Run once:  docker pull {args.image}")
    os.makedirs(args.workdir, exist_ok=True)
    agent = Agent(Api(args.server, args.token), args.workdir, args.dev, args.dev_mem_gb, args.docker,
                  args.unsafe_native_scripts, args.image, args.docker_network, args.cpus, args.ram_gb)
    # Crash recovery on start: remove job processes/containers left by a previous run.
    killed = agent.kill_orphans()
    if killed:
        print(f"[agent] stopped {killed} job process(es)/container(s) left from the previous run", flush=True)
    mode = "Docker sandbox" if args.docker else ("UNSAFE native scripts" if args.unsafe_native_scripts else "prepared workloads only")
    print(f"[agent] connected to {args.server} ({mode})", flush=True)
    # 'warned' avoids printing the same error every loop; last_full is the time of the last full cycle.
    warned, last_full = False, 0.0
    while True:
        try:
            # Full heartbeat cycle every 2 s; in between, if a terminal is open, only pump it (fast ~0.3 s
            # feel).
            if time.time() - last_full >= config.AGENT_POLL_S:
                last_full = time.time()
                agent.step()
            elif agent.term_jobs:
                agent.pump_terminals()
            warned = False
        # Server down or restarting: just keep retrying; running jobs are not touched.
        except (urllib.error.URLError, OSError, ValueError) as e:
            if not warned:
                print(f"[agent] server unreachable ({e}); retrying", flush=True)
                warned = True
        # Short sleep while a terminal is open so typing feels responsive, otherwise relax.
        time.sleep(0.3 if agent.term_jobs else 0.5)


if __name__ == "__main__":
    main()
