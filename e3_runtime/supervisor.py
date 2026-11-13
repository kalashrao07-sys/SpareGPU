"""E3 supervisor: the parent process. Starts the web server (E4), the scheduler (E1) and the
provider monitor (E3) as separate OS processes, restarts any that dies, and checks that the stored
state is consistent after every restart.

    python run.py          (calls main() below)

Process isolation: each child has its own address space, so a crash of the scheduler leaves
the web pages (job status, GPU pool) available.

Recovery: the supervisor polls its children every second; a dead child is restarted and the database
is checked for consistency. Running jobs are on the provider laptops, so stopping or restarting the server
never disturbs them; the agents keep running and keep sending heartbeats.
"""
import os
import signal
import subprocess
import psutil
import sys
import time

import config
from e2_persistence import Repository

from . import recovery

# (name, python module) of each child process. They are started with 'python -m module'.
CHILDREN = [("web", "e4_web.app"), ("scheduler", "e1_scheduling.scheduler"), ("monitor", "e3_runtime.provider_monitor")]


class Supervisor:
    def __init__(self, repo):
        self.repo = repo
        self.procs = {}
        # Timestamps of recent restarts per child (used to detect a crash loop).
        self.restarts = {name: [] for name, _ in CHILDREN}
        self._logs = {}

    def spawn(self, name, module):
        """Start one child as a separate OS process whose output goes to its own log file."""
        config.ensure_dirs()
        logf = open(os.path.join(config.LOG_DIR, f"{name}.log"), "a", encoding="utf-8")
        # Keep the file object referenced so it stays open while the child runs.
        self._logs[name] = logf
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # Same Python interpreter, '-u' = unbuffered so log lines appear at once; cwd lets imports find
        # config.
        p = subprocess.Popen([sys.executable, "-u", "-m", module], cwd=config.BASE_DIR,
                             stdout=logf, stderr=subprocess.STDOUT, creationflags=flags)
        self.procs[name] = p
        print(f"[E3] started {name} (pid {p.pid})", flush=True)

    def start_all(self):
        for name, module in CHILDREN:
            self.spawn(name, module)

    def check_children(self):
        """Restart any child that has exited. Returns the names restarted."""
        restarted = []
        now = time.time()
        for name, module in CHILDREN:
            p = self.procs.get(name)
            # poll() is None means the child is still alive; nothing to do.
            if p is not None and p.poll() is None:
                continue
            code = None if p is None else p.returncode
            # Keep only restarts from the last 60 s.
            hist = [t for t in self.restarts[name] if now - t < 60]
            self.restarts[name] = hist + [now]
            print(f"[E3] {name} exited (code {code}); restarting", flush=True)
            # Check the stored data before bringing the child back.
            problems = recovery.verify(self.repo)
            print("[E3] stored state is consistent" if not problems else f"[E3] INCONSISTENT: {problems}", flush=True)
            # Already 5 restarts within a minute: wait 5 s so a broken child does not spin the CPU.
            if len(hist) >= 5:
                time.sleep(5)                       # crash loop: slow down
            self.spawn(name, module)
            restarted.append(name)
        return restarted

    def stop_all(self):
        """Ask all children to stop (terminate), wait up to 5 s in total, then force-kill any that remain."""
        for name, p in self.procs.items():
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 5
        for name, p in self.procs.items():
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                recovery.kill_tree(psutil.Process(p.pid))
        # running jobs live on the provider laptops and are NOT disturbed by a server stop


def main():
    """Entry point of run.py: set up the database and seed users, then supervise forever until Ctrl+C."""
    config.ensure_dirs()
    repo = Repository()
    repo.init()
    # Create the default demo accounts on first run (skipped for ones that already exist).
    for username, password, role in config.SEED_USERS:
        if config.BEHIND_PROXY and role != "admin":
            continue                               # public host: no default-password demo accounts, people sign up / admin adds them      # first run: create the seed accounts
        try:
            uid = repo.create_user(username, password, role)
            if config.CREDITS_ENABLED and config.PORTAL == "exchange" and role == "submitter":
                repo.credit(uid, config.SIGNUP_CREDITS, "Welcome credits")
            print(f"[E3] created {role} account: {username}")
        except Exception:
            pass
    print(f"[E3] portal: {config.PORTAL}; data folder: {config.DATA_DIR}")
    print(f"[E3] server starting; providers: {[p['name'] for p in repo.list_providers()] or 'none registered yet'}")
    problems = recovery.verify(repo)
    print("[E3] stored state is consistent" if not problems else f"[E3] INCONSISTENT at start: {problems}")
    sup = Supervisor(repo)
    # Turn SIGTERM into KeyboardInterrupt so 'terminate' also reaches the clean shutdown below.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # clean stop on terminate too
    sup.start_all()
    print(f"[E3] open http://localhost:{config.PORT}  (Ctrl+C to stop)", flush=True)
    try:
        while True:
            # Supervision loop: look at the children once per second.
            sup.check_children()
            time.sleep(1)
    except KeyboardInterrupt:
        print("[E3] stopping ...", flush=True)
    finally:
        sup.stop_all()


if __name__ == "__main__":
    main()
