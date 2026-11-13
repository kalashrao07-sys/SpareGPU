"""Shared settings for all four engines. Edit values here, nowhere else."""
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Which site this process serves: "college" (KLE GPU booking portal) or "exchange" (SpareGPU, people lend idle GPUs).
# Start with:  python run.py --portal college   |   python run.py --portal exchange
# Each portal has its own folder, database, accounts and port; the scheduler, agent and sandbox code are shared.
PORTAL = os.environ.get("GPUSCHED_PORTAL", "college")
if PORTAL not in ("college", "exchange"):
    raise SystemExit("GPUSCHED_PORTAL must be 'college' or 'exchange'")
_DEFAULT_DATA = os.path.join(BASE_DIR, "data") if PORTAL == "college" else os.path.join(BASE_DIR, "data", "exchange")
DATA_DIR = os.environ.get("GPUSCHED_DATA_DIR", _DEFAULT_DATA)
DB_PATH = os.environ.get("GPUSCHED_DB", os.path.join(DATA_DIR, "scheduler.db"))
LOG_DIR = os.path.join(DATA_DIR, "logs")            # process logs (server side)
JOBLOG_DIR = os.path.join(DATA_DIR, "joblogs")      # live job output, one file per job (server side)
LOG_MAX_BYTES = 2_000_000                            # per job; the rest is dropped with a note
# (os.environ.get(name, default) reads an environment variable, so a host such as Render can change a setting
#  without editing code; the default after the comma is used on a normal laptop.)

# Simulated GPU records are OFF until the guide approves them (admin "add GPU" form is refused).
# "Simulated" = a GPU row with no real laptop behind it. Off so only real, agent-reported GPUs get jobs.
ALLOW_SIMULATED_GPUS = False

SCHEDULER_POLL_S = 1.0      # E1: how often a scheduling pass runs
AGENT_POLL_S = 2.0          # E3 agent: how often it reports to the server and asks for work
PROVIDER_OFFLINE_S = 20     # a provider silent for this long is offline; its running jobs are re-queued
KILL_GRACE_S = 15           # E3 agent: extra time after the declared duration before a job is force-killed
MAX_DURATION_MIN = 60       # E4: upper limit accepted on the form (protects the shared laptop)
MAX_MEMORY_GB = 64          # E4: sanity limit on the form; HC1 does the real check
CUDA_OVERHEAD_GB = 0.4      # job scripts keep this much of the declared memory free for CUDA itself

HOST = os.environ.get("GPUSCHED_HOST", "0.0.0.0")   # 0.0.0.0 = reachable from other devices
PORT = int(os.environ.get("PORT") or os.environ.get("GPUSCHED_PORT", "5000" if PORTAL == "college" else "5100"))   # $PORT is set by Render
BEHIND_PROXY = bool(os.environ.get("RENDER") or os.environ.get("GPUSCHED_BEHIND_PROXY"))   # HTTPS terminated by the host
SECURE_COOKIES = BEHIND_PROXY or bool(os.environ.get("GPUSCHED_SECURE_COOKIES"))
# Availability windows and shown times use this zone (Render runs on UTC). e.g. GPUSCHED_TZ=Asia/Kolkata
# time.tzset() exists on Linux/macOS (where Render runs) but not on Windows, hence the hasattr guard.
if os.environ.get("GPUSCHED_TZ") and hasattr(time, "tzset"):
    os.environ["TZ"] = os.environ["GPUSCHED_TZ"]
    time.tzset()

# Seed accounts created by `python -m e2_persistence.seed`. CHANGE THESE before the demo.
SEED_USERS = [
    ("admin", os.environ.get("GPUSCHED_ADMIN_PASSWORD", "admin-change-me"), "admin"),
    ("kalash", "kalash-change-me", "submitter"),
    ("komal", "komal-change-me", "submitter"),
    ("omkar", "omkar-change-me", "submitter"),
    ("bhoomika", "bhoomika-change-me", "submitter"),
]

# SpareGPU (exchange) settings. Credits are VIRTUAL points, not money: no payment gateway in this version.
SIGNUP_CREDITS = 100          # every new SpareGPU account starts with this many demo credits
PROVIDER_AUTO_APPROVE = True  # exchange: a lender's laptop is usable at once (admin can still suspend it)
CURRENCY = "₹"
CREDITS_ENABLED = False       # no wallet: prices are only the lender's listed rate, settled directly between people

# The ONLY workloads a user can choose. No user text ever reaches a command line.
FRAMEWORKS = ["PyTorch", "TensorFlow", "Other"]   # recorded on the job (information only)

WORKLOADS = {
    "matmul": "Matrix multiplication benchmark",
    "train": "Small neural network training",
}

def ensure_dirs():
    """Create the data, log and job-log folders if they do not exist yet (safe to call repeatedly)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(JOBLOG_DIR, exist_ok=True)
