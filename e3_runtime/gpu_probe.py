"""E3 (agent side): find the laptop's GPUs and read live memory / utilisation with nvidia-smi."""
import subprocess


def _smi(query):
    """Run nvidia-smi for the given fields and return rows as lists of strings (one row per GPU)."""
    try:
        # csv,noheader,nounits gives plain numbers like '0, NVIDIA GeForce RTX 3060, 12288'.
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        # Driver missing: tell the lender what to do instead of crashing with a vague error.
        raise RuntimeError("nvidia-smi not available - is the NVIDIA driver installed? "
                           "(use --dev only on a machine without an NVIDIA GPU)") from e
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def probe_gpus(dev=False, dev_mem_gb=4.0):
    """[{'device_index','name','gpu_type','memory_gb'}]

    Static facts about each GPU, sent in every heartbeat. --dev fakes one GPU so the system can be demoed
    on a laptop with no NVIDIA card."""
    if dev:
        return [{"device_index": 0, "name": "DEV (no real GPU)", "gpu_type": "DEV", "memory_gb": float(dev_mem_gb)}]
    gpus = []
    for idx, name, mem_mib in _smi("index,name,memory.total"):
        # nvidia-smi reports memory in MiB; divide by 1024 for GB. gpu_type is the short name (e.g. 'RTX
        # 3060').
        gpus.append({"device_index": int(idx), "name": name,
                     "gpu_type": name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").strip(),
                     "memory_gb": round(float(mem_mib) / 1024.0, 2)})
    return gpus


def read_stats(dev=False):
    """{device_index: (mem_used_mb, util_pct)}; empty when unavailable.

    Live numbers shown on the website's GPU pool page. Failure is not fatal, the stats are just left out."""
    if dev:
        return {0: (0.0, 0.0)}
    try:
        return {int(i): (float(m), float(u)) for i, m, u in _smi("index,memory.used,utilization.gpu")}
    except (RuntimeError, ValueError):
        return {}
