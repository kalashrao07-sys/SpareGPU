# GPU Sharing Platform with a Constraint-Based Job Scheduler (Team 7, KLE Tech)

Guide: Prof. Amey Muchandi · Team: Kalash Rao (E1), Komal Suryavanshi (E2), Omkar Naik (E3), Bhoomika Kalal (E4)

GPU owners (providers) register their laptops. Other students request GPU time through a website form. The
constraint-based scheduler decides who runs where and when; the provider's agent runs the job and the website shows live logs.

```
  Kalash / Komal / Bhoomika (browser)            Omkar's laptop (GPU)
          |                                       agent  (e3_runtime.agent)
          v                                          ^   |
   SERVER laptop: website (E4) + DB (E2) + scheduler (E1) + monitor (E3)
          <------------- HTTP heartbeat / logs / finish -------------
```

| Folder | Engine | Owner | Runs on |
|---|---|---|---|
| e1_scheduling | Scheduling & Constraint (HC1,HC2,HC4, SP1-SP3, heap queue) | Kalash | server |
| e2_persistence | Persistence & Transaction (SQLite, atomic assignment commit, providers, logs) | Komal | server |
| e3_runtime | Process Isolation & Recovery (supervisor, provider monitor, **agent**) | Omkar | server + provider |
| e4_web | Submission & Status Interface (forms, status, pool, providers, agent API) | Bhoomika | server |

## 1. Install (every laptop)
```
pip install -r requirements.txt
```
Provider laptop with NVIDIA GPU: also install PyTorch from https://pytorch.org (CUDA build).
Without PyTorch the prepared jobs fall back to NumPy on CPU (development only).

## 2. Server laptop (Kalash or Bhoomika)
```
python run.py
```
First run creates accounts `admin, kalash, komal, omkar, bhoomika` with password `<name>-change-me`. **Change them** before the demo
(edit SEED_USERS in config.py before the first run, or delete data/ and rerun). Open `http://SERVER_IP:5000`
(find SERVER_IP with `ipconfig`). Allow Python through the Windows firewall for private networks.
Start fresh: delete the `data` folder (this also removes old logs).

## 3. Provider laptop (Omkar): one-time setup
1. **Docker Desktop** (WSL2 backend; Windows 11 Home is fine): install, restart, make sure it says "running".
   NVIDIA driver 592.82 already supports GPU in WSL2; no extra CUDA install is needed on Windows.
2. **Check GPU inside Docker** (this is the key test, tell the team the result):
   ```
   docker run --rm --gpus all pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   It must print `True NVIDIA GeForce RTX 2050`. (First run downloads ~3 GB.)
3. `pip install flask psutil` in the project folder.
4. On the website: log in as `omkar` > **Providers** > register `omkar-laptop`, copy the token; the admin approves it.
5. Start the agent (leave the window open, laptop awake and plugged in):
```
python -m e3_runtime.agent --server http://SERVER_ADDRESS:5000 --token PASTE_TOKEN --docker
```
Without `--docker` the laptop still runs the two prepared workloads natively, but **refuses users' own code**.
`--docker-network bridge` gives jobs internet (for pip/datasets); default is none. `--ram-gb 8 --cpus 2` set container limits.
Closing the laptop or the agent makes the provider offline after 20 s: running jobs are re-queued. On agent restart, leftover containers are removed.

## 4. Overview
1. **Pool**: Omkar's RTX 2050 (4 GB) online and free. **Connect** page shows the addresses (Tailscale one for remote users).
2. **Editor**: Kalash writes `proj/train.py` in the browser editor and saves; Komal and Bhoomika do the same in their own projects (each sees only their own files).
3. Three people press **Run on GPU** within seconds (priorities 2, 5, 3). Scheduler order: Komal, Bhoomika, Kalash. Others show why they wait.
4. Open a job: **live log** streams; Stop button ends it; the pool page shows live GPU use.
5. Book an **Interactive session**: the job page shows a **Terminal**; run `nvidia-smi` and `python -c "import torch;print(torch.cuda.is_available())"`.
6. Isolation demo (inside a session): `ls /` (no laptop files), `ping 8.8.8.8` (no network), `touch /x` (read-only), `cat /etc/shadow` (denied).
7. Failure demo: close the agent mid-job; ~20 s later it is re-queued. Restart the agent and it runs again.

## 5. How the three added features work
* **Remote access**: Tailscale (free VPN) on the server laptop, the provider laptop and each user's device. Use the 100.x address shown on **Connect**.
  Traffic is encrypted by Tailscale; never port-forward the router.
* **Docker isolation**: user code only ever runs in a container with no network, read-only root, all capabilities dropped, non-root user,
  memory/CPU/process limits, exactly one GPU, and only a copy of the user's own files. PyTorch GPU memory is capped to the declared amount.
* **Editor**: Monaco (the VS Code editor) from a CDN, with a plain text box fallback when offline. Files are stored on the server per user;
  a copy is sent to the provider when the job starts and deleted when it ends. Download a **backup .zip** from the Editor page; also keep Git.
* **Terminal**: the provider is behind NAT, so the agent polls the server (about every 0.3 s while the terminal is open) and relays lines to
  `docker exec` inside your job's container. Only the job's owner can open it. It is line-based (no vim/nano).

## 6. Tests
```
python -m unittest discover -s . -t . -p "test_*.py"
```
67 tests: constraints, atomic commit, forced-kill of the scheduler inside the commit transaction (5 runs), agent protocol,
provider offline / agent restart / server restart, path-escape, Docker command lock-down, editor privacy, terminal relay, stop button, token checks, role guard, CSRF, validation.

## Current Limits 
* The Docker path (containers, GPU passthrough, `docker exec` terminal) could not be run in the build environment: command construction,
  agent logic and the relay are unit-tested, but the real container run must be verified on Omkar's laptop (step 2 above).
* GPU memory is capped only for PyTorch code; TensorFlow users' declared memory is advisory (the job is still killed on timeout).
* `--unsafe-native-scripts` exists for development only; never use it on a real provider.
* Plain HTTP on the same network; use Tailscale (private VPN) for remote access, never expose port 5000 to the internet.
* Downloadable result files and full-TTY terminal (xterm.js) are future work.
