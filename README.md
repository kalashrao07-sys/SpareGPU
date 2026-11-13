# SpareGPU

Got a GPU sitting idle? Share it. Need a GPU and don't have one? Rent one.

SpareGPU is a small platform that connects people with spare GPUs to people who need them. A scheduler decides who gets which GPU, so a job never lands on a card that can't handle it, and two people never end up on the same GPU at once. Your code runs in a locked-down Docker sandbox on the lender's machine, and you see real output, or an honest error if nothing is available.

This repo contains **two portals** built on the same engine:

| Portal | What it is | Command | Port |
|---|---|---|---|
| **SpareGPU** | The marketplace. Lenders list GPUs, renters rent them. | `python run.py --portal exchange` | 5100 |
| **KLE Tech portal** | A college GPU lab. Students submit code, the admin manages the lab's GPUs. | `python run.py --portal college` | 5000 |

They share the scheduler, agent and sandbox code, but each has its own database, accounts and look.

---

## What you can do

**On SpareGPU**
- Lend a GPU: register your laptop or PC, set a price per hour (or free), the hours it's available, and whether it's private (only people you trust).
- Pause or stop sharing any time.
- Rent a GPU: browse the market, pick one that fits, and get an editor to run your code. The editor only appears once a GPU is actually allocated to you.

**On the KLE Tech portal**
- Students get a VS Code-style editor: folders, tabs, syntax highlighting, create/rename/delete files, Run and Stop, a live output panel and a GPU terminal.
- The admin adds servers with their GPU model and memory, sees each GPU as available, occupied, offline or disabled, watches live utilisation, and can edit, disable or remove anything.

**On both**
- If no GPU is online, nothing starts and the portal tells you why. No fake runs.
- If a GPU machine drops off mid-job, the job goes back in the queue.
- Jobs can be stopped at any time and end on their own at their time limit.

---

## How matching works

Hard rules that are never broken:

- The GPU has enough memory for the job.
- One job per GPU at a time.
- The GPU's owner has the sandbox switched on (needed to run your own code).
- The job fits the owner's available hours.
- The price is within the renter's maximum.
- Private GPUs only serve people the owner trusts.
- The assignment is saved in one atomic step, so a crash can't leave half a booking.

Among the GPUs that pass, it picks by: priority and arrival order, then lowest price, then the GPU that's been idle the shortest, then a GPU of the type you used before.

---

## Run it on your own machine

You need Python 3.10+.

```bash
git clone https://github.com/kalashrao07-sys/SpareGPU.git
cd SpareGPU
pip install -r requirements.txt

python run.py --portal exchange     # SpareGPU  -> http://localhost:5100
python run.py --portal college      # KLE Tech  -> http://localhost:5000
```

Sign in as `admin` with the password from the `GPUSCHED_ADMIN_PASSWORD` environment variable (the default `admin-change-me` is for local testing only).

### Connect a real GPU

The machine that owns the GPU runs a small agent. It needs Linux with **Docker** and the **NVIDIA container toolkit**, and `nvidia-smi` must work.

1. In the portal, register the machine (SpareGPU: *Share my GPU*; KLE Tech: *Admin → Devices → Add a server*). You get a token, shown once.
2. On the GPU machine:

```bash
python -m e3_runtime.agent --server http://YOUR_SERVER_ADDRESS --token YOUR_TOKEN
```

The agent reports every couple of seconds. If it goes quiet for 20 seconds, the machine counts as offline and its jobs are re-queued.

The agent and the server can be on different machines. If they aren't on the same network, use a tunnel such as Tailscale, or host the server publicly (below).

---

## Deploy on Render

`render.yaml` defines both portals as two web services from this one repo.

1. Push this repo to GitHub.
2. On Render: **New → Blueprint**, pick the repo. It creates `sparegpu` and `kle-gpu-portal`. Delete the one you don't need.
3. Set `GPUSCHED_ADMIN_PASSWORD` for each service in the dashboard (a strong one). `GPUSCHED_SECRET_KEY` is generated for you.
4. Deploy. Every push to `main` redeploys automatically.
5. Start the agent on the GPU machine with `--server https://YOUR-SERVICE.onrender.com`.

Settings you can change through environment variables:

| Variable | What it does |
|---|---|
| `GPUSCHED_PORTAL` | `exchange` or `college` (set by `--portal`) |
| `GPUSCHED_ADMIN_PASSWORD` | Password for the seeded admin account |
| `GPUSCHED_SECRET_KEY` | Signs login sessions |
| `GPUSCHED_TZ` | Time zone, e.g. `Asia/Kolkata` |
| `GPUSCHED_DATA_DIR` | Where the database and logs live |
| `PORT` | Set by Render automatically |

**Free tier heads-up.** Render's free plan wipes the local disk on every restart or deploy and puts the app to sleep when idle. That means the database (users, listings, jobs) resets. Fine for a quick look; for a real demo either use a paid plan with a persistent disk or move the data to a hosted database (a Supabase Postgres port is planned). Wake the service a few minutes before presenting.

---

## Project layout

```
e1_scheduling/   Picks the GPU: priority queue, rules, ranking
e2_persistence/  Database schema and all reads/writes, including the atomic booking
e3_runtime/      The agent, Docker sandbox launcher, heartbeat watchdog, GPU probe
e4_web/          Flask app: login, editor, market, admin pages, templates, styling
rules.py         The shared constraint checks
config.py        All settings
run.py           Starts everything for the chosen portal
render.yaml      Render blueprint for both portals
```

---

## Tests

```bash
python -m unittest discover -s . -t . -p "test_*.py"
```

About 98 tests cover the scheduler, database, agent, web app and 12 end-to-end scenarios: simultaneous submissions, two jobs fighting for one GPU, queueing, a GPU going offline mid-job, cancel and timeout, admin changes, file operations in the editor, access control, and a database failure mid-booking. A full run takes around a minute and a half.

---

## Security in short

Your code runs in Docker with no network, all Linux capabilities dropped, no privilege escalation, a read-only filesystem, a non-root user, and memory and CPU limits, and only the one allocated GPU is visible. Agent tokens are stored hashed. Users can't see each other's jobs, logs or files, and admin routes are checked on the server.

---

## Status and roadmap

Working: both portals, the scheduler, the editor, admin GPU management, failure handling and the test suite.
Still to do: verify end-to-end runs on a real GPU under Ubuntu + Docker, and move storage to Supabase Postgres so data survives restarts on free hosting.

## Team

Built by **Team 7** at KLE Technological University, Belagavi, guided by Prof. Amey Muchandi: Kalash Rao (scheduling engine, project lead), Komal Suryavanshi (persistence), Omkar Naik (runtime and agent), Bhoomika Kalal (web).