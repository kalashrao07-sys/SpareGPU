--
-- E2 PERSISTENCE: database schema (SQLite). Read by db.init_schema() on every start.
--
-- How to read this file:
--   * Every statement is CREATE ... IF NOT EXISTS, so running it again never damages existing data.
--   * PRIMARY KEY AUTOINCREMENT = the database numbers rows 1, 2, 3 ... and never re-uses a number.
--   * REFERENCES x(y) = FOREIGN KEY (enforced because db.connect sets PRAGMA foreign_keys=ON):
--     a row cannot point at a parent row that does not exist, and a parent with children cannot be deleted.
--   * CHECK (...) = a rule the database itself enforces on every INSERT/UPDATE, so even buggy code
--     cannot store invalid data (e.g. memory_gb <= 0).
--   * Booleans are INTEGER 0/1 (SQLite has no boolean type). Times are epoch seconds (REAL).
--   * Versions: v2 -> v3 (project files, terminal relay) -> v4 (SpareGPU: price, hours, trust, credits).
--     Columns added in v4 are listed again in db.py (_V4_COLUMNS) so old databases can be upgraded.
--
-- E2 schema, version 4. Times are stored as epoch seconds (REAL).
-- app_user: every person who can log in (renters, lenders and admins are all users).
CREATE TABLE IF NOT EXISTS app_user (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    -- UNIQUE: two users cannot share a username (create_user relies on this error).
    username      TEXT NOT NULL UNIQUE,
    -- Only the salted hash is stored, never the password itself.
    password_hash TEXT NOT NULL,
    -- Two roles only: normal users ('submitter') and administrators.
    role          TEXT NOT NULL CHECK (role IN ('submitter','admin'))
);

-- A GPU provider = one laptop running the agent. Its token is stored only as a hash.
-- provider: the owner's laptop is the provider, it runs the agent that actually executes jobs.
CREATE TABLE IF NOT EXISTS provider (
    provider_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    -- The lender / laptop owner. Used for HC7 (owner may always use a private GPU) and for paying earnings.
    owner_user_id   INTEGER NOT NULL REFERENCES app_user(user_id),
    name            TEXT NOT NULL UNIQUE,
    -- SHA-256 hash of the agent's secret token. UNIQUE so a token identifies exactly one provider.
    token_hash      TEXT NOT NULL UNIQUE,
    approved        INTEGER NOT NULL DEFAULT 0,      -- the admin must approve a provider
    accepts_scripts INTEGER NOT NULL DEFAULT 0,      -- provider allows users' own scripts (set by the agent)
    workspace_root  TEXT,                            -- folder on the provider's laptop holding user workspaces
    last_seen       REAL,                            -- last heartbeat from the agent
    created_at      REAL NOT NULL,
    -- v4 (SpareGPU): terms set by the lender. College portal leaves them at the defaults.
    -- Price in credits per hour. CHECK >= 0 (no negative prices). Used by HC6 and by complete_job settlement.
    price_per_hour  REAL NOT NULL DEFAULT 0 CHECK (price_per_hour >= 0),   -- 0 = free
    -- Daily availability window as minutes after midnight (0..1439). NULL = no limit. Used by HC5.
    avail_from      INTEGER CHECK (avail_from IS NULL OR (avail_from >= 0 AND avail_from < 1440)),  -- minutes after midnight
    -- avail_to may be smaller than avail_from: the window then crosses midnight (decided in rules.fits_window).
    avail_to        INTEGER CHECK (avail_to IS NULL OR (avail_to >= 0 AND avail_to < 1440)),
    sharing_enabled INTEGER NOT NULL DEFAULT 1,      -- lender's pause switch: 0 = no new jobs
    private_only    INTEGER NOT NULL DEFAULT 0,      -- 1 = only the owner and trusted users may rent
    description     TEXT
);

-- gpu: one graphics card. Belongs to a provider (a lender's laptop) or to the server itself.
CREATE TABLE IF NOT EXISTS gpu (
    gpu_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id    INTEGER REFERENCES provider(provider_id),   -- NULL only for test / simulated records
    name           TEXT NOT NULL,
    gpu_type       TEXT NOT NULL,
    -- GPU memory must be positive. HC1 compares this with job.memory_gb.
    memory_gb      REAL NOT NULL CHECK (memory_gb > 0),
    -- 'removed' = soft delete by the admin, so old assignments keep pointing at a valid row.
    status         TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','removed')),
    -- 1 = fake GPU for tests/demos, 0 = a real card found by nvidia-smi.
    is_simulated   INTEGER NOT NULL DEFAULT 0,
    device_index   INTEGER,                 -- nvidia-smi index on the provider's laptop
    -- HC2 'one job per GPU' is stored here: a single column can hold only ONE job id at a time.
    current_job_id INTEGER REFERENCES job(job_id),   -- NULL = free (HC2)
    free_since     REAL NOT NULL,           -- when the GPU last became free (used by SP2)
    mem_used_mb    REAL,                    -- live stats reported by the agent
    util_pct       REAL,
    stats_at       REAL
);
-- A provider cannot report the same device_index twice. The WHERE makes the index partial: local
-- GPUs (provider_id NULL) are excluded because NULLs are not compared by a UNIQUE index anyway.
CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_provider_dev ON gpu(provider_id, device_index) WHERE provider_id IS NOT NULL;

-- job: one request to use a GPU. Moves queued/held -> running -> completed.
CREATE TABLE IF NOT EXISTS job (
    job_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES app_user(user_id),
    -- Empty string (not NULL) when the user gave no project name.
    project_name TEXT NOT NULL DEFAULT '',
    description  TEXT,
    -- template = a prepared workload, script = the user's own .py file, session = interactive terminal.
    kind         TEXT NOT NULL DEFAULT 'template' CHECK (kind IN ('template','script','session')),
    -- Only the two prepared workloads are allowed.
    template     TEXT CHECK (template IS NULL OR template IN ('matmul','train')),
    script_path  TEXT,                      -- relative path inside the owner's workspace (kind = script)
    framework    TEXT,                      -- information only
    -- (CHECK allows NULL, which means 'not given'.)
    cpu_ram_gb   REAL CHECK (cpu_ram_gb IS NULL OR cpu_ram_gb > 0),   -- information only
    -- GPU memory the job needs. HC1. Must be positive.
    memory_gb    REAL NOT NULL CHECK (memory_gb > 0),                  -- GPU memory: HC1
    -- Declared run time in minutes. Used for the booking window (HC5) and the credit estimate (HC6).
    duration_min REAL NOT NULL CHECK (duration_min > 0),
    -- 1 (lowest) .. 5 (highest). The scheduler serves higher priority first.
    priority     INTEGER NOT NULL CHECK (priority BETWEEN 1 AND 5),
    deadline     TEXT,                      -- metadata only (FR2), never used for scheduling
    submitted_at REAL NOT NULL,
    -- queued = waiting, held = waiting with a reason (held_reason), running = on a GPU, completed = finished/cancelled.
    state        TEXT NOT NULL CHECK (state IN ('queued','held','running','completed')),
    -- Why a job is held (e.g. no GPU is big enough). NULL when not held.
    held_reason  TEXT,
    stop_requested INTEGER NOT NULL DEFAULT 0,   -- 0 none, 1 the user pressed Stop, 2 the lender stopped it
    max_price_per_hour REAL CHECK (max_price_per_hour IS NULL OR max_price_per_hour >= 0),  -- v4: renter's limit (NULL = any)
    cost REAL,                                    -- v4: credits charged when the job ended
    -- Message shown to the user at the end (success text, 'cancelled by user', requeue reason ...).
    result_note  TEXT,
    -- Table-level CHECK: ties 'kind' to which columns may be filled, so a row can never be half template,
        -- half script. template jobs: template set, no script. script jobs: script set, no template. session: neither.
    CHECK ((kind = 'template' AND template IS NOT NULL AND script_path IS NULL) OR
           (kind = 'script' AND script_path IS NOT NULL AND template IS NULL) OR
           (kind = 'session' AND script_path IS NULL AND template IS NULL))
);

-- assignment: the scheduler's decision. One row = one job placed on one GPU for a time window.
-- It is written only by Repository.commit_assignment, in the same transaction that sets job.state='running'
-- and gpu.current_job_id (HC3). requeue_job deletes the row if the job has to go back to the queue.
CREATE TABLE IF NOT EXISTS assignment (
    assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- UNIQUE: a job can have at most one assignment (no double booking of one job).
    job_id        INTEGER NOT NULL UNIQUE REFERENCES job(job_id),
    gpu_id        INTEGER NOT NULL REFERENCES gpu(gpu_id),
    -- Planned booking window (epoch seconds). start_time <= end_time is checked by the scheduler.
    start_time    REAL NOT NULL,
    end_time      REAL NOT NULL,
    -- Which scheduling rule made the decision (kept so the choice can be explained).
    deciding_rule TEXT NOT NULL,
    -- When the transaction committed.
    committed_at  REAL NOT NULL,
    launched      INTEGER NOT NULL DEFAULT 0,   -- 1 once the agent reported the process started
    -- When the agent started the process, and its operating-system process id.
    launched_at   REAL,
    pid           INTEGER,
    rate          REAL NOT NULL DEFAULT 0       -- v4: credits per hour this renter pays (after trust discount)
);

-- Indexes speed up the most common lookups (open jobs by state, a user's jobs). They do not change results.
CREATE INDEX IF NOT EXISTS idx_job_state ON job(state);
CREATE INDEX IF NOT EXISTS idx_job_user ON job(user_id);

-- (composite PRIMARY KEY (user_id, path): each user has at most one file per path, so UPSERT is possible.)
-- v3: the user's project files (edited in the website's code editor), kept on the server as the single copy.
CREATE TABLE IF NOT EXISTS project_file (
    user_id    INTEGER NOT NULL REFERENCES app_user(user_id),
    path       TEXT NOT NULL,
    content    TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, path)
);

-- Rows here are deleted by complete_job / requeue_job / term_clear when the job ends.
-- term_input: text typed in the browser, waiting for the agent. delivered=1 once the agent has fetched it.
-- v3: relay for the interactive terminal (provider laptops are behind NAT, so the agent polls for input).
CREATE TABLE IF NOT EXISTS term_input (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id    INTEGER NOT NULL REFERENCES job(job_id),
    data      TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0
);
-- term_output: text produced by the job, read by the browser using 'id > last seen id'.
CREATE TABLE IF NOT EXISTS term_output (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES job(job_id),
    data   TEXT NOT NULL
);
-- term_state: when the browser last polled. The terminal counts as open for a few seconds after that.
CREATE TABLE IF NOT EXISTS term_state (
    job_id       INTEGER PRIMARY KEY REFERENCES job(job_id),
    browser_seen REAL NOT NULL
);

-- Composite PRIMARY KEY: a user can be trusted by a provider only once.
-- v4 (SpareGPU): users a lender trusts: they rent that lender's GPU for free (and may use it when it is private).
CREATE TABLE IF NOT EXISTS provider_trusted (
    provider_id INTEGER NOT NULL REFERENCES provider(provider_id),
    user_id     INTEGER NOT NULL REFERENCES app_user(user_id),
    PRIMARY KEY (provider_id, user_id)
);

-- ledger is APPEND-ONLY: balances are never edited, a new row is added for each movement of credits,
-- so the full history is kept. Balance = SUM(amount). complete_job writes a negative row for the renter and
-- a positive row for the lender with the same amount (credits move, none are created).
-- v4: virtual credits. Balance = SUM(amount). Positive = earned/granted, negative = spent. Not real money.
CREATE TABLE IF NOT EXISTS ledger (
    entry_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES app_user(user_id),
    -- Positive = earned or granted, negative = spent.
    amount     REAL NOT NULL,
    note       TEXT NOT NULL,
    -- The job this entry belongs to (NULL for e.g. welcome credits).
    job_id     INTEGER REFERENCES job(job_id),
    created_at REAL NOT NULL
);
-- Fast balance / history lookup per user.
CREATE INDEX IF NOT EXISTS idx_ledger_user ON ledger(user_id);
