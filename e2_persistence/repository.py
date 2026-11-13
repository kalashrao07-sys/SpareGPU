"""E2 Persistence & Transaction Engine: the ONLY code in the project that talks SQL.

WHAT THIS ENGINE DOES
    Every other engine (E1 scheduler, E3 execution/agent, E4 web/validation) calls the methods of
    the Repository class below and gets plain Python dicts / lists / numbers back. Nobody else
    opens the database, so all the rules about stored data live in exactly one place.

KEY DESIGN IDEAS (be ready to explain each one)
    1. SQLite in WAL mode (see db.py): readers do not block the writer and the writer does not
       block readers, so the web pages can read while the scheduler writes.
    2. Foreign keys are switched ON (PRAGMA foreign_keys=ON) and tables have CHECK constraints
       (see schema.sql), so the database itself refuses nonsense such as memory_gb <= 0.
    3. Manual transactions: connections use isolation_level=None, and _tx() issues
       "BEGIN IMMEDIATE" itself. BEGIN IMMEDIATE takes the database WRITE LOCK at the start, so two
       writers can never interleave: the second one waits (busy_timeout) until the first commits.
    4. commit_assignment() is the heart of the engine. It writes the assignment row, flips the job
       to 'running' and marks the GPU occupied in ONE transaction (all or nothing, NFR1), and it
       RE-CHECKS the hard constraints HC1..HC7 inside that transaction. The scheduler (E1) decided
       using data it read a moment earlier; by the time it commits, another process may have changed
       things. Re-checking under the write lock means a stale decision is rejected instead of
       producing a double booking (race-condition safety).
    5. Provider tokens are stored ONLY as a SHA-256 hash (token_hash). The plaintext token is
       shown once at creation, so a leaked database does not leak working tokens.
    6. Passwords are stored only as salted PBKDF2 hashes (werkzeug), never as plain text.
    7. Schema versions: v2 -> v3 (project files, terminal relay) -> v4 (SpareGPU: price, hours,
       trusted users, credits ledger). db.init_schema() migrates v3 -> v4 in place.
    8. Times are epoch seconds (float). Every method that needs "now" takes an optional now=
       argument so tests can control the clock.

THE HARD CONSTRAINTS (HC) ENFORCED OR CHECKED HERE
    HC1 GPU memory >= job memory.            HC2 at most one running job per GPU.
    HC3 assignment is atomic: assignment row + job 'running' + GPU occupied, all or nothing.
    HC4 user code (kind script/session) only on a provider that has a sandbox (accepts_scripts).
    HC5 the booking must fit inside the lender's daily availability window.
    HC6 price must not exceed the renter's max price, and the renter needs enough free credits.
    HC7 a private GPU may only be used by its owner or by users the owner trusts.
"""
import base64
import hashlib
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager

from werkzeug.security import check_password_hash, generate_password_hash

import config
import rules

from . import db


# ---- exceptions: each one tells the caller exactly what went wrong ----
class NotFound(Exception):
    """Raised when a job / GPU / provider id does not exist."""
    pass


class BadState(Exception):
    """Raised when the object exists but is in the wrong state for the operation (e.g. job already running)."""
    pass


class GpuBusy(Exception):
    """Raised when a GPU with a running job is removed (UC6)."""


class AssignmentRejected(Exception):
    """Raised when a commit would break HC1, HC2 or the state rules; nothing is written."""


# A script path must: start with a letter/digit/underscore, use only safe characters, end in .py,
# and be at most 200 chars. ':' and backslash are not allowed, so drive letters (C:) are impossible.
_SCRIPT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\- /]{0,198}\.py$")


def valid_script_path(path):
    """Return True if `path` is a safe relative .py path inside the user's workspace.
    Rejects non-strings, absolute paths, drive letters, backslashes, '..' segments and '//'.
    Used by submit_job() for kind='script' so a user cannot point at files outside their own folder."""
    # Not a string, or does not match the whitelist pattern -> unsafe.
    if not isinstance(path, str) or not _SCRIPT_RE.match(path):
        return False
    # Split on '/' so a folder named 'a..b' is fine but a segment that is exactly '..' (go up) is refused.
    return ".." not in path.split("/") and "//" not in path


# Editor file: same character whitelist as scripts but any extension (the .py ending is not required).
_FILE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\- /]{0,198}$")


def valid_file_path(path):
    """Return True if `path` is a safe relative file path for the code editor.
    Rules: relative, no '..' or '.' segments, no empty segments (so no '//' or leading '/'),
    no trailing '/', no backslashes, at most 200 characters. Protects the server from path tricks."""
    # Reject non-strings, characters outside the whitelist, and names that end with '/' (a folder).
    if not isinstance(path, str) or not _FILE_RE.match(path) or path.endswith("/"):
        return False
    # Every segment must be a real name: empty ('//'), '.' and '..' are all refused.
    return all(seg not in ("", ".", "..") for seg in path.split("/"))


def _hash_token(token):
    """SHA-256 hex digest of a provider token. Only this hash is ever stored (never the token itself)."""
    return hashlib.sha256(token.encode()).hexdigest()


def _d(row):
    """Convert a sqlite3.Row to a plain dict (or None when no row was found)."""
    return dict(row) if row is not None else None


# =====================================================================================
# Repository: one object = one SQLite connection. Every process / thread makes its own.
# =====================================================================================
class Repository:
    """All database access for the project. Methods return plain dicts so callers never see SQL.
    Exceptions used: NotFound (unknown id), BadState (wrong state), GpuBusy (GPU has a job),
    AssignmentRejected (a hard constraint failed), ValueError (bad input / constraint violation)."""
    def __init__(self, path=None):
        """path: SQLite file to use (None = config.DB_PATH). The connection is opened lazily on first use."""
        # Lazy connection: opening is delayed until the first query (see the conn property).
        self._path = path
        self._conn = None

    # ------------------------------------------------------------------ plumbing
    # (connection handling and the transaction helper that every write goes through)
    # ------------------------------------------------------------------ plumbing
    @property
    def conn(self):
        """The shared connection; created on first access (lazy) and reused afterwards."""
        if self._conn is None:
            self._conn = db.connect(self._path)
        return self._conn

    def init(self):
        """Create the tables if needed and migrate an old database (see db.init_schema)."""
        db.init_schema(self.conn)

    def close(self):
        """Close the connection (safe to call twice). A new one is opened automatically if used again."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def _tx(self):
        """Context manager for ONE write transaction: `with self._tx() as c:`.
        BEGIN IMMEDIATE grabs the database write lock immediately, so no other writer can slip in between
        our reads and our writes (this is what makes the re-checks in commit_assignment race-safe).
        If the block raises ANY exception (BaseException also covers KeyboardInterrupt) everything is rolled
        back and the error is re-raised; otherwise the changes are committed together (all or nothing)."""
        # Start the transaction and take the write lock now (not lazily at the first INSERT).
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            # Hand the connection to the caller's `with` block; the block's statements run inside the transaction.
            yield self.conn
        except BaseException:
            # Something failed: undo every change made in this transaction, then let the error propagate.
            self.conn.execute("ROLLBACK")
            raise
        else:
            # No error: make all changes permanent at once.
            self.conn.execute("COMMIT")

    # ------------------------------------------------------------------ users
    # ------------------------------------------------------------------ users
    def create_user(self, username, password, role="submitter"):
        """Insert a user with a salted PBKDF2 password hash. Returns the new user_id.
        Raises sqlite3.IntegrityError if the username already exists (UNIQUE) or role breaks the CHECK."""
        # Never store the password itself: only a salted one-way hash.
        h = generate_password_hash(password, method="pbkdf2:sha256")
        # INSERT inside a transaction; `return` leaves the with-block, which then COMMITs.
        with self._tx() as c:
            cur = c.execute("INSERT INTO app_user(username,password_hash,role) VALUES(?,?,?)",
                            (username, h, role))
            return cur.lastrowid

    def verify_login(self, username, password):
        """Check a username/password. Returns {user_id, username, role} on success, else None.
        check_password_hash re-hashes the typed password with the stored salt and compares the results."""
        row = self.conn.execute("SELECT * FROM app_user WHERE username=?", (username,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            return {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}
        return None

    def get_user_by_name(self, username):
        """Look a user up by name (without the password hash). Returns a dict or None."""
        return _d(self.conn.execute("SELECT user_id,username,role FROM app_user WHERE username=?", (username,)).fetchone())

    def register_user(self, username, password, welcome_credits=0, now=None):
        """SpareGPU self sign-up: creates a renter and gives the welcome credits (one ledger row).
        Returns the new user_id. Raises sqlite3.IntegrityError when the username is taken.
        Note: create_user and credit are two separate transactions, each atomic on its own."""
        uid = self.create_user(username, password, "submitter")
        if welcome_credits:
            self.credit(uid, welcome_credits, "Welcome credits", now=now)
        return uid

    def list_market(self, now=None):
        """Every approved provider with its terms (price, hours, privacy, online flag) for the browse page."""
        rows = self.list_providers(now)
        return [r for r in rows if r["approved"]]

    def get_user(self, user_id):
        """Return {user_id, username, role} for an id, or None."""
        row = self.conn.execute("SELECT user_id,username,role FROM app_user WHERE user_id=?",
                                (user_id,)).fetchone()
        return _d(row)

    # ------------------------------------------------------------------ jobs
    # ------------------------------------------------------------------ jobs
    def submit_job(self, user_id, template, memory_gb, duration_min, priority, deadline=None, now=None,
                   project_name="", description=None, kind="template", script_path=None,
                   framework=None, cpu_ram_gb=None, max_price_per_hour=None):
        """Store a validated job as 'queued' (hand-off from E4 to E1). Returns the new job_id.
        kind='template': template names one of the prepared workloads ('matmul' or 'train').
        kind='script'  : script_path is a file inside the owner's workspace on the provider's laptop.
        kind='session' : an interactive terminal session; it has neither a template nor a script.
        The schema CHECK on job enforces that exactly the right one of template/script_path is set.
        max_price_per_hour: the renter's price limit (None = any price), checked again in commit_assignment (HC6).
        Raises ValueError for a bad script path or if the database constraints reject the values
        (memory_gb > 0, duration_min > 0, priority 1..5, ...)."""
        now = time.time() if now is None else now
        # Normalise the kind so the row always satisfies the CHECK in schema.sql
        # (template jobs have no script_path, script jobs have no template, sessions have neither).
        if kind == "script":
            if not valid_script_path(script_path):
                raise ValueError("script path must be a relative .py path without '..'")
            template = None
        elif kind == "session":
            template = script_path = None
        else:
            script_path = None
        # IntegrityError (a CHECK / NOT NULL failure) is converted to ValueError so callers need no sqlite knowledge.
        try:
            with self._tx() as c:
                # New jobs always start in state 'queued'; E1 later moves them to 'running' via commit_assignment.
                cur = c.execute(
                    "INSERT INTO job(user_id,project_name,description,kind,template,script_path,framework,cpu_ram_gb,"
                    "memory_gb,duration_min,priority,deadline,submitted_at,state,max_price_per_hour) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'queued',?)",
                    (user_id, project_name or "", description, kind, template, script_path, framework, cpu_ram_gb,
                     memory_gb, duration_min, priority, deadline, now, max_price_per_hour))
                return cur.lastrowid
        # The transaction was already rolled back by _tx() before we get here.
        except sqlite3.IntegrityError as e:
            raise ValueError(f"job rejected by database constraints: {e}")

    # Shared SELECT used by get_job / list_jobs / list_held_jobs. j.* is the job row.
    # JOIN app_user: every job has an owner (so a plain JOIN). LEFT JOINs for assignment -> gpu -> provider:
    # a queued job has no assignment yet, and we still want that job row, with NULLs in those columns.
    _JOB_SELECT = (
        "SELECT j.*, u.username, a.assignment_id, a.gpu_id, a.start_time, a.end_time, a.deciding_rule, "
        "a.launched, a.pid, g.name AS gpu_name, pr.name AS provider_name "
        "FROM job j JOIN app_user u ON u.user_id=j.user_id "
        "LEFT JOIN assignment a ON a.job_id=j.job_id LEFT JOIN gpu g ON g.gpu_id=a.gpu_id "
        "LEFT JOIN provider pr ON pr.provider_id=g.provider_id ")

    def get_job(self, job_id):
        """Return one job (with owner name, assignment, GPU and provider info) or None."""
        return _d(self.conn.execute(self._JOB_SELECT + "WHERE j.job_id=?", (job_id,)).fetchone())

    def list_jobs(self, user_id=None, limit=200):
        """Newest jobs first. user_id=None means all users (admin view); otherwise only that user's jobs.
        limit caps the number of rows so a page never loads the whole table."""
        # Optional WHERE clause is appended to the shared SELECT; values are bound with ? (no SQL injection).
        sql = self._JOB_SELECT + ("WHERE j.user_id=? " if user_id is not None else "")
        sql += "ORDER BY j.job_id DESC LIMIT ?"
        args = (user_id, limit) if user_id is not None else (limit,)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def list_open_jobs(self):
        """Queued + held jobs for E1 (the scheduler's input). Each job dict also carries:
        pref_gpu_type     - the GPU type this user's most recent job ran on (used by SP3, GPU-type continuity);
        trusted_providers - set of provider ids that trust this user (their GPUs are free for the user);
        available_credits - balance minus credits reserved by running jobs (used for HC6 pre-checks)."""
        # Correlated subquery: for the current job's user, find the GPU type of their latest committed
        # assignment (ORDER BY committed_at DESC LIMIT 1). It is NULL for a user with no history.
        sql = ("SELECT j.*, (SELECT g.gpu_type FROM assignment a JOIN job j2 ON j2.job_id=a.job_id "
               "JOIN gpu g ON g.gpu_id=a.gpu_id WHERE j2.user_id=j.user_id "
               "ORDER BY a.committed_at DESC LIMIT 1) AS pref_gpu_type "
               "FROM job j WHERE j.state IN ('queued','held')")
        # cache: trusted ids and credits are per USER, so look each user up only once even with many jobs.
        jobs, cache = [], {}
        for r in self.conn.execute(sql):
            j = dict(r)
            uid = j["user_id"]
            if uid not in cache:
                cache[uid] = (self._trusted_ids(uid), self.available_credits(uid))
            j["trusted_providers"], j["available_credits"] = cache[uid]
            jobs.append(j)
        return jobs

    def list_held_jobs(self):
        """Held jobs (waiting for a reason such as 'no GPU big enough'), highest priority first."""
        # The shared SELECT has a.assignment_id; a held job has no assignment, so it is replaced by NULL.
        return [dict(r) for r in self.conn.execute(
            self._JOB_SELECT.replace("a.assignment_id,", "NULL AS assignment_id,") +
            "WHERE j.state='held' ORDER BY j.priority DESC, j.submitted_at")]

    def queue_position(self, job_id):
        """1-based place among open jobs in scheduling order (priority desc, submission asc).
        Returns None if the job is not queued/held. job_id is the final tie-breaker so the order is stable."""
        rows = self.conn.execute(
            "SELECT job_id FROM job WHERE state IN ('queued','held') ORDER BY priority DESC, submitted_at, job_id"
        ).fetchall()
        for i, r in enumerate(rows, 1):
            if r["job_id"] == job_id:
                return i
        return None

    def set_held(self, job_id, reason):
        """queued/held -> held with a specific reason (FR11 / FR12). Returns True if a row changed.
        The WHERE clause only touches queued/held jobs (never running ones) and skips the write when the
        job is already held for the same reason (`IS NOT ?` is NULL-safe), so nothing is rewritten needlessly."""
        with self._tx() as c:
            cur = c.execute(
                "UPDATE job SET state='held', held_reason=? WHERE job_id=? AND state IN ('queued','held') "
                "AND (state!='held' OR held_reason IS NOT ?)", (reason, job_id, reason))
            return cur.rowcount > 0

    # ------------------------------------------------------------------ GPUs
    # ------------------------------------------------------------------ GPUs
    def add_gpu(self, name, gpu_type, memory_gb, is_simulated=0, device_index=None, now=None):
        """Add a local/simulated GPU (no provider). Returns gpu_id.
        free_since is set to now: SP2 uses it to prefer GPUs that have been idle longest.
        Raises ValueError if the CHECK fails (memory_gb > 0)."""
        now = time.time() if now is None else now
        try:
            with self._tx() as c:
                cur = c.execute(
                    "INSERT INTO gpu(name,gpu_type,memory_gb,is_simulated,device_index,free_since) VALUES(?,?,?,?,?,?)",
                    (name, gpu_type, memory_gb, int(is_simulated), device_index, now))
                return cur.lastrowid
        except sqlite3.IntegrityError as e:
            raise ValueError(f"GPU rejected by database constraints: {e}")

    def upsert_real_gpu(self, device_index, name, gpu_type, memory_gb, now=None):
        """Register a GPU found by nvidia-smi on the server itself. Returns gpu_id.
        Existing rows only get name/type/memory refreshed; their status (active/removed) stays the admin's choice."""
        now = time.time() if now is None else now
        with self._tx() as c:
            # Look for an existing real (non-simulated) GPU at this device index so restarts do not duplicate it.
            row = c.execute("SELECT gpu_id FROM gpu WHERE device_index=? AND is_simulated=0",
                            (device_index,)).fetchone()
            if row:
                c.execute("UPDATE gpu SET name=?, gpu_type=?, memory_gb=? WHERE gpu_id=?",
                          (name, gpu_type, memory_gb, row["gpu_id"]))
                return row["gpu_id"]
            cur = c.execute(
                "INSERT INTO gpu(name,gpu_type,memory_gb,is_simulated,device_index,free_since) VALUES(?,?,?,0,?,?)",
                (name, gpu_type, memory_gb, device_index, now))
            return cur.lastrowid

    def _cutoff(self, now):
        """Epoch time before which a provider counts as OFFLINE (now minus PROVIDER_OFFLINE_S)."""
        now = time.time() if now is None else now
        return now - config.PROVIDER_OFFLINE_S

    def list_active_gpus(self, now=None):
        """GPUs the scheduler may use: status 'active', and (if they belong to a provider) the provider is
        approved, has sharing switched on and has reported (heartbeat last_seen) recently.
        Each row carries the provider's terms: accepts_scripts, price_per_hour, window, private_only, owner."""
        # LEFT JOIN keeps server-local GPUs (provider_id NULL). COALESCE gives local GPUs safe defaults
        # (0 = no scripts, free, not private). last_seen >= cutoff is the heartbeat / offline test.
        sql = ("SELECT g.*, p.name AS provider_name, COALESCE(p.accepts_scripts,0) AS accepts_scripts, "
               "p.owner_user_id, COALESCE(p.price_per_hour,0) AS price_per_hour, p.avail_from, p.avail_to, "
               "COALESCE(p.private_only,0) AS private_only "
               "FROM gpu g LEFT JOIN provider p ON p.provider_id=g.provider_id "
               "WHERE g.status='active' AND (g.provider_id IS NULL OR "
               "(p.approved=1 AND p.sharing_enabled=1 AND p.last_seen >= ?)) "
               "ORDER BY g.gpu_id")
        return [dict(r) for r in self.conn.execute(sql, (self._cutoff(now),))]

    def list_all_gpus(self):
        """Every GPU (including removed ones) with its provider name, for the admin pages."""
        return [dict(r) for r in self.conn.execute(
            "SELECT g.*, p.name AS provider_name FROM gpu g LEFT JOIN provider p ON p.provider_id=g.provider_id ORDER BY g.gpu_id")]

    def pool_view(self, now=None):
        """Every GPU with the job currently on it (if any) and whether its provider is online."""
        # `online` is 1 for local GPUs or for approved providers with a fresh heartbeat. The LEFT JOIN chain
        # gpu -> job (current_job_id) -> user / assignment fills the run_* columns only for busy GPUs.
        sql = ("SELECT g.*, p.name AS provider_name, p.approved AS provider_approved, "
               "(g.provider_id IS NULL OR (p.approved=1 AND p.last_seen >= ?)) AS online, "
               "p.price_per_hour, p.avail_from, p.avail_to, p.sharing_enabled, p.private_only, "
               "j.job_id AS run_job_id, j.template AS run_template, j.user_id AS run_user_id, "
               "u.username AS run_username, a.end_time AS run_end_time FROM gpu g "
               "LEFT JOIN provider p ON p.provider_id=g.provider_id "
               "LEFT JOIN job j ON j.job_id=g.current_job_id LEFT JOIN app_user u ON u.user_id=j.user_id "
               "LEFT JOIN assignment a ON a.job_id=j.job_id ORDER BY g.gpu_id")
        return [dict(r) for r in self.conn.execute(sql, (self._cutoff(now),))]

    def remove_gpu(self, gpu_id):
        """Admin removes a GPU from the pool (soft delete: status='removed'); blocked while a job runs on it (UC6).
        Raises NotFound / GpuBusy."""
        with self._tx() as c:
            # Read and update in the same transaction so a job cannot be assigned between the check and the update.
            row = c.execute("SELECT * FROM gpu WHERE gpu_id=?", (gpu_id,)).fetchone()
            if row is None:
                raise NotFound(f"GPU {gpu_id} not found")
            if row["current_job_id"] is not None:
                raise GpuBusy(f"GPU {gpu_id} has a running job")
            c.execute("UPDATE gpu SET status='removed' WHERE gpu_id=?", (gpu_id,))

    def restore_gpu(self, gpu_id, now=None):
        """Put a removed GPU back in the pool. free_since is reset so SP2 treats it as newly free.
        Raises NotFound."""
        now = time.time() if now is None else now
        with self._tx() as c:
            cur = c.execute("UPDATE gpu SET status='active', free_since=? WHERE gpu_id=?", (now, gpu_id))
            if cur.rowcount == 0:
                raise NotFound(f"GPU {gpu_id} not found")

    def rename_gpu(self, gpu_id, name):
        """Change a GPU's display name. Raises NotFound."""
        with self._tx() as c:
            cur = c.execute("UPDATE gpu SET name=? WHERE gpu_id=?", (name, gpu_id))
            if cur.rowcount == 0:
                raise NotFound(f"GPU {gpu_id} not found")

    def update_gpu(self, gpu_id, name, gpu_type, memory_gb):
        """Admin edits a GPU's model name, type and memory. Refused while a job runs on it (the job was placed
        using the old memory figure). Raises NotFound / GpuBusy / ValueError (CHECK constraint)."""
        try:
            with self._tx() as c:
                row = c.execute("SELECT current_job_id FROM gpu WHERE gpu_id=?", (gpu_id,)).fetchone()
                if row is None:
                    raise NotFound(f"GPU {gpu_id} not found")
                if row["current_job_id"] is not None:
                    raise GpuBusy(f"GPU {gpu_id} has a running job")
                c.execute("UPDATE gpu SET name=?, gpu_type=?, memory_gb=? WHERE gpu_id=?",
                          (name, gpu_type, memory_gb, gpu_id))
        except sqlite3.IntegrityError as e:
            raise ValueError(f"GPU rejected by database constraints: {e}")

    def add_server(self, owner_user_id, name, gpu_name, gpu_type, memory_gb, now=None):
        """Admin registers a GPU server: a provider row (approved, accepts user code) plus its declared GPU
        (device 0). It stays OFFLINE, so it gets no jobs, until the agent started with the returned token
        connects; the agent then replaces the declared model/memory with what nvidia-smi really reports.
        Returns (provider_id, plaintext_token). Raises ValueError on duplicate name / bad values."""
        pid, token = self.create_provider(owner_user_id, name, accepts_scripts=1, approved=1, now=now)
        try:
            self.sync_provider_gpus(pid, [{"device_index": 0, "name": gpu_name, "gpu_type": gpu_type,
                                           "memory_gb": memory_gb}], now=now)
        except sqlite3.IntegrityError as e:
            self.delete_provider(pid)                       # do not leave a server without its GPU row
            raise ValueError(f"GPU rejected by database constraints: {e}")
        return pid, token

    def update_gpu_stats(self, gpu_id, mem_used_mb, util_pct, now=None):
        """Store live stats (memory used in MB, utilisation %) reported by the agent, with a timestamp."""
        now = time.time() if now is None else now
        with self._tx() as c:
            c.execute("UPDATE gpu SET mem_used_mb=?, util_pct=?, stats_at=? WHERE gpu_id=?",
                      (mem_used_mb, util_pct, now, gpu_id))

    # ------------------------------------------------------------------ the atomic commit (NFR1, HC3)
    # ------------------------------------------------------------------ the atomic commit (NFR1, HC3)
    def commit_assignment(self, job_id, gpu_id, start_time, end_time, rule, now=None, _fail_before_commit=False):
        """Store a scheduling decision. ONE transaction: INSERT assignment + job -> 'running' + GPU occupied.

        Inputs: job_id and gpu_id chosen by E1; start_time/end_time (epoch s) of the booking; rule = name of the
        rule that decided (stored for explanation); now = clock override for tests;
        _fail_before_commit = test hook that raises just before COMMIT to prove nothing is half-written.
        Returns True. Raises NotFound (no such job), BadState (job not queued/held) or AssignmentRejected.

        WHY RE-CHECK INSIDE THE TRANSACTION? E1 chose this GPU using data read earlier. Meanwhile a heartbeat may
        have been missed, another scheduler may have taken the GPU, the lender may have paused sharing, etc.
        _tx() holds the database write lock (BEGIN IMMEDIATE), so while we check, nobody else can change the data;
        if any check fails we raise and the transaction rolls back with NOTHING written. That is what prevents
        double booking and invalid assignments even if the caller is buggy or two schedulers race.

        Rules re-checked here:
          HC1 GPU memory >= job memory.
          HC2 the GPU has no job (current_job_id IS NULL) - enforced by a conditional UPDATE, see below.
          HC3 atomicity: GPU occupied + assignment row + job 'running' are written together or not at all.
          HC4 kind script/session only on a provider that accepts users' own scripts (has the sandbox).
          HC5 start..end must fit the lender's daily availability window (rules.fits_window).
          HC6 rate <= job's max price, and the renter's free credits (balance - reserved) cover rate x duration.
          HC7 a private GPU only for its owner or a trusted user.
        Also: the GPU must be active, and its provider approved, sharing enabled and online (recent heartbeat).
        The price (rate) is stored on the assignment: owner and trusted users pay 0, everyone else the provider's price."""
        now = time.time() if now is None else now
        with self._tx() as c:
            # 1. Re-read the job INSIDE the transaction: its state may have changed since E1 looked at it.
            job = c.execute("SELECT * FROM job WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise NotFound(f"job {job_id} not found")
            # Only waiting jobs can be assigned (a running/completed job would be double-booked).
            if job["state"] not in ("queued", "held"):
                raise BadState(f"job {job_id} is {job['state']}, cannot be assigned")
            # rate = credits per hour this renter will pay; stays 0 for free / server-local GPUs.
            rate = 0.0
            # 2. Re-read the GPU and check it is still an active pool member.
            gpu = c.execute("SELECT * FROM gpu WHERE gpu_id=?", (gpu_id,)).fetchone()
            if gpu is None or gpu["status"] != "active":
                raise AssignmentRejected("GPU is not an active pool member")
            # HC1: the GPU must have at least as much memory as the job needs.
            if gpu["memory_gb"] < job["memory_gb"]:
                raise AssignmentRejected("HC1 violated: GPU memory below the job requirement")
            # GPUs owned by a provider (a lender's laptop) get the extra provider checks; local GPUs skip them.
            if gpu["provider_id"] is not None:
                prov = c.execute("SELECT * FROM provider WHERE provider_id=?", (gpu["provider_id"],)).fetchone()
                # Provider must be approved by the admin AND online (heartbeat newer than the offline cutoff).
                if not prov or not prov["approved"] or prov["last_seen"] is None or prov["last_seen"] < self._cutoff(now):
                    raise AssignmentRejected("the GPU's provider is not approved and online")
                # HC4: user-written code (script/session) must never run on a laptop without the sandbox switch.
                if job["kind"] in ("script", "session") and not prov["accepts_scripts"]:
                    raise AssignmentRejected("HC4 violated: provider does not accept users' own scripts")
                # trusted = the lender has put this renter on the trusted list; own = renter owns the laptop.
                trusted = c.execute("SELECT 1 FROM provider_trusted WHERE provider_id=? AND user_id=?",
                                    (prov["provider_id"], job["user_id"])).fetchone() is not None
                own = prov["owner_user_id"] == job["user_id"]
                # Lender's pause switch: no NEW jobs while sharing is off.
                if not prov["sharing_enabled"]:
                    raise AssignmentRejected("the lender has paused sharing on this GPU")
                # HC7: private GPU -> only the owner or trusted users.
                if prov["private_only"] and not (own or trusted):
                    raise AssignmentRejected("HC7 violated: this GPU is private")
                # HC5: the whole booking must lie inside the lender's available hours (None = always available).
                if not rules.fits_window(prov["avail_from"], prov["avail_to"], start_time, end_time):
                    raise AssignmentRejected("HC5 violated: booking does not fit the lender's available hours")
                # Owner and trusted users rent for free; everybody else pays the lender's price per hour.
                rate = 0.0 if (own or trusted) else float(prov["price_per_hour"])
                # HC6: the price/credit checks only matter when the job actually costs something.
                if rate > 0:
                    if config.CREDITS_ENABLED:
                        pass
                    # HC6a: the renter set a maximum price per hour and this GPU is more expensive.
                    if job["max_price_per_hour"] is not None and rate > job["max_price_per_hour"]:
                        raise AssignmentRejected("HC6 violated: price is above the renter's maximum")
                    # bal = credits balance (SUM of the ledger). If credits are disabled, treat the balance as infinite.
                    bal = c.execute("SELECT COALESCE(SUM(amount),0) FROM ledger WHERE user_id=?", (job["user_id"],)).fetchone()[0] if config.CREDITS_ENABLED else float("inf")
                    # held = credits already promised to the user's OTHER running jobs (rate x declared hours); these
                    # cannot be spent twice, so they are subtracted from the balance.
                    held = c.execute("SELECT COALESCE(SUM(a.rate * j.duration_min / 60.0),0) FROM assignment a "
                                     "JOIN job j ON j.job_id=a.job_id WHERE j.user_id=? AND j.state='running'",
                                     (job["user_id"],)).fetchone()[0]
                    # HC6b: free credits must cover the cost of the booked time (1e-9 tolerates float rounding).
                    if bal - held < rules.estimate_cost(rate, job["duration_min"]) - 1e-9:
                        raise AssignmentRejected("HC6 violated: not enough credits for the booked time")
            # Server-local GPU (no provider): user code is refused, because only provider laptops have a sandbox (HC4).
            elif job["kind"] in ("script", "session"):
                raise AssignmentRejected("HC4 violated: user code needs a provider sandbox")
            # HC2 + HC3 step 1: claim the GPU with a CONDITIONAL update. `current_job_id IS NULL` is the test and
            # the write in one atomic statement: if another job took the GPU, rowcount is 0 and we refuse.
            cur = c.execute("UPDATE gpu SET current_job_id=? WHERE gpu_id=? AND current_job_id IS NULL",
                            (job_id, gpu_id))
            # rowcount != 1 means the GPU was not free -> HC2 would be broken.
            if cur.rowcount != 1:
                raise AssignmentRejected("HC2 violated: GPU already has a running job")
            # HC3 step 2: record the assignment (job_id is UNIQUE in the table: one assignment per job).
            c.execute("INSERT INTO assignment(job_id,gpu_id,start_time,end_time,deciding_rule,committed_at,rate) "
                      "VALUES(?,?,?,?,?,?,?)", (job_id, gpu_id, start_time, end_time, rule, now, rate))
            # HC3 step 3: the job becomes 'running' and any 'held' reason is cleared.
            c.execute("UPDATE job SET state='running', held_reason=NULL WHERE job_id=?", (job_id,))
            # ---- test hooks (below): used only by the NFR1 / NFR4 tests to simulate a crash before COMMIT
            # ---- test hooks: used only by the NFR1 / NFR4 tests to simulate a crash before COMMIT
            if _fail_before_commit:
                raise RuntimeError("injected failure before COMMIT")
            if os.environ.get("GPUSCHED_FAULT") == "sleep_in_commit":
                mark = os.environ.get("GPUSCHED_FAULT_MARK")
                if mark:                        # tells the test "I am inside the transaction now"
                    with open(mark, "w") as f:
                        f.write(str(os.getpid()))
                time.sleep(float(os.environ.get("GPUSCHED_FAULT_SLEEP", "3")))
        return True

    # ------------------------------------------------------------------ providers (GPU owners' laptops)
    # ------------------------------------------------------------------ providers (GPU owners' laptops)
    def create_provider(self, owner_user_id, name, accepts_scripts=0, workspace_root=None, now=None,
                        approved=0, price_per_hour=0.0, avail_from=None, avail_to=None, private_only=0, description=None):
        """Register a laptop. Returns (provider_id, plaintext_token); only the token's hash is stored,
        so this return value is the ONLY time the plaintext token exists on the server.
        College portal: not approved until the admin approves. SpareGPU: the lender sets price / hours / privacy.
        avail_from / avail_to: minutes after midnight (0..1439), None = always available.
        Raises ValueError if a CHECK / UNIQUE constraint fails (duplicate name, negative price, bad minutes)."""
        # Random URL-safe secret (24 random bytes). The agent sends it with every request to prove who it is.
        now = time.time() if now is None else now
        token = secrets.token_urlsafe(24)
        try:
        # Booleans are stored as 0/1 integers (SQLite has no boolean type).
            with self._tx() as c:
                cur = c.execute(
                    "INSERT INTO provider(owner_user_id,name,token_hash,approved,accepts_scripts,workspace_root,created_at,"
                    "price_per_hour,avail_from,avail_to,private_only,description) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (owner_user_id, name, _hash_token(token), int(bool(approved)), int(bool(accepts_scripts)),
                     workspace_root, now, float(price_per_hour), avail_from, avail_to, int(bool(private_only)), description))
                return cur.lastrowid, token
        except sqlite3.IntegrityError as e:
            raise ValueError(f"provider rejected by database constraints: {e}")

    def update_provider_terms(self, provider_id, price_per_hour, avail_from, avail_to, private_only, description):
        """Lender changes price, available hours, privacy and description. Raises NotFound / ValueError.
        Only affects NEW bookings: commit_assignment reads these terms at commit time."""
        try:
            with self._tx() as c:
                cur = c.execute("UPDATE provider SET price_per_hour=?, avail_from=?, avail_to=?, private_only=?, description=? "
                                "WHERE provider_id=?", (float(price_per_hour), avail_from, avail_to,
                                                        int(bool(private_only)), description, provider_id))
                # rowcount == 0: no provider has that id.
                if cur.rowcount == 0:
                    raise NotFound(f"provider {provider_id} not found")
        except sqlite3.IntegrityError as e:
            raise ValueError(f"terms rejected: {e}")

    def set_sharing(self, provider_id, enabled):
        """Lender's pause switch: no NEW jobs are placed while it is off (running jobs continue).
        Raises NotFound."""
        with self._tx() as c:
            if c.execute("UPDATE provider SET sharing_enabled=? WHERE provider_id=?", (int(bool(enabled)), provider_id)).rowcount == 0:
                raise NotFound(f"provider {provider_id} not found")

    def owner_stop_all(self, provider_id):
        """Lender's kill switch: pause sharing AND ask the agent to stop every job running on this laptop.
        stop_requested=2 means 'the lender stopped it' (1 means the user pressed Stop). The agent polls
        provider_stop_requests() and kills the processes. Both updates are in one transaction."""
        with self._tx() as c:
            c.execute("UPDATE provider SET sharing_enabled=0 WHERE provider_id=?", (provider_id,))
            # The subquery finds all jobs whose assignment sits on a GPU of this provider; only running ones are flagged.
            c.execute("UPDATE job SET stop_requested=2 WHERE state='running' AND job_id IN ("
                      "SELECT a.job_id FROM assignment a JOIN gpu g ON g.gpu_id=a.gpu_id WHERE g.provider_id=?)", (provider_id,))

    # ---- trusted users (rent this lender's GPU for free; can use it when it is private)
    # ---- trusted users (rent this lender's GPU for free; can use it when it is private)
    def trust_user(self, provider_id, user_id):
        """Lender trusts a user. INSERT OR IGNORE makes it idempotent (the PRIMARY KEY would reject a duplicate)."""
        with self._tx() as c:
            c.execute("INSERT OR IGNORE INTO provider_trusted(provider_id,user_id) VALUES(?,?)", (provider_id, user_id))

    def untrust_user(self, provider_id, user_id):
        """Remove a user from the lender's trusted list (no error if they were not on it)."""
        with self._tx() as c:
            c.execute("DELETE FROM provider_trusted WHERE provider_id=? AND user_id=?", (provider_id, user_id))

    def list_trusted(self, provider_id):
        """Trusted users of one provider as [{user_id, username}], sorted by name."""
        return [dict(r) for r in self.conn.execute(
            "SELECT u.user_id, u.username FROM provider_trusted t JOIN app_user u ON u.user_id=t.user_id "
            "WHERE t.provider_id=? ORDER BY u.username", (provider_id,))]

    def _trusted_ids(self, user_id):
        """Set of provider ids that trust this user (used for pricing and HC7 in the scheduler)."""
        return {r[0] for r in self.conn.execute("SELECT provider_id FROM provider_trusted WHERE user_id=?", (user_id,))}

    # ---- credits (virtual points; NOT real money). Balance = SUM of the ledger; every change is a ledger row.
    # ---- credits (virtual points)
    def balance(self, user_id):
        """Credit balance = sum of all ledger rows (COALESCE turns 'no rows' into 0). Rounded to 4 places."""
        return round(self.conn.execute("SELECT COALESCE(SUM(amount),0) FROM ledger WHERE user_id=?", (user_id,)).fetchone()[0], 4)

    def reserved(self, user_id):
        """Credits a user's RUNNING jobs may still cost (rate x declared time): they cannot be spent twice.
        Uses the declared duration (worst case); complete_job later charges only the time actually used."""
        return round(self.conn.execute(
            "SELECT COALESCE(SUM(a.rate * j.duration_min / 60.0),0) FROM assignment a JOIN job j ON j.job_id=a.job_id "
            "WHERE j.user_id=? AND j.state='running'", (user_id,)).fetchone()[0], 4)

    def available_credits(self, user_id):
        """Credits the user can still spend: balance minus reserved."""
        return round(self.balance(user_id) - self.reserved(user_id), 4)

    def credit(self, user_id, amount, note, job_id=None, now=None):
        """Add a ledger row (positive = earned/granted, negative = spent). The ledger is append-only: balances are
        never edited, only new rows are added, so there is a full history of every credit movement."""
        now = time.time() if now is None else now
        with self._tx() as c:
            c.execute("INSERT INTO ledger(user_id,amount,note,job_id,created_at) VALUES(?,?,?,?,?)",
                      (user_id, float(amount), note, job_id, now))

    def list_ledger(self, user_id, limit=100):
        """Newest ledger entries first (for the user's credit history page)."""
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM ledger WHERE user_id=? ORDER BY entry_id DESC LIMIT ?", (user_id, limit))]

    def get_provider_by_token(self, token):
        """Find a provider from the plaintext token sent by its agent: hash it, then look up token_hash.
        Returns the provider dict or None (also None for an empty token)."""
        if not token:
            return None
        return _d(self.conn.execute("SELECT * FROM provider WHERE token_hash=?", (_hash_token(token),)).fetchone())

    def get_provider(self, provider_id):
        """Return the provider row as a dict, or None."""
        return _d(self.conn.execute("SELECT * FROM provider WHERE provider_id=?", (provider_id,)).fetchone())

    def list_providers(self, now=None):
        """All providers with owner name, `online` flag and number of active GPUs (for admin / market pages).
        online = approved AND heartbeat newer than the offline cutoff. The correlated COUNT subquery counts GPUs."""
        sql = ("SELECT p.provider_id,p.owner_user_id,p.name,p.approved,p.accepts_scripts,p.workspace_root,"
               "p.last_seen,p.created_at,p.price_per_hour,p.avail_from,p.avail_to,p.sharing_enabled,p.private_only,p.description,"
               "u.username AS owner, "
               "(p.approved=1 AND p.last_seen IS NOT NULL AND p.last_seen >= ?) AS online, "
               "(SELECT COUNT(*) FROM gpu g WHERE g.provider_id=p.provider_id AND g.status='active') AS gpu_count "
               "FROM provider p JOIN app_user u ON u.user_id=p.owner_user_id ORDER BY p.provider_id")
        return [dict(r) for r in self.conn.execute(sql, (self._cutoff(now),))]

    def delete_provider(self, provider_id):
        """Decline a provider request. Only possible while it has never run a job; otherwise suspend it instead.
        Deletes in child -> parent order (gpu, trusted rows, then provider) because of the foreign keys.
        Raises NotFound / BadState."""
        with self._tx() as c:
            if c.execute("SELECT 1 FROM provider WHERE provider_id=?", (provider_id,)).fetchone() is None:
                raise NotFound(f"provider {provider_id} not found")
            # If any assignment ever used one of its GPUs, keep the history: refuse deletion (foreign keys would block it).
            used = c.execute("SELECT 1 FROM assignment a JOIN gpu g ON g.gpu_id=a.gpu_id WHERE g.provider_id=? LIMIT 1",
                             (provider_id,)).fetchone()
            if used:
                raise BadState("this provider has already run jobs; suspend it instead of declining")
            c.execute("DELETE FROM gpu WHERE provider_id=?", (provider_id,))
            c.execute("DELETE FROM provider_trusted WHERE provider_id=?", (provider_id,))
            c.execute("DELETE FROM provider WHERE provider_id=?", (provider_id,))

    # ------------------------------------------------------------------ admin dashboard data
    # ------------------------------------------------------------------ admin dashboard data
    def admin_stats(self, now=None):
        """Counters for the admin dashboard: providers (pending/online/total), GPUs, users and jobs."""
        # Tiny helper: run a one-value query (COUNT ...) and return that single value.
        q = lambda sql, *a: self.conn.execute(sql, a).fetchone()[0]
        # cut = oldest heartbeat time that still counts as online.
        cut = self._cutoff(now)
        return {
            "pending_providers": q("SELECT COUNT(*) FROM provider WHERE approved=0"),
            "online_providers": q("SELECT COUNT(*) FROM provider WHERE approved=1 AND last_seen >= ?", cut),
            "total_providers": q("SELECT COUNT(*) FROM provider"),
            "total_gpus": q("SELECT COUNT(*) FROM gpu WHERE status='active'"),
            "active_users": q("SELECT COUNT(DISTINCT user_id) FROM job WHERE state IN ('queued','held','running')"),
            "running_jobs": q("SELECT COUNT(*) FROM job WHERE state='running'"),
            "waiting_jobs": q("SELECT COUNT(*) FROM job WHERE state IN ('queued','held')"),
        }

    def fleet_view(self, now=None):
        """Providers (plus any server-local GPUs) with their GPUs, for the dashboard rings and the fleet page.
        Each group gets total / allocated (busy) / available (free and online) / percent busy."""
        now = time.time() if now is None else now
        groups = {}
        for p in self.list_providers(now):
            groups[p["provider_id"]] = dict(p, gpus=[])
        # Pseudo-provider that collects the GPUs with provider_id NULL (installed on the server itself).
        local = {"provider_id": None, "name": "Server (local GPUs)", "owner": "-", "approved": 1, "online": 1,
                 "accepts_scripts": 0, "last_seen": None, "gpus": []}
        for r in self.pool_view(now):
            if r["status"] != "active":
                continue
            grp = groups.get(r["provider_id"]) if r["provider_id"] is not None else local
            if grp is None:
                continue
            grp["gpus"].append({
                "gpu_id": r["gpu_id"], "device_index": r["device_index"], "name": r["name"], "memory_gb": r["memory_gb"],
                "busy": r["current_job_id"] is not None, "online": bool(r["online"]), "job_id": r["run_job_id"],
                "username": r["run_username"], "end_time": r["run_end_time"], "util_pct": r["util_pct"],
                "mem_used_mb": r["mem_used_mb"]})
        # Local GPUs are shown last, and only when there is at least one.
        out = list(groups.values()) + ([local] if local["gpus"] else [])
        for g in out:
            n = len(g["gpus"])
            g["total"], g["allocated"] = n, sum(1 for x in g["gpus"] if x["busy"])
            g["available"] = sum(1 for x in g["gpus"] if not x["busy"] and x["online"])
            g["percent"] = round(100 * g["allocated"] / n) if n else 0
        return out

    def set_provider_approved(self, provider_id, approved):
        """Admin approves (or suspends) a provider. Unapproved providers are never scheduled. Raises NotFound."""
        with self._tx() as c:
            cur = c.execute("UPDATE provider SET approved=? WHERE provider_id=?", (int(bool(approved)), provider_id))
            if cur.rowcount == 0:
                raise NotFound(f"provider {provider_id} not found")

    def set_provider_scripts(self, provider_id, accepts_scripts, workspace_root=None):
        """Agent tells the server whether it accepts users' own scripts (HC4). COALESCE(?, workspace_root) keeps
        the old workspace folder when no new one is given. Raises NotFound."""
        with self._tx() as c:
            cur = c.execute("UPDATE provider SET accepts_scripts=?, workspace_root=COALESCE(?,workspace_root) "
                            "WHERE provider_id=?", (int(bool(accepts_scripts)), workspace_root, provider_id))
            if cur.rowcount == 0:
                raise NotFound(f"provider {provider_id} not found")

    def touch_provider(self, provider_id, now=None):
        """Heartbeat: record that the provider's agent is alive right now (last_seen). Offline = no recent heartbeat."""
        now = time.time() if now is None else now
        with self._tx() as c:
            c.execute("UPDATE provider SET last_seen=? WHERE provider_id=?", (now, provider_id))

    def sync_provider_gpus(self, provider_id, gpus, now=None):
        """The agent reports the GPUs it found. New devices (by device_index) are added, known ones refreshed.
        Returns the list of gpu_ids. Status and free_since of existing GPUs are left alone."""
        now = time.time() if now is None else now
        ids = []
        with self._tx() as c:
            for g in gpus:
                row = c.execute("SELECT gpu_id FROM gpu WHERE provider_id=? AND device_index=?",
                                (provider_id, g["device_index"])).fetchone()
                if row:
                    c.execute("UPDATE gpu SET name=?, gpu_type=?, memory_gb=? WHERE gpu_id=?",
                              (g["name"], g["gpu_type"], g["memory_gb"], row["gpu_id"]))
                    ids.append(row["gpu_id"])
                else:
                    cur = c.execute(
                        "INSERT INTO gpu(provider_id,name,gpu_type,memory_gb,is_simulated,device_index,free_since) "
                        "VALUES(?,?,?,?,0,?,?)",
                        (provider_id, g["name"], g["gpu_type"], g["memory_gb"], g["device_index"], now))
                    ids.append(cur.lastrowid)
        return ids

    # ------------------------------------------------------------------ run lifecycle (E3)
    # ------------------------------------------------------------------ run lifecycle (E3)
    # Shared SELECT for E3: assignment + the job fields the agent needs + GPU and owner info.
    # Inner JOINs are fine here because a running job always has all of these rows.
    _RUN_SELECT = ("SELECT a.*, j.template, j.kind, j.script_path, j.memory_gb, j.duration_min, j.user_id, "
                   "j.stop_requested, g.device_index, g.is_simulated, g.provider_id, u.username FROM assignment a "
                   "JOIN job j ON j.job_id=a.job_id JOIN gpu g ON g.gpu_id=a.gpu_id "
                   "JOIN app_user u ON u.user_id=j.user_id ")

    def unlaunched_assignments(self, provider_id=None):
        """Running jobs that were committed but whose process has not started yet (launched=0).
        provider_id=None: all of them; otherwise only those on that provider's GPUs. Oldest first."""
        sql = self._RUN_SELECT + "WHERE j.state='running' AND a.launched=0 "
        args = ()
        if provider_id is not None:
            sql += "AND g.provider_id=? "
            args = (provider_id,)
        return [dict(r) for r in self.conn.execute(sql + "ORDER BY a.assignment_id", args)]

    def running_assignments(self, provider_id=None):
        """All running jobs (optionally of one provider), oldest assignment first."""
        sql = self._RUN_SELECT + "WHERE j.state='running' "
        args = ()
        if provider_id is not None:
            sql += "AND g.provider_id=? "
            args = (provider_id,)
        return [dict(r) for r in self.conn.execute(sql + "ORDER BY a.assignment_id", args)]

    def provider_launches(self, provider_id):
        """Jobs committed for this provider that its agent has not started yet."""
        return self.unlaunched_assignments(provider_id)

    def provider_active(self, provider_id):
        """Jobs the server believes are running on this provider (launched)."""
        return [a for a in self.running_assignments(provider_id) if a["launched"]]

    def mark_launched(self, assignment_id, pid, now=None):
        """The agent reports the process started: launched=1, start timestamp and operating-system pid."""
        now = time.time() if now is None else now
        with self._tx() as c:
            c.execute("UPDATE assignment SET launched=1, launched_at=?, pid=? WHERE assignment_id=?",
                      (now, pid, assignment_id))

    def offline_provider_ids(self, now=None):
        """Providers that still have running jobs but have stopped reporting (laptop closed, network lost).
        E3 uses this to requeue those jobs. DISTINCT avoids listing a provider once per job."""
        # last_seen IS NULL (never reported) or older than the cutoff = offline.
        sql = ("SELECT DISTINCT g.provider_id FROM gpu g JOIN job j ON j.job_id=g.current_job_id "
               "JOIN provider p ON p.provider_id=g.provider_id "
               "WHERE j.state='running' AND (p.last_seen IS NULL OR p.last_seen < ?)")
        return [r["provider_id"] for r in self.conn.execute(sql, (self._cutoff(now),))]

    def overdue_assignments(self, now=None, slack_s=None):
        """Running jobs far past their planned end: the agent should have reported them by now.
        slack_s defaults to twice the kill grace period; E3 then forces them to finish or requeues them."""
        now = time.time() if now is None else now
        slack = config.KILL_GRACE_S * 2 if slack_s is None else slack_s
        return [a for a in self.running_assignments() if a["end_time"] + slack < now]

    # ------------------------------------------------------------------ live logs (files, not DB rows)
    # Logs can be large and are append-only, so they live in plain files (data/joblogs), not in SQLite.
    # ------------------------------------------------------------------ live logs (files, not DB rows)
    def _log_path(self, job_id):
        """Log file for a job. int() guarantees the name cannot contain path tricks."""
        return os.path.join(config.JOBLOG_DIR, f"job_{int(job_id)}.log")

    def append_log(self, job_id, text):
        """Append agent output; the file is capped so one job cannot fill the disk."""
        if not text:
            return 0
        os.makedirs(config.JOBLOG_DIR, exist_ok=True)
        path = self._log_path(job_id)
        # Cap the file at LOG_MAX_BYTES so one chatty job cannot fill the disk.
        size = os.path.getsize(path) if os.path.exists(path) else 0
        room = config.LOG_MAX_BYTES - size
        if room <= 0:
            return 0
        # Truncate the new text to the remaining room; errors='replace' avoids crashes on bad bytes.
        data = text.encode("utf-8", "replace")[:room]
        # 'ab' = append in binary mode so offsets are exact byte counts.
        with open(path, "ab") as f:
            f.write(data)
        return len(data)

    def read_log(self, job_id, offset=0):
        """Return (text_from_offset, new_offset). Offsets are in bytes so the page can poll incrementally."""
        path = self._log_path(job_id)
        if not os.path.exists(path):
        # Binary read from the byte offset the page last saw; the new offset is old offset + bytes read.
            return "", 0
        with open(path, "rb") as f:
            f.seek(max(0, int(offset)))
            data = f.read()
        return data.decode("utf-8", "replace"), max(0, int(offset)) + len(data)

    def complete_job(self, job_id, result_note, now=None):
        """running -> completed; releases the GPU (FR16) and settles the credits. Returns False if not running.

        Everything below happens in ONE transaction:
          1. job -> 'completed' (only if it is still 'running': makes a second call a harmless no-op, returns False);
          2. the GPU is freed (current_job_id NULL) and free_since = now (SP2 idle-time fairness);
          3. leftover interactive-terminal rows for the job are deleted;
          4. SETTLEMENT: the renter pays rate x time actually used (never more than the booked end_time), written
             as a negative ledger row; the lender gets the same amount as a positive row. cost is stored on the job.
             Free jobs (rate 0, owner or trusted users) create no ledger rows."""
        now = time.time() if now is None else now
        with self._tx() as c:
            # The `AND state='running'` guard is what makes completion idempotent.
            cur = c.execute("UPDATE job SET state='completed', result_note=? WHERE job_id=? AND state='running'",
                            (result_note, job_id))
            if cur.rowcount == 0:
                return False
            # Free the GPU so E1 can schedule the next job on it.
            c.execute("UPDATE gpu SET current_job_id=NULL, free_since=? WHERE current_job_id=?", (now, job_id))
            # Terminal relay rows are only needed while the job runs.
            for t in ("term_input", "term_output", "term_state"):
                c.execute(f"DELETE FROM {t} WHERE job_id=?", (job_id,))
            # ---- SpareGPU settlement ----
            # SpareGPU settlement: the renter pays for the time actually used (never more than booked); the lender earns it.
            # Fetch the price (rate), booked window, renter (j.user_id) and lender (p.owner_user_id).
            # LEFT JOIN provider because a server-local GPU has no provider (then nobody is paid).
            a = c.execute("SELECT a.rate, a.start_time, a.end_time, j.user_id, p.owner_user_id, p.name FROM assignment a "
                          "JOIN job j ON j.job_id=a.job_id JOIN gpu g ON g.gpu_id=a.gpu_id "
                          "LEFT JOIN provider p ON p.provider_id=g.provider_id WHERE a.job_id=?", (job_id,)).fetchone()
            if config.CREDITS_ENABLED and a and a["rate"] > 0 and a["owner_user_id"] is not None:
                # Seconds used = min(now, end_time) - start_time, clamped at 0; divide by 3600 for hours. Rounded to 2 places.
                cost = round(a["rate"] * max(0.0, min(now, a["end_time"]) - a["start_time"]) / 3600.0, 2)
                c.execute("UPDATE job SET cost=? WHERE job_id=?", (cost, job_id))
                # Two ledger rows with opposite signs: the credits move from renter to lender, none are created or lost.
                if cost > 0:
                    c.execute("INSERT INTO ledger(user_id,amount,note,job_id,created_at) VALUES(?,?,?,?,?)",
                              (a["user_id"], -cost, f"Job #{job_id} on {a['name']}", job_id, now))
                    c.execute("INSERT INTO ledger(user_id,amount,note,job_id,created_at) VALUES(?,?,?,?,?)",
                              (a["owner_user_id"], cost, f"Earned: job #{job_id} on {a['name']}", job_id, now))
            return True

    def requeue_job(self, job_id, note, now=None):
        """Recovery: a running job whose process is gone (provider offline, agent restart) goes back to
        'queued' and frees its GPU so it can be scheduled again. Returns False if it was not running.
        The assignment row is DELETED so the job looks never-assigned (job_id is UNIQUE in assignment, and
        check_consistency forbids a queued job with an assignment). No credits are charged for the lost run."""
        now = time.time() if now is None else now
        with self._tx() as c:
            # Back to the queue, with a note explaining why (shown to the user).
            cur = c.execute("UPDATE job SET state='queued', held_reason=NULL, result_note=? "
                            "WHERE job_id=? AND state='running'", (note, job_id))
            if cur.rowcount == 0:
                return False
            # Remove the assignment, free the GPU and clear terminal data, all in the same transaction.
            c.execute("DELETE FROM assignment WHERE job_id=?", (job_id,))
            c.execute("UPDATE gpu SET current_job_id=NULL, free_since=? WHERE current_job_id=?", (now, job_id))
            for t in ("term_input", "term_output", "term_state"):
                c.execute(f"DELETE FROM {t} WHERE job_id=?", (job_id,))
            return True

    # ------------------------------------------------------------------ stop / cancel (user)
    # stop_requested on a job: 0 = nobody asked, 1 = the user pressed Stop, 2 = the lender stopped it (owner_stop_all).
    # ------------------------------------------------------------------ stop / cancel (user)
    def cancel_job(self, job_id, user_id=None, now=None):
        """User pressed Stop/Cancel.
        queued/held -> completed with result_note 'cancelled by user' (nothing was running, nothing to kill).
        running     -> stop_requested=1: the agent sees it on its next poll and kills the process, then reports
                       completion (so the GPU is released and credits settled the normal way, by complete_job).
        user_id: if given, only the owner of the job may cancel it (returns None for someone else's job).
        Returns 'cancelled', 'stopping' or None (not found / already finished / not yours)."""
        now = time.time() if now is None else now
        with self._tx() as c:
            row = c.execute("SELECT * FROM job WHERE job_id=?", (job_id,)).fetchone()
            if row is None or (user_id is not None and row["user_id"] != user_id):
                return None
            if row["state"] in ("queued", "held"):
                c.execute("UPDATE job SET state='completed', held_reason=NULL, result_note='cancelled by user' "
                          "WHERE job_id=?", (job_id,))
                return "cancelled"
            if row["state"] == "running":
                c.execute("UPDATE job SET stop_requested=1 WHERE job_id=?", (job_id,))
                return "stopping"
            return None

    def provider_stop_requests(self, provider_id):
        """{job_id: 'user' | 'owner'} for running jobs of this provider that someone asked to stop
        (stop_requested 1 -> 'user', 2 -> 'owner'). The agent polls this and kills those processes."""
        return {a["job_id"]: ("owner" if a["stop_requested"] == 2 else "user")
                for a in self.running_assignments(provider_id) if a.get("stop_requested")}

    def provider_terminal_jobs(self, provider_id, now=None):
        """Running jobs of this provider whose owner has the terminal open right now (see term_active)."""
        return [a["job_id"] for a in self.running_assignments(provider_id) if self.term_active(a["job_id"], now)]

    # ------------------------------------------------------------------ project files (code editor)
    # A user's files live in the project_file table (one row per user+path): the server keeps the single copy.
    # The limits below protect the server's disk and memory.
    # ------------------------------------------------------------------ project files (code editor)
    # per-file, per-user file count, and per-user total size limits
    MAX_FILE_BYTES = 200_000
    MAX_FILES = 200
    MAX_TOTAL_BYTES = 5_000_000

    def list_files(self, user_id):
        """[{path, size_in_bytes, updated_at}] for the user. LENGTH(CAST(content AS BLOB)) counts BYTES, not characters."""
        return [dict(r) for r in self.conn.execute(
            "SELECT path, LENGTH(CAST(content AS BLOB)) AS size, updated_at FROM project_file "
            "WHERE user_id=? ORDER BY path", (user_id,))]

    def get_file(self, user_id, path):
        """Content of one file, or None if it does not exist."""
        r = self.conn.execute("SELECT content FROM project_file WHERE user_id=? AND path=?", (user_id, path)).fetchone()
        return None if r is None else r["content"]

    def get_project(self, user_id):
        """All of a user's files as {path: content}; this snapshot is what a job receives."""
        return {r["path"]: r["content"] for r in self.conn.execute(
            "SELECT path, content FROM project_file WHERE user_id=?", (user_id,))}

    def save_file(self, user_id, path, content, now=None):
        """Create or overwrite one file. Path must be a safe relative path; size limits protect the server's disk.
        Raises ValueError for an unsafe path, a file over MAX_FILE_BYTES, too many files or a project over 5 MB."""
        now = time.time() if now is None else now
        if not valid_file_path(path):
            raise ValueError("file path: letters, digits, _ - . and / only, relative, no '..', at most 200 characters")
        # Size in BYTES of the UTF-8 encoding (what the disk and the limits actually count).
        size = len(content.encode("utf-8"))
        if size > self.MAX_FILE_BYTES:
            raise ValueError(f"file too large (limit {self.MAX_FILE_BYTES // 1000} KB)")
        with self._tx() as c:
            # Count the user's OTHER files (path != this one) so overwriting an existing file never hits the limit.
            n, total = c.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(CAST(content AS BLOB))),0) FROM project_file "
                                 "WHERE user_id=? AND path!=?", (user_id, path)).fetchone()
            if n >= self.MAX_FILES:
                raise ValueError(f"too many files (limit {self.MAX_FILES})")
            if total + size > self.MAX_TOTAL_BYTES:
                raise ValueError("project too large (limit 5 MB)")
            # UPSERT: insert a new file, or if (user_id, path) already exists update its content and time.
            c.execute("INSERT INTO project_file(user_id,path,content,updated_at) VALUES(?,?,?,?) "
                      "ON CONFLICT(user_id,path) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at",
                      (user_id, path, content, now))

    def delete_file(self, user_id, path):
        """Delete one file. Returns True if a file was actually deleted."""
        with self._tx() as c:
            return c.execute("DELETE FROM project_file WHERE user_id=? AND path=?", (user_id, path)).rowcount > 0

    def delete_prefix(self, user_id, folder):
        """Delete every file inside a folder (paths that start with 'folder/'). Returns how many were deleted."""
        folder = folder.strip("/")
        if not folder or not valid_file_path(folder + "/x"):
            raise ValueError("invalid folder name")
        with self._tx() as c:
            # '/' is part of the pattern, so deleting 'src' never touches 'src2/a.py'. ESCAPE keeps '_' and '%' literal.
            esc = folder.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            return c.execute("DELETE FROM project_file WHERE user_id=? AND path LIKE ? ESCAPE '\\'",
                             (user_id, esc + "/%")).rowcount

    def rename_path(self, user_id, old, new):
        """Rename a file, or a whole folder (every path starting with 'old/'), in one transaction.
        Raises ValueError for an invalid name, a missing source or a name that is already used (nothing is lost)."""
        old, new = old.strip("/"), new.strip("/")
        if not valid_file_path(old) or not valid_file_path(new):
            raise ValueError("file path: letters, digits, _ - . and / only, relative, no '..', at most 200 characters")
        if old == new:
            return 0
        if new.startswith(old + "/"):
            raise ValueError("a folder cannot be moved inside itself")
        with self._tx() as c:
            esc = old.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = c.execute("SELECT path FROM project_file WHERE user_id=? AND (path=? OR path LIKE ? ESCAPE '\\')",
                             (user_id, old, esc + "/%")).fetchall()
            if not rows:
                raise ValueError("nothing to rename: file or folder not found")
            moves = [(r["path"], new + r["path"][len(old):]) for r in rows]
            for _, dst in moves:
                if not valid_file_path(dst):
                    raise ValueError("the new name makes a path that is too long or invalid")
            # Refuse to overwrite anything that is not itself being moved away.
            moving = {src for src, _ in moves}
            for _, dst in moves:
                if dst not in moving and c.execute("SELECT 1 FROM project_file WHERE user_id=? AND path=?", (user_id, dst)).fetchone():
                    raise ValueError(f"'{dst}' already exists")
            # Two-step rename through temporary names so swapping overlapping names cannot hit the UNIQUE key.
            for i, (src, _) in enumerate(moves):
                c.execute("UPDATE project_file SET path=? WHERE user_id=? AND path=?", (f"\x00tmp{i}", user_id, src))
            for i, (_, dst) in enumerate(moves):
                c.execute("UPDATE project_file SET path=? WHERE user_id=? AND path=?", (dst, user_id, f"\x00tmp{i}"))
            return len(moves)

    # ------------------------------------------------------------------ interactive terminal relay
    # Provider laptops are behind NAT (the server cannot reach them), so the browser and the agent both talk
    # to the server and the server stores the text between them: term_input (browser -> agent) and
    # term_output (agent -> browser). term_state remembers when the browser last looked.
    # ------------------------------------------------------------------ interactive terminal relay
    MAX_TERM_CHUNK = 4000

    def term_browser_seen(self, job_id, now=None):
        """Browser heartbeat: the terminal page is open right now (UPSERT of the time)."""
        now = time.time() if now is None else now
        with self._tx() as c:
            c.execute("INSERT INTO term_state(job_id,browser_seen) VALUES(?,?) "
                      "ON CONFLICT(job_id) DO UPDATE SET browser_seen=excluded.browser_seen", (job_id, now))

    def term_active(self, job_id, now=None, window_s=10):
        """True if the browser polled within the last window_s seconds (the terminal is 'open')."""
        now = time.time() if now is None else now
        r = self.conn.execute("SELECT browser_seen FROM term_state WHERE job_id=?", (job_id,)).fetchone()
        return bool(r and r["browser_seen"] >= now - window_s)

    def term_send(self, job_id, data):
        """Queue keyboard input from the browser for the agent. Raises ValueError if one chunk is too long
        or 50 chunks are already waiting (back-pressure so the table cannot grow without limit)."""
        if len(data) > self.MAX_TERM_CHUNK:
            raise ValueError("input too long")
        with self._tx() as c:
            pending = c.execute("SELECT COUNT(*) FROM term_input WHERE job_id=? AND delivered=0", (job_id,)).fetchone()[0]
            if pending >= 50:
                raise ValueError("too much unsent input; wait for the terminal to catch up")
            c.execute("INSERT INTO term_input(job_id,data) VALUES(?,?)", (job_id, data))

    def term_take_input(self, job_id):
        """The agent collects (and thereby acknowledges) all pending input for a job. Returns a list of strings.
        Delivered rows are marked delivered=1 up to the highest id read, in the same transaction, so none is lost or repeated."""
        with self._tx() as c:
            rows = c.execute("SELECT id,data FROM term_input WHERE job_id=? AND delivered=0 ORDER BY id", (job_id,)).fetchall()
            if rows:
                c.execute("UPDATE term_input SET delivered=1 WHERE job_id=? AND id<=?", (job_id, rows[-1]["id"]))
            return [r["data"] for r in rows]

    def term_append_output(self, job_id, data):
        """Store output text from the agent. After inserting, old rows are trimmed so only about the latest 500 remain."""
        if not data:
            return
        with self._tx() as c:
            c.execute("INSERT INTO term_output(job_id,data) VALUES(?,?)", (job_id, data[:self.MAX_TERM_CHUNK * 4]))
            # Delete rows older than (newest id - 500): a rolling buffer instead of unbounded growth.
            c.execute("DELETE FROM term_output WHERE job_id=? AND id <= "
                      "(SELECT COALESCE(MAX(id),0)-500 FROM term_output WHERE job_id=?)", (job_id, job_id))

    def term_read_output(self, job_id, after_id=0):
        """Output after after_id. Returns (joined_text, newest_id) so the browser can ask for 'only what is new'."""
        rows = self.conn.execute("SELECT id,data FROM term_output WHERE job_id=? AND id>? ORDER BY id",
                                 (job_id, int(after_id))).fetchall()
        return "".join(r["data"] for r in rows), (rows[-1]["id"] if rows else int(after_id))

    def term_clear(self, job_id):
        """Delete all terminal data of a job (input, output and state)."""
        with self._tx() as c:
            for t in ("term_input", "term_output", "term_state"):
                c.execute(f"DELETE FROM {t} WHERE job_id=?", (job_id,))

    # ------------------------------------------------------------------ integrity check (NFR2 / NFR4 / NFR9)
    def check_consistency(self):
        """Return a list of problems. An empty list means the stored state is consistent.
        Used by tests and the admin page to PROVE the HC rules hold in the stored data. Each query below
        looks for one kind of contradiction (typically a LEFT JOIN where the other side is missing)."""
        # q(sql) runs a query and returns the rows as dicts.
        q = lambda s: [dict(r) for r in self.conn.execute(s)]
        problems = []
        # HC3: a running job must have an assignment row (LEFT JOIN + 'a.job_id IS NULL' finds the missing ones).
        for r in q("SELECT j.job_id FROM job j LEFT JOIN assignment a ON a.job_id=j.job_id "
                   "WHERE j.state='running' AND a.job_id IS NULL"):
            problems.append(f"job {r['job_id']} is running without an assignment (HC3)")
        # A queued/held job must NOT have an assignment (requeue_job deletes it).
        for r in q("SELECT a.job_id FROM assignment a JOIN job j ON j.job_id=a.job_id "
                   "WHERE j.state IN ('queued','held')"):
            problems.append(f"job {r['job_id']} has an assignment but is not running/completed")
        # HC3: a running job's GPU must show that job as occupying it (IS NOT is the NULL-safe !=).
        for r in q("SELECT j.job_id FROM job j JOIN assignment a ON a.job_id=j.job_id "
                   "JOIN gpu g ON g.gpu_id=a.gpu_id WHERE j.state='running' AND g.current_job_id IS NOT j.job_id"):
            problems.append(f"job {r['job_id']} is running but its GPU does not show it as occupied")
        # Reverse check: a GPU marked busy must point at a job that exists and is running.
        for r in q("SELECT g.gpu_id, g.current_job_id FROM gpu g LEFT JOIN job j ON j.job_id=g.current_job_id "
                   "WHERE g.current_job_id IS NOT NULL AND (j.job_id IS NULL OR j.state!='running')"):
            problems.append(f"GPU {r['gpu_id']} is marked busy with job {r['current_job_id']}, which is not running")
        # HC2: GROUP BY gpu with HAVING n>1 finds a GPU with more than one running job.
        for r in q("SELECT a.gpu_id, COUNT(*) n FROM assignment a JOIN job j ON j.job_id=a.job_id "
                   "WHERE j.state='running' GROUP BY a.gpu_id HAVING n>1"):
            problems.append(f"GPU {r['gpu_id']} has {r['n']} running jobs (HC2)")
        # HC1: a running job must not sit on a GPU with less memory than it needs.
        for r in q("SELECT j.job_id FROM job j JOIN assignment a ON a.job_id=j.job_id "
                   "JOIN gpu g ON g.gpu_id=a.gpu_id WHERE j.state='running' AND g.memory_gb < j.memory_gb"):
            problems.append(f"job {r['job_id']} runs on a GPU with too little memory (HC1)")
        return problems
