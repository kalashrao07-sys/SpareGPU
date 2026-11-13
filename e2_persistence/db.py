"""E2: SQLite connection helper (used only by repository.py and the tests).

Responsibilities:
  * connect()      - open a connection with the settings the whole project relies on:
                     WAL journal (readers and the writer do not block each other), foreign keys ON,
                     a busy timeout (wait for the write lock instead of failing at once) and
                     manual transactions (the Repository issues BEGIN IMMEDIATE / COMMIT itself).
  * init_schema()  - create the tables from schema.sql and migrate an older database in place.

Schema versions are tracked with SQLite's PRAGMA user_version:
  v2 -> v3 added project files and the terminal relay;  v3 -> v4 (SpareGPU) added price, hours,
  trusted users and the credits ledger. Only v3 -> v4 is migrated automatically; older files must be
  deleted and re-seeded (init_schema tells the user how).
"""
import os
import sqlite3

# config.py (project root) provides DB_PATH, ensure_dirs() and the other settings.
import config

# Full path of schema.sql, which sits next to this file (works wherever the project is started from).
_SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
# Current schema version, written to PRAGMA user_version after the schema is applied.
SCHEMA_VERSION = 4


def connect(path=None):
    """Open a new connection and return it. Every process / thread must open its own
    (a sqlite3 connection must not be shared between threads).
    path: database file (None = config.DB_PATH). Rows come back as sqlite3.Row (access columns by name)."""
    # Make sure the data/ folder exists before SQLite tries to create the file in it.
    config.ensure_dirs()
    # timeout=10: wait up to 10 s for a lock. isolation_level=None = autocommit mode: Python does NOT
    # open transactions behind our back, so Repository._tx() controls BEGIN/COMMIT exactly.
    conn = sqlite3.connect(path or config.DB_PATH, timeout=10, isolation_level=None)
    # Row objects allow row['name'] as well as row[0].
    conn.row_factory = sqlite3.Row
    # WAL (write-ahead log): readers never block the writer and the writer never blocks readers;
    # a crash before COMMIT leaves the main database untouched (supports the atomicity requirement).
    conn.execute("PRAGMA journal_mode=WAL")
    # SQLite ignores FOREIGN KEY constraints unless this is switched on, for EVERY connection.
    conn.execute("PRAGMA foreign_keys=ON")
    # Same idea as timeout above, in milliseconds: how long a busy database is retried before 'database is locked'.
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_schema(conn):
    """Create / upgrade the schema on this connection.
    Steps: detect an existing database (is there a 'job' table?), read its version, migrate v3 -> v4 if
    needed, refuse any other old version, then run schema.sql (all CREATE ... IF NOT EXISTS, so it is
    safe to run every start) and stamp the new version.
    Raises RuntimeError for a database from an unsupported older version."""
    # Is this a fresh (empty) database or an existing one? sqlite_master lists every table.
    have_job = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='job'").fetchone()
    # user_version is a free integer slot in the file header; 0 for a brand-new database.
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    # Existing v3 database: add the new v4 columns first so the data is kept.
    if have_job and version == 3:
        _migrate_3_to_4(conn)                 # keeps your existing data (users, jobs, providers)
        version = 4
    # An existing database that is still not at version 4 (e.g. v2) cannot be upgraded safely.
    if have_job and version != SCHEMA_VERSION:
        raise RuntimeError("This database was created by an older version of the project. "
                           "Stop everything and delete the file data/scheduler.db (and data/joblogs), "
                           "then run  python -m e2_persistence.seed  again.")
    # New tables / indexes (CREATE ... IF NOT EXISTS) are created here; existing tables are left alone.
    with open(_SCHEMA, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    # f-string is safe here because SCHEMA_VERSION is our own constant (PRAGMA cannot take ? parameters).
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


# Columns added in version 4: {table: [(column, SQL declaration), ...]}.
# ALTER TABLE ADD COLUMN needs a DEFAULT for NOT NULL columns, hence the DEFAULTs here.
# This mirrors the new columns in schema.sql so old and new databases end up identical.
_V4_COLUMNS = {
    "provider": [("price_per_hour", "REAL NOT NULL DEFAULT 0"), ("avail_from", "INTEGER"), ("avail_to", "INTEGER"),
                 ("sharing_enabled", "INTEGER NOT NULL DEFAULT 1"), ("private_only", "INTEGER NOT NULL DEFAULT 0"),
                 ("description", "TEXT")],
    "job": [("max_price_per_hour", "REAL"), ("cost", "REAL")],
    "assignment": [("rate", "REAL NOT NULL DEFAULT 0")],
}


def _migrate_3_to_4(conn):
    """Add the SpareGPU columns to a version-3 database. New tables come from schema.sql (IF NOT EXISTS).
    Idempotent: a column that already exists is skipped, so running it twice is harmless."""
    for table, cols in _V4_COLUMNS.items():
        # PRAGMA table_info lists the table's current columns (each row has a 'name').
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in cols:
            # SQLite has no 'ADD COLUMN IF NOT EXISTS', so we check first.
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
