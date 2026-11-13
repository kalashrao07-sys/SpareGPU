"""E2 Persistence & Transaction Engine (package).

Only this engine touches the SQLite database. Other engines import from here:
    Repository         - all reads/writes (see repository.py for the full description)
    AssignmentRejected - commit_assignment() refused: a hard constraint (HC1-HC7) would be broken
    GpuBusy            - tried to remove a GPU that has a running job
    NotFound, BadState - unknown id / operation not allowed in the job's current state
    valid_script_path, valid_file_path - helpers that validate user-supplied file paths
Other files: db.py (connection + schema migration), schema.sql (tables), seed.py (demo users).
"""
# Re-export the public names so callers can write `from e2_persistence import Repository`.
# (noqa: F401 tells the linter these imports are intentionally 'unused' in this file.)
from .repository import Repository, AssignmentRejected, GpuBusy, NotFound, BadState, valid_script_path, valid_file_path  # noqa: F401
