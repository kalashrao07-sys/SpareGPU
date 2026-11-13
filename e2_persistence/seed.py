"""Create the database and the seed accounts:  python -m e2_persistence.seed

Run once before starting the web app. It creates the tables (init_schema) and inserts the demo
users listed in config.SEED_USERS as (username, password, role). Safe to run again: users that
already exist are reported as 'exists' and left unchanged."""
import config
from .repository import Repository


def main():
    """Entry point: create the schema, then try to create every seed user."""
    # Make sure the data/ folder exists, then open the database and create / migrate the tables.
    config.ensure_dirs()
    repo = Repository()
    repo.init()
    # Insert each seed account.
    for username, password, role in config.SEED_USERS:
        try:
            repo.create_user(username, password, role)
            print(f"created {role}: {username}")
        # create_user fails (username is UNIQUE) when the account already exists from an earlier run;
        # that is expected, so we only report it. (A broad except is acceptable in this one-off script.)
        except Exception:
            print(f"exists  {role}: {username}")
    # Always close the connection so the WAL file is flushed cleanly.
    repo.close()


# Run main() only when started as a script (python -m e2_persistence.seed), not when imported.
if __name__ == "__main__":
    main()
