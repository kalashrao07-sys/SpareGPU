# E4 package entry point: re-export the app factory so other code can write
# `from e4_web import create_app`. The noqa tells the linter the "unused" import is intentional.
from .app import create_app  # noqa: F401
