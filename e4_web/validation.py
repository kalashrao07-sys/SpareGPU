"""E4: input validation. Runs on the server for every request (browser checks can be bypassed).

Each validate_* function takes the raw request data (a dict from a form or JSON) and returns
two things: `clean` (trusted, correctly typed values) and `err` (field name -> message).
The route handlers in app.py only continue when `err` is empty, so nothing unchecked reaches
the database or the scheduler. Limits come from config.py so they live in one place.
"""
import math
from datetime import datetime

import config
from e2_persistence import valid_script_path


# Convert a form/JSON value to a finite float, or raise ValueError.
# Form fields arrive as strings, JSON may give numbers. bool is refused because True would
# silently become 1.0; NaN/inf are refused because every comparison with them is False.
def _num(v):
    if isinstance(v, bool):
        raise ValueError
    x = float(v)
    if not math.isfinite(x):
        raise ValueError
    return x


def validate_job(data):
    """Return (cleaned, errors). errors maps field name -> message; empty means valid (FR1, FR3).

    This is where form input becomes scheduler constraints: memory_gb, duration_min, priority,
    max_price_per_hour, framework and cpu_ram_gb are the limits the scheduler must respect."""
    # clean = the safe values we will use; err = field name -> message for the form.
    clean, err = {}, {}
    name = str(data.get("project_name") or "").strip()
    if not 1 <= len(name) <= 80:
        err["project_name"] = "give the project a name (1 to 80 characters)"
    clean["project_name"] = name
    desc = str(data.get("description") or "").strip()
    if len(desc) > 500:
        err["description"] = "at most 500 characters"
    clean["description"] = desc or None
    # The job kind decides which extra field is needed: template (prepared workload),
    # script (a file in the user's workspace) or session (interactive shell).
    kind = data.get("kind") or "template"
    if kind not in ("template", "script", "session"):
        err["kind"] = "choose a prepared workload, a script or an interactive session"
        kind = "template"
    clean["kind"] = kind
    clean["template"] = clean["script_path"] = None
    if kind == "template":
        t = data.get("template")
        # Only names listed in config.WORKLOADS are allowed, so users cannot invent a command.
        t = t.strip() if isinstance(t, str) else t
        if t not in config.WORKLOADS:
            err["template"] = "choose one of: " + ", ".join(config.WORKLOADS)
        else:
            clean["template"] = t
    elif kind == "session":
        pass                                   # a shell in a sandbox; nothing else to give
    else:
        # Windows backslashes become / first; valid_script_path then rejects '..', absolute paths
        # and odd characters, which stops a user escaping their own workspace (path traversal).
        sp = str(data.get("script_path") or "").strip().replace("\\", "/")
        if not valid_script_path(sp):
            err["script_path"] = "a path like myproject/train.py inside your workspace (letters, digits, _ - . /; no '..')"
        else:
            clean["script_path"] = sp
    fw = str(data.get("framework") or "").strip()
    if fw and fw not in config.FRAMEWORKS:
        err["framework"] = "choose one of: " + ", ".join(config.FRAMEWORKS)
    clean["framework"] = fw or None
    # cpu_ram_gb is optional: empty means 'no preference' (stored as None), else 0 < r <= 256.
    raw_ram = data.get("cpu_ram_gb")
    if raw_ram in (None, ""):
        clean["cpu_ram_gb"] = None
    else:
        try:
            r = _num(raw_ram)
            if r <= 0 or r > 256:
                raise ValueError
            clean["cpu_ram_gb"] = r
        except (TypeError, ValueError):
            err["cpu_ram_gb"] = "a number above 0 and at most 256 (or leave empty)"
    try:
        # GPU memory needed (GB). The scheduler uses this as a hard constraint when choosing a GPU.
        m = _num(data.get("memory_gb"))
        if m <= 0 or m > config.MAX_MEMORY_GB:
            raise ValueError
        clean["memory_gb"] = m
    except (TypeError, ValueError):
        err["memory_gb"] = f"must be a number above 0 and at most {config.MAX_MEMORY_GB}"
    try:
        # Estimated run time in minutes; the scheduler uses it to plan when the job finishes.
        d = _num(data.get("duration_min"))
        if d <= 0 or d > config.MAX_DURATION_MIN:
            raise ValueError
        clean["duration_min"] = d
    except (TypeError, ValueError):
        err["duration_min"] = f"must be a number above 0 and at most {config.MAX_DURATION_MIN} (minutes)"
    try:
        # Priority must be a WHOLE number 1..5. Reject True/False and 2.5, but accept '3' and 3.0.
        raw = data.get("priority")
        if isinstance(raw, bool) or (isinstance(raw, float) and raw != int(raw)):
            raise ValueError
        p = int(str(raw).strip()) if not isinstance(raw, int) else raw
        if not 1 <= p <= 5:
            raise ValueError
        clean["priority"] = p
    except (TypeError, ValueError):
        err["priority"] = "must be a whole number between 1 and 5"
    # Optional budget: the most the user will pay per GPU hour. Empty = no limit (None).
    raw_mp = data.get("max_price_per_hour")
    if raw_mp in (None, ""):
        clean["max_price_per_hour"] = None
    else:
        try:
            mp = _num(raw_mp)
            if mp < 0 or mp > 100000:
                raise ValueError
            clean["max_price_per_hour"] = mp
        except (TypeError, ValueError):
            err["max_price_per_hour"] = "a number 0 or more (or leave empty for no limit)"
    # Deadline is optional; fromisoformat accepts both dates and date+time. It is stored
    # normalised to 'YYYY-MM-DDTHH:MM'.
    dl = data.get("deadline")
    if dl in (None, ""):
        clean["deadline"] = None                      # optional, metadata only (FR2)
    else:
        try:
            clean["deadline"] = datetime.fromisoformat(str(dl).strip()).isoformat(timespec="minutes")
        except ValueError:
            err["deadline"] = "use a date like 2026-11-05 or 2026-11-05T18:30 (or leave empty)"
    return clean, err


def _minutes(v):
    """'HH:MM' -> minutes since midnight.

    Example: "09:30" -> 9*60 + 30 = 570. Storing a plain integer makes "is the GPU available
    now?" a simple number comparison later. Raises ValueError for bad input such as "25:00",
    "9" (no colon -> unpack error) or "ab:cd"."""
    h, m = str(v).strip().split(":")
    h, m = int(h), int(m)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError
    return h * 60 + m


def validate_terms(data):
    """SpareGPU lender terms: price per hour, available hours (both empty = always), private only, description."""
    clean, err = {}, {}
    try:
        p = _num(data.get("price_per_hour") or 0)
        if p < 0 or p > 100000:
            raise ValueError
        clean["price_per_hour"] = p
    except (TypeError, ValueError):
        err["price_per_hour"] = "a number 0 or more (0 = free)"
    # Availability window: both empty = always available. Otherwise both must be 'HH:MM'
    # and are stored as minutes since midnight (e.g. 09:30 -> 570). A window like 22:00-06:00
    # (start later than end) is allowed and means 'overnight'.
    f, t = data.get("avail_from"), data.get("avail_to")
    if f in (None, "") and t in (None, ""):
        clean["avail_from"] = clean["avail_to"] = None
    else:
        try:
            clean["avail_from"], clean["avail_to"] = _minutes(f), _minutes(t)
        except (TypeError, ValueError):
            err["avail_from"] = "give both times as HH:MM (or leave both empty for always)"
    # A checkbox sends 'on' when ticked and nothing when not; JSON may send true. Store as 1/0 for the database.
    clean["private_only"] = 1 if str(data.get("private_only", "")).lower() in ("1", "true", "on", "yes") else 0
    d = str(data.get("description") or "").strip()
    if len(d) > 200:
        err["description"] = "at most 200 characters"
    clean["description"] = d or None
    return clean, err


def validate_gpu(data):
    """Check a lender's GPU description (name, GPU type, memory in GB). Returns (clean, errors).

    Unlike validate_job, a bad field is simply left out of `clean` and reported in `err`."""
    clean, err = {}, {}
    name = str(data.get("name") or "").strip()
    gtype = str(data.get("gpu_type") or "").strip()
    if not 1 <= len(name) <= 60:
        err["name"] = "1 to 60 characters"
    else:
        clean["name"] = name
    if not 1 <= len(gtype) <= 30:
        err["gpu_type"] = "1 to 30 characters"
    else:
        clean["gpu_type"] = gtype
    try:
        m = _num(data.get("memory_gb"))
        if m <= 0 or m > 1024:
            raise ValueError
        clean["memory_gb"] = m
    except (TypeError, ValueError):
        err["memory_gb"] = "must be a number above 0"
    return clean, err
