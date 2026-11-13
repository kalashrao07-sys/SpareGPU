"""E4: login session, role guard and CSRF protection.

This small module holds the security helpers that app.py plugs into every request:

* current_user()   - who is logged in (read from the signed session cookie).
* login_required   - decorator: no login -> 401 (JSON clients) or redirect to /login (browsers).
* admin_required   - decorator: login AND role == "admin", otherwise 403.
* csrf_token() / check_csrf() - stops another website from making the browser post a form for us.
* too_many_failures / note_failure / clear_failures - login lockout (5 wrong tries per minute per IP).

All checks run on the SERVER. Hiding a button in the page is not security, because anyone can
send a request by hand, so every protected route is guarded here.
"""
import hmac
import secrets
import time
from functools import wraps

from flask import abort, g, jsonify, redirect, request, session, url_for

# In-memory table for the login limiter. Key = client IP, value = (number of failures, time of the
# first failure in the current window). It resets when the server restarts, which is fine for a
# small project; a real system would keep it in a database or Redis.
_FAILS = {}            # ip -> (count, first_failure_time): tiny brute-force limiter


def wants_json():
    """True if the caller is an API client (sent JSON, or prefers JSON over HTML).

    Used to decide the error style: JSON clients get {"error": ...} with a status code,
    browsers get a redirect or an HTML error page."""
    return request.is_json or request.accept_mimetypes.best_match(["text/html", "application/json"]) == "application/json"


def current_user():
    """Return the logged-in user row (dict-like) or None.

    The session cookie stores only the user id. We look the user up once per request and cache it
    on flask.g (a per-request scratch object), so many calls cost one database read.
    Looking up the row each time also means a deleted user or a changed role takes effect at once."""
    if "user" not in g:
        uid = session.get("user_id")
        g.user = g.repo.get_user(uid) if uid else None
    return g.user


def login_required(view):
    """Decorator: run `view` only for a logged-in user.

    Not logged in -> 401 JSON for API callers, otherwise redirect to the login page with
    ?next= so we can send them back to where they wanted to go."""
    @wraps(view)       # keep the original function name; Flask uses it as the endpoint name
    def wrapper(*a, **kw):
        if current_user() is None:
            if wants_json():
                return jsonify(error="login required"), 401
            return redirect(url_for("login", next=request.path))
        return view(*a, **kw)
    return wrapper


def admin_required(view):
    """Role check on the SERVER for every admin route (FR19, FR20): hiding buttons is not access control."""
    @wraps(view)
    @login_required        # login is checked first, so current_user() below is never None
    def wrapper(*a, **kw):
        if current_user()["role"] != "admin":
            if wants_json():
                return jsonify(error="admin only"), 403
            abort(403)
        return view(*a, **kw)
    return wrapper


def csrf_token():
    """Return this session's secret form token, creating it on first use.

    It is put in a hidden field of every HTML form. An attacker's page cannot read it
    (same-origin policy), so it cannot forge a valid form post."""
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)     # 16 random bytes = 32 hex characters
    return session["csrf"]


def check_csrf():
    """Form posts need the hidden token. JSON posts are protected by the content type
    (a cross-site page cannot send application/json without a CORS preflight, which we never allow)."""
    # Safe methods (GET) change nothing, and JSON requests are covered by the content-type rule.
    if request.method not in ("POST", "PUT", "DELETE") or request.is_json:
        return
    sent = request.form.get("csrf_token", "")
    # compare_digest compares in constant time so timing cannot leak the token.
    if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
        abort(400, "missing or invalid form token; reload the page and try again")


def too_many_failures(ip):
    """True if this IP already failed 5 or more logins within the last 60 seconds."""
    c, t = _FAILS.get(ip, (0, 0))
    if time.time() - t > 60:        # the window is over, so the old failures no longer count
        return False
    return c >= 5


def note_failure(ip):
    """Record one failed login. A failure after the 60 s window starts a fresh count of 1;
    otherwise we add 1 but keep the original window start time."""
    c, t = _FAILS.get(ip, (0, 0))
    _FAILS[ip] = (1, time.time()) if time.time() - t > 60 else (c + 1, t)


def clear_failures(ip):
    """Forget this IP's failures (called after a successful login)."""
    _FAILS.pop(ip, None)
