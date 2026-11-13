"""E4 Submission & Status Interface Engine: the web server.

Run as its own process (started by the supervisor):  python -m e4_web.app
Talks only to E2 (through Repository). It never decides which GPU a job gets.

WHAT THIS FILE DOES
  It is a Flask web app. Users log in, submit GPU jobs, watch status and logs, edit code and use
  a terminal. Providers (laptops that share a GPU) talk to it through the /agent/* API. All data
  goes through the Repository class (E2); the scheduler (E5) decides placement, not this file.

TWO PORTALS, ONE CODEBASE
  create_app(portal=...) builds either the "college" portal (students submit jobs, only the admin
  adds GPUs) or the "exchange" portal, called SpareGPU (anyone can register, lend their laptop GPU,
  rent others' GPUs, optionally with credits). The decorator `portal_only("exchange")` makes a
  route answer 404 on the other portal, so it looks like the page does not exist there.

SECURITY FEATURES (each one is explained where it is used)
  * Session login: after /login the signed cookie stores only user_id (SECRET_KEY signs it).
  * Role guard: auth.admin_required checks the role on the server for every admin route.
  * CSRF: HTML form posts must carry a hidden token (auth.check_csrf runs before every request).
  * JSON content-type protection: JSON requests skip the token because a foreign website cannot
    send application/json to us without a CORS preflight, which we never allow.
  * Secure headers: nosniff, X-Frame-Options DENY (no clickjacking), Cache-Control no-store.
  * MAX_CONTENT_LENGTH: requests above 600 KB are rejected by Flask (413), so nobody can flood us.
  * Login lockout: 5 wrong passwords per IP per minute -> HTTP 429 (see auth.py).
  * ProxyFix + secure cookies: on Render, HTTPS ends at a proxy and reaches us as plain HTTP. ProxyFix
    trusts the X-Forwarded-* headers so the real client IP/scheme are seen, and SESSION_COOKIE_SECURE
    makes browsers send the cookie only over HTTPS.
  * Validation: validation.py turns form text into checked numbers (memory, minutes, priority,
    price) that become the constraints the scheduler uses.

ANSWER STYLE: routes return JSON when the caller asks for it (auth.wants_json) and HTML otherwise.
"""
import os
import re
import secrets
import time

from flask import (Flask, abort, flash, g, jsonify, redirect, render_template, request, session, url_for)

import config
from e2_persistence import BadState, GpuBusy, NotFound, Repository
from e3_runtime.provider_monitor import handle_heartbeat

from . import auth
from .validation import validate_gpu, validate_job, validate_terms

# Whitelist of job columns that may be shown to the browser. job_view() copies only these, so internal
# or private columns never leak into JSON/HTML.
PUBLIC_JOB_FIELDS = ("stop_requested", "job_id", "state", "template", "memory_gb", "duration_min", "priority", "deadline",
                     "held_reason", "result_note", "gpu_id", "gpu_name", "start_time", "end_time", "submitted_at",
                     "cost", "max_price_per_hour", "project_name", "description", "kind", "script_path", "framework", "cpu_ram_gb", "provider_name")


def _secret_key():
    """Return the secret used to sign session cookies.

    Stored in data/secret_key so it survives restarts (otherwise everyone would be logged out).
    It is created once from 32 random bytes. GPUSCHED_SECRET_KEY (env) overrides it in production.
    """
    config.ensure_dirs()
    path = os.path.join(config.DATA_DIR, "secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


def create_app(db_path=None, portal=None):
    """Application factory: build and return a configured Flask app.

    db_path: optional database file (tests pass a temporary one). portal: 'college' or 'exchange'
    (default from config.PORTAL). Everything below (routes, hooks) is defined inside so each app
    instance has its own portal setting.
    """
    portal = portal or config.PORTAL
    app = Flask(__name__)
    # HTTPONLY: JavaScript cannot read the cookie (limits XSS damage). SAMESITE=Lax: the browser does not send it
    # on cross-site POSTs (extra CSRF defence). SECURE: only over HTTPS when enabled. MAX_CONTENT_LENGTH caps request size.
    app.config.update(SECRET_KEY=os.environ.get("GPUSCHED_SECRET_KEY") or _secret_key(), SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=config.SECURE_COOKIES,
                      MAX_CONTENT_LENGTH=600 * 1024)
    if config.BEHIND_PROXY:                        # Render terminates HTTPS and forwards over HTTP
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    # Values every template can use without being passed in each render_template call.
    app.jinja_env.globals.update(now_ts=lambda: time.time(), csrf_token=auth.csrf_token, workloads=config.WORKLOADS, frameworks=config.FRAMEWORKS,
                              portal=portal, currency=config.CURRENCY)

    # Template filter: {{ ts|dt }} shows a unix timestamp as '07 Nov 14:05:09', or '-' when empty.
    @app.template_filter("dt")
    def fmt_dt(v):
        return time.strftime("%d %b %H:%M:%S", time.localtime(v)) if v else "-"

    @app.before_request
    def _open():
        """Runs before EVERY request: open a database handle and run the CSRF check.

        g is Flask's per-request storage, so each request gets its own Repository connection.
        """
        g.repo = Repository(db_path)
        auth.check_csrf()

    @app.teardown_request
    def _close(exc):
        """Runs after every request (even after errors): close the database connection opened in _open.
        """
        repo = g.pop("repo", None)
        if repo is not None:
            repo.close()

    @app.after_request
    def _headers(resp):
        """Add security headers to every response.

        nosniff: browser must trust our content types. X-Frame-Options DENY: our pages cannot be put in
        an <iframe> on another site (clickjacking). no-store: private pages are not kept in the cache.
        """
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.context_processor
    def _ctx():
        """Context processor: extra variables available in every template.

        user = logged-in user; pending_count = providers waiting for admin approval (admins only, for the
        menu badge); balance = spendable credits (exchange portal with credits on); has_gpu = the user
        has a running job, i.e. a GPU is allocated (the exchange menu uses it to enable the Editor).
        """
        u = auth.current_user()
        pending = g.repo.admin_stats()["pending_providers"] if (u and u["role"] == "admin") else 0
        bal = g.repo.available_credits(u["user_id"]) if (u and portal == "exchange" and config.CREDITS_ENABLED) else None
        has_gpu = bool(u and portal == "exchange" and any(j["state"] == "running" for j in g.repo.list_jobs(u["user_id"], limit=50)))
        return {"user": u, "pending_count": pending, "balance": bal, "has_gpu": has_gpu}

    def portal_only(name):
        """Decorator factory: allow the route only when this app's portal equals `name`.

        Otherwise abort(404), so e.g. /market does not even exist on the college portal.
        """
        def deco(view):
            # Imported here, inside the function, to keep the top of the file short; it works the same.
            from functools import wraps

            @wraps(view)
            def wrapper(*a, **kw):
                if portal != name:
                    abort(404)
                return view(*a, **kw)
            return wrapper
        return deco

    def fail(msg, status, **extra):
        """Report an error in the right style for the caller.

        JSON callers get ({error, ...}, status). Browser callers get a flash message and None, which
        lets the caller write `fail(...) or render_template(...)` to show the page again.
        """
        if auth.wants_json():
            return jsonify(error=msg, **extra), status
        flash(msg, "error")
        return None

    # ------------------------------------------------------------------ login
    @app.route("/")
    def index():
        """GET / (anyone). Send logged-in users to /home and everyone else to /login.
        """
        return redirect(url_for("home") if auth.current_user() else url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        """GET/POST /login (anyone).

        GET shows the form. POST reads username/password (JSON or form), applies the lockout, checks the
        password through the repository (hashed), then starts a session. Success: JSON {ok, username, role}
        or a redirect to ?next= / home. Failure: 401 (wrong login) or 429 (locked out).
        """
        if request.method == "GET":
            return render_template("login.html")
        data = request.get_json(silent=True) or request.form
        # After ProxyFix this is the real client IP, which is what the lockout needs.
        ip = request.remote_addr or "?"
        if auth.too_many_failures(ip):
            return fail("too many failed attempts; wait a minute", 429) or (render_template("login.html"), 429)
        u = g.repo.verify_login(str(data.get("username", "")), str(data.get("password", "")))
        if not u:
            auth.note_failure(ip)
            return fail("wrong username or password", 401) or (render_template("login.html"), 401)
        auth.clear_failures(ip)
        # Clear first, then set the id: any old session data (e.g. an old CSRF token) is discarded, which
        # prevents session fixation.
        session.clear()
        session["user_id"] = u["user_id"]
        if auth.wants_json():
            return jsonify(ok=True, username=u["username"], role=u["role"])
        # Only allow local paths for ?next= ('//host' would be an external site), else this is an open redirect.
        nxt = request.args.get("next", "")
        return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("home"))

    @app.route("/logout", methods=["POST"])
    def logout():
        """POST /logout (logged-in user; the CSRF token is checked in before_request). Clears the session.
        """
        session.clear()
        return redirect(url_for("login"))

    # ------------------------------------------------------------------ jobs
    def job_view(job):
        """Turn a job row into the dict shown to users: only PUBLIC_JOB_FIELDS plus queue position
        (for waiting jobs), the owner's username and seconds left (for running jobs).
        """
        d = {k: job.get(k) for k in PUBLIC_JOB_FIELDS}
        d["queue_position"] = g.repo.queue_position(job["job_id"]) if job["state"] in ("queued", "held") else None
        d["username"] = job.get("username")
        d["remaining_s"] = max(0, int(job["end_time"] - time.time())) if job["state"] == "running" and job.get("end_time") else None
        return d

    @app.route("/jobs/new")
    @auth.login_required
    def new_job():
        """GET /jobs/new (any logged-in user). Show the job submission form.

        Reads the GPU list to tell the form the biggest GPU memory and whether any GPU accepts scripts.
        The query string can prefill kind/script_path (the Editor's Run button uses this).
        """
        gpus = g.repo.list_active_gpus()
        return render_template("submit.html", errors={}, form={"priority": "3", "duration_min": "1", "memory_gb": "1", "kind": request.args.get("kind", "template"),
                                     "script_path": request.args.get("script_path", "")},
                               files=g.repo.list_files(auth.current_user()["user_id"]),
                               max_gpu=max([x["memory_gb"] for x in gpus], default=0),
                               script_ok=any(x["accepts_scripts"] for x in gpus))

    @app.route("/jobs", methods=["GET", "POST"])
    @auth.login_required
    def jobs():
        """GET/POST /jobs (any logged-in user).

        GET: list jobs (admin sees everyone's, a user only their own) as HTML or JSON.
        POST: submit a job. validate_job() cleans the input; on errors reply 400 (JSON) or redraw the form.
        A 'script' job must name a file already saved in the user's workspace. On success the job is
        stored as 'queued' (E2 submit_job) and the scheduler picks it up later: 201 JSON or redirect.
        """
        me = auth.current_user()
        if request.method == "POST":                                         # FR1, FR3
            # silent=True returns None on bad JSON instead of raising, so we can answer 400 ourselves.
            data = request.get_json(silent=True) if request.is_json else request.form
            if data is None:
                return jsonify(error="invalid JSON"), 400
            clean, errors = validate_job(data)
            if errors:
                if auth.wants_json():
                    field = next(iter(errors))
                    return jsonify(field=field, error=errors[field], errors=errors), 400
                gpus = g.repo.list_active_gpus()
                return render_template("submit.html", errors=errors, form=data, files=g.repo.list_files(me["user_id"]),
                                       max_gpu=max([x["memory_gb"] for x in gpus], default=0),
                                       script_ok=any(x["accepts_scripts"] for x in gpus)), 400
            if clean["kind"] == "script" and g.repo.get_file(me["user_id"], clean["script_path"]) is None:
                errors = {"script_path": "this file is not in your project: write and save it in the Editor first"}
                if auth.wants_json():
                    return jsonify(field="script_path", error=errors["script_path"], errors=errors), 400
                gpus = g.repo.list_active_gpus()
                return render_template("submit.html", errors=errors, form=data, files=g.repo.list_files(me["user_id"]),
                                       max_gpu=max([x["memory_gb"] for x in gpus], default=0),
                                       script_ok=any(x["accepts_scripts"] for x in gpus)), 400
            job_id = g.repo.submit_job(me["user_id"], **clean)
            if auth.wants_json():
                return jsonify(job_id=job_id, state="queued"), 201
            return redirect(url_for("job_detail", job_id=job_id))
        rows = g.repo.list_jobs(None if me["role"] == "admin" else me["user_id"])
        if auth.wants_json():
            return jsonify([job_view(j) for j in rows])
        return render_template("jobs.html", jobs=[job_view(j) for j in rows])

    @app.route("/jobs/<int:job_id>")
    @auth.login_required
    def job_detail(job_id):
        """GET /jobs/<id> (the owner or an admin). Job page (HTML) or job status (JSON).
        Others get 404 (not 403) so the existence of other people's jobs is not revealed (FR18).
        """
        me = auth.current_user()
        job = g.repo.get_job(job_id)
        if job is None or (job["user_id"] != me["user_id"] and me["role"] != "admin"):   # FR18: own jobs only
            if auth.wants_json():
                return jsonify(error="job not found"), 404
            abort(404)
        v = job_view(job)
        return jsonify(v) if auth.wants_json() else render_template("job.html", job=v)

    # ------------------------------------------------------------------ live log (polled by the job page)
    @app.route("/jobs/<int:job_id>/log")
    @auth.login_required
    def job_log(job_id):
        """GET /jobs/<id>/log?offset=N (owner or admin). Returns JSON {text, offset, state}.
        text is the log from character N onward; the page polls with the returned offset (incremental read).
        """
        me = auth.current_user()
        job = g.repo.get_job(job_id)
        if job is None or (job["user_id"] != me["user_id"] and me["role"] != "admin"):
            return jsonify(error="job not found"), 404
        try:
            offset = max(0, int(request.args.get("offset", 0)))
        except ValueError:
            offset = 0
        text, new = g.repo.read_log(job_id, offset)
        return jsonify(text=text, offset=new, state=job["state"])

    # ------------------------------------------------------------------ providers (GPU owners)
    @app.route("/providers", methods=["GET", "POST"])
    @auth.login_required
    def providers():
        """GET/POST /providers (logged-in). Provider (GPU owner) management for the college portal.

        On the exchange portal this redirects to /share. In college only the admin may use it (students get 404).
        POST registers a provider named `name` and returns a secret agent token ONCE (token is stored hashed,
        so it cannot be shown again). GET lists providers.
        """
        me = auth.current_user()
        if portal == "exchange":
            return redirect(url_for("share"))
        if me["role"] != "admin":                       # college: students never register GPUs; the admin adds demo nodes
            abort(404)
        token = None
        if request.method == "POST":
            data = (request.get_json(silent=True) if request.is_json else request.form) or {}
            name = str(data.get("name", "")).strip()
            # fullmatch = the whole name must fit the pattern (no hidden characters).
            if not re.fullmatch(r"[A-Za-z0-9_.\- ]{1,40}", name):
                msg = "name: 1 to 40 letters, digits, space, - _ ."
                return (jsonify(error=msg), 400) if auth.wants_json() else (flash(msg, "error") or redirect(url_for("providers")))
            try:
                pid, token = g.repo.create_provider(me["user_id"], name, approved=config.PROVIDER_AUTO_APPROVE)
            except ValueError:
                msg = "a provider with this name already exists"
                return (jsonify(error=msg), 409) if auth.wants_json() else (flash(msg, "error") or redirect(url_for("providers")))
            if auth.wants_json():
                return jsonify(provider_id=pid, token=token), 201
        rows = [p for p in g.repo.list_providers() if me["role"] == "admin" or p["owner_user_id"] == me["user_id"]]
        if auth.wants_json():
            return jsonify(rows)
        return render_template("providers.html", providers=rows, token=token, public_view=g.repo.list_providers())

    @app.route("/providers/<int:provider_id>/approve", methods=["POST"])
    @auth.admin_required
    def provider_approve(provider_id):
        """POST /providers/<id>/approve (admin). Approve (approved=1) or suspend (approved=0) a provider.
        Unapproved providers get no jobs (see agent_heartbeat). 404 if the provider does not exist.
        """
        data = (request.get_json(silent=True) if request.is_json else request.form) or {}
        approved = str(data.get("approved", "1")) in ("1", "true", "True")
        try:
            g.repo.set_provider_approved(provider_id, approved)
        except NotFound:
            return admin_result("provider not found", 404, ok=False)
        return admin_result("provider approved" if approved else "provider suspended") if not auth.wants_json() else jsonify(ok=True)

    @app.route("/providers/<int:provider_id>/decline", methods=["POST"])
    @auth.admin_required
    def provider_decline(provider_id):
        """POST /providers/<id>/decline (admin). Reject and delete a provider request.
        409 if the provider still has running work (BadState), 404 if unknown.
        """
        try:
            g.repo.delete_provider(provider_id)
        except NotFound:
            return admin_result("provider not found", 404, ok=False)
        except BadState as e:
            return admin_result(str(e), 409, ok=False)
        return admin_result("provider request declined and removed")

    # ------------------------------------------------------------------ SpareGPU exchange
    def _terms_form(data):
        """Thin wrapper: validate lender terms (price, hours, private-only, description).
        """
        return validate_terms(data)

    def _my_provider(provider_id):
        """Load a provider that the current user owns (admin may open any). Otherwise 404, so users cannot
        manage or even discover other people's laptops.
        """
        me = auth.current_user()
        prov = g.repo.get_provider(provider_id)
        if prov is None or (prov["owner_user_id"] != me["user_id"] and me["role"] != "admin"):
            abort(404)
        return prov

    @app.route("/register", methods=["GET", "POST"])
    @portal_only("exchange")
    def register():
        """GET/POST /register (anyone; exchange portal only). Create a SpareGPU account.

        Username 3-30 safe characters, password 8-100 chars. New users get SIGNUP_CREDITS if credits are on.
        A taken username makes register_user raise, which we report as a form error. Success logs the user in
        (new session) and goes to /market (JSON: 201).
        """
        if request.method == "GET":
            return render_template("register.html", errors={}, form={})
        data = request.get_json(silent=True) if request.is_json else request.form
        data = data or {}
        name, pw = str(data.get("username", "")).strip(), str(data.get("password", ""))
        errors = {}
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{3,30}", name):
            errors["username"] = "3 to 30 letters, digits, _ - ."
        if not 8 <= len(pw) <= 100:
            errors["password"] = "at least 8 characters"
        if not errors:
            try:
                uid = g.repo.register_user(name, pw, config.SIGNUP_CREDITS if config.CREDITS_ENABLED else 0)
            except Exception:
                errors["username"] = "this username is taken"
        if errors:
            if auth.wants_json():
                return jsonify(errors=errors), 400
            return render_template("register.html", errors=errors, form=data), 400
        session.clear()
        session["user_id"] = uid
        if auth.wants_json():
            return jsonify(ok=True, user_id=uid), 201
        flash("Welcome to SpareGPU!", "ok")
        return redirect(url_for("market"))

    @app.route("/market")
    @portal_only("exchange")
    @auth.login_required
    def market():
        """GET /market (logged-in; exchange only). The marketplace: every shared laptop with its GPUs.

        Per provider we compute: own (my laptop), trusted (the owner trusts me), locked (private-only and I am
        neither owner nor trusted: cannot book) and free (price 0, my own, or trusted: costs nothing).
        """
        me = auth.current_user()
        # set of provider ids whose owners trust this user (private/free access).
        trusted = g.repo._trusted_ids(me["user_id"])
        rows = []
        for p in g.repo.list_market():
            own = p["owner_user_id"] == me["user_id"]
            rows.append({**p, "own": own, "trusted": p["provider_id"] in trusted,
                         "locked": bool(p["private_only"]) and not own and p["provider_id"] not in trusted,
                         "free": p["price_per_hour"] == 0 or own or p["provider_id"] in trusted,
                         "gpus": [x for x in g.repo.list_active_gpus() if x["provider_id"] == p["provider_id"]]})
        if auth.wants_json():
            return jsonify(rows)
        return render_template("market.html", providers=rows)

    @app.route("/share", methods=["GET", "POST"])
    @portal_only("exchange")
    @auth.login_required
    def share():
        """GET/POST /share (logged-in; exchange only). The lender page: 'share my laptop'.

        POST registers a laptop (provider) with the chosen terms (validate_terms) and shows its agent token
        once; it is auto-approved here (approved=1) because owners are trusted to share their own machine.
        GET lists my laptops and my total earnings (ledger rows whose note starts with 'Earned').
        """
        me = auth.current_user()
        token = None
        if request.method == "POST":
            data = (request.get_json(silent=True) if request.is_json else request.form) or {}
            name = str(data.get("name", "")).strip()
            terms, errs = _terms_form(data)
            if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,40}", name):
                errs["name"] = "1 to 40 letters, digits, - _ ."
            if errs:
                if auth.wants_json():
                    return jsonify(errors=errs), 400
                for m in errs.values():
                    flash(m, "error")
                return redirect(url_for("share"))
            try:
                pid, token = g.repo.create_provider(me["user_id"], name, approved=1, **terms)
            except ValueError:
                msg = "you already have a laptop with this name"
                return (jsonify(error=msg), 409) if auth.wants_json() else (flash(msg, "error") or redirect(url_for("share")))
            if auth.wants_json():
                return jsonify(provider_id=pid, token=token), 201
        mine = [p for p in g.repo.list_providers() if p["owner_user_id"] == me["user_id"]]
        earned = sum(r["amount"] for r in g.repo.list_ledger(me["user_id"], 1000) if r["note"].startswith("Earned"))
        return render_template("share.html", providers=mine, token=token, earned=round(earned, 2), port=config.PORT)

    @app.route("/share/<int:provider_id>")
    @portal_only("exchange")
    @auth.login_required
    def share_manage(provider_id):
        """GET /share/<id> (the laptop's owner or admin; exchange only). Manage one laptop: current terms,
        trusted users and recent earnings. `f` converts minutes-since-midnight back to 'HH:MM' for the form.
        """
        prov = _my_provider(provider_id)
        p = next(x for x in g.repo.list_providers() if x["provider_id"] == provider_id)
        return render_template("share_manage.html", p=p, trusted=g.repo.list_trusted(provider_id),
                               ledger=[r for r in g.repo.list_ledger(prov["owner_user_id"], 50) if r["note"].startswith("Earned")],
                               f=lambda m: "" if m is None else "%02d:%02d" % divmod(m, 60))

    @app.route("/share/<int:provider_id>/terms", methods=["POST"])
    @portal_only("exchange")
    @auth.login_required
    def share_terms(provider_id):
        """POST /share/<id>/terms (owner/admin; exchange only). Update price, availability hours,
        private-only and description after validate_terms. Applies to NEW bookings only.
        """
        _my_provider(provider_id)
        data = (request.get_json(silent=True) if request.is_json else request.form) or {}
        terms, errs = _terms_form(data)
        if errs:
            if auth.wants_json():
                return jsonify(errors=errs), 400
            for m in errs.values():
                flash(m, "error")
        else:
            g.repo.update_provider_terms(provider_id, **terms)
            if auth.wants_json():
                return jsonify(ok=True)
            flash("Sharing terms saved. They apply to new bookings only.", "ok")
        return redirect(url_for("share_manage", provider_id=provider_id))

    @app.route("/share/<int:provider_id>/pause", methods=["POST"])
    @portal_only("exchange")
    @auth.login_required
    def share_pause(provider_id):
        """POST /share/<id>/pause (owner/admin; exchange only). enabled=1 resumes sharing, 0 pauses it.
        Pausing blocks new bookings but jobs already running continue.
        """
        _my_provider(provider_id)
        data = (request.get_json(silent=True) if request.is_json else request.form) or {}
        on = str(data.get("enabled", "0")) in ("1", "true", "True")
        g.repo.set_sharing(provider_id, on)
        if auth.wants_json():
            return jsonify(ok=True, sharing=on)
        flash("Sharing is ON: new renters can book." if on else "Sharing paused: no new bookings (running jobs continue).", "ok")
        return redirect(url_for("share_manage", provider_id=provider_id))

    @app.route("/share/<int:provider_id>/stop", methods=["POST"])
    @portal_only("exchange")
    @auth.login_required
    def share_stop(provider_id):
        """POST /share/<id>/stop (owner/admin; exchange only). Emergency stop: sharing off AND every running
        job on this laptop is killed (the owner always has the final say over their machine).
        """
        _my_provider(provider_id)
        g.repo.owner_stop_all(provider_id)
        if auth.wants_json():
            return jsonify(ok=True)
        flash("Stopped: sharing is off and every running job on this laptop is being killed.", "ok")
        return redirect(url_for("share_manage", provider_id=provider_id))

    @app.route("/share/<int:provider_id>/trust", methods=["POST"])
    @portal_only("exchange")
    @auth.login_required
    def share_trust(provider_id):
        """POST /share/<id>/trust (owner/admin; exchange only). Add (or with remove=1 remove) a trusted
        user by username. Trusted users can use a private-only laptop for free. 404 if the user is unknown.
        """
        _my_provider(provider_id)
        data = (request.get_json(silent=True) if request.is_json else request.form) or {}
        u = g.repo.get_user_by_name(str(data.get("username", "")).strip())
        remove = str(data.get("remove", "")) in ("1", "true")
        if u is None:
            msg = "no such user"
            return (jsonify(error=msg), 404) if auth.wants_json() else (flash(msg, "error") or redirect(url_for("share_manage", provider_id=provider_id)))
        # Pick the method first, then call it: untrust_user(...) or trust_user(...).
        (g.repo.untrust_user if remove else g.repo.trust_user)(provider_id, u["user_id"])
        if auth.wants_json():
            return jsonify(ok=True)
        return redirect(url_for("share_manage", provider_id=provider_id))

    @app.route("/wallet")
    @portal_only("exchange")
    @auth.login_required
    def wallet():
        """GET /wallet (logged-in; exchange only, and only when CREDITS_ENABLED, else 404).
        Shows balance, credits reserved for running jobs, and the ledger (history). JSON gives balance,
        available (balance minus reserved) and ledger.
        """
        if not config.CREDITS_ENABLED:
            abort(404)
        me = auth.current_user()
        if auth.wants_json():
            return jsonify(balance=g.repo.balance(me["user_id"]), available=g.repo.available_credits(me["user_id"]),
                           ledger=g.repo.list_ledger(me["user_id"]))
        return render_template("wallet.html", balance=g.repo.balance(me["user_id"]), reserved=g.repo.reserved(me["user_id"]),
                               ledger=g.repo.list_ledger(me["user_id"]))

    @app.route("/admin/credits", methods=["GET", "POST"])
    @portal_only("exchange")
    @auth.admin_required
    def admin_credits():
        """GET/POST /admin/credits (admin; exchange only; needs CREDITS_ENABLED).
        POST gives (positive) or removes (negative) credits for a user, recorded in the ledger.
        The amount must be a real number within +/-100000; `amt != amt` is True only for NaN, so NaN is refused.
        """
        if not config.CREDITS_ENABLED:
            abort(404)
        if request.method == "POST":
            data = (request.get_json(silent=True) if request.is_json else request.form) or {}
            u = g.repo.get_user_by_name(str(data.get("username", "")).strip())
            try:
                amt = float(data.get("amount"))
                if not -100000 <= amt <= 100000 or amt != amt:
                    raise ValueError
            except (TypeError, ValueError):
                amt = None
            if u is None or amt is None or amt == 0:
                return admin_result("give an existing username and a non-zero amount", 400, ok=False) or redirect(url_for("admin_credits"))
            g.repo.credit(u["user_id"], amt, "Admin grant" if amt > 0 else "Admin adjustment")
            return admin_result(f"{amt:+g} credits for {u['username']}") or redirect(url_for("admin_credits"))
        return render_template("admin_credits.html")

    # ------------------------------------------------------------------ agent API (called by the provider laptops)
    def agent_provider():
        """Identify the calling laptop from its header 'Authorization: Bearer <token>'.
        Returns the provider row or None. h[7:] drops the 7 characters 'Bearer '. No session or CSRF is
        used here: the agent is a program, and the secret token is its proof of identity.
        """
        h = request.headers.get("Authorization", "")
        prov = g.repo.get_provider_by_token(h[7:].strip()) if h.startswith("Bearer ") else None
        return prov

    @app.route("/agent/heartbeat", methods=["POST"])
    def agent_heartbeat():
        """POST /agent/heartbeat (provider agent; Bearer token). Called every few seconds by the laptop agent.

        The body is JSON (GPU stats, finished jobs...). handle_heartbeat (E3) records it and replies with
        jobs the laptop should launch/stop. This also tells us the laptop is online. Unapproved providers get
        an empty launch list plus a note. 401 bad token, 400 bad JSON.
        """
        prov = agent_provider()
        if prov is None:
            return jsonify(error="invalid provider token"), 401
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="invalid JSON"), 400
        try:
            reply = handle_heartbeat(g.repo, prov, data)
        except (KeyError, TypeError, ValueError) as e:
            return jsonify(error=f"bad heartbeat: {e}"), 400
        if not prov["approved"]:
            reply["launches"] = []
            reply["note"] = "waiting for the admin to approve this provider"
        return jsonify(reply)

    # Security rule shared by the agent routes: a laptop may only touch jobs running on ITS OWN GPUs.
    def _own_job(prov, job_id):
        """A provider may only touch jobs assigned to one of ITS GPUs."""
        return any(a["job_id"] == job_id for a in g.repo.running_assignments(prov["provider_id"]))

    @app.route("/agent/log", methods=["POST"])
    def agent_log():
        """POST /agent/log (provider agent). Append up to 20000 characters of job output to the job's log
        (shown live on the job page). 403 if the job is not on this provider.
        """
        prov = agent_provider()
        if prov is None:
            return jsonify(error="invalid provider token"), 401
        data = request.get_json(silent=True) or {}
        try:
            job_id = int(data.get("job_id"))
        except (TypeError, ValueError):
            return jsonify(error="job_id required"), 400
        if not _own_job(prov, job_id):
            return jsonify(error="not your job"), 403
        g.repo.append_log(job_id, str(data.get("text", ""))[:20000])
        return jsonify(ok=True)

    @app.route("/agent/finish", methods=["POST"])
    def agent_finish():
        """POST /agent/finish (provider agent). Report that a job ended; E2 marks it completed with a note.
        Always answers ok, so a repeated or late report is harmless (idempotent).
        """
        prov = agent_provider()
        if prov is None:
            return jsonify(error="invalid provider token"), 401
        data = request.get_json(silent=True) or {}
        try:
            job_id = int(data.get("job_id"))
        except (TypeError, ValueError):
            return jsonify(error="job_id required"), 400
        if _own_job(prov, job_id):
            g.repo.complete_job(job_id, str(data.get("note", "finished"))[:300])
        return jsonify(ok=True)       # idempotent: a repeated or late finish is simply acknowledged

    @app.route("/agent/project", methods=["POST"])
    def agent_project():
        """The provider fetches the project files of the user whose job it is about to run (and only then)."""
        prov = agent_provider()
        if prov is None:
            return jsonify(error="invalid provider token"), 401
        data = request.get_json(silent=True) or {}
        try:
            job_id = int(data.get("job_id"))
        except (TypeError, ValueError):
            return jsonify(error="job_id required"), 400
        a = next((x for x in g.repo.running_assignments(prov["provider_id"]) if x["job_id"] == job_id), None)
        if a is None or a["kind"] == "template":
            return jsonify(error="not your job"), 403
        return jsonify(files=g.repo.get_project(a["user_id"]))

    @app.route("/agent/term", methods=["POST"])
    def agent_term():
        """POST /agent/term (provider agent). Terminal bridge: the agent uploads new shell output and
        receives the keystrokes the user typed in the browser (term_take_input), all through the database.
        """
        prov = agent_provider()
        if prov is None:
            return jsonify(error="invalid provider token"), 401
        data = request.get_json(silent=True) or {}
        try:
            job_id = int(data.get("job_id"))
        except (TypeError, ValueError):
            return jsonify(error="job_id required"), 400
        if not _own_job(prov, job_id):
            return jsonify(error="not your job"), 403
        g.repo.term_append_output(job_id, str(data.get("output", "")))
        return jsonify(input=g.repo.term_take_input(job_id))

    # ------------------------------------------------------------------ stop, terminal (browser side)
    @app.route("/jobs/<int:job_id>/stop", methods=["POST"])
    @auth.login_required
    def job_stop(job_id):
        """POST /jobs/<id>/stop (owner or admin). Cancel a job. A queued job is cancelled at once; a running
        one gets a stop request that the provider carries out in a few seconds. 404 if not found/finished.
        """
        me = auth.current_user()
        res = g.repo.cancel_job(job_id, None if me["role"] == "admin" else me["user_id"])
        if res is None:
            return (jsonify(error="job not found or already finished"), 404) if auth.wants_json() else abort(404)
        if auth.wants_json():
            return jsonify(result=res)
        flash("Job cancelled." if res == "cancelled" else "Stop requested; the provider ends the job within a few seconds.", "ok")
        return redirect(url_for("job_detail", job_id=job_id))

    def _term_job(job_id):
        """Check that the current user may use job_id's terminal. Returns (job, None) or (None, error response).

        Rules: only the job's OWNER (not even admin: it is a private shell), only for script/session jobs
        (not templates), and only while the job is running (else 409).
        """
        me = auth.current_user()
        job = g.repo.get_job(job_id)
        if job is None or job["user_id"] != me["user_id"]:       # only the owner (not even admin) gets a shell
            return None, (jsonify(error="job not found"), 404)
        if job["kind"] == "template":
            return None, (jsonify(error="terminal is for your own code jobs and sessions"), 400)
        if job["state"] != "running":
            return None, (jsonify(error="the terminal is available while the job is running", state=job["state"]), 409)
        return job, None

    @app.route("/jobs/<int:job_id>/term", methods=["GET"])
    @auth.login_required
    def term_read(job_id):
        """GET /jobs/<id>/term?after=N (owner). Terminal output after position N as JSON {text, after}.
        term_browser_seen tells the agent somebody is watching, so it keeps streaming output.
        """
        job, err = _term_job(job_id)
        if err:
            return err
        g.repo.term_browser_seen(job_id)                        # tells the agent someone is watching
        text, last = g.repo.term_read_output(job_id, request.args.get("after", 0))
        return jsonify(text=text, after=last)

    @app.route("/jobs/<int:job_id>/term", methods=["POST"])
    @auth.login_required
    def term_write(job_id):
        """POST /jobs/<id>/term (owner). Send typed text (JSON {data}) to the job's shell via the database queue.
        ValueError from the repository (e.g. too much input) becomes 400.
        """
        job, err = _term_job(job_id)
        if err:
            return err
        data = request.get_json(silent=True) or {}
        try:
            g.repo.term_send(job_id, str(data.get("data", "")))
        except ValueError as e:
            return jsonify(error=str(e)), 400
        g.repo.term_browser_seen(job_id)
        return jsonify(ok=True)

    # ------------------------------------------------------------------ code editor
    @app.route("/editor")
    @auth.login_required
    def editor():
        """GET /editor (logged-in). The in-browser code editor page.

        On the exchange portal the Editor is blocked until the user has a running job, i.e. a GPU is allocated
        (the same test as has_gpu above): the editor exists to prepare code for rented GPU time, and this
        stops people using the platform as a free code host. Others are sent to /market with a message.
        """
        if portal == "exchange" and not any(j["state"] == "running" for j in g.repo.list_jobs(auth.current_user()["user_id"], limit=50)):
            flash("The editor opens once a GPU has been allocated to you.", "error")
            return redirect(url_for("market"))
        return render_template("editor.html")

    @app.route("/api/files", methods=["GET"])
    @auth.login_required
    def files_get():
        """GET /api/files (logged-in). Without ?path: list the user's files. With ?path=x: return that file's
        text. Files are per user, so one user can never read another's workspace. 404 if missing.
        """
        me = auth.current_user()
        path = request.args.get("path")
        if path is None:
            return jsonify(files=g.repo.list_files(me["user_id"]))
        content = g.repo.get_file(me["user_id"], path)
        if content is None:
            return jsonify(error="file not found"), 404
        return jsonify(path=path, content=content)

    @app.route("/api/files", methods=["PUT"])
    @auth.login_required
    def files_put():
        """PUT /api/files (logged-in). Create or overwrite a file from JSON {path, content}.
        Both must be strings. The repository checks the path (no '..') and size and raises ValueError -> 400.
        """
        me = auth.current_user()
        data = request.get_json(silent=True)                    # JSON only: protected from cross-site requests
        if not isinstance(data, dict) or not isinstance(data.get("path"), str) or not isinstance(data.get("content"), str):
            return jsonify(error="path and content required"), 400
        try:
            g.repo.save_file(me["user_id"], data["path"], data["content"])
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True)

    @app.route("/api/files/rename", methods=["POST"])
    @auth.login_required
    def files_rename():
        """POST /api/files/rename (logged-in). JSON {path, new_path}: rename a file or a whole folder.
        400 with a message when the name is invalid, missing or already used (nothing is changed then)."""
        me = auth.current_user()
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("path"), str) or not isinstance(data.get("new_path"), str):
            return jsonify(error="path and new_path required"), 400
        try:
            n = g.repo.rename_path(me["user_id"], data["path"], data["new_path"])
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True, moved=n)

    @app.route("/api/gpu-status")
    @auth.login_required
    def gpu_status():
        """GET /api/gpu-status (logged-in). Honest GPU availability for the editor's Run button: how many GPUs
        are online, how many are free, the largest memory, and whether any of them accepts user code (sandbox on).
        The editor refuses to start a run when this says there is nothing to run on, instead of pretending."""
        gpus = g.repo.list_active_gpus()
        return jsonify(online=len(gpus), free=sum(1 for x in gpus if x["current_job_id"] is None),
                       max_memory_gb=max([x["memory_gb"] for x in gpus], default=0),
                       scripts_ok=any(x["accepts_scripts"] for x in gpus))

    @app.route("/api/files", methods=["DELETE"])
    @auth.login_required
    def files_delete():
        """DELETE /api/files (logged-in). Delete the file named in JSON {path}, or a whole folder with
        {path, folder: true}. 404 if it does not exist.
        """
        me = auth.current_user()
        data = request.get_json(silent=True) or {}
        if data.get("folder"):
            try:
                n = g.repo.delete_prefix(me["user_id"], str(data.get("path", "")))
            except ValueError as e:
                return jsonify(error=str(e)), 400
            return (jsonify(ok=True, deleted=n) if n else (jsonify(error="folder not found"), 404))
        if not g.repo.delete_file(me["user_id"], str(data.get("path", ""))):
            return jsonify(error="file not found"), 404
        return jsonify(ok=True)

    @app.route("/api/files/backup.zip")
    @auth.login_required
    def files_backup():
        """Download the whole project as a zip: the user's own backup (the server is not the only copy)."""
        # Build the zip in memory (BytesIO), so no temporary file is written to disk.
        import io, zipfile
        me = auth.current_user()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for path, content in g.repo.get_project(me["user_id"]).items():
                z.writestr(path, content)
        return app.response_class(buf.getvalue(), mimetype="application/zip", headers={
            "Content-Disposition": f'attachment; filename="{me["username"]}_project.zip"'})

    # ------------------------------------------------------------------ connect info (remote access)
    @app.route("/connect")
    @auth.login_required
    def connect():
        """GET /connect (logged-in; on the college portal admin only). Shows this machine's network
        addresses so a provider laptop knows what URL to put in its agent. 100.x = Tailscale VPN range,
        127.x (localhost) is skipped. Needs psutil.
        """
        if portal == "college" and auth.current_user()["role"] != "admin":
            abort(404)
        import socket
        import psutil
        addrs = []
        for name, infos in psutil.net_if_addrs().items():
            for i in infos:
                if i.family == socket.AF_INET and not i.address.startswith("127."):
                    kind = "Tailscale (works from any network)" if i.address.startswith("100.") else "local network (same Wi-Fi)"
                    addrs.append({"iface": name, "ip": i.address, "kind": kind})
        return render_template("connect.html", addrs=addrs, port=config.PORT)

    # ------------------------------------------------------------------ pool (FR17)
    @app.route("/pool")
    @auth.login_required
    def pool():
        """GET /pool (logged-in; FR17). Live view of all GPUs: free/busy, memory, utilisation and queue length.

        Privacy: admin sees who runs what; a normal user sees only their own job and 'another user' for others.
        """
        me = auth.current_user()
        rows = []
        for r in g.repo.pool_view():
            busy = r["current_job_id"] is not None
            mine = busy and r["run_user_id"] == me["user_id"]
            rows.append({
                "gpu_id": r["gpu_id"], "name": r["name"], "gpu_type": r["gpu_type"], "memory_gb": r["memory_gb"],
                "status": r["status"], "simulated": bool(r["is_simulated"]), "busy": busy,
                "provider": r["provider_name"], "online": bool(r["online"]),
                "job_id": r["run_job_id"] if (mine or me["role"] == "admin") else None,
                "who": r["run_username"] if me["role"] == "admin" else ("you" if mine else ("another user" if busy else None)),
                "free_in_s": max(0, int(r["run_end_time"] - time.time())) if busy and r["run_end_time"] else None,
                "mem_used_mb": r["mem_used_mb"], "util_pct": r["util_pct"]})
        queue = len(g.repo.list_open_jobs())
        if auth.wants_json():
            return jsonify(gpus=rows, queue_length=queue)
        return render_template("pool.html", gpus=rows, queue=queue)

    # ------------------------------------------------------------------ home (landing page after sign-in)
    @app.route("/home")
    @auth.login_required
    def home():
        """GET /home (logged-in). Landing page after login: admin -> /admin, exchange user -> /market,
        college student -> a small dashboard with counts of running and waiting jobs.
        """
        me = auth.current_user()
        if me["role"] == "admin":
            return redirect(url_for("admin"))
        if portal == "exchange":
            return redirect(url_for("market"))
        mine = g.repo.list_jobs(me["user_id"], limit=200)
        counts = {"running": sum(1 for j in mine if j["state"] == "running"),
                  "waiting": sum(1 for j in mine if j["state"] in ("queued", "held"))}
        return render_template("home.html", counts=counts, online=g.repo.admin_stats()["online_providers"])

    # ------------------------------------------------------------------ admin (FR19, FR20)
    @app.route("/admin")
    @auth.admin_required
    def admin():
        """GET /admin (admin; FR19/FR20). Admin dashboard (HTML) or JSON with all GPUs and held jobs
        (held = could not be placed yet; held_reason explains why).
        """
        if auth.wants_json():
            return jsonify(gpus=g.repo.list_all_gpus(), held=[job_view(j) for j in g.repo.list_held_jobs()])
        return render_template("admin_dashboard.html", stats=g.repo.admin_stats(), fleet=g.repo.fleet_view())

    @app.route("/admin/requests")
    @auth.admin_required
    def admin_requests():
        """GET /admin/requests (admin). Providers waiting for approval with their GPUs, plus held jobs.
        """
        pending = [p for p in g.repo.list_providers() if not p["approved"]]
        gpus_by = {p["provider_id"]: [x for x in g.repo.list_all_gpus() if x["provider_id"] == p["provider_id"]] for p in pending}
        return render_template("admin_requests.html", pending=pending, gpus_by=gpus_by,
                               held=[job_view(j) for j in g.repo.list_held_jobs()])

    @app.route("/admin/fleet")
    @auth.admin_required
    def admin_fleet():
        """GET /admin/fleet (admin). Overview of every provider and its GPUs, with statistics.
        """
        return render_template("admin_fleet.html", fleet=g.repo.fleet_view(), stats=g.repo.admin_stats())

    def _devices_page(token=None, new_server=None):
        """Render the GPU management page: every server with its GPUs, true state and live utilisation."""
        now = time.time()
        rows = g.repo.pool_view()
        for r in rows:
            # One honest state per GPU, most specific first.
            if r["status"] == "removed":
                r["state"], r["cls"] = "disabled", "removed"
            elif r["provider_id"] is not None and not r["provider_approved"]:
                r["state"], r["cls"] = "server suspended", "removed"
            elif not r["online"]:
                r["state"], r["cls"] = "offline (no worker)", "held"
            elif r["current_job_id"] is not None:
                r["state"], r["cls"] = "occupied", "running"
            else:
                r["state"], r["cls"] = "available", "free"
            fresh = r["stats_at"] is not None and now - r["stats_at"] <= 15
            r["live"] = fresh and r["online"]
        servers = g.repo.list_providers()
        return render_template("admin_devices.html", gpus=rows, servers=servers, token=token, new_server=new_server)

    @app.route("/admin/devices")
    @auth.admin_required
    def admin_devices():
        """GET /admin/devices (admin). GPU management: servers, their GPUs, state and utilisation."""
        return _devices_page()

    @app.route("/admin/servers", methods=["POST"])
    @auth.admin_required
    def admin_add_server():
        """POST /admin/servers (admin). Register a GPU server with its GPU model and memory.
        Shows the agent token once. The server is offline until its worker connects (no fake availability)."""
        data = (request.get_json(silent=True) if request.is_json else request.form) or {}
        name = str(data.get("name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\- ]{1,40}", name):
            return admin_result("server name: 1 to 40 letters, digits, space, - _ .", 400, ok=False)
        clean, errors = validate_gpu({"name": data.get("gpu_name"), "gpu_type": data.get("gpu_type") or data.get("gpu_name"),
                                      "memory_gb": data.get("memory_gb")})
        if errors:
            return admin_result("; ".join(f"{k}: {v}" for k, v in errors.items()), 400, ok=False)
        me = auth.current_user()
        try:
            pid, token = g.repo.add_server(me["user_id"], name, clean["name"], clean["gpu_type"], clean["memory_gb"])
        except ValueError:
            return admin_result("a server with this name already exists (or the values were rejected)", 409, ok=False)
        if auth.wants_json():
            return jsonify(provider_id=pid, token=token), 201
        return _devices_page(token=token, new_server=name)

    @app.route("/admin/gpus/<int:gpu_id>/update", methods=["POST"])
    @auth.admin_required
    def admin_update_gpu(gpu_id):
        """POST /admin/gpus/<id>/update (admin). Change a GPU's model, type and memory (not while it is busy)."""
        data = (request.get_json(silent=True) if request.is_json else request.form) or {}
        clean, errors = validate_gpu({"name": data.get("name"), "gpu_type": data.get("gpu_type") or data.get("name"),
                                      "memory_gb": data.get("memory_gb")})
        if errors:
            return admin_result("; ".join(f"{k}: {v}" for k, v in errors.items()), 400, ok=False)
        try:
            g.repo.update_gpu(gpu_id, clean["name"], clean["gpu_type"], clean["memory_gb"])
        except GpuBusy:
            return admin_result("cannot edit: a job is running on this GPU", 409, ok=False)
        except NotFound:
            return admin_result("GPU not found", 404, ok=False)
        except ValueError as e:
            return admin_result(str(e), 400, ok=False)
        return admin_result("GPU updated")

    @app.route("/admin/held")
    @auth.admin_required
    def admin_held():
        """GET /admin/held (admin). JSON list of held jobs.
        """
        return jsonify([job_view(j) for j in g.repo.list_held_jobs()])

    def admin_result(msg, status=200, ok=True):
        """Shared reply for admin actions: JSON {ok, message} / {error} for API callers; for browsers a
        flash message and a redirect to the hidden field `back` (local path only, to avoid open redirects)
        or to the devices page.
        """
        if auth.wants_json():
            return jsonify(ok=ok, message=msg) if ok else (jsonify(error=msg), status)
        flash(msg, "ok" if ok else "error")
        back = request.form.get("back", "")
        return redirect(back if back.startswith("/") and not back.startswith("//") else url_for("admin_devices"))

    @app.route("/admin/gpus", methods=["POST"])
    @auth.admin_required
    def admin_add_gpu():
        """POST /admin/gpus (admin). Add a SIMULATED GPU record (for demos). Real GPUs are found
        automatically from agents. Disabled (403) unless ALLOW_SIMULATED_GPUS is set. Validated by validate_gpu.
        """
        if not config.ALLOW_SIMULATED_GPUS:
            return admin_result("simulated GPU records are disabled until the guide approves them; "
                                "real GPUs are discovered automatically", 403, ok=False)
        data = request.get_json(silent=True) if request.is_json else request.form
        clean, errors = validate_gpu(data or {})
        if errors:
            return admin_result("; ".join(f"{k}: {v}" for k, v in errors.items()), 400, ok=False)
        gid = g.repo.add_gpu(clean["name"], clean["gpu_type"], clean["memory_gb"], is_simulated=1)
        return admin_result(f"GPU {gid} added (simulated)")

    @app.route("/admin/gpus/<int:gpu_id>/remove", methods=["POST"])
    @auth.admin_required
    def admin_remove_gpu(gpu_id):
        """POST /admin/gpus/<id>/remove (admin). Take a GPU out of the pool. 409 if a job is running on it.
        """
        try:
            g.repo.remove_gpu(gpu_id)
        except GpuBusy:
            return admin_result("cannot remove: a job is running on this GPU (wait until it finishes)", 409, ok=False)
        except NotFound:
            return admin_result("GPU not found", 404, ok=False)
        return admin_result(f"GPU {gpu_id} removed from the pool")

    @app.route("/admin/gpus/<int:gpu_id>/restore", methods=["POST"])
    @auth.admin_required
    def admin_restore_gpu(gpu_id):
        """POST /admin/gpus/<id>/restore (admin). Put a removed GPU back into the pool.
        """
        try:
            g.repo.restore_gpu(gpu_id)
        except NotFound:
            return admin_result("GPU not found", 404, ok=False)
        return admin_result(f"GPU {gpu_id} is back in the pool")

    @app.route("/admin/gpus/<int:gpu_id>/rename", methods=["POST"])
    @auth.admin_required
    def admin_rename_gpu(gpu_id):
        """POST /admin/gpus/<id>/rename (admin). Rename a GPU (1 to 60 characters).
        """
        data = request.get_json(silent=True) if request.is_json else request.form
        name = str((data or {}).get("name", "")).strip()
        if not 1 <= len(name) <= 60:
            return admin_result("name must be 1 to 60 characters", 400, ok=False)
        try:
            g.repo.rename_gpu(gpu_id, name)
        except NotFound:
            return admin_result("GPU not found", 404, ok=False)
        return admin_result("GPU renamed")

    # Error pages: JSON for API callers, a friendly HTML page for browsers.
    @app.errorhandler(403)
    def _403(e):
        return (jsonify(error="not permitted"), 403) if auth.wants_json() else (render_template("error.html", code=403, msg="You are not allowed to do that."), 403)

    @app.errorhandler(404)
    def _404(e):
        return (jsonify(error="not found"), 404) if auth.wants_json() else (render_template("error.html", code=404, msg="Page or job not found."), 404)

    @app.errorhandler(400)
    def _400(e):
        return (jsonify(error=getattr(e, "description", "bad request")), 400) if auth.wants_json() else (render_template("error.html", code=400, msg=getattr(e, "description", "Bad request")), 400)

    return app


def main():
    """Start the development server (used when run as `python -m e4_web.app`). threaded=True lets
    several requests (e.g. the agents' heartbeats and a user's polling) be served at once.
    """
    app = create_app()
    print(f"[E4] web server on http://{config.HOST}:{config.PORT}", flush=True)
    app.run(host=config.HOST, port=config.PORT, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
