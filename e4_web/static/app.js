// Live refresh: poll the JSON endpoints every 3 s and update the page. Text only (textContent), never HTML.
// Wrapped in an immediately-invoked function so our variables stay private and do not clash with other scripts.
// Every element is looked up defensively: this one file serves several pages (job page, pool page, submit form),
// and only the parts whose elements exist on the current page do anything.
(function () {
  var jobEl = document.getElementById("job");
  var poolBody = document.getElementById("pool-body");
  // Shared fetch options: ask for JSON and send the session cookie so the server knows who we are (GET needs no CSRF token).
  var opts = { headers: { "Accept": "application/json" }, credentials: "same-origin" };

  // Put a value into the element marked data-field=NAME. textContent (not innerHTML) means server text can never inject HTML (XSS safe).
  function setField(name, value) {
    var el = document.querySelector('[data-field="' + name + '"]');
    if (el) el.textContent = (value === null || value === undefined || value === "") ? "-" : value;
  }

  // Job page: GET /jobs/<id> as JSON and update state badge, GPU, queue position, note and a friendly status message.
  function refreshJob() {
    fetch("/jobs/" + jobEl.dataset.jobId, opts).then(function (r) { return r.json(); }).then(function (j) {
      var s = document.querySelector('[data-field="state"]');
      if (s) { s.textContent = j.state; s.className = "badge " + j.state; }
      setField("gpu_name", j.gpu_name);
      setField("queue_position", j.queue_position);
      setField("result_note", j.result_note);
      // Turn the state into a sentence for the student. remaining_s is null when unknown.
      var msg = "";
      if (j.state === "held") msg = "Waiting: " + (j.held_reason || "");
      else if (j.state === "queued") msg = "Waiting for the scheduler to place this job.";
      else if (j.state === "running") msg = "Running on " + (j.gpu_name || "a GPU") + (j.remaining_s !== null ? " - about " + j.remaining_s + " s left." : ".");
      else if (j.state === "completed") msg = "Finished.";
      setField("message", msg);
    // A failed poll (e.g. server restarting) is ignored; the next tick simply tries again.
    }).catch(function () {});
  }

  // Helper: append a <td> to row tr. With cls the text is shown as a coloured badge, otherwise plain text.
  function cell(tr, text, cls) {
    var td = document.createElement("td");
    if (cls) { var b = document.createElement("span"); b.className = "badge " + cls; b.textContent = text; td.appendChild(b); }
    else td.textContent = text;
    tr.appendChild(td);
  }

  // Pool page: GET /pool and rebuild the table of GPUs and the queue length. The table is cleared and redrawn each time.
  function refreshPool() {
    fetch("/pool", opts).then(function (r) { return r.json(); }).then(function (p) {
      document.getElementById("queue").textContent = p.queue_length;
      poolBody.textContent = "";
      p.gpus.forEach(function (g) {
        var tr = document.createElement("tr");
        cell(tr, g.name + (g.simulated ? " (simulated)" : ""));
        var pv = document.createElement("td");
        pv.textContent = (g.provider || "-") + " ";
        var ob = document.createElement("span"); ob.className = "badge " + (g.online ? "free" : "removed"); ob.textContent = g.online ? "online" : "offline";
        pv.appendChild(ob); tr.appendChild(pv);
        cell(tr, g.memory_gb + " GB");
        // Status is derived: busy beats removed beats free.
        var st = g.busy ? "busy" : (g.status === "removed" ? "removed" : "free");
        cell(tr, st, g.busy ? "running" : (g.status === "removed" ? "removed" : "free"));
        var who = g.busy ? (g.who + (g.job_id ? " - job #" + g.job_id : "") + (g.free_in_s !== null ? " - free in ~" + g.free_in_s + " s" : "")) : "-";
        cell(tr, who);
        cell(tr, g.util_pct !== null ? Math.round(g.util_pct) + "% - " + Math.round(g.mem_used_mb) + " MB" : "-");
        poolBody.appendChild(tr);
      });
    }).catch(function () {});
  }

  // Live job log. logOffset = how many characters we already have, so the server only sends the new part (incremental tail).
  var logEl = document.getElementById("log"), logOffset = 0, logDone = false;
  function refreshLog() {
    if (!logEl || logDone) return;
    fetch("/jobs/" + jobEl.dataset.jobId + "/log?offset=" + logOffset, opts).then(function (r) { return r.json(); }).then(function (d) {
      if (d.text) {
        // Only auto-scroll if the user was already at the bottom, so scrolling up to read is not disturbed.
        var atEnd = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20;
        logEl.textContent += d.text;
        if (atEnd) logEl.scrollTop = logEl.scrollHeight;
      }
      logOffset = d.offset;
      // Once the job is finished and nothing new arrives, stop polling the log.
      if (d.state === "completed" && !d.text) logDone = true;
    }).catch(function () {});
  }

  // Submit form: show only the input block for the chosen job kind (template / script / session).
  var kind = document.getElementById("kind");
  if (kind) {
    var sync = function () {
      document.getElementById("by-template").style.display = kind.value === "template" ? "" : "none";
      document.getElementById("by-script").style.display = kind.value === "script" ? "" : "none";
      document.getElementById("by-session").style.display = kind.value === "session" ? "" : "none";
    };
    kind.addEventListener("change", sync); sync();
  }

  // ---- interactive terminal (only shown while the job is running)
  var termBox = document.getElementById("term-box");
  // termAfter = index of the last terminal output we received, so each poll only fetches new text.
  var termAfter = 0, termTimer = null;
  // JSON posts need no CSRF form token: the application/json content type already protects them (see auth.check_csrf).
  function csrfJson() { return { "Content-Type": "application/json", "Accept": "application/json" }; }
  // GET /jobs/<id>/term?after=N. 409/400 means the job is not running (or not a session), so hide the terminal.
  function pollTerm() {
    fetch("/jobs/" + jobEl.dataset.jobId + "/term?after=" + termAfter, opts).then(function (r) {
      if (r.status === 409 || r.status === 400) { termBox.hidden = true; return null; }
      return r.json();
    }).then(function (d) {
      if (!d) return;
      termBox.hidden = false;
      if (d.text) { var t = document.getElementById("term"); t.textContent += d.text; t.scrollTop = t.scrollHeight; }
      termAfter = d.after;
    }).catch(function () {});
  }
  // Poll the terminal output every 0.7 s; typed lines are sent with POST /jobs/<id>/term as {data: line + newline}.
  if (termBox) {
    termTimer = setInterval(pollTerm, 700); pollTerm();
    document.getElementById("term-form").addEventListener("submit", function (ev) {
      // Stop the normal page-reloading form submit; we send it with fetch instead.
      ev.preventDefault();
      var inp = document.getElementById("term-input"); var line = inp.value; inp.value = "";
      var t = document.getElementById("term"); t.textContent += "$ " + line + "\n";
      fetch("/jobs/" + jobEl.dataset.jobId + "/term", { method: "POST", headers: csrfJson(), credentials: "same-origin",
        body: JSON.stringify({ data: line + "\n" }) }).catch(function () {});
    });
  }

  // Start the timers: job status every 3 s, log every 1.5 s, pool table every 3 s (only on pages that have those elements).
  if (jobEl) { refreshJob(); refreshLog(); setInterval(refreshJob, 3000); setInterval(refreshLog, 1500); }
  if (poolBody) { setInterval(refreshPool, 3000); }
})();
