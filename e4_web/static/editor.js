// VS Code-style workspace: explorer tree, tabs, Monaco editor, Output panel, GPU terminal, Run/Stop.
// Everything talks to the server over JSON (the JSON content type also protects these calls from CSRF):
//   /api/files            list/read (GET), save (PUT), delete file or folder (DELETE)
//   /api/files/rename     rename a file or folder (POST)
//   /api/gpu-status       is a GPU online/free/accepting user code? (GET)  -> used so Run never pretends
//   /jobs                 submit a job (POST), /jobs/<id> status, /jobs/<id>/log output, /jobs/<id>/stop cancel
//   /jobs/<id>/term       interactive terminal of a running session job (GET output, POST input)
// The code itself is executed on the allocated GPU worker (in a sandbox), never in this browser or on this server.
(function () {
  var $ = function (id) { return document.getElementById(id); };

  // ---------------------------------------------------------------- state
  var files = [];                 // [{path, size}] as last listed by the server
  var tabs = [];                  // open file paths, in tab order
  var buf = {};                   // path -> {text, saved}: current text and the text last saved (dirty = they differ)
  var models = {};                // path -> Monaco text model (only when Monaco is in use)
  var active = null;              // path of the visible tab
  var open = {};                  // folder path -> true when expanded in the tree
  var selFolder = "";             // folder where "+ File" creates (the folder row last clicked)
  var monacoEditor = null, useMonaco = false;
  var run = { id: null, off: 0, timer: null, lastState: null };   // the script job started by Run
  var term = { id: null, after: 0, timer: null, ready: false };    // the interactive GPU session
  var gpu = { online: 0, free: 0, max: 0, scriptsOk: false };      // last /api/gpu-status

  // ---------------------------------------------------------------- small helpers
  function api(method, url, body) {
    var o = { method: method, credentials: "same-origin", headers: { "Accept": "application/json" } };
    if (body !== undefined) { o.headers["Content-Type"] = "application/json"; o.body = JSON.stringify(body); }
    return fetch(url, o).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (d) { return { ok: r.ok, status: r.status, data: d }; });
    });
  }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text) e.textContent = text; return e; }
  // Pick syntax highlighting from the file extension (Monaco has many more; these are the common ones for GPU work).
  function langOf(p) {
    var m = /\.([A-Za-z0-9]+)$/.exec(p); var ext = m ? m[1].toLowerCase() : "";
    return ({ py: "python", json: "json", md: "markdown", js: "javascript", sh: "shell", yaml: "yaml", yml: "yaml", txt: "plaintext",
              cpp: "cpp", c: "c", cu: "cpp", h: "cpp", html: "html", css: "css", csv: "plaintext", ipynb: "json" })[ext] || "plaintext";
  }
  function isDirty(p) { return buf[p] && buf[p].text !== buf[p].saved; }
  function anyDirty() { return tabs.some(isDirty); }
  function getText() { return useMonaco ? monacoEditor.getValue() : $("fallback").value; }
  // Output panel: append text, optionally styled as an error; keep scrolled to the bottom.
  function out(text, isErr) {
    var box = $("out");
    if (isErr) { var s = el("span", "err", text); box.appendChild(s); } else { box.appendChild(document.createTextNode(text)); }
    box.scrollTop = box.scrollHeight;
  }
  function setStatus(text, cls) { var s = $("job-status"); s.textContent = text || ""; s.className = "job-status " + (cls || ""); }
  function showPanel(which) {
    $("out").hidden = which !== "out"; $("term-wrap").hidden = which !== "term";
    $("tab-out").className = which === "out" ? "on" : ""; $("tab-term").className = which === "term" ? "on" : "";
  }

  // ---------------------------------------------------------------- explorer tree
  function buildTree() {
    // Turn flat paths ("a/b/c.py") into nested folders. ".keep" files only mark empty folders and are never shown.
    var root = { dirs: {}, files: [] };
    files.forEach(function (f) {
      var parts = f.path.split("/"), node = root;
      for (var i = 0; i < parts.length - 1; i++) { node = node.dirs[parts[i]] = node.dirs[parts[i]] || { dirs: {}, files: [] }; }
      var leaf = parts[parts.length - 1];
      if (leaf !== ".keep") node.files.push({ name: leaf, path: f.path });
    });
    return root;
  }
  function renderTree() {
    var host = $("tree"); host.textContent = "";
    (function walk(node, prefix, depth, parent) {
      Object.keys(node.dirs).sort().forEach(function (name) {
        var path = prefix + name; if (open[path] === undefined) open[path] = true;
        var row = el("div", "row"); row.style.paddingLeft = (0.4 + depth * 0.9) + "rem"; row.setAttribute("role", "treeitem");
        row.appendChild(el("span", "twisty", open[path] ? "▾" : "▸"));
        row.appendChild(el("span", "name", "📁 " + name));
        row.appendChild(actions(path, true));
        row.addEventListener("click", function () { open[path] = !open[path]; selFolder = path; renderTree(); });
        host.appendChild(row);
        if (open[path]) walk(node.dirs[name], path + "/", depth + 1);
      });
      node.files.sort(function (a, b) { return a.name < b.name ? -1 : 1; }).forEach(function (f) {
        var row = el("div", "row" + (f.path === active ? " active" : "")); row.style.paddingLeft = (1.4 + depth * 0.9) + "rem";
        row.setAttribute("role", "treeitem");
        row.appendChild(el("span", "name", f.name));
        row.appendChild(actions(f.path, false));
        row.addEventListener("click", function () { selFolder = prefix.replace(/\/$/, ""); openFile(f.path); });
        host.appendChild(row);
      });
    })(buildTree(), "", 0);
    if (!files.length) host.appendChild(el("div", "row", "No files yet. Click + File."));
  }
  // Hover buttons on every row: rename and delete.
  function actions(path, isFolder) {
    var box = el("span", "acts");
    var r = el("button", "", "✎"); r.title = "Rename";
    var d = el("button", "", "🗑"); d.title = "Delete";
    r.addEventListener("click", function (e) { e.stopPropagation(); renameItem(path, isFolder); });
    d.addEventListener("click", function (e) { e.stopPropagation(); deleteItem(path, isFolder); });
    box.appendChild(r); box.appendChild(d); return box;
  }
  function loadList() {
    return api("GET", "/api/files").then(function (x) { files = (x.data.files || []); renderTree(); });
  }

  // ---------------------------------------------------------------- tabs
  function renderTabs() {
    var bar = $("tabs"); bar.textContent = "";
    tabs.forEach(function (p) {
      var t = el("div", "tab" + (p === active ? " active" : "")); t.setAttribute("role", "tab");
      t.appendChild(el("span", "", p.split("/").pop())); t.title = p;
      if (isDirty(p)) t.appendChild(el("span", "dot", "●"));
      var x = el("button", "x", "×"); x.title = "Close";
      x.addEventListener("click", function (e) { e.stopPropagation(); closeTab(p); });
      t.addEventListener("click", function () { activate(p); });
      t.addEventListener("auxclick", function (e) { if (e.button === 1) closeTab(p); });   // middle click closes
      t.appendChild(x); bar.appendChild(t);
    });
    $("empty").hidden = tabs.length > 0;
  }
  // Show one tab: in Monaco switch the model; in the plain fallback save/restore the textarea text.
  function activate(p) {
    if (active && !useMonaco && buf[active]) buf[active].text = $("fallback").value;
    active = p;
    if (p === null) { $("empty").hidden = false; renderTabs(); renderTree(); return; }
    if (useMonaco) { monacoEditor.setModel(models[p]); monacoEditor.focus(); } else { $("fallback").value = buf[p].text; }
    renderTabs(); renderTree();
  }
  function makeModel(p, text) {
    if (!useMonaco) return;
    models[p] = monaco.editor.createModel(text, langOf(p));
    models[p].onDidChangeContent(function () { buf[p].text = models[p].getValue(); renderTabs(); });
  }
  function openFile(p) {
    if (buf[p]) { activate(p); return; }
    api("GET", "/api/files?path=" + encodeURIComponent(p)).then(function (x) {
      if (!x.ok) { out((x.data.error || "cannot open " + p) + "\n", true); return; }
      buf[p] = { text: x.data.content, saved: x.data.content };
      tabs.push(p); makeModel(p, x.data.content); activate(p);
    });
  }
  function dropBuffer(p) {
    tabs = tabs.filter(function (t) { return t !== p; });
    if (models[p]) { models[p].dispose(); delete models[p]; }
    delete buf[p];
  }
  function closeTab(p) {
    if (isDirty(p) && !confirm("Close " + p + " without saving?")) return;
    var i = tabs.indexOf(p); dropBuffer(p);
    if (active === p) { active = null; activate(tabs.length ? tabs[Math.min(i, tabs.length - 1)] : null); } else renderTabs();
  }

  // ---------------------------------------------------------------- file operations
  function saveFile(p) {
    if (!buf[p]) return Promise.resolve(true);
    if (p === active && !useMonaco) buf[p].text = $("fallback").value;
    var text = buf[p].text;
    return api("PUT", "/api/files", { path: p, content: text }).then(function (x) {
      if (x.ok) { buf[p].saved = text; renderTabs(); return true; }
      out("Save failed for " + p + ": " + (x.data.error || x.status) + "\n", true); return false;
    });
  }
  function saveAll() {
    return Promise.all(tabs.filter(isDirty).map(saveFile)).then(function (r) { return r.every(Boolean); });
  }
  function newFile() {
    var name = prompt("New file path (for example src/train.py):", selFolder ? selFolder + "/" : "");
    if (!name) return;
    name = name.trim().replace(/\\/g, "/");
    api("PUT", "/api/files", { path: name, content: "" }).then(function (x) {
      if (!x.ok) { alert(x.data.error || "could not create the file"); return; }
      loadList().then(function () { openFile(name); });
    });
  }
  function newFolder() {
    var name = prompt("New folder name:", selFolder ? selFolder + "/" : "");
    if (!name) return;
    name = name.trim().replace(/\\/g, "/").replace(/\/$/, "");
    // Folders exist only as path prefixes, so a hidden ".keep" file makes an empty folder appear.
    api("PUT", "/api/files", { path: name + "/.keep", content: "" }).then(function (x) {
      if (!x.ok) { alert(x.data.error || "could not create the folder"); return; }
      open[name] = true; selFolder = name; loadList();
    });
  }
  function renameItem(path, isFolder) {
    var to = prompt("Rename to:", path);
    if (!to || to === path) return;
    to = to.trim().replace(/\\/g, "/");
    saveAll().then(function () {
      return api("POST", "/api/files/rename", { path: path, new_path: to });
    }).then(function (x) {
      if (!x || !x.ok) { alert((x && x.data.error) || "rename failed"); return; }
      // Re-point open tabs: a tab for the renamed file (or a file inside the renamed folder) now has a new path.
      tabs.slice().forEach(function (t) {
        if (t === path || t.indexOf(path + "/") === 0) {
          var nt = to + t.slice(path.length), text = buf[t].text, wasActive = active === t;
          dropBuffer(t); buf[nt] = { text: text, saved: text }; tabs.push(nt); makeModel(nt, text);
          if (wasActive) active = nt;
        }
      });
      loadList().then(function () { if (active) activate(active); else renderTabs(); });
    });
  }
  function deleteItem(path, isFolder) {
    if (!confirm("Delete " + (isFolder ? "the folder " + path + " and everything in it" : path) + "? This cannot be undone.")) return;
    api("DELETE", "/api/files", isFolder ? { path: path, folder: true } : { path: path }).then(function (x) {
      if (!x.ok) { alert(x.data.error || "delete failed"); return; }
      tabs.slice().forEach(function (t) { if (t === path || t.indexOf(path + "/") === 0) { if (active === t) active = null; dropBuffer(t); } });
      loadList().then(function () { if (!active) activate(tabs.length ? tabs[0] : null); else renderTabs(); });
    });
  }

  // ---------------------------------------------------------------- GPU availability (honest status)
  function refreshGpu() {
    api("GET", "/api/gpu-status").then(function (x) {
      if (!x.ok) return;
      gpu = { online: x.data.online, free: x.data.free, max: x.data.max_memory_gb, scriptsOk: x.data.scripts_ok };
      var c = $("gpu-chip");
      if (!gpu.online) { c.textContent = "No GPU online"; c.className = "chip bad"; }
      else if (!gpu.scriptsOk) { c.textContent = "GPU online, but not accepting user code"; c.className = "chip bad"; }
      else if (!gpu.free) { c.textContent = gpu.online + " GPU online, busy (jobs will wait)"; c.className = "chip busy"; }
      else { c.textContent = gpu.free + " GPU free · up to " + gpu.max + " GB"; c.className = "chip ok"; }
    });
  }
  // Returns an error message if nothing can run the job right now, or null if it is fine to submit.
  function cannotRun(memGb) {
    if (!gpu.online) return "No GPU is online right now, so nothing was started. Ask the admin to bring a GPU worker online.";
    if (!gpu.scriptsOk) return "A GPU is online, but no worker accepts user code (its sandbox is not enabled), so nothing was started.";
    if (memGb > gpu.max) return "You asked for " + memGb + " GB but the largest GPU online has " + gpu.max + " GB. Lower the memory and try again.";
    return null;
  }

  // ---------------------------------------------------------------- Run / Stop (script job)
  function stopWatching() { clearInterval(run.timer); run.timer = null; $("run-btn").disabled = false; $("stop-btn").disabled = true; }
  var BAD_NOTE = /fail|error|killed|stopped|time limit|timed out|cancel|exit code [1-9]|traceback/i;

  function runFile() {
    if (run.timer) return;                                         // one run at a time
    if (!active) { showPanel("out"); out("Open the file you want to run first.\n", true); return; }
    var mem = parseFloat($("mem").value), mins = parseInt($("mins").value, 10);
    showPanel("out"); $("out").textContent = "";
    if (!(mem > 0) || !(mins >= 1)) { out("Enter a GPU memory and a time limit.\n", true); return; }
    var path = active;
    saveAll().then(function (ok) {
      if (!ok) { out("Fix the save error before running.\n", true); return; }
      return api("GET", "/api/gpu-status").then(function (g) {
        if (g.ok) { gpu = { online: g.data.online, free: g.data.free, max: g.data.max_memory_gb, scriptsOk: g.data.scripts_ok }; }
        var why = cannotRun(mem);
        if (why) { out(why + "\n", true); setStatus("Not started: no GPU", "bad"); return; }    // never pretend a run happened
        return api("POST", "/jobs", { kind: "script", script_path: path, memory_gb: mem, duration_min: mins, priority: 3,
                                       project_name: path.split("/")[0] }).then(function (x) {
          if (!x.ok) { out((x.data.error || "the server refused the job") + "\n", true); setStatus("Not started", "bad"); return; }
          run = { id: x.data.job_id, off: 0, timer: null, lastState: null };
          out("Job #" + run.id + " submitted: " + path + "\n");
          $("run-btn").disabled = true; $("stop-btn").disabled = false;
          run.timer = setInterval(pollRun, 1500); pollRun();
        });
      });
    });
  }
  // Poll the job's status and its log. The status text always reflects the TRUE state from the server.
  function pollRun() {
    var id = run.id;
    api("GET", "/jobs/" + id).then(function (s) {
      if (!s.ok) { setStatus("Lost contact with the server, retrying…", "wait"); return; }
      var j = s.data;
      return api("GET", "/jobs/" + id + "/log?offset=" + run.off).then(function (l) {
        if (l.ok && l.data.text) { out(l.data.text); run.off = l.data.offset; }
        if (j.state === "queued" || j.state === "held") {
          setStatus("Waiting for a GPU" + (j.held_reason ? ": " + j.held_reason : "") + (j.queue_position ? " (position " + j.queue_position + ")" : ""), "wait");
          if (run.lastState !== "wait") { out("Waiting for a GPU. Your code has NOT started yet.\n"); run.lastState = "wait"; }
        } else if (j.state === "running") {
          setStatus("Running on " + (j.gpu_name || "GPU") + (j.provider_name ? " (" + j.provider_name + ")" : "") + (j.remaining_s != null ? " · " + j.remaining_s + " s left" : ""), "ok");
          if (run.lastState !== "run") { out("Started on " + (j.gpu_name || "the GPU") + ".\n"); run.lastState = "run"; }
        } else if (j.state === "completed") {
          var bad = BAD_NOTE.test(j.result_note || "");
          out("\n" + (bad ? "Finished with a problem: " : "Finished: ") + (j.result_note || "done") + "\n", bad);
          setStatus(bad ? "Failed / stopped: " + j.result_note : "Finished: " + (j.result_note || "done"), bad ? "bad" : "ok");
          stopWatching();
        }
      });
    });
  }
  function stopRun() {
    if (!run.id) return;
    api("POST", "/jobs/" + run.id + "/stop", {}).then(function (x) {
      if (x.ok) { out("Stop requested. The GPU worker ends the job within a few seconds.\n"); setStatus("Stopping…", "wait"); }
      else { out((x.data.error || "could not stop") + "\n", true); }
    });
  }

  // ---------------------------------------------------------------- GPU terminal (interactive session job)
  function startTerminal() {
    if (term.id) { endTerminal(); return; }
    showPanel("term"); $("term").textContent = "";
    var mem = parseFloat($("mem").value) || 1, mins = parseInt($("mins").value, 10) || 5;
    var why = cannotRun(mem);
    if (why) { $("term").textContent = why + "\n"; return; }
    api("POST", "/jobs", { kind: "session", memory_gb: mem, duration_min: mins, priority: 3, project_name: "terminal" }).then(function (x) {
      if (!x.ok) { $("term").textContent = (x.data.error || "the server refused the session") + "\n"; return; }
      term = { id: x.data.job_id, after: 0, timer: setInterval(pollTerm, 800), ready: false };
      $("term").textContent = "Requesting a GPU session #" + term.id + "… waiting for allocation.\n";
      $("term-start").textContent = "End terminal session";
    });
  }
  function endTerminal() {
    if (term.id) api("POST", "/jobs/" + term.id + "/stop", {});
    clearInterval(term.timer); $("term-input").disabled = true; $("term-start").textContent = "Start a GPU terminal (uses a GPU session)";
    $("term").textContent += "\nSession ended.\n"; term = { id: null, after: 0, timer: null, ready: false };
  }
  function pollTerm() {
    if (!term.id) return;
    api("GET", "/jobs/" + term.id + "/term?after=" + term.after).then(function (x) {
      var t = $("term");
      if (x.ok) {
        if (!term.ready) { term.ready = true; $("term-input").disabled = false; $("term-input").focus(); t.textContent += "Connected to the GPU sandbox.\n"; }
        if (x.data.text) { t.textContent += x.data.text; t.scrollTop = t.scrollHeight; }
        term.after = x.data.after;
      } else if (x.status === 409) {
        // Not running yet (still queued) or already finished: ask the job itself so the message is truthful.
        api("GET", "/jobs/" + term.id).then(function (s) {
          if (s.ok && s.data.state === "completed") { endTerminal(); $("term").textContent += (s.data.result_note || "") + "\n"; }
        });
      }
    });
  }
  $("term-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var line = $("term-input").value; $("term-input").value = ""; if (!term.id) return;
    $("term").textContent += "$ " + line + "\n";
    api("POST", "/jobs/" + term.id + "/term", { data: line + "\n" }).then(function (x) {
      if (!x.ok) { $("term").textContent += (x.data.error || "could not send") + "\n"; }
    });
  });

  // ---------------------------------------------------------------- wiring
  $("run-btn").addEventListener("click", runFile);
  $("stop-btn").addEventListener("click", stopRun);
  $("new-file").addEventListener("click", newFile);
  $("new-folder").addEventListener("click", newFolder);
  $("tab-out").addEventListener("click", function () { showPanel("out"); });
  $("tab-term").addEventListener("click", function () { showPanel("term"); });
  $("term-start").addEventListener("click", startTerminal);
  $("clear-out").addEventListener("click", function () { $("out").textContent = ""; });
  $("fallback").addEventListener("input", function () { if (active) { buf[active].text = $("fallback").value; renderTabs(); } });
  document.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); if (active) saveFile(active); }
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); runFile(); }
  });
  window.addEventListener("beforeunload", function (e) { if (anyDirty()) { e.preventDefault(); e.returnValue = ""; } });

  // Offline mode: plain textarea instead of Monaco (no syntax colours, everything else works).
  function useFallback() { useMonaco = false; $("monaco").hidden = true; $("fallback").hidden = false; boot(); }
  function boot() { showPanel("out"); loadList(); renderTabs(); refreshGpu(); setInterval(refreshGpu, 5000); }

  // Load Monaco from a CDN; if it cannot be reached within 4 s, use the plain textarea.
  var failTimer = setTimeout(useFallback, 4000);
  var s = document.createElement("script");
  s.src = "https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs/loader.js";
  s.onerror = function () { clearTimeout(failTimer); useFallback(); };
  s.onload = function () {
    require.config({ paths: { vs: "https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs" } });
    require(["vs/editor/editor.main"], function () {
      clearTimeout(failTimer);
      monacoEditor = monaco.editor.create($("monaco"), { model: null, automaticLayout: true, minimap: { enabled: false }, fontSize: 14,
        theme: "vs-dark", tabSize: 4, insertSpaces: true, lineNumbers: "on", renderWhitespace: "none", scrollBeyondLastLine: false });
      useMonaco = true; boot();
    }, function () { clearTimeout(failTimer); useFallback(); });
  };
  document.head.appendChild(s);
})();
