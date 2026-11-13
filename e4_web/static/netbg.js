// Soft animated "network" background for the login and home pages (light red nodes joined by faint lines).
// Pauses when the tab is hidden; shows one still frame if the visitor prefers reduced motion.
// Purely decorative; it has no connection to the server. Exits quietly if the page has no <canvas id=netbg>.
(function () {
  var cv = document.getElementById("netbg"); if (!cv) return;
  // Colour depends on the portal: teal for the exchange (body class 'exchange'), red for the college portal.
  var rgb = document.body.classList.contains("exchange") ? "15,118,110" : "214,72,72";
  // dpr = device pixel ratio (capped at 2) so the canvas is sharp on high-DPI screens without being slow.
  var ctx = cv.getContext("2d"), w = 0, h = 0, dpr = Math.min(window.devicePixelRatio || 1, 2), nodes = [], raf = null;
  var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Resize the canvas and create a number of nodes proportional to screen area (between 24 and 70).
  function size() {
    w = window.innerWidth; h = window.innerHeight;
    cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var n = Math.max(24, Math.min(70, Math.round(w * h / 17000)));
    nodes = [];
    for (var i = 0; i < n; i++) nodes.push({ x: Math.random() * w, y: Math.random() * h, vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35, r: 1.6 + Math.random() * 2 });
  }
  // Draw one frame: move each node (bounce at edges), join nodes closer than 150 px with a line that fades with distance, draw the dots.
  function frame() {
    ctx.clearRect(0, 0, w, h);
    for (var i = 0; i < nodes.length; i++) {
      var a = nodes[i];
      if (!still) { a.x += a.vx; a.y += a.vy; if (a.x < 0 || a.x > w) a.vx *= -1; if (a.y < 0 || a.y > h) a.vy *= -1; }
      for (var j = i + 1; j < nodes.length; j++) {
        var b = nodes[j], dx = a.x - b.x, dy = a.y - b.y, d = Math.sqrt(dx * dx + dy * dy);
        if (d < 150) { ctx.strokeStyle = "rgba(" + rgb + "," + (0.16 * (1 - d / 150)).toFixed(3) + ")"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
      }
      ctx.fillStyle = "rgba(" + rgb + ",.38)"; ctx.beginPath(); ctx.arc(a.x, a.y, a.r, 0, 6.2832); ctx.fill();
    }
    // Keep animating unless the user prefers reduced motion.
    if (!still) raf = requestAnimationFrame(frame);
  }
  size(); frame();
  window.addEventListener("resize", function () { size(); if (still) frame(); });
  // Stop animating while the tab is hidden to save battery/CPU.
  document.addEventListener("visibilitychange", function () {
    if (still) return;
    if (document.hidden) { cancelAnimationFrame(raf); } else { raf = requestAnimationFrame(frame); }
  });
})();
