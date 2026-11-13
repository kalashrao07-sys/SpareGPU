// Admin pages: GPU detail pop-up (Fleet) and the Approve pop-up (Requests). Text only (textContent), never HTML.
// Admin-only pages. The buttons here only open pop-ups; the real permission check happens on the server (admin_required).
(function () {
  // Wire every element marked data-close inside dialog d so clicking it closes the dialog.
  function closeAll(d) { d.querySelectorAll("[data-close]").forEach(function (b) { b.addEventListener("click", function () { d.close(); }); }); }
  // Fleet page: clicking a GPU chip fills the <dialog> from its data-* attributes (no extra server request) and opens it.
  var gd = document.getElementById("gpu-dialog");
  if (gd) {
    closeAll(gd);
    document.querySelectorAll("button.chip[data-gpu]").forEach(function (c) {
      c.addEventListener("click", function () {
        // d = the chip's data-* attributes. set() writes text into an element by id (textContent = safe).
        var d = c.dataset, set = function (id, t) { document.getElementById(id).textContent = t; };
        set("gd-title", d.gpu); set("gd-sub", "GPU on " + (c.closest(".fleetcard").querySelector("b").textContent));
        set("gd-state", d.state === "busy" ? "Allocated" : d.state === "free" ? "Available" : "Offline");
        set("gd-mem", d.mem + " GB");
        set("gd-job", d.job ? "Job #" + d.job + " (" + d.user + ")" + (d.left !== "" ? ", about " + Math.max(0, d.left) + " s left" : "") : "-");
        set("gd-util", d.util !== "" ? d.util + "% busy, " + d.used + " MB used" : "-");
        gd.showModal();
      });
    });
  }
  // Requests page: the Approve pop-up. Its form is pointed at /providers/<id>/approve for the clicked provider.
  var ap = document.getElementById("approve-dialog");
  if (ap) {
    closeAll(ap);
    document.querySelectorAll("[data-review]").forEach(function (b) {
      b.addEventListener("click", function () {
        var d = b.dataset;
        // encodeURIComponent makes sure the id cannot break out of the URL path.
        document.getElementById("approve-form").action = "/providers/" + encodeURIComponent(d.review) + "/approve";
        document.getElementById("ap-name").textContent = d.name;
        document.getElementById("ap-owner").textContent = d.owner;
        document.getElementById("ap-gpus").textContent = d.gpus;
        document.getElementById("ap-sandbox").textContent = d.sandbox;
        ap.showModal();
      });
    });
  }
})();
