/* report-only behaviour; the shared design system (design.js) runs before this and provides window.RagauditUI. */
(function () {
  "use strict";
  var UI = window.RagauditUI;
  var data = {};
  try {
    data = JSON.parse(document.getElementById("ra-data").textContent);
  } catch (e) {}

  var $ = UI.$, $$ = UI.$$;

  /* ---- report tabs (the shared pill-tab widget, mirrored in the URL hash) */
  var reportTabs = UI.Tabs($(".report"), { hash: true });
  function selectTab(index) { reportTabs.select(index); }
  function tabIndexFor(name) { return reportTabs.indexOf(name); }
  $$("[data-goto]").forEach(function (el) {
    el.addEventListener("click", function () {
      var i = tabIndexFor(el.getAttribute("data-goto"));
      if (i >= 0) { selectTab(i); window.scrollTo(0, 0); }
    });
  });

  /* ---- welcome popup ---------------------------------------------------- */
  var dialog = $("[data-welcome]");
  var lastFocus = null;
  var SEEN = "ragaudit.welcome.seen";
  function onDialogKey(ev) {
    if (ev.key === "Escape" && !(UI.openSlides && UI.openSlides.isOpen())) { closeWelcome(); ev.preventDefault(); }
    if (ev.key === "Tab") {
      var focusable = $$("button, a[href], [tabindex='0']", dialog).filter(function (el) { return el.offsetParent !== null; });
      if (!focusable.length) return;
      var first = focusable[0], last = focusable[focusable.length - 1];
      if (ev.shiftKey && document.activeElement === first) { last.focus(); ev.preventDefault(); }
      else if (!ev.shiftKey && document.activeElement === last) { first.focus(); ev.preventDefault(); }
    }
  }
  function openWelcome() {
    if (!dialog) return;
    lastFocus = document.activeElement;
    dialog.hidden = false;
    document.addEventListener("keydown", onDialogKey);
    var close = $("[data-welcome-close]", dialog);
    if (close) close.focus();
    try { localStorage.setItem(SEEN, "1"); } catch (e) {}
  }
  function closeWelcome() {
    if (!dialog || dialog.hidden) return;
    if (UI.openSlides && UI.openSlides.isOpen()) UI.openSlides.close();
    dialog.hidden = true;
    document.removeEventListener("keydown", onDialogKey);
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  if (dialog) {
    $$("[data-welcome-close]", dialog).forEach(function (b) { b.addEventListener("click", closeWelcome); });
    dialog.addEventListener("click", function (ev) { if (ev.target === dialog) closeWelcome(); });
    $$("[data-welcome-run]", dialog).forEach(function (b) {
      b.addEventListener("click", function () { closeWelcome(); selectTab(tabIndexFor("run")); window.scrollTo(0, 0); });
    });
    $$("[data-welcome-open]").forEach(function (b) { b.addEventListener("click", openWelcome); });
    var seen = null, embedded = false;
    try { seen = localStorage.getItem(SEEN); } catch (e) { seen = null; }
    try { embedded = window.self !== window.top; } catch (e) { embedded = true; }
    // Shown inside another page (the website's sample viewer): don't cover it; "What is this?" still opens it.
    if (!seen && !embedded) openWelcome();
  }

  /* ---- score history ---------------------------------------------------- */
  var runs = data.history || [];
  var chart = $("[data-chart]");
  if (chart && runs.length) {
    var band = $("[data-sel-band]", chart);
    var selLabel = $("[data-sel-label]", chart);
    var points = $$("[data-point]", chart);
    var buttons = $$("[data-run]");
    var detail = $("[data-run-detail]");
    function fill(name, value) { var el = $("[data-d=" + name + "]", detail); if (el) el.textContent = value; }
    function pick(k) {
      var r = runs[k];
      if (!r) return;
      band.setAttribute("x1", r.x); band.setAttribute("x2", r.x);
      selLabel.setAttribute("x", r.x); selLabel.setAttribute("y", r.label_y); selLabel.textContent = r.tp_text;
      points.forEach(function (p, j) { p.setAttribute("r", j === k ? 9 : 6); });
      buttons.forEach(function (b, j) { b.setAttribute("aria-pressed", j === k ? "true" : "false"); });
      fill("meta", r.meta); fill("note", r.note); fill("tp", r.tp_text); fill("hit", r.hit_text); fill("delta", r.delta_text);
      var d = $("[data-d=delta]", detail);
      d.className = "num" + (r.delta_class ? " " + r.delta_class : "");
      var warn = $("[data-d=warn]", detail);
      warn.textContent = r.warn || "";
      warn.hidden = !r.warn;
    }
    buttons.forEach(function (b, j) { b.addEventListener("click", function () { pick(j); }); });
  }

  /* ---- what failed ------------------------------------------------------ */
  var failButtons = $$("[data-fail]");
  var failDetails = $$("[data-fail-detail]");
  function pickFail(k) {
    failButtons.forEach(function (b, j) { b.setAttribute("aria-pressed", j === k ? "true" : "false"); });
    failDetails.forEach(function (d, j) { d.hidden = j !== k; });
  }
  failButtons.forEach(function (b, j) { b.addEventListener("click", function () { pickFail(j); }); });
  if (failDetails.length > 1) pickFail(0);

  /* ---- all-questions filters -------------------------------------------- */
  var filters = $$("[data-filter]");
  var caseRows = $$("[data-case]");
  filters.forEach(function (f) {
    f.addEventListener("click", function () {
      var want = f.getAttribute("data-filter");
      filters.forEach(function (o) { o.setAttribute("aria-pressed", o === f ? "true" : "false"); });
      caseRows.forEach(function (row) {
        row.hidden = !(want === "all" || row.getAttribute("data-case") === want);
      });
    });
  });
})();
