(function () {
  "use strict";
  var doc = document.documentElement;
  doc.classList.remove("no-js");
  doc.classList.add("js");

  var data = {};
  try {
    data = JSON.parse(document.getElementById("ra-data").textContent);
  } catch (e) {}

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  /* ---- copy buttons ---------------------------------------------------- */
  function fallbackCopy(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.top = "-1000px";
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(area);
    return ok;
  }
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).then(function () { return true; }, function () { return fallbackCopy(text); });
    }
    return Promise.resolve(fallbackCopy(text));
  }
  $$("[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy");
      copyText(text).then(function (ok) {
        btn.textContent = ok ? "Copied ✓" : "Select & copy";
        clearTimeout(btn._t);
        btn._t = setTimeout(function () { btn.textContent = "Copy"; }, 1500);
      });
    });
  });

  /* ---- tabs ------------------------------------------------------------- */
  var tabs = $$("[role=tab]");
  var panels = $$("[role=tabpanel]");
  var pos = $("[data-tab-pos]");
  var current = 0;
  function tabIndexFor(name) {
    for (var i = 0; i < tabs.length; i++) if (tabs[i].getAttribute("data-tab") === name) return i;
    return -1;
  }
  function selectTab(index, opts) {
    opts = opts || {};
    var n = tabs.length;
    if (!n) return;
    index = (index + n) % n;
    current = index;
    tabs.forEach(function (tab, i) {
      var on = i === index;
      tab.setAttribute("aria-selected", on ? "true" : "false");
      tab.tabIndex = on ? 0 : -1;
      if (panels[i]) panels[i].hidden = !on;
    });
    if (pos) pos.textContent = (index + 1) + " / " + n;
    if (opts.focus) tabs[index].focus();
    if (!opts.noHash) {
      var hash = "#" + tabs[index].getAttribute("data-tab");
      try { history.replaceState(null, "", hash); } catch (e) { location.hash = hash; }
    }
  }
  tabs.forEach(function (tab, i) {
    tab.addEventListener("click", function () { selectTab(i); });
  });
  var tablist = $("[role=tablist]");
  if (tablist) {
    tablist.addEventListener("keydown", function (ev) {
      var key = ev.key;
      if (key === "ArrowRight") { selectTab(current + 1, { focus: true }); ev.preventDefault(); }
      else if (key === "ArrowLeft") { selectTab(current - 1, { focus: true }); ev.preventDefault(); }
      else if (key === "Home") { selectTab(0, { focus: true }); ev.preventDefault(); }
      else if (key === "End") { selectTab(tabs.length - 1, { focus: true }); ev.preventDefault(); }
    });
  }
  var prevBtn = $("[data-tab-prev]"), nextBtn = $("[data-tab-next]");
  if (prevBtn) prevBtn.addEventListener("click", function () { selectTab(current - 1); });
  if (nextBtn) nextBtn.addEventListener("click", function () { selectTab(current + 1); });
  $$("[data-goto]").forEach(function (el) {
    el.addEventListener("click", function () {
      var i = tabIndexFor(el.getAttribute("data-goto"));
      if (i >= 0) { selectTab(i); window.scrollTo(0, 0); }
    });
  });
  function fromHash() {
    var i = tabIndexFor((location.hash || "").replace("#", ""));
    selectTab(i >= 0 ? i : 0, { noHash: i < 0 });
  }
  fromHash();
  window.addEventListener("hashchange", fromHash);

  /* ---- slides ----------------------------------------------------------- */
  var openSlides = null;
  function Slides(root) {
    var slides = $$("[data-slide]", root);
    var dots = $$("[data-slide-dot]", root);
    var label = $("[data-slide-label]", root);
    var fsBtn = $("[data-slide-fs]", root);
    var fsText = $("[data-slide-fs-text]", root);
    var i = 0, fs = false, overlay = false;
    function show(k) {
      var n = slides.length;
      i = (k + n) % n;
      slides.forEach(function (s, j) { s.hidden = j !== i; });
      dots.forEach(function (d, j) {
        if (j === i) d.setAttribute("aria-current", "true"); else d.removeAttribute("aria-current");
      });
      if (label) label.textContent = (i + 1) + " / " + n;
    }
    function setFs(on) {
      fs = on;
      root.classList.toggle("is-fs", on);
      fsText.textContent = on ? "Close" : "Full screen";
      fsBtn.setAttribute("aria-label", on ? "Close full screen" : "Open pictures full screen");
      if (on) { openSlides = api; document.addEventListener("keydown", onKey, true); }
      else { if (openSlides === api) openSlides = null; document.removeEventListener("keydown", onKey, true); }
    }
    function openFs() {
      var req = root.requestFullscreen || root.webkitRequestFullscreen;
      var useOverlay = function () { overlay = true; root.classList.add("fs"); setFs(true); fsBtn.focus(); };
      if (!req) return useOverlay();
      try {
        var p = req.call(root);
        if (p && p.then) {
          p.then(function () { setFs(true); }, useOverlay);
        } else {
          setFs(true);
        }
      } catch (e) { useOverlay(); }
    }
    function closeFs() {
      if (overlay) { overlay = false; root.classList.remove("fs"); setFs(false); return; }
      var fsEl = document.fullscreenElement || document.webkitFullscreenElement;
      if (fsEl === root) {
        var exit = document.exitFullscreen || document.webkitExitFullscreen;
        try { exit.call(document); } catch (e) {}
      }
      setFs(false);
    }
    function onKey(ev) {
      if (!fs) return;
      if (ev.key === "ArrowRight") { show(i + 1); ev.preventDefault(); ev.stopPropagation(); }
      else if (ev.key === "ArrowLeft") { show(i - 1); ev.preventDefault(); ev.stopPropagation(); }
      else if (ev.key === "Escape") { closeFs(); ev.preventDefault(); ev.stopPropagation(); }
    }
    document.addEventListener("fullscreenchange", function () {
      if (!document.fullscreenElement && fs && !overlay) setFs(false);
    });
    $("[data-slide-prev]", root).addEventListener("click", function () { show(i - 1); });
    $("[data-slide-next]", root).addEventListener("click", function () { show(i + 1); });
    dots.forEach(function (d, j) { d.addEventListener("click", function () { show(j); }); });
    fsBtn.addEventListener("click", function () { if (fs) closeFs(); else openFs(); });
    var api = { show: show, close: closeFs, isOpen: function () { return fs; } };
    show(0);
    return api;
  }
  $$("[data-slides]").forEach(function (root) { Slides(root); });

  /* ---- welcome popup ---------------------------------------------------- */
  var dialog = $("[data-welcome]");
  var lastFocus = null;
  var SEEN = "ragaudit.welcome.seen";
  function onDialogKey(ev) {
    if (ev.key === "Escape" && !(openSlides && openSlides.isOpen())) { closeWelcome(); ev.preventDefault(); }
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
    if (openSlides && openSlides.isOpen()) openSlides.close();
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
    var seen = null;
    try { seen = localStorage.getItem(SEEN); } catch (e) { seen = null; }
    if (!seen) openWelcome();
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
