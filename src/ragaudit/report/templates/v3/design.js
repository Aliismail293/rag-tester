/* ragaudit design system: shared behaviour for copy buttons, pill tabs and the slides widget.
   Canonical source. The report inlines it; the website loads a synced copy
   (docs/assets/ragaudit.js, written by scripts/sync_site.py). No dependencies. */
(function () {
  "use strict";
  var UI = window.RagauditUI = window.RagauditUI || {};
  var doc = document.documentElement;
  doc.classList.remove("no-js");
  doc.classList.add("js");

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  UI.$ = $;
  UI.$$ = $$;

  /* ---- copy buttons: any button whose data-copy attribute holds the text */
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
  UI.copyText = copyText;
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

  /* ---- pill tabs --------------------------------------------------------
     root holds one [role=tablist] of [role=tab] buttons, each with aria-controls
     naming its [role=tabpanel]; optional [data-tab-prev], [data-tab-next],
     [data-tab-pos]. opts.hash: mirror data-tab names in the URL hash. */
  UI.Tabs = function (root, opts) {
    opts = opts || {};
    var tablist = $("[role=tablist]", root);
    var tabs = $$("[role=tab]", tablist);
    var panels = tabs.map(function (t) { return document.getElementById(t.getAttribute("aria-controls")); });
    var pos = $("[data-tab-pos]", root);
    var current = 0;
    function indexOf(name) {
      for (var i = 0; i < tabs.length; i++) if (tabs[i].getAttribute("data-tab") === name) return i;
      return -1;
    }
    function select(index, o) {
      o = o || {};
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
      if (o.focus) tabs[index].focus();
      if (opts.hash && !o.noHash) {
        var hash = "#" + tabs[index].getAttribute("data-tab");
        try { history.replaceState(null, "", hash); } catch (e) { location.hash = hash; }
      }
      if (opts.onSelect) opts.onSelect(index, tabs[index], panels[index]);
    }
    tabs.forEach(function (tab, i) { tab.addEventListener("click", function () { select(i); }); });
    tablist.addEventListener("keydown", function (ev) {
      var key = ev.key;
      if (key === "ArrowRight") { select(current + 1, { focus: true }); ev.preventDefault(); }
      else if (key === "ArrowLeft") { select(current - 1, { focus: true }); ev.preventDefault(); }
      else if (key === "Home") { select(0, { focus: true }); ev.preventDefault(); }
      else if (key === "End") { select(tabs.length - 1, { focus: true }); ev.preventDefault(); }
    });
    var prev = $("[data-tab-prev]", root), next = $("[data-tab-next]", root);
    if (prev) prev.addEventListener("click", function () { select(current - 1); });
    if (next) next.addEventListener("click", function () { select(current + 1); });
    if (opts.hash) {
      var fromHash = function () {
        var i = indexOf((location.hash || "").replace("#", ""));
        select(i >= 0 ? i : 0, { noHash: i < 0 });
      };
      fromHash();
      window.addEventListener("hashchange", fromHash);
    } else {
      var start = 0;
      tabs.forEach(function (t, i) { if (t.getAttribute("aria-selected") === "true") start = i; });
      select(start);
    }
    return { select: select, indexOf: indexOf, current: function () { return current; } };
  };

  /* ---- slides: [data-slides] with [data-slide] figures ------------------- */
  UI.openSlides = null;
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
      if (on) { UI.openSlides = api; document.addEventListener("keydown", onKey, true); }
      else { if (UI.openSlides === api) UI.openSlides = null; document.removeEventListener("keydown", onKey, true); }
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
  UI.Slides = Slides;
  $$("[data-slides]").forEach(function (root) { Slides(root); });
})();
