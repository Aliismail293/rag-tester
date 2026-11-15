/* Website wiring only: the shared widgets come from ragaudit.js (the report's design system). */
(function () {
  "use strict";
  var UI = window.RagauditUI;
  if (!UI) return;
  var viewer = document.querySelector("[data-viewer]");
  if (viewer) UI.Tabs(viewer);
  var snips = document.querySelector("[data-snips]");
  if (snips) UI.Tabs(snips);
})();
