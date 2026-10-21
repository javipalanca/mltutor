// Runs before the first paint (classic, blocking script) to avoid a flash of the wrong theme.
(function () {
  var root = document.documentElement;
  root.classList.add("js");
  var stored = null;
  try { stored = localStorage.getItem("mltutor-theme"); } catch (e) { /* storage blocked */ }
  var dark = stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.setAttribute("data-theme", dark ? "dark" : "light");
})();
