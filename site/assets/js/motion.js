// Live `prefers-reduced-motion` state shared by the animated parts of the page.
const query = window.matchMedia("(prefers-reduced-motion: reduce)");
const listeners = new Set();

query.addEventListener("change", () => {
  for (const fn of listeners) fn(query.matches);
});

/** True when the visitor asked the system to reduce motion. */
export function reducedMotion() {
  return query.matches;
}

/** Calls `fn(reduced)` whenever the preference changes. */
export function onReducedMotionChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
