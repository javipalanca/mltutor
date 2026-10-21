/**
 * macOS WKWebView only moves Tab through text fields and pop-up menus unless the user turns
 * on "Keyboard navigation" in System Settings (WebKit `tabsToAllFormControls`): buttons,
 * checkboxes, sliders and links are skipped. An explicit `tabindex` makes WebKit treat
 * them as keyboard-focusable regardless of that setting, so on macOS every native control
 * without one gets `tabindex="0"` (elements that set their own, such as the inactive
 * tabs with -1, are left alone). WebView2 and WebKitGTK already tab to every control.
 */
export const TABBABLE_SELECTOR = [
  "button:not([tabindex])",
  "a[href]:not([tabindex])",
  'input[type="checkbox"]:not([tabindex])',
  'input[type="radio"]:not([tabindex])',
  'input[type="range"]:not([tabindex])',
  "select:not([tabindex])",
  "summary:not([tabindex])",
].join(", ");

export function makeTabbable(root: ParentNode): void {
  for (const element of root.querySelectorAll<HTMLElement>(TABBABLE_SELECTOR)) {
    element.tabIndex = 0;
  }
}

export function isMacWebKit(nav: Pick<Navigator, "userAgent"> = navigator): boolean {
  return /Macintosh/.test(nav.userAgent) && /AppleWebKit/.test(nav.userAgent) && !/Chrome|Edg\//.test(nav.userAgent);
}

/** Keeps every control tabbable as Svelte adds and removes them. Returns a disposer. */
export function installMacTabbing(root: HTMLElement = document.body): () => void {
  if (typeof MutationObserver === "undefined" || !isMacWebKit()) {
    return () => {};
  }
  makeTabbable(root);
  let queued = false;
  const observer = new MutationObserver((records) => {
    // Plotly redraws its SVG on every hover: those mutations never add controls.
    const relevant = records.some(
      (record) =>
        record.addedNodes.length > 0 &&
        !(record.target instanceof Element && record.target.closest(".js-plotly-plot, svg, canvas")),
    );
    if (relevant && !queued) {
      queued = true;
      queueMicrotask(() => {
        queued = false;
        makeTabbable(root);
      });
    }
  });
  observer.observe(root, { childList: true, subtree: true });
  return () => observer.disconnect();
}
