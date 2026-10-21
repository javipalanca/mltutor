// WAI-ARIA tabs with roving tabindex (← → Inicio Fin).
export function initTabs(container) {
  const tabs = [...container.querySelectorAll('[role="tab"]')];
  const panels = tabs.map((tab) => document.getElementById(tab.getAttribute("aria-controls")));

  function select(index, focus) {
    tabs.forEach((tab, i) => {
      const selected = i === index;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (panels[i]) panels[i].hidden = !selected;
    });
    if (focus) tabs[index].focus();
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(i, false));
    tab.addEventListener("keydown", (event) => {
      const last = tabs.length - 1;
      let next = null;
      if (event.key === "ArrowRight") next = i === last ? 0 : i + 1;
      else if (event.key === "ArrowLeft") next = i === 0 ? last : i - 1;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = last;
      if (next !== null) {
        event.preventDefault();
        select(next, true);
      }
    });
  });

  return {
    /** Selects the tab whose `data-os` matches, without moving the focus. */
    selectOs(os) {
      const index = tabs.findIndex((tab) => tab.dataset.os === os);
      if (index >= 0) select(index, false);
    },
  };
}
