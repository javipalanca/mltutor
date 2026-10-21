// Animated counters of "Hecho en Rust" (and any [data-count]). The numbers live in the HTML,
// taken from README.md and benchmarks.md; this only animates them.
import { reducedMotion } from "./motion.js";

function formatNumber(value, decimals) {
  return new Intl.NumberFormat("es-ES", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

function factorText(ratio, kind) {
  const digits = ratio >= 10 ? 0 : 1;
  const n = formatNumber(ratio, digits);
  return kind === "size" ? `${n}× más pequeña` : `${n}× más rápido`;
}

function prepareBars(group) {
  const kind = group.dataset.bars;
  for (const row of group.querySelectorAll(".bar-row")) {
    const python = Number(row.dataset.python);
    const rust = Number(row.dataset.rust);
    const max = Number(row.dataset.max) || python;
    const py = row.querySelector(".bar-py .bar-fill");
    const rs = row.querySelector(".bar-rs .bar-fill");
    const rsPct = (100 * rust) / max;
    py.style.setProperty("--w", `${(100 * python) / max}%`);
    rs.style.setProperty("--w", `${rsPct}%`);
    const factor = row.querySelector("[data-factor]");
    if (factor && python > 0 && rust > 0) factor.textContent = factorText(python / rust, kind);
  }
}

function animateCounter(el) {
  const end = Number(el.dataset.count);
  const decimals = Number(el.dataset.decimals || 0);
  if (reducedMotion()) {
    el.textContent = formatNumber(end, decimals);
    return;
  }
  const duration = 1400;
  const start = performance.now();
  const tick = (now) => {
    const p = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = formatNumber(end * eased, decimals);
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

export function initStats(root = document) {
  const groups = [...root.querySelectorAll("[data-bars]")];
  groups.forEach(prepareBars);
  const counters = [...root.querySelectorAll("[data-count]")];

  const reveal = (el) => {
    if (el.matches("[data-bars]")) el.classList.add("ready");
    el.querySelectorAll?.("[data-count]").forEach(animateCounter);
    if (el.matches("[data-count]")) animateCounter(el);
  };

  if (!("IntersectionObserver" in window) || reducedMotion()) {
    groups.forEach((g) => g.classList.add("ready"));
    counters.forEach((c) => (c.textContent = formatNumber(Number(c.dataset.count), Number(c.dataset.decimals || 0))));
    return;
  }

  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        io.unobserve(entry.target);
        reveal(entry.target);
      }
    },
    { threshold: 0.25 },
  );
  groups.forEach((g) => io.observe(g));
  // Counters outside a bar group (the start-up time).
  counters.filter((c) => !c.closest("[data-bars]")).forEach((c) => io.observe(c));
}
