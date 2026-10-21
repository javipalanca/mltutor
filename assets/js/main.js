// Entry point (ES module, no build step).
import { initTheme } from "./theme.js";
import { initKnnDemo } from "./knn-demo.js";
import { initStats } from "./stats.js";
import { initGallery } from "./gallery.js";
import { initTabs } from "./tabs.js";
import { initDownloads } from "./downloads.js";
import { reducedMotion } from "./motion.js";

const $ = (selector) => document.querySelector(selector);

function safely(name, fn) {
  try {
    fn();
  } catch (error) {
    console.warn(`MLTutor: no se ha podido iniciar ${name}`, error);
  }
}

safely("el tema", () => initTheme($("#theme-toggle")));

safely("la demo KNN", () =>
  initKnnDemo({
    canvas: $("#knn-canvas"),
    kInput: $("#knn-k"),
    kOutput: $("#knn-k-out"),
    classInputs: [...document.querySelectorAll('input[name="knn-class"]')],
    resetButton: $("#knn-reset"),
    readout: $("#knn-readout"),
  }),
);

safely("las cifras", () => initStats());
safely("la galería", () => initGallery({ list: $("#gallery"), dialog: $("#lightbox") }));

let tabs = null;
safely("las pestañas", () => {
  tabs = initTabs($("[data-tabs]"));
});

safely("las descargas", () =>
  initDownloads({
    cards: [...document.querySelectorAll(".dl[data-os]")],
    status: $("#release-status"),
    heroButton: $("#hero-download"),
    heroLabel: $("#hero-download-label"),
    osNote: $("#os-note"),
    tabs,
  }),
);

// Gentle reveal on scroll (only elements marked here, so nothing hides if this fails).
safely("las animaciones de aparición", () => {
  if (reducedMotion() || !("IntersectionObserver" in window)) return;
  const targets = document.querySelectorAll(".section-head, .lesson, .shot, .panel, .feature, .dl, .firstrun, .reqs");
  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          io.unobserve(entry.target);
        }
      }
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
  );
  targets.forEach((target) => {
    const rect = target.getBoundingClientRect();
    if (rect.top < window.innerHeight) return; // already on screen: no flash
    target.classList.add("reveal");
    io.observe(target);
  });
});
