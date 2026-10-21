// Screenshot gallery: thumbnails follow the page theme when a light/dark pair exists,
// and a <dialog> lightbox shows them large, with a light/dark switch and ← → navigation.
import { currentTheme } from "./theme.js";

const BASE = "assets/img/capturas/";

function variantsOf(button) {
  return { light: button.dataset.light || null, dark: button.dataset.dark || null };
}

function preferred(button, theme) {
  const v = variantsOf(button);
  return theme === "dark" ? v.dark || v.light : v.light || v.dark;
}

export function initGallery({ list, dialog }) {
  const buttons = [...list.querySelectorAll(".shot")];
  const img = dialog.querySelector("#lightbox-img");
  const caption = dialog.querySelector("#lightbox-caption");
  const count = dialog.querySelector("#lightbox-count");
  const variantButtons = [...dialog.querySelectorAll("[data-variant]")];
  const prev = dialog.querySelector("#lightbox-prev");
  const next = dialog.querySelector("#lightbox-next");
  const close = dialog.querySelector("#lightbox-close");
  let index = 0;
  let variant = "light";
  let opener = null;

  function syncThumbs() {
    const theme = currentTheme();
    for (const button of buttons) {
      const v = variantsOf(button);
      if (!v.light || !v.dark) continue;
      const name = preferred(button, theme);
      const thumb = button.querySelector("img");
      const src = `${BASE}${name}-640.webp`;
      if (!thumb.getAttribute("src").endsWith(src)) thumb.src = src;
    }
  }

  function show() {
    const button = buttons[index];
    const v = variantsOf(button);
    if (!v[variant]) variant = v.light ? "light" : "dark";
    const name = v[variant];
    const thumb = button.querySelector("img");
    img.src = `${BASE}${name}.webp`;
    img.removeAttribute("width");
    img.removeAttribute("height");
    img.alt = `${thumb.alt}${variant === "dark" ? " (tema oscuro)" : " (tema claro)"}`;
    caption.textContent = button.dataset.caption || "";
    count.textContent = `Captura ${index + 1} de ${buttons.length}`;
    for (const vb of variantButtons) {
      const key = vb.dataset.variant;
      vb.disabled = !v[key];
      vb.setAttribute("aria-pressed", String(key === variant));
      vb.title = v[key] ? "" : "Esta captura solo está en el otro tema";
    }
  }

  function open(i, from) {
    index = i;
    const theme = currentTheme();
    variant = theme;
    opener = from;
    show();
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    close.focus();
  }

  function move(delta) {
    index = (index + delta + buttons.length) % buttons.length;
    show();
  }

  buttons.forEach((button, i) => button.addEventListener("click", () => open(i, button)));
  prev.addEventListener("click", () => move(-1));
  next.addEventListener("click", () => move(1));
  close.addEventListener("click", () => dialog.close());
  variantButtons.forEach((vb) =>
    vb.addEventListener("click", () => {
      variant = vb.dataset.variant;
      show();
    }),
  );
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      move(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      move(1);
    }
  });
  // Click on the backdrop (outside the inner box) closes.
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener("close", () => {
    opener?.focus();
    opener = null;
  });

  document.addEventListener("themechange", syncThumbs);
  syncThumbs();
}
