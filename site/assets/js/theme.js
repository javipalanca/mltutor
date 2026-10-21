// Light/dark theme: follows prefers-color-scheme until the visitor picks one with the toggle.
const KEY = "mltutor-theme";
const root = document.documentElement;
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

function stored() {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function currentTheme() {
  return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

function apply(theme, button) {
  root.setAttribute("data-theme", theme);
  if (button) {
    const next = theme === "dark" ? "claro" : "oscuro";
    button.setAttribute("aria-label", `Cambiar a tema ${next}`);
    button.title = `Cambiar a tema ${next}`;
  }
  document.dispatchEvent(new CustomEvent("themechange", { detail: { theme } }));
}

export function initTheme(button) {
  apply(currentTheme(), button);
  button?.addEventListener("click", () => {
    const theme = currentTheme() === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      /* storage blocked: the choice lasts until the page is reloaded */
    }
    apply(theme, button);
  });
  systemDark.addEventListener("change", (event) => {
    if (!stored()) apply(event.matches ? "dark" : "light", button);
  });
}
