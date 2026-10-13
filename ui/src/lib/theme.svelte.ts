import { LIGHT_THEME, themeFromCss, type ChartTheme } from "./charts";

/** Chart colours, refreshed when the system switches between light and dark mode. */
class ThemeState {
  chart: ChartTheme = $state(LIGHT_THEME);

  constructor() {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const refresh = (): void => {
      this.chart = themeFromCss();
    };
    query.addEventListener("change", () => requestAnimationFrame(refresh));
    queueMicrotask(refresh);
  }
}

export const theme = new ThemeState();
