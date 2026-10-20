import { LIGHT_THEME, themeFromCss, type ChartTheme } from "./charts";

export type ThemeMode = "light" | "dark";

/**
 * Light/dark mode. `theme.css` keys the dark tokens on `<html data-theme="dark">` (not on
 * the `prefers-color-scheme` media query) so that captures can force a theme
 * (`--scenario nombre@oscuro`) and a future toggle needs no CSS changes. Without a forced
 * mode the attribute follows the system setting live.
 */
class ThemeState {
  chart: ChartTheme = $state(LIGHT_THEME);
  mode: ThemeMode = $state("light");
  /** `devicePixelRatio`, updated when the window moves between a 1× and a 2× screen, so
   *  canvases (NetworkMap) redraw their bitmap at the new density instead of blurring. */
  dpr: number = $state(1);
  #forced: ThemeMode | null = null;
  #query: MediaQueryList | null = null;

  constructor() {
    if (typeof window === "undefined" || typeof document === "undefined") {
      return;
    }
    if (typeof window.matchMedia === "function") {
      this.#query = window.matchMedia("(prefers-color-scheme: dark)");
      this.#query.addEventListener("change", () => this.#apply());
    }
    this.#apply();
    this.#watchPixelRatio();
  }

  #watchPixelRatio(): void {
    this.dpr = window.devicePixelRatio || 1;
    if (typeof window.matchMedia !== "function") {
      return;
    }
    // A `resolution` query only fires when crossing its value: re-arm it on every change.
    const query = window.matchMedia(`(resolution: ${this.dpr}dppx)`);
    query.addEventListener("change", () => this.#watchPixelRatio(), { once: true });
  }

  /** Forces a theme (`null` = follow the system again). */
  force(mode: ThemeMode | null): void {
    this.#forced = mode;
    this.#apply();
  }

  #apply(): void {
    const mode = this.#forced ?? (this.#query?.matches ? "dark" : "light");
    this.mode = mode;
    document.documentElement.dataset["theme"] = mode;
    // The CSS variables change synchronously with the attribute.
    this.chart = themeFromCss();
  }
}

export const theme = new ThemeState();
