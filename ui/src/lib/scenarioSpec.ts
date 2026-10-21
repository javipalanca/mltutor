/**
 * `--scenario` specification: `nombre[@mod…][:ruta.csv]`. The modifiers only exist for the
 * F7 visual review and change how the scenario is shown, not what it computes:
 *
 * - `@oscuro` / `@claro`: forces the theme (otherwise it follows the system);
 * - `@1024x668`: lays the app out in a box of that CSS size (the inner size of a
 *   1024 × 700 window), top-left in the window, so smaller windows can be checked without
 *   resizing the real one; crop the PNG to that box;
 * - `@vista`: scrolls to the main view of the tab (its first chart, diagram, canvas or table) so a
 *   window-sized capture shows it instead of the page header.
 */
import type { ThemeMode } from "./theme.svelte";

export interface ScenarioSpec {
  name: string;
  arg: string | undefined;
  theme: ThemeMode | null;
  viewport: { width: number; height: number } | null;
  scrollToView: boolean;
}

export function parseScenarioSpec(spec: string): ScenarioSpec {
  const colon = spec.indexOf(":");
  const head = colon < 0 ? spec : spec.slice(0, colon);
  const arg = colon < 0 ? undefined : spec.slice(colon + 1);
  const [name = "", ...mods] = head.split("@");
  const result: ScenarioSpec = { name, arg, theme: null, viewport: null, scrollToView: false };
  for (const mod of mods) {
    const size = /^(\d{3,4})x(\d{3,4})$/.exec(mod);
    if (mod === "oscuro") {
      result.theme = "dark";
    } else if (mod === "claro") {
      result.theme = "light";
    } else if (mod === "vista") {
      result.scrollToView = true;
    } else if (size) {
      result.viewport = { width: Number(size[1]), height: Number(size[2]) };
    } else {
      throw new Error(`Modificador de escenario desconocido: @${mod} (oscuro, claro, vista o ANCHOxALTO)`);
    }
  }
  return result;
}

/** Elements that count as "the main view" for `@vista`, in document order. */
export const MAIN_VIEW_SELECTOR = [
  "figure.chart",
  "[data-tree-diagram]",
  "[data-network-map]",
  "[data-network-diagram]",
  "[data-playground]",
  ".table-block",
]
  .map((s) => `[data-content] ${s}`)
  .join(", ");
