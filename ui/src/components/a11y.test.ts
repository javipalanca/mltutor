/**
 * Accessibility checks (F7): ARIA roles and relations, keyboard handling and the WCAG AA
 * contrast of the design tokens in both themes. Focused tests instead of axe-core, which
 * would be a heavy dependency for what is checked here.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { tick } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import { isMacWebKit, makeTabbable } from "../lib/keyboard";
import { parseScenarioSpec } from "../lib/scenarioSpec";
import DataTable from "./DataTable.svelte";
import LessonTabs from "./LessonTabs.svelte";
import MultiSelect from "./MultiSelect.svelte";
import NumberInput from "./NumberInput.svelte";
import Select from "./Select.svelte";
import Slider from "./Slider.svelte";
import StatusBar from "./StatusBar.svelte";
import ExpanderHarness from "./test/ExpanderHarness.svelte";
import TabsHarness from "./test/TabsHarness.svelte";

afterEach(() => cleanup());

/** Every id referenced by `attr` on `root` exists (IDREF lists split on spaces). */
function referencesResolve(root: ParentNode, attr: string): void {
  for (const element of root.querySelectorAll(`[${attr}]`)) {
    for (const id of (element.getAttribute(attr) ?? "").split(/\s+/).filter(Boolean)) {
      expect(document.getElementById(id), `${attr}="${id}"`).not.toBeNull();
    }
  }
}

describe("LessonTabs (WAI-ARIA tabs)", () => {
  const tabs = ["📊 Datos", "🏋️ Entrenamiento", "📈 Evaluación", "📉 Visualización", "🔍 Coeficientes", "🔮 Predicciones", "💾 Exportar"];

  it("links each tab to the panel and back", async () => {
    const { container } = render(TabsHarness, { tabs });
    const tab = screen.getByRole("tab", { name: "📊 Datos" });
    expect(tab.getAttribute("aria-controls")).toBe("t-panel");
    const panel = screen.getByRole("tabpanel");
    expect(panel.getAttribute("aria-labelledby")).toBe(tab.id);
    referencesResolve(container, "aria-controls");
    referencesResolve(container, "aria-labelledby");
    expect(screen.getByRole("tablist").getAttribute("aria-label")).toBeTruthy();
  });

  it("moves selection and focus with the arrows, Home and End (roving tabindex)", async () => {
    render(TabsHarness, { tabs });
    const list = screen.getByRole("tablist");
    screen.getByRole("tab", { name: "📊 Datos" }).focus();
    await fireEvent.keyDown(list, { key: "ArrowRight" });
    await tick();
    const second = screen.getByRole("tab", { name: "🏋️ Entrenamiento" });
    expect(second.getAttribute("aria-selected")).toBe("true");
    expect(document.activeElement).toBe(second);
    expect(screen.getAllByRole("tab").filter((t) => t.tabIndex === 0)).toEqual([second]);
    await fireEvent.keyDown(list, { key: "End" });
    await tick();
    expect(document.activeElement).toBe(screen.getByRole("tab", { name: "💾 Exportar" }));
    await fireEvent.keyDown(list, { key: "ArrowRight" });
    await tick();
    expect(document.activeElement).toBe(screen.getByRole("tab", { name: "📊 Datos" }));
    await fireEvent.keyDown(list, { key: "Home" });
    expect(screen.getByRole("tab", { name: "📊 Datos" }).getAttribute("aria-selected")).toBe("true");
  });

  it("ignores other keys", async () => {
    const onchange = vi.fn();
    render(LessonTabs, { tabs, active: 2, onchange, idPrefix: "x" });
    await fireEvent.keyDown(screen.getByRole("tablist"), { key: "a" });
    expect(onchange).not.toHaveBeenCalled();
  });
});

describe("Expander", () => {
  it("exposes aria-expanded/aria-controls and closes with Escape", async () => {
    const { container } = render(ExpanderHarness, { id: "a11y-expander", label: "💡 Ayuda" });
    const header = screen.getByRole("button", { name: /Ayuda/ });
    expect(header.getAttribute("aria-expanded")).toBe("false");
    await fireEvent.click(header);
    expect(header.getAttribute("aria-expanded")).toBe("true");
    referencesResolve(container, "aria-controls");
    const body = screen.getByText("cuerpo");
    await fireEvent.keyDown(body, { key: "Escape" });
    expect(screen.queryByText("cuerpo")).toBeNull();
    expect(header.getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(header);
  });
});

describe("Form controls have accessible names", () => {
  it("Slider, Select, NumberInput and MultiSelect", () => {
    const { container } = render(Slider, { id: "k", label: "Valor K:", min: 1, max: 10, value: 5, onchange: vi.fn(), hint: "vecinos" });
    expect(screen.getByRole("slider", { name: "Valor K:" })).toBeTruthy();
    expect(screen.getByRole("spinbutton", { name: /Valor K:/ })).toBeTruthy();
    referencesResolve(container, "aria-describedby");
    cleanup();

    render(Select, { id: "m", label: "Métrica:", options: [{ label: "Euclídea", value: "e" }], value: "e", onchange: vi.fn() });
    expect(screen.getByRole("combobox", { name: "Métrica:" })).toBeTruthy();
    cleanup();

    render(NumberInput, { id: "n", label: "Épocas:", min: 1, max: 10, value: 3, onchange: vi.fn() });
    expect(screen.getByRole("spinbutton", { name: "Épocas:" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Aumentar Épocas:" })).toBeTruthy();
    cleanup();

    // The label has spaces and emojis: the group must still be labelled.
    const multi = render(MultiSelect, { label: "🎯 Selecciona características:", options: ["a", "b"], value: ["a"], onchange: vi.fn() });
    expect(screen.getByRole("group", { name: "🎯 Selecciona características:" })).toBeTruthy();
    referencesResolve(multi.container, "aria-labelledby");
    referencesResolve(multi.container, "aria-describedby");
  });
});

describe("DataTable", () => {
  it("sortable headers are buttons (keyboard) and report aria-sort", async () => {
    const rows = [[{ kind: "number", v: 2 }], [{ kind: "number", v: 1 }]] as const;
    const { container } = render(DataTable, { columns: ["x"], rows: rows.map((r) => [...r]), sortable: true, label: "Datos" });
    expect(screen.getByRole("region", { name: "Datos" }).tabIndex).toBe(0);
    const button = screen.getByRole("button", { name: "x" });
    await fireEvent.click(button);
    expect(container.querySelector("th")?.getAttribute("aria-sort")).toBe("ascending");
  });
});

describe("StatusBar", () => {
  it("announces the status text in a polite live region", () => {
    render(StatusBar);
    const live = screen.getByRole("status");
    expect(live.getAttribute("aria-live")).toBe("polite");
    expect(live.querySelector("button")).toBeNull();
  });
});

describe("macOS tabbing", () => {
  it("adds tabindex=0 to native controls without one", () => {
    document.body.innerHTML = '<button id="a">a</button><button id="b" tabindex="-1">b</button><a id="c" href="#x">c</a><input id="d" type="checkbox" /><input id="e" type="text" />';
    makeTabbable(document.body);
    expect(document.getElementById("a")?.getAttribute("tabindex")).toBe("0");
    expect(document.getElementById("b")?.getAttribute("tabindex")).toBe("-1");
    expect(document.getElementById("c")?.getAttribute("tabindex")).toBe("0");
    expect(document.getElementById("d")?.getAttribute("tabindex")).toBe("0");
    expect(document.getElementById("e")?.hasAttribute("tabindex")).toBe(false);
    document.body.innerHTML = "";
  });

  it("only applies to WKWebView/Safari", () => {
    const wk = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)";
    expect(isMacWebKit({ userAgent: wk })).toBe(true);
    expect(isMacWebKit({ userAgent: `${wk} Chrome/120.0 Safari/537.36` })).toBe(false);
    expect(isMacWebKit({ userAgent: "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 Edg/120" })).toBe(false);
  });
});

describe("Scenario modifiers", () => {
  it("parses theme, viewport, scroll and the CSV path", () => {
    expect(parseScenarioSpec("knn_frontera_iris@vista@1024x668@oscuro")).toEqual({
      name: "knn_frontera_iris",
      arg: undefined,
      theme: "dark",
      viewport: { width: 1024, height: 668 },
      scrollToView: true,
    });
    expect(parseScenarioSpec("csv_cargado_clasificacion@claro:/tmp/a:b.csv")).toMatchObject({
      name: "csv_cargado_clasificacion",
      arg: "/tmp/a:b.csv",
      theme: "light",
    });
    expect(() => parseScenarioSpec("inicio@grande")).toThrow(/desconocido/);
  });
});

// ---------------------------------------------------------------------------------------
// WCAG AA contrast of the CSS tokens (theme.css), light and dark.

function tokens(css: string, selector: string): Record<string, string> {
  const start = css.indexOf(selector);
  const body = css.slice(css.indexOf("{", start) + 1, css.indexOf("\n}", start));
  return Object.fromEntries([...body.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})\b/g)].map((m) => [m[1], m[2]]));
}

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (hi + 0.05) / (lo + 0.05);
}

describe("Contrast of the design tokens (WCAG 2.1 AA)", () => {
  const css = readFileSync(resolve(process.cwd(), "src/styles/theme.css"), "utf8");
  const light = tokens(css, ":root {");
  const dark = { ...light, ...tokens(css, ':root[data-theme="dark"] {') };

  // [foreground, background, minimum]: 4.5 for text, 3 for large text and UI components.
  const TEXT: [string, string][] = [
    ["color-text", "color-bg"],
    ["color-text", "color-sidebar-bg"],
    ["color-text", "color-input-bg"],
    ["color-text", "color-table-alt"],
    ["color-text", "color-info-bg"],
    ["color-text", "color-warning-bg"],
    ["color-text", "color-error-bg"],
    ["color-text", "color-success-bg"],
    ["color-text", "color-tooltip-bg"],
    ["color-text", "color-code-inline-bg"],
    ["color-text-muted", "color-bg"],
    ["color-text-muted", "color-sidebar-bg"],
    ["color-text-muted", "color-input-bg"],
    ["color-text-muted", "color-metric-bg"],
    ["color-text-muted", "color-info-bg"],
    ["color-text-muted", "color-table-index-bg"],
    ["color-footer", "color-bg"],
    ["color-link", "color-bg"],
    ["color-heading", "color-bg"],
    ["color-heading", "color-sidebar-bg"],
    ["color-on-primary", "color-primary"],
    ["color-on-primary", "color-primary-hover"],
    ["color-chip-text", "color-chip-bg"],
    ["color-chip-on-text", "color-chip-on-bg"],
    ["color-table-head-text", "color-table-head-bg"],
    ["color-table-selected-text", "color-table-selected"],
    ["color-metric-value", "color-metric-bg"],
    ["color-code-text", "color-code-bg"],
    ["color-code-label", "color-code-bg"],
    ["color-statusbar-text", "color-statusbar-bg"],
    ["chart-text", "chart-bg"],
  ];
  const NON_TEXT: [string, string][] = [
    ["color-title", "color-bg"], // 32 px bold: large text
    ["color-focus", "color-bg"],
    ["color-focus", "color-sidebar-bg"],
    ["color-input-border", "color-bg"],
    ["color-input-border", "color-input-bg"],
    ["color-input-focus-border", "color-bg"],
    ["color-slider-handle", "color-bg"],
    ["color-primary", "color-bg"],
    ["chart-axis", "chart-bg"],
  ];

  for (const [name, theme] of [
    ["claro", light],
    ["oscuro", dark],
  ] as const) {
    it(`text ≥ 4.5:1 and controls ≥ 3:1 in the ${name} theme`, () => {
      const failures: string[] = [];
      for (const [pairs, min] of [
        [TEXT, 4.5],
        [NON_TEXT, 3],
      ] as const) {
        for (const [fg, bg] of pairs) {
          const a = theme[fg];
          const b = theme[bg];
          expect(a, fg).toBeDefined();
          expect(b, bg).toBeDefined();
          const ratio = contrast(a ?? "#000000", b ?? "#000000");
          if (ratio < min) {
            failures.push(`${fg} ${a} / ${bg} ${b}: ${ratio.toFixed(2)} < ${min}`);
          }
        }
      }
      expect(failures).toEqual([]);
    });
  }
});
