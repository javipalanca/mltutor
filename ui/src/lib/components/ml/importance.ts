/**
 * C10 (fase3-arboles.md §C10): bar chart and table of the feature importance, native or by
 * permutation. Shared by 🌲 Árboles and 🏘️ KNN (F4, permutation only). Pure functions.
 *
 * Python: horizontal bars in ascending order, `viridis(linspace(0, 1, n))` by position,
 * alpha 0.8, the value with 3 decimals to the right of each bar, `xerr = std` (capsize 3)
 * for the permutation, x axis "Importancia" and the title
 * "Importancia de Características\n({método})".
 */
import type { Data } from "plotly.js";
import type { FeatureImportanceView } from "../../bindings/FeatureImportanceView";
import { axis, baseLayout, LIGHT_THEME, withAlpha, type ChartTheme, type Figure } from "../../charts";
import { compText } from "../../content";
import { escapePlotly } from "../../escape";
import { fmtText } from "../../fmt";
import { fixed } from "../../format";
import { VIRIDIS } from "./figures";

function hex(color: string): [number, number, number] {
  const n = parseInt(color.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** `viridis(t)` interpolated between the samples of `VIRIDIS`. */
export function viridisAt(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  let i = 0;
  while (i < VIRIDIS.length - 2 && (VIRIDIS[i + 1]?.[0] ?? 1) < x) {
    i += 1;
  }
  const [t0, c0] = VIRIDIS[i] ?? [0, "#440154"];
  const [t1, c1] = VIRIDIS[i + 1] ?? [1, "#fde725"];
  const f = t1 > t0 ? (x - t0) / (t1 - t0) : 0;
  const a = hex(c0);
  const b = hex(c1);
  const mix = a.map((v, k) => Math.round(v + ((b[k] ?? v) - v) * f));
  return `#${mix.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

/** `viridis(linspace(0, 1, n))`. */
export function viridisColors(n: number): string[] {
  return Array.from({ length: n }, (_, i) => viridisAt(n > 1 ? i / (n - 1) : 0));
}

export function methodName(view: FeatureImportanceView): string {
  return compText("importancia", view.method === "native" ? "metodo_nativo" : "metodo_perm");
}

/** Height of the chart: Python's `max(8, 0.4·n)` inches. */
export function importanceHeight(n: number): number {
  return Math.max(380, Math.min(1600, 90 + n * 34));
}

export function importanceFigure(view: FeatureImportanceView, theme: ChartTheme = LIGHT_THEME): Figure {
  const rows = view.rows;
  const colors = viridisColors(rows.length);
  const values = rows.map((r) => r.importance);
  const errors = rows.map((r) => r.std ?? 0);
  const hi = Math.max(0, ...values.map((v, i) => v + (errors[i] ?? 0)));
  const lo = Math.min(0, ...values.map((v, i) => v - (errors[i] ?? 0)));
  const span = hi - lo || 1;
  const title = fmtText(compText("importancia", "grafico_title"), { metodo: methodName(view) })
    .split("\n")
    .map(escapePlotly)
    .join("<br>");
  const layout = baseLayout(theme, title);
  layout.margin = { l: 30, r: 40, t: 80, b: 60 };
  layout.xaxis = { ...axis(theme, escapePlotly(compText("importancia", "grafico_xlabel"))), range: [lo - span * 0.02, hi + span * 0.14], zeroline: true };
  // Positions instead of categories: two features may share a friendly name.
  layout.yaxis = {
    ...axis(theme, ""),
    tickmode: "array",
    tickvals: rows.map((_, i) => i),
    ticktext: rows.map((r) => escapePlotly(r.name)),
    showgrid: false,
    automargin: true,
    range: [-0.6, rows.length - 0.4],
  };
  layout.bargap = 0.2;
  const permutation = view.method === "permutation";
  const trace = {
    type: "bar",
    orientation: "h",
    x: values,
    y: rows.map((_, i) => i),
    customdata: rows.map((r) => escapePlotly(r.name)),
    marker: { color: colors.map((c) => withAlpha(c, 0.8)), line: { color: colors, width: 1 } },
    text: values.map((v) => fixed(v, 3)),
    textposition: "outside",
    cliponaxis: false,
    textfont: { size: 12, color: theme.text },
    hovertemplate: permutation ? "%{customdata}: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>" : "%{customdata}: %{x:.4f}<extra></extra>",
    ...(permutation ? { error_x: { type: "data", array: errors, visible: true, color: theme.text, thickness: 1.2, width: 3 } } : {}),
  } as Data;
  return { data: [trace], layout };
}

/** Rows of "### Valores Detallados": descending, with the original column as index. */
export function importanceTable(view: FeatureImportanceView): { index: string[]; rows: [string, string, string?][] } {
  const ordered = [...view.rows].sort((a, b) => b.importance - a.importance || a.feature_index - b.feature_index);
  return {
    index: ordered.map((r) => String(r.feature_index)),
    rows: ordered.map((r) => {
      const row: [string, string, string?] = [r.name, fixed(r.importance, 4)];
      if (view.method === "permutation") {
        row.push(fixed(r.std ?? 0, 4));
      }
      return row;
    }),
  };
}
