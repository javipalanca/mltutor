/**
 * Plotly figures of the shared ML components (fase2-regresion.md §3, C1–C8 and the
 * coefficient chart). Pure functions (DTO → figure) so they can be unit-tested without a
 * browser; the components draw them with `PlotlyChart.svelte`.
 *
 * Colours, titles and axes follow the matplotlib figures of the Python version (§8.2),
 * with the fixes of D21–D23. Strings that come from data (class and feature names) go
 * through `escapePlotly`, and hover templates use `%{…}` instead of embedding them.
 */
import type { Data, Layout, Shape } from "plotly.js";
import type { ClassCurves } from "../../bindings/ClassCurves";
import type { ClassHistogram } from "../../bindings/ClassHistogram";
import type { ClassMetrics } from "../../bindings/ClassMetrics";
import type { CoefficientRow } from "../../bindings/CoefficientRow";
import type { DecisionGrid } from "../../bindings/DecisionGrid";
import type { Histogram } from "../../bindings/Histogram";
import type { KdeCurve } from "../../bindings/KdeCurve";
import type { Matrix } from "../../bindings/Matrix";
import type { RocData } from "../../bindings/RocData";
import { axis, baseLayout, LIGHT_THEME, withAlpha, type ChartTheme, type Figure } from "../../charts";
import { compList, compText, regList, regText } from "../../content";
import { escapePlotly } from "../../escape";
import { fmtText } from "../../fmt";
import { fixed } from "../../format";

export type { Figure };

const esc = (values: readonly string[]): string[] => values.map(escapePlotly);
/** Text of a template from content, already interpolated, escaped for Plotly. */
const label = (template: string, values: Record<string, string | number>): string =>
  escapePlotly(fmtText(template, values));

// --- Palettes (fase2-regresion.md §8.2) ---

/** matplotlib `Blues`, sampled. */
export const BLUES: [number, string][] = [
  [0, "#f7fbff"],
  [0.125, "#deebf7"],
  [0.25, "#c6dbef"],
  [0.375, "#9ecae1"],
  [0.5, "#6baed6"],
  [0.625, "#4292c6"],
  [0.75, "#2171b5"],
  [0.875, "#08519c"],
  [1, "#08306b"],
];

/** matplotlib `viridis`, sampled. */
export const VIRIDIS: [number, string][] = [
  [0, "#440154"],
  [0.111, "#482878"],
  [0.222, "#3e4989"],
  [0.333, "#31688e"],
  [0.444, "#26828e"],
  [0.556, "#1f9e89"],
  [0.667, "#35b779"],
  [0.778, "#6ece58"],
  [0.889, "#b5de2b"],
  [1, "#fde725"],
];

/** matplotlib `Set3` (multiclass boundary and probability histograms). */
export const SET3 = [
  "#8dd3c7",
  "#ffffb3",
  "#bebada",
  "#fb8072",
  "#80b1d3",
  "#fdb462",
  "#b3de69",
  "#fccde5",
  "#d9d9d9",
  "#bc80bd",
  "#ccebc5",
  "#ffed6f",
] as const;

/** D23: one colour per class in binary boundaries, from the ends of `RdBu`. */
export const RDBU_BINARY = ["#b2182b", "#2166ac"] as const;

/** CSS colours of `colores_clases` (per-class ROC curves, cyclic). */
const CSS_COLORS: Readonly<Record<string, string>> = {
  aqua: "#00ffff",
  darkorange: "#ff8c00",
  cornflowerblue: "#6495ed",
  red: "#ff0000",
  green: "#008000",
  purple: "#800080",
  brown: "#a52a2a",
  pink: "#ffc0cb",
  gray: "#808080",
  olive: "#808000",
};

export const ROC_COLORS = {
  binary: "#ff8c00",
  pr: "#006400",
  navy: "#000080",
  micro: "#ff1493",
  random: "#000000",
} as const;

export const COEF_COLORS = { positive: "#4ecdc4", negative: "#ff6b6b" } as const;
export const PRECISION_COLOR = "#3274a1";
export const PROB_COLORS = ["#f08080", "#add8e6"] as const;
export const RESIDUAL_COLORS = { points: "#1f77b4", hist: "#87ceeb", box: "#f5deb3", std: "#add8e6" } as const;

export function classCurveColor(index: number): string {
  const names = compList("roc", "colores_clases");
  const name = names[index % Math.max(1, names.length)] ?? "gray";
  return CSS_COLORS[name] ?? name;
}

/** Colour of class `index` of `n` in boundaries and probability histograms. */
export function classColor(index: number, n: number): string {
  if (n === 2) {
    return RDBU_BINARY[index] ?? RDBU_BINARY[0];
  }
  return SET3[index % SET3.length] ?? SET3[0];
}

// --- Helpers ---

function centers(edges: readonly number[]): number[] {
  return edges.slice(0, -1).map((edge, i) => (edge + (edges[i + 1] ?? edge)) / 2);
}

function widths(edges: readonly number[]): number[] {
  return edges.slice(0, -1).map((edge, i) => (edges[i + 1] ?? edge) - edge);
}

function minMax(values: readonly number[]): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  return Number.isFinite(lo) ? [lo, hi] : [0, 1];
}

function line(x0: number, y0: number, x1: number, y1: number, color: string, dash: string, width = 2, extra: Partial<Shape> = {}): Partial<Shape> {
  return { type: "line", x0, y0, x1, y1, line: { color, dash: dash as "dash", width }, ...extra };
}

function insideLegend(x: number, y: number, xanchor: "left" | "right", yanchor: "top" | "bottom", theme: ChartTheme) {
  return {
    x,
    y,
    xanchor,
    yanchor,
    bgcolor: withAlpha(theme.background.startsWith("#") ? theme.background : "#ffffff", 0.85),
    bordercolor: theme.grid,
    borderwidth: 1,
    font: { size: 12 },
  };
}

// --- C1: confusion matrix and precision by class ---

export function confusionFigure(confusion: Matrix, classNames: readonly string[], theme: ChartTheme = LIGHT_THEME): Figure {
  const n = confusion.rows;
  const z: number[][] = [];
  let max = 0;
  for (let r = 0; r < n; r += 1) {
    const row: number[] = [];
    for (let c = 0; c < confusion.cols; c += 1) {
      const v = confusion.values[r * confusion.cols + c] ?? 0;
      row.push(v);
      max = Math.max(max, v);
    }
    z.push(row);
  }
  const names = esc(classNames.length === n ? classNames : z.map((_, i) => String(i)));
  const layout = baseLayout(theme, escapePlotly(compText("clasificacion", "matriz_title")));
  layout.xaxis = { ...axis(theme, escapePlotly(compText("clasificacion", "matriz_xlabel"))), type: "category", showgrid: false, side: "bottom" };
  layout.yaxis = {
    ...axis(theme, escapePlotly(compText("clasificacion", "matriz_ylabel"))),
    type: "category",
    showgrid: false,
    autorange: "reversed",
    scaleanchor: "x",
    constrain: "domain",
  };
  layout.margin = { l: 110, r: 30, t: 60, b: 90 };
  // Dark cells get white text, as seaborn's `annot`.
  const textColors = z.map((row) => row.map((v) => (max > 0 && v / max > 0.55 ? "#ffffff" : "#08306b")));
  return {
    data: [
      {
        type: "heatmap",
        x: names,
        y: names,
        z,
        zmin: 0,
        zmax: Math.max(1, max),
        colorscale: BLUES,
        xgap: 1,
        ygap: 1,
        colorbar: { thickness: 16, outlinewidth: 0, tickfont: { color: theme.text } },
        hovertemplate: "Real %{y} · Predicción %{x}: %{z}<extra></extra>",
      } as Data,
    ],
    layout: {
      ...layout,
      annotations: z.flatMap((row, r) =>
        row.map((v, c) => ({
          x: names[c] ?? String(c),
          y: names[r] ?? String(r),
          text: String(v),
          showarrow: false,
          font: { size: n <= 5 ? 16 : 12, color: textColors[r]?.[c] ?? "#000000" },
        })),
      ),
    },
  };
}

export function precisionByClassFigure(perClass: readonly ClassMetrics[], theme: ChartTheme = LIGHT_THEME): Figure {
  const layout = baseLayout(theme, escapePlotly(compText("clasificacion", "precision_clase_title")));
  layout.xaxis = { ...axis(theme, escapePlotly(compText("clasificacion", "precision_clase_xlabel"))), type: "category" };
  layout.yaxis = { ...axis(theme, escapePlotly(compText("clasificacion", "precision_clase_ylabel"))), range: [0, 1] };
  layout.bargap = 0.2;
  return {
    data: [
      {
        type: "bar",
        x: esc(perClass.map((c) => c.name)),
        y: perClass.map((c) => c.precision),
        marker: { color: PRECISION_COLOR },
        hovertemplate: "%{x}: %{y:.4f}<extra></extra>",
      },
    ],
    layout,
  };
}

// --- C2 / C7a: predictions against real values ---

export interface PredVsRealOptions {
  /** C2: colour by |error| (viridis) and diagonal from min(y) to max(y). */
  colorByError?: boolean;
  /** C7a: "R² = …" box (D22). */
  r2?: number | null;
}

export function predVsRealFigure(
  yTrue: readonly number[],
  yPred: readonly number[],
  options: PredVsRealOptions = {},
  theme: ChartTheme = LIGHT_THEME,
): Figure {
  const errorMode = options.colorByError ?? false;
  const t = errorMode
    ? { title: compText("regresion", "pred_title"), x: compText("regresion", "pred_xlabel"), y: compText("regresion", "pred_ylabel") }
    : { title: compText("residuos", "pred_title"), x: compText("residuos", "pred_xlabel"), y: compText("residuos", "pred_ylabel") };
  const layout = baseLayout(theme, escapePlotly(t.title));
  layout.xaxis = axis(theme, escapePlotly(t.x));
  layout.yaxis = axis(theme, escapePlotly(t.y));
  layout.hovermode = "closest";
  const [lo, hi] = errorMode ? minMax(yTrue) : minMax([...yTrue, ...yPred]);
  const data: Data[] = [];
  if (errorMode) {
    const absError = yTrue.map((y, i) => Math.abs(y - (yPred[i] ?? Number.NaN)));
    data.push({
      type: "scatter",
      mode: "markers",
      x: [...yTrue],
      y: [...yPred],
      customdata: absError,
      marker: {
        color: absError,
        colorscale: VIRIDIS,
        opacity: 0.5,
        size: 8,
        showscale: true,
        colorbar: { title: { text: escapePlotly(compText("regresion", "pred_colorbar")), side: "right" }, thickness: 16, outlinewidth: 0 },
      },
      showlegend: false,
      hovertemplate: "Real %{x:.4g}<br>Predicción %{y:.4g}<br>|error| %{customdata:.4g}<extra></extra>",
    } as Data);
    data.push({
      type: "scatter",
      mode: "lines",
      x: [lo, hi],
      y: [lo, hi],
      line: { color: "#ff0000", dash: "dash", width: 2 },
      showlegend: false,
      hoverinfo: "skip",
    });
  } else {
    data.push({
      type: "scatter",
      mode: "markers",
      x: [...yTrue],
      y: [...yPred],
      marker: { color: withAlpha(RESIDUAL_COLORS.points, 0.6), size: 9, line: { color: "#000000", width: 0.5 } },
      showlegend: false,
      hovertemplate: "Real %{x:.4g}<br>Predicción %{y:.4g}<extra></extra>",
    });
    data.push({
      type: "scatter",
      mode: "lines",
      x: [lo, hi],
      y: [lo, hi],
      name: escapePlotly(compText("residuos", "pred_leyenda")),
      line: { color: "#ff0000", dash: "dash", width: 2 },
      hoverinfo: "skip",
    });
    layout.showlegend = true;
    layout.legend = insideLegend(0.99, 0.02, "right", "bottom", theme);
  }
  if (options.r2 !== undefined && options.r2 !== null) {
    layout.annotations = [
      {
        xref: "paper",
        yref: "paper",
        x: 0.02,
        y: 0.98,
        xanchor: "left",
        yanchor: "top",
        text: label(compText("residuos", "pred_r2"), { r2: options.r2 }),
        showarrow: false,
        bgcolor: RESIDUAL_COLORS.box,
        bordercolor: "#8a6d3b",
        borderwidth: 1,
        borderpad: 5,
        font: { size: 13, color: "#243247" },
      },
    ];
  }
  return { data, layout };
}

/** C2: histogram of `y − ŷ` with its KDE and the line x = 0. */
export function errorHistogramFigure(hist: Histogram, kde: KdeCurve | null, theme: ChartTheme = LIGHT_THEME): Figure {
  const layout = baseLayout(theme, escapePlotly(compText("regresion", "errores_title")));
  layout.xaxis = axis(theme, escapePlotly(compText("regresion", "errores_xlabel")));
  layout.yaxis = { ...axis(theme, "Count"), rangemode: "tozero" };
  layout.bargap = 0;
  const data: Data[] = [
    {
      type: "bar",
      x: centers(hist.edges),
      y: [...hist.counts],
      width: widths(hist.edges),
      marker: { color: withAlpha("#1f77b4", 0.55), line: { color: "#1f77b4", width: 1 } },
      customdata: hist.counts.map((_, i) => [hist.edges[i] ?? 0, hist.edges[i + 1] ?? 0]),
      hovertemplate: "[%{customdata[0]:.4g}, %{customdata[1]:.4g}): %{y}<extra></extra>",
      showlegend: false,
    },
  ];
  if (kde && kde.density.length > 1) {
    const steps = kde.density.length - 1;
    data.push({
      type: "scatter",
      mode: "lines",
      x: kde.density.map((_, i) => kde.grid_min + ((kde.grid_max - kde.grid_min) * i) / steps),
      y: kde.density,
      line: { color: "#1f77b4", width: 2 },
      showlegend: false,
      hoverinfo: "skip",
    });
  }
  layout.shapes = [line(0, 0, 0, 1, "#ff0000", "dash", 2, { yref: "paper" })];
  return { data, layout };
}

// --- C4: ROC and precision-recall ---

export type Average = "macro" | "micro";

export interface RocOptions {
  showPerClass?: boolean;
  /** Multiclass averages in the order chosen; empty = macro (as Python). */
  averages?: readonly Average[];
}

function rocLayout(theme: ChartTheme, rocTitle: string, prTitle: string): Partial<Layout> {
  const layout = baseLayout(theme, "");
  const frame = { showline: true, mirror: true, linecolor: theme.axis, zeroline: false };
  layout.xaxis = { ...axis(theme, escapePlotly(compText("roc", "roc_xlabel"))), ...frame, domain: [0, 0.46], range: [0, 1] };
  layout.yaxis = { ...axis(theme, escapePlotly(compText("roc", "roc_ylabel"))), ...frame, range: [0, 1.05] };
  layout.xaxis2 = { ...axis(theme, escapePlotly(compText("roc", "pr_xlabel"))), ...frame, domain: [0.54, 1], range: [0, 1], anchor: "y2" };
  layout.yaxis2 = { ...axis(theme, escapePlotly(compText("roc", "pr_ylabel"))), ...frame, range: [0, 1.05], anchor: "x2" };
  layout.annotations = [
    { text: `<b>${escapePlotly(rocTitle)}</b>`, xref: "paper", yref: "paper", x: 0.23, y: 1.04, xanchor: "center", yanchor: "bottom", showarrow: false, font: { size: 15 } },
    { text: `<b>${escapePlotly(prTitle)}</b>`, xref: "paper", yref: "paper", x: 0.77, y: 1.04, xanchor: "center", yanchor: "bottom", showarrow: false, font: { size: 15 } },
  ];
  layout.margin = { l: 70, r: 30, t: 60, b: 70 };
  layout.showlegend = true;
  layout.hovermode = "closest";
  // One legend per subplot, inside, as matplotlib (Plotly ≥ 2.23 supports `legend2`).
  layout.legend = insideLegend(0.45, 0.03, "right", "bottom", theme);
  (layout as Record<string, unknown>)["legend2"] = insideLegend(0.555, 0.03, "left", "bottom", theme);
  return layout;
}

function rocTrace(curve: ClassCurves["roc"], name: string, color: string, extra: Record<string, unknown> = {}): Data {
  return {
    type: "scatter",
    mode: "lines",
    x: curve.x,
    y: curve.y,
    name,
    line: { color, width: 2, ...(extra["line"] as object | undefined) },
    hovertemplate: "%{fullData.name}<br>FPR %{x:.3f} · TPR %{y:.3f}<extra></extra>",
    ...Object.fromEntries(Object.entries(extra).filter(([key]) => key !== "line")),
  } as Data;
}

function prTrace(curve: ClassCurves["pr"], name: string, color: string, extra: Record<string, unknown> = {}): Data {
  return {
    type: "scatter",
    mode: "lines",
    x: curve.x,
    y: curve.y,
    xaxis: "x2",
    yaxis: "y2",
    legend: "legend2",
    name,
    // `drawstyle='steps-post'` would be sklearn's display; the Python app joins the points.
    line: { color, width: 2, ...(extra["line"] as object | undefined) },
    hovertemplate: "%{fullData.name}<br>Recall %{x:.3f} · Precision %{y:.3f}<extra></extra>",
    ...Object.fromEntries(Object.entries(extra).filter(([key]) => key !== "line")),
  } as Data;
}

export function rocPrFigure(roc: RocData, options: RocOptions = {}, theme: ChartTheme = LIGHT_THEME): Figure {
  if (roc.kind === "binary") {
    const layout = rocLayout(theme, compText("roc", "roc_title_binaria"), compText("roc", "pr_title_binaria"));
    const data: Data[] = [
      rocTrace(roc.roc, label(compText("roc", "leyenda_roc"), { auc: roc.auc }), ROC_COLORS.binary),
      rocTrace({ x: [0, 1], y: [0, 1], thresholds: null }, escapePlotly(compText("roc", "leyenda_aleatorio")), ROC_COLORS.navy, {
        line: { dash: "dash" },
        hoverinfo: "skip",
      }),
      prTrace(roc.pr, label(compText("roc", "leyenda_pr"), { ap: roc.ap }), ROC_COLORS.pr),
      prTrace({ x: [0, 1], y: [roc.baseline, roc.baseline], thresholds: null }, label(compText("roc", "leyenda_baseline"), { baseline: roc.baseline }), ROC_COLORS.navy, {
        line: { dash: "dash" },
        hoverinfo: "skip",
      }),
    ];
    return { data, layout };
  }
  const showPerClass = options.showPerClass ?? true;
  const averages: readonly Average[] = options.averages && options.averages.length > 0 ? options.averages : ["macro"];
  const layout = rocLayout(theme, compText("roc", "roc_title_multiclase"), compText("roc", "pr_title_multiclase"));
  const data: Data[] = [];
  if (showPerClass) {
    roc.per_class.forEach((curves, i) => {
      const color = classCurveColor(i);
      const name = escapePlotly(curves.name);
      data.push(rocTrace(curves.roc, label(compText("roc", "leyenda_clase_roc"), { clase: curves.name, auc: curves.auc }), color, { opacity: 0.7, legendgroup: name }));
      data.push(prTrace(curves.pr, label(compText("roc", "leyenda_clase_pr"), { clase: curves.name, ap: curves.ap }), color, { opacity: 0.7, legendgroup: name }));
    });
  }
  // Python draws macro first, then micro, whatever the order of the multiselect.
  if (averages.includes("macro")) {
    data.push(rocTrace(roc.macro_roc, label(compText("roc", "leyenda_macro_roc"), { auc: roc.macro_auc }), ROC_COLORS.navy, { line: { dash: "dash", width: 4 } }));
    data.push(prTrace({ x: [0, 1], y: [roc.macro_ap, roc.macro_ap], thresholds: null }, label(compText("roc", "leyenda_macro_pr"), { ap: roc.macro_ap }), ROC_COLORS.navy, {
      line: { dash: "dash", width: 4 },
    }));
  }
  if (averages.includes("micro")) {
    data.push(rocTrace(roc.micro.roc, label(compText("roc", "leyenda_micro_roc"), { auc: roc.micro.auc }), ROC_COLORS.micro, { line: { dash: "dot", width: 4 } }));
    data.push(prTrace(roc.micro.pr, label(compText("roc", "leyenda_micro_pr"), { ap: roc.micro.ap }), ROC_COLORS.micro, { line: { dash: "dot", width: 4 } }));
  }
  data.push(rocTrace({ x: [0, 1], y: [0, 1], thresholds: null }, escapePlotly(compText("roc", "leyenda_aleatorio")), withAlpha("#000000", 0.5), {
    line: { dash: "dash", width: 1.5 },
    hoverinfo: "skip",
  }));
  return { data, layout };
}

// --- C5: probabilities ---

export function binaryProbabilityFigure(perRealClass: readonly ClassHistogram[], theme: ChartTheme = LIGHT_THEME): Figure {
  const layout = baseLayout(theme, escapePlotly(compText("probabilidades", "title_binaria")));
  layout.xaxis = { ...axis(theme, escapePlotly(compText("probabilidades", "xlabel_binaria"))), range: [0, 1] };
  layout.yaxis = { ...axis(theme, escapePlotly(compText("probabilidades", "ylabel"))), rangemode: "tozero" };
  layout.barmode = "overlay";
  layout.bargap = 0;
  layout.showlegend = true;
  layout.legend = insideLegend(0.01, 0.99, "left", "top", theme);
  const data: Data[] = perRealClass
    .filter((group) => group.histogram.counts.length > 0)
    .map((group) => ({
      type: "bar",
      x: centers(group.histogram.edges),
      y: [...group.histogram.counts],
      width: widths(group.histogram.edges),
      name: label(compText("probabilidades", "leyenda_clase"), { clase: group.name }),
      marker: { color: withAlpha(PROB_COLORS[group.class_index] ?? PROB_COLORS[0], 0.7), line: { color: "#000000", width: 1 } },
      hovertemplate: "%{fullData.name}<br>p ≈ %{x:.3f}: %{y}<extra></extra>",
    }));
  data.push({
    type: "scatter",
    mode: "lines",
    x: [0.5, 0.5],
    y: [0, Math.max(1, ...perRealClass.flatMap((g) => g.histogram.counts))],
    name: escapePlotly(compText("probabilidades", "leyenda_umbral")),
    line: { color: "#ff0000", dash: "dash", width: 2 },
    hoverinfo: "skip",
  });
  return { data, layout };
}

export function classProbabilityFigure(group: ClassHistogram, theme: ChartTheme = LIGHT_THEME): Figure {
  const layout = baseLayout(theme, label(compText("probabilidades", "title_multiclase"), { clase: group.name }));
  layout.title = { ...(layout.title as object), font: { size: 14, color: theme.text } };
  layout.xaxis = { ...axis(theme, escapePlotly(compText("probabilidades", "xlabel_multiclase"))), range: [0, 1] };
  layout.yaxis = { ...axis(theme, escapePlotly(compText("probabilidades", "ylabel"))), rangemode: "tozero" };
  layout.bargap = 0;
  layout.margin = { l: 60, r: 20, t: 50, b: 60 };
  return {
    data: [
      {
        type: "bar",
        x: centers(group.histogram.edges),
        y: [...group.histogram.counts],
        width: widths(group.histogram.edges),
        marker: { color: withAlpha(SET3[group.class_index % SET3.length] ?? SET3[0], 0.85), line: { color: "#000000", width: 1 } },
        hovertemplate: "p ≈ %{x:.3f}: %{y}<extra></extra>",
      },
    ],
    layout,
  };
}

// --- C7b: residuals ---

export interface ResidualsInput {
  y_true: readonly number[];
  y_pred: readonly number[];
  std: number;
  mean: number;
  hist: Histogram;
}

export function residualsFigure(view: ResidualsInput, theme: ChartTheme = LIGHT_THEME): Figure {
  const residuals = view.y_true.map((y, i) => y - (view.y_pred[i] ?? Number.NaN));
  const layout = baseLayout(theme, "");
  const frame = { showline: true, mirror: true, linecolor: theme.axis, zeroline: false };
  layout.xaxis = { ...axis(theme, escapePlotly(compText("residuos", "res_xlabel"))), ...frame, domain: [0, 0.46] };
  layout.yaxis = { ...axis(theme, escapePlotly(compText("residuos", "res_ylabel"))), ...frame };
  layout.xaxis2 = { ...axis(theme, escapePlotly(compText("residuos", "hist_xlabel"))), ...frame, domain: [0.54, 1], anchor: "y2" };
  layout.yaxis2 = { ...axis(theme, escapePlotly(compText("residuos", "hist_ylabel"))), ...frame, anchor: "x2", rangemode: "tozero" };
  layout.bargap = 0;
  layout.showlegend = true;
  layout.hovermode = "closest";
  layout.legend = insideLegend(0.45, 0.99, "right", "top", theme);
  (layout as Record<string, unknown>)["legend2"] = insideLegend(0.99, 0.99, "right", "top", theme);
  const maxCount = Math.max(1, ...view.hist.counts);
  layout.annotations = [
    { text: `<b>${escapePlotly(compText("residuos", "res_title"))}</b>`, xref: "paper", yref: "paper", x: 0.23, y: 1.04, xanchor: "center", yanchor: "bottom", showarrow: false, font: { size: 15 } },
    { text: `<b>${escapePlotly(compText("residuos", "hist_title"))}</b>`, xref: "paper", yref: "paper", x: 0.77, y: 1.04, xanchor: "center", yanchor: "bottom", showarrow: false, font: { size: 15 } },
    {
      text: label(compText("residuos", "res_desv"), { std: view.std }),
      xref: "paper",
      yref: "paper",
      x: 0.01,
      y: 0.98,
      xanchor: "left",
      yanchor: "top",
      showarrow: false,
      bgcolor: RESIDUAL_COLORS.std,
      bordercolor: "#5b8fa8",
      borderwidth: 1,
      borderpad: 4,
      font: { size: 12, color: "#243247" },
    },
  ];
  const [lo, hi] = minMax(view.y_pred);
  return {
    data: [
      {
        type: "scatter",
        mode: "markers",
        x: [...view.y_pred],
        y: residuals,
        marker: { color: withAlpha(RESIDUAL_COLORS.points, 0.6), size: 7, line: { color: "#000000", width: 0.4 } },
        showlegend: false,
        hovertemplate: "Predicción %{x:.4g}<br>Residuo %{y:.4g}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        x: [lo, hi],
        y: [0, 0],
        name: escapePlotly(compText("residuos", "res_leyenda")),
        line: { color: "#ff0000", dash: "dash", width: 2 },
        hoverinfo: "skip",
      },
      {
        type: "bar",
        x: centers(view.hist.edges),
        y: [...view.hist.counts],
        width: widths(view.hist.edges),
        xaxis: "x2",
        yaxis: "y2",
        marker: { color: withAlpha(RESIDUAL_COLORS.hist, 0.7), line: { color: "#000000", width: 1 } },
        showlegend: false,
        hovertemplate: "%{x:.4g}: %{y}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        x: [view.mean, view.mean],
        y: [0, maxCount],
        xaxis: "x2",
        yaxis: "y2",
        legend: "legend2",
        name: label(compText("residuos", "hist_media"), { media: view.mean }),
        line: { color: "#ff0000", dash: "dash", width: 2 },
        hoverinfo: "skip",
      } as Data,
      {
        type: "scatter",
        mode: "lines",
        x: [0, 0],
        y: [0, maxCount],
        xaxis: "x2",
        yaxis: "y2",
        legend: "legend2",
        name: escapePlotly(compText("residuos", "hist_ideal")),
        line: { color: withAlpha("#008000", 0.7), width: 2 },
        hoverinfo: "skip",
      } as Data,
    ],
    layout,
  };
}

// --- C8: decision boundary and prediction surface ---

/** Discrete colour scale: class i of n gets a flat band (heatmap with `zsmooth: false`). */
export function discreteScale(colors: readonly string[]): [number, string][] {
  const n = colors.length;
  if (n === 1) {
    return [
      [0, colors[0] ?? "#cccccc"],
      [1, colors[0] ?? "#cccccc"],
    ];
  }
  return colors.flatMap((color, i) => [
    [i / n, color],
    [(i + 1) / n, color],
  ]) as [number, string][];
}

function gridRows<T>(z: readonly T[], cols: number): T[][] {
  const rows: T[][] = [];
  for (let start = 0; start < z.length; start += cols) {
    rows.push(z.slice(start, start + cols));
  }
  return rows;
}

function pointsColumn(grid: DecisionGrid, column: 0 | 1): number[] {
  const out: number[] = [];
  for (let r = 0; r < grid.points.rows; r += 1) {
    out.push(grid.points.values[r * grid.points.cols + column] ?? Number.NaN);
  }
  return out;
}

export function boundaryFigure(grid: DecisionGrid, theme: ChartTheme = LIGHT_THEME, webgl = true): Figure {
  if (grid.values.kind !== "classes") {
    return surfaceFigure(grid, theme, webgl);
  }
  const names = grid.class_names ?? [];
  // Class codes of the points (indices into `class_names`), never the target values:
  // a dataset with classes {3, 6} has codes {0, 1}.
  const codes = grid.point_codes ?? [];
  const n = Math.max(names.length, 1 + Math.max(0, ...grid.values.z), 1 + Math.max(0, ...codes), 1);
  const colors = Array.from({ length: n }, (_, i) => classColor(i, n));
  const [fx, fy] = grid.feature_names;
  const layout = baseLayout(theme, label(compText("frontera", "title_frontera"), { f1: fx, f2: fy }));
  layout.xaxis = { ...axis(theme, escapePlotly(fx)), range: [grid.x[0], grid.x[grid.x.length - 1]], showline: true, mirror: true, zeroline: false };
  layout.yaxis = { ...axis(theme, escapePlotly(fy)), range: [grid.y[0], grid.y[grid.y.length - 1]], showline: true, mirror: true, zeroline: false };
  layout.showlegend = true;
  layout.legend = { title: { text: escapePlotly(compText("frontera", "leyenda_clases")) }, ...insideLegend(0.99, 0.99, "right", "top", theme) };
  layout.hovermode = "closest";
  const xs = pointsColumn(grid, 0);
  const ys = pointsColumn(grid, 1);
  const classNames = colors.map((_, i) => names[i] ?? `Clase ${i}`);
  const data: Data[] = [
    {
      type: "heatmap",
      x: grid.x,
      y: grid.y,
      z: gridRows(grid.values.z, grid.x.length),
      zmin: -0.5,
      zmax: n - 0.5,
      zsmooth: false,
      colorscale: discreteScale(colors),
      showscale: false,
      opacity: 0.35,
      customdata: gridRows(grid.values.z.map((c) => classNames[c] ?? String(c)), grid.x.length),
      hovertemplate: "%{x:.3g}, %{y:.3g} → %{customdata}<extra></extra>",
    } as Data,
  ];
  classNames.forEach((name, classIndex) => {
    const px: number[] = [];
    const py: number[] = [];
    codes.forEach((code, i) => {
      if (code === classIndex) {
        px.push(xs[i] ?? Number.NaN);
        py.push(ys[i] ?? Number.NaN);
      }
    });
    data.push({
      type: webgl ? "scattergl" : "scatter",
      mode: "markers",
      x: px,
      y: py,
      name: escapePlotly(name),
      marker: { color: colors[classIndex], size: 9, opacity: 0.85, line: { color: "#000000", width: 1 } },
      hovertemplate: "%{fullData.name}<br>%{x:.4g}, %{y:.4g}<extra></extra>",
    } as Data);
  });
  return { data, layout };
}

/**
 * 🧠 (D78, review of F5c): below this fraction of the real range, the predicted range gets its
 * own colour scale. With a shared scale a network that predicts ≈ 0.8 for targets of 25–346
 * (Diabetes with the defaults) is one flat colour; but separate scales make a surface that
 * predicts well look as if it matched the points when the colours mean different values. 20 %
 * keeps the shared (honest) scale for any network whose predictions span a visible part of
 * the colour bar.
 */
export const OWN_SCALE_FRACTION = 0.2;

/** Whether the surface needs its own scale (`ownScales` mode): see [`OWN_SCALE_FRACTION`]. */
export function needsOwnScales(predMin: number, predMax: number, realMin: number, realMax: number): boolean {
  const real = realMax - realMin;
  const pred = predMax - predMin;
  if (!(real > 0) || !Number.isFinite(pred)) {
    return false;
  }
  return pred < OWN_SCALE_FRACTION * real;
}

/**
 * `ownScales` (🧠, D78): a colour scale shared by the surface and the points, unless the
 * predictions cover less than [`OWN_SCALE_FRACTION`] of the real range; then the surface takes
 * the range of the predictions and the points their own, each with its colour bar ("Valor
 * predicho" / "Valor real"). Python's contourf always scaled the surface to the predictions.
 */
export function surfaceFigure(
  grid: DecisionGrid,
  theme: ChartTheme = LIGHT_THEME,
  webgl = true,
  requestedScales: { surface: string; points: string } | null = null,
): Figure {
  const [fx, fy] = grid.feature_names;
  const layout = baseLayout(theme, label(compText("frontera", "title_superficie"), { f1: fx, f2: fy }));
  layout.xaxis = { ...axis(theme, escapePlotly(fx)), range: [grid.x[0], grid.x[grid.x.length - 1]], showline: true, mirror: true, zeroline: false };
  layout.yaxis = { ...axis(theme, escapePlotly(fy)), range: [grid.y[0], grid.y[grid.y.length - 1]], showline: true, mirror: true, zeroline: false };
  layout.hovermode = "closest";
  const values = grid.values.kind === "values" ? grid.values : { z: [] as number[], min: 0, max: 1 };
  const [tmin, tmax] = minMax(grid.point_target);
  // Separate bars only when a shared one would flatten the surface (see OWN_SCALE_FRACTION).
  const ownScales = requestedScales && needsOwnScales(values.min, values.max, tmin, tmax) ? requestedScales : null;
  const cmin = ownScales ? values.min : Math.min(values.min, tmin);
  const cmax = ownScales ? (values.max > values.min ? values.max : values.min + 1e-9) : Math.max(values.max, tmax);
  const pmin = ownScales ? tmin : cmin;
  const pmax = ownScales ? tmax : cmax;
  if (ownScales) {
    layout.margin = { l: 70, r: 150, t: 60, b: 70 };
  }
  const bar = (title: string, x: number) => ({ title: { text: escapePlotly(title), side: "right" as const }, len: 0.8, thickness: 16, outlinewidth: 0, x });
  return {
    data: [
      {
        type: "contour",
        x: grid.x,
        y: grid.y,
        z: gridRows(values.z, grid.x.length),
        colorscale: VIRIDIS,
        zmin: cmin,
        zmax: cmax,
        ncontours: 20,
        contours: { coloring: "fill", showlines: false },
        opacity: 0.6,
        colorbar: ownScales ? bar(ownScales.surface, 1.02) : { title: { text: escapePlotly(compText("frontera", "colorbar_superficie")), side: "right" }, len: 0.8, thickness: 16, outlinewidth: 0 },
        hovertemplate: "%{x:.3g}, %{y:.3g} → %{z:.4g}<extra></extra>",
      } as Data,
      {
        type: webgl ? "scattergl" : "scatter",
        mode: "markers",
        x: pointsColumn(grid, 0),
        y: pointsColumn(grid, 1),
        marker: {
          color: [...grid.point_target],
          colorscale: VIRIDIS,
          cmin: pmin,
          cmax: pmax,
          size: 8,
          line: { color: "#000000", width: 1 },
          showscale: ownScales !== null,
          ...(ownScales ? { colorbar: bar(ownScales.points, 1.16) } : {}),
        },
        customdata: [...grid.point_target],
        showlegend: false,
        hovertemplate: "%{x:.4g}, %{y:.4g}<br>objetivo %{customdata:.4g}<extra></extra>",
      } as Data,
    ],
    layout,
  };
}

// --- Coefficients (§2.5) ---

/** Horizontal bars by |coef|, the largest on top (Python draws it at the bottom). */
export function coefficientFigure(rows: readonly CoefficientRow[], theme: ChartTheme = LIGHT_THEME): Figure {
  const ordered = [...rows].reverse();
  const values = ordered.map((r) => r.coef);
  const [lo, hi] = minMax([0, ...values]);
  const pad = (hi - lo) * 0.12 || 1;
  const legend = regList("coeficientes", "grafico_leyenda");
  const layout = baseLayout(theme, escapePlotly(regText("coeficientes", "grafico_title")));
  layout.xaxis = { ...axis(theme, escapePlotly(regText("coeficientes", "grafico_xlabel"))), range: [lo - pad, hi + pad], zeroline: false };
  layout.yaxis = { ...axis(theme, ""), type: "category", showgrid: false, automargin: true };
  layout.shapes = [line(0, 0, 0, 1, theme.text, "solid", 1, { yref: "paper" })];
  layout.bargap = 0.35;
  layout.showlegend = true;
  // Above the plot: inside it would cover the labels of the longest bars.
  layout.legend = { orientation: "h", x: 1, xanchor: "right", y: 1, yanchor: "bottom", font: { size: 12 } };
  layout.margin = { l: 30, r: 30, t: 80, b: 60 };
  const trace = (effect: "positive" | "negative"): Data => {
    const pick = ordered.filter((r) => r.effect === effect);
    return {
      type: "bar",
      orientation: "h",
      x: pick.map((r) => r.coef),
      y: esc(pick.map((r) => r.feature)),
      name: escapePlotly(legend[effect === "positive" ? 0 : 1] ?? effect),
      marker: { color: withAlpha(COEF_COLORS[effect], 0.7), line: { color: "#000000", width: 1 } },
      text: pick.map((r) => fixed(r.coef, 3)),
      textposition: "outside",
      cliponaxis: false,
      textfont: { size: 12, color: theme.text },
      hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
    } as Data;
  };
  layout.yaxis = { ...layout.yaxis, categoryorder: "array", categoryarray: esc(ordered.map((r) => r.feature)) };
  return { data: [trace("positive"), trace("negative")], layout };
}

/** Height of the coefficient chart: Python's `max(6, 0.4·n)` inches. */
export function coefficientHeight(n: number): number {
  return Math.max(380, Math.min(1400, 60 + n * 42));
}
