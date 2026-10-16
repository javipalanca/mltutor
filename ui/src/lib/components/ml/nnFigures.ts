/**
 * C27: the 1×2 history figure of `viz/nn.py::plot_training_history` (loss | accuracy or
 * MAE, Entrenamiento in blue and Validación in red, height 400, x axis "Época"). NEW: the
 * epochs start at 1 (Python plotted the index from 0), the epoch whose weights EarlyStopping
 * restored is marked, and so are the reductions of the learning rate.
 */
import type { Data, Layout, Shape } from "plotly.js";
import type { EpochLog } from "../../bindings/EpochLog";
import type { NnMetric } from "../../bindings/NnMetric";
import { axis, baseLayout, LIGHT_THEME, type ChartTheme, type Figure } from "../../charts";
import { nnText } from "../../content";

const ev = (key: string): string => nnText("evaluacion", key);

export interface HistoryOptions {
  bestEpoch?: number | null;
  lrDrops?: readonly number[];
  /** Fixed x range (the live chart shows the whole training from the start). */
  maxEpochs?: number | null;
}

const value = (v: number | null): number | null => (v === null || !Number.isFinite(v) ? null : v);

export function historyFigure(
  logs: readonly EpochLog[],
  metric: NnMetric,
  options: HistoryOptions = {},
  theme: ChartTheme = LIGHT_THEME,
): Figure {
  const x = logs.map((l) => l.epoch);
  const regression = metric === "mae";
  const train = ev("serie_entrenamiento");
  const val = ev("serie_validacion");
  const line = (y: (number | null)[], name: string, color: string, xaxis: string, yaxis: string, legend: boolean): Data => ({
    type: "scatter",
    mode: logs.length === 1 ? "markers" : "lines",
    x,
    y,
    name,
    legendgroup: name,
    showlegend: legend,
    xaxis,
    yaxis,
    line: { color, width: 2 },
    marker: { color, size: 6 },
    hovertemplate: `${name}<br>Época %{x}<br>%{y:.4f}<extra></extra>`,
  });
  const data: Data[] = [
    line(logs.map((l) => value(l.loss)), train, "blue", "x", "y", true),
    line(logs.map((l) => value(l.val_loss)), val, "red", "x", "y", true),
    line(logs.map((l) => value(l.metric)), train, "blue", "x2", "y2", false),
    line(logs.map((l) => value(l.val_metric)), val, "red", "x2", "y2", false),
  ];
  const shapes: Partial<Shape>[] = [];
  const marks = (epoch: number, color: string, dash: "dot" | "dash"): void => {
    for (const ref of ["x", "x2"] as const) {
      shapes.push({
        type: "line",
        xref: ref,
        yref: ref === "x" ? "y domain" : "y2 domain",
        x0: epoch,
        x1: epoch,
        y0: 0,
        y1: 1,
        line: { color, width: 1.5, dash },
      } as Partial<Shape>);
    }
  };
  if (options.bestEpoch) {
    marks(options.bestEpoch, "#2ca02c", "dash");
  }
  for (const drop of options.lrDrops ?? []) {
    marks(drop + 0.5, "#9467bd", "dot");
  }
  const range: [number, number] | undefined = options.maxEpochs ? [0.5, Math.max(options.maxEpochs, 2) + 0.5] : undefined;
  const xa = (domain: [number, number], anchor: "y" | "y2") => ({
    ...axis(theme, ev("eje_epoca")),
    domain,
    anchor,
    ...(range ? { range, autorange: false } : {}),
  });
  const { title: _title, ...base } = baseLayout(theme, "");
  const layout: Partial<Layout> = {
    ...base,
    showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.22 },
    margin: { l: 64, r: 24, t: 48, b: 80 },
    xaxis: xa([0, 0.45], "y"),
    xaxis2: xa([0.55, 1], "y2"),
    yaxis: { ...axis(theme, regression ? ev("eje_mse") : ev("eje_perdida")), anchor: "x" as const },
    yaxis2: { ...axis(theme, regression ? ev("eje_mae") : ev("eje_precision")), anchor: "x2" as const },
    shapes,
    annotations: [
      { text: regression ? ev("grafica_perdida_regresion") : ev("grafica_perdida"), x: 0.225 },
      { text: regression ? ev("grafica_mae") : ev("grafica_precision"), x: 0.775 },
    ].map((a) => ({
      ...a,
      xref: "paper" as const,
      yref: "paper" as const,
      y: 1.04,
      yanchor: "bottom" as const,
      xanchor: "center" as const,
      showarrow: false,
      font: { size: 14, color: theme.text },
    })),
    uirevision: "history",
  };
  return { data, layout };
}

// --- C25 / C26: histograms of weights, biases and activations (binned in Rust, K2) ---

const centersOf = (edges: readonly number[]): number[] => edges.slice(0, -1).map((e, i) => (e + (edges[i + 1] ?? e)) / 2);
const widthsOf = (edges: readonly number[]): number[] => edges.slice(0, -1).map((e, i) => (edges[i + 1] ?? e) - e);

export interface BinnedHistogramOptions {
  title: string;
  xLabel: string;
  yLabel: string;
  color: string;
  /** Green dashed line with the annotation "Media" (🧠 Pesos), plus the grey line at 0. */
  mean?: number | null;
  meanLabel?: string;
  /** "Escala log en frecuencia". */
  log?: boolean;
  height?: number;
}

/**
 * `go.Histogram(x=values, nbinsx=…)` of 🧠 Pesos and 📉 Activaciones, drawn as bars from the
 * bins Rust computed (the raw values do not cross the IPC). `bargap` 0.02 as in Python.
 */
export function binnedHistogramFigure(
  hist: { edges: readonly number[]; counts: readonly number[] },
  options: BinnedHistogramOptions,
  theme: ChartTheme = LIGHT_THEME,
): Figure {
  const { title: _title, ...base } = baseLayout(theme, "");
  const widths = widthsOf(hist.edges);
  const data: Data[] = [
    {
      type: "bar",
      x: centersOf(hist.edges),
      y: [...hist.counts],
      width: widths.map((w) => w * 0.98),
      marker: { color: options.color, line: { color: "rgba(255,255,255,0.6)", width: 0.5 } },
      customdata: hist.counts.map((_, i) => [hist.edges[i] ?? 0, hist.edges[i + 1] ?? 0]),
      hovertemplate: "[%{customdata[0]:.4g}, %{customdata[1]:.4g}): %{y}<extra></extra>",
      showlegend: false,
    },
  ];
  const shapes: Partial<Shape>[] = [];
  const annotations: Layout["annotations"] = [];
  if (options.mean !== undefined && options.mean !== null && Number.isFinite(options.mean)) {
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: { color: "#808080", width: 1 } } as Partial<Shape>);
    shapes.push({
      type: "line",
      xref: "x",
      yref: "paper",
      x0: options.mean,
      x1: options.mean,
      y0: 0,
      y1: 1,
      line: { color: "green", width: 2, dash: "dash" },
    } as Partial<Shape>);
    annotations.push({
      x: options.mean,
      y: 1,
      xref: "x",
      yref: "paper",
      text: options.meanLabel ?? "Media",
      showarrow: false,
      xanchor: "left",
      yanchor: "top",
      font: { size: 12, color: theme.text },
    });
  }
  const layout: Partial<Layout> = {
    ...base,
    title: { text: options.title, font: { size: 16, color: theme.text }, x: 0.02, xanchor: "left" },
    margin: { l: 60, r: 20, t: 50, b: 56 },
    bargap: 0.02,
    xaxis: axis(theme, options.xLabel),
    yaxis: { ...axis(theme, options.yLabel), ...(options.log ? { type: "log" as const } : { rangemode: "tozero" as const }) },
    shapes,
    annotations,
  };
  return { data, layout };
}

/** "Medias de Activación (capa)": `go.Bar(y=neuron_means)`, x = neuron index. */
export function neuronMeansFigure(means: readonly number[], title: string, xLabel: string, yLabel: string, theme: ChartTheme = LIGHT_THEME): Figure {
  const { title: _title, ...base } = baseLayout(theme, "");
  return {
    data: [
      {
        type: "bar",
        x: means.map((_, i) => i),
        y: [...means],
        marker: { color: "#1f77b4" },
        hovertemplate: `${xLabel} %{x}<br>${yLabel}: %{y:.4f}<extra></extra>`,
        showlegend: false,
      },
    ],
    layout: {
      ...base,
      title: { text: title, font: { size: 16, color: theme.text }, x: 0.02, xanchor: "left" },
      margin: { l: 60, r: 20, t: 50, b: 56 },
      xaxis: axis(theme, xLabel),
      yaxis: axis(theme, yLabel),
    },
  };
}
