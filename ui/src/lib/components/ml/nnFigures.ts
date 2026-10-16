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
