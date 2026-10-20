/**
 * Plotly figures of the 📊 Datos tab and the CSV page. Pure functions (data → figure) so
 * they can be unit-tested without a browser; `PlotlyChart.svelte` draws them.
 * Colours follow the Python version (fase1-shell-datos.md §2.4–§2.6).
 *
 * Plotly reads pseudo-HTML (`<a href>`, `<b>`…) in every label, so strings that come from
 * data (column names, class values, car names) go through `escapePlotly`, and hover
 * templates refer to them with `%{…}` instead of embedding them.
 */
import type { Annotations, Data, Layout } from "plotly.js";
import type { CorrelationMatrix } from "./bindings/CorrelationMatrix";
import type { CsvTargetStats } from "./bindings/CsvTargetStats";
import type { DiagPanel } from "./bindings/DiagPanel";
import type { PairplotData } from "./bindings/PairplotData";
import type { TargetDistribution } from "./bindings/TargetDistribution";
import { escapePlotly } from "./escape";

const esc = (values: readonly string[]): string[] => values.map(escapePlotly);

export interface Figure {
  data: Data[];
  layout: Partial<Layout>;
}

export interface ChartTheme {
  text: string;
  grid: string;
  axis: string;
  background: string;
}

export const LIGHT_THEME: ChartTheme = {
  text: "#243247",
  grid: "#e5eaf1",
  axis: "#7b8aa0",
  background: "#ffffff",
};

/** Theme from the CSS variables of `theme.css` (dark mode included). */
export function themeFromCss(element: Element = document.documentElement): ChartTheme {
  const style = getComputedStyle(element);
  const read = (name: string, fallback: string): string => style.getPropertyValue(name).trim() || fallback;
  return {
    text: read("--chart-text", LIGHT_THEME.text),
    grid: read("--chart-grid", LIGHT_THEME.grid),
    axis: read("--chart-axis", LIGHT_THEME.axis),
    background: read("--chart-bg", LIGHT_THEME.background),
  };
}

/** matplotlib `tab10`, the hue palette of seaborn for ≤ 10 levels. */
export const TAB10 = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#7f7f7f",
  "#bcbd22",
  "#17becf",
] as const;

/** seaborn barplot C0 with saturation 0.75. */
export const BAR_COLOR = "#3274a1";
/** matplotlib C0. */
export const HIST_COLOR = "#1f77b4";

/** matplotlib `coolwarm`, sampled. */
export const COOLWARM: [number, string][] = [
  [0, "#3b4cc0"],
  [0.125, "#5977e3"],
  [0.25, "#7b9ff9"],
  [0.375, "#9ebeff"],
  [0.5, "#dddddd"],
  [0.625, "#f7b89c"],
  [0.75, "#f49a7b"],
  [0.875, "#de604d"],
  [1, "#b40426"],
];

/** `sns.cubehelix_palette(as_cmap=True)`: light pink → dark purple (numeric hue, D11). */
export const CUBEHELIX: [number, string][] = [
  [0, "#edd1cb"],
  [0.2, "#d8a6a6"],
  [0.4, "#b77f93"],
  [0.6, "#8a5d82"],
  [0.8, "#583e66"],
  [1, "#2d1e3e"],
];

/** Colour of hue level `index`: tab10, then an evenly spaced HSL wheel (seaborn husl). */
export function levelColor(index: number, levels: number): string {
  if (levels <= TAB10.length) {
    return TAB10[index % TAB10.length] ?? TAB10[0];
  }
  const hue = Math.round((360 * index) / levels);
  return `hsl(${hue}, 65%, 50%)`;
}

export function withAlpha(hex: string, alpha: number): string {
  const match = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!match) {
    return hex.startsWith("hsl(") ? hex.replace("hsl(", "hsla(").replace(")", `, ${alpha})`) : hex;
  }
  const [r, g, b] = [match[1], match[2], match[3]].map((part) => parseInt(part ?? "0", 16));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function baseLayout(theme: ChartTheme, title: string): Partial<Layout> {
  return {
    title: { text: title, font: { size: 16, color: theme.text }, x: 0.5, xanchor: "center" },
    font: { family: "Inter Variable, Inter, Segoe UI, Arial, sans-serif", size: 13, color: theme.text },
    paper_bgcolor: theme.background,
    plot_bgcolor: theme.background,
    margin: { l: 70, r: 30, t: 60, b: 70 },
    hoverlabel: { font: { family: "Inter Variable, Inter, Arial, sans-serif" } },
    showlegend: false,
  };
}

export function axis(theme: ChartTheme, title: string) {
  return {
    title: { text: title, font: { size: 13 } },
    gridcolor: theme.grid,
    linecolor: theme.axis,
    zerolinecolor: theme.grid,
    tickcolor: theme.axis,
    automargin: true,
  };
}

/** Classification: bars sorted by class name; regression: 30-bin histogram. */
export function targetDistributionFigure(dist: TargetDistribution, theme: ChartTheme = LIGHT_THEME): Figure {
  if (dist.kind === "bar") {
    const layout = baseLayout(theme, "Distribución de Clases");
    layout.xaxis = { ...axis(theme, "Clase"), type: "category", tickangle: dist.labels.length > 3 ? -45 : 0 };
    layout.yaxis = axis(theme, "Cantidad");
    layout.bargap = 0.2;
    return {
      data: [
        {
          type: "bar",
          x: esc(dist.labels),
          y: dist.counts,
          marker: { color: BAR_COLOR },
          hovertemplate: "%{x}: %{y}<extra></extra>",
        },
      ],
      layout,
    };
  }
  return histogramFigure(dist.edges, dist.counts, "Distribución de Valores Objetivo", "Valor", "Frecuencia", theme);
}

/** Histogram from precomputed edges/counts (matplotlib `hist`, alpha 0.7, black edges). */
export function histogramFigure(
  edges: readonly number[],
  counts: readonly number[],
  title: string,
  xTitle: string,
  yTitle: string,
  theme: ChartTheme = LIGHT_THEME,
): Figure {
  const centers = counts.map((_, i) => ((edges[i] ?? 0) + (edges[i + 1] ?? 0)) / 2);
  const widths = counts.map((_, i) => (edges[i + 1] ?? 0) - (edges[i] ?? 0));
  const layout = baseLayout(theme, title);
  layout.xaxis = axis(theme, xTitle);
  layout.yaxis = axis(theme, yTitle);
  layout.bargap = 0;
  return {
    data: [
      {
        type: "bar",
        x: centers,
        y: [...counts],
        width: widths,
        marker: { color: withAlpha(HIST_COLOR, 0.7), line: { color: "#000000", width: 1 } },
        customdata: counts.map((_, i) => [edges[i] ?? 0, edges[i + 1] ?? 0]),
        hovertemplate: "[%{customdata[0]:.4g}, %{customdata[1]:.4g}): %{y}<extra></extra>",
      },
    ],
    layout,
  };
}

/** CSV page: "Distribución de {col}" bars (pandas `plot.bar`, C0) or histogram. */
export function csvTargetFigure(column: string, stats: CsvTargetStats, theme: ChartTheme = LIGHT_THEME): Figure {
  const name = escapePlotly(column);
  const title = `Distribución de ${name}`;
  if (stats.kind === "classes") {
    const layout = baseLayout(theme, title);
    layout.xaxis = { ...axis(theme, name), type: "category", tickangle: -45 };
    layout.yaxis = axis(theme, "Cantidad");
    return {
      data: [
        {
          type: "bar",
          x: stats.value_counts.map(([label]) => escapePlotly(label)),
          y: stats.value_counts.map(([, count]) => count),
          marker: { color: HIST_COLOR },
          hovertemplate: "%{x}: %{y}<extra></extra>",
        },
      ],
      layout,
    };
  }
  return histogramFigure(stats.hist_edges, stats.hist_counts, title, name, "Frecuencia", theme);
}

/** Heatmap of the strict lower triangle (Python's `np.triu` mask), scale fixed to [-1, 1] (D12). */
export function correlationFigure(corr: CorrelationMatrix, theme: ChartTheme = LIGHT_THEME): Figure {
  const n = corr.labels.length;
  const z: (number | null)[][] = [];
  const text: string[][] = [];
  for (let i = 0; i < n; i += 1) {
    const row: (number | null)[] = [];
    const textRow: string[] = [];
    for (let j = 0; j < n; j += 1) {
      const value = j < i ? (corr.values.values[i * n + j] ?? null) : null;
      row.push(value);
      textRow.push(value === null ? "" : value.toFixed(2));
    }
    z.push(row);
    text.push(textRow);
  }
  const annotate = n <= 20;
  const fontSize = n <= 8 ? 13 : n <= 14 ? 11 : 9;
  const layout = baseLayout(theme, "Matriz de Correlación de Características");
  layout.xaxis = { ...axis(theme, ""), showgrid: false, tickangle: -45, constrain: "domain", type: "category" };
  layout.yaxis = {
    ...axis(theme, ""),
    showgrid: false,
    autorange: "reversed",
    scaleanchor: "x",
    constrain: "domain",
    type: "category",
  };
  layout.margin = { l: 150, r: 40, t: 60, b: 140 };
  (layout.xaxis as Record<string, unknown>)["showline"] = false;
  (layout.yaxis as Record<string, unknown>)["showline"] = false;
  return {
    data: [
      {
        type: "heatmap",
        x: esc(corr.labels),
        y: esc(corr.labels),
        z,
        zmin: -1,
        zmax: 1,
        zmid: 0,
        colorscale: COOLWARM,
        xgap: 1,
        ygap: 1,
        hoverongaps: false,
        text,
        texttemplate: annotate ? "%{text}" : "",
        textfont: { size: fontSize },
        colorbar: { len: 0.8, thickness: 18, outlinewidth: 0, tickfont: { color: theme.text } },
        hovertemplate: "%{y} · %{x}: %{z:.3f}<extra></extra>",
      } as unknown as Data,
    ],
    layout,
  };
}

/** Height (px) of the correlation heatmap: square cells that use the content width. */
export function correlationHeight(n: number): number {
  return n <= 20 ? 760 : 980;
}

/** Height (px) of a pairplot with `k` features. */
export function pairplotHeight(k: number): number {
  return Math.max(360, Math.min(1100, k * 190 + 110));
}

function axisName(kind: "x" | "y", index: number): string {
  return index === 0 ? kind : `${kind}${index + 1}`;
}

function layoutAxisKey(kind: "x" | "y", index: number): `${"x" | "y"}axis${string}` {
  return `${kind}axis${index === 0 ? "" : index + 1}`;
}

function diagonalTraces(panel: DiagPanel, levels: readonly string[], xa: string, ya: string, alpha: number): Data[] {
  const continuous = levels.length === 0;
  const colorOf = (level: string): string => {
    if (continuous) {
      return "#6b4c7a";
    }
    const index = levels.indexOf(level);
    return levelColor(index < 0 ? 0 : index, levels.length);
  };
  if (panel.kind === "kde") {
    return panel.curves
      .filter((curve) => curve.density.length > 0)
      .map((curve) => {
        const steps = curve.density.length - 1;
        const x = curve.density.map((_, i) => curve.grid_min + ((curve.grid_max - curve.grid_min) * i) / steps);
        const color = colorOf(curve.level);
        return {
          type: "scatter",
          mode: "lines",
          x,
          y: curve.density,
          xaxis: xa,
          yaxis: ya,
          fill: "tozeroy",
          fillcolor: withAlpha(color, alpha * 0.55),
          line: { color, width: 1.5 },
          legendgroup: curve.level,
          showlegend: false,
          name: escapePlotly(curve.level),
          hovertemplate: `${continuous ? "" : "%{fullData.name}<br>"}%{x:.3g}: %{y:.3g}<extra></extra>`,
        } as Data;
      });
  }
  const centers = panel.edges.slice(0, -1).map((edge, i) => (edge + (panel.edges[i + 1] ?? edge)) / 2);
  const widths = panel.edges.slice(0, -1).map((edge, i) => (panel.edges[i + 1] ?? edge) - edge);
  return panel.series.map((series) => {
    const color = colorOf(series.level);
    return {
      type: "bar",
      x: centers,
      y: series.counts,
      width: widths,
      xaxis: xa,
      yaxis: ya,
      marker: { color: withAlpha(color, alpha), line: { color: withAlpha(color, 1), width: 0.6 } },
      legendgroup: series.level,
      showlegend: false,
      name: escapePlotly(series.level),
      hovertemplate: `${continuous ? "" : "%{fullData.name}<br>"}%{x:.3g}: %{y}<extra></extra>`,
    } as Data;
  });
}

/**
 * `sns.pairplot(hue="target")`: k×k grid, WebGL scatter off the diagonal (fluid with
 * thousands of points), KDE or histogram per level on the diagonal, legend "target".
 * Regression (D11): continuous cubehelix colour with a colour bar and a single diagonal.
 * Without WebGL (some lab machines, software rendering) it falls back to SVG `scatter`.
 */
export function pairplotFigure(plot: PairplotData, theme: ChartTheme = LIGHT_THEME, webgl = true): Figure {
  const scatterType = webgl ? "scattergl" : "scatter";
  const k = plot.features.length;
  const n = plot.n_points;
  const gap = k > 1 ? 0.035 : 0;
  const size = (1 - gap * (k - 1)) / k;
  const start = (i: number): number => i * (size + gap);
  const categorical = plot.hue.kind === "categorical";
  const levels = plot.hue.kind === "categorical" ? plot.hue.levels : [];
  const column = (c: number): number[] => {
    const out = new Array<number>(n);
    for (let r = 0; r < n; r += 1) {
      out[r] = plot.points.values[r * k + c] ?? Number.NaN;
    }
    return out;
  };
  const columns = plot.features.map((_, c) => column(c));
  const alpha = plot.high_quality ? 0.7 : 0.6;
  const markerSize = plot.high_quality ? 7 : 5;
  const markerLine = plot.high_quality ? { color: "#ffffff", width: 0.5 } : { width: 0 };
  const diagAlpha = plot.high_quality ? 0.8 : 0.6;
  const tooltips = plot.tooltips ? esc(plot.tooltips) : null;
  const features = esc(plot.features);

  const layout = baseLayout(theme, escapePlotly(plot.title));
  layout.showlegend = categorical;
  layout.legend = {
    title: { text: "target" },
    x: 1.02,
    xanchor: "left",
    y: 0.5,
    yanchor: "middle",
    itemsizing: "constant",
    font: { size: 13 },
  };
  layout.margin = { l: 80, r: categorical ? 140 : 100, t: 60, b: 70 };
  layout.barmode = "overlay";
  layout.bargap = 0;
  layout.hovermode = "closest";
  layout.dragmode = "zoom";

  const data: Data[] = [];
  for (let row = 0; row < k; row += 1) {
    for (let col = 0; col < k; col += 1) {
      const index = row * k + col;
      const xa = axisName("x", index);
      const ya = axisName("y", index);
      const xKey = layoutAxisKey("x", index);
      const yKey = layoutAxisKey("y", index);
      const isDiagonal = row === col;
      const xTitle = row === k - 1 ? (features[col] ?? "") : "";
      const yTitle = col === 0 ? (features[row] ?? "") : "";
      (layout as Record<string, unknown>)[xKey] = {
        ...axis(theme, xTitle),
        domain: [start(col), start(col) + size],
        anchor: ya,
        // Columns share the x scale, rows share the y scale (seaborn PairGrid).
        ...(index === col ? {} : { matches: axisName("x", col) }),
        showticklabels: row === k - 1,
        ticks: "outside",
        ticklen: 3,
        zeroline: false,
      };
      (layout as Record<string, unknown>)[yKey] = {
        ...axis(theme, yTitle),
        domain: [1 - start(row) - size, 1 - start(row)],
        anchor: xa,
        ...(isDiagonal ? {} : row === 0 && col === 1 ? {} : { matches: axisName("y", row === 0 ? 1 : row * k) }),
        showticklabels: col === 0 && !isDiagonal,
        showgrid: !isDiagonal,
        ticks: col === 0 && !isDiagonal ? "outside" : "",
        ticklen: 3,
        zeroline: false,
        rangemode: isDiagonal ? "tozero" : "normal",
      };
      if (isDiagonal) {
        const panel = plot.diagonal[row];
        if (panel) {
          data.push(...diagonalTraces(panel, levels, xa, ya, diagAlpha));
        }
        continue;
      }
      const xs = columns[col] ?? [];
      const ys = columns[row] ?? [];
      if (plot.hue.kind === "categorical") {
        const codes = plot.hue.codes;
        levels.forEach((level, levelIndex) => {
          const px: number[] = [];
          const py: number[] = [];
          const pt: string[] = [];
          for (let r = 0; r < n; r += 1) {
            if (codes[r] === levelIndex) {
              px.push(xs[r] ?? Number.NaN);
              py.push(ys[r] ?? Number.NaN);
              if (tooltips) {
                pt.push(tooltips[r] ?? "");
              }
            }
          }
          data.push({
            type: scatterType,
            mode: "markers",
            x: px,
            y: py,
            xaxis: xa,
            yaxis: ya,
            name: escapePlotly(level),
            legendgroup: level,
            showlegend: index === 1 || (k === 1 && index === 0),
            marker: { color: levelColor(levelIndex, levels.length), size: markerSize, opacity: alpha, line: markerLine },
            ...(tooltips ? { text: pt } : {}),
            hovertemplate: `${tooltips ? "%{text}<br>" : ""}%{fullData.name}<br>%{x:.4g}, %{y:.4g}<extra></extra>`,
          } as Data);
        });
      } else {
        const hue = plot.hue;
        data.push({
          type: scatterType,
          mode: "markers",
          x: xs,
          y: ys,
          xaxis: xa,
          yaxis: ya,
          name: "target",
          showlegend: false,
          marker: {
            color: hue.values,
            cmin: hue.min ?? undefined,
            cmax: hue.max ?? undefined,
            colorscale: CUBEHELIX,
            size: markerSize,
            opacity: alpha,
            line: markerLine,
            showscale: index === 1,
            colorbar: { title: { text: "target" }, len: 0.6, thickness: 16, outlinewidth: 0 },
          },
          ...(tooltips ? { text: tooltips } : {}),
          customdata: hue.values,
          hovertemplate: `${tooltips ? "%{text}<br>" : ""}target = %{customdata:.4g}<br>%{x:.4g}, %{y:.4g}<extra></extra>`,
        } as Data);
      }
    }
  }
  if (k === 1) {
    layout.showlegend = false;
  }
  return { data, layout };
}

/** Plotly annotations are not used; kept for typing consumers that add notes. */
export type { Annotations };

let webglSupport: boolean | null = null;

/** Whether this WebView can create a WebGL context (cached). */
export function hasWebGL(): boolean {
  if (webglSupport === null) {
    try {
      const canvas = document.createElement("canvas");
      webglSupport = Boolean(canvas.getContext("webgl") ?? canvas.getContext("experimental-webgl"));
    } catch {
      webglSupport = false;
    }
  }
  return webglSupport;
}
