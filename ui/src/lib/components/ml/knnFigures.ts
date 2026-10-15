/**
 * Plotly figures of 🔍 K-Nearest Neighbors (fase4-knn.md §3): the interactive canvas (C18,
 * D54) and the heatmap of the distances between classes (C19, D67). Pure functions (DTO →
 * figure), unit-tested without a browser.
 *
 * The canvas keeps what the Python HTML showed (palette `colors[code % 8]`, border `#333`,
 * gold star, dashed `#FF4444` lines to the neighbours, green confidence arc, "NN%" and
 * the class name) and adds named axes and a tooltip with the duplicates (×n).
 */
import type { Data, Layout } from "plotly.js";
import type { ClassDistancesView } from "../../bindings/ClassDistancesView";
import type { Matrix } from "../../bindings/Matrix";
import type { PlaygroundView } from "../../bindings/PlaygroundView";
import { axis, baseLayout, LIGHT_THEME, type ChartTheme, type Figure } from "../../charts";
import { knnText } from "../../content";
import { escapePlotly, escapePlotlyTemplate } from "../../escape";
import { fmtText } from "../../fmt";
import { fixed } from "../../format";
import { VIRIDIS } from "./figures";

export type { Figure };

/** Python's canvas colours (also sent by the backend in `PlaygroundView.colors`). */
export const PLAYGROUND_PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA726", "#AB47BC", "#66BB6A", "#EF5350", "#42A5F5"] as const;
export const NEIGHBOR_LINE = "#FF4444";
export const STAR_COLOR = "#FFD700";
export const RING_COLOR = "#4CAF50";
export const POINT_BORDER = "#333333";

/** Pixels of the canvas → data units along each axis (for circles drawn in data space). */
export interface PixelScale {
  dx: number;
  dy: number;
}

export interface PlaygroundOptions {
  showConfidence: boolean;
  showDistances: boolean;
  /** Lines shown per test point (animation, D54); `undefined` = all of them. */
  revealed?: readonly number[];
  /** Index of the test point being dragged (drawn on top, no tooltip). */
  dragging?: number | null;
  scale: PixelScale;
  /** `scattergl` for big datasets. */
  webgl?: boolean;
}

/** Points with the same coordinates and class, drawn once with their count (×n). */
export interface Aggregated {
  x: number[];
  y: number[];
  count: number[];
  rows: number[][];
}

export function classColor(code: number, colors: readonly string[] = PLAYGROUND_PALETTE): string {
  return colors[code] ?? PLAYGROUND_PALETTE[code % PLAYGROUND_PALETTE.length] ?? "#888888";
}

/** Rows of `data` (2 columns) of each class code, duplicates merged. */
export function aggregateByClass(data: Matrix, codes: readonly number[]): Map<number, Aggregated> {
  const out = new Map<number, Aggregated>();
  const index = new Map<string, [Aggregated, number]>();
  for (let r = 0; r < data.rows; r += 1) {
    const x = data.values[r * data.cols] ?? Number.NaN;
    const y = data.values[r * data.cols + 1] ?? Number.NaN;
    const code = codes[r] ?? 0;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      continue;
    }
    const key = `${code}|${x}|${y}`;
    const found = index.get(key);
    if (found) {
      const [group, i] = found;
      group.count[i] = (group.count[i] ?? 1) + 1;
      group.rows[i]?.push(r);
      continue;
    }
    let group = out.get(code);
    if (!group) {
      group = { x: [], y: [], count: [], rows: [] };
      out.set(code, group);
    }
    index.set(key, [group, group.x.length]);
    group.x.push(x);
    group.y.push(y);
    group.count.push(1);
    group.rows.push([r]);
  }
  return out;
}

/** Marker diameter of a training point: Python's radius 4–8 px, a bit bigger with duplicates. */
export function pointSize(count: number): number {
  return Math.min(22, 12 + 3 * Math.log2(Math.max(1, count)));
}

/**
 * The arc of the confidence ring: from 12 o'clock clockwise, `fraction` of a circle of
 * `radiusPx` pixels around `(cx, cy)`, in data coordinates.
 */
export function arc(cx: number, cy: number, radiusPx: number, fraction: number, scale: PixelScale, steps = 48): { x: number[]; y: number[] } {
  const f = Math.max(0, Math.min(1, fraction));
  const n = Math.max(2, Math.ceil(steps * f) + 1);
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const t = Math.PI / 2 - (2 * Math.PI * f * i) / (n - 1);
    x.push(cx + radiusPx * Math.cos(t) * scale.dx);
    y.push(cy + radiusPx * Math.sin(t) * scale.dy);
  }
  return { x, y };
}

/** `Math.round(c·100)%`, as the cards and labels of the Python canvas. */
export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

const RING_RADIUS_PX = 18;
/** Distance (px) from the centre of a test point to its labels. */
const LABEL_OFFSET_PX = 32;

/** `point` kept inside `bounds` (`[x_min, x_max, y_min, y_max]`), as the backend requires. */
export function clampPoint([x, y]: readonly [number, number], bounds: readonly [number, number, number, number]): [number, number] {
  const [x0, x1, y0, y1] = bounds;
  return [Math.min(x1, Math.max(x0, x)), Math.min(y1, Math.max(y0, y))];
}

export function playgroundFigure(view: PlaygroundView, data: Matrix, codes: readonly number[], options: PlaygroundOptions, theme: ChartTheme = LIGHT_THEME): Figure {
  const [fx, fy] = view.feature_names;
  const [x0, x1, y0, y1] = view.bounds;
  const layout: Partial<Layout> = {
    ...baseLayout(theme, ""),
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 70, r: 24, t: 54, b: 64 },
    showlegend: true,
    legend: {
      orientation: "h",
      x: 0,
      xanchor: "left",
      y: 1.02,
      yanchor: "bottom",
      title: { text: escapePlotly(knnText("interactiva", "leyenda_titulo")) },
      bgcolor: "rgba(0,0,0,0)",
      font: { color: theme.text },
      itemclick: false,
      itemdoubleclick: false,
    },
    hovermode: "closest",
    dragmode: false,
    xaxis: { ...axis(theme, escapePlotly(fx)), range: [x0, x1], zeroline: false, gridcolor: "rgba(120,130,150,0.18)" },
    yaxis: { ...axis(theme, escapePlotly(fy)), range: [y0, y1], zeroline: false, gridcolor: "rgba(120,130,150,0.18)" },
    uirevision: `${fx}|${fy}`,
  };
  delete layout.title;
  const traces: Data[] = [];
  const groups = aggregateByClass(data, codes);
  const type = options.webgl ? "scattergl" : "scatter";
  const shown = [...new Set([...view.classes, ...groups.keys()])].sort((a, b) => a - b);
  for (const code of shown) {
    const group = groups.get(code) ?? { x: [], y: [], count: [], rows: [] };
    const name = view.class_names[code] ?? fmtText(knnText("interactiva", "etiqueta_clase"), { codigo: code });
    traces.push({
      type,
      mode: "markers",
      x: group.x,
      y: group.y,
      name: escapePlotly(name),
      customdata: group.count.map((n) => (n > 1 ? ` ×${n}` : "")),
      marker: { color: classColor(code, view.colors), size: group.count.map(pointSize), line: { color: POINT_BORDER, width: 1.5 }, opacity: 0.92 },
      hovertemplate: `%{fullData.name}%{customdata}<br>${escapePlotlyTemplate(fx)}: %{x:.4g}<br>${escapePlotlyTemplate(fy)}: %{y:.4g}<extra></extra>`,
    } as Data);
  }
  const points = view.points;
  // Neighbours: dashed lines (revealed one by one) and a thick red border.
  if (options.showDistances) {
    const lx: (number | null)[] = [];
    const ly: (number | null)[] = [];
    const hx: number[] = [];
    const hy: number[] = [];
    const ht: string[] = [];
    points.forEach((p, i) => {
      const limit = options.revealed?.[i] ?? p.neighbors.length;
      p.neighbors.slice(0, limit).forEach((n, order) => {
        const nx = data.values[n.dataset_row * data.cols] ?? Number.NaN;
        const ny = data.values[n.dataset_row * data.cols + 1] ?? Number.NaN;
        lx.push(p.point[0], nx, null);
        ly.push(p.point[1], ny, null);
        hx.push(nx);
        hy.push(ny);
        const clase = view.class_names[n.class_index ?? 0] ?? String(n.class_index ?? 0);
        ht.push(escapePlotly(fmtText(knnText("interactiva", "vecino_tooltip"), { orden: order + 1, clase, distancia: n.distance })));
      });
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x: lx,
      y: ly,
      line: { color: NEIGHBOR_LINE, width: 2, dash: "dash" },
      hoverinfo: "skip",
      showlegend: false,
    } as Data);
    traces.push({
      type: "scatter",
      mode: "markers",
      x: hx,
      y: hy,
      text: ht,
      marker: { color: "rgba(0,0,0,0)", size: 17, line: { color: NEIGHBOR_LINE, width: 3 } },
      hovertemplate: "%{text}<extra></extra>",
      showlegend: false,
    } as Data);
  }
  if (options.showConfidence) {
    const rx: (number | null)[] = [];
    const ry: (number | null)[] = [];
    const gx: (number | null)[] = [];
    const gy: (number | null)[] = [];
    for (const p of points) {
      const full = arc(p.point[0], p.point[1], RING_RADIUS_PX, 1, options.scale);
      const part = arc(p.point[0], p.point[1], RING_RADIUS_PX, p.confidence, options.scale);
      rx.push(...full.x, null);
      ry.push(...full.y, null);
      gx.push(...part.x, null);
      gy.push(...part.y, null);
    }
    traces.push({ type: "scatter", mode: "lines", x: rx, y: ry, line: { color: "rgba(76,175,80,0.22)", width: 4 }, hoverinfo: "skip", showlegend: false } as Data);
    traces.push({ type: "scatter", mode: "lines", x: gx, y: gy, line: { color: RING_COLOR, width: 4 }, hoverinfo: "skip", showlegend: false } as Data);
  }
  const px = points.map((p) => p.point[0]);
  const py = points.map((p) => p.point[1]);
  const names = points.map((p) => view.class_names[p.class_index] ?? String(p.class_index));
  // Test points: circle of the predicted class with a gold star (legend: gold diamond).
  traces.push({
    type: "scatter",
    mode: "markers",
    x: px,
    y: py,
    customdata: points.map((p, i) => [i + 1, escapePlotly(names[i] ?? ""), percent(p.confidence), fixed(p.mean_distance, 3)]),
    marker: { color: points.map((p) => classColor(p.class_index, view.colors)), size: 24, line: { color: POINT_BORDER, width: 2.5 } },
    hovertemplate: `Punto %{customdata[0]}: %{customdata[1]} (%{customdata[2]})<br>${escapePlotlyTemplate(fx)}: %{x:.3f}<br>${escapePlotlyTemplate(fy)}: %{y:.3f}<br>Distancia promedio: %{customdata[3]}<extra></extra>`,
    showlegend: false,
  } as Data);
  traces.push({
    type: "scatter",
    mode: "markers",
    x: px,
    y: py,
    marker: { color: STAR_COLOR, symbol: "star", size: 13, line: { color: "#8a6d00", width: 1 } },
    hoverinfo: "skip",
    showlegend: false,
  } as Data);
  // Class name above (bold) and confidence below, around an invisible ring-sized marker.
  traces.push({
    type: "scatter",
    mode: "text",
    x: px,
    y: py.map((y) => y + LABEL_OFFSET_PX * options.scale.dy),
    text: names.map((n) => `<b>${escapePlotly(n)}</b>`),
    textposition: "middle center",
    textfont: { size: 14, color: theme.text },
    hoverinfo: "skip",
    showlegend: false,
  } as Data);
  if (options.showConfidence) {
    traces.push({
      type: "scatter",
      mode: "text",
      x: px,
      y: py.map((y) => y - LABEL_OFFSET_PX * options.scale.dy),
      text: points.map((p) => percent(p.confidence)),
      textposition: "middle center",
      textfont: { size: 13, color: theme.text },
      hoverinfo: "skip",
      showlegend: false,
    } as Data);
  }
  // Legend entry of the test points (Python: gold diamond "Puntos de Prueba").
  traces.push({
    type: "scatter",
    mode: "markers",
    x: [null],
    y: [null],
    name: escapePlotly(knnText("interactiva", "leyenda_prueba")),
    marker: { color: STAR_COLOR, symbol: "diamond", size: 12, line: { color: POINT_BORDER, width: 1.5 } },
    hoverinfo: "skip",
  } as Data);
  return { data: traces, layout };
}

/** Relative luminance (sRGB) of `#rrggbb`. */
export function luminance(hex: string): number {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) {
    return 0;
  }
  const [r, g, b] = [m[1], m[2], m[3]].map((c) => {
    const v = parseInt(c ?? "0", 16) / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * (r ?? 0) + 0.7152 * (g ?? 0) + 0.0722 * (b ?? 0);
}

/** Colour of `VIRIDIS` at `t` (0..1), linear between the samples. */
export function viridisAt(t: number): string {
  const v = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  for (let i = 1; i < VIRIDIS.length; i += 1) {
    const [p1, c1] = VIRIDIS[i] ?? [1, "#fde725"];
    const [p0, c0] = VIRIDIS[i - 1] ?? [0, "#440154"];
    if (v <= p1) {
      const f = p1 > p0 ? (v - p0) / (p1 - p0) : 0;
      const mix = (a: string, b: string, k: number): string => {
        const x = parseInt(a.slice(k, k + 2), 16);
        const y = parseInt(b.slice(k, k + 2), 16);
        return Math.round(x + (y - x) * f)
          .toString(16)
          .padStart(2, "0");
      };
      return `#${mix(c0, c1, 1)}${mix(c0, c1, 3)}${mix(c0, c1, 5)}`;
    }
  }
  return VIRIDIS[VIRIDIS.length - 1]?.[1] ?? "#fde725";
}

/** Text over a heatmap cell: white on dark cells, black on light ones (D-vistosidad). */
export function cellTextColor(t: number): string {
  return luminance(viridisAt(t)) > 0.35 ? "#111111" : "#ffffff";
}

/**
 * C19: `imshow(viridis)` of the mean distances with `{:.2f}` outside the diagonal and,
 * with D67, the within-class means on the diagonal (marked, in italics).
 */
export function distanceHeatmapFigure(view: ClassDistancesView, theme: ChartTheme = LIGHT_THEME): Figure {
  const n = view.classes.length;
  const names = view.class_names.map(escapePlotly);
  const showWithin = view.within.some((w) => w !== null);
  const z = view.matrix.map((row, i) => row.map((v, j) => (i === j && showWithin ? (view.within[i] ?? 0) : v)));
  const flat = z.flat().filter(Number.isFinite);
  const zmax = Math.max(...flat, 0) || 1;
  const metric = view.metric;
  const title = escapePlotly(fmtText(knnText("distancias", "grafico_title"), { metrica: metric })).replace("\n", "<br>");
  const layout = baseLayout(theme, title);
  layout.margin = { l: 110, r: 30, t: 86, b: 110 };
  layout.xaxis = { ...axis(theme, ""), tickangle: -45, type: "category", showgrid: false, ticks: "outside", automargin: true };
  layout.yaxis = { ...axis(theme, ""), autorange: "reversed", type: "category", showgrid: false, ticks: "outside", automargin: true };
  const annotations = [];
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const value = z[i]?.[j] ?? 0;
      const diagonal = i === j;
      if (diagonal && !showWithin) {
        continue;
      }
      annotations.push({
        x: names[j],
        y: names[i],
        text: diagonal ? `<i>${fixed(value, 2)}</i>` : fixed(value, 2),
        showarrow: false,
        font: { size: n > 6 ? 11 : 15, color: cellTextColor(value / zmax) },
      });
    }
  }
  layout.annotations = annotations as Layout["annotations"];
  const hover = z.map((row, i) =>
    row.map((v, j) => `${names[i] ?? ""} ↔ ${names[j] ?? ""}: ${fixed(v, 3)}${i === j && showWithin ? " (dentro de la clase)" : ""}`),
  );
  const data: Data[] = [
    {
      type: "heatmap",
      x: names,
      y: names,
      z,
      zmin: 0,
      zmax,
      colorscale: VIRIDIS,
      xgap: 1,
      ygap: 1,
      text: hover,
      hovertemplate: "%{text}<extra></extra>",
      colorbar: { title: { text: escapePlotly(knnText("distancias", "grafico_colorbar")), side: "right" }, outlinewidth: 0, tickfont: { color: theme.text } },
    } as unknown as Data,
  ];
  return { data, layout };
}

/** Height of the heatmap: 8 × 6 in in Python, more rows for Dígitos. */
export function distanceHeatmapHeight(n: number): number {
  return Math.max(480, Math.min(820, 300 + 52 * n));
}
