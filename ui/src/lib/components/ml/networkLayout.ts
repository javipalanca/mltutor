/**
 * C24 (D80): geometry of the network diagram of 🏗️ Arquitectura, the same as Python's
 * canvas (`viz/nn.py::create_neural_network_visualization`): margin 40, layers evenly
 * spaced, neurons at `margin + espacio·(j + 1)` with `espacio = (alto − 2·margin − 60)/(n + 1)`,
 * radius `min(20, ancho/(capas·8), max(8, (alto − 80)/(máx_neuronas·3)))` and the labels
 * 25 px above the bottom. Improvements: a layer with more than [`MAX_DRAWN`] neurons draws
 * the first and last [`HALF`] with "⋮" between them (Python drew 256 overlapping circles and
 * 65 536 lines), and the connections of each pair of layers are a single path whose opacity
 * falls with the number of lines.
 */

export const COLORS = {
  input: "#4ECDC4",
  hidden: "#45B7D1",
  output: "#FF6B6B",
  connection: "#BDC3C7",
  border: "#2C3E50",
  text: "#2C3E50",
} as const;

/** Layers with more neurons are summarised. */
export const MAX_DRAWN = 24;
export const HALF = MAX_DRAWN / 2;
const MARGIN = 40;
const LABEL_SPACE = 60;

export type LayerKind = "input" | "hidden" | "output";

export interface NodeLayout {
  /** Index of the neuron in its layer. */
  index: number;
  y: number;
}

export interface LayerLayout {
  kind: LayerKind;
  x: number;
  neurons: number;
  nodes: NodeLayout[];
  /** y of the "⋮" of a summarised layer, and how many neurons are not drawn. */
  gapY: number | null;
  hidden: number;
}

export interface NetworkLayout {
  width: number;
  height: number;
  radius: number;
  layers: LayerLayout[];
  /** One SVG path per pair of consecutive layers, with its opacity. */
  links: { d: string; opacity: number; lines: number }[];
  labelY: number;
  fontSize: number;
}

/**
 * Height of the canvas: Python's 2:1 with at least 300 px, and taller when a layer draws
 * many neurons (Python overlapped them); at most 460.
 */
export function diagramHeight(width: number, sizes: readonly number[] = []): number {
  const drawn = Math.max(1, ...sizes.map(slots));
  return Math.round(Math.min(460, Math.max(300, width / 2, 140 + drawn * 13)));
}

/** Slots drawn for a layer of `n` neurons (the summary takes one slot for "⋮"). */
export function slots(n: number): number {
  return n > MAX_DRAWN ? MAX_DRAWN + 1 : n;
}

function slotY(slot: number, count: number, height: number): number {
  const available = height - 2 * MARGIN - LABEL_SPACE;
  if (count <= 1) {
    return MARGIN + available / 2;
  }
  return MARGIN + (available / (count + 1)) * (slot + 1);
}

/** Opacity of the links of a pair of layers: Python's 0.3, lower when there are many. */
export function linkOpacity(lines: number): number {
  return Math.max(0.05, Math.min(0.3, 0.3 * Math.sqrt(100 / Math.max(1, lines))));
}

const r1 = (v: number): number => Math.round(v * 10) / 10;

export function layoutNetwork(sizes: readonly number[], width: number, height = diagramHeight(width, sizes)): NetworkLayout {
  const n = sizes.length;
  const layerWidth = n > 1 ? (width - 2 * MARGIN) / (n - 1) : 0;
  const drawnMax = Math.max(1, ...sizes.map(slots));
  const maxRadius = Math.min(20, width / (Math.max(1, n) * 8));
  // Python's radius, capped so that the circles of the fullest column do not overlap.
  const spacing = (height - 2 * MARGIN - LABEL_SPACE) / (drawnMax + 1);
  const radius = Math.min(maxRadius, Math.max(8, (height - 2 * MARGIN) / (drawnMax * 3)), Math.max(3, spacing * 0.42));
  const layers: LayerLayout[] = sizes.map((neurons, i) => {
    const kind: LayerKind = i === 0 ? "input" : i === n - 1 ? "output" : "hidden";
    const x = n > 1 ? MARGIN + i * layerWidth : width / 2;
    const count = slots(neurons);
    const nodes: NodeLayout[] = [];
    let gapY: number | null = null;
    if (neurons > MAX_DRAWN) {
      for (let s = 0; s < count; s++) {
        const y = slotY(s, count, height);
        if (s < HALF) {
          nodes.push({ index: s, y });
        } else if (s === HALF) {
          gapY = y;
        } else {
          nodes.push({ index: neurons - (count - s), y });
        }
      }
    } else {
      for (let j = 0; j < neurons; j++) {
        nodes.push({ index: j, y: slotY(j, count, height) });
      }
    }
    return { kind, x, neurons, nodes, gapY, hidden: Math.max(0, neurons - MAX_DRAWN) };
  });
  const links = layers.slice(0, -1).map((from, i) => {
    const to = layers[i + 1] as LayerLayout;
    const parts: string[] = [];
    for (const a of from.nodes) {
      for (const b of to.nodes) {
        parts.push(`M${r1(from.x)} ${r1(a.y)}L${r1(to.x)} ${r1(b.y)}`);
      }
    }
    const lines = from.neurons * to.neurons;
    return { d: parts.join(""), opacity: linkOpacity(parts.length), lines };
  });
  return {
    width,
    height,
    radius,
    layers,
    links,
    labelY: height - 25,
    fontSize: Math.max(10, Math.min(14, width / 50)),
  };
}

// --- C26: the map of dead (💀) and constant (😴) neurons of 📉 Activaciones ---

/** Colours of the map canvas of `show_layer_activations_tab`. */
export const MAP_COLORS = {
  input: "#4ECDC4",
  hidden: "#1976d2",
  output: "#FF6B6B",
  connection: "#777777",
  border: "#1b1f23",
} as const;

/** Height of Python's canvas. */
export const MAP_HEIGHT = 460;

/**
 * Deterministic hash in [0, 1) of a layer, a neuron and a connection: the jitter of the nodes
 * and the subset of connections drawn. **Fix:** Python's canvas used
 * `((h ^ (h >> 13)) * 1274126177) ^ …` on JavaScript doubles; the product exceeds 2^53 and the
 * result never reached 0.5, so "% conexiones dibujadas" ≥ 50 drew every line (the default is
 * 70) and the jitter always moved the nodes to the left. Same mixing constants, in 32-bit
 * integer arithmetic (`Math.imul`).
 */
export function mapHash(l: number, i: number, j = 0): number {
  let h = (Math.imul(l, 374761393) + Math.imul(i, 668265263) + Math.imul(j, 2147483647)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

export interface MapOptions {
  /** "Dispersión nodos", 0..50 px. */
  jitter: number;
  /** "% conexiones dibujadas", 10..100. */
  connectionPct: number;
  /** "Radio adaptativo". */
  adaptiveRadius: boolean;
  /** "Agrupar capas grandes". */
  group: boolean;
  /** "Espaciado mínimo vertical", 12..50. */
  minSpacing: number;
  /** "Máx subcolumnas capa", 1..6. */
  maxColumns: number;
}

export const DEFAULT_MAP_OPTIONS: MapOptions = { jitter: 18, connectionPct: 70, adaptiveRadius: true, group: true, minSpacing: 22, maxColumns: 3 };

export interface MapNode {
  x: number;
  y: number;
  r: number;
}

export interface MapLayout {
  width: number;
  height: number;
  marginX: number;
  layerGap: number;
  /** Per layer, per neuron. */
  nodes: MapNode[][];
}

/**
 * Coordinates of every neuron (Python's `draw()`): margins 50 / clamp(5 % of the width,
 * 40, 80), layers evenly spaced, radius `min(14, max(5, alto/(n+6)))` or 10, and with
 * "Agrupar" up to `maxColumns` sub-columns of `floor(alto_útil/espaciado)` neurons
 * separated by `min(32, hueco·0.25)`. **Fix:** Python capped the number of columns but not
 * the neurons per column, so the neurons that did not fit were drawn in extra columns
 * beyond the cap; here they are spread evenly over the allowed columns.
 */
export function layoutMap(arch: readonly number[], width: number, options: MapOptions, height = MAP_HEIGHT): MapLayout {
  const marginY = 50;
  const marginX = Math.min(80, Math.max(40, width * 0.05));
  const layerGap = arch.length > 1 ? (width - 2 * marginX) / (arch.length - 1) : 0;
  const radius = arch.map((n) => (options.adaptiveRadius ? Math.min(14, Math.max(5, height / (n + 6))) : 10));
  const jitter = (l: number, i: number): number => (options.jitter ? (mapHash(l, i) - 0.5) * options.jitter : 0);
  const nodes = arch.map((n, l) => {
    const baseX = marginX + l * layerGap;
    const r = radius[l] ?? 10;
    const out: MapNode[] = [];
    if (options.group && n > 0) {
      const available = height - 2 * marginY;
      const maxPerCol = Math.max(1, Math.floor(available / options.minSpacing));
      const cols = Math.min(Math.ceil(n / maxPerCol), Math.max(1, options.maxColumns));
      const perCol = Math.max(maxPerCol, Math.ceil(n / cols));
      const offset = Math.min(32, layerGap * 0.25);
      for (let i = 0; i < n; i++) {
        const col = Math.floor(i / perCol);
        const row = i % perCol;
        const inCol = col === cols - 1 ? n - col * perCol : perCol;
        const y = inCol === 1 ? height / 2 : marginY + row * ((height - 2 * marginY) / (inCol - 1));
        out.push({ x: baseX + (col - (cols - 1) / 2) * offset + jitter(l, i), y, r });
      }
    } else {
      for (let i = 0; i < n; i++) {
        const y = n === 1 ? height / 2 : marginY + i * ((height - 2 * marginY) / (n - 1));
        out.push({ x: baseX + jitter(l, i), y, r });
      }
    }
    return out;
  });
  return { width, height, marginX, layerGap, nodes };
}

/** Whether the connection `i → j` between layers `l` and `l + 1` is drawn (`hash ≤ pct`). */
export function drawsConnection(l: number, i: number, j: number, pct: number): boolean {
  return pct >= 99.9 || mapHash(l, i, j) <= pct / 100;
}

/** Python's `formatVal` of the tooltip: exponential below 1e-4, otherwise 4 decimals. */
export function formatMapValue(v: number): string {
  return Math.abs(v) < 1e-4 ? v.toExponential(2) : v.toFixed(4);
}

/** The neuron under the pointer (`(r + 4)²` as in Python), or `null`. */
export function hitNeuron(layout: MapLayout, x: number, y: number): { layer: number; index: number } | null {
  for (let l = 0; l < layout.nodes.length; l++) {
    const layer = layout.nodes[l] ?? [];
    for (let i = 0; i < layer.length; i++) {
      const c = layer[i];
      if (!c) {
        continue;
      }
      const dx = x - c.x;
      const dy = y - c.y;
      if (dx * dx + dy * dy <= (c.r + 4) * (c.r + 4)) {
        return { layer: l, index: i };
      }
    }
  }
  return null;
}
