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
