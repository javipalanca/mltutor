/**
 * C13 (fase3-arboles.md, D41/D42): geometry of the decision tree diagram. Pure functions
 * (DTO → boxes, edges and a standalone SVG) so they can be tested without a browser;
 * `TreeDiagram.svelte` draws them and handles zoom, panning and the tooltip.
 *
 * The positions are the Buchheim coordinates of `plot_tree` (`(x + 0.5) / max_x`, one
 * level per depth), scaled so that the widest box fits one unit: with a fixed, legible
 * font the boxes never overlap, whatever the size of the tree. Texts and colours are the
 * ones of `plot_tree` (computed in Rust).
 */
import type { TreeDiagram } from "../../bindings/TreeDiagram";
import type { TreeDiagramNode } from "../../bindings/TreeDiagramNode";

export const FONT_SIZE = 12;
export const FONT_FAMILY = "Inter Variable, Inter, Helvetica, Arial, sans-serif";
export const LINE_HEIGHT = 15;
const PAD_X = 9;
const PAD_Y = 7;
/** Horizontal gap between the widest boxes of two neighbouring units. */
const GAP_X = 14;
/** Vertical room for the arrows (and the True/False labels of the root). */
const GAP_Y = 46;
export const MARGIN = 16;

export interface BoxLayout {
  node: TreeDiagramNode;
  /** Top-left corner and size, in diagram units (px at zoom 1). */
  x: number;
  y: number;
  w: number;
  h: number;
  cx: number;
}

export interface EdgeLayout {
  from: number;
  to: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** "True" / "False" on the two edges of the root, as `plot_tree`. */
  label: "True" | "False" | null;
}

export interface DiagramLayout {
  boxes: BoxLayout[];
  edges: EdgeLayout[];
  width: number;
  height: number;
}

export type Measure = (text: string) => number;

/** Width estimate without a canvas (jsdom): Inter at 12 px averages ~0.56 em per glyph. */
export const estimateWidth: Measure = (text) => [...text].length * FONT_SIZE * 0.56;

let canvasMeasure: Measure | null = null;

/** Text width with the real font (canvas `measureText`), or the estimate. */
export function textMeasure(): Measure {
  if (canvasMeasure) {
    return canvasMeasure;
  }
  const context = typeof document === "undefined" ? null : document.createElement("canvas").getContext("2d");
  if (!context) {
    return estimateWidth;
  }
  context.font = `${FONT_SIZE}px ${FONT_FAMILY}`;
  const cache = new Map<string, number>();
  canvasMeasure = (text) => {
    let width = cache.get(text);
    if (width === undefined) {
      width = context.measureText(text).width;
      cache.set(text, width);
    }
    return width;
  };
  return canvasMeasure;
}

export function layoutTree(diagram: TreeDiagram, measure: Measure = estimateWidth): DiagramLayout {
  const sizes = diagram.nodes.map((node) => ({
    w: Math.ceil(Math.max(30, ...node.lines.map(measure)) + 2 * PAD_X),
    h: node.lines.length * LINE_HEIGHT + 2 * PAD_Y,
  }));
  const maxW = Math.max(40, ...sizes.map((s) => s.w));
  const maxH = Math.max(20, ...sizes.map((s) => s.h));
  const unitX = maxW + GAP_X;
  const unitY = maxH + GAP_Y;
  const boxes = diagram.nodes.map((node, i): BoxLayout => {
    const { w, h } = sizes[i] ?? { w: maxW, h: maxH };
    const cx = MARGIN + (node.x + 0.5) * unitX;
    return { node, x: cx - w / 2, y: MARGIN + node.depth * unitY, w, h, cx };
  });
  const edges: EdgeLayout[] = [];
  for (const box of boxes) {
    const parent = box.node.parent === null ? undefined : boxes[box.node.parent];
    if (!parent) {
      continue;
    }
    edges.push({
      from: parent.node.id,
      to: box.node.id,
      x1: parent.cx,
      y1: parent.y + parent.h,
      x2: box.cx,
      y2: box.y,
      label: parent.node.parent === null ? (box.node.is_left ? "True" : "False") : null,
    });
  }
  const width = Math.max(...boxes.map((b) => b.x + b.w), 0) + MARGIN;
  const height = Math.max(...boxes.map((b) => b.y + b.h), 0) + MARGIN;
  return { boxes, edges, width, height };
}

/** Nodes from the root to `leaf` (path highlighted from 🔮 Predicciones). */
export function pathTo(diagram: TreeDiagram, leaf: number | null): Set<number> {
  const out = new Set<number>();
  let node = leaf === null ? undefined : diagram.nodes[leaf];
  while (node) {
    out.add(node.id);
    node = node.parent === null ? undefined : diagram.nodes[node.parent];
  }
  return out;
}

/** Below this scale the whole tree is unreadable: the first view zooms on the root. */
export const MIN_READABLE_SCALE = 0.45;

/**
 * First view: the whole tree if it is readable at that scale; otherwise (deep trees, D41)
 * the root at the top centre at a legible scale, so the student starts reading from it.
 */
export function initialTransform(layout: DiagramLayout, width: number, height: number): { k: number; x: number; y: number } {
  const fit = fitTransform(layout, width, height);
  if (fit.k >= MIN_READABLE_SCALE) {
    return fit;
  }
  const k = 0.8;
  const root = layout.boxes[0];
  const cx = root ? root.cx : layout.width / 2;
  return { k, x: width / 2 - cx * k, y: 8 };
}

/** Scale and translation that fit the whole diagram in a `width × height` viewport. */
export function fitTransform(layout: DiagramLayout, width: number, height: number): { k: number; x: number; y: number } {
  const k = Math.min(width / layout.width, height / layout.height, 1.5);
  return { k, x: (width - layout.width * k) / 2, y: Math.max(0, (height - layout.height * k) / 2) };
}

function xml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

export interface DrawColors {
  edge: string;
  text: string;
  background: string;
}

export const PRINT_COLORS: DrawColors = { edge: "#000000", text: "#000000", background: "#ffffff" };

/** Standalone SVG of the whole diagram (D42: `arbol_decision.svg`). */
export function diagramSvg(layout: DiagramLayout, colors: DrawColors = PRINT_COLORS): string {
  const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${Math.ceil(layout.width)}" height="${Math.ceil(layout.height)}" viewBox="0 0 ${Math.ceil(layout.width)} ${Math.ceil(layout.height)}" font-family="${xml(FONT_FAMILY)}" font-size="${FONT_SIZE}">`,
    `<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${colors.edge}"/></marker></defs>`,
    `<rect width="100%" height="100%" fill="${colors.background}"/>`,
  ];
  for (const e of layout.edges) {
    parts.push(`<line x1="${e.x1.toFixed(1)}" y1="${e.y1.toFixed(1)}" x2="${e.x2.toFixed(1)}" y2="${e.y2.toFixed(1)}" stroke="${colors.edge}" stroke-width="1.2" marker-end="url(#arrow)"/>`);
    if (e.label) {
      const mx = (e.x1 + e.x2) / 2;
      const my = (e.y1 + e.y2) / 2;
      const anchor = e.label === "True" ? "end" : "start";
      const dx = e.label === "True" ? -6 : 6;
      parts.push(`<text x="${(mx + dx).toFixed(1)}" y="${my.toFixed(1)}" text-anchor="${anchor}" dominant-baseline="middle" fill="${colors.text}">${e.label}</text>`);
    }
  }
  for (const b of layout.boxes) {
    parts.push(
      `<g><rect x="${b.x.toFixed(1)}" y="${b.y.toFixed(1)}" width="${b.w}" height="${b.h}" rx="6" fill="${xml(b.node.fill)}" stroke="#000000" stroke-width="1"/>`,
    );
    b.node.lines.forEach((line, i) => {
      parts.push(
        `<text x="${b.cx.toFixed(1)}" y="${(b.y + PAD_Y + (i + 0.5) * LINE_HEIGHT).toFixed(1)}" text-anchor="middle" dominant-baseline="central" fill="#000000">${xml(line)}</text>`,
      );
    });
    parts.push("</g>");
  }
  parts.push("</svg>");
  return parts.join("\n");
}

/** Text baseline of line `i` of a box (shared by the SVG and the canvas). */
export function lineY(box: BoxLayout, i: number): number {
  return box.y + PAD_Y + (i + 0.5) * LINE_HEIGHT;
}

/** Longest side and area of a canvas that WebKit/WebView2 still encode as a PNG. */
export const PNG_MAX_SIDE = 16000;
export const PNG_MAX_AREA = 120_000_000;
/** Below this scale the 12 px texts of the boxes become unreadable in the PNG. */
export const PNG_MIN_SCALE = 0.5;

/**
 * Largest scale ≤ `wanted` whose canvas stays within the limits of WebKit/WebView2. No
 * floor: a floor would break `PNG_MAX_SIDE`; huge trees get [`pngTooLarge`] instead.
 */
export function pngScale(layout: DiagramLayout, wanted = 3): number {
  const bySide = Math.min(PNG_MAX_SIDE / layout.width, PNG_MAX_SIDE / layout.height);
  const byArea = Math.sqrt(PNG_MAX_AREA / (layout.width * layout.height));
  return Math.min(wanted, bySide, byArea);
}

/** The PNG would need a scale below `PNG_MIN_SCALE`: offer only the SVG. */
export function pngTooLarge(layout: DiagramLayout): boolean {
  return !(pngScale(layout) >= PNG_MIN_SCALE);
}

/** Draws the diagram on a canvas (PNG download, `arbol_decision.png`). */
export function drawDiagram(context: CanvasRenderingContext2D, layout: DiagramLayout, scale: number): void {
  context.save();
  context.scale(scale, scale);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, layout.width, layout.height);
  context.strokeStyle = "#000000";
  context.fillStyle = "#000000";
  context.lineWidth = 1.2;
  context.font = `${FONT_SIZE}px ${FONT_FAMILY}`;
  for (const e of layout.edges) {
    context.beginPath();
    context.moveTo(e.x1, e.y1);
    context.lineTo(e.x2, e.y2);
    context.stroke();
    const angle = Math.atan2(e.y2 - e.y1, e.x2 - e.x1);
    context.beginPath();
    context.moveTo(e.x2, e.y2);
    context.lineTo(e.x2 - 8 * Math.cos(angle - 0.4), e.y2 - 8 * Math.sin(angle - 0.4));
    context.lineTo(e.x2 - 8 * Math.cos(angle + 0.4), e.y2 - 8 * Math.sin(angle + 0.4));
    context.closePath();
    context.fill();
    if (e.label) {
      context.textAlign = e.label === "True" ? "right" : "left";
      context.textBaseline = "middle";
      context.fillText(e.label, (e.x1 + e.x2) / 2 + (e.label === "True" ? -6 : 6), (e.y1 + e.y2) / 2);
    }
  }
  context.textAlign = "center";
  context.textBaseline = "middle";
  for (const b of layout.boxes) {
    context.beginPath();
    context.roundRect(b.x, b.y, b.w, b.h, 6);
    context.fillStyle = b.node.fill;
    context.fill();
    context.lineWidth = 1;
    context.strokeStyle = "#000000";
    context.stroke();
    context.fillStyle = "#000000";
    b.node.lines.forEach((line, i) => context.fillText(line, b.cx, lineY(b, i)));
  }
  context.restore();
}
