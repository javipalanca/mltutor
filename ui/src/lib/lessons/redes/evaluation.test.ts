/// <reference types="node" />
/**
 * F5c (📈 Evaluación and 🎯 Visualización) against the Python renders of the `full` cases of
 * `tests/parity/nn/exact/*.json` (`expected.views`, `expected.diagnostics`, `expected.weights`,
 * `expected.activations_default`): the view tree of the diagnostics, the texts and numbers of
 * 🧠 Pesos and 📉 Activaciones, and the geometry of the map of dead neurons.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cleanup, render, screen, waitFor } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ActivationsView } from "../../bindings/ActivationsView";
import type { NnViewNode } from "../../bindings/NnViewNode";
import type { WeightsView } from "../../bindings/WeightsView";
import { binnedHistogramFigure } from "../../components/ml/nnFigures";
import { DEFAULT_MAP_OPTIONS, drawsConnection, formatMapValue, layoutMap, mapHash, MAP_HEIGHT } from "../../components/ml/networkLayout";
import NnViewTree from "../../components/ml/NnViewTree.svelte";
import { nnText } from "../../content";
import { session } from "../../session.svelte";

const mocks = vi.hoisted(() => ({ weights: null as unknown, activations: null as unknown }));

vi.mock("../../backend", async (importOriginal) => {
  const real = await importOriginal<typeof import("../../backend")>();
  return {
    ...real,
    nnWeights: () => Promise.resolve(mocks.weights),
    nnActivations: () => ({ promise: Promise.resolve(mocks.activations), cancel: () => undefined }),
  };
});

afterEach(() => cleanup());

interface PyNode {
  kind: string;
  text?: string;
  label?: string;
  value?: string;
  children?: PyNode[];
  table?: { columns: string[]; rows: unknown[][] };
}

interface WeightLayer {
  shape: [number, number];
  mean_w: number;
  std_w: number;
  near_zero_pct: number;
  near_zero_pct_raw: number;
  mean_b: number;
  std_b: number;
  n_params: number;
  min_w: number;
  max_w: number;
  min_b: number;
  max_b: number;
}

interface ActLayer {
  name: string;
  units: number;
  dead: number[];
  constant: number[];
  saturation_pct: number;
  mean_act: number;
  std_act: number;
  neuron_means?: number[];
}

interface Case {
  expected: {
    weights: { layers: WeightLayer[]; global_mean: number; global_std: number; global_near_zero_pct: number };
    activations_default: { params: { rows_used: number }; layers: ActLayer[] };
    diagnostics: { view: { children: PyNode[] }; history_tab: PyNode };
    views: Record<string, PyNode>;
  };
}

function load(name: string): Case {
  return JSON.parse(readFileSync(resolve(process.cwd(), `../tests/parity/nn/exact/${name}.json`), "utf8")) as Case;
}

const FULL = ["iris_default", "breast_cancer_default", "diabetes_default"] as const;

function walk(node: PyNode, visit: (n: PyNode) => void): void {
  visit(node);
  for (const child of node.children ?? []) {
    walk(child, visit);
  }
}

function collect(node: PyNode, kind: string): PyNode[] {
  const out: PyNode[] = [];
  walk(node, (n) => {
    if (n.kind === kind) {
      out.push(n);
    }
  });
  return out;
}

const EMPTY_HIST = { edges: [0, 1], counts: [1] };

/** The `WeightsView` the backend sends for the weights of the fixture (D77 names). */
function weightsView(c: Case): WeightsView {
  const w = c.expected.weights;
  const n = w.layers.length;
  return {
    layers: w.layers.map((l, i) => ({
      name: i === n - 1 ? "Salida" : `Oculta ${i + 1}`,
      keras_name: i === n - 1 ? "output_layer" : `dense_${i + 1}`,
      inputs: l.shape[0],
      units: l.shape[1],
      mean_w: l.mean_w,
      std_w: l.std_w,
      near_zero_pct: l.near_zero_pct,
      near_zero_pct_raw: l.near_zero_pct_raw,
      mean_b: l.mean_b,
      std_b: l.std_b,
      n_params: l.n_params,
      min_w: l.min_w,
      max_w: l.max_w,
      min_b: l.min_b,
      max_b: l.max_b,
      weights_hist: EMPTY_HIST,
      biases_hist: EMPTY_HIST,
    })),
    global_mean: w.global_mean,
    global_std: w.global_std,
    global_near_zero_pct: w.global_near_zero_pct,
    verdict: w.global_std < 0.01 ? "too_small" : w.global_std > 2 ? "too_large" : "healthy",
    insights: ["balanced"],
  };
}

function activationsView(c: Case): ActivationsView {
  const a = c.expected.activations_default;
  const units = a.layers.map((l) => l.units);
  const dead = a.layers.reduce((s, l) => s + l.dead.length, 0);
  return {
    rows_used: a.params.rows_used,
    dead_pct: (dead / units.reduce((s, u) => s + u, 0)) * 100,
    layers: a.layers.map((l, i) => ({
      name: `Oculta ${i + 1}`,
      keras_name: l.name,
      units: l.units,
      dead: l.dead,
      constant: l.constant,
      saturation_pct: l.saturation_pct,
      mean_act: l.mean_act,
      std_act: l.std_act,
      neuron_means: l.neuron_means ?? Array.from({ length: l.units }, () => 0),
      hist: EMPTY_HIST,
    })),
    map: {
      layers: [4, ...units, 3],
      neurons: [...units, 3].map((u) => ({
        bias: Array.from({ length: u }, () => 0),
        mean_w: Array.from({ length: u }, () => 0),
        std_w: Array.from({ length: u }, () => 0),
        max_abs_w: Array.from({ length: u }, () => 0),
      })),
    },
  };
}

describe("C23: the diagnostics tree renders as Python drew it", () => {
  for (const name of FULL) {
    it(`${name}: every text, notice and metric of the diagnostics`, () => {
      const c = load(name);
      const nodes = c.expected.diagnostics.view.children as unknown as NnViewNode[];
      const { container } = render(NnViewTree, { props: { nodes, idPrefix: "t" } });
      const text = container.textContent ?? "";
      const root = { kind: "page", children: c.expected.diagnostics.view.children };
      for (const m of collect(root, "metric")) {
        expect(text).toContain(m.label);
        expect(text).toContain(m.value);
      }
      for (const e of collect(root, "expander")) {
        expect(text).toContain(e.label);
      }
      // Notices keep their Markdown: compare the plain words of the first line.
      for (const n of collect(root, "notice")) {
        const words = (n.text ?? "").split("\n")[0]?.replace(/[*_`]/g, "").trim() ?? "";
        expect(text.replace(/\s+/g, " ")).toContain(words.replace(/\s+/g, " ").slice(0, 40));
      }
      expect(container.querySelectorAll("[data-view-row]").length).toBe(collect(root, "row").length);
    });
  }

  it("the 4 cards of 📊 Historial", () => {
    const c = load("iris_default");
    const row = collect(c.expected.diagnostics.history_tab, "row")[0];
    expect(row).toBeDefined();
    const { container } = render(NnViewTree, { props: { nodes: [row as unknown as NnViewNode], idPrefix: "h" } });
    for (const m of collect(row as PyNode, "metric")) {
      expect(container.textContent).toContain(m.label);
      expect(container.textContent).toContain(m.value);
    }
  });
});

describe("C25: 🧠 Análisis de Pesos with the texts and numbers of Python", () => {
  for (const name of FULL) {
    it(`${name}`, async () => {
      const c = load(name);
      mocks.weights = weightsView(c);
      const { default: NnWeights } = await import("../../components/ml/NnWeights.svelte");
      const { container } = render(NnWeights, { props: { modelId: 1, onstale: () => undefined } });
      await waitFor(() => expect(container.querySelector("[data-nn-weights]")?.getAttribute("data-nn-weights")).toBe(String(c.expected.weights.layers.length)));
      const text = container.textContent ?? "";
      const py = c.expected.views["visualizacion_pesos"] as PyNode;
      for (const m of collect(py, "metric")) {
        expect(text, m.label).toContain(m.label);
        expect(text, m.label).toContain(m.value);
      }
      // The captions under the histograms ("Media=… | Std=…") are identical.
      const captions = collect(py, "caption").map((n) => n.text ?? "").filter((t) => t.startsWith("Media="));
      expect(captions.length).toBe(2 * c.expected.weights.layers.length);
      for (const caption of captions) {
        expect(text).toContain(caption);
      }
      // D77: "Capa 2" of Python is "Oculta 1" here; the shapes are the same.
      const loaded = collect(py, "caption").map((n) => n.text ?? "").filter((t) => t.startsWith("✅"));
      loaded.forEach((caption, i) => {
        const shapes = caption.slice(caption.indexOf(":"));
        expect(text).toContain(shapes);
        expect(text).toContain(i === loaded.length - 1 ? "✅ Salida" : `✅ Oculta ${i + 1}`);
      });
      const table = collect(py, "table")[0]?.table;
      for (const column of table?.columns ?? []) {
        expect(screen.getAllByText(column).length).toBeGreaterThan(0);
      }
      for (const notice of collect(py, "notice").slice(2)) {
        expect(text).toContain((notice.text ?? "").replace(/\*\*/g, ""));
      }
    });
  }
});

describe("C26: 📉 Análisis de Activaciones with the texts and numbers of Python", () => {
  for (const name of FULL) {
    it(`${name}`, async () => {
      const c = load(name);
      mocks.activations = activationsView(c);
      c.expected.activations_default.layers.forEach((_, i) => session.setExpander(`nn-act-capa-${i}`, true));
      const { default: NnActivations } = await import("../../components/ml/NnActivations.svelte");
      const { container } = render(NnActivations, { props: { modelId: 1, onstale: () => undefined } });
      await waitFor(() => expect(container.querySelector("[data-network-map]")).not.toBeNull());
      const text = container.textContent ?? "";
      const py = c.expected.views["visualizacion_activaciones"] as PyNode;
      // The verdict on dead neurons ("✅ Neuronas muertas bajo control (5.0%)").
      const verdict = collect(py, "notice").find((n) => (n.text ?? "").includes("muertas"));
      expect(verdict).toBeDefined();
      expect(text).toContain(verdict?.text ?? "");
      // "Muertas" / "Constantes" of every layer, as in the expanders of Python.
      const metrics = collect(py, "metric");
      for (const m of metrics) {
        expect(text).toContain(m.label);
      }
      const dead = metrics.filter((m) => m.label === "Muertas").map((m) => m.value);
      expect(dead).toEqual(c.expected.activations_default.layers.map((l) => String(l.dead.length)));
      const map = container.querySelector("[data-network-map]");
      expect(map?.getAttribute("data-dead")).toBe(String(c.expected.activations_default.layers.reduce((s, l) => s + l.dead.length, 0)));
      for (const column of collect(py, "table")[0]?.table?.columns ?? []) {
        expect(screen.getAllByText(column).length).toBeGreaterThan(0);
      }
    });
  }
});

describe("map of dead neurons (C24, map mode)", () => {
  it("hashes uniformly in [0, 1) (Python's canvas never passed 0.5)", () => {
    const python = (l: number, i: number, j = 0): number => {
      let x = l * 374761393 + i * 668265263 + j * 2147483647;
      x = (x ^ (x >> 13)) * 1274126177;
      x = x ^ (x >> 16);
      return (x >>> 0) / 4294967295;
    };
    let pyMax = 0;
    const values: number[] = [];
    for (let i = 0; i < 64; i++) {
      for (let j = 0; j < 64; j++) {
        pyMax = Math.max(pyMax, python(1, i, j));
        values.push(mapHash(1, i, j));
      }
    }
    expect(pyMax).toBeLessThan(0.5);
    expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...values)).toBeLessThan(1);
    expect(values.filter((v) => v >= 0.5).length / values.length).toBeGreaterThan(0.45);
    expect(mapHash(2, 7, 3)).toBe(mapHash(2, 7, 3));
  });

  it("keeps big layers inside the allowed sub-columns and the canvas", () => {
    const width = 1000;
    const layout = layoutMap([64, 256, 256, 256, 256, 256, 10], width, DEFAULT_MAP_OPTIONS);
    const gap = layout.layerGap;
    layout.nodes.forEach((layer, l) => {
      const base = layout.marginX + l * gap;
      const xs = new Set(layer.map((n) => Math.round((n.x - base) / Math.min(32, gap * 0.25) + 100)));
      for (const node of layer) {
        expect(node.y).toBeGreaterThanOrEqual(49);
        expect(node.y).toBeLessThanOrEqual(MAP_HEIGHT - 49);
        expect(Math.abs(node.x - base)).toBeLessThanOrEqual(Math.min(32, gap * 0.25) * 1.5 + DEFAULT_MAP_OPTIONS.jitter / 2 + 1e-9);
      }
      expect(xs.size).toBeGreaterThan(0);
    });
    // Python: radius min(14, max(5, 460/(n+6))).
    expect(layout.nodes[1]?.[0]?.r).toBe(5);
    expect(layout.nodes[0]?.[0]?.r).toBeCloseTo(Math.min(14, Math.max(5, 460 / 70)), 12);
  });

  it("draws about the requested share of connections", () => {
    let drawn = 0;
    for (let i = 0; i < 64; i++) {
      for (let j = 0; j < 64; j++) {
        drawn += drawsConnection(1, i, j, 70) ? 1 : 0;
      }
    }
    expect(drawn / 4096).toBeGreaterThan(0.6);
    expect(drawn / 4096).toBeLessThan(0.8);
    expect(drawsConnection(1, 5, 7, 100)).toBe(true);
  });

  it("formats the tooltip values as Python's formatVal", () => {
    expect(formatMapValue(0.123456)).toBe("0.1235");
    expect(formatMapValue(0.00001234)).toBe("1.23e-5");
    expect(formatMapValue(-2)).toBe("-2.0000");
  });
});

describe("histograms of 🧠 Pesos", () => {
  it("bars from the Rust bins, mean line and log scale", () => {
    const fig = binnedHistogramFigure(
      { edges: [-1, 0, 1, 2], counts: [1, 4, 2] },
      { title: "t", xLabel: nnText("pesos", "eje_peso"), yLabel: nnText("pesos", "eje_frecuencia"), color: "#1f77b4", mean: 0.3, log: true },
    );
    const bars = fig.data[0] as { x: number[]; y: number[]; marker: { color: string } };
    expect(bars.x).toEqual([-0.5, 0.5, 1.5]);
    expect(bars.y).toEqual([1, 4, 2]);
    expect(bars.marker.color).toBe("#1f77b4");
    expect((fig.layout.yaxis as { type?: string }).type).toBe("log");
    expect(fig.layout.shapes?.length).toBe(2);
    expect(fig.layout.bargap).toBe(0.02);
  });
});

describe("🧠 surface with its own colour scales (D78)", () => {
  it("scales the surface to the predictions and the points to the targets", async () => {
    const { surfaceFigure } = await import("../../components/ml/figures");
    const grid = {
      x: [0, 1],
      y: [0, 1],
      values: { kind: "values" as const, z: [0.78, 0.8, 0.79, 0.81], min: 0.78, max: 0.81 },
      notices: [],
      points: { rows: 2, cols: 2, values: [0, 0, 1, 1] },
      point_target: [25, 346],
      point_codes: null,
      feature_names: ["age", "sex"] as [string, string],
      class_names: null,
      show_2d_warning: true,
      fixed: [0, 0],
    };
    const fig = surfaceFigure(grid, undefined, false, { surface: "Valor predicho", points: "Valor real" });
    const [contour, points] = fig.data as [{ zmin: number; zmax: number }, { marker: { cmin: number; cmax: number; showscale: boolean } }];
    expect([contour.zmin, contour.zmax]).toEqual([0.78, 0.81]);
    expect([points.marker.cmin, points.marker.cmax, points.marker.showscale]).toEqual([25, 346, true]);
    const shared = surfaceFigure(grid, undefined, false) as unknown as { data: [{ zmin: number; zmax: number }] };
    expect([shared.data[0].zmin, shared.data[0].zmax]).toEqual([0.78, 346]);
  });
});
