/// <reference types="node" />
/**
 * C10, C13, C14 and C16 (fase3-arboles.md) against the golden fixtures of scikit-learn
 * (`tests/parity/trees/*.json`): number of boxes, colours and texts of `plot_tree`, the
 * Buchheim positions, the importance bars and the lines of the decision path.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cleanup, render } from "@testing-library/svelte";
import { afterEach, describe, expect, it } from "vitest";
import type { DecisionPath } from "../../bindings/DecisionPath";
import type { FeatureImportanceView } from "../../bindings/FeatureImportanceView";
import type { TreeDiagram as Diagram } from "../../bindings/TreeDiagram";
import type { TreeDiagramNode } from "../../bindings/TreeDiagramNode";
import DecisionPathList from "./DecisionPathList.svelte";
import FeatureImportanceChart from "./FeatureImportanceChart.svelte";
import { importanceFigure, importanceTable, viridisColors } from "./importance";
import TreeDiagram from "./TreeDiagram.svelte";
import TreeRulesText from "./TreeRulesText.svelte";
import { diagramSvg, fitTransform, initialTransform, layoutTree, MIN_READABLE_SCALE, pathTo, pngScale } from "./treeLayout";

afterEach(() => cleanup());

interface Box {
  text: string;
  xy_axes: [number, number];
  facecolor: string;
}

interface Fixture {
  expected: {
    feature_names: string[];
    class_names: string[] | null;
    feature_importances: number[];
    tree: {
      children_left: number[];
      children_right: number[];
      feature: number[];
      threshold: number[];
      impurity: number[];
      n_node_samples: number[];
      value: number[][] | number[][][];
      max_depth: number;
      node_count: number;
      n_leaves: number;
    };
    plot_tree: { boxes_in_drawing_order: Box[]; edge_labels: { text: string }[] };
    prediction_form: { default: { path: { leaf: number; steps: { node: number; feature: number; value_float32: number; threshold: number; goes_left: boolean }[] } } };
    export_text: { view_text: string };
  };
}

function fixture(name: string): Fixture {
  // vitest runs in `ui/`.
  const path = resolve(process.cwd(), "../tests/parity/trees", `${name}.json`);
  return JSON.parse(readFileSync(path, "utf8")) as Fixture;
}

/** The DTO the backend builds (`tree_diagram`): Buchheim x recovered from `xy_axes`. */
function diagramOf(f: Fixture): Diagram {
  const t = f.expected.tree;
  const boxes = f.expected.plot_tree.boxes_in_drawing_order;
  const maxXy = Math.max(...boxes.map((b) => b.xy_axes[0]));
  const maxX = 0.5 / (1 - maxXy);
  const depth: number[] = new Array(t.node_count).fill(0);
  const parent: (number | null)[] = new Array(t.node_count).fill(null);
  const isLeft: boolean[] = new Array(t.node_count).fill(false);
  for (let n = 0; n < t.node_count; n += 1) {
    for (const [child, left] of [
      [t.children_left[n] ?? -1, true],
      [t.children_right[n] ?? -1, false],
    ] as const) {
      if (child >= 0) {
        depth[child] = (depth[n] ?? 0) + 1;
        parent[child] = n;
        isLeft[child] = left;
      }
    }
  }
  const nodes = boxes.map((b, n): TreeDiagramNode => {
    const leaf = (t.children_left[n] ?? -1) < 0;
    const raw = t.value[n] as number[] | number[][];
    const value = (Array.isArray(raw[0]) ? raw[0] : raw) as number[];
    return {
      id: n,
      parent: parent[n] ?? null,
      is_left: isLeft[n] ?? false,
      depth: depth[n] ?? 0,
      x: b.xy_axes[0] * maxX - 0.5,
      y: depth[n] ?? 0,
      lines: b.text.split("\n"),
      fill: b.facecolor,
      leaf,
      feature: leaf ? null : (t.feature[n] ?? null),
      threshold: leaf ? null : (t.threshold[n] ?? null),
      impurity: t.impurity[n] ?? 0,
      n_samples: t.n_node_samples[n] ?? 0,
      value,
      value_counts: null,
      class_index: null,
    };
  });
  return {
    nodes,
    max_x: maxX,
    max_y: t.max_depth + 1,
    criterion: "gini",
    summary: { node_count: t.node_count, n_leaves: t.n_leaves, max_depth: t.max_depth, n_features: f.expected.feature_names.length },
    class_names: f.expected.class_names,
  };
}

function overlaps(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }): boolean {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

describe("TreeDiagram layout (C13)", () => {
  it("keeps the boxes, texts, colours and order of plot_tree", () => {
    const f = fixture("iris_gini_d3");
    const layout = layoutTree(diagramOf(f));
    const boxes = f.expected.plot_tree.boxes_in_drawing_order;
    expect(layout.boxes).toHaveLength(9);
    layout.boxes.forEach((b, i) => {
      expect(b.node.fill).toBe(boxes[i]?.facecolor);
      expect(b.node.lines.join("\n")).toBe(boxes[i]?.text);
    });
    expect(layout.boxes[0]?.node.lines[0]).toBe("petal length (cm) <= 2.45");
    expect(layout.boxes[0]?.node.fill).toBe("#ffffff");
    // True/False only on the two edges of the root, as plot_tree.
    expect(layout.edges.filter((e) => e.label).map((e) => e.label)).toEqual(["True", "False"]);
    expect(f.expected.plot_tree.edge_labels.map((e) => e.text.trim())).toEqual(["True", "False"]);
    // The x order of Buchheim is the order of plot_tree.
    const order = (xs: number[]) => xs.map((_, i) => i).sort((a, b) => (xs[a] ?? 0) - (xs[b] ?? 0));
    expect(order(layout.boxes.map((b) => b.cx))).toEqual(order(boxes.map((b) => b.xy_axes[0])));
  });

  it("never overlaps boxes, even with 407 nodes (California d8)", () => {
    const f = fixture("california_housing_squared_error_d8_mss10");
    const diagram = diagramOf(f);
    const started = performance.now();
    const layout = layoutTree(diagram);
    const ms = performance.now() - started;
    expect(layout.boxes).toHaveLength(407);
    const byDepth = new Map<number, typeof layout.boxes>();
    for (const b of layout.boxes) {
      byDepth.set(b.node.depth, [...(byDepth.get(b.node.depth) ?? []), b]);
    }
    for (const level of byDepth.values()) {
      const sorted = [...level].sort((a, b) => a.x - b.x);
      for (let i = 1; i < sorted.length; i += 1) {
        expect(overlaps(sorted[i - 1]!, sorted[i]!)).toBe(false);
      }
    }
    expect(ms).toBeLessThan(200);
    const fit = fitTransform(layout, 900, 640);
    expect(layout.width * fit.k).toBeLessThanOrEqual(900 + 1e-6);
    // Fitting 407 nodes would be unreadable: the first view zooms on the root (D41).
    expect(fit.k).toBeLessThan(MIN_READABLE_SCALE);
    const first = initialTransform(layout, 900, 640);
    expect(first.k).toBeGreaterThanOrEqual(MIN_READABLE_SCALE);
    expect((layout.boxes[0]?.cx ?? 0) * first.k + first.x).toBeCloseTo(450, 6);
    expect(initialTransform(layoutTree(diagramOf(fixture("iris_gini_d3"))), 900, 640).k).toBeGreaterThanOrEqual(MIN_READABLE_SCALE);
    expect(pngScale(layout)).toBeLessThanOrEqual(3);
    expect(layout.width * pngScale(layout)).toBeLessThanOrEqual(16000 + 1e-6);
  });

  it("exports a standalone SVG with every box and the path to a leaf", () => {
    const diagram = diagramOf(fixture("iris_gini_d3"));
    const svg = diagramSvg(layoutTree(diagram));
    expect(svg.startsWith("<svg xmlns=\"http://www.w3.org/2000/svg\"")).toBe(true);
    expect(svg.match(/<rect /g)).toHaveLength(10); // background + 9 boxes
    expect(svg).toContain('fill="#e58139"');
    expect(svg).toContain("petal length (cm) &lt;= 2.45");
    expect([...pathTo(diagram, 4)].sort()).toEqual([0, 2, 3, 4]);
  });

  it("renders the SVG component with one group per node", () => {
    const diagram = diagramOf(fixture("breast_cancer_entropy_d10"));
    const started = performance.now();
    const { container } = render(TreeDiagram, { diagram, pngLabel: "PNG", svgLabel: "SVG" });
    const ms = performance.now() - started;
    expect(container.querySelectorAll("[data-node]")).toHaveLength(27);
    expect(container.querySelector("[data-tree-diagram]")?.getAttribute("data-nodes")).toBe("27");
    expect(container.querySelector("[data-node='0'] rect")?.getAttribute("fill")).toBe(
      fixture("breast_cancer_entropy_d10").expected.plot_tree.boxes_in_drawing_order[0]?.facecolor,
    );
    expect(ms).toBeLessThan(2000);
  });

  it("renders 407 nodes", () => {
    const diagram = diagramOf(fixture("california_housing_squared_error_d8_mss10"));
    const started = performance.now();
    const { container } = render(TreeDiagram, { diagram, pngLabel: "PNG", svgLabel: "SVG" });
    const ms = performance.now() - started;
    expect(container.querySelectorAll("[data-node]")).toHaveLength(407);
    // jsdom is much slower than a WebView; the measure in the app goes to the log.
    console.info(`TreeDiagram 407 nodos (jsdom): ${ms.toFixed(0)} ms`);
    expect(ms).toBeLessThan(5000);
  });
});

describe("TreeRulesText (C14)", () => {
  it("keeps the indentation of export_text", () => {
    const text = fixture("iris_gini_d3").expected.export_text.view_text;
    const { container } = render(TreeRulesText, { text, downloadName: "arbol_texto.txt", downloadLabel: "📥 Descargar texto del árbol" });
    expect(container.querySelector("pre")?.textContent).toBe(text);
  });
});

describe("FeatureImportanceChart (C10)", () => {
  const f = fixture("iris_gini_d3");
  const imp = f.expected.feature_importances;
  const rows = imp
    .map((importance, i) => ({ feature_index: i, name: f.expected.feature_names[i] ?? "", importance, std: null }))
    .sort((a, b) => a.importance - b.importance || a.feature_index - b.feature_index);
  const view: FeatureImportanceView = { method: "native", rows, n_repeats: null, baseline: null, seed: null, task: "classification" };

  it("draws viridis by position, ascending, as matplotlib", () => {
    // The bar colours of the Python figure (views.features), without the alpha.
    expect(viridisColors(4)).toEqual(["#440154", "#31688e", "#35b779", "#fde725"]);
    const figure = importanceFigure(view);
    const trace = figure.data[0] as { x: number[]; text: string[] };
    expect(trace.text).toEqual(["0.000", "0.000", "0.075", "0.925"]);
    expect(String(figure.layout.title && (figure.layout.title as { text: string }).text)).toBe(
      "Importancia de Características<br>(Importancia Nativa del Modelo)",
    );
  });

  it("lists the values in descending order with the column index", () => {
    const table = importanceTable(view);
    expect(table.index).toEqual(["2", "3", "0", "1"]);
    expect(table.rows[0]).toEqual(["petal length (cm)", "0.9251"]);
    const { container } = render(FeatureImportanceChart, { view, idPrefix: "test" });
    expect(container.querySelector("[data-importance='native']")).toBeTruthy();
    expect(container.textContent).toContain("Valores Detallados");
  });

  it("adds the standard deviation for the permutation", () => {
    const perm: FeatureImportanceView = { ...view, method: "permutation", rows: rows.map((r) => ({ ...r, std: 0.01 })), n_repeats: 10 };
    expect(importanceTable(perm).rows[0]).toEqual(["petal length (cm)", "0.9251", "0.0100"]);
    const trace = importanceFigure(perm).data[0] as { error_x?: { array: number[] } };
    expect(trace.error_x?.array).toEqual([0.01, 0.01, 0.01, 0.01]);
  });
});

describe("DecisionPathList (C16)", () => {
  it("writes the lines of show_prediction_path and the leaf (D46)", () => {
    const f = fixture("iris_gini_d3");
    const p = f.expected.prediction_form.default.path;
    const path: DecisionPath = {
      steps: p.steps.map((s) => ({
        node: s.node,
        feature: s.feature,
        name: f.expected.feature_names[s.feature] ?? "",
        value: s.value_float32,
        threshold: s.threshold,
        goes_left: s.goes_left,
        display: null,
      })),
      leaf: p.leaf,
      leaf_samples: 32,
      leaf_value: [0, 1, 0],
      leaf_class: 1,
    };
    const { container } = render(DecisionPathList, { path, classNames: f.expected.class_names });
    const items = [...container.querySelectorAll("li")].map((li) => li.textContent?.trim());
    // The three lines of the Python view `prediction_default`.
    expect(items.slice(0, 3)).toEqual([
      "petal length (cm) = 3.8705 > 2.4500 ✅",
      "petal length (cm) = 3.8705 ≤ 4.7500 ✅",
      "petal width (cm) = 1.2390 ≤ 1.6000 ✅",
    ]);
    expect(items[3]).toContain("Hoja 4");
    expect(items[3]).toContain("versicolor");
  });

  it("shows the label of a categorical value", () => {
    const path: DecisionPath = {
      steps: [{ node: 0, feature: 9, name: "Hombre_Adulto", value: 0, threshold: 0.5, goes_left: true, display: "0 (No)" }],
      leaf: 1,
      leaf_samples: 10,
      leaf_value: [0.2, 0.8],
      leaf_class: 1,
    };
    const { container } = render(DecisionPathList, { path, classNames: ["No sobrevivió", "Sobrevivió"] });
    expect(container.querySelector("li")?.textContent?.trim()).toBe("Hombre_Adulto = 0 (No) ≤ 0.5000 ✅");
  });
});
