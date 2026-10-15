/// <reference types="node" />
/**
 * C18–C20 and the state of 🔍 KNN (fase4-knn.md §9.2) against the golden fixtures of
 * scikit-learn (`tests/parity/knn/*.json`): palette, limits and initial controls of the
 * canvas, the heatmap of distances and the table of neighbours.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cleanup, render, screen } from "@testing-library/svelte";
import { afterEach, describe, expect, it } from "vitest";
import type { ClassDistancesView } from "../../bindings/ClassDistancesView";
import type { Matrix } from "../../bindings/Matrix";
import type { PlaygroundView } from "../../bindings/PlaygroundView";
import type { TrainedModelInfo } from "../../bindings/TrainedModelInfo";
import { knn } from "../../lessons/knn/state.svelte";
import ClassDistanceHeatmap from "./ClassDistanceHeatmap.svelte";
import { aggregateByClass, arc, cellTextColor, classColor, distanceHeatmapFigure, percent, PLAYGROUND_PALETTE, playgroundFigure, pointSize } from "./knnFigures";
import KnnNeighborsTable from "./KnnNeighborsTable.svelte";

afterEach(() => cleanup());

interface Interactive {
  features: [number, number];
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number };
  palette: string[];
  class_info: { id: number; name: string; color: string }[];
  initial_controls: { k: number; metric: string; weights: string; show_confidence: boolean; show_distances: boolean };
  samples: { point: [number, number]; sklearn_2d_all_data: { predict: number; proba: number[]; confidence_max_proba: number; neighbours: number[]; distances: number[] } }[];
}

interface DistanceApp {
  classes: number[];
  class_labels: string[];
  matrix: number[][];
  min_distance: number;
  max_distance: number;
  ratio: number;
  metrics_shown: Record<string, string>;
  within_class_mean_D67: number[];
}

interface Fixture {
  expected: {
    feature_names: string[];
    class_names: string[];
    interactive: { default: Interactive };
    distance_analysis: { app: DistanceApp };
  };
}

function fixture(name: string): Fixture {
  return JSON.parse(readFileSync(resolve(process.cwd(), `../tests/parity/knn/${name}.json`), "utf8")) as Fixture;
}

const IRIS = fixture("iris_k5_uniform_minkowski");
const CANCER = fixture("breast_cancer_k5_uniform_minkowski");

describe("canvas of 🎮 Visualización Interactiva (C18)", () => {
  const interactive = IRIS.expected.interactive.default;

  it("uses the palette of the Python canvas", () => {
    expect([...PLAYGROUND_PALETTE]).toEqual(interactive.palette);
    for (const info of interactive.class_info) {
      expect(classColor(info.id)).toBe(info.color);
    }
    expect(classColor(9)).toBe(interactive.palette[1]);
  });

  it("starts with the controls of the trained model (D55: K up to 20)", () => {
    knn.reset();
    const info = { model_id: 7, feature_names: IRIS.expected.feature_names, config: { kind: "knn", n_neighbors: 20, weights: "distance", metric: "manhattan" } } as unknown as TrainedModelInfo;
    knn.syncPlayground(info);
    expect(knn.playground.k).toBe(20);
    expect([knn.playground.weights, knn.playground.metric]).toEqual(["distance", "manhattan"]);
    expect([knn.playground.showConfidence, knn.playground.showDistances]).toEqual([interactive.initial_controls.show_confidence, interactive.initial_controls.show_distances]);
    expect(knn.playground.pair).toEqual(interactive.features);
    // Points survive navigation, and are emptied by another pair or another model.
    knn.addPoint(6.1, 3.2);
    knn.syncPlayground(info);
    expect(knn.playground.points).toHaveLength(1);
    knn.setPair([2, 3]);
    expect(knn.playground.points).toHaveLength(0);
    knn.addPoint(1, 1);
    knn.syncPlayground({ ...info, model_id: 8 } as TrainedModelInfo);
    expect(knn.playground.points).toHaveLength(0);
    expect(knn.playground.pair).toEqual([2, 3]);
  });

  /** A view as the backend sends it, with the sklearn results of the fixture samples. */
  function view(): { view: PlaygroundView; data: Matrix; codes: number[] } {
    const data: Matrix = { rows: 4, cols: 2, values: [5.1, 3.5, 5.1, 3.5, 7.0, 3.2, 6.3, 3.3] };
    const codes = [0, 0, 1, 2];
    const samples = interactive.samples.slice(0, 2);
    return {
      data,
      codes,
      view: {
        data: null,
        codes: null,
        classes: [0, 1, 2],
        class_names: IRIS.expected.class_names,
        colors: interactive.palette.slice(0, 3),
        bounds: [interactive.bounds.xMin, interactive.bounds.xMax, interactive.bounds.yMin, interactive.bounds.yMax],
        feature_names: [IRIS.expected.feature_names[0] ?? "", IRIS.expected.feature_names[1] ?? ""],
        points: samples.map((s) => ({
          point: s.point,
          class_index: s.sklearn_2d_all_data.predict,
          confidence: s.sklearn_2d_all_data.confidence_max_proba,
          proba: s.sklearn_2d_all_data.proba,
          mean_distance: 0.2,
          neighbors: [0, 2, 3].map((row, i) => ({ train_row: row, dataset_row: row, distance: 0.1 * (i + 1), weight: 1, class_index: codes[row] ?? 0, value: null, features: [] })),
        })),
        mean_confidence: 0.8,
        engine: "kd_tree",
        n_neighbors: 3,
      },
    };
  }

  it("draws named axes, the limits min − 1 / max + 1 and one trace per class", () => {
    const { view: v, data, codes } = view();
    const fig = playgroundFigure(v, data, codes, { showConfidence: true, showDistances: true, scale: { dx: 0.01, dy: 0.01 } });
    const xaxis = fig.layout.xaxis as { title: { text: string }; range: number[] };
    expect(xaxis.title.text).toBe("sepal length (cm)");
    expect(xaxis.range).toEqual([interactive.bounds.xMin, interactive.bounds.xMax]);
    const names = fig.data.map((d) => (d as { name?: string }).name).filter(Boolean);
    expect(names).toEqual(["setosa", "versicolor", "virginica", "Puntos de Prueba"]);
    // Duplicates are merged and marked ×n.
    const setosa = fig.data[0] as { x: number[]; customdata: string[]; marker: { size: number[] } };
    expect(setosa.x).toEqual([5.1]);
    expect(setosa.customdata).toEqual([" ×2"]);
    expect(setosa.marker.size[0]).toBe(pointSize(2));
    // Dashed red lines to the neighbours: 3 per point, separated by gaps.
    const lines = fig.data.find((d) => (d as { line?: { dash?: string } }).line?.dash === "dash") as { x: (number | null)[] };
    expect(lines.x.filter((x) => x === null)).toHaveLength(6);
  });

  it("reveals the lines one by one, nearest first (D54)", () => {
    const { view: v, data, codes } = view();
    const fig = playgroundFigure(v, data, codes, { showConfidence: false, showDistances: true, revealed: [1, 0], scale: { dx: 0.01, dy: 0.01 } });
    const lines = fig.data.find((d) => (d as { line?: { dash?: string } }).line?.dash === "dash") as { x: (number | null)[] };
    expect(lines.x).toEqual([v.points[0]?.point[0], 5.1, null]);
    const hidden = playgroundFigure(v, data, codes, { showConfidence: false, showDistances: false, scale: { dx: 0.01, dy: 0.01 } });
    expect(hidden.data.some((d) => (d as { line?: { dash?: string } }).line?.dash === "dash")).toBe(false);
  });

  it("draws the confidence arc proportional to max(proba) and NN% labels", () => {
    const a = arc(0, 0, 10, 0.5, { dx: 1, dy: 1 });
    expect(a.x[0]).toBeCloseTo(0);
    expect(a.y[0]).toBeCloseTo(10);
    expect(a.x[a.x.length - 1]).toBeCloseTo(0);
    expect(a.y[a.y.length - 1]).toBeCloseTo(-10);
    expect(percent(0.6)).toBe("60%");
    expect(percent(1)).toBe("100%");
  });

  it("merges points by coordinates and class", () => {
    const groups = aggregateByClass({ rows: 3, cols: 2, values: [1, 2, 1, 2, 1, 2] }, [0, 0, 1]);
    expect(groups.get(0)?.count).toEqual([2]);
    expect(groups.get(1)?.count).toEqual([1]);
  });
});

function distances(app: DistanceApp, verdict: ClassDistancesView["verdict"]): ClassDistancesView {
  return {
    classes: app.classes,
    class_names: app.class_labels,
    matrix: app.matrix,
    within: app.within_class_mean_D67,
    min: app.min_distance,
    max: app.max_distance,
    ratio: app.ratio,
    verdict,
    metric: "minkowski",
    n_samples: 105,
  };
}

describe("📏 Análisis de Distancias (C19)", () => {
  it("writes {:.2f} outside the diagonal and the within-class means on it (D67)", () => {
    const app = IRIS.expected.distance_analysis.app;
    const fig = distanceHeatmapFigure(distances(app, "moderate"));
    const texts = (fig.layout.annotations ?? []).map((a) => (a as { text: string }).text);
    expect(texts).toContain("3.24");
    expect(texts).toContain("4.81");
    expect(texts).toContain("1.90");
    expect(texts).toContain("<i>0.69</i>");
    const title = (fig.layout.title as { text: string }).text;
    expect(title).toBe("Distancias Promedio entre Clases<br>(Métrica: minkowski)");
    // Text colour by luminance: black on yellow, white on purple.
    expect(cellTextColor(1)).toBe("#111111");
    expect(cellTextColor(0)).toBe("#ffffff");
  });

  it("shows the metrics of Python and the verdict", () => {
    const app = IRIS.expected.distance_analysis.app;
    render(ClassDistanceHeatmap, { view: distances(app, "moderate"), idPrefix: "t" });
    for (const value of Object.values(app.metrics_shown)) {
      expect(screen.getByText(value)).toBeTruthy();
    }
    expect(screen.getByText(/Separación moderada entre clases/)).toBeTruthy();
  });

  it("replaces the binary verdict, always 🔴 in Python (D67)", () => {
    const app = CANCER.expected.distance_analysis.app;
    render(ClassDistanceHeatmap, { view: distances(app, "binary"), idPrefix: "t" });
    expect(screen.queryByText(/muy próximas/)).toBeNull();
    expect(screen.getByText(/Con dos clases solo hay una distancia/)).toBeTruthy();
  });
});

describe("👥 vecinos de una predicción (C20)", () => {
  it("lists the K neighbours with distance, class, weight and features", () => {
    render(KnnNeighborsTable, {
      neighbors: [
        { train_row: 80, dataset_row: 3, distance: 0, weight: 1, class_index: 1, value: null, features: [5.8, 3] },
        { train_row: 100, dataset_row: 9, distance: 0.4481027, weight: 0, class_index: 2, value: null, features: [6, 2.9] },
      ],
      featureNames: ["a", "b"],
      classNames: ["setosa", "versicolor", "virginica"],
      weights: "distance",
      tie: { n_tied: 3, slots: 1, distance: 0.4481027 },
    });
    expect(screen.getByText("versicolor")).toBeTruthy();
    expect(screen.getByText("0.4481")).toBeTruthy();
    expect(screen.getByText(/Los 2 vecinos más cercanos/)).toBeTruthy();
    expect(screen.getByText(/Hay 3 muestras de entrenamiento/)).toBeTruthy();
    expect(screen.getByText(/solo cuentan esas/)).toBeTruthy();
  });
});
