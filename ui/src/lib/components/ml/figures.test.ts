import { describe, expect, it } from "vitest";
import type { DecisionGrid } from "../../bindings/DecisionGrid";
import type { RocData } from "../../bindings/RocData";
import {
  binaryProbabilityFigure,
  boundaryFigure,
  classColor,
  classCurveColor,
  classProbabilityFigure,
  coefficientFigure,
  confusionFigure,
  discreteScale,
  errorHistogramFigure,
  precisionByClassFigure,
  predVsRealFigure,
  residualsFigure,
  rocPrFigure,
  RDBU_BINARY,
  SET3,
  surfaceFigure,
} from "./figures";

type Trace = Record<string, unknown>;
const trace = (data: unknown[], i: number): Trace => data[i] as Trace;
const names = (data: unknown[]): unknown[] => data.map((t) => (t as Trace)["name"]);

const curve = (x: number[], y: number[]) => ({ x, y, thresholds: null });

const MULTI: RocData = {
  kind: "multiclass",
  per_class: ["setosa", "versicolor", "virginica"].map((name, i) => ({
    class_index: i,
    name,
    roc: curve([0, 0, 1], [0, 1, 1]),
    auc: 1,
    pr: curve([1, 0], [1, 1]),
    ap: 1,
  })),
  micro: { class_index: 0, name: "micro", roc: curve([0, 1], [0, 1]), auc: 0.987, pr: curve([1, 0], [0.5, 1]), ap: 0.95 },
  macro_roc: curve([0, 1], [1, 1]),
  macro_auc: 0.99,
  macro_ap: 0.97,
  weighted_auc: 0.99,
  absent_classes: [],
};

describe("C1 figures", () => {
  it("draws the confusion matrix with Blues, integer annotations and real classes as rows", () => {
    const fig = confusionFigure({ rows: 2, cols: 2, values: [19, 1, 0, 13] }, ["<b>a</b>", "b"]);
    const heat = trace(fig.data, 0);
    expect(heat["type"]).toBe("heatmap");
    expect(heat["z"]).toEqual([
      [19, 1],
      [0, 13],
    ]);
    expect(heat["x"]).toEqual(["&lt;b&gt;a&lt;/b&gt;", "b"]);
    expect(fig.layout.annotations?.map((a) => a.text)).toEqual(["19", "1", "0", "13"]);
    expect((fig.layout.yaxis as Trace)["autorange"]).toBe("reversed");
  });

  it("draws the precision of each class from 0 to 1 in #3274a1", () => {
    const fig = precisionByClassFigure([
      { name: "x", precision: 0.5, recall: 1, f1: 0.6, support: 3 },
      { name: "y", precision: 1, recall: 0.5, f1: 0.6, support: 2 },
    ]);
    expect(trace(fig.data, 0)["y"]).toEqual([0.5, 1]);
    expect((trace(fig.data, 0)["marker"] as Trace)["color"]).toBe("#3274a1");
    expect((fig.layout.yaxis as Trace)["range"]).toEqual([0, 1]);
  });
});

describe("C2/C7 figures", () => {
  it("colours predictions by absolute error and draws the diagonal of the real values", () => {
    const fig = predVsRealFigure([1, 2, 4], [1.5, 2, 3], { colorByError: true });
    const marker = trace(fig.data, 0)["marker"] as Trace;
    expect(marker["color"]).toEqual([0.5, 0, 1]);
    expect(trace(fig.data, 1)["x"]).toEqual([1, 4]);
  });

  it("shows the real R² (D22) and the perfect-prediction line over y and ŷ", () => {
    const fig = predVsRealFigure([1, 2], [0, 5], { r2: 0.4773 });
    expect(fig.layout.annotations?.[0]?.text).toBe("R² = 0.4773");
    expect(trace(fig.data, 1)["x"]).toEqual([0, 5]);
    expect(trace(fig.data, 1)["name"]).toBe("Predicción Perfecta");
  });

  it("draws the error histogram with its KDE and the line x = 0", () => {
    const fig = errorHistogramFigure(
      { edges: [-1, 0, 1], counts: [2, 3] },
      { level: "errores", n: 5, bandwidth: null, grid_min: -1, grid_max: 1, density: [0, 1, 0] },
    );
    expect(trace(fig.data, 0)["x"]).toEqual([-0.5, 0.5]);
    expect(trace(fig.data, 1)["x"]).toEqual([-1, 0, 1]);
    expect(fig.layout.shapes?.[0]?.x0).toBe(0);
    expect((fig.layout.yaxis as Trace)["title"]).toMatchObject({ text: "Count" });
  });

  it("draws residuals against predictions and their 20-bin histogram with the mean", () => {
    const fig = residualsFigure({ y_true: [3, 1], y_pred: [2, 2], std: 1.414, mean: 0, hist: { edges: [-1, 0, 1], counts: [1, 1] } });
    expect(trace(fig.data, 0)["y"]).toEqual([1, -1]);
    expect(names(fig.data)).toContain("Media: 0.000");
    expect(names(fig.data)).toContain("Ideal (0)");
    expect(fig.layout.annotations?.some((a) => a.text === "Desv. Estándar: 1.414")).toBe(true);
  });
});

describe("C4 figures", () => {
  it("binary: ROC in darkorange, random diagonal, PR in darkgreen and the baseline", () => {
    const fig = rocPrFigure({
      kind: "binary",
      positive_class: "benign",
      roc: curve([0, 0, 1], [0, 1, 1]),
      auc: 0.998,
      pr: curve([1, 0], [0.9, 1]),
      ap: 0.999,
      baseline: 0.6316,
    });
    expect(names(fig.data)).toEqual([
      "Curva ROC (AUC = 0.998)",
      "Clasificador Aleatorio",
      "Curva P-R (AP = 0.999)",
      "Baseline (0.632)",
    ]);
    expect((trace(fig.data, 0)["line"] as Trace)["color"]).toBe("#ff8c00");
    expect(trace(fig.data, 2)["xaxis"]).toBe("x2");
    expect(trace(fig.data, 3)["y"]).toEqual([0.6316, 0.6316]);
  });

  it("multiclass: per-class curves, macro (horizontal AP) and micro, filtered in the UI", () => {
    const all = rocPrFigure(MULTI, { showPerClass: true, averages: ["macro", "micro"] });
    expect(all.data).toHaveLength(3 * 2 + 2 + 2 + 1);
    expect(names(all.data)[0]).toBe("setosa (AUC = 1.000)");
    expect(names(all.data)).toContain("Macro-promedio (AP = 0.970)");
    const macroPr = all.data.find((t) => (t as Trace)["name"] === "Macro-promedio (AP = 0.970)") as Trace;
    expect(macroPr["y"]).toEqual([0.97, 0.97]);
    const micro = rocPrFigure(MULTI, { showPerClass: false, averages: ["micro"] });
    expect(names(micro.data)).toEqual(["Micro-promedio (AUC = 0.987)", "Micro-promedio (AP = 0.950)", "Clasificador Aleatorio"]);
    const none = rocPrFigure(MULTI, { showPerClass: false, averages: [] });
    expect(names(none.data)[0]).toBe("Macro-promedio (AUC = 0.990)");
  });

  it("cycles the class colours of matplotlib", () => {
    expect(classCurveColor(0)).toBe("#00ffff");
    expect(classCurveColor(10)).toBe("#00ffff");
  });
});

describe("C5 figures", () => {
  it("overlays the histograms of each real class and the 0.5 threshold", () => {
    const fig = binaryProbabilityFigure([
      { class_index: 0, name: "malignant", histogram: { edges: [0, 0.1, 0.2], counts: [5, 1] } },
      { class_index: 1, name: "benign", histogram: { edges: [0.8, 0.9, 1], counts: [2, 9] } },
    ]);
    expect(names(fig.data)).toEqual(["Clase malignant (Real)", "Clase benign (Real)", "Umbral de decisión (0.5)"]);
    expect(trace(fig.data, 2)["y"]).toEqual([0, 9]);
    expect(fig.layout.barmode).toBe("overlay");
  });

  it("titles each multiclass histogram with its class and Set3 colour", () => {
    const fig = classProbabilityFigure({ class_index: 2, name: "virginica", histogram: { edges: [0, 1], counts: [3] } });
    expect((fig.layout.title as Trace)["text"]).toBe("Probabilidades para virginica");
    expect(((trace(fig.data, 0)["marker"] as Trace)["color"] as string).startsWith("rgba(190, 186, 218")).toBe(true);
  });
});

describe("C8 figures", () => {
  const grid = (values: DecisionGrid["values"], classNames: string[] | null): DecisionGrid => ({
    x: [0, 1, 2],
    y: [0, 1],
    values,
    notices: [{ kind: "created", rows: 2, cols: 3 }],
    points: { rows: 2, cols: 2, values: [0.5, 0.5, 1.5, 0.5] },
    point_target: [0, 1],
    feature_names: ["a", "b"],
    class_names: classNames,
    show_2d_warning: true,
  });

  it("D23: the same colour per class in the background, the points and the legend", () => {
    const fig = boundaryFigure(grid({ kind: "classes", z: [0, 0, 1, 0, 1, 1] }, ["no", "sí"]), undefined, false);
    const heat = trace(fig.data, 0);
    expect(heat["z"]).toEqual([
      [0, 0, 1],
      [0, 1, 1],
    ]);
    expect(heat["zsmooth"]).toBe(false);
    const scale = heat["colorscale"] as [number, string][];
    expect(scale[0]?.[1]).toBe(RDBU_BINARY[0]);
    expect(scale[3]?.[1]).toBe(RDBU_BINARY[1]);
    expect(names(fig.data).slice(1)).toEqual(["no", "sí"]);
    expect((trace(fig.data, 2)["marker"] as Trace)["color"]).toBe(RDBU_BINARY[1]);
    expect((fig.layout.title as Trace)["text"]).toBe("Frontera de Decisión para a vs b");
  });

  it("uses Set3 in multiclass and a flat band per class", () => {
    expect(classColor(2, 3)).toBe(SET3[2]);
    expect(discreteScale(["#a", "#b"])).toEqual([
      [0, "#a"],
      [0.5, "#a"],
      [0.5, "#b"],
      [1, "#b"],
    ]);
  });

  it("draws the prediction surface as viridis contours with the training points", () => {
    const fig = surfaceFigure(grid({ kind: "values", z: [1, 2, 3, 4, 5, 6], min: 1, max: 6 }, null), undefined, false);
    expect(trace(fig.data, 0)["type"]).toBe("contour");
    expect(trace(fig.data, 1)["x"]).toEqual([0.5, 1.5]);
    expect((fig.layout.title as Trace)["text"]).toBe("Superficie de Predicción para a vs b");
  });
});

describe("Coefficient chart", () => {
  it("puts the largest |coef| on top with the colours of the effect", () => {
    const row = (feature: string, coef: number) => ({
      feature,
      feature_position: 0,
      coef,
      abs: Math.abs(coef),
      odds_ratio: null,
      change_pct: null,
      effect: coef > 0 ? ("positive" as const) : ("negative" as const),
      importance: null,
      strength: null,
      intensity: "low" as const,
    });
    const fig = coefficientFigure([row("big", -2.278), row("small", 0.869)]);
    expect((fig.layout.yaxis as Trace)["categoryarray"]).toEqual(["small", "big"]);
    expect(trace(fig.data, 0)["text"]).toEqual(["0.869"]);
    expect(trace(fig.data, 1)["name"]).toBe("Efecto Negativo");
  });
});
