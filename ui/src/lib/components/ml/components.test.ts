import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import { compRaw } from "../../content";
import { fmt } from "../../fmt";
import { pickLevel } from "../../ml/levels";
import FeaturePairSelector from "./FeaturePairSelector.svelte";
import MetricGrid from "./MetricGrid.svelte";
import PredictionForm from "./PredictionForm.svelte";
import PredictionResultCard from "./PredictionResultCard.svelte";
import ThresholdAnalysis from "./ThresholdAnalysis.svelte";

afterEach(() => cleanup());

describe("quality levels from the TOML", () => {
  it("picks the first level reached and shows its icon (D34)", () => {
    const table = compRaw("clasificacion", "accuracy_niveles");
    expect(pickLevel(table, 0.95)).toEqual({ severity: "success", markdown: "🌟 Excelente" });
    expect(pickLevel(table, 0.75)?.severity).toBe("warning");
    expect(pickLevel(table, 0.1)?.markdown).toContain("Necesita mejora");
  });

  it("evaluates the baseline thresholds of the average precision", () => {
    const table = compRaw("roc", "binaria_ap");
    expect(pickLevel(table, 0.95, { ap: 0.95 }, { baseline: 0.6 })?.severity).toBe("success");
    expect(pickLevel(table, 0.65, { ap: 0.65 }, { baseline: 0.6 })?.severity).toBe("warning");
    expect(pickLevel(table, 0.5, { ap: 0.5 }, { baseline: 0.6 })?.severity).toBe("error");
  });

  it("uses `<` for the probability separation", () => {
    const table = compRaw("probabilidades", "separacion");
    expect(pickLevel(table, 0.029, {}, { below: true })?.severity).toBe("success");
    expect(pickLevel(table, 0.5, {}, { below: true })?.severity).toBe("warning");
  });

  it("formats percentages and signs like Python", () => {
    expect(fmt("{p:.1%} {c:+.1f}", { p: 0.02924, c: 138.44 })).toBe("2.9% +138.4");
  });
});

describe("MetricGrid", () => {
  it("renders one card per metric with its badge", () => {
    render(MetricGrid, {
      items: [
        { label: "Exactitud (Accuracy)", value: "1.0000", badge: { severity: "success", markdown: "🌟 Excelente" } },
        { label: "RMSE", value: "53.1202" },
      ],
    });
    expect(screen.getByText("1.0000")).toBeTruthy();
    expect(screen.getByText("🌟 Excelente").closest(".notice")?.classList.contains("success")).toBe(true);
  });
});

describe("PredictionForm (C9)", () => {
  const controls = [
    { index: 0, column: "sex", label: "Sexo", description: "Sexo del pasajero", control: { kind: "select" as const, options: [{ label: "Mujer", code: 0 }, { label: "Hombre", code: 1 }], default_index: 0 } },
    { index: 1, column: "age", label: "Edad", description: null, control: { kind: "slider" as const, min: 0, max: 80, step: 0.8, default: 29 } },
    { index: 2, column: "alone", label: "alone", description: null, control: { kind: "checkbox" as const, default: false } },
  ];

  it("emits the codes the model receives", async () => {
    const onchange = vi.fn();
    render(PredictionForm, { controls, values: [0, 29, 0], onchange, idPrefix: "t" });
    const select = screen.getByLabelText("Sexo") as HTMLSelectElement;
    await fireEvent.change(select, { target: { value: "1" } });
    expect(onchange).toHaveBeenLastCalledWith([1, 29, 0]);
    await fireEvent.click(screen.getByLabelText("alone"));
    expect(onchange).toHaveBeenLastCalledWith([0, 29, 1]);
    expect(screen.getByText("Sexo del pasajero")).toBeTruthy();
  });

  it("puts the first ⌈n/2⌉ controls in the left column", () => {
    const { container } = render(PredictionForm, { controls, values: [0, 29, 0], onchange: vi.fn(), idPrefix: "t" });
    const columns = container.querySelectorAll(".column");
    expect(columns[0]?.querySelectorAll(":scope > .control")).toHaveLength(2);
    expect(columns[1]?.querySelectorAll(":scope > .control")).toHaveLength(1);
  });
});

describe("PredictionResultCard (C9)", () => {
  it("shows the class and the probability table with 4 decimals (D33)", () => {
    render(PredictionResultCard, {
      result: { kind: "class", index: 1, name: "versicolor", proba: [{ code: 0, name: "setosa", probability: 0.0427128 }, { code: 1, name: "versicolor", probability: 0.928777 }], path: null, neighbors: null },
    });
    expect(screen.getByText("Clase: versicolor")).toBeTruthy();
    expect(screen.getByText("0.0427")).toBeTruthy();
  });

  it("shows a value and the coefficients of a linear model (D25)", () => {
    render(PredictionResultCard, {
      result: { kind: "value", value: 1.23456, coefficients: [{ feature: "bmi", coef: 546.29972 }], path: null, neighbors: null },
    });
    expect(screen.getByText("Valor: 1.2346")).toBeTruthy();
    expect(screen.getByText("546.2997")).toBeTruthy();
  });
});

describe("ThresholdAnalysis (C6)", () => {
  it("shows the rows 0.3, 0.5 and 0.7 (D21) and the advice for a far optimum", () => {
    const row = (threshold: number, f1: number) => ({ threshold, accuracy: 0.9, precision: 0.9, recall: 0.9, f1 });
    render(ThresholdAnalysis, {
      thresholds: [row(0.1, 0.8), row(0.30000000000000004, 0.85), row(0.5, 0.9), row(0.7000000000000001, 0.95), row(0.8, 0.97)],
      best: 0.8,
      shown: [0.30000000000000004, 0.5, 0.7000000000000001],
      idPrefix: "t",
    });
    expect(screen.getAllByRole("row")).toHaveLength(4);
    expect(screen.getByText("0.3")).toBeTruthy();
    expect(screen.getByText("0.80")).toBeTruthy();
    expect(screen.getByText(/ajustar el umbral a 0.80/)).toBeTruthy();
    // The best threshold comes from the test set: the leakage warning sits under the advice.
    const warning = document.querySelector("[data-threshold-warning]");
    expect(warning?.textContent).toContain("validación");
  });

  it("shows neither the advice nor its warning when the optimum is close to 0.5", () => {
    const row = (threshold: number) => ({ threshold, accuracy: 0.9, precision: 0.9, recall: 0.9, f1: 0.9 });
    render(ThresholdAnalysis, { thresholds: [row(0.5)], best: 0.55, shown: [0.5], idPrefix: "t" });
    expect(screen.queryByText(/ajustar el umbral/)).toBeNull();
    expect(document.querySelector("[data-threshold-warning]")).toBeNull();
  });
});

describe("FeaturePairSelector (C8)", () => {
  it("reports the new pair", async () => {
    const onchange = vi.fn();
    render(FeaturePairSelector, { features: ["a", "b", "c"], pair: [0, 1], onchange, idPrefix: "t" });
    await fireEvent.change(screen.getByLabelText("Segunda característica:"), { target: { value: "2" } });
    expect(onchange).toHaveBeenCalledWith([0, 2]);
  });
});
