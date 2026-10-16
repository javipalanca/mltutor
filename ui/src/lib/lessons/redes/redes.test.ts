import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import type { ArchitectureView } from "../../bindings/ArchitectureView";
import type { EpochLog } from "../../bindings/EpochLog";
import NetworkDiagram from "../../components/ml/NetworkDiagram.svelte";
import { historyFigure } from "../../components/ml/nnFigures";
import { diagramHeight, layoutNetwork, linkOpacity, MAX_DRAWN } from "../../components/ml/networkLayout";
import { hasMd, nnText } from "../../content";
import { nn, taskSignature, thousands } from "./state.svelte";

function view(task: "classification" | "regression", nClasses: number | null): ArchitectureView {
  return {
    dataset: { kind: "builtin", id: "iris" },
    input: 4,
    output: nClasses && nClasses > 2 ? nClasses : 1,
    task,
    n_classes: nClasses,
    n_rows: 150,
    suggested_neurons: [10, 10, 10, 10, 10],
    default_neurons: [10, 10, 10, 10, 10],
    layers: [],
    total_params: 193,
    complexity_ratio: 1.287,
    complexity: "high",
    output_options: task === "regression" ? ["linear", "sigmoid", "tanh", "softmax"] : ["sigmoid", "softmax", "linear", "tanh"],
    recommended_output: task === "regression" ? "linear" : nClasses === 2 ? "sigmoid" : "softmax",
    output_check: { level: "success", key: "multiclase_softmax", untrainable: null },
    batch_hint: "large",
    size_tips: { half: 2, double: 8, quadruple: 16, max_params: 15, batch_recommended: 32 },
    default_epochs: 50,
    suggested_epochs: 50,
    split: { train: 96, val: 24, test: 30 },
    cost: null,
  };
}

describe("state of 🧠 Redes Neuronales", () => {
  it("goes back to the recommended output when the task changes (D85)", () => {
    nn.reset();
    const iris = view("classification", 3);
    const cancer = view("classification", 2);
    nn.syncTask(iris);
    nn.setOutput(iris, "tanh");
    nn.syncTask(iris);
    expect(nn.outputOf(iris)).toBe("tanh");
    // Python kept "softmax" for Cáncer and the training failed (§12.4).
    nn.syncTask(cancer);
    expect(nn.outputOf(cancer)).toBe("sigmoid");
    expect(taskSignature(cancer)).toBe("classification:2");
    nn.syncTask(view("regression", null));
    expect(nn.output).toBe("linear");
  });

  it("builds the configuration of the controls (D70, D90)", () => {
    nn.reset();
    const iris = view("classification", 3);
    nn.syncTask(iris);
    nn.layers = 3;
    nn.neurons = [32, null, null, null, null];
    nn.earlyStopping = false;
    expect(nn.config("builtin:iris", iris)).toEqual({
      kind: "neural_net",
      arch: { hidden: [32, 10, 10], activation: "relu", output_activation: "softmax", dropout: 0.2, batch_size: 64, optimizer: "adam" },
      training: { learning_rate: 0.001, epochs: 50, validation_pct: 20, patience: null, lr_factor: 0.5, seed: 42 },
    });
    nn.setEpochs("builtin:iris", 120);
    expect(nn.epochsOf("builtin:iris", iris)).toBe(120);
    expect(nn.epochsOf("builtin:wine", iris)).toBe(50);
  });

  it("only accepts the epochs of the training in progress, without duplicates", () => {
    nn.reset();
    nn.startLive(10);
    nn.live.taskId = 7;
    const log = (epoch: number): EpochLog => ({ epoch, loss: 1 / epoch, metric: 0.5, val_loss: 1, val_metric: 0.4, learning_rate: null });
    nn.onEpochEvent({ task_id: 6, phase: "training", epoch: 1, epochs: 10, logs: [log(1)], sent_at_ms: 0 });
    expect(nn.live.logs).toHaveLength(0);
    nn.onEpochEvent({ task_id: 7, phase: "training", epoch: 3, epochs: 10, logs: [log(1), log(2), log(3)], sent_at_ms: 0 });
    nn.onEpochEvent({ task_id: 7, phase: "training", epoch: 3, epochs: 10, logs: [log(3)], sent_at_ms: 0 });
    expect(nn.live.logs.map((l) => l.epoch)).toEqual([1, 2, 3]);
    expect(nn.live.events).toBe(2);
  });

  it("formats thousands like Python's {:,}", () => {
    expect(thousands(282378)).toBe("282,378");
    expect(thousands(193)).toBe("193");
  });
});

describe("C24: network diagram", () => {
  it("follows Python's geometry for small networks", () => {
    const layout = layoutNetwork([4, 10, 10, 3], 600);
    expect(layout.height).toBe(300);
    expect(layout.layers.map((l) => l.nodes.length)).toEqual([4, 10, 10, 3]);
    expect(layout.layers[0]?.x).toBe(40);
    expect(layout.layers[3]?.x).toBe(560);
    expect(layout.links).toHaveLength(3);
    expect(layout.links[1]?.opacity).toBeCloseTo(0.3);
  });

  it("summarises big layers and stays cheap with 5 × 256 neurons", () => {
    const sizes = [64, 256, 256, 256, 256, 256, 10];
    const started = performance.now();
    const layout = layoutNetwork(sizes, 640);
    const elapsed = performance.now() - started;
    expect(layout.layers.every((l) => l.nodes.length <= MAX_DRAWN)).toBe(true);
    expect(layout.layers[1]?.hidden).toBe(256 - MAX_DRAWN);
    expect(layout.layers[1]?.nodes.at(-1)?.index).toBe(255);
    const segments = layout.links.reduce((n, l) => n + (l.d.match(/M/g)?.length ?? 0), 0);
    expect(segments).toBeLessThanOrEqual(6 * 24 * 24);
    expect(linkOpacity(65536)).toBeLessThan(0.3);
    expect(diagramHeight(640, sizes)).toBeLessThanOrEqual(460);
    // The circles of a full column do not overlap.
    const ys = layout.layers[1]?.nodes.map((n) => n.y) ?? [];
    expect(Math.min(...ys.slice(1).map((y, i) => y - (ys[i] ?? 0)))).toBeGreaterThanOrEqual(2 * layout.radius - 1e-9);
    expect(elapsed).toBeLessThan(50);
  });

  it("renders the legend, the labels and the total", () => {
    const rows = [
      { name: "Entrada", neurons: 4, inputs: 0, params: 0, activation: null },
      { name: "Oculta 1", neurons: 30, inputs: 4, params: 150, activation: "relu" as const },
      { name: "Salida", neurons: 3, inputs: 30, params: 93, activation: "softmax" as const },
    ];
    const { container } = render(NetworkDiagram, { rows, activation: "relu", outputActivation: "softmax", task: "clasificación", totalParams: 243 });
    expect(container.querySelectorAll("circle")).toHaveLength(4 + MAX_DRAWN + 3);
    expect(container.textContent).toContain("Capas Ocultas (RELU)");
    expect(container.textContent).toContain("Capa de Salida (SOFTMAX)");
    expect(container.textContent).toContain("+6");
    expect(container.querySelector("[data-network-total]")?.textContent).toContain("243");
  });
});

describe("C27: history chart", () => {
  it("has the two panels of Python with the epochs from 1 and the marks", () => {
    const logs: EpochLog[] = [1, 2, 3].map((epoch) => ({ epoch, loss: 1 / epoch, metric: 0.1 * epoch, val_loss: null, val_metric: 0.2, learning_rate: 0.001 }));
    const figure = historyFigure(logs, "accuracy", { bestEpoch: 2, lrDrops: [1], maxEpochs: 50 });
    expect(figure.data).toHaveLength(4);
    expect((figure.data[0] as { x: number[] }).x).toEqual([1, 2, 3]);
    expect((figure.data[1] as { y: (number | null)[] }).y).toEqual([null, null, null]);
    expect(figure.layout.shapes).toHaveLength(4);
    expect(figure.layout.xaxis?.range).toEqual([0.5, 50.5]);
    const titles = (figure.layout.annotations ?? []).map((a) => a.text);
    expect(titles).toEqual([nnText("evaluacion", "grafica_perdida"), nnText("evaluacion", "grafica_precision")]);
    const regression = historyFigure(logs, "mae");
    expect(regression.layout.yaxis2?.title).toMatchObject({ text: "MAE" });
  });
});

describe("content of 🏗️ and ⚙️", () => {
  it("has every text the tabs load by name", () => {
    const sources = ["ArchitectureTab.svelte", "TrainingTab.svelte", "../../components/ml/NetworkDiagram.svelte"].map((f) =>
      readFileSync(resolve(process.cwd(), "src/lib/lessons/redes", f), "utf8"),
    );
    const keys = new Set<string>();
    for (const source of sources) {
      for (const match of source.matchAll(/\bt\("([a-z0-9_]+)"\)/g)) {
        keys.add(match[1] ?? "");
      }
    }
    const architecture = sources[0] ?? "";
    for (const key of keys) {
      const section = architecture.includes(`t("${key}")`) || (sources[2] ?? "").includes(`t("${key}")`) ? "arquitectura" : "entrenamiento";
      expect(() => nnText(section, key), `${section}.${key}`).not.toThrow();
    }
    expect(keys.size).toBeGreaterThan(40);
    for (const path of ["arquitectura-tamano.md", "arquitectura-consejos.md", "activaciones-perdidas.md", "entrenamiento-conceptos.md", "entrenamiento-tecnicas.md", "escalado.md", "semilla.md", "valores-defecto.md"]) {
      expect(hasMd(`redes/${path}`), path).toBe(true);
    }
  });
});
