/**
 * `--screenshot --scenario` screens of 🧠 Redes Neuronales (🏗️ and ⚙️, F5b), named after the
 * Qt captures of `docs/migration/capturas/qt/fase5/` with the prefix `redes_`. Also used by
 * the smoke test.
 */
import { logToBackend } from "../../backend";
import { busy } from "../../busy.svelte";
import type { DatasetId } from "../../bindings/DatasetId";
import type { NnActivation } from "../../bindings/NnActivation";
import type { NnOptimizer } from "../../bindings/NnOptimizer";
import { nextPaint, waitFor, waitForIdle } from "../../dom";
import { models } from "../../ml/modelSession.svelte";
import { router } from "../../router.svelte";
import { chooseDataset } from "../../selection";
import { datasetKey, session } from "../../session.svelte";
import { expandAll } from "../regression/scenarios";
import { architecture } from "./architecture.svelte";
import { LESSON, nn } from "./state.svelte";
import { nnViz } from "./visualization.svelte";
import { training } from "./training.svelte";
import type { ExportFormat } from "../../bindings/ExportFormat";
import { nnPredictions, type ExportSubview, type PredictionSubview } from "./predictions.svelte";

export interface NnScenario {
  dataset?: DatasetId;
  tab: number;
  layers?: number;
  neurons?: number[];
  activation?: NnActivation;
  output?: NnActivation;
  dropout?: number;
  batch?: number;
  optimizer?: NnOptimizer;
  learningRate?: number;
  epochs?: number;
  validation?: number;
  earlyStopping?: boolean;
  reduceLr?: boolean;
  train?: boolean;
  /** Press "Cancelar" once this epoch has arrived live (D83). */
  cancelAt?: number;
  /** Press "📝 Generar Código de la Arquitectura". */
  code?: boolean;
  open?: string[];
  expandAll?: boolean;
  /** 🎯 subview (`Historial`, `Pesos`, `Superficie`, `Activaciones`). */
  viz?: string;
  /** Features of the surface. */
  pair?: [number, number];
  /** Tick every "Mostrar histograma/medias" of 📉 (the `_expandido` captures). */
  layerCharts?: boolean;
  /** Waits for this element after training (a view of 📈/🎯 is ready). */
  ready?: string;
  /** 🔮 subview (`Individual`, `Lotes`, `Exploracion`). */
  subview?: PredictionSubview;
  /** 🔮 Individual: press "Realizar predicción". */
  predict?: boolean;
  /** 🔮 Lotes: press "📂 Elegir CSV…" (automation: the scenario CSV or one written from the dataset). */
  batchCsv?: boolean;
  /** 🎲 Exploración: base sample and feature. */
  explore?: [number, number];
  /** 💾 panel. */
  exportView?: ExportSubview;
  /** 💾: press the button of this format (automation: saved without dialog). */
  exportSave?: ExportFormat;
}

async function click(selector: string, what: string): Promise<void> {
  const button = (await waitFor(selector, 30000, what)) as HTMLButtonElement;
  button.click();
}

/** Waits for the view of 🏗️ of the selected dataset with the current controls. */
async function architectureReady(): Promise<void> {
  const deadline = performance.now() + 30000;
  for (;;) {
    const view = architecture.view;
    const wanted = nn.arch(view);
    if (
      view &&
      architecture.matches(session.selected) &&
      view.layers.length === wanted.hidden.length + 2 &&
      view.layers.slice(1, -1).every((l, i) => l.neurons === wanted.hidden[i]) &&
      view.layers.at(-1)?.activation === wanted.output_activation
    ) {
      return;
    }
    if (performance.now() > deadline) {
      throw new Error("Tiempo agotado esperando la vista de la arquitectura");
    }
    architecture.request(session.selected, wanted, nn.validationPct, true);
    await new Promise((resolve) => setTimeout(resolve, 30));
  }
}

export async function nnScenario(options: NnScenario): Promise<void> {
  const dataset = { kind: "builtin", id: options.dataset ?? "iris" } as const;
  await chooseDataset(dataset);
  nn.reset();
  nnViz.reset();
  nnPredictions.reset();
  nnPredictions.subview = options.subview ?? "Individual";
  nnPredictions.exportView = options.exportView ?? "Keras";
  const key = datasetKey(dataset);
  if (options.layers !== undefined) {
    nn.layers = options.layers;
  }
  if (options.neurons) {
    nn.neurons = [0, 1, 2, 3, 4].map((i) => options.neurons?.[i] ?? null);
  }
  nn.activation = options.activation ?? "relu";
  nn.dropout = options.dropout ?? 0.2;
  nn.batchSize = options.batch ?? 64;
  nn.optimizer = options.optimizer ?? "adam";
  nn.learningRate = options.learningRate ?? 0.001;
  nn.validationPct = options.validation ?? 20;
  nn.earlyStopping = options.earlyStopping ?? true;
  nn.reduceLr = options.reduceLr ?? true;
  await architectureReady();
  const view = architecture.view;
  if (options.output && view) {
    nn.setOutput(view, options.output);
    await architectureReady();
  }
  if (options.epochs !== undefined) {
    nn.setEpochs(key, options.epochs);
  }
  for (const id of options.open ?? []) {
    session.setExpander(id, true);
  }
  const state = models.lesson(LESSON);
  state.viz = options.viz ?? null;
  if (options.pair) {
    state.pair = options.pair;
  }
  if (options.layerCharts) {
    nnViz.layerHist = { 0: true, 1: true, 2: true, 3: true, 4: true };
    nnViz.layerMeans = { 0: true, 1: true, 2: true, 3: true, 4: true };
  }
  session.activeTab[LESSON] = options.tab;
  router.go(LESSON);
  await nextPaint();
  if (options.train && architecture.view) {
    void logToBackend("debug", `escenario: entrenando la red en ${dataset.id}`);
    const run = training.train(dataset, architecture.view);
    if (options.cancelAt !== undefined) {
      const at = options.cancelAt;
      const deadline = performance.now() + 60000;
      while ((nn.live.logs.at(-1)?.epoch ?? 0) < at && state.training && performance.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 5));
      }
      training.cancel();
    }
    await run;
    if (state.trainError && options.tab !== 2) {
      throw new Error(state.trainError);
    }
    if (options.pair) {
      // Training a new model resets the pair if the features changed (`#setModel`).
      state.pair = options.pair;
    }
  }
  if (options.ready) {
    await waitFor(options.ready, 60000, "la vista de la red");
  }
  const model = models.lesson(LESSON).info;
  if (options.explore && model) {
    nnPredictions.exploreRow = { [model.model_id]: options.explore[0] };
    nnPredictions.exploreFeature = { [model.model_id]: options.explore[1] };
  }
  if (options.predict) {
    await click("[data-nn-predict-single] [data-predict-button]", "el botón de predecir");
    await waitFor("[data-nn-predict-single] [data-prediction-result]", 30000, "la predicción");
  }
  if (options.batchCsv) {
    await click("[data-nn-batch-pick]", "el botón del CSV");
    await waitFor("[data-nn-batch-file]", 30000, "el resultado del lote");
  }
  if (options.subview === "Exploracion" && model) {
    await waitFor('[data-nn-explore-chart] [data-plot-rendered="true"]', 30000, "la curva de exploración");
  }
  if (options.exportSave) {
    await click(`[data-nn-export] button[data-export-format="${options.exportSave}"]`, "el botón de exportar");
    await waitFor("[data-nn-export] [data-export-result]", 30000, "el resultado de la exportación");
  }
  if (options.code) {
    const button = await waitFor("[data-nn-code-button]", 30000, "el botón del código");
    (button as HTMLButtonElement).click();
    await waitFor("[data-nn-architecture] pre", 30000, "el código generado");
  }
  await waitForIdle(() => busy.active, 60000);
  if (options.expandAll) {
    await expandAll();
  }
}

const S = (options: NnScenario) => (): Promise<void> => nnScenario(options);

/** §12.4 / D85: softmax chosen on Iris, then 🔬 Cáncer (Python kept softmax and failed). */
async function inheritedOutput(): Promise<void> {
  await nnScenario({ dataset: "iris", tab: 1, output: "softmax" });
  await chooseDataset({ kind: "builtin", id: "breast_cancer" });
  await architectureReady();
  await nextPaint();
  await waitFor('[data-output-check="binaria_sigmoid"]', 30000, "la salida recomendada de Cáncer");
  await waitForIdle(() => busy.active, 60000);
}
const IRIS = { dataset: "iris" } as const;
const DIAG = "[data-nn-diagnostics]";
const CARDS = "[data-nn-history-cards] [data-view-row]";
const WEIGHTS = '[data-weights-layer="output_layer"] [data-plot-rendered="true"]';
const GRID = '[data-grid-view] [data-plot-rendered="true"]';
const MAP = "[data-network-map]";
/** K1: Diabetes (targets ≈ 150) with tanh hidden layers and SGD at LR 0.3 explodes. */
const DIVERGES: NnScenario = {
  dataset: "diabetes",
  tab: 3,
  train: true,
  activation: "tanh",
  optimizer: "sgd",
  learningRate: 0.3,
  dropout: 0,
  epochs: 30,
  earlyStopping: false,
  reduceLr: false,
};
const ADVANCED = ["nn-avanzada"];

/** Names = Qt captures without the number (`02_arquitectura_iris` → `redes_arquitectura_iris`). */
export const NN_SCENARIOS: Readonly<Record<string, () => Promise<void>>> = {
  redes_arquitectura_iris: S({ ...IRIS, tab: 1 }),
  redes_arquitectura_iris_expandido: S({ ...IRIS, tab: 1, open: ["nn-tamano", "nn-consejos"] }),
  redes_arquitectura_salida_linear: S({ ...IRIS, tab: 1, output: "linear" }),
  redes_arquitectura_salida_sigmoid_multiclase: S({ ...IRIS, tab: 1, output: "sigmoid" }),
  redes_arquitectura_oculta_sigmoid: S({ ...IRIS, tab: 1, activation: "sigmoid" }),
  redes_arquitectura_1_capa: S({ ...IRIS, tab: 1, layers: 1 }),
  redes_arquitectura_5_capas: S({ ...IRIS, tab: 1, layers: 5 }),
  redes_arquitectura_5x256: S({ dataset: "digits", tab: 1, layers: 5, neurons: [256, 256, 256, 256, 256] }),
  redes_arquitectura_codigo: S({ ...IRIS, tab: 1, code: true }),
  redes_entrenamiento_iris: S({ ...IRIS, tab: 2 }),
  redes_entrenamiento_iris_avanzado: S({ ...IRIS, tab: 2, open: ADVANCED }),
  redes_entrenamiento_sin_callbacks: S({ ...IRIS, tab: 2, open: ADVANCED, earlyStopping: false, reduceLr: false }),
  redes_entrenado_iris: S({ ...IRIS, tab: 2, train: true }),
  redes_entrenado_iris_lr01: S({ ...IRIS, tab: 2, train: true, learningRate: 0.1, epochs: 100 }),
  redes_entrenamiento_cancelado: S({ dataset: "california_housing", tab: 2, train: true, epochs: 500, cancelAt: 3 }),
  redes_arquitectura_cancer_softmax_heredado: inheritedOutput,
  redes_arquitectura_cancer: S({ dataset: "breast_cancer", tab: 1 }),
  redes_entrenado_cancer: S({ dataset: "breast_cancer", tab: 2, train: true }),
  redes_arquitectura_diabetes: S({ dataset: "diabetes", tab: 1 }),
  redes_entrenado_diabetes: S({ dataset: "diabetes", tab: 2, train: true }),
  redes_error_salida_sigmoid_iris: S({ ...IRIS, tab: 2, output: "sigmoid" }),
  redes_evaluacion_sin_entrenar: S({ ...IRIS, tab: 3 }),
  // F5c: 📈 Evaluación and 🎯 Visualización (Qt captures 14–22, 32–33, 37–39b, 41).
  redes_evaluacion_iris: S({ ...IRIS, tab: 3, train: true, ready: DIAG }),
  redes_evaluacion_iris_expandido: S({ ...IRIS, tab: 3, train: true, ready: DIAG, expandAll: true }),
  redes_historial_iris: S({ ...IRIS, tab: 4, train: true, viz: "Historial", ready: CARDS }),
  redes_pesos_iris: S({ ...IRIS, tab: 4, train: true, viz: "Pesos", ready: WEIGHTS }),
  redes_pesos_iris_expandido: S({ ...IRIS, tab: 4, train: true, viz: "Pesos", ready: WEIGHTS, expandAll: true }),
  redes_superficie_iris: S({ ...IRIS, tab: 4, train: true, viz: "Superficie", ready: GRID }),
  redes_superficie_petalos_iris: S({ ...IRIS, tab: 4, train: true, viz: "Superficie", pair: [2, 3], ready: GRID }),
  redes_activaciones_iris: S({ ...IRIS, tab: 4, train: true, viz: "Activaciones", ready: MAP }),
  redes_activaciones_iris_expandido: S({ ...IRIS, tab: 4, train: true, viz: "Activaciones", ready: MAP, expandAll: true, layerCharts: true }),
  redes_evaluacion_cancer: S({ dataset: "breast_cancer", tab: 3, train: true, ready: DIAG }),
  redes_superficie_cancer: S({ dataset: "breast_cancer", tab: 4, train: true, viz: "Superficie", ready: GRID }),
  redes_evaluacion_diabetes: S({ dataset: "diabetes", tab: 3, train: true, ready: DIAG }),
  redes_historial_diabetes: S({ dataset: "diabetes", tab: 4, train: true, viz: "Historial", ready: CARDS }),
  redes_superficie_diabetes: S({ dataset: "diabetes", tab: 4, train: true, viz: "Superficie", ready: GRID }),
  redes_activaciones_diabetes: S({ dataset: "diabetes", tab: 4, train: true, viz: "Activaciones", ready: MAP }),
  redes_superficie_moons: S({ dataset: "moons", tab: 4, train: true, viz: "Superficie", ready: GRID }),
  // NEW: K1 (Diabetes with tanh and SGD at LR 0.3 diverges in epoch 7) and the biggest network.
  redes_evaluacion_divergencia: S({ ...DIVERGES, tab: 3, ready: "[data-nn-diverged]" }),
  redes_activaciones_5x256: S({ dataset: "digits", tab: 4, train: true, layers: 5, neurons: [256, 256, 256, 256, 256], epochs: 10, viz: "Activaciones", ready: MAP }),
  // F5d: 🔮 Predicciones and 💾 Exportar (Qt captures 00d, 00e, 23–29, 34, 40; the rest NEW).
  redes_predicciones_sin_entrenar: S({ ...IRIS, tab: 5 }),
  redes_exportar_sin_entrenar: S({ ...IRIS, tab: 6 }),
  redes_prediccion_formulario_iris: S({ ...IRIS, tab: 5, train: true, ready: "[data-nn-predict-single] [data-predict-button]" }),
  redes_prediccion_resultado_iris: S({ ...IRIS, tab: 5, train: true, predict: true }),
  redes_prediccion_resultado_cancer: S({ dataset: "breast_cancer", tab: 5, train: true, predict: true }),
  redes_prediccion_resultado_diabetes: S({ dataset: "diabetes", tab: 5, train: true, predict: true }),
  redes_lotes_formato_iris: S({ ...IRIS, tab: 5, train: true, subview: "Lotes", ready: "[data-nn-batch-pick]" }),
  redes_lotes_iris: S({ ...IRIS, tab: 5, train: true, subview: "Lotes", batchCsv: true }),
  redes_lotes_diabetes: S({ dataset: "diabetes", tab: 5, train: true, subview: "Lotes", batchCsv: true }),
  redes_exploracion_iris: S({ ...IRIS, tab: 5, train: true, subview: "Exploracion", explore: [0, 2] }),
  redes_exploracion_iris_expandido: S({ ...IRIS, tab: 5, train: true, subview: "Exploracion", explore: [0, 2], expandAll: true }),
  redes_exploracion_cancer: S({ dataset: "breast_cancer", tab: 5, train: true, subview: "Exploracion", explore: [0, 0] }),
  redes_exploracion_diabetes: S({ dataset: "diabetes", tab: 5, train: true, subview: "Exploracion", explore: [0, 2] }),
  redes_prediccion_divergencia: S({ ...DIVERGES, tab: 5, ready: "[data-nn-diverged]" }),
  redes_exportar_keras: S({ ...IRIS, tab: 6, train: true, exportView: "Keras", exportSave: "keras_package" }),
  redes_exportar_onnx: S({ ...IRIS, tab: 6, train: true, exportView: "Onnx", exportSave: "onnx" }),
  redes_exportar_python: S({ ...IRIS, tab: 6, train: true, exportView: "Python", ready: '[data-nn-export] [data-export-format="nn_numpy_predictor"]' }),
  redes_exportar_metadatos: S({ ...IRIS, tab: 6, train: true, exportView: "Metadatos", ready: "[data-nn-metadata]" }),
  redes_exportar_divergencia: S({ ...DIVERGES, tab: 6, ready: "[data-nn-diverged]" }),
  redes_pesos_5x256: S({ dataset: "digits", tab: 4, train: true, layers: 5, neurons: [256, 256, 256, 256, 256], epochs: 10, viz: "Pesos", ready: WEIGHTS }),
};
