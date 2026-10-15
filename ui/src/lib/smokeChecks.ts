/**
 * Data checks of `--smoke-test` (fase1-shell-datos.md §8.2.5): 📊 Datos of a lesson with
 * Iris, then its correlation heatmap and its pairplot drawn by Plotly (WebGL included).
 */
import * as backend from "./backend";
import { busy } from "./busy.svelte";
import { hasWebGL } from "./charts";
import { waitFor, waitForIdle } from "./dom";
import { router } from "./router.svelte";
import { chooseDataset } from "./selection";
import { session } from "./session.svelte";
import { regressionScenario, type RegressionScenario } from "./lessons/regression/scenarios";
import { IRIS_POINTS, knnScenario, type KnnScenario } from "./lessons/knn/scenarios";
import { knn } from "./lessons/knn/state.svelte";
import { models } from "./ml/modelSession.svelte";
import { treesScenario, type TreesScenario } from "./lessons/trees/scenarios";

const LESSON = "regresion";
const IRIS = { kind: "builtin", id: "iris" } as const;

async function datosVistas(): Promise<void> {
  await chooseDataset(IRIS);
  session.activeTab[LESSON] = 0;
  router.go(LESSON);
  await waitFor(`[data-dataset-tab="${LESSON}"][data-ready="true"]`, 30000, "la pestaña 📊 Datos");
  await waitFor(`[data-dataset-tab="${LESSON}"] table`, 10000, "las tablas");
  await waitFor(`[data-dataset-tab="${LESSON}"] [data-plot-rendered="true"]`, 30000, "el gráfico del objetivo");
}

async function correlacion(): Promise<void> {
  session.datasetTab(LESSON, IRIS).analysis = "correlation";
  await waitFor('[data-analysis="correlation"] [data-plot-rendered="true"]', 30000, "el heatmap de correlación");
  const cells = document.querySelectorAll('[data-analysis="correlation"] .heatmaplayer, [data-analysis="correlation"] image');
  if (cells.length === 0) {
    throw new Error("el heatmap no tiene celdas");
  }
}

async function pairplot(): Promise<void> {
  const state = session.datasetTab(LESSON, IRIS);
  state.analysis = "pairplot";
  state.generated = true;
  await waitFor('[data-analysis="pairplot"] [data-plot-rendered="true"]', 30000, "el pairplot");
  // scattergl draws on a WebGL canvas: it fails if the CSP breaks it. Without WebGL the
  // pairplot falls back to SVG, which is logged.
  if (hasWebGL()) {
    await waitFor('[data-analysis="pairplot"] canvas', 10000, "el canvas WebGL del pairplot");
  } else {
    void backend.logToBackend("warn", "WebGL no disponible: el pairplot usa SVG");
  }
  await waitForIdle(() => busy.active);
}

/** Opens a screen of 📊 Regresión and waits for `selector` and its charts. */
async function regressionView(options: RegressionScenario, selector: string, what: string): Promise<void> {
  void backend.logToBackend("info", `smoke test: ${what}`);
  await regressionScenario(options);
  await waitFor(selector, 30000, what);
  await waitForIdle(() => busy.active, 60000);
  const error = document.querySelector('[data-content] .notice.error, [data-plot-rendered="error"]');
  if (error) {
    throw new Error(`${what}: ${error.textContent?.trim() ?? "error"}`);
  }
}

/** Every tab of a trained model (fase2-regresion.md §9.2.4). */
async function regressionChecks(base: Omit<RegressionScenario, "tab">, vizs: readonly [string, string][]): Promise<void> {
  await regressionView({ ...base, train: true, tab: 1 }, "[data-train-tab] .notice.success", "el entrenamiento");
  await regressionView({ ...base, tab: 2 }, "[data-evaluation] [data-plot-rendered=\"true\"]", "la evaluación");
  for (const [viz, selector] of vizs) {
    await regressionView({ ...base, tab: 3, viz }, `${selector} [data-plot-rendered="true"]`, `la vista ${viz}`);
  }
  await regressionView({ ...base, tab: 4 }, "[data-regression-tab=\"4\"] table", "los coeficientes");
  await regressionView({ ...base, tab: 5, predict: true }, "[data-prediction-result]", "la predicción");
  await regressionView({ ...base, tab: 6, code: true }, "[data-export-panel] h4", "el código generado");
}

/** Opens a screen of 🌲 Árboles and waits for `selector`. */
async function treesView(options: TreesScenario, selector: string, what: string): Promise<void> {
  void backend.logToBackend("info", `smoke test: árboles, ${what}`);
  await treesScenario(options);
  await waitFor(selector, 30000, what);
  await waitForIdle(() => busy.active, 60000);
  const error = document.querySelector('[data-content] .notice.error, [data-plot-rendered="error"]');
  if (error) {
    throw new Error(`${what}: ${error.textContent?.trim() ?? "error"}`);
  }
}

/** Every tab of a trained tree (fase3-arboles.md §9.2.3). */
async function treesChecks(base: Omit<TreesScenario, "tab">, vizs: readonly [string, string][]): Promise<void> {
  await treesView({ ...base, train: true, tab: 1 }, "[data-train-tab] .notice.success", "el entrenamiento");
  await treesView({ ...base, tab: 2 }, "[data-evaluation] [data-plot-rendered=\"true\"]", "la evaluación");
  for (const [viz, selector] of vizs) {
    await treesView({ ...base, tab: 3, viz }, selector, `la vista ${viz}`);
  }
  await treesView({ ...base, tab: 4 }, "[data-importance=\"native\"] [data-plot-rendered=\"true\"]", "la importancia nativa");
  await treesView({ ...base, tab: 4, importance: "permutation" }, "[data-importance=\"permutation\"] [data-plot-rendered=\"true\"]", "la importancia por permutación");
  await treesView({ ...base, tab: 5, predict: true }, "[data-decision-path]", "el camino de decisión");
  await treesView({ ...base, tab: 6, open: ["arboles-vista-previa"] }, "[data-export-panel] [data-export-format=\"tree_python\"]", "la exportación");
}

/** Opens a screen of 🔍 KNN and waits for `selector`. */
async function knnView(options: KnnScenario, selector: string, what: string): Promise<void> {
  void backend.logToBackend("info", `smoke test: knn, ${what}`);
  await knnScenario(options);
  await waitFor(selector, 30000, what);
  await waitForIdle(() => busy.active, 60000);
  const error = document.querySelector('[data-content] .notice.error, [data-plot-rendered="error"]');
  if (error) {
    throw new Error(`${what}: ${error.textContent?.trim() ?? "error"}`);
  }
}

/**
 * Round trips of `knn_playground` like a drag (one request after another, 3 and 20 test
 * points), logged for fase4-knn.md (presupuesto < 10 ms con 20 puntos).
 */
async function playgroundLatency(): Promise<void> {
  const info = models.lesson("knn").info;
  if (!info) {
    throw new Error("no hay modelo KNN");
  }
  const pg = knn.playground;
  for (const n of [3, 20]) {
    const times: number[] = [];
    for (let i = 0; i < 40; i += 1) {
      const points = Array.from({ length: n }, (_, j) => [5 + ((i + j) % 20) * 0.1, 2.5 + (j % 10) * 0.15] as [number, number]);
      const started = performance.now();
      await backend.knnPlayground({
        model_id: info.model_id,
        feature_x: pg.pair[0],
        feature_y: pg.pair[1],
        n_neighbors: pg.k,
        weights: pg.weights,
        metric: pg.metric,
        points,
        include_data: false,
      }).promise;
      times.push(performance.now() - started);
    }
    times.sort((a, b) => a - b);
    const median = times[Math.floor(times.length / 2)] ?? 0;
    const p95 = times[Math.floor(times.length * 0.95)] ?? 0;
    void backend.logToBackend("info", `smoke test: knn_playground con ${n} puntos: mediana ${median.toFixed(1)} ms, p95 ${p95.toFixed(1)} ms`);
  }
}

/**
 * Drags the first test point of the canvas with real pointer events (down, 20 moves, up)
 * and checks that it moved; logs the time from each move to the next Plotly redraw.
 */
async function playgroundDrag(): Promise<void> {
  const plot = document.querySelector<HTMLElement>("[data-playground] .plot");
  const layout = (plot as unknown as { _fullLayout?: { xaxis: { _offset: number; l2p(v: number): number }; yaxis: { _offset: number; l2p(v: number): number } } } | null)?._fullLayout;
  const first = knn.playground.points[0];
  if (!plot || !layout || !first) {
    throw new Error("el lienzo no está listo para arrastrar");
  }
  const rect = plot.getBoundingClientRect();
  const at = (x: number, y: number): { clientX: number; clientY: number } => ({
    clientX: rect.left + layout.xaxis._offset + layout.xaxis.l2p(x),
    clientY: rect.top + layout.yaxis._offset + layout.yaxis.l2p(y),
  });
  const start = at(first.x, first.y);
  const fire = (type: string, dx: number): void => {
    plot.dispatchEvent(new PointerEvent(type, { bubbles: true, cancelable: true, pointerId: 41, button: 0, clientX: start.clientX + dx, clientY: start.clientY - dx / 2 }));
  };
  // Timers, not frames: the smoke-test window may be hidden (no requestAnimationFrame).
  const frame = (): Promise<void> => new Promise((done) => setTimeout(done, 4));
  fire("pointerdown", 0);
  const cycles: number[] = [];
  for (let i = 1; i <= 20; i += 1) {
    const before = plot.dataset["renderedAt"];
    const sent = performance.now();
    fire("pointermove", i * 6);
    const deadline = sent + 2000;
    while (plot.dataset["renderedAt"] === before && performance.now() < deadline) {
      await frame();
    }
    cycles.push(performance.now() - sent);
  }
  fire("pointerup", 120);
  await waitForIdle(() => busy.active, 10000);
  const moved = knn.playground.points[0];
  if (!moved || Math.abs(moved.x - first.x) < 1e-6) {
    throw new Error("el punto de prueba no se ha movido al arrastrarlo");
  }
  cycles.sort((a, b) => a - b);
  const median = cycles[Math.floor(cycles.length / 2)] ?? 0;
  const worst = cycles[cycles.length - 1] ?? 0;
  void backend.logToBackend("info", `smoke test: arrastre en el lienzo, de mover el puntero a redibujar: mediana ${median.toFixed(0)} ms, máximo ${worst.toFixed(0)} ms`);
}

/** Every tab of a trained KNN (fase4-knn.md §9.2.3). */
async function knnChecks(base: Omit<KnnScenario, "tab">, vizs: readonly [string, string][]): Promise<void> {
  await knnView({ ...base, train: true, tab: 1 }, "[data-train-tab] .notice.success", "el entrenamiento");
  await knnView({ ...base, tab: 2 }, "[data-evaluation] [data-plot-rendered=\"true\"]", "la evaluación");
  for (const [viz, selector] of vizs) {
    await knnView({ ...base, tab: 3, viz, ...(viz === "Interactiva" ? { points: IRIS_POINTS } : {}) }, selector, `la vista ${viz}`);
    if (viz === "Interactiva" && base.dataset === "iris") {
      await playgroundLatency();
      await playgroundDrag();
    }
  }
  await knnView({ ...base, tab: 4 }, "[data-importance=\"permutation\"] [data-plot-rendered=\"true\"]", "la importancia por permutación");
  await knnView({ ...base, tab: 5, predict: true }, "[data-knn-neighbors]", "los vecinos de la predicción");
  await knnView({ ...base, tab: 6, code: true }, "[data-export-panel] [data-export-format=\"python_package\"]", "la exportación");
}

const KNN_CLASSIFICATION: readonly [string, string][] = [
  ["Probabilidades", "[data-probabilities] [data-plot-rendered=\"true\"]"],
  ["ROC", "[data-roc] [data-plot-rendered=\"true\"]"],
  ["Frontera", "[data-grid-view] [data-plot-rendered=\"true\"]"],
  ["Interactiva", "[data-playground][data-points=\"3\"] [data-test-points]"],
  ["Distancias", "[data-class-distances] [data-plot-rendered=\"true\"]"],
];
const KNN_REGRESSION: readonly [string, string][] = [
  ["Predicciones", "[data-residuals] [data-plot-rendered=\"true\"]"],
  ["Superficie", "[data-grid-view] [data-plot-rendered=\"true\"]"],
];

const TREE_VIZ: readonly [string, string][] = [
  ["Árbol", "[data-tree-diagram] [data-node]"],
  ["Texto", "[data-tree-rules]"],
];

const CHECKS: Readonly<Record<string, () => Promise<void>>> = {
  datos_vistas: datosVistas,
  datos_correlacion: correlacion,
  datos_pairplot: pairplot,
  regresion_logistica_iris: () =>
    regressionChecks({ dataset: "iris", model: "logistic" }, [
      ["ROC", "[data-roc]"],
      ["Frontera", "[data-grid-view]"],
      ["Probs", "[data-probabilities]"],
    ]),
  regresion_binaria_cancer: () =>
    regressionChecks({ dataset: "breast_cancer", model: "logistic" }, [
      ["ROC", "[data-roc=\"binary\"]"],
      ["Frontera", "[data-grid-view]"],
      ["Probs", "[data-probabilities=\"binary\"]"],
    ]),
  regresion_lineal_diabetes: () =>
    regressionChecks({ dataset: "diabetes", model: "linear" }, [
      ["Residuos", "[data-residuals]"],
      ["Superficie", "[data-grid-view]"],
    ]),
  arboles_gini_iris: () =>
    treesChecks({ dataset: "iris" }, [
      ...TREE_VIZ,
      ["Frontera", "[data-grid-view] [data-plot-rendered=\"true\"]"],
      ["ROC", "[data-roc] [data-plot-rendered=\"true\"]"],
    ]),
  arboles_regresion_diabetes: () =>
    treesChecks({ dataset: "diabetes" }, [
      ...TREE_VIZ,
      ["Residuos", "[data-residuals] [data-plot-rendered=\"true\"]"],
      ["Superficie", "[data-grid-view] [data-plot-rendered=\"true\"]"],
    ]),
  arboles_binaria_cancer: () =>
    treesChecks({ dataset: "breast_cancer", criterion: "entropy", depth: 10 }, [
      ...TREE_VIZ,
      ["ROC", "[data-roc=\"binary\"] [data-plot-rendered=\"true\"]"],
    ]),
  knn_iris_k5: () => knnChecks({ dataset: "iris" }, KNN_CLASSIFICATION),
  knn_binaria_cancer_k10_distance: () =>
    knnChecks({ dataset: "breast_cancer", k: 10, weights: "distance", metric: "euclidean" }, [
      ["Probabilidades", "[data-probabilities=\"binary\"] [data-plot-rendered=\"true\"]"],
      ["ROC", "[data-roc=\"binary\"] [data-plot-rendered=\"true\"]"],
      ["Frontera", "[data-grid-view] [data-plot-rendered=\"true\"]"],
      // J2: binary has a verdict on the separation too (1.26 → moderate).
      ["Distancias", "[data-class-distances=\"moderate\"][data-n-classes=\"2\"] [data-plot-rendered=\"true\"]"],
    ]),
  knn_regresion_diabetes_k5: () => knnChecks({ dataset: "diabetes" }, KNN_REGRESSION),
  knn_california_k5: () => knnChecks({ dataset: "california_housing" }, KNN_REGRESSION),
};

export async function runSmokeCheck(name: string): Promise<void> {
  const check = CHECKS[name];
  if (!check) {
    throw new Error(`la interfaz no conoce esta comprobación: ${name}`);
  }
  await check();
}
