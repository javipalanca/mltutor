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
};

export async function runSmokeCheck(name: string): Promise<void> {
  const check = CHECKS[name];
  if (!check) {
    throw new Error(`la interfaz no conoce esta comprobación: ${name}`);
  }
  await check();
}
