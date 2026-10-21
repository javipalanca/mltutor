/**
 * `--screenshot --scenario` screens of 📊 Regresión, named after the Qt captures of
 * `docs/migration/capturas/qt/fase2/` (fase2-regresion.md §0). Also used by the smoke test.
 */
import { logToBackend } from "../../backend";
import { busy } from "../../busy.svelte";
import type { DatasetId } from "../../bindings/DatasetId";
import { nextPaint, waitFor, waitForIdle } from "../../dom";
import { models } from "../../ml/modelSession.svelte";
import { router } from "../../router.svelte";
import { chooseDataset } from "../../selection";
import { session } from "../../session.svelte";
import type { Average } from "../../components/ml/figures";
import { LESSON, regression, type RegressionModelType } from "./state.svelte";

export interface RegressionScenario {
  dataset?: DatasetId;
  model?: RegressionModelType;
  /** Train before showing the tab. */
  train?: boolean;
  tab: number;
  viz?: string;
  pair?: [number, number];
  rocPerClass?: boolean;
  rocAverages?: Average[];
  coefClass?: number;
  /** Press "Realizar predicción" (default values). */
  predict?: boolean;
  /** Press "📄 Generar Código" in 💾 Exportar. */
  code?: boolean;
  /** Open every expander of the page (`expand_all` of the Qt captures). */
  expandAll?: boolean;
}

/** Clicks every closed expander until none is left (some appear when others open). */
export async function expandAll(): Promise<void> {
  for (let round = 0; round < 6; round += 1) {
    const closed = [...document.querySelectorAll<HTMLButtonElement>("[data-content] section.expander:not(.open) > button")];
    if (closed.length === 0) {
      return;
    }
    closed.forEach((button) => button.click());
    await waitForIdle(() => busy.active, 30000);
  }
}

export async function regressionScenario(options: RegressionScenario): Promise<void> {
  const dataset = { kind: "builtin", id: options.dataset ?? "iris" } as const;
  await chooseDataset(dataset);
  regression.modelType = options.model ?? "logistic";
  regression.maxIter = 1000;
  regression.testSize = 0.3;
  const state = models.lesson(LESSON);
  if (options.train) {
    void logToBackend("debug", `escenario: entrenando ${dataset.id}`);
    await models.train(LESSON, dataset, regression.config(), regression.testSize);
    if (state.trainError && options.tab !== 1) {
      throw new Error(state.trainError);
    }
  }
  state.viz = options.viz ?? null;
  state.pair = options.pair ?? [0, 1];
  state.rocPerClass = options.rocPerClass ?? true;
  state.rocAverages = options.rocAverages ?? ["macro", "micro"];
  state.coefClass = options.coefClass ?? 0;
  session.activeTab[LESSON] = options.tab;
  router.go(LESSON);
  void logToBackend("debug", `escenario: pestaña ${options.tab}`);
  await nextPaint();
  if (options.predict) {
    const button = await waitFor("[data-predict-button]", 30000, "el botón de predecir");
    (button as HTMLButtonElement).click();
    await waitFor("[data-prediction-result]", 30000, "el resultado de la predicción");
  }
  if (options.code) {
    const button = await waitFor("[data-export-panel] button", 30000, "el botón de generar código");
    (button as HTMLButtonElement).click();
    await nextPaint();
  }
  await waitForIdle(() => busy.active, 60000);
  if (options.expandAll) {
    await expandAll();
  }
}

const S = (options: RegressionScenario) => (): Promise<void> => regressionScenario(options);
const IRIS_LOG = { dataset: "iris", model: "logistic", train: true } as const;
const CANCER = { dataset: "breast_cancer", model: "logistic", train: true } as const;
const TITANIC = { dataset: "titanic", model: "logistic", train: true } as const;
const DIABETES = { dataset: "diabetes", model: "linear", train: true } as const;

/** Names = Qt captures without the number (`02_entrenado_logistica_iris` → `regresion_entrenado_logistica_iris`). */
export const REGRESSION_SCENARIOS: Readonly<Record<string, () => Promise<void>>> = {
  regresion_evaluacion_sin_entrenar: S({ tab: 2 }),
  regresion_visualizacion_sin_entrenar: S({ tab: 3 }),
  regresion_entrenamiento_logistica_iris: S({ dataset: "iris", model: "logistic", tab: 1 }),
  regresion_entrenado_logistica_iris: S({ ...IRIS_LOG, tab: 1 }),
  regresion_evaluacion_clasificacion_iris: S({ ...IRIS_LOG, tab: 2 }),
  regresion_evaluacion_clasificacion_iris_expandido: S({ ...IRIS_LOG, tab: 2, expandAll: true }),
  regresion_roc_multiclase_iris: S({ ...IRIS_LOG, tab: 3, viz: "ROC" }),
  regresion_roc_multiclase_iris_expandido: S({ ...IRIS_LOG, tab: 3, viz: "ROC", expandAll: true }),
  regresion_roc_micro_sin_clases_iris: S({ ...IRIS_LOG, tab: 3, viz: "ROC", rocPerClass: false, rocAverages: ["micro"] }),
  regresion_frontera_multiclase_iris: S({ ...IRIS_LOG, tab: 3, viz: "Frontera" }),
  regresion_frontera_petalos_iris: S({ ...IRIS_LOG, tab: 3, viz: "Frontera", pair: [2, 3] }),
  regresion_frontera_misma_caracteristica: S({ ...IRIS_LOG, tab: 3, viz: "Frontera", pair: [1, 1] }),
  regresion_probabilidades_multiclase_iris: S({ ...IRIS_LOG, tab: 3, viz: "Probs" }),
  regresion_coeficientes_logistica_iris: S({ ...IRIS_LOG, tab: 4 }),
  regresion_coeficientes_logistica_iris_expandido: S({ ...IRIS_LOG, tab: 4, expandAll: true }),
  regresion_coeficientes_clase_versicolor: S({ ...IRIS_LOG, tab: 4, coefClass: 1 }),
  regresion_prediccion_formulario_iris: S({ ...IRIS_LOG, tab: 5 }),
  regresion_prediccion_resultado_iris: S({ ...IRIS_LOG, tab: 5, predict: true }),
  regresion_exportar: S({ ...IRIS_LOG, tab: 6 }),
  regresion_exportar_codigo: S({ ...IRIS_LOG, tab: 6, code: true }),
  regresion_entrenado_cancer: S({ ...CANCER, tab: 1 }),
  regresion_evaluacion_binaria_cancer: S({ ...CANCER, tab: 2 }),
  regresion_roc_binaria_cancer: S({ ...CANCER, tab: 3, viz: "ROC" }),
  regresion_frontera_binaria_cancer: S({ ...CANCER, tab: 3, viz: "Frontera" }),
  regresion_probabilidades_umbrales_cancer: S({ ...CANCER, tab: 3, viz: "Probs" }),
  regresion_probabilidades_umbrales_cancer_expandido: S({ ...CANCER, tab: 3, viz: "Probs", expandAll: true }),
  regresion_prediccion_formulario_titanic: S({ ...TITANIC, tab: 5 }),
  regresion_prediccion_resultado_titanic: S({ ...TITANIC, tab: 5, predict: true }),
  regresion_coeficientes_logistica_titanic: S({ ...TITANIC, tab: 4 }),
  regresion_entrenamiento_lineal_diabetes: S({ dataset: "diabetes", model: "linear", tab: 1 }),
  regresion_entrenado_lineal_diabetes: S({ ...DIABETES, tab: 1 }),
  regresion_evaluacion_regresion_diabetes: S({ ...DIABETES, tab: 2 }),
  regresion_evaluacion_regresion_diabetes_expandido: S({ ...DIABETES, tab: 2, expandAll: true }),
  regresion_residuos_diabetes: S({ ...DIABETES, tab: 3, viz: "Residuos" }),
  regresion_residuos_diabetes_expandido: S({ ...DIABETES, tab: 3, viz: "Residuos", expandAll: true }),
  regresion_superficie_diabetes: S({ ...DIABETES, tab: 3, viz: "Superficie" }),
  regresion_superficie_bmi_s5_diabetes: S({ ...DIABETES, tab: 3, viz: "Superficie", pair: [2, 8] }),
  regresion_coeficientes_lineal_diabetes: S({ ...DIABETES, tab: 4 }),
  regresion_coeficientes_lineal_diabetes_expandido: S({ ...DIABETES, tab: 4, expandAll: true }),
  regresion_prediccion_formulario_diabetes: S({ ...DIABETES, tab: 5 }),
  regresion_prediccion_resultado_diabetes: S({ ...DIABETES, tab: 5, predict: true }),
  regresion_aviso_lineal_en_clasificacion_iris: S({ dataset: "iris", model: "linear", tab: 1 }),
  regresion_aviso_logistica_en_regresion_diabetes: S({ dataset: "diabetes", model: "logistic", tab: 1 }),
  // Measures (California: 20 640 rows, 6 192 in the test set).
  regresion_evaluacion_california: S({ dataset: "california_housing", model: "linear", train: true, tab: 2 }),
  regresion_superficie_california: S({ dataset: "california_housing", model: "linear", train: true, tab: 3, viz: "Superficie", pair: [0, 6] }),
  regresion_logistica_bloqueada_diabetes: S({ dataset: "diabetes", model: "logistic", train: true, tab: 1 }),
};
