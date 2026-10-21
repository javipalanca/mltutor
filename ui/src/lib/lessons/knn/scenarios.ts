/**
 * `--screenshot --scenario` screens of 🔍 K-Nearest Neighbors, named after the Qt captures
 * of `docs/migration/capturas/qt/fase4/` (fase4-knn.md §0) with the prefix `knn_`. Also
 * used by the smoke test.
 */
import { logToBackend } from "../../backend";
import { busy } from "../../busy.svelte";
import type { DatasetId } from "../../bindings/DatasetId";
import type { KnnMetric } from "../../bindings/KnnMetric";
import type { KnnWeights } from "../../bindings/KnnWeights";
import { nextPaint, waitFor, waitForIdle } from "../../dom";
import { models } from "../../ml/modelSession.svelte";
import { router } from "../../router.svelte";
import { chooseDataset } from "../../selection";
import { session } from "../../session.svelte";
import { expandAll } from "../regression/scenarios";
import { knn, LESSON } from "./state.svelte";

export interface KnnScenario {
  dataset?: DatasetId;
  k?: number;
  weights?: KnnWeights;
  metric?: KnnMetric;
  testSize?: number;
  train?: boolean;
  tab: number;
  viz?: string;
  /** Pair of the boundary/surface. */
  pair?: [number, number];
  /** Pair of the canvas. */
  canvasPair?: [number, number];
  /** Test points of the canvas (data coordinates). */
  points?: [number, number][];
  predict?: boolean;
  /** Expanders to open (ids of `session.expanders`). */
  open?: string[];
  expandAll?: boolean;
  /** Press the 📄 training-code button of 💾 Exportar. */
  code?: boolean;
}

export async function knnScenario(options: KnnScenario): Promise<void> {
  const dataset = { kind: "builtin", id: options.dataset ?? "iris" } as const;
  await chooseDataset(dataset);
  knn.reset();
  knn.k = options.k ?? 5;
  knn.weights = options.weights ?? "uniform";
  knn.metric = options.metric ?? "minkowski";
  knn.testSize = options.testSize ?? 0.3;
  const state = models.lesson(LESSON);
  if (options.train) {
    void logToBackend("debug", `escenario: entrenando KNN en ${dataset.id}`);
    await models.train(LESSON, dataset, knn.config(), knn.testSize);
    if (state.trainError && options.tab !== 1) {
      throw new Error(state.trainError);
    }
  }
  if (state.info) {
    knn.syncPlayground(state.info);
    if (options.canvasPair) {
      knn.setPair(options.canvasPair);
    }
    for (const [x, y] of options.points ?? []) {
      knn.addPoint(x, y);
    }
  }
  state.viz = options.viz ?? null;
  state.pair = options.pair ?? [0, 1];
  for (const id of options.open ?? []) {
    session.setExpander(id, true);
  }
  session.activeTab[LESSON] = options.tab;
  router.go(LESSON);
  await nextPaint();
  if (options.viz === "Interactiva" && state.info && (options.canvasPair ?? [0, 1])[0] !== (options.canvasPair ?? [0, 1])[1]) {
    await waitFor('[data-playground][data-playground-ready="true"] [data-plot-rendered="true"]', 30000, "el lienzo interactivo");
    if (options.points?.length) {
      await waitFor(`[data-playground][data-points="${options.points.length}"] [data-test-points]`, 30000, "los puntos de prueba");
    }
  }
  if (options.predict) {
    const button = await waitFor("[data-predict-button]", 30000, "el botón de predecir");
    (button as HTMLButtonElement).click();
    await waitFor("[data-prediction-result]", 30000, "la predicción");
  }
  if (options.code) {
    const button = await waitFor("[data-export-panel] button", 30000, "el botón del código de entrenamiento");
    (button as HTMLButtonElement).click();
    await nextPaint();
  }
  await waitForIdle(() => busy.active, 60000);
  if (options.expandAll) {
    await expandAll();
  }
}

const S = (options: KnnScenario) => (): Promise<void> => knnScenario(options);
const IRIS = { dataset: "iris", train: true } as const;
const CANCER = { dataset: "breast_cancer", train: true } as const;
const TITANIC = { dataset: "titanic", train: true } as const;
const K20 = { dataset: "iris", k: 20, weights: "distance", metric: "manhattan", train: true } as const;
const DIABETES = { dataset: "diabetes", train: true } as const;
const CALIFORNIA = { dataset: "california_housing", train: true } as const;
/** Three test points like `09d_interactiva_iris_lienzo_con_puntos` (setosa, the overlap, virginica). */
export const IRIS_POINTS: [number, number][] = [
  [5.0, 3.4],
  [6.2, 2.85],
  [6.9, 3.2],
];

/** Names = Qt captures without the number (`09_interactiva_iris` → `knn_interactiva_iris`). */
export const KNN_SCENARIOS: Readonly<Record<string, () => Promise<void>>> = {
  knn_evaluacion_sin_entrenar: S({ tab: 2 }),
  knn_visualizacion_sin_entrenar: S({ tab: 3 }),
  knn_caracteristicas_sin_entrenar: S({ tab: 4 }),
  knn_entrenamiento_iris: S({ dataset: "iris", tab: 1 }),
  knn_entrenado_iris: S({ ...IRIS, tab: 1 }),
  knn_evaluacion_clasificacion_iris: S({ ...IRIS, tab: 2 }),
  knn_evaluacion_clasificacion_iris_expandido: S({ ...IRIS, tab: 2, expandAll: true }),
  knn_probabilidades_multiclase_iris: S({ ...IRIS, tab: 3, viz: "Probabilidades" }),
  knn_roc_multiclase_iris: S({ ...IRIS, tab: 3, viz: "ROC" }),
  knn_frontera_iris: S({ ...IRIS, tab: 3, viz: "Frontera" }),
  knn_frontera_petalos_iris: S({ ...IRIS, tab: 3, viz: "Frontera", pair: [2, 3] }),
  knn_interactiva_iris: S({ ...IRIS, tab: 3, viz: "Interactiva" }),
  knn_interactiva_iris_con_puntos: S({ ...IRIS, tab: 3, viz: "Interactiva", points: IRIS_POINTS }),
  knn_interactiva_iris_expandido: S({ ...IRIS, tab: 3, viz: "Interactiva", points: IRIS_POINTS, open: ["knn-lienzo-guia", "knn-lienzo-exacto"] }),
  knn_interactiva_misma_caracteristica: S({ ...IRIS, tab: 3, viz: "Interactiva", canvasPair: [0, 0] }),
  knn_distancias_iris: S({ ...IRIS, tab: 3, viz: "Distancias" }),
  knn_distancias_iris_expandido: S({ ...IRIS, tab: 3, viz: "Distancias", open: ["knn-distancias-interpretar"] }),
  knn_caracteristicas_iris: S({ ...IRIS, tab: 4 }),
  knn_caracteristicas_iris_expandido: S({ ...IRIS, tab: 4, expandAll: true }),
  knn_prediccion_formulario_iris: S({ ...IRIS, tab: 5 }),
  knn_prediccion_resultado_iris: S({ ...IRIS, tab: 5, predict: true }),
  knn_exportar_iris: S({ ...IRIS, tab: 6 }),
  knn_exportar_codigo: S({ ...IRIS, tab: 6, code: true }),
  knn_entrenado_cancer: S({ ...CANCER, tab: 1 }),
  knn_probabilidades_umbrales_cancer: S({ ...CANCER, tab: 3, viz: "Probabilidades" }),
  knn_probabilidades_umbrales_cancer_expandido: S({ ...CANCER, tab: 3, viz: "Probabilidades", expandAll: true }),
  knn_roc_binaria_cancer: S({ ...CANCER, tab: 3, viz: "ROC" }),
  knn_frontera_binaria_cancer: S({ ...CANCER, tab: 3, viz: "Frontera" }),
  knn_interactiva_cancer: S({ ...CANCER, tab: 3, viz: "Interactiva", points: [[14, 20], [18, 22]] }),
  knn_distancias_cancer: S({ ...CANCER, tab: 3, viz: "Distancias" }),
  knn_prediccion_formulario_titanic: S({ ...TITANIC, tab: 5 }),
  knn_prediccion_resultado_titanic: S({ ...TITANIC, tab: 5, predict: true }),
  knn_entrenamiento_k20_distance_manhattan: S({ ...K20, tab: 1 }),
  knn_interactiva_k20_distance_manhattan: S({ ...K20, tab: 3, viz: "Interactiva", points: IRIS_POINTS }),
  knn_frontera_k20_distance_manhattan: S({ ...K20, tab: 3, viz: "Frontera" }),
  knn_prediccion_resultado_k20_distance_manhattan: S({ ...K20, tab: 5, predict: true }),
  knn_entrenamiento_regresion_diabetes: S({ dataset: "diabetes", tab: 1 }),
  knn_entrenado_regresion_diabetes: S({ ...DIABETES, tab: 1 }),
  knn_evaluacion_regresion_diabetes: S({ ...DIABETES, tab: 2 }),
  knn_predicciones_residuos_diabetes: S({ ...DIABETES, tab: 3, viz: "Predicciones" }),
  knn_predicciones_residuos_diabetes_expandido: S({ ...DIABETES, tab: 3, viz: "Predicciones", expandAll: true }),
  knn_superficie_diabetes: S({ ...DIABETES, tab: 3, viz: "Superficie" }),
  knn_superficie_bmi_s5_diabetes: S({ ...DIABETES, tab: 3, viz: "Superficie", pair: [2, 8] }),
  knn_caracteristicas_regresion_diabetes: S({ ...DIABETES, tab: 4 }),
  knn_prediccion_regresion_diabetes: S({ ...DIABETES, tab: 5, predict: true }),
  knn_exportar_regresion_diabetes: S({ ...DIABETES, tab: 6 }),
  knn_superficie_california: S({ ...CALIFORNIA, tab: 3, viz: "Superficie" }),
  knn_caracteristicas_california: S({ ...CALIFORNIA, tab: 4 }),
};
