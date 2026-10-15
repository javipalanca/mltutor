/**
 * `--screenshot --scenario` screens of 🌲 Árboles de Decisión, named after the Qt captures of
 * `docs/migration/capturas/qt/fase3/` (fase3-arboles.md §0) with the prefix `arboles_`.
 * Also used by the smoke test.
 */
import { logToBackend } from "../../backend";
import { busy } from "../../busy.svelte";
import type { DatasetId } from "../../bindings/DatasetId";
import type { ImportanceMethod } from "../../bindings/ImportanceMethod";
import type { TreeCriterion } from "../../bindings/TreeCriterion";
import type { TreeTask } from "../../bindings/TreeTask";
import { nextPaint, waitFor, waitForIdle } from "../../dom";
import { models } from "../../ml/modelSession.svelte";
import { router } from "../../router.svelte";
import { chooseDataset } from "../../selection";
import { datasetKey, session } from "../../session.svelte";
import { expandAll } from "../regression/scenarios";
import { LESSON, trees } from "./state.svelte";

export interface TreesScenario {
  dataset?: DatasetId;
  /** `null`/absent: the task of the dataset (D37). */
  task?: TreeTask;
  criterion?: TreeCriterion;
  depth?: number;
  minSamples?: number;
  train?: boolean;
  tab: number;
  viz?: string;
  pair?: [number, number];
  figure?: [number, number];
  importance?: ImportanceMethod;
  predict?: boolean;
  /** Expanders to open (ids of `session.expanders`). */
  open?: string[];
  expandAll?: boolean;
  /** Press the 📄 training-code button of 💾 Exportar. */
  code?: boolean;
}

export async function treesScenario(options: TreesScenario): Promise<void> {
  const dataset = { kind: "builtin", id: options.dataset ?? "iris" } as const;
  const summary = await chooseDataset(dataset);
  trees.reset();
  if (summary) {
    // What 🏋️ Entrenamiento does on entering (D37), so that it does not override `task`.
    trees.preset(datasetKey(summary.dataset), summary.task);
  }
  if (options.task) {
    trees.setTask(options.task);
  }
  if (options.criterion) {
    trees.criterion = options.criterion;
  }
  trees.maxDepth = options.depth ?? 3;
  trees.minSamplesSplit = options.minSamples ?? 2;
  trees.importance = options.importance ?? "native";
  [trees.figWidth, trees.figHeight] = options.figure ?? [14, 10];
  const state = models.lesson(LESSON);
  if (options.train) {
    void logToBackend("debug", `escenario: entrenando árbol en ${dataset.id}`);
    await models.train(LESSON, dataset, trees.config(), trees.testSize);
    if (state.trainError && options.tab !== 1) {
      throw new Error(state.trainError);
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
  if (options.predict) {
    const button = await waitFor("[data-predict-button]", 30000, "el botón de predecir");
    (button as HTMLButtonElement).click();
    await waitFor("[data-decision-path]", 30000, "el camino de decisión");
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

const S = (options: TreesScenario) => (): Promise<void> => treesScenario(options);
const IRIS = { dataset: "iris", train: true } as const;
const CANCER_D10 = { dataset: "breast_cancer", criterion: "entropy", depth: 10, train: true } as const;
const TITANIC = { dataset: "titanic", depth: 4, train: true } as const;
const DIABETES = { dataset: "diabetes", train: true } as const;

/** Names = Qt captures without the number (`05_arbol_iris` → `arboles_arbol_iris`). */
export const TREES_SCENARIOS: Readonly<Record<string, () => Promise<void>>> = {
  arboles_evaluacion_sin_entrenar: S({ tab: 2 }),
  arboles_visualizacion_sin_entrenar: S({ tab: 3 }),
  arboles_entrenamiento_iris: S({ dataset: "iris", tab: 1 }),
  arboles_entrenado_iris: S({ ...IRIS, tab: 1 }),
  arboles_evaluacion_clasificacion_iris: S({ ...IRIS, tab: 2 }),
  arboles_evaluacion_clasificacion_iris_expandido: S({ ...IRIS, tab: 2, expandAll: true }),
  arboles_arbol_iris: S({ ...IRIS, tab: 3, viz: "Árbol" }),
  arboles_arbol_iris_20x15: S({ ...IRIS, tab: 3, viz: "Árbol", figure: [20, 15] }),
  arboles_arbol_iris_codigo: S({ ...IRIS, tab: 3, viz: "Árbol", open: ["arboles-codigo-arbol"] }),
  arboles_texto_iris: S({ ...IRIS, tab: 3, viz: "Texto" }),
  arboles_frontera_iris: S({ ...IRIS, tab: 3, viz: "Frontera" }),
  arboles_frontera_petalos_iris: S({ ...IRIS, tab: 3, viz: "Frontera", pair: [2, 3] }),
  arboles_roc_iris: S({ ...IRIS, tab: 3, viz: "ROC" }),
  arboles_caracteristicas_iris: S({ ...IRIS, tab: 4 }),
  arboles_caracteristicas_iris_expandido: S({ ...IRIS, tab: 4, expandAll: true }),
  arboles_caracteristicas_permutacion_iris: S({ ...IRIS, tab: 4, importance: "permutation" }),
  arboles_prediccion_formulario_iris: S({ ...IRIS, tab: 5 }),
  arboles_prediccion_camino_iris: S({ ...IRIS, tab: 5, predict: true }),
  arboles_prediccion_camino_iris_codigo: S({ ...IRIS, tab: 5, predict: true, open: ["arboles-codigo-camino"] }),
  arboles_exportar_iris: S({ ...IRIS, tab: 6 }),
  arboles_exportar_iris_expandido: S({ ...IRIS, tab: 6, open: ["arboles-vista-previa"] }),
  arboles_exportar_codigo_entrenamiento: S({ ...IRIS, tab: 6, code: true }),
  arboles_entrenamiento_cancer_entropy_d10: S({ ...CANCER_D10, tab: 1 }),
  arboles_arbol_cancer_d10: S({ ...CANCER_D10, tab: 3, viz: "Árbol" }),
  arboles_roc_binaria_cancer: S({ dataset: "breast_cancer", train: true, tab: 3, viz: "ROC" }),
  arboles_prediccion_formulario_titanic: S({ ...TITANIC, tab: 5 }),
  arboles_prediccion_camino_titanic: S({ ...TITANIC, tab: 5, predict: true }),
  arboles_aviso_clasificacion_en_diabetes: S({ dataset: "diabetes", task: "classification", tab: 1 }),
  arboles_entrenamiento_regresion_diabetes: S({ dataset: "diabetes", tab: 1 }),
  arboles_entrenado_regresion_diabetes: S({ ...DIABETES, tab: 1 }),
  arboles_evaluacion_regresion_diabetes: S({ ...DIABETES, tab: 2 }),
  arboles_arbol_regresion_diabetes: S({ ...DIABETES, tab: 3, viz: "Árbol" }),
  arboles_texto_regresion_diabetes: S({ ...DIABETES, tab: 3, viz: "Texto" }),
  arboles_residuos_diabetes: S({ ...DIABETES, tab: 3, viz: "Residuos" }),
  arboles_superficie_diabetes: S({ ...DIABETES, tab: 3, viz: "Superficie" }),
  arboles_superficie_bmi_s5_diabetes: S({ ...DIABETES, tab: 3, viz: "Superficie", pair: [2, 8] }),
  arboles_caracteristicas_regresion_diabetes: S({ ...DIABETES, tab: 4 }),
  arboles_caracteristicas_regresion_diabetes_expandido: S({ ...DIABETES, tab: 4, expandAll: true }),
  arboles_prediccion_regresion_diabetes: S({ ...DIABETES, tab: 5, predict: true }),
  arboles_exportar_regresion_diabetes: S({ ...DIABETES, tab: 6 }),
  arboles_arbol_california_d8: S({ dataset: "california_housing", depth: 8, minSamples: 10, train: true, tab: 3, viz: "Árbol" }),
  arboles_error_clasificacion_california: S({ dataset: "california_housing", task: "classification", train: true, tab: 1 }),
  arboles_clasificacion_bloqueada_diabetes: S({ dataset: "diabetes", task: "classification", train: true, tab: 1 }),
  arboles_aviso_regresion_en_iris: S({ dataset: "iris", task: "regression", tab: 1 }),
  arboles_regresion_entrenada_en_iris: S({ dataset: "iris", task: "regression", train: true, tab: 2 }),
};
