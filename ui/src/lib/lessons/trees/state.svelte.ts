/**
 * UI state of 🌲 Árboles de Decisión (fase3-arboles.md §1.1 and §5.2): the controls of
 * 🏋️ Entrenamiento and the options of tabs 3–4 that are not in `ModelLessonState`. The
 * model and the shared options (subview, pair, ROC, form) live in `models.lesson("arboles")`.
 */
import type { ImportanceMethod } from "../../bindings/ImportanceMethod";
import type { ModelConfig } from "../../bindings/ModelConfig";
import type { TaskType } from "../../bindings/TaskType";
import type { TreeCriterion } from "../../bindings/TreeCriterion";
import type { TreeTask } from "../../bindings/TreeTask";
import { arbList } from "../../content";

export const LESSON = "arboles" as const;

/** Options of "Criterio de División:" for each tree type (`content/arboles/mensajes.toml`). */
export function criterionOptions(task: TreeTask): TreeCriterion[] {
  const key = task === "classification" ? "criterio_opciones_clasificacion" : "criterio_opciones_regresion";
  return arbList("entrenamiento", key) as TreeCriterion[];
}

class TreesState {
  /** `tree_type`. D37: preselected with the task of the dataset (see `preset`). */
  task: TreeTask = $state("classification");
  criterion: TreeCriterion = $state("gini");
  /** 1..10 */
  maxDepth = $state(3);
  /** 2..10 */
  minSamplesSplit = $state(2);
  /** 0.1..0.5, step 0.05. */
  testSize = $state(0.3);
  /** "Ancho de figura:" 8..20 and "Alto de figura:" 6..15 (inches of matplotlib). */
  figWidth = $state(14);
  figHeight = $state(10);
  /** D38: 🌳 native (Python) or 🔀 permutation. */
  importance: ImportanceMethod = $state("native");
  /** Dataset whose task last preselected the type (D37). */
  #presetFor: string | null = null;

  setTask(task: TreeTask): void {
    this.task = task;
    // The selectbox has no key: an option missing from the new list goes back to the first.
    const options = criterionOptions(task);
    if (!options.includes(this.criterion)) {
      this.criterion = options[0] ?? (task === "classification" ? "gini" : "squared_error");
    }
  }

  /** D37: the first time a dataset is seen in 🏋️ Entrenamiento, take its task. */
  preset(datasetKey: string, task: TaskType): void {
    if (this.#presetFor !== datasetKey) {
      this.#presetFor = datasetKey;
      this.setTask(task);
    }
  }

  config(): ModelConfig {
    return {
      kind: "tree",
      task: this.task,
      criterion: this.criterion,
      max_depth: this.maxDepth,
      min_samples_split: this.minSamplesSplit,
    };
  }

  /** Back to the defaults (scenarios). */
  reset(): void {
    this.task = "classification";
    this.criterion = "gini";
    this.maxDepth = 3;
    this.minSamplesSplit = 2;
    this.testSize = 0.3;
    this.figWidth = 14;
    this.figHeight = 10;
    this.importance = "native";
    this.#presetFor = null;
  }
}

export const trees = new TreesState();
