/**
 * UI session state (fase1-shell-datos.md §5.2): what the backend does not need to
 * compute. It survives navigation between pages for the whole session.
 */
import type { CsvLoaded } from "./bindings/CsvLoaded";
import type { CsvPreview } from "./bindings/CsvPreview";
import type { DatasetEntry } from "./bindings/DatasetEntry";
import type { DatasetRef } from "./bindings/DatasetRef";
import type { DiagKind } from "./bindings/DiagKind";
import type { TaskChoice } from "./bindings/TaskChoice";
import type { LessonId } from "./content";

export type Analysis = "correlation" | "pairplot" | null;

/** State of the 📊 Datos tab, one per lesson (D13). */
export interface DatasetTabState {
  analysis: Analysis;
  diag: DiagKind;
  /** `null` until the user changes it: then `min(4, n)`. */
  maxFeatures: number | null;
  /** `null` = the first `maxFeatures` features. */
  features: string[] | null;
  /** Pairplot requested with "🚀 Generar…"; later changes regenerate it. */
  generated: boolean;
}

export interface CsvPageState {
  preview: CsvPreview | null;
  target: string | null;
  task: TaskChoice | null;
  loaded: CsvLoaded | null;
  error: string | null;
}

export function datasetKey(dataset: DatasetRef): string {
  return dataset.kind === "builtin" ? `builtin:${dataset.id}` : `csv:${dataset.id}`;
}

export function sameDataset(a: DatasetRef | null, b: DatasetRef | null): boolean {
  return a !== null && b !== null && datasetKey(a) === datasetKey(b);
}

function createDatasetTab(previous: DatasetTabState | undefined): DatasetTabState {
  // `$state` in a declaration: the object is deeply reactive. The analysis and the
  // diagonal type are kept when the dataset changes, as `active_analysis` in Python.
  const state = $state<DatasetTabState>({
    analysis: previous?.analysis ?? null,
    diag: previous?.diag ?? "kde",
    maxFeatures: null,
    features: null,
    generated: false,
  });
  return state;
}

class Session {
  /** Active tab of each lesson (`active_tab`, `active_tab_lr`…), 0 by default. */
  activeTab: Record<LessonId, number> = $state({ arboles: 0, regresion: 0, knn: 0, redes: 0 });
  /** 📊 Datos state by lesson and dataset (plain map: entries are reactive themselves). */
  #datasetTabs = new Map<string, DatasetTabState>();
  #lastDatasetTab = new Map<LessonId, DatasetTabState>();
  /** Open/closed state of expanders, by id. */
  expanders: Record<string, boolean> = $state({});
  /** Selector options (built-ins + loaded CSVs). */
  datasets: DatasetEntry[] = $state([]);
  /** Shared by every lesson and the CSV page, as in Python. */
  selected: DatasetRef = $state({ kind: "builtin", id: "iris" });
  csv: CsvPageState = $state({ preview: null, target: null, task: null, loaded: null, error: null });

  expander(id: string, defaultOpen: boolean): boolean {
    return this.expanders[id] ?? defaultOpen;
  }

  setExpander(id: string, open: boolean): void {
    this.expanders[id] = open;
  }

  /** The 📊 Datos state of `lesson` for `dataset` (feature choices are per dataset). */
  datasetTab(lesson: LessonId, dataset: DatasetRef): DatasetTabState {
    const key = `${lesson}|${datasetKey(dataset)}`;
    let state = this.#datasetTabs.get(key);
    if (!state) {
      state = createDatasetTab(this.#lastDatasetTab.get(lesson));
      this.#datasetTabs.set(key, state);
    }
    this.#lastDatasetTab.set(lesson, state);
    return state;
  }

  entry(dataset: DatasetRef): DatasetEntry | undefined {
    return this.datasets.find((candidate) => sameDataset(candidate.dataset, dataset));
  }
}

export const session = new Session();
