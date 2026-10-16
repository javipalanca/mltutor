/**
 * Thin typed wrappers around the Tauri IPC commands (`src-tauri/src/commands.rs` and
 * `src-tauri/src/data.rs`). Payload types come from `./bindings`, generated from Rust
 * with ts-rs (`cargo test`).
 */
import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { ClassDistancesView } from "./bindings/ClassDistancesView";
import type { CodeExtra } from "./bindings/CodeExtra";
import type { CodeTemplate } from "./bindings/CodeTemplate";
import type { CoefficientsView } from "./bindings/CoefficientsView";
import type { CommandError } from "./bindings/CommandError";
import type { DecisionGrid } from "./bindings/DecisionGrid";
import type { Evaluation } from "./bindings/Evaluation";
import type { ExportFormat } from "./bindings/ExportFormat";
import type { FeatureImportanceView } from "./bindings/FeatureImportanceView";
import type { ImportanceMethod } from "./bindings/ImportanceMethod";
import type { RulesKind } from "./bindings/RulesKind";
import type { RulesRequest } from "./bindings/RulesRequest";
import type { TreeDiagram } from "./bindings/TreeDiagram";
import type { GridRequest } from "./bindings/GridRequest";
import type { LessonId as BackendLessonId } from "./bindings/LessonId";
import type { PlaygroundRequest } from "./bindings/PlaygroundRequest";
import type { PlaygroundView } from "./bindings/PlaygroundView";
import type { PredictionForm } from "./bindings/PredictionForm";
import type { PredictionResult } from "./bindings/PredictionResult";
import type { PredictRequest } from "./bindings/PredictRequest";
import type { ProbabilityView } from "./bindings/ProbabilityView";
import type { ResidualsView } from "./bindings/ResidualsView";
import type { RocData } from "./bindings/RocData";
import type { TaskProgress } from "./bindings/TaskProgress";
import type { ArchitectureRequest } from "./bindings/ArchitectureRequest";
import type { ArchitectureView } from "./bindings/ArchitectureView";
import type { HistoryView } from "./bindings/HistoryView";
import type { NnCodeRequest } from "./bindings/NnCodeRequest";
import type { NnEpochEvent } from "./bindings/NnEpochEvent";
import type { TrainedModelInfo } from "./bindings/TrainedModelInfo";
import type { TrainRequest } from "./bindings/TrainRequest";
import type { LessonId as RouteLesson } from "./content";
import type { CorrelationMatrix } from "./bindings/CorrelationMatrix";
import type { CsvLoaded } from "./bindings/CsvLoaded";
import type { CsvPreview } from "./bindings/CsvPreview";
import type { DatasetEntry } from "./bindings/DatasetEntry";
import type { DatasetRef } from "./bindings/DatasetRef";
import type { DatasetSummary } from "./bindings/DatasetSummary";
import type { DescribeTable } from "./bindings/DescribeTable";
import type { LaunchMode } from "./bindings/LaunchMode";
import type { LogLevel } from "./bindings/LogLevel";
import type { PairplotData } from "./bindings/PairplotData";
import type { PairplotRequest } from "./bindings/PairplotRequest";
import type { SelectedDataset } from "./bindings/SelectedDataset";
import type { TablePreview } from "./bindings/TablePreview";
import type { TargetDistribution } from "./bindings/TargetDistribution";
import type { TargetInfo } from "./bindings/TargetInfo";
import type { TaskChoice } from "./bindings/TaskChoice";
import { busy } from "./busy.svelte";

export type { LaunchMode, LogLevel };

export { isTauri };

const GENERIC_ERROR = "No se pudo completar la operación. Puedes volver a intentarlo o elegir otra sección.";

export function isCommandError(value: unknown): value is CommandError {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { kind?: unknown }).kind === "string" &&
    typeof (value as { message?: unknown }).message === "string"
  );
}

/** Spanish message of any rejection of `invoke`. */
export function errorMessage(error: unknown): string {
  if (isCommandError(error)) {
    return error.message;
  }
  if (typeof error === "string" && error.trim()) {
    return error;
  }
  return GENERIC_ERROR;
}

/** The user cancelled, or a newer request superseded this one: not an error to show. */
export function isCancelled(error: unknown): boolean {
  return isCommandError(error) && error.kind === "cancelled";
}

/** `invoke` that shows the busy indicator of the status bar while it runs. */
async function call<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  busy.start();
  try {
    return await invoke<T>(command, args);
  } finally {
    busy.end();
  }
}

/** A request that can be cancelled while it runs (status bar button or a newer request). */
export interface Cancellable<T> {
  promise: Promise<T>;
  /** Id of the task in the backend (events such as `nn-epoch` carry it). */
  taskId?: number;
  /** Asks the backend to stop; the promise then rejects with a `cancelled` error. */
  cancel(): void;
}

let nextTaskId = 0;

/** `call` with a `taskId` registered in the backend (`src-tauri/src/tasks.rs`). */
function cancellable<T>(command: string, args: Record<string, unknown> = {}): Cancellable<T> {
  nextTaskId = (nextTaskId + 1) % 0x7fffffff;
  const taskId = nextTaskId;
  busy.start(taskId);
  const promise = invoke<T>(command, { ...args, taskId }).finally(() => busy.end(taskId));
  return { promise, cancel: () => void cancelTask(taskId) };
}

export function cancelTask(taskId: number): Promise<boolean> {
  return invoke<boolean>("cancel_task", { taskId }).catch(() => false);
}

/** Cancels every cancellable request in flight ("Cancelar" of the status bar). */
export function cancelAll(): void {
  for (const taskId of busy.cancellable) {
    void cancelTask(taskId);
  }
}

// Requests that change the selection carry an increasing `seq`: the backend applies the
// newest one even if an older one finishes later. Seeded with the clock so that a reload
// of the page (the backend keeps its session) keeps increasing; `continueSeqAfter` then
// raises it above the last `seq` the backend applied, in case the clock went backwards.
let requestSeq = Date.now() * 1000;

/** Makes the next `seq` greater than `lastApplied` (from `get_selected_dataset`). */
export function continueSeqAfter(lastApplied: number): void {
  requestSeq = Math.max(requestSeq, Date.now() * 1000, lastApplied);
}

export function nextSeq(): number {
  requestSeq += 1;
  return requestSeq;
}

export function currentSeq(): number {
  return requestSeq;
}

export function launchMode(): Promise<LaunchMode> {
  return invoke<LaunchMode>("launch_mode");
}

/** Tells the backend that the first page is on screen (startup timing). */
export function frontendReady(route: string): Promise<void> {
  return invoke("frontend_ready", { route });
}

/** `--screenshot`: the scenario is ready; `height` (CSS px) for `--full-page`. */
export function screenshotReady(height: number | null, error: string | null = null): Promise<void> {
  return invoke("screenshot_ready", { height, error });
}

export function smokeRouteRendered(route: string, title: string): Promise<void> {
  return invoke("smoke_route_rendered", { route, title });
}

export function smokeCheckPassed(check: string): Promise<void> {
  return invoke("smoke_check_passed", { check });
}

export function smokeFailed(message: string): Promise<void> {
  return invoke("smoke_failed", { message });
}

/** Panics on a backend worker thread; resolves `true` if the app survived it. */
export function smokePanicProbe(): Promise<boolean> {
  return invoke<boolean>("smoke_panic_probe");
}

export function logToBackend(level: LogLevel, message: string): Promise<void> {
  return invoke("log_message", { level, message });
}

// --- Data (fase1-shell-datos.md §5.4) ---

export function listDatasets(): Promise<DatasetEntry[]> {
  return call("list_datasets");
}

/** The shared selection and the last `seq` the backend applied. */
export function getSelectedDataset(): Promise<SelectedDataset> {
  return call("get_selected_dataset");
}

export function selectDataset(dataset: DatasetRef, seq: number = nextSeq()): Promise<DatasetSummary> {
  return call("select_dataset", { dataset, seq });
}

export function datasetSummary(dataset: DatasetRef): Promise<DatasetSummary> {
  return call("dataset_summary", { dataset });
}

export function datasetPreview(dataset: DatasetRef, rows = 10): Promise<TablePreview> {
  return call("dataset_preview", { dataset, rows });
}

export function datasetDescribe(dataset: DatasetRef): Promise<DescribeTable> {
  return call("dataset_describe", { dataset });
}

export function targetDistribution(dataset: DatasetRef): Promise<TargetDistribution> {
  return call("target_distribution", { dataset });
}

export function correlationMatrix(dataset: DatasetRef): Cancellable<CorrelationMatrix> {
  return cancellable("correlation_matrix", { dataset });
}

export function pairplotData(req: PairplotRequest): Promise<PairplotData> {
  return call("pairplot_data", { req });
}

/** Native "Abrir" dialog + reading, all in Rust: the path never goes through JS. */
export function pickAndLoadCsv(filterName: string): Cancellable<CsvPreview | null> {
  return cancellable("pick_and_load_csv", { filterName });
}

/** `--screenshot --scenario csv_…:<ruta.csv>`: the backend reads the path of its command line. */
export function loadScenarioCsv(): Cancellable<CsvPreview> {
  return cancellable("load_scenario_csv");
}

/** "Quitar archivo": releases the draft in the backend. */
export function discardCsv(draftId: number): Promise<boolean> {
  return call("discard_csv", { draftId });
}

export function csvTargetInfo(draftId: number, target: string): Promise<TargetInfo> {
  return call("csv_target_info", { draftId, target });
}

/** Confirms the target and registers the CSV (cancellable: big files take a while). */
export function setCsvTarget(draftId: number, target: string, task: TaskChoice, seq: number): Cancellable<CsvLoaded> {
  return cancellable("set_csv_target", { draftId, target, task, seq });
}

export type SaveKind = "png" | "svg" | "python" | "text";

/**
 * Native save dialog + writing, in Rust (`save_file`). The bytes go as the raw IPC body;
 * the suggested name and the kind as headers. Resolves the saved name, or `null`.
 */
export function saveFile(name: string, kind: SaveKind, bytes: Uint8Array): Promise<string | null> {
  return invoke<string | null>("save_file", bytes, {
    headers: { "x-mltutor-file-name": encodeURIComponent(name), "x-mltutor-file-kind": kind },
  });
}

// --- Models (fase2-regresion.md §5.4) ---

/** Lesson of the backend (`LessonId` in Rust) for each lesson route. */
export const BACKEND_LESSON: Readonly<Record<RouteLesson, BackendLessonId>> = {
  arboles: "trees",
  regresion: "regression",
  knn: "knn",
  redes: "nn",
};

/** A model trained by a newer request or retrained meanwhile: reload, do not show it. */
export function isStale(error: unknown): boolean {
  return isCommandError(error) && error.kind === "model_stale";
}

/** Training in the background: progress by `task-progress`, cancellable, newest `seq` wins. */
export function trainModel(req: Omit<TrainRequest, "task_id" | "seq">, seq: number = nextSeq()): Cancellable<TrainedModelInfo> {
  nextTaskId = (nextTaskId + 1) % 0x7fffffff;
  const taskId = nextTaskId;
  busy.start(taskId);
  const promise = invoke<TrainedModelInfo>("train_model", { req: { ...req, seq, task_id: taskId } }).finally(() =>
    busy.end(taskId),
  );
  return { promise, taskId, cancel: () => void cancelTask(taskId) };
}

export function lessonModel(lesson: BackendLessonId): Promise<TrainedModelInfo | null> {
  return call("lesson_model", { lesson });
}

export function evaluateModel(modelId: number): Promise<Evaluation> {
  return call("evaluate_model", { modelId });
}

export function rocCurves(modelId: number): Promise<RocData> {
  return call("roc_curves", { modelId });
}

export function probabilityView(modelId: number): Promise<ProbabilityView> {
  return call("probability_view", { modelId });
}

export function residualsView(modelId: number): Promise<ResidualsView> {
  return call("residuals_view", { modelId });
}

export function decisionGrid(modelId: number, featureX: number, featureY: number): Cancellable<DecisionGrid> {
  nextTaskId = (nextTaskId + 1) % 0x7fffffff;
  const taskId = nextTaskId;
  busy.start(taskId);
  const req: GridRequest = { model_id: modelId, feature_x: featureX, feature_y: featureY, task_id: taskId };
  const promise = invoke<DecisionGrid>("decision_grid", { req }).finally(() => busy.end(taskId));
  return { promise, cancel: () => void cancelTask(taskId) };
}

export function coefficientsView(modelId: number, classIndex: number | null): Promise<CoefficientsView> {
  return call("coefficients_view", { modelId, classIndex });
}

export function predictionForm(modelId: number): Promise<PredictionForm> {
  return call("prediction_form", { modelId });
}

export function predictOne(modelId: number, values: number[]): Promise<PredictionResult> {
  const req: PredictRequest = { model_id: modelId, values };
  return call("predict_one", { req });
}

/** ONNX or Python package, built and saved (native dialog) in Rust. Saved name or `null`. */
export function exportModel(modelId: number, format: ExportFormat): Promise<string | null> {
  return call("export_model", { modelId, format });
}

export function generateCode(modelId: number, template: CodeTemplate, extra: CodeExtra | null = null): Promise<string> {
  return call("generate_code", { modelId, template, extra });
}

// --- Decision trees (fase3-arboles.md §5.4) ---

/** C13: boxes, colours and Buchheim layout of `plot_tree`. */
export function treeDiagram(modelId: number): Promise<TreeDiagram> {
  return call("tree_diagram", { modelId });
}

/** C14: `export_text` (📝 Texto with weights, or the rules of 💾 Exportar); D40 names. */
export function treeRules(modelId: number, kind: RulesKind, classNames = true): Promise<string> {
  const req: RulesRequest = { model_id: modelId, kind, class_names: classNames };
  return call("tree_rules", { req });
}

/** C10: native importance or permutation importance (cancellable, with progress). */
export function featureImportance(modelId: number, method: ImportanceMethod): Cancellable<FeatureImportanceView> {
  return cancellable("feature_importance", { modelId, method });
}

// --- K-Nearest Neighbors (fase4-knn.md §5.4) ---

/** C19: 📏 Análisis de Distancias (O(n²) pairs: cancellable when the view changes). */
export function classDistances(modelId: number): Cancellable<ClassDistancesView> {
  return cancellable("class_distances", { modelId });
}

/**
 * C18: the 2D KNN of the canvas and its test points. Cancellable: a drag sends a request
 * per frame and cancels the previous one. Not counted by the busy indicator (it would
 * flicker while dragging).
 */
export function knnPlayground(req: Omit<PlaygroundRequest, "task_id">): Cancellable<PlaygroundView> {
  nextTaskId = (nextTaskId + 1) % 0x7fffffff;
  const taskId = nextTaskId;
  const promise = invoke<PlaygroundView>("knn_playground", { req: { ...req, task_id: taskId } });
  return { promise, cancel: () => void cancelTask(taskId) };
}

// --- Neural networks (fase5-redes.md §5.4) ---

/** 🏗️ Arquitectura of the dataset with the current controls (cheap: called on every change). */
export function nnArchitecture(req: ArchitectureRequest): Promise<ArchitectureView> {
  return invoke<ArchitectureView>("nn_architecture", { req });
}

/** C27: the history of a trained network. */
export function nnHistory(modelId: number): Promise<HistoryView> {
  return call("nn_history", { modelId });
}

/** D74: the Keras script with the controls of 🏗️ and ⚙️ (no model needed). */
export function nnTrainingCode(req: NnCodeRequest): Promise<string> {
  return call("nn_training_code", { req });
}

/** Subscribes to `nn-epoch` (phases and epochs of a training); returns the unsubscribe function. */
export async function onNnEpoch(handler: (event: NnEpochEvent) => void): Promise<() => void> {
  if (!isTauri()) {
    return () => undefined;
  }
  return listen<NnEpochEvent>("nn-epoch", (event) => handler(event.payload));
}

/** Subscribes to `task-progress`; returns the unsubscribe function. */
export async function onTaskProgress(handler: (progress: TaskProgress) => void): Promise<() => void> {
  if (!isTauri()) {
    return () => undefined;
  }
  return listen<TaskProgress>("task-progress", (event) => handler(event.payload));
}
