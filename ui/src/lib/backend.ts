/**
 * Thin typed wrappers around the Tauri IPC commands (`src-tauri/src/commands.rs` and
 * `src-tauri/src/data.rs`). Payload types come from `./bindings`, generated from Rust
 * with ts-rs (`cargo test`).
 */
import { invoke, isTauri } from "@tauri-apps/api/core";
import type { CommandError } from "./bindings/CommandError";
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
// of the page (the backend keeps its session) keeps increasing.
let requestSeq = Date.now() * 1000;

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

export function getSelectedDataset(): Promise<DatasetRef> {
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

export type SaveKind = "png" | "python";

/**
 * Native save dialog + writing, in Rust (`save_file`). The bytes go as the raw IPC body;
 * the suggested name and the kind as headers. Resolves the saved name, or `null`.
 */
export function saveFile(name: string, kind: SaveKind, bytes: Uint8Array): Promise<string | null> {
  return invoke<string | null>("save_file", bytes, {
    headers: { "x-mltutor-file-name": encodeURIComponent(name), "x-mltutor-file-kind": kind },
  });
}
