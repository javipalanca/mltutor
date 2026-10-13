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

/** `invoke` that shows the busy indicator of the status bar while it runs. */
async function call<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  busy.start();
  try {
    return await invoke<T>(command, args);
  } finally {
    busy.end();
  }
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

export function selectDataset(dataset: DatasetRef): Promise<DatasetSummary> {
  return call("select_dataset", { dataset });
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

export function correlationMatrix(dataset: DatasetRef): Promise<CorrelationMatrix> {
  return call("correlation_matrix", { dataset });
}

export function pairplotData(req: PairplotRequest): Promise<PairplotData> {
  return call("pairplot_data", { req });
}

export function loadCsv(path: string): Promise<CsvPreview> {
  return call("load_csv", { path });
}

export function csvTargetInfo(draftId: number, target: string): Promise<TargetInfo> {
  return call("csv_target_info", { draftId, target });
}

export function setCsvTarget(draftId: number, target: string, task: TaskChoice): Promise<CsvLoaded> {
  return call("set_csv_target", { draftId, target, task });
}
