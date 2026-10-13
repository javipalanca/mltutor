/**
 * Flow of 📁 Cargar CSV Personalizado (fase1-shell-datos.md §3) on `session.csv`:
 * read → register with the default target and the suggested task → re-register on
 * every change (Python registers and selects the CSV on each rerun).
 */
import * as backend from "./backend";
import type { TaskChoice } from "./bindings/TaskChoice";
import type { TaskType } from "./bindings/TaskType";
import { session } from "./session.svelte";

export function taskChoiceOf(task: TaskType): TaskChoice {
  return task;
}

async function register(): Promise<void> {
  const csv = session.csv;
  if (!csv.preview || csv.target === null || csv.task === null) {
    return;
  }
  const loaded = await backend.setCsvTarget(csv.preview.draft_id, csv.target, csv.task);
  csv.loaded = loaded;
  session.datasets = await backend.listDatasets();
  session.selected = loaded.dataset.dataset;
}

async function guarded(work: () => Promise<void>): Promise<void> {
  try {
    session.csv.error = null;
    await work();
  } catch (error) {
    session.csv.error = backend.errorMessage(error);
  }
}

export function loadCsvPath(path: string): Promise<void> {
  return guarded(async () => {
    const csv = session.csv;
    csv.preview = null;
    csv.loaded = null;
    const preview = await backend.loadCsv(path);
    csv.preview = preview;
    csv.target = preview.default_target;
    csv.task = taskChoiceOf(preview.target.suggested_task);
    await register();
  });
}

export function setTarget(target: string): Promise<void> {
  return guarded(async () => {
    const csv = session.csv;
    if (!csv.preview) {
      return;
    }
    const info = await backend.csvTargetInfo(csv.preview.draft_id, target);
    csv.preview = { ...csv.preview, target: info };
    csv.target = target;
    csv.task = taskChoiceOf(info.suggested_task);
    await register();
  });
}

export function setTask(task: TaskChoice): Promise<void> {
  return guarded(async () => {
    session.csv.task = task;
    await register();
  });
}

/** "Quitar archivo": empties the page; the CSV stays registered and selected (as Python). */
export function clearCsv(): void {
  session.csv.preview = null;
  session.csv.loaded = null;
  session.csv.target = null;
  session.csv.task = null;
  session.csv.error = null;
}
