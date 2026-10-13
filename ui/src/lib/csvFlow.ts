/**
 * Flow of 📁 Cargar CSV Personalizado (fase1-shell-datos.md §3) on `session.csv`:
 * read → register with the default target and the suggested task → re-register on
 * every change (Python registers and selects the CSV on each rerun).
 *
 * Responses can arrive out of order, so:
 * - every user action bumps `flow`; an action only applies its results (and errors) if
 *   no newer action happened meanwhile;
 * - registrations run one at a time and only the newest pending one is sent ("the last
 *   request wins"); each carries the selection `seq` of the moment the user acted, so
 *   the backend cannot be left with an older target or task either.
 */
import * as backend from "./backend";
import type { CsvPreview } from "./bindings/CsvPreview";
import type { TaskChoice } from "./bindings/TaskChoice";
import type { TaskType } from "./bindings/TaskType";
import { sameDataset, session } from "./session.svelte";
import { status } from "./status.svelte";

export function taskChoiceOf(task: TaskType): TaskChoice {
  return task;
}

let flow = 0;
let registerSeq = 0;
let chain: Promise<void> = Promise.resolve();

/** Queues a registration of the current page state; resolves (or rejects) when it is done or superseded. */
function register(): Promise<void> {
  registerSeq += 1;
  const mine = registerSeq;
  const seq = backend.nextSeq();
  const run = async (): Promise<void> => {
    const csv = session.csv;
    if (mine !== registerSeq || !csv.preview || csv.target === null || csv.task === null) {
      return;
    }
    const loaded = await backend.setCsvTarget(csv.preview.draft_id, csv.target, csv.task, seq).promise;
    if (mine !== registerSeq) {
      return;
    }
    csv.loaded = loaded;
    const datasets = await backend.listDatasets();
    if (mine !== registerSeq) {
      return;
    }
    session.datasets = datasets;
    // Same rule as the backend: select it unless the user chose another dataset later
    // (or the selected CSV was the one this registration replaced).
    const selectedGone = session.selected.kind === "csv" && !datasets.some((e) => sameDataset(e.dataset, session.selected));
    if (seq > session.selectionSeq || selectedGone) {
      session.selected = loaded.dataset.dataset;
      session.selectionSeq = Math.max(session.selectionSeq, seq);
    }
  };
  const link = chain.then(run);
  // The chain itself never rejects: the caller gets the error of its own link.
  chain = link.catch(() => undefined);
  return link;
}

async function guarded(work: (mine: number) => Promise<void>): Promise<void> {
  flow += 1;
  const mine = flow;
  try {
    session.csv.error = null;
    await work(mine);
  } catch (error) {
    if (mine !== flow) {
      return;
    }
    if (backend.isCancelled(error)) {
      status.flash(backend.errorMessage(error));
    } else {
      session.csv.error = backend.errorMessage(error);
    }
  }
}

function clearPage(): void {
  const csv = session.csv;
  csv.preview = null;
  csv.loaded = null;
  csv.target = null;
  csv.task = null;
}

/** Reads a CSV with `request` (dialog or `--scenario`) and registers it. */
function loadWith(request: () => backend.Cancellable<CsvPreview | null>): Promise<void> {
  return guarded(async (mine) => {
    let preview: CsvPreview | null;
    try {
      preview = await request().promise;
    } catch (error) {
      // The backend released the previous file before reading this one.
      if (mine === flow) {
        clearPage();
      }
      throw error;
    }
    if (!preview || mine !== flow) {
      return; // Dialog closed: the page does not change.
    }
    const csv = session.csv;
    csv.preview = preview;
    csv.loaded = null;
    csv.target = preview.default_target;
    csv.task = taskChoiceOf(preview.target.suggested_task);
    await register();
  });
}

/** "Seleccionar archivo": native dialog and reading in Rust (the path never reaches JS). */
export function pickCsv(filterName: string): Promise<void> {
  return loadWith(() => backend.pickAndLoadCsv(filterName));
}

/** `--screenshot --scenario csv_…:<ruta.csv>` (the backend knows the path). */
export function loadScenarioCsv(): Promise<void> {
  return loadWith(() => backend.loadScenarioCsv());
}

export function setTarget(target: string): Promise<void> {
  return guarded(async (mine) => {
    const csv = session.csv;
    if (!csv.preview) {
      return;
    }
    const draftId = csv.preview.draft_id;
    csv.target = target;
    const info = await backend.csvTargetInfo(draftId, target);
    if (mine !== flow || csv.preview?.draft_id !== draftId) {
      return;
    }
    csv.preview = { ...csv.preview, target: info };
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

/**
 * "Quitar archivo": empties the page and releases the file in the backend. The CSV stays
 * registered and selected (as Python).
 */
export function clearCsv(): void {
  flow += 1;
  registerSeq += 1; // Drops registrations still queued for this file.
  const draftId = session.csv.preview?.draft_id;
  clearPage();
  session.csv.error = null;
  if (draftId !== undefined) {
    void backend.discardCsv(draftId).catch(() => false);
  }
}
