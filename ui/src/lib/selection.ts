/** Changing the dataset shared by the lessons and the CSV page. */
import * as backend from "./backend";
import type { DatasetRef } from "./bindings/DatasetRef";
import type { DatasetSummary } from "./bindings/DatasetSummary";
import { session } from "./session.svelte";

/**
 * Selects `dataset` in the UI at once and in the backend with a new `seq`, so that if the
 * user changes it again quickly the newest choice wins in both (review scenario A).
 * Resolves `null` if a newer selection superseded this one before it finished.
 */
export async function chooseDataset(dataset: DatasetRef): Promise<DatasetSummary | null> {
  const seq = backend.nextSeq();
  session.choose(dataset, seq);
  const summary = await backend.selectDataset(dataset, seq).catch((error: unknown) => {
    if (session.selectionSeq !== seq) {
      return null;
    }
    throw error;
  });
  return session.selectionSeq === seq ? summary : null;
}
