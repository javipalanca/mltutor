/**
 * The `ArchitectureView` of the selected dataset with the current controls, shared by
 * 🏗️ Arquitectura and ⚙️ Entrenamiento. Every change asks the backend again after a 50 ms
 * debounce (it answers in < 5 ms); the newest request wins.
 */
import * as backend from "../../backend";
import type { ArchitectureView } from "../../bindings/ArchitectureView";
import type { DatasetRef } from "../../bindings/DatasetRef";
import type { NnArchitecture } from "../../bindings/NnArchitecture";
import { datasetKey } from "../../session.svelte";
import { nn } from "./state.svelte";

export const DEBOUNCE_MS = 50;

class ArchitectureLoader {
  view = $state<ArchitectureView | null>(null);
  error = $state<string | null>(null);
  #seq = 0;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #lastKey = "";

  /** The view belongs to `dataset` (a stale one is not shown after switching). */
  matches(dataset: DatasetRef): boolean {
    return this.view !== null && datasetKey(this.view.dataset) === datasetKey(dataset);
  }

  /** Asks for the view of `dataset` with `arch` (debounced; repeated requests are skipped). */
  request(dataset: DatasetRef, arch: NnArchitecture, validationPct: number, immediate = false): void {
    const key = JSON.stringify([dataset, arch, validationPct]);
    if (key === this.#lastKey) {
      return;
    }
    this.#lastKey = key;
    if (this.#timer !== null) {
      clearTimeout(this.#timer);
    }
    const seq = ++this.#seq;
    const run = (): void => {
      this.#timer = null;
      backend
        .nnArchitecture({ dataset, arch, validation_pct: validationPct })
        .then((view) => {
          if (seq === this.#seq) {
            this.error = null;
            this.view = view;
            nn.syncTask(view);
          }
        })
        .catch((error: unknown) => {
          if (seq === this.#seq) {
            this.error = backend.errorMessage(error);
            this.#lastKey = "";
          }
        });
    };
    // The first view of a dataset is not delayed (the tab is still empty).
    if (immediate || !this.matches(dataset)) {
      run();
    } else {
      this.#timer = setTimeout(run, DEBOUNCE_MS);
    }
  }
}

export const architecture = new ArchitectureLoader();
