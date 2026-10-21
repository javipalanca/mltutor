/**
 * Loads one backend view for the current model (evaluation, ROC, grid…). Old answers are
 * ignored when the key changes, a cancellable request is cancelled, and a `model_stale`
 * answer asks the lesson for its current model instead of showing an error.
 */
import { untrack } from "svelte";
import * as backend from "../backend";
import type { Cancellable } from "../backend";

export class ViewLoader<T> {
  data = $state<T | null>(null);
  error = $state<string | null>(null);
  loading = $state(false);
  /** Key of `data` (e.g. `model_id|pair`). */
  key = $state<string | null>(null);
  #pending: string | null = null;

  /**
   * Starts loading `key` unless it is already loaded or loading. Call it from an `$effect`
   * and return its cleanup. `onStale` runs on `model_stale`.
   */
  load(key: string, request: () => Promise<T> | Cancellable<T>, onStale: () => void, delay = 0): () => void {
    return untrack(() => {
      if (this.key === key && this.error === null) {
        return () => undefined;
      }
      let stale = false;
      let cancel: (() => void) | null = null;
      this.#pending = key;
      this.loading = true;
      this.error = null;
      const run = (): void => {
        const started = request();
        const promise = started instanceof Promise ? started : started.promise;
        if (!(started instanceof Promise)) {
          cancel = started.cancel;
        }
        promise
          .then((data) => {
            if (!stale && this.#pending === key) {
              this.data = data;
              this.key = key;
              this.loading = false;
            }
          })
          .catch((error: unknown) => {
            if (stale || this.#pending !== key) {
              return;
            }
            this.loading = false;
            if (backend.isStale(error)) {
              onStale();
            } else if (!backend.isCancelled(error)) {
              this.error = backend.errorMessage(error);
            }
          });
      };
      const timer = delay > 0 ? setTimeout(run, delay) : null;
      if (timer === null) {
        run();
      }
      return () => {
        stale = true;
        if (timer !== null) {
          clearTimeout(timer);
        }
        cancel?.();
        if (this.#pending === key) {
          this.#pending = null;
          this.loading = false;
        }
      };
    });
  }

  /** `data` if it belongs to `key`. */
  for(key: string): T | null {
    return this.key === key ? this.data : null;
  }
}
