import { untrack } from "svelte";

/**
 * Backend requests in flight, for the status bar ("Actualizando…" / "Listo"), and the ids
 * of the cancellable ones (CSV reading, correlation), for its "Cancelar" button.
 */
class Busy {
  pending = $state(0);
  cancellable = $state<number[]>([]);
  /** Last `task-progress` message while something runs (cleared when idle). */
  progress = $state<string | null>(null);

  get active(): boolean {
    return this.pending > 0;
  }

  /** The newest cancellable task still running (the one the status bar follows). */
  get activeTask(): number | null {
    return this.cancellable.at(-1) ?? null;
  }

  get canCancel(): boolean {
    return this.cancellable.length > 0;
  }

  // Requests start inside effects: reading the state there would make the effect depend
  // on it and loop, hence `untrack`.
  start(task?: number): void {
    untrack(() => {
      this.pending += 1;
      if (task !== undefined) {
        this.cancellable = [...this.cancellable, task];
      }
    });
  }

  end(task?: number): void {
    untrack(() => {
      this.pending = Math.max(0, this.pending - 1);
      if (this.pending === 0) {
        this.progress = null;
      }
      if (task !== undefined) {
        const wasActive = this.activeTask === task;
        this.cancellable = this.cancellable.filter((id) => id !== task);
        // Its last message must not linger under the next task.
        if (wasActive) {
          this.progress = null;
        }
      }
    });
  }
}

export const busy = new Busy();
