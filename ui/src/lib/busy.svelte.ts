import { untrack } from "svelte";

/**
 * Backend requests in flight, for the status bar ("Actualizando…" / "Listo"), and the ids
 * of the cancellable ones (CSV reading, correlation), for its "Cancelar" button.
 */
class Busy {
  pending = $state(0);
  cancellable = $state<number[]>([]);

  get active(): boolean {
    return this.pending > 0;
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
      if (task !== undefined) {
        this.cancellable = this.cancellable.filter((id) => id !== task);
      }
    });
  }
}

export const busy = new Busy();
