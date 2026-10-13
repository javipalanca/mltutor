import { untrack } from "svelte";

/** Number of backend requests in flight, for the status bar ("Actualizando…" / "Listo"). */
class Busy {
  pending = $state(0);

  get active(): boolean {
    return this.pending > 0;
  }

  // Requests start inside effects: reading `pending` there would make the effect depend
  // on it and loop, hence `untrack`.
  start(): void {
    untrack(() => {
      this.pending += 1;
    });
  }

  end(): void {
    untrack(() => {
      this.pending = Math.max(0, this.pending - 1);
    });
  }
}

export const busy = new Busy();
