import { busy } from "./busy.svelte";

/** Text of the status bar: "Actualizando…" while the backend works, else "Listo" or a message. */
class Status {
  message = $state<string | null>(null);

  #timer: ReturnType<typeof setTimeout> | undefined;

  get text(): string {
    if (busy.active) {
      return busy.progress ?? "Actualizando…";
    }
    return this.message ?? "Listo";
  }

  /** `task-progress` event: shown while the task runs; cleared when it reaches 100 %. */
  onProgress(message: string, percent: number | null): void {
    busy.progress = busy.active && !(percent !== null && percent >= 100) ? message : null;
  }

  flash(message: string, ms = 4000): void {
    this.message = message;
    clearTimeout(this.#timer);
    this.#timer = setTimeout(() => (this.message = null), ms);
  }
}

export const status = new Status();
