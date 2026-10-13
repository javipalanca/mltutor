import { busy } from "./busy.svelte";

/** Text of the status bar: "Actualizando…" while the backend works, else "Listo" or a message. */
class Status {
  message = $state<string | null>(null);
  #timer: ReturnType<typeof setTimeout> | undefined;

  get text(): string {
    if (busy.active) {
      return "Actualizando…";
    }
    return this.message ?? "Listo";
  }

  flash(message: string, ms = 4000): void {
    this.message = message;
    clearTimeout(this.#timer);
    this.#timer = setTimeout(() => (this.message = null), ms);
  }
}

export const status = new Status();
