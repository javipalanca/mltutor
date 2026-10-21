import type { TaskProgress } from "./bindings/TaskProgress";
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

  /**
   * `task-progress` event: shown while the task runs; cleared when it reaches 100 %.
   * Only the newest task in flight may write it: a superseded training that is still
   * winding down (or a late event of a finished task) must not overwrite the message.
   */
  onProgress(progress: Pick<TaskProgress, "task_id" | "message" | "percent">): void {
    if (busy.activeTask !== progress.task_id) {
      return;
    }
    busy.progress = !(progress.percent !== null && progress.percent >= 100) ? progress.message : null;
  }

  flash(message: string, ms = 4000): void {
    this.message = message;
    clearTimeout(this.#timer);
    this.#timer = setTimeout(() => (this.message = null), ms);
  }
}

export const status = new Status();
