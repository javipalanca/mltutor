import { afterEach, describe, expect, it } from "vitest";
import { busy } from "./busy.svelte";
import { status } from "./status.svelte";

describe("task-progress in the status bar", () => {
  afterEach(() => {
    for (const id of [...busy.cancellable]) {
      busy.end(id);
    }
    while (busy.pending > 0) {
      busy.end();
    }
  });

  it("follows only the newest task in flight", () => {
    busy.start(1);
    status.onProgress({ task_id: 1, message: "Entrenando 1", percent: 10 });
    expect(status.text).toBe("Entrenando 1");
    // A newer training supersedes the first one, which is still winding down.
    busy.start(2);
    status.onProgress({ task_id: 2, message: "Entrenando 2", percent: 5 });
    status.onProgress({ task_id: 1, message: "Entrenando 1", percent: 50 });
    expect(status.text).toBe("Entrenando 2");
    // A late event of the old task after it ended changes nothing either.
    busy.end(1);
    status.onProgress({ task_id: 1, message: "Modelo entrenado", percent: 100 });
    expect(status.text).toBe("Entrenando 2");
    // Unknown ids (another window, a finished task) are ignored.
    status.onProgress({ task_id: 99, message: "otra", percent: null });
    expect(status.text).toBe("Entrenando 2");
    // 100 % of the active task clears it.
    status.onProgress({ task_id: 2, message: "Modelo entrenado", percent: 100 });
    expect(status.text).toBe("Actualizando…");
    busy.end(2);
    expect(status.text).toBe("Listo");
  });

  it("does not keep the message of an ended active task", () => {
    busy.start(3);
    busy.start(4);
    status.onProgress({ task_id: 4, message: "Generando frontera de decisión...", percent: null });
    busy.end(4);
    expect(busy.activeTask).toBe(3);
    expect(status.text).toBe("Actualizando…");
  });
});
