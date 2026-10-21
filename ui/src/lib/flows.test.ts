/**
 * Out-of-order backend answers (review scenarios A, B and C) and cancellation, with a
 * mocked `invoke` whose answers the test releases in any order.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

interface Call {
  command: string;
  args: Record<string, unknown>;
  resolve(value: unknown): void;
  reject(error: unknown): void;
}

const calls: Call[] = [];

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: () => true,
  invoke: (command: string, args: Record<string, unknown> = {}) =>
    new Promise((resolve, reject) => {
      calls.push({ command, args, resolve, reject });
    }),
}));

const { chooseDataset } = await import("./selection");
const { clearCsv, loadScenarioCsv, retryRegistration, setTask } = await import("./csvFlow");
const backend = await import("./backend");
const { session } = await import("./session.svelte");
const { busy } = await import("./busy.svelte");
const StatusBar = (await import("../components/StatusBar.svelte")).default;
const { models } = await import("./ml/modelSession.svelte");

const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

function pending(command: string): Call[] {
  return calls.filter((call) => call.command === command);
}

function take(command: string): Call {
  const index = calls.findIndex((call) => call.command === command);
  const [call] = index < 0 ? [] : calls.splice(index, 1);
  if (!call) {
    throw new Error(`no hay ninguna llamada a ${command}: ${calls.map((c) => c.command).join(", ")}`);
  }
  return call;
}

const IRIS = { kind: "builtin", id: "iris" } as const;
const CALIFORNIA = { kind: "builtin", id: "california_housing" } as const;
const WINE = { kind: "builtin", id: "wine" } as const;

function preview(draftId: number, file: string): unknown {
  return {
    draft_id: draftId,
    file_name: file,
    n_rows: 3,
    n_cols: 2,
    missing_values: 0,
    columns: [],
    head: { columns: [], index: [], rows: [], total_rows: 3 },
    default_target: "y",
    target: { column: "y", nunique: 2, dtype: "int64", value_counts: null, suggested_task: "classification" },
    notices: [],
  };
}

function loaded(id: number, task: string): unknown {
  return {
    dataset: { dataset: { kind: "csv", id }, label: "📄 a.csv", blurb: null, task, synthetic: false },
    task_choice: task,
    resolved_task: task,
    n_rows: 3,
    n_cols: 2,
    target: "y",
    describe: { columns: [], stats: [], values: { rows: 0, cols: 0, values: [] }, excluded_columns: [] },
    target_stats: { kind: "classes", value_counts: [] },
  };
}

beforeEach(() => {
  calls.length = 0;
  session.selected = IRIS;
  session.selectionSeq = 0;
  session.csv = { preview: null, target: null, task: null, loaded: null, error: null, registerCancelled: false };
});

afterEach(() => cleanup());

describe("dataset selection (scenario A)", () => {
  it("keeps the last choice and sends increasing seqs whatever the answer order", async () => {
    const results = [chooseDataset(IRIS), chooseDataset(CALIFORNIA), chooseDataset(WINE)];
    const [iris, california, wine] = [take("select_dataset"), take("select_dataset"), take("select_dataset")];
    const seqs = [iris, california, wine].map((call) => call.args["seq"] as number);
    expect(seqs[0]).toBeLessThan(seqs[1] ?? 0);
    expect(seqs[1]).toBeLessThan(seqs[2] ?? 0);
    expect(session.selected).toEqual(WINE);
    // Vino answers first, California (slow) last: the UI stays on Vino.
    wine.resolve({ dataset: WINE });
    iris.reject({ kind: "internal", message: "fallo viejo" });
    california.resolve({ dataset: CALIFORNIA });
    const answers = await Promise.all(results);
    expect(answers).toEqual([null, null, { dataset: WINE }]);
    expect(session.selected).toEqual(WINE);
  });

  it("continues above the last seq the backend applied (the clock went backwards)", () => {
    const ahead = Date.now() * 1000 + 1e9;
    backend.continueSeqAfter(ahead);
    expect(backend.nextSeq()).toBe(ahead + 1);
    backend.continueSeqAfter(5);
    expect(backend.nextSeq()).toBe(ahead + 2);
  });
});

describe("CSV page", () => {
  async function loadA(): Promise<void> {
    const done = loadScenarioCsv();
    take("load_scenario_csv").resolve(preview(1, "a.csv"));
    await flush();
    const register = take("set_csv_target");
    register.resolve(loaded(1, "classification"));
    await flush();
    take("list_datasets").resolve([]);
    await done;
  }

  it("serialises registrations and only sends the newest pending one (scenario C)", async () => {
    await loadA();
    const first = setTask("regression");
    await flush();
    const second = setTask("classification");
    const third = setTask("auto");
    await flush();
    // One request at a time: the second and third wait for the first.
    expect(pending("set_csv_target")).toHaveLength(1);
    const a = take("set_csv_target");
    expect(a.args["task"]).toBe("regression");
    a.resolve(loaded(1, "regression"));
    await flush();
    // The second was superseded before it started: only "auto" is sent.
    const c = take("set_csv_target");
    expect(c.args["task"]).toBe("auto");
    expect(pending("set_csv_target")).toHaveLength(0);
    expect((c.args["seq"] as number) > (a.args["seq"] as number)).toBe(true);
    c.resolve(loaded(1, "auto"));
    await flush();
    take("list_datasets").resolve([{ dataset: { kind: "csv", id: 1 } }]);
    await Promise.all([first, second, third]);
    expect(session.csv.task).toBe("auto");
    expect(session.csv.loaded?.task_choice).toBe("auto");
    expect(session.csv.error).toBeNull();
  });

  it("ignores the answers of a file replaced meanwhile (scenario B)", async () => {
    await loadA();
    const slow = setTask("regression");
    await flush();
    const old = take("set_csv_target");
    const next = loadScenarioCsv();
    take("load_scenario_csv").resolve(preview(2, "b.csv"));
    await flush();
    // The backend rejects the old draft; the error is not shown for the new file.
    old.reject({ kind: "csv_io", message: "❌ Error al procesar el archivo: el archivo ya no está disponible" });
    await flush();
    const register = take("set_csv_target");
    expect(register.args["draftId"]).toBe(2);
    register.resolve(loaded(2, "classification"));
    await flush();
    take("list_datasets").resolve([]);
    await Promise.all([slow, next]);
    expect(session.csv.preview?.file_name).toBe("b.csv");
    expect(session.csv.error).toBeNull();
  });

  it("does not select the CSV if the user chose another dataset meanwhile", async () => {
    const done = loadScenarioCsv();
    take("load_scenario_csv").resolve(preview(1, "a.csv"));
    await flush();
    const register = take("set_csv_target");
    void chooseDataset(WINE);
    take("select_dataset").resolve({ dataset: WINE });
    register.resolve(loaded(1, "classification"));
    await flush();
    take("list_datasets").resolve([{ dataset: { kind: "csv", id: 1 } }]);
    await done;
    expect(session.selected).toEqual(WINE);
  });

  it("'Quitar archivo' releases the draft in the backend", async () => {
    await loadA();
    clearCsv();
    expect(take("discard_csv").args).toEqual({ draftId: 1 });
    expect(session.csv.preview).toBeNull();
  });

  it("a cancelled registration keeps the preview and offers 'Registrar de nuevo'", async () => {
    await loadA();
    const change = setTask("regression");
    await flush();
    take("set_csv_target").reject({ kind: "cancelled", message: "Operación cancelada" });
    await change;
    expect(session.csv.preview?.file_name).toBe("a.csv");
    expect(session.csv.loaded).toBeNull();
    expect(session.csv.registerCancelled).toBe(true);
    expect(session.csv.error).toBeNull();
    const retry = retryRegistration();
    expect(session.csv.registerCancelled).toBe(false);
    await flush();
    const again = take("set_csv_target");
    expect(again.args["task"]).toBe("regression");
    again.resolve(loaded(1, "regression"));
    await flush();
    take("list_datasets").resolve([{ dataset: { kind: "csv", id: 1 } }]);
    await retry;
    expect(session.csv.loaded?.task_choice).toBe("regression");
  });

  it("a cancelled read empties the page without an error", async () => {
    await loadA();
    const done = loadScenarioCsv();
    take("load_scenario_csv").reject({ kind: "cancelled", message: "Operación cancelada" });
    await done;
    expect(session.csv.preview).toBeNull();
    expect(session.csv.error).toBeNull();
  });
});

describe("status bar", () => {
  it("offers 'Cancelar' while a cancellable operation runs and cancels it", async () => {
    render(StatusBar);
    expect(screen.queryByRole("button", { name: "Cancelar" })).toBeNull();
    const done = loadScenarioCsv();
    const read = take("load_scenario_csv");
    const taskId = read.args["taskId"] as number;
    expect(busy.cancellable).toEqual([taskId]);
    await flush();
    await fireEvent.click(screen.getByRole("button", { name: "Cancelar" }));
    expect(take("cancel_task").args).toEqual({ taskId });
    read.reject({ kind: "cancelled", message: "Operación cancelada" });
    await done;
    await flush();
    expect(busy.cancellable).toEqual([]);
    expect(screen.queryByRole("button", { name: "Cancelar" })).toBeNull();
  });
});

describe("training errors and classes", () => {
  it("shows the Spanish message of too_many_classes (I5)", async () => {
    const done = models.train("regresion", IRIS, { kind: "logistic", max_iter: 1000 }, 0.3);
    const message = "El objetivo tiene 58 clases y el máximo para clasificación es 50.";
    take("train_model").reject({ kind: "too_many_classes", n_classes: 58, max: 50, message });
    await done;
    const state = models.lesson("regresion");
    expect(state.trainError).toBe(message);
    expect(state.trainErrorKind).toBe("too_many_classes");
    expect(state.training).toBe(false);
  });

  it("keeps the coefficient row within the classes the model saw", async () => {
    const state = models.lesson("regresion");
    state.coefClass = 2;
    const done = models.train("regresion", IRIS, { kind: "logistic", max_iter: 1000 }, 0.3);
    take("train_model").resolve({
      model_id: 7,
      lesson: "regression",
      dataset: IRIS,
      dataset_label: "Iris",
      config: { kind: "logistic", max_iter: 1000 },
      test_size: 0.3,
      model_task: "classification",
      dataset_task: "classification",
      n_train: 4,
      n_test: 2,
      feature_columns: ["a", "b"],
      feature_names: ["a", "b"],
      class_names: ["setosa", "versicolor", "virginica"],
      // 2 of 3 classes in the train set: a binary model with one row of coefficients.
      model_classes: [0, 2],
      notices: [],
    });
    await done;
    expect(state.info?.model_id).toBe(7);
    expect(state.coefClass).toBe(0);
  });
});
