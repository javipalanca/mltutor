/**
 * UI state of 🧠 Redes Neuronales (fase5-redes.md §5.2): the controls of 🏗️ Arquitectura and
 * ⚙️ Entrenamiento and the live training (C27). The trained model and the shared options
 * live in `models.lesson("redes")`.
 *
 * D85: the output activation goes back to the recommended one when the task or the number
 * of classes changes (Python's selector had no key and kept "softmax" for Cáncer). D90: a
 * default configuration always exists. The neurons of each layer keep their value across
 * datasets (Python's `layer_{i}` keys) and start at the suggestion of the dataset.
 */
import type { ArchitectureView } from "../../bindings/ArchitectureView";
import type { EpochLog } from "../../bindings/EpochLog";
import type { ModelConfig } from "../../bindings/ModelConfig";
import type { NnActivation } from "../../bindings/NnActivation";
import type { NnArchitecture } from "../../bindings/NnArchitecture";
import type { NnEpochEvent } from "../../bindings/NnEpochEvent";
import type { NnOptimizer } from "../../bindings/NnOptimizer";
import type { NnTraining } from "../../bindings/NnTraining";
import type { TrainPhase } from "../../bindings/TrainPhase";

export const LESSON = "redes" as const;

/** Test size of the network: fixed at 0.2 (no control in Python). */
export const NN_TEST_SIZE = 0.2;
export const MAX_LAYERS = 5;
export const MAX_NEURONS = 256;
export const HIDDEN_ACTIVATIONS: readonly NnActivation[] = ["relu", "tanh", "sigmoid"];
export const BATCH_SIZES = [16, 32, 64, 128, 256] as const;
export const OPTIMIZERS: readonly NnOptimizer[] = ["adam", "sgd", "rmsprop"];
export const LEARNING_RATES = [0.001, 0.01, 0.1, 0.3] as const;
export const DEFAULT_SEED = 42;

/** The training in progress (or the last one of this visit), as the `nn-epoch` events tell it. */
export interface LiveTraining {
  taskId: number | null;
  phase: TrainPhase | null;
  epochs: number;
  logs: EpochLog[];
  /** The user pressed "Cancelar" (D83: the previous model stays). */
  cancelRequested: boolean;
  /** Set when the training ended cancelled: the epoch it stopped at. */
  cancelledAt: number | null;
  /** `nn-epoch` events received (smoke test: the progress is live). */
  events: number;
  /** Backend → UI latency of each event with epochs, in ms (smoke test). */
  latencies: number[];
}

function emptyLive(): LiveTraining {
  return { taskId: null, phase: null, epochs: 0, logs: [], cancelRequested: false, cancelledAt: null, events: 0, latencies: [] };
}

class NnState {
  /** "Número de capas ocultas" 1..5. */
  layers = $state(2);
  /** Neurons of the 5 possible layers; `null` = the suggestion of the dataset (not touched). */
  neurons = $state<(number | null)[]>([null, null, null, null, null]);
  activation = $state<NnActivation>("relu");
  /** `null` = the recommended output of the task (D85). */
  output = $state<NnActivation | null>(null);
  /** Task signature the output was chosen for ("classification:3"). */
  outputFor = $state<string | null>(null);
  /** 0..0.8 in tenths. */
  dropout = $state(0.2);
  batchSize = $state(64);
  optimizer = $state<NnOptimizer>("adam");
  learningRate = $state(0.001);
  /** Epochs chosen by the user, by dataset (otherwise `min(100, sugerencia)` of the dataset). */
  epochsBy = $state<Record<string, number>>({});
  validationPct = $state(20);
  earlyStopping = $state(true);
  patience = $state(10);
  reduceLr = $state(true);
  lrFactor = $state(0.5);
  /** D70: fixed seed (Python had none). */
  seed = $state(DEFAULT_SEED);
  live = $state<LiveTraining>(emptyLive());

  /** The neurons of layer `i` with the defaults of `view` (or 10 before the first view). */
  neuronsOf(i: number, view: ArchitectureView | null): number {
    return this.neurons[i] ?? view?.default_neurons[i] ?? 10;
  }

  /** D85: the output for the task of `view`, reset when the task signature changes. */
  outputOf(view: ArchitectureView | null): NnActivation {
    if (!view) {
      return this.output ?? "softmax";
    }
    const signature = taskSignature(view);
    if (this.outputFor !== signature || this.output === null) {
      return view.recommended_output;
    }
    return this.output;
  }

  /** Follows the task of the view (D85); call when a new view arrives. */
  syncTask(view: ArchitectureView): void {
    const signature = taskSignature(view);
    if (this.outputFor !== signature) {
      this.outputFor = signature;
      this.output = view.recommended_output;
    }
  }

  setOutput(view: ArchitectureView, output: NnActivation): void {
    this.outputFor = taskSignature(view);
    this.output = output;
  }

  epochsOf(datasetKey: string, view: ArchitectureView | null): number {
    return this.epochsBy[datasetKey] ?? view?.default_epochs ?? 50;
  }

  setEpochs(datasetKey: string, epochs: number): void {
    this.epochsBy = { ...this.epochsBy, [datasetKey]: epochs };
  }

  arch(view: ArchitectureView | null): NnArchitecture {
    return {
      hidden: Array.from({ length: this.layers }, (_, i) => this.neuronsOf(i, view)),
      activation: this.activation,
      output_activation: this.outputOf(view),
      dropout: Math.round(this.dropout * 10) / 10,
      batch_size: this.batchSize,
      optimizer: this.optimizer,
    };
  }

  training(datasetKey: string, view: ArchitectureView | null): NnTraining {
    return {
      learning_rate: this.learningRate,
      epochs: this.epochsOf(datasetKey, view),
      validation_pct: this.validationPct,
      patience: this.earlyStopping ? this.patience : null,
      lr_factor: this.reduceLr ? Math.round(this.lrFactor * 100) / 100 : null,
      seed: this.seed,
    };
  }

  config(datasetKey: string, view: ArchitectureView | null): ModelConfig {
    return { kind: "neural_net", arch: this.arch(view), training: this.training(datasetKey, view) };
  }

  /** A new training starts: the live chart empties. */
  startLive(epochs: number): void {
    this.live = { ...emptyLive(), epochs };
  }

  /** `nn-epoch`: only the events of the training in progress count. */
  onEpochEvent(event: NnEpochEvent): void {
    const live = this.live;
    if (live.taskId === null || event.task_id !== live.taskId) {
      return;
    }
    live.events += 1;
    if (event.logs.length > 0 && live.latencies.length < 2000) {
      live.latencies.push(Date.now() - event.sent_at_ms);
    }
    live.phase = event.phase;
    live.epochs = event.epochs;
    if (event.logs.length > 0) {
      const known = live.logs.length > 0 ? (live.logs[live.logs.length - 1]?.epoch ?? 0) : 0;
      const fresh = event.logs.filter((log) => log.epoch > known);
      if (fresh.length > 0) {
        live.logs = [...live.logs, ...fresh];
      }
    }
  }

  /** Back to the defaults (scenarios). */
  reset(): void {
    this.layers = 2;
    this.neurons = [null, null, null, null, null];
    this.activation = "relu";
    this.output = null;
    this.outputFor = null;
    this.dropout = 0.2;
    this.batchSize = 64;
    this.optimizer = "adam";
    this.learningRate = 0.001;
    this.epochsBy = {};
    this.validationPct = 20;
    this.earlyStopping = true;
    this.patience = 10;
    this.reduceLr = true;
    this.lrFactor = 0.5;
    this.seed = DEFAULT_SEED;
    this.live = emptyLive();
  }
}

/** Task and number of classes: D85 resets the output when it changes. */
export function taskSignature(view: Pick<ArchitectureView, "task" | "n_classes">): string {
  return `${view.task}:${view.n_classes ?? 0}`;
}

/** `{:,}` of Python (the texts of the lesson keep the English separator, §8.3). */
export function thousands(value: number): string {
  return Math.round(value)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

export const nn = new NnState();
