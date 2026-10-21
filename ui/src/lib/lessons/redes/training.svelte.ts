/**
 * "🧠 Entrenar Red Neuronal" (⚙️ Entrenamiento and the scenarios): trains with the current
 * controls, follows the `nn-epoch` events of that task and, when it ends, takes the
 * authoritative history from the backend (the last batch of events may still be in flight).
 */
import * as backend from "../../backend";
import type { ArchitectureView } from "../../bindings/ArchitectureView";
import type { DatasetRef } from "../../bindings/DatasetRef";
import { models } from "../../ml/modelSession.svelte";
import { datasetKey } from "../../session.svelte";
import { LESSON, NN_TEST_SIZE, nn } from "./state.svelte";

/** Marks of the chart of the last finished training. */
export interface FinishedHistory {
  modelId: number;
  bestEpoch: number | null;
  lrDrops: number[];
}

class TrainingRun {
  history = $state<FinishedHistory | null>(null);

  async train(dataset: DatasetRef, view: ArchitectureView): Promise<void> {
    const key = datasetKey(dataset);
    const config = nn.config(key, view);
    const lessonState = models.lesson(LESSON);
    nn.startLive(config.kind === "neural_net" ? config.training.epochs : 0);
    this.history = null;
    const before = lessonState.info?.model_id ?? null;
    await models.train(LESSON, dataset, config, NN_TEST_SIZE, (taskId) => {
      nn.live.taskId = taskId;
    });
    const info = lessonState.info;
    if (lessonState.justTrained && info && info.model_id !== before) {
      try {
        const full = await backend.nnHistory(info.model_id);
        nn.live.logs = full.epochs;
        this.history = { modelId: info.model_id, bestEpoch: full.best_epoch, lrDrops: full.lr_drops };
      } catch (error) {
        if (!backend.isStale(error)) {
          void backend.logToBackend("warn", `historial de la red: ${backend.errorMessage(error)}`);
        }
      }
    } else if (nn.live.cancelRequested && !lessonState.trainError) {
      nn.live.cancelledAt = nn.live.logs.at(-1)?.epoch ?? 0;
    }
  }

  /** "Cancelar" (D83): the previous model stays. */
  cancel(): void {
    nn.live.cancelRequested = true;
    models.cancelTraining(LESSON);
  }
}

export const training = new TrainingRun();
