/**
 * UI state of the lessons that train models (fase2-regresion.md §5.2): the trained model
 * of each lesson, the training in flight and the options of tabs 2–6 (subview, pair of
 * features, ROC options, coefficient class, form values). It survives navigation; the
 * backend keeps the model itself.
 */
import * as backend from "../backend";
import type { Cancellable } from "../backend";
import type { ModelConfig } from "../bindings/ModelConfig";
import type { PredictionResult } from "../bindings/PredictionResult";
import type { TrainedModelInfo } from "../bindings/TrainedModelInfo";
import type { DatasetRef } from "../bindings/DatasetRef";
import type { LessonId } from "../content";
import { datasetKey } from "../session.svelte";
import type { Average } from "../components/ml/figures";

export interface ModelLessonState {
  /** The current model of the lesson (`null` until trained). */
  info: TrainedModelInfo | null;
  training: boolean;
  /** Message of the last failed training (Spanish, from the backend). */
  trainError: string | null;
  /** Kind of the last failed training error (`invalid_target`…). */
  trainErrorKind: string | null;
  /** "¡Modelo entrenado exitosamente!" after the last training of this visit. */
  justTrained: boolean;
  /** Subview of 📉 Visualización (D13: per lesson); `null` = the first one. */
  viz: string | null;
  /** Features of the boundary/surface (`viz_feature1/2`). */
  pair: [number, number];
  rocPerClass: boolean;
  rocAverages: Average[];
  /** Class of the coefficient table (multiclass logistic). */
  coefClass: number;
  /** Values of the prediction form, by model id. */
  form: Record<number, number[]>;
  /** Last prediction, by model id. */
  prediction: Record<number, PredictionResult>;
}

function createLessonState(): ModelLessonState {
  const state = $state<ModelLessonState>({
    info: null,
    training: false,
    trainError: null,
    trainErrorKind: null,
    justTrained: false,
    viz: null,
    pair: [0, 1],
    rocPerClass: true,
    rocAverages: ["macro", "micro"],
    coefClass: 0,
    form: {},
    prediction: {},
  });
  return state;
}

class ModelSession {
  #lessons = new Map<LessonId, ModelLessonState>();
  #training = new Map<LessonId, Cancellable<TrainedModelInfo>>();
  #restored = new Set<LessonId>();

  lesson(id: LessonId): ModelLessonState {
    let state = this.#lessons.get(id);
    if (!state) {
      state = createLessonState();
      this.#lessons.set(id, state);
    }
    return state;
  }

  /** After a reload of the page the backend still has the model: fetch it once. */
  async restore(id: LessonId): Promise<void> {
    if (this.#restored.has(id) || !backend.isTauri()) {
      return;
    }
    this.#restored.add(id);
    try {
      const info = await backend.lessonModel(backend.BACKEND_LESSON[id]);
      const state = this.lesson(id);
      if (info && !state.info) {
        this.#setModel(state, info);
      }
    } catch (error) {
      void backend.logToBackend("warn", `no se pudo recuperar el modelo de ${id}: ${backend.errorMessage(error)}`);
    }
  }

  /** A view answered `model_stale`: the lesson was retrained, take the current model. */
  async refresh(id: LessonId): Promise<void> {
    const info = await backend.lessonModel(backend.BACKEND_LESSON[id]).catch(() => null);
    const state = this.lesson(id);
    if (info?.model_id !== state.info?.model_id) {
      if (info) {
        this.#setModel(state, info);
      } else {
        state.info = null;
      }
    }
  }

  #setModel(state: ModelLessonState, info: TrainedModelInfo): void {
    const previous = state.info;
    state.info = info;
    const n = info.feature_names.length;
    // Keep the pair if it still fits the new model (same features), as Python's keys.
    const [a, b] = state.pair;
    if (!previous || previous.feature_names.join("\u0000") !== info.feature_names.join("\u0000") || a >= n || b >= n) {
      state.pair = [0, Math.min(1, n - 1)];
    }
    const classes = info.class_names?.length ?? 0;
    if (state.coefClass >= Math.max(1, classes)) {
      state.coefClass = 0;
    }
  }

  /** "🚀 Entrenar Modelo": a newer training cancels the one in flight. */
  async train(id: LessonId, dataset: DatasetRef, config: ModelConfig, testSize: number): Promise<void> {
    const state = this.lesson(id);
    this.#training.get(id)?.cancel();
    const request = backend.trainModel({ lesson: backend.BACKEND_LESSON[id], dataset, config, test_size: testSize });
    this.#training.set(id, request);
    state.training = true;
    state.trainError = null;
    state.trainErrorKind = null;
    state.justTrained = false;
    try {
      const info = await request.promise;
      if (this.#training.get(id) !== request) {
        return;
      }
      this.#setModel(state, info);
      state.justTrained = true;
    } catch (error) {
      if (this.#training.get(id) !== request || backend.isCancelled(error)) {
        return;
      }
      state.trainError = backend.errorMessage(error);
      state.trainErrorKind = backend.isCommandError(error) ? error.kind : null;
    } finally {
      if (this.#training.get(id) === request) {
        this.#training.delete(id);
        state.training = false;
      }
    }
  }

  /** "Cancelar" next to the training spinner. */
  cancelTraining(id: LessonId): void {
    this.#training.get(id)?.cancel();
  }

  /** D32: the model was trained on another dataset than the selected one. */
  trainedOnOther(id: LessonId, selected: DatasetRef): boolean {
    const info = this.lesson(id).info;
    return info !== null && datasetKey(info.dataset) !== datasetKey(selected);
  }
}

export const models = new ModelSession();

/** Same model configuration (to tell whether the controls changed since training). */
export function sameConfig(a: ModelConfig, b: ModelConfig): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
