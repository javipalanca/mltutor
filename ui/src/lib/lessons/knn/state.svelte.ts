/**
 * UI state of 🔍 K-Nearest Neighbors (fase4-knn.md §1.1 and §5.2): the controls of
 * 🏋️ Entrenamiento and the state of the interactive canvas (C18, D54). The model and the
 * shared options (subview, pair of the boundary, ROC, form) live in `models.lesson("knn")`.
 */
import type { KnnMetric } from "../../bindings/KnnMetric";
import type { KnnWeights } from "../../bindings/KnnWeights";
import type { ModelConfig } from "../../bindings/ModelConfig";
import type { TrainedModelInfo } from "../../bindings/TrainedModelInfo";

export const LESSON = "knn" as const;

/** "Vecinos (K)" 1..20, and the K of the canvas (D55). */
export const K_MIN = 1;
export const K_MAX = 20;

/**
 * Test points of the canvas: the same limit as the backend (`MAX_PLAYGROUND_POINTS` in
 * `src-tauri/src/knn.rs`, which also rejects longer requests with this text).
 */
export const MAX_PLAYGROUND_POINTS = 200;
export const MAX_POINTS_MESSAGE = `El lienzo admite como máximo ${MAX_PLAYGROUND_POINTS} puntos de prueba.`;

export interface TestPoint {
  /** Stable id (keys of the table, drag target). */
  id: number;
  x: number;
  y: number;
}

/** The canvas of one trained model: controls, pair and test points. */
export interface PlaygroundState {
  /** Model the state belongs to: retraining (or another dataset) empties it. */
  modelId: number | null;
  pair: [number, number];
  k: number;
  weights: KnnWeights;
  metric: KnnMetric;
  showConfidence: boolean;
  showDistances: boolean;
  points: TestPoint[];
}

class KnnState {
  k = $state(5);
  weights: KnnWeights = $state("uniform");
  metric: KnnMetric = $state("minkowski");
  /** 0.1..0.5, step 0.05. */
  testSize = $state(0.3);
  playground = $state<PlaygroundState>(emptyPlayground());
  #nextPoint = 1;

  config(): ModelConfig {
    return { kind: "knn", n_neighbors: this.k, weights: this.weights, metric: this.metric };
  }

  /**
   * The canvas starts with the K, weights and metric of the trained model (Python's
   * `currentK` and selects) and is emptied when the model changes; otherwise it keeps its
   * state across navigation and subviews (D13).
   */
  syncPlayground(info: TrainedModelInfo): void {
    if (this.playground.modelId === info.model_id) {
      return;
    }
    const n = info.feature_names.length;
    const knn = info.config.kind === "knn" ? info.config : null;
    const keepPair =
      this.playground.modelId !== null && this.playground.pair[0] < n && this.playground.pair[1] < n;
    this.playground = {
      ...emptyPlayground(),
      modelId: info.model_id,
      pair: keepPair ? this.playground.pair : [0, Math.min(1, n - 1)],
      k: Math.min(K_MAX, Math.max(K_MIN, knn?.n_neighbors ?? 5)),
      weights: knn?.weights ?? "uniform",
      metric: knn?.metric ?? "minkowski",
    };
  }

  /** Whether another test point fits ([`MAX_PLAYGROUND_POINTS`]). */
  canAddPoint(): boolean {
    return this.playground.points.length < MAX_PLAYGROUND_POINTS;
  }

  /** Adds a test point; `null` when the canvas is full. */
  addPoint(x: number, y: number): TestPoint | null {
    if (!this.canAddPoint()) {
      return null;
    }
    const point = { id: this.#nextPoint++, x, y };
    this.playground.points = [...this.playground.points, point];
    return point;
  }

  movePoint(id: number, x: number, y: number): void {
    this.playground.points = this.playground.points.map((p) => (p.id === id ? { id, x, y } : p));
  }

  removePoint(id: number): void {
    this.playground.points = this.playground.points.filter((p) => p.id !== id);
  }

  clearPoints(): void {
    this.playground.points = [];
  }

  /** A new pair of features: the old points mean nothing in the new axes. */
  setPair(pair: [number, number]): void {
    this.playground.pair = pair;
    this.playground.points = [];
  }

  /** Back to the defaults (scenarios). */
  reset(): void {
    this.k = 5;
    this.weights = "uniform";
    this.metric = "minkowski";
    this.testSize = 0.3;
    this.playground = emptyPlayground();
  }
}

function emptyPlayground(): PlaygroundState {
  return {
    modelId: null,
    pair: [0, 1],
    k: 5,
    weights: "uniform",
    metric: "minkowski",
    showConfidence: true,
    showDistances: true,
    points: [],
  };
}

export const knn = new KnnState();
