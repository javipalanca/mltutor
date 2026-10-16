/**
 * UI state of 🔮 Predicciones and 💾 Exportar of 🧠 Redes Neuronales (fase5-redes.md §5.2,
 * C29, D75), kept while the student navigates (D13): the subview of each panel, the base
 * sample and feature of 🎲 Exploración and the last batch of 📊 Predicción por lotes, by model.
 */
import type { BatchPredictView } from "../../bindings/BatchPredictView";

/** `[predicciones] subvistas`. */
export type PredictionSubview = "Individual" | "Lotes" | "Exploracion";
/** `[exportar_rust] botones`. */
export type ExportSubview = "Keras" | "Onnx" | "Python" | "Metadatos";

class NnPredictionsState {
  subview = $state<PredictionSubview>("Individual");
  exportView = $state<ExportSubview>("Keras");
  /** "Índice de muestra" and "Característica a explorar", by model id. */
  exploreRow = $state<Record<number, number>>({});
  exploreFeature = $state<Record<number, number>>({});
  /** The last batch of each model (the CSV with every row stays in Rust). */
  batch = $state<Record<number, BatchPredictView>>({});
  /** The saved file of 💾 Exportar, by model and format (success notice). */
  saved = $state<Record<string, string>>({});

  reset(): void {
    this.subview = "Individual";
    this.exportView = "Keras";
    this.exploreRow = {};
    this.exploreFeature = {};
    this.batch = {};
    this.saved = {};
  }
}

export const nnPredictions = new NnPredictionsState();
