/**
 * Options of 🎯 Visualización of 🧠 Redes Neuronales (fase5-redes.md §2.5), kept while the
 * student navigates (D13): bins and log scale of 🧠 Pesos, the 10 parameters of 📉
 * Activaciones and the per-layer checkboxes. The subview and the pair of the surface live in
 * `models.lesson("redes")` (`viz`, `pair`), like the other lessons.
 */
import { DEFAULT_MAP_OPTIONS, type MapOptions } from "../../components/ml/networkLayout";

export const WEIGHT_BINS = { min: 10, max: 120, value: 50 } as const;
export const BIAS_BINS = { min: 5, max: 60, value: 20 } as const;
export const SAMPLES = { min: 10, max: 200, value: 50 } as const;

class NnVizState {
  binsWeights = $state<number>(WEIGHT_BINS.value);
  binsBiases = $state<number>(BIAS_BINS.value);
  logScale = $state(false);
  sampleSize = $state<number>(SAMPLES.value);
  deadThreshold = $state(0);
  constantStd = $state(1e-3);
  showMap = $state(true);
  map = $state<MapOptions>({ ...DEFAULT_MAP_OPTIONS });
  /** "Mostrar histograma activaciones {capa}" by hidden layer (Python `hist_act_{i}`). */
  layerHist = $state<Record<number, boolean>>({});
  /** "Mostrar medias por neurona {capa}" (Python `means_{i}`). */
  layerMeans = $state<Record<number, boolean>>({});

  reset(): void {
    this.binsWeights = WEIGHT_BINS.value;
    this.binsBiases = BIAS_BINS.value;
    this.logScale = false;
    this.sampleSize = SAMPLES.value;
    this.deadThreshold = 0;
    this.constantStd = 1e-3;
    this.showMap = true;
    this.map = { ...DEFAULT_MAP_OPTIONS };
    this.layerHist = {};
    this.layerMeans = {};
  }
}

export const nnViz = new NnVizState();
