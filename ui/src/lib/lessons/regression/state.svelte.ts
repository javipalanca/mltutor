/**
 * UI state of 📊 Regresión (fase2-regresion.md §1.1 and §5.2): the controls of
 * 🏋️ Entrenamiento. The model and the options of tabs 2–6 live in `models.lesson("regresion")`.
 */
import type { ModelConfig } from "../../bindings/ModelConfig";

export type RegressionModelType = "linear" | "logistic";

class RegressionState {
  /** `model_type_lr`: "Linear" by default. */
  modelType: RegressionModelType = $state("linear");
  /** 100..2000, step 100. */
  maxIter = $state(1000);
  /** 0.1..0.5, step 0.05. */
  testSize = $state(0.3);

  config(): ModelConfig {
    return this.modelType === "linear" ? { kind: "linear" } : { kind: "logistic", max_iter: this.maxIter };
  }
}

export const regression = new RegressionState();

export const LESSON = "regresion" as const;
