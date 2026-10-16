/**
 * Texts of 📈/🎯 that `ml-lesson-author` adds to `content/redes/mensajes.toml` in F5c; the
 * fallbacks keep the app working (and the smoke test green) until the keys exist.
 */
import { nnRaw, optionalText } from "../../content";
import { fmt } from "../../fmt";

/** K1: the training diverged at `epoch` (1-based). */
export function divergenceText(epoch: number): string {
  return fmt(
    optionalText(
      nnRaw,
      "evaluacion",
      "aviso_divergencia",
      "🚨 El entrenamiento divergió en la época {epoca}: la pérdida se hizo infinita o NaN (no es un número), así que los pesos de la red no sirven y las métricas, la superficie, los pesos y las activaciones no se pueden calcular. Suele deberse a un learning rate demasiado alto: prueba 0.01 o 0.001 y vuelve a entrenar.",
    ),
    { epoca: epoch },
  );
}

/** D87: a tanh output is not a probability. */
export function tanhOutputText(): string {
  return optionalText(
    nnRaw,
    "evaluacion",
    "aviso_salida_tanh",
    "⚠️ La salida de esta red es **tanh** (entre −1 y 1): sus valores no son probabilidades. La clase se decide igual (mayor que 0,5 o la neurona mayor), pero no interpretes las salidas como la confianza de la red.",
  );
}

/** K6: the training may take long (`level` of `training_cost`). */
export function costText(level: "high" | "very_high", seconds: number): string {
  const rounded = seconds < 60 ? Math.max(1, Math.round(seconds)) : Math.round(seconds / 60);
  return level === "very_high"
    ? fmt(
        optionalText(
          nnRaw,
          "entrenamiento",
          "coste_muy_alto",
          "⏳ Esta red es grande para estos datos: el entrenamiento puede tardar varios minutos (unos {minutos} min en un equipo rápido). Puedes cancelarlo en cualquier momento; para probar, reduce las épocas o las neuronas.",
        ),
        { minutos: rounded },
      )
    : fmt(
        optionalText(
          nnRaw,
          "entrenamiento",
          "coste_alto",
          "⏳ El entrenamiento puede tardar unos {segundos} s. Verás el progreso época a época y podrás cancelarlo.",
        ),
        { segundos: rounded },
      );
}
