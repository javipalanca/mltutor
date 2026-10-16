/** F5d: 🔮 Predicciones and 💾 Exportar of 🧠 Redes Neuronales (C29, D75, D81). */
import { describe, expect, it } from "vitest";
import { needsOwnScales, OWN_SCALE_FRACTION } from "../../components/ml/figures";
import { exploreFigure } from "../../components/ml/nnFigures";
import { hasMd, nnRaw, nnText } from "../../content";
import { invalidNetworkText, networkBlocked, recoveredDivergenceText } from "./nnTexts";
import { nnPredictions } from "./predictions.svelte";

const xs = Array.from({ length: 50 }, (_, i) => 1 + (i * 5.9) / 49);

describe("🎲 Exploración", () => {
  it("binary: probability, the 0.5 threshold and the base value line", () => {
    const fig = exploreFigure(
      { kind: "binary", xs, ys: xs.map((x) => x / 10), classes: null, confidence: null, baseValue: 3, outputIsProbability: true, baseClass: null },
      "mean <radius>",
      ["malignant", "benign"],
    );
    expect(fig.data).toHaveLength(1);
    expect(fig.layout.shapes).toHaveLength(2);
    expect(JSON.stringify(fig.layout)).not.toContain("<radius>");
    expect(String((fig.layout.title as { text: string }).text)).toContain("Probabilidad");
  });

  it("multiclass: class names on the left axis and the confidence on the right", () => {
    const fig = exploreFigure(
      { kind: "multiclass", xs, ys: xs.map(() => 0.5), classes: xs.map((x) => (x < 3 ? 0 : 2)), confidence: xs.map(() => 0.7), baseValue: 1.4, outputIsProbability: true, baseClass: 0 },
      "petal length (cm)",
      ["setosa", "versicolor", "virginica"],
    );
    // NEW: the curve of the analysis (probability of the base class) on the right axis.
    expect(fig.data).toHaveLength(3);
    expect(fig.data[2]?.name).toBe(nnText("exploracion", "probabilidad_clase_base").replace("{clase}", "setosa"));
    expect((fig.data[2] as { yaxis?: string }).yaxis).toBe("y2");
    expect(fig.layout.yaxis?.ticktext).toEqual(["setosa", "versicolor", "virginica"]);
    expect(fig.layout.yaxis2?.overlaying).toBe("y");
  });

  it("tanh: the neuron's output, not a probability", () => {
    const binary = exploreFigure(
      { kind: "binary", xs, ys: xs.map(() => -0.3), classes: null, confidence: null, baseValue: 3, outputIsProbability: false, baseClass: null },
      "x",
      ["a", "b"],
    );
    const neuron = nnText("exploracion", "salida_neurona");
    expect(binary.data[0]?.name).toBe(neuron);
    expect(String((binary.layout.yaxis?.title as { text: string }).text)).toBe(neuron);
    const multi = exploreFigure(
      { kind: "multiclass", xs, ys: xs.map(() => -0.2), classes: xs.map(() => 1), confidence: xs.map(() => 0.4), baseValue: 1, outputIsProbability: false, baseClass: 1 },
      "x",
      ["a", "b", "c"],
    );
    expect(multi.layout.yaxis2?.range).toEqual([-1.05, 1.05]);
    expect(JSON.stringify(multi.data)).not.toContain(nnText("exploracion", "serie_confianza"));
  });

  it("regression: one series, the predicted value axis", () => {
    const fig = exploreFigure({ kind: "regression", xs, ys: xs, classes: null, confidence: null, baseValue: 2, outputIsProbability: true, baseClass: null }, "bmi", []);
    expect(fig.data).toHaveLength(1);
    expect(fig.layout.shapes).toHaveLength(1);
  });
});

describe("contenido de 🔮 y 💾", () => {
  it("has every key the tabs read", () => {
    const keys: Record<string, string[]> = {
      comun: ["red_no_valida", "motivo_no_finita", "sin_entrenar"],
      predicciones: ["header", "probabilidad_positiva", "umbral_nota", "salida_tanh_binaria", "nota_tanh_multiclase", "escalado_nota", "fuera_rango", "fuera_rango_relu", "fuera_rango_item", "error_prediccion"],
      evaluacion: ["aviso_divergencia", "aviso_divergencia_recuperada", "aviso_casi_divergencia", "aviso_salida_tanh"],
      activaciones: ["muertas_no_relu"],
      lotes: [
        "titulo", "texto", "btn_elegir", "nota_unidades", "probabilidad_help", "confianza_help", "cargado", "faltan", "faltan_ayuda", "sobran",
        "reordenadas", "columna_prediccion", "columna_probabilidad", "columna_confianza", "columna_fuera_rango", "columna_fuera_rango_help",
        "generadas", "btn_descargar", "error", "formato", "columnas", "confianza_help_binaria", "confianza_help_multiclase", "fila_no_finita",
        "demasiadas_filas", "una_columna_ok",
      ],
      exploracion: [
        "titulo", "que_es_expander", "intro", "paso_1", "tip_base", "muestra_label", "paso_2", "paso_2_caption", "base_multiclase", "base_binaria",
        "base_regresion", "paso_3", "objetivo", "caracteristica_label", "caracteristica_help", "grafica_multiclase", "grafica_binaria",
        "grafica_regresion", "serie_clase", "serie_confianza", "serie_probabilidad", "serie_prediccion", "eje_valor_predicho", "umbral",
        "valor_base", "analisis", "rango_probabilidades", "rango_probabilidades_help", "volatilidad", "volatilidad_help", "rango_predicciones",
        "rango_predicciones_help", "impacto_relativo", "impacto_relativo_help", "muy_influyente", "moderada", "poco_influyente",
        "muy_influyente_regresion", "moderada_regresion", "poco_influyente_regresion", "consejos_expander", "multiclase_nota", "nota_una_muestra", "error",
        "salida_neurona", "nota_tanh_binaria", "probabilidad_clase_base", "impacto_relativo_nota",
      ],
      exportar: ["header", "info_titulo", "info_arquitectura", "info_parametros", "opciones_titulo", "python_titulo", "metadatos_titulo", "metadatos_descargar", "adicional_titulo"],
      exportar_rust: [
        "keras_titulo", "keras_texto", "btn_keras", "keras_ok", "keras_cuando", "onnx_titulo", "onnx_texto", "btn_onnx", "onnx_ok",
        "onnx_salidas_clasificacion", "onnx_salidas_regresion", "onnx_cuando", "python_texto", "python_cuando", "codigo_entrenamiento",
        "metadatos_texto", "metadatos_fecha_nota", "metadatos_cuando", "ayuda_expander", "archivo_numpy", "archivo_entrenamiento",
      ],
    };
    for (const [section, list] of Object.entries(keys)) {
      for (const key of list) {
        expect(nnRaw(section, key), `[${section}] ${key}`).toBeTypeOf("string");
      }
    }
    expect(nnRaw("predicciones", "subvistas")).toHaveLength(3);
    expect(nnRaw("exportar_rust", "botones")).toHaveLength(4);
    for (const file of ["exportar.md", "exportar-recomendaciones.md", "exploracion-que-es.md", "exploracion-consejos.md"]) {
      expect(hasMd(`redes/${file}`), file).toBe(true);
    }
    expect(invalidNetworkText()).not.toContain("{motivo}");
    // The agreed placeholders.
    const recovered = recoveredDivergenceText(3, 1);
    expect(recovered).not.toMatch(/\{(epoca|mejor)\}/);
    expect(recovered).toContain("época 3");
    expect(recovered).toContain("época 1");
    expect(nnText("exploracion", "probabilidad_clase_base")).toContain("{clase}");
    expect(nnText("lotes", "demasiadas_filas")).toContain("{max}");
    for (const key of ["fuera_rango", "fuera_rango_relu"]) {
      expect(nnText("predicciones", key)).toContain("{detalles}");
    }
    expect(nnText("exportar_rust", "archivo_numpy")).toMatch(/\.zip$/);
  });

  it("keeps the subviews and resets them", () => {
    nnPredictions.subview = "Lotes";
    nnPredictions.exploreRow = { 3: 7 };
    nnPredictions.reset();
    expect(nnPredictions.subview).toBe("Individual");
    expect(nnPredictions.exploreRow).toEqual({});
  });
});

describe("K1 y la superficie", () => {
  it("only non-finite weights block the views", () => {
    expect(networkBlocked({ network_valid: false })).toBe(true);
    expect(networkBlocked({ network_valid: true })).toBe(false);
  });

  it("shares the colour scale unless the predictions cover < 20 % of the real range", () => {
    expect(OWN_SCALE_FRACTION).toBe(0.2);
    // Diabetes with the defaults: predictions ≈ 0.8 for targets 25–346.
    expect(needsOwnScales(0.7, 0.9, 25, 346)).toBe(true);
    // A network that learnt: shared, so equal colours mean equal values.
    expect(needsOwnScales(60, 300, 25, 346)).toBe(false);
    expect(needsOwnScales(1, 2, 3, 3)).toBe(false);
  });
});
