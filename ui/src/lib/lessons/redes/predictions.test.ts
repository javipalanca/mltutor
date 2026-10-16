/** F5d: 🔮 Predicciones and 💾 Exportar of 🧠 Redes Neuronales (C29, D75, D81). */
import { describe, expect, it } from "vitest";
import { exploreFigure } from "../../components/ml/nnFigures";
import { hasMd, nnRaw, nnText } from "../../content";
import { invalidNetworkText } from "./nnTexts";
import { nnPredictions } from "./predictions.svelte";

const xs = Array.from({ length: 50 }, (_, i) => 1 + (i * 5.9) / 49);

describe("🎲 Exploración", () => {
  it("binary: probability, the 0.5 threshold and the base value line", () => {
    const fig = exploreFigure(
      { kind: "binary", xs, ys: xs.map((x) => x / 10), classes: null, confidence: null, baseValue: 3 },
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
      { kind: "multiclass", xs, ys: xs.map(() => 0.5), classes: xs.map((x) => (x < 3 ? 0 : 2)), confidence: xs.map(() => 0.7), baseValue: 1.4 },
      "petal length (cm)",
      ["setosa", "versicolor", "virginica"],
    );
    expect(fig.data).toHaveLength(2);
    expect(fig.layout.yaxis?.ticktext).toEqual(["setosa", "versicolor", "virginica"]);
    expect(fig.layout.yaxis2?.overlaying).toBe("y");
  });

  it("regression: one series, the predicted value axis", () => {
    const fig = exploreFigure({ kind: "regression", xs, ys: xs, classes: null, confidence: null, baseValue: 2 }, "bmi", []);
    expect(fig.data).toHaveLength(1);
    expect(fig.layout.shapes).toHaveLength(1);
  });
});

describe("contenido de 🔮 y 💾", () => {
  it("has every key the tabs read", () => {
    const keys: Record<string, string[]> = {
      comun: ["red_no_valida", "motivo_no_finita", "sin_entrenar"],
      predicciones: ["header", "probabilidad_positiva", "umbral_nota", "salida_tanh_binaria", "nota_tanh_multiclase", "escalado_nota", "fuera_rango", "fuera_rango_item", "error_prediccion"],
      lotes: [
        "titulo", "texto", "btn_elegir", "nota_unidades", "probabilidad_help", "confianza_help", "cargado", "faltan", "faltan_ayuda", "sobran",
        "reordenadas", "columna_prediccion", "columna_probabilidad", "columna_confianza", "columna_fuera_rango", "columna_fuera_rango_help",
        "generadas", "btn_descargar", "error", "formato", "columnas",
      ],
      exploracion: [
        "titulo", "que_es_expander", "intro", "paso_1", "tip_base", "muestra_label", "paso_2", "paso_2_caption", "base_multiclase", "base_binaria",
        "base_regresion", "paso_3", "objetivo", "caracteristica_label", "caracteristica_help", "grafica_multiclase", "grafica_binaria",
        "grafica_regresion", "serie_clase", "serie_confianza", "serie_probabilidad", "serie_prediccion", "eje_valor_predicho", "umbral",
        "valor_base", "analisis", "rango_probabilidades", "rango_probabilidades_help", "volatilidad", "volatilidad_help", "rango_predicciones",
        "rango_predicciones_help", "impacto_relativo", "impacto_relativo_help", "muy_influyente", "moderada", "poco_influyente",
        "muy_influyente_regresion", "moderada_regresion", "poco_influyente_regresion", "consejos_expander", "multiclase_nota", "nota_una_muestra", "error",
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
