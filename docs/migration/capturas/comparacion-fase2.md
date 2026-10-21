# Comparación visual Qt ↔ Rust — Fase 2 (📊 Regresión)

Fecha: 2026-10-14. Capturas Qt en [`qt/fase2/`](qt/fase2/) (`tools/capture_qt.py --phase 2`) y
Rust en [`rust/fase2/`](rust/fase2/), con los mismos números y nombres. Las Rust se generan con el
binario de release y `RealFacade` (los cálculos son de `mltutor-core` y `mltutor-export`):

```sh
target/release/mltutor --screenshot rust/fase2/05_roc_multiclase_iris.png \
    --scenario regresion_roc_multiclase_iris --full-page
```

El escenario es el nombre Qt sin el número y con el prefijo `regresion_`
(`ui/src/lib/lessons/regression/scenarios.ts`). Las `_ventana` se capturan sin `--full-page`.
Cuantizadas a 256 colores con Pillow: 47 capturas, 6,2 MB. Igual que en F1, las Qt `NNN.png` son
solo el contenido (1135 px) y las Rust la ventana entera a 1400 px con la barra lateral.

No hay equivalente Rust de `19_exportar_pickle` (D28: sin pickle directo; hay ONNX y paquete
Python) ni de `44_error_roc_logistica_diabetes` (D26: la logística ya no se entrena con un objetivo
de regresión). `43_logistica_bloqueada_diabetes` sustituye a `43_logistica_entrenada_en_diabetes`.

## Resumen

| Pantalla | Coincide | Mejora | Difiere |
|---|---|---|---|
| 🏋️ Entrenamiento (01, 02, 20, 29, 30, 41, 42, 43) | Selector, "Configuración del Modelo", caja de información, botones de tipo, sliders (1000 / 0,3), botón "🚀 Entrenar Modelo", "¡Modelo entrenado exitosamente!", avisos de tipo de modelo | Spinner con "Cancelar" mientras entrena y progreso en la barra de estado; D26: error en español con la explicación en vez de entrenar 192 clases | Cáncer converge en Rust, así que no sale el aviso D19 (Python no convergía, pero no avisaba) |
| Pestañas 2–6 (todas) | "📊 Dataset actual" | D32: "🏋️ Modelo entrenado con …" y aviso si se cambió de dataset | — |
| 📈 Evaluación clasificación (03, 04, 21) | 4 métricas con insignia, matriz de confusión Blues con recuentos, informe por clase, tablas de muestras, precisión por clase #3274a1, interpretación | D33: informe con 4 decimales; D34: insignias con icono; matriz y barras interactivas | El informe y las muestras van a dos columnas, como en Qt |
| 📈 Evaluación regresión (31, 32) | MSE/RMSE/MAE/R², dispersión coloreada por error (viridis) con diagonal roja, histograma de errores con KDE y línea en 0 | Los dos gráficos lado a lado (en Python, uno debajo de otro); textos de métricas corregidos por el autor de contenido | — |
| 📉 ROC (05, 06, 07, 22) | Opciones (checkbox + macro/micro), colores por clase de matplotlib, micro deeppink punteado, macro navy discontinuo (AP como línea horizontal), interpretación por promedio, tabla por clase | Filtrado sin volver a calcular; leyenda propia en cada subgráfico; nota de la clase positiva en binaria (Cáncer: *benign*) | — |
| 🌈 Frontera (08, 09, 10, 23) | Mallas y avisos idénticos ("210 x 270 = 56,700", reducción a 200 × 200), puntos de train con borde negro, título, aviso 2D, código | D23: el mismo color por clase en fondo, puntos y leyenda; D30: el código usa el par elegido; con la misma característica dos veces sale un aviso | El fondo es un heatmap a nivel de celda (sin el antialiasing de `contourf`) |
| 📊 Probabilidades (11, 24, 25) | Histogramas por clase real (lightcoral/lightblue), umbral 0,5, métricas de incertidumbre, aviso de separación, umbrales | D21: la tabla muestra 0,3, 0,5 y 0,7 (Python solo 0,5) con nota | — |
| 📈 Residuos (33, 34) | Dispersión con "Predicción Perfecta", caja wheat, residuos vs predicción con "Desv. Estándar", histograma skyblue con media e ideal, tres comprobaciones | D22: R² real (0,4773) en lugar de 0,0000 | Textos de las comprobaciones corregidos (contenido) |
| 🌐 Superficie (35, 36) | Malla ±1 (D24), contornos viridis, puntos de train, barra "Valor Objetivo" | Interactiva | — |
| 🔍 Coeficientes (12, 13, 14, 28, 37, 38) | Tabla, intercepto, barras #4ecdc4/#ff6b6b con valores, análisis detallado, resumen, recomendaciones, aviso de extremos | D27: intercepto de la clase elegida, numeración 1-2-3, fuerza simétrica; la mayor barra arriba (Python la dibuja abajo); leyenda fuera del gráfico | "Odds base" solo en binaria (texto corregido por el autor de contenido) |
| 🔮 Predicciones (15, 16, 26, 27, 39, 40) | Formulario a dos columnas, selects con los textos de los metadatos, sliders con la media, tarjeta verde "Resultado de la predicción", tabla de probabilidades | Nombre en negrita y descripción en cursiva bajo el nombre; D25: la lineal muestra "Valor:" y un aviso | En Iris se ven descripciones que Python nunca mostraba (MEMORY, F1) |
| 💾 Exportar (17, 18) | "📄 Generar Código" | D28: "💾 Exportar a ONNX" y "📦 Paquete Python (.zip)" con diálogo nativo; D29: script completo y ejecutable; expander "¿Qué formato elijo?" | Sin pickle |

## Diferencias que quedan

- **Tema oscuro** no capturado (la captura sigue el aspecto del sistema). Los gráficos usan los
  colores del tema; la tarjeta de predicción y las cajas tienen variantes oscuras.
- **Matplotlib frente a Plotly:** tipografía Inter en lugar de DejaVu Sans, rejilla más suave y
  leyendas con fondo semitransparente. Los títulos de los subgráficos van en negrita como en
  matplotlib.
- **Frontera:** Plotly no suaviza las fronteras del heatmap; a 200 × 200 se nota un escalonado leve
  en diagonales (también en `contourf`, pero menos).
- **Altura de los gráficos:** fija por tipo (Python usaba pulgadas por figura). Se adaptan al ancho.
