# Comparación visual Qt ↔ Rust — Fase 5d (🧠 Redes Neuronales: 🔮 Predicciones y 💾 Exportar)

Fecha: 2026-10-16. Capturas Qt en [`qt/fase5/`](qt/fase5/) y Rust en [`rust/fase5/`](rust/fase5/), con los mismos
números y nombres cuando la pantalla existe en Python. Binario de release (`npm --prefix ui run tauri build`) con
`RealFacade` (motor `mltutor-nn`, semilla 42):

```sh
target/release/mltutor --screenshot rust/fase5/24_prediccion_resultado_iris.png --scenario redes_prediccion_resultado_iris --full-page
```

Escenario = nombre de la captura sin el número, con el prefijo `redes_` (`ui/src/lib/lessons/redes/scenarios.ts`).
En modo captura no se abre ningún diálogo: el lote lee el CSV de `--scenario <nombre>:<ruta.csv>` o, si no se da,
uno con las 25 primeras filas del dataset y una columna de más (`comentario`); las exportaciones se guardan en
`$TMPDIR/mltutor-automatizacion-<pid>/`. Cuantizadas a 256 colores con Pillow: 19 capturas, 2,3 MB.

**Nuevas** (no existen en Python, D81 y K1): `44_lotes_formato_iris`, `45_lotes_iris`, `45b_lotes_diabetes`,
`46_exploracion_iris`, `46b_exploracion_iris_expandido`, `47_exploracion_cancer`, `48_exploracion_diabetes`,
`49_prediccion_divergencia`, `49b_exportar_divergencia` y `25b_exportar_onnx`.

## Resumen

| Pantalla | Qt | Rust | Diferencias |
|---|---|---|---|
| 00d/00e sin modelo | «⚠️ Primero debes entrenar un modelo.» | igual | — |
| 23 🔮 formulario Iris | C9 con rangos de todo el dataset, «Realizar predicción» | igual + panel NUEVO de 3 subvistas (🔮 Individual · 📊 Lotes · 🎲 Exploración) y la nota `escalado_nota` bajo el formulario | panel de subvistas (D81) |
| 24 🔮 resultado Iris | tarjeta «Clase: virginica» + tabla Clase/Probabilidad | igual | si un valor sale del rango de las filas de entrenamiento, aviso NUEVO `fuera_rango` |
| 34 🔮 Cáncer | tarjeta con «Probabilidad clase positiva: 0.9584» dentro | igual (dentro de la tarjeta) + la nota del umbral (> 0,5, D87) | Python usaba ≥ 0,5 (D87) |
| 40 🔮 Diabetes | «Valor: …» | igual | — |
| 44/45 📊 Lotes (nuevas) | — (inalcanzable) | «📋 Formato esperado» con 3 filas y las columnas, `nota_unidades`, «📂 Elegir CSV…» (diálogo nativo desde Rust); tras leerlo: cargado, sobran/reordenadas, generadas, fuera de rango, tabla con Predicción, Probabilidad y Confianza (binaria) o Confianza (multiclase) y «Fuera de rango», ayudas de las columnas y «📥 Descargar Resultados» | las comprobaciones de `[lotes]` en orden (sin_filas → duplicadas → faltan → no_numericas → vacías) |
| 46–48 🎲 Exploración (nuevas) | — (inalcanzable) | muestra base, valores, «🎯 Predicción Base», característica, curva Plotly (50 valores del mínimo al máximo del dataset entero, línea verde «Valor Base», umbral rojo en binaria, doble eje en multiclase), análisis con 2 métricas y el veredicto, notas y el expander de consejos | multiclase: eje izquierdo con los nombres de clase (Python mostraba el índice) y análisis sobre la probabilidad de la clase base (NUEVO) |
| 25 💾 Paquete Keras | «🤖 Modelo TensorFlow» con radio SavedModel/HDF5/TFLite y «Modo seguro» | «📦 Paquete Keras (.zip)»: texto, «Cuándo usarlo» y «💾 Guardar paquete Keras…» con el aviso de éxito | D75: TF no está en Rust; el ZIP genera esos formatos al ejecutarlo |
| 25b 💾 ONNX (nueva) | — | texto, salidas, «Cuándo usarlo», «💾 Guardar ONNX…» | — |
| 26/27 SavedModel, Modelo completo | descarga del SavedModel / pickle | incluidos en el paquete Keras (`reconstruir_modelo.py`) | D75 |
| 28 💾 Código Python | plantilla que devuelve ceros + aviso | predictor NumPy con los pesos reales (código y «💾 Guardar predictor NumPy (red_predictor_numpy.zip)…») y el script de entrenamiento con la semilla del modelo (L1) | D75; el .py va en ZIP con su `modelo.json` (decisión de F5d) |
| 29 💾 Metadatos | `st.json` + «📥 Descargar Metadatos» | igual, con la fecha del **entrenamiento**, `num_classes` correcto, MAE en regresión, métricas de prueba, escalador y `config` (semilla) | D75 |
| Información del Modelo | dos `st.info` en una línea | dos tarjetas con listas | la fecha es la del entrenamiento |
| 49/49b divergencia (nuevas) | (Python: predicción NaN) | `[comun] red_no_valida` en 🔮 y 💾 en lugar del formulario y de los botones | K1 |

## Diferencias que quedan

- Rust muestra la barra de navegación también en 🔮 y 💾 (Python no la tiene en las pestañas 5 y 6); viene de F5b.
- El lote predice al elegir el archivo (Python tenía un segundo botón «🚀 Generar Predicciones»), y la tabla muestra las
  primeras 100 filas; el CSV descargado las contiene todas.
- La tabla de resultados del lote contiene solo las características del modelo (en su orden) y las columnas de la
  predicción: Python también quitaba las columnas sobrantes (`new_df[feature_cols]`).
- «Índice de muestra» es un campo numérico (0 … n − 1) y no un desplegable con todas las filas (California tiene
  20 640).
- La exploración de Python hacía 50 `model.predict`; Rust predice las 50 filas en una pasada (mismo resultado en f32).
