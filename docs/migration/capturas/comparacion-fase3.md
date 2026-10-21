# Comparación visual Qt ↔ Rust — Fase 3 (🌲 Árboles de Decisión)

Fecha: 2026-10-15. Capturas Qt en [`qt/fase3/`](qt/fase3/) (`tools/capture_qt.py --phase 3`) y Rust en
[`rust/fase3/`](rust/fase3/), con los mismos números y nombres. Las de Rust se han hecho con el binario de
release y `RealFacade`, así que los cálculos son de `mltutor-core` y `mltutor-export`:

```sh
target/release/mltutor --screenshot rust/fase3/05_arbol_iris.png --scenario arboles_arbol_iris --full-page
```

El escenario es el nombre Qt sin el número y con el prefijo `arboles_`
(`ui/src/lib/lessons/trees/scenarios.ts`). Las capturas `_ventana` se hacen sin `--full-page`. Están
cuantizadas a 256 colores con Pillow: 49 capturas, 4,7 MB. Como en F1 y F2, las `NNN.png` de Qt muestran
solo el contenido (1135 px) y las de Rust, la ventana entera a 1400 px con la barra lateral.

**Sin equivalente en Rust:**
- `19_exportar_pickle` (D43: no hay pickle directo; el paquete Python lo reconstruye);
- `20_exportar_reglas` y `21_exportar_codigo_python`: en Qt, el botón muestra otro botón de descarga; en
  Rust abre directamente el diálogo nativo de guardar, que no se puede capturar;
- `43_error_evaluacion_clasificacion_diabetes`: con D37 ya no se entrena el clasificador de 192 clases.

`42_clasificacion_bloqueada_diabetes` sustituye a `42_clasificacion_entrenada_en_diabetes`.

**Nuevas:**
- `13b_caracteristicas_permutacion_iris` (D38);
- `21b_exportar_codigo_entrenamiento` (el script de entrenamiento);
- `44b_regresion_entrenada_en_iris` (evaluación de regresión sobre Iris);
- `40_arbol_california_d8_ventana`.

## Resumen

| Pantalla | Coincide | Mejora | Difiere |
|---|---|---|---|
| 🏋️ Entrenamiento (01, 02, 22, 27, 28, 29, 41, 42, 44) | Selector, "Configuración del Modelo", caja de información, botones 🏷️/📈 con ayuda, aviso de tipo, criterio (gini/entropy o los tres de regresión), sliders 3/2/0,3, "Entrenar Modelo" sin emoji, "¡Modelo entrenado con éxito!" | D37: el tipo se preselecciona con la tarea del dataset, y la clasificación con un objetivo continuo da un error en español con la explicación (Python entrenaba 192 clases en Diabetes y fallaba en inglés en California); D49: "👉 Ve a la pestaña '📉 Visualización'"; expanders de criterios y parámetros; aviso de regresión sobre códigos de clase; spinner con "Cancelar" | Al entrar en Diabetes sale 📈 Regresión (en Python, 🏷️ Clasificación) |
| Pestañas 2–6 (todas) | "📊 Dataset actual", el `st.warning` sin modelo | D32/D45: "Modelo entrenado con…", la línea de configuración del árbol (criterio, profundidad, nodos, hojas) y un aviso si los controles han cambiado sin reentrenar | Mandan el modelo y su tipo, no el botón actual (D45) |
| 📈 Evaluación (03, 04, 30) | C1/C2/C3 de F2: métricas, matriz de confusión, informe, muestras, interpretación (incluido el literal "regresión lineal") | Igual que en F2 | — |
| 🌲 Árbol (05, 06, 07, 23, 31, 40) | Mismas cajas, textos (`gini = 0.664`, `samples = 100.0%`, `value = [0.295, 0.352, 0.352]`, `class = versicolor`), colores de `_color_brew` mezclados con blanco (raíz de Iris blanca), bordes redondeados, flechas, "True"/"False" en la raíz y disposición de Buchheim. Sliders de ancho y alto (14 × 10 y 20 × 15) | D41: SVG con letra fija y legible a cualquier tamaño, zoom con rueda y botones, arrastre, teclado, tooltip por nodo (con los recuentos por clase) y primera vista centrada en la raíz cuando el árbol entero no se puede leer (California d8, 407 nodos: Python no mostraba nada legible). D42: descargas PNG (a 3×, limitada a 16 000 px) y SVG. Expander "📖 Cómo leer el árbol" | La letra es Inter y no DejaVu Sans; las cajas tienen ancho propio según el texto; el lienzo usa `ancho × 96 × 0,66` px con la proporción ancho/alto |
| 📝 Texto (08, 32) | `export_text` con pesos, monoespaciado | D40: "class: setosa" en lugar de "class: 0"; nota sobre cómo leerlo; descarga `arbol_texto.txt` | — |
| 🌈 Frontera / 🌐 Superficie (09, 10, 34, 35) | C8 de F2 con el árbol 2D reentrenado (mallas y avisos idénticos) | Los mismos colores por clase en el fondo, los puntos y la leyenda (D23) | El fondo escalonado de un árbol se dibuja a nivel de celda |
| 📉 ROC (11, 24) | C4 con las probabilidades de hoja (curvas de pocos puntos), binaria en Cáncer | Igual que en F2 | — |
| 📈 Residuos (33) | C7 de F2 | R² real (D22) | — |
| 🔍 Características (12, 13, 13b, 36, 37) | Barras viridis por posición, orden ascendente (empates en 0 por índice), valor con 3 decimales, título en dos líneas, tabla "Valores Detallados" descendente con el índice de la columna, expander con los textos nativos y los consejos | D38: panel "🌳 Importancia del modelo" / "🔀 Importancia por permutación" (barras de error, columna "Desv. Estándar", explicación del método) con nota en cada subvista y el expander "¿Por qué no coinciden…?" | La tabla muestra 0.0000 en lugar de 0 (4 decimales fijos); el eje x llega un 14 % más allá de la mayor barra |
| 🔮 Predicciones (14, 15, 16, 25, 26, 38) | Formulario a dos columnas con los selects de metadatos, "Realizar predicción", tarjeta verde, "### Camino de decisión" con las mismas líneas (`- petal length (cm) = 3.8705 > 2.4500 ✅`), código del camino | D46: tabla "Probabilidades de la hoja" con su nota, valor legible de las categóricas ("Hombre_Adulto = 0 (No)", "Clase_Social = 1 (Segunda)"), línea final con la hoja y expander "Cómo se obtiene la predicción" | — |
| 💾 Exportar (17, 18, 21b, 39) | "Formatos de Exportación", "📝 Exportar Reglas como Texto", "🐍 Exportar como Código Python", vista previa en expander con la "Información del código generado", "Código para Usar el Modelo" | D43: "💾 Exportar a ONNX" y "📦 Paquete Python (.zip)", más "📄 Código de Entrenamiento"; D44: la vista previa y la descarga son la función corregida (compila y da las mismas predicciones), con el expander de los umbrales float32; "¿Qué formato elijo?" | Cinco botones en una fila (el texto se parte en dos líneas a 1400 px) |

## Diferencias que quedan

- **Árbol grande (California d8):** la primera vista muestra la raíz al 80 %. Para ver el resto hay que
  arrastrar o pulsar "Ajustar" (que muestra el árbol entero sin poder leerse, como en Python). No hay
  minimapa.
- **Tema oscuro:** no se ha capturado. El lienzo del diagrama es siempre blanco, porque las cajas usan
  los colores de matplotlib y el texto va en negro.
- **Matplotlib frente a SVG/Plotly:** cambian la tipografía y la rejilla, como en F2.
- **California en RealFacade:** el split del core usa 14 447 filas de entrenamiento y scikit-learn,
  14 448. Es un fallo del core (ver el informe de la fase). Los valores del árbol de California difieren
  en el tercer decimal (`squared_error = 0.564` frente a `0.565` en el nodo 2) y, con una fila menos, el
  árbol completo puede no coincidir con el de scikit-learn. El resto de los datasets coincide exactamente
  (`real_trees::tests`).
