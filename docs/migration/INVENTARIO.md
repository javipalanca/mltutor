# Inventario funcional de MLTutor (referencia para la migración)

Este documento lista lo que el estudiante puede ver y hacer en la versión Python.
La referencia es el código de `codex/desktop-migration-study`, la versión Qt, que
reproduce fielmente la app Streamlit de `main`. **La versión Rust debe cubrir cada
casilla.** Cuando se porte un elemento, se marca con `[x]` y se añade el PR o commit.

Cada `migration-porter` detalla su módulo en `docs/migration/<módulo>.md`: controles
con rangos y valores por defecto, textos y casos límite.

Leyenda: 🐍 = origen en Python · ⚠️ = requiere decisión o implementación propia.

---

## 0. Estructura común

- [x] Barra lateral: "Navegación" con 🏠 Inicio, 🌲 Árboles de Decisión, 📊 Regresión,
      🔍 K-Nearest Neighbors y 🧠 Redes Neuronales; "🔧 Herramientas" con 📁 Cargar CSV
      Personalizado. El botón activo aparece resaltado. 🐍 `app.py`
- [x] Página de inicio: bienvenida, tarjetas "📚 Algoritmos disponibles" con los botones
      "✨ Explorar …", "📝 Cómo usar MLTutor" y pie de página. 🐍 `ui.py`
      (`show_welcome_page`, `show_footer`)
- [x] Cada lección tiene pestañas en secuencia, con botones "Volver a …" y *(F1: estructura; contenido de cada pestaña en F2–F5)*
      "Continuar a …" (`navbar`). Se conserva la pestaña activa por lección. 🐍 `apps/navbar.py`
- [x] Panel de botones para elegir subvista (`create_button_panel`). 🐍 `ui.py`
- [x] Avisos (info, warning, error, success), métricas, expanders, columnas, tablas
      (en Qt no se pueden ordenar; ordenarlas es una mejora opcional) y bloques de código Python con resaltado, botón de copiar y descarga.
- [x] Los cálculos se hacen en segundo plano, con progreso y cancelación; la interfaz *(F1: Cancelar en la lectura de CSV y la correlación; el entrenamiento se añade en F2–F5)*
      nunca se bloquea.
- [ ] Diálogos nativos para guardar las exportaciones y abrir CSV. *(F1: abrir CSV y guardar PNG/código; exportaciones en F2–F6)*
- [x] Funciona sin conexión: datasets, gráficos y fuentes van incluidos.

## 1. Datos (pestaña "📊 Datos", común a las cuatro lecciones) 🐍 `dataset/`

- [x] Selector de 11 datasets incluidos: 🌸 Iris, 🍷 Vino, 🔬 Cáncer, 🔢 Dígitos, 🩺 Diabetes,
      💎 Diamantes (muestreo), ⛽ MPG, 🚢 Titanic, 💰 Propinas, 🏠 Viviendas California y
      🐧 Pingüinos. También 🌙 Moons sintético, sin controles en Python (500 muestras,
      ruido 0.2). En Rust se genera una vez por sesión con semilla 42 (D9). En MPG, `name` se muestra en la exploración pero no es
      característica del modelo (D8).
      ⚠️ California y Diamantes se descargan hoy de internet: hay que incluirlos.
- [x] Carga de CSV propio: elegir la columna objetivo, detectar automáticamente si es
      clasificación o regresión, codificar etiquetas y nombrar las clases.
- [ ] Información del dataset: muestras, características y clases. La descripción aparece en *(F1: parte de Datos hecha)*
      la pestaña Entrenamiento, que también muestra el selector en árboles, regresión y KNN.
- [x] El dataset seleccionado es **global**: lo comparten las 4 lecciones y la página CSV.
- [x] Vista previa de los datos y estadísticas descriptivas (`describe`).
- [x] Distribución del objetivo: barras en clasificación, histograma en regresión.
- [x] Análisis avanzado: matriz de correlación (heatmap) y matriz de dispersión
      (pairplot), con diagonal histograma o KDE, nº máximo de características y
      multiselección.
- [x] Preprocesado: división train/test con `test_size` y `random_state=42`. La versión *(core: split.rs)*
      Python **no estratifica**; Rust lo reproduce exactamente (decisión D7).
- [x] Metadatos por dataset para el formulario de predicción: rangos, valores *(datos en content/datasets; formulario en F2)*
      categóricos y nombres legibles. 🐍 `dataset/dataset_metadata.py`

## 2. Página "📁 Cargar CSV Personalizado" 🐍 `app.py::run_csv_loader_app`

- [x] Expander de ayuda, carga de CSV y métricas (filas, columnas, objetivo, tipo de tarea).
- [x] Estadísticas descriptivas y distribución del objetivo, con barras e histograma y
      el mín/máx/media/mediana/desviación típica.
- [x] Ejemplos de datasets cuando todavía no hay ninguno cargado.

## 3. 🌲 Árboles de Decisión 🐍 `apps/decission_tree_app.py`

Pestañas: 📊 Datos · 🏋️ Entrenamiento · 📈 Evaluación · 📉 Visualización · 🔍 Características ·
🔮 Predicciones · 💾 Exportar

- [x] Entrenamiento: tipo de tarea (clasificación o 📈 regresión), criterio de división
      (gini/entropía; squared_error… en regresión), profundidad máxima, muestras mínimas
      para dividir y tamaño de test. ⚠️ Los árboles de regresión no los cubre linfa.
- [x] Evaluación de clasificación: accuracy, precision, recall, F1, informe por clase,
      matriz de confusión, muestras bien y mal clasificadas e interpretación.
      🐍 `algorithms/model_evaluation.py`
- [x] Evaluación de regresión: MSE, RMSE, MAE, R², predicciones frente a reales.
- [x] Visualización: 🌲 árbol gráfico (con ancho y alto ajustables) y 📝 texto de reglas.
      En clasificación, 🌈 frontera de decisión (eligiendo 2 características) y
      📉 ROC/PR. En regresión, 📈 residuos y 🌐 superficie de decisión.
      🐍 `viz/tree_visualizer.py`, `viz/decision_boundary.py`, `viz/roc.py`, `viz/residual.py`
- [x] Características: importancia nativa y por permutación (D38; C10, reutilizado en KNN), con valores
      detallados. 🐍 `viz/features.py`
- [x] Predicciones: formulario por característica guiado por los metadatos, resultado
      con probabilidades de la hoja y camino de decisión (D46). 🐍 `ui.py::create_prediction_interface`,
      `model_evaluation.py::show_prediction_path`
- [x] Exportar: ONNX (TreeEnsemble*), paquete Python ZIP (D43), 📝 reglas .txt, 🐍 código Python
      corregido (D44; el de Python no compila), imagen PNG/SVG del árbol (D42) y código de uso.

## 4. 📊 Regresión (lineal y logística) 🐍 `apps/linear_regression_app.py`

Pestañas: 📊 Datos · 🏋️ Entrenamiento · 📈 Evaluación · 📉 Visualización · 🔍 Coeficientes ·
🔮 Predicciones · 💾 Exportar

- [x] Entrenamiento: 📈 regresión lineal o logística, máximo de iteraciones (logística)
      y tamaño de test.
- [x] Evaluación: métricas de regresión o de clasificación, igual que en el apartado 3.
- [x] Visualización lineal: 📈 residuos (predicciones frente a reales y gráfico de
      residuos) y 🌐 superficie de predicción con un modelo reentrenado con 2 características.
- [x] Visualización logística: 📉 ROC (con curvas por clase y promediados macro y micro;
      D20), análisis de umbrales, 🌈 frontera de decisión y 📊 distribución de probabilidades.
- [x] Coeficientes: tabla y gráfico, con selección de clase en el caso multiclase,
      intercepto e interpretación textual.
- [x] Predicciones con formulario.
- [x] Exportar: código generado, ONNX y paquete Python ZIP (que reconstruye el modelo y genera el `.pkl`; D1, D28).

## 5. 🔍 K-Nearest Neighbors 🐍 `apps/knn_app.py`

Pestañas: 📊 Datos · 🏋️ Entrenamiento · 📈 Evaluación · 📉 Visualización · 🔍 Características ·
🔮 Predicciones · 💾 Exportar

- [x] Entrenamiento: vecinos (K), pesos (uniform/distance), métrica
      (minkowski/euclidean/manhattan) y tamaño de test. KNN propio con kd-tree exacto (D52)
- [x] Evaluación: clasificación o regresión.
- [x] Visualización de clasificación: 📊 distribución de probabilidades, 📉 ROC,
      🌈 frontera de decisión, 🎮 visualización interactiva nativa (D54) y 📏 análisis de distancias
      (separabilidad y distancias entre clases).
      🐍 `create_interactive_knn_visualization`, con un punto de prueba de valores
      editables y su animación HTML de vecinos.
- [x] Visualización de regresión: 📊 distribución de predicciones (residuos) y
      🌐 superficie de predicción.
- [x] Características: permutation importance (C10 de F3).
- [x] Predicciones con formulario, probabilidades y los K vecinos (D58).
- [x] Exportar: código de entrenamiento, ONNX propio y paquete Python ZIP (D57). Python solo tenía el
      pickle (`linear_regression_model.pkl`) y el fragmento de Regresión.

## 6. 🧠 Redes Neuronales 🐍 `apps/neural_network_app.py`, `viz/nn.py`, `viz/nn_extra.py`

Pestañas: 📊 Datos · 🏗️ Arquitectura · ⚙️ Entrenamiento · 📈 Evaluación · 🎯 Visualización ·
🔮 Predicciones · 💾 Exportar

- [x] Arquitectura: nº de capas ocultas, neuronas por capa, activación oculta
      (relu/tanh/sigmoid), activación de salida (sigmoid/softmax/linear/tanh, según la
      tarea), dropout y diagrama nativo con tooltip y nº de parámetros por capa (D80). Incluye
      el código Keras con descarga (script corregido, D74). 🐍 `create_neural_network_visualization`
- [x] Entrenamiento: tamaño de batch, optimizador (adam/sgd/rmsprop), learning rate,
      épocas y % de validación. Opciones avanzadas: parada temprana (paciencia) y
      reducción automática del LR (factor). Progreso por pasos (1/4…4/4) y cancelación.
      Muestra la distribución train/val/test. Split estratificado con test 0,2 fijo y
      escalado (D72); semilla (D70); salidas no entrenables bloqueadas (D71).
- [x] Evaluación: métricas principales, informe de clasificación, matriz de confusión y
      diagnóstico del entrenamiento (estado general, tendencias por fase, patrones, alertas
      y recomendaciones; D84). 🐍 `model_evaluation.py::neural_network_diagnostics` y siguientes
- [x] Visualización:
  - [x] 📊 Historial: pérdida y métricas por época, train frente a validación.
  - [x] 🧠 Pesos: histogramas de pesos y sesgos (bins, escala log) y resumen por capa.
  - [x] 🎯 Superficie de decisión: 2 características, el resto fijas en su media, en
        clasificación y en regresión.
  - [x] 📉 Activaciones: nº de muestras, umbral de neurona muerta, desviación típica
        constante, mapa de la red con opciones de dibujo (dispersión, % de conexiones,
        radio adaptativo, agrupación de capas, espaciado, subcolumnas), histogramas y
        medias por capa, y mapa de neuronas muertas 💀 y constantes 😴.
- [x] Predicciones: individual (formulario), por lotes y exploración (índice de muestra y
      característica). Por lotes y exploración son nuevas (D81): en Python eran inalcanzables.
- [x] Exportar: 📦 paquete Keras ZIP (genera .keras, .h5, SavedModel y .tflite), 💾 ONNX, 📝
      predictor NumPy + script y 📋 metadatos (D75).

## 7. Código Python generado 🐍 `algorithms/code_examples.py`, `utils.py`

- [ ] Para cada algoritmo: código de entrenamiento, predicción, frontera de decisión,
      exportación, arquitectura, evaluación y visualización de la red, y código completo.
      Debe reproducir los hiperparámetros elegidos en la app.

## 8. Funcionalidad que NO se porta (código muerto en la versión Python)

Estas funciones no se usan en la navegación actual. No se migran salvo que se decida
lo contrario: `show_network_anatomy_tab`, `show_enhanced_decision_surface_tab`,
`show_live_activations_tab`, `show_interpretability_tab`,
`show_interactive_simulator_tab` y sus auxiliares (`viz/nn_extra.py`);
`show_sidebar_config`, `display_tree_visualization`, `display_feature_importance`,
`display_model_export_options`, `export_model_pickle`, `export_modelo_onnx` (`ui.py`);
`viz/tree_visualization.py` entero; `tree_visualizer.create_tree_visualization` y
`render_tree_visualization`; `utils.generate_model_code`; `model_training.predict_sample`;
`DECISION_BOUNDARY_CODE`; `viz/nn_extra.py` entero; `show_neural_network_evaluation`,
`show_neural_network_visualizations`, `initialize_model_safely`,
`generate_neural_network_{evaluation,visualization,complete}_code` y el surrogate KNN de la superficie de
regresión; `create_animated_neighbors_plot`, la tabla y el respaldo de puntos de
prueba de `create_interactive_knn_visualization`, `knn_metrics` y `dataset_loaded` (`knn_app.py`).
Antes de descartar cada una, `migration-porter` debe confirmar que no se puede alcanzar.

## 9. Gráficos: de matplotlib a Plotly

Unas 40 llamadas a matplotlib/seaborn (`st.pyplot`) y unas 20 a Plotly. En Rust,
**todo pasa a Plotly** (interactivo) o a SVG propio, que es el caso del diagrama del
árbol y del de la red. Se conservan las paletas, los títulos y los ejes. El aspecto
debe ser igual o mejor que el de las capturas de referencia.
