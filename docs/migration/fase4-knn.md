# Fase 4 — 🔍 K-Nearest Neighbors

Plan de port de `migration-porter` (2026-10-15). Cubre INVENTARIO §5 (🔍 K-Nearest Neighbors), la parte de §0
(diálogos de exportación), §7 (código Python generado de KNN) y §9 de estas pantallas. Referencia: rama
`codex/desktop-migration-study`, la versión Qt. Plantilla: [fase3-arboles.md](fase3-arboles.md).

La lección **reutiliza** los componentes compartidos de F2/F3 (C1–C12, C10 en modo permutación) y añade
tres nuevos: el buscador de vecinos con paridad exacta (C17), el lienzo interactivo nativo (C18) y el
análisis de distancias entre clases (C19). Los vecinos de una predicción (C20) son una vista pequeña nueva.

**Hallazgo principal (D52):** en KNN no hay entrenamiento; la paridad consiste en elegir **los mismos
vecinos**. Con `algorithm='auto'`, scikit-learn usa un **kd-tree** (≤ 15 características y K < n_train // 2)
o **fuerza bruta**, y cada uno desempata de forma distinta cuando dos muestras están a la misma distancia del
K-ésimo vecino. Los empates son frecuentes con puntos duplicados (Titanic, `sex` de Diabetes, Iris en 2D,
las coordenadas de California): con la regla "gana el índice menor" (la de la fuerza bruta, y la que
MEMORY daba por buena) cambian hasta **5 340 de 11 655 celdas** de la superficie por defecto de Diabetes y
179 de 40 000 de la frontera de Iris. `tools/knn_reference.py` transcribe el kd-tree de sklearn 1.7.2
(construcción con `std::nth_element` de **libc++**, recorrido en profundidad, heap que rechaza `dist >=
mayor`, `simultaneous_sort`) y reproduce **bit a bit** los índices (en orden) y las distancias de los 19
casos y de todas las celdas sensibles a empates de sus rejillas. Rust debe seguirlo.

**El contrato IPC (§5) es la frontera.** `rust-ml-engineer` trabaja en `crates/mltutor-core` y
`crates/mltutor-export`; `tauri-ui-engineer`, en `src-tauri` y `ui`; `ml-lesson-author`, en `content/`.

Índice: [0 Capturas](#0-capturas-de-referencia-qt) · [1 Estructura](#1-estructura-de-la-lección) ·
[2 Pestañas](#2-pestañas) · [3 Componentes](#3-componentes-reutilizados-y-nuevos) ·
[4 Casos límite](#4-casos-límite-y-comportamientos-de-python) · [5 Contrato IPC](#5-contrato-ipc) ·
[6 Fixtures](#6-fixtures-de-paridad) · [7 Contenido](#7-contenido-extraído) ·
[8 Mapeo a Rust](#8-mapeo-a-rust) · [9 Reparto y checklist](#9-reparto-de-tareas-y-checklist) ·
[10 Decisiones](#10-decisiones-para-el-usuario) · [11 Riesgos](#11-riesgos) ·
[12 Hallazgos](#12-hallazgos-de-la-versión-python) · [13 Cambios propuestos](#13-cambios-propuestos-en-otros-documentos)

## 0. Capturas de referencia (Qt)

En `docs/migration/capturas/qt/fase4/`, generadas con

```sh
uv run python tools/capture_qt.py docs/migration/capturas/qt/fase4 --phase 4 --quantize
```

(ventana real, no `offscreen`; unos 77 s; 49 PNG, más el HTML y 2 PNG del lienzo; 5,4 MB ya cuantizados). Solo 02, 09 y 35 tienen
además `NNN_ventana.png`. Como en F2/F3, los expanders abiertos con `expand_all` siguen abiertos en las
capturas siguientes (por eso el expander de la cabecera aparece abierto).

**El lienzo de 🎮 Visualización Interactiva es una vista WebEngine y sale en blanco** en `09_…` y `30_…`
(MEMORY: `capture_qt` no captura WebEngine). Para tener su referencia visual se ha guardado el HTML que
genera Python (`09_interactiva_iris_lienzo.html`, ábrelo en un navegador) y dos capturas hechas con Chrome
headless: `09c_interactiva_iris_lienzo.png` (estado inicial, sin puntos) y
`09d_interactiva_iris_lienzo_con_puntos.png` (con tres puntos de prueba precargados en `existingTestPoints`,
para ver las líneas a los vecinos, el anillo de confianza y la etiqueta de clase).

| Archivo | Pantalla |
|---|---|
| `00_evaluacion_sin_entrenar`, `00b_visualizacion_sin_entrenar`, `00c_caracteristicas_sin_entrenar` | Pestañas sin modelo (`st.info`: "Primero entrena un modelo KNN." / "…para ver la importancia de características.") |
| `01_entrenamiento_iris`, `02_entrenado_iris(_ventana)` | 🏋️ Entrenamiento (K 5, uniform, minkowski, 0,3) antes y después; **la caja "Información del Dataset" desaparece tras entrenar** (D60) |
| `03_evaluacion_clasificacion_iris`, `04_…_expandido` | 📈 Evaluación (C1 + C3) |
| `05_probabilidades_multiclase_iris` | 📊 Distribución de Probabilidades (C5) con títulos "Probabilidades para Clase 0" (D61) |
| `06_roc_multiclase_iris` | 📉 Curva ROC (C4), leyendas "Clase 0 (AUC = …)" (D61) |
| `07_frontera_iris`, `08_frontera_petalos_iris` | 🌈 Frontera (C8) con **todos** los datos (D63) |
| `09_interactiva_iris(_ventana)`, `09c_…_lienzo`, `09d_…_lienzo_con_puntos`, `10_interactiva_iris_expandido`, `11_interactiva_misma_caracteristica` | 🎮 Visualización Interactiva: selects, lienzo (en blanco en Qt), `st.info` "💡 ¿Cómo usar…", guía y aviso de misma característica |
| `12_distancias_iris`, `13_…_expandido` | 📏 Análisis de Distancias: heatmap viridis, 3 métricas, aviso 🟡 |
| `14_caracteristicas_iris`, `15_…_expandido` | 🔍 Permutation importance (C10) |
| `16_prediccion_formulario_iris`, `17_prediccion_resultado_iris` | 🔮 Formulario (rangos de **todo** el dataset) y resultado: tarjeta + "Información del modelo KNN" |
| `18_exportar_iris`, `19_exportar_codigo`, `20_exportar_pickle` | 💾 Exportar: los dos botones de Regresión (F2 §2.7) |
| `21_entrenado_cancer` … `26_distancias_cancer` | Binaria: probabilidades + umbrales (C5 + C6), ROC + PR, frontera RdBu, distancias (ratio 1,00 → 🔴: **siempre** en binaria, D67) |
| `27_prediccion_formulario_titanic`, `28_prediccion_resultado_titanic` | Formulario con selects de metadatos |
| `29_entrenamiento_k20_distance_manhattan`, `30_interactiva_k20_…`, `31_frontera_k20_…` | K = 20 (el lienzo de Python limita su deslizador a 15; D55), pesos por distancia, manhattan |
| `32_entrenamiento_regresion_diabetes` … `41_exportar_regresion_diabetes` | Regresión: sin botones de tipo, evaluación (C2), 📊 Distribución de Predicciones (C7), 🌐 superficie age/sex (la más sensible a empates) y bmi/s5, importancia, predicción y exportar |
| `42_superficie_california`, `43_caracteristicas_california` | California: paso 0,1 (n > 1000) y permutación sobre 6 192 muestras |

No se ha capturado: los diálogos nativos, el spinner, el error de K > n_train (solo con un CSV pequeño; está
en el fixture `csv_string_target_k20_test05_error`) ni la tabla de puntos de prueba (inalcanzable, §12).

---

## 1. Estructura de la lección

`mltutor/apps/knn_app.py::run_knn_app`.

1. Cabecera: `st.header("🔍 K-Nearest Neighbors (KNN)")`, "Aprende sobre K-vecinos más cercanos de forma
   interactiva." y el expander "ℹ️ ¿Qué es K-Nearest Neighbors?" (cerrado; `content/knn/intro.md`). Ya está en
   F1 (`content/inicio/navegacion.toml [lessons.knn]`).
2. 7 pestañas (botones `tab_knn_0..6`, estado `active_tab_knn`, 0 por defecto) y `---`: 📊 Datos ·
   🏋️ Entrenamiento · 📈 Evaluación · 📉 Visualización · 🔍 Características · 🔮 Predicciones · 💾 Exportar. El
   CSS `tab-button-knn` no tiene efecto en Qt.
3. `run_dataset_tab(tab)` (F1), como en Regresión y Árboles.
4. Contenido de la pestaña (§2) y navbar (F1): (—, "Continuar a Entrenamiento"), ("Volver a Datos",
   "Continuar a Evaluación"), ("Volver a Entrenamiento", "Continuar a Visualización"), ("Volver a Evaluación",
   "Continuar a Características"), ("Volver a Visualización", "Continuar a Predicciones"), ("Volver a
   Características", "Continuar a Exportar"), ("Volver a Predicciones", —).

### 1.1 Estado de sesión (Python)

| Clave | Escribe | Contenido |
|---|---|---|
| `active_tab_knn` | pestañas, navbar | 0..6 |
| `dataset_option_knn` | Entrenamiento | último dataset cargado; si difiere de `selected_dataset` (o `knn_X` es None) se recarga |
| `knn_dataset`, `knn_X`, `knn_y`, `knn_feature_names`, `knn_class_names`, `knn_task_type` | al cargar (pestaña 1) | datos **completos** (DataFrame/Series con su índice), nombres, clases y **la tarea del dataset** (no hay botón de tipo) |
| `knn_n_neighbors`, `knn_weights`, `knn_metric` | controles con key | K, pesos, métrica |
| `knn_Xtrain`, `knn_Xtest`, `knn_ytrain`, `knn_ytest`, `knn_model`, `knn_trained` | "🚀 Entrenar Modelo KNN" | split y modelo; `knn_trained` **no se borra al cambiar de dataset** (D32) |
| `knn_metrics` | — | se inicializa y no se usa nunca |
| `viz_type` | `create_button_panel` | global (D13: por lección en Rust) |
| `viz_feature1`, `viz_feature2` | frontera/superficie | compartidos con Regresión y Árboles |
| `interactive_feature1`, `interactive_feature2` | 🎮 Interactiva | propios de KNN |
| `test_points` | se inicializa a `[]` | **nunca recibe puntos** (§12) |
| `feature_{i}` | formulario | compartidos con las otras lecciones |

Flujo de entrenamiento: `load_data(selected_dataset)` → `preprocess_data(knn_X, knn_y, test_size)` (D7, sin
estratificar) → `train_knn_model(Xtrain, ytrain, task_type=knn_task_type, n_neighbors, weights, metric)` =
`KNeighborsClassifier` o `KNeighborsRegressor(n_neighbors, weights, metric)` con el resto por defecto
(`algorithm='auto'`, `leaf_size=30`, `p=2`, `metric_params=None`, `n_jobs=None`) sobre los datos **sin
escalar** (no hay `StandardScaler` en ninguna parte de la lección; D62). `fit` no comprueba `K <= n_train`:
el error aparece al predecir (D59).

---

## 2. Pestañas

### 2.1 📊 Datos (pestaña 0)

Exactamente la de F1 (`DatasetTab.svelte`).

### 2.2 🏋️ Entrenamiento (pestaña 1)

Orden: selector de F1 → `st.header("Configuración del Modelo KNN")` → (solo en el render que carga el
dataset) `### Información del Dataset` + `create_info_box(info)` → `### Parámetros del Modelo` + "Configura
los hiperparámetros del modelo KNN:" → 4 columnas → botón → resultado → navbar. Si falla la carga:
`st.error("Error al cargar el dataset: {e}")`. La rama `st.info("Primero selecciona y carga un dataset en la
pestaña de Datos.")` es inalcanzable.

| Control | Tipo | Etiqueta exacta | Rango / opciones | Por defecto | key | Efecto |
|---|---|---|---|---|---|---|
| Vecinos | number_input entero (col. 1) | `Vecinos (K)` | 1 … 20, paso 1 | 5 | `knn_n_neighbors` | `n_neighbors` |
| Pesos | selectbox (col. 2) | `Pesos` | `uniform`, `distance` (sin traducir) | `uniform` | `knn_weights` | `weights` |
| Métrica | selectbox (col. 3) | `Métrica` | `minkowski`, `euclidean`, `manhattan` (sin traducir) | `minkowski` | `knn_metric` | `metric` (minkowski con p = 2 **es** euclidean) |
| Tamaño de test | slider real (col. 4) | `Tamaño del Conjunto de Prueba:` | 0,1 … 0,5, paso 0,05 | 0,3 | **sin key** | help "Proporción de datos que se usará para evaluar el modelo." |
| Entrenar | botón primary | `🚀 Entrenar Modelo KNN` | — | — | `train_knn_button` | Spinner "Entrenando modelo..."; guarda el estado de §1.1 |

Ningún control tiene `help` salvo el slider. Resultado: `st.success("¡Modelo KNN entrenado correctamente!")` +
`st.info("👉 Ve a la pestaña '📈 Evaluación' para ver los resultados del modelo.")` o
`st.error("Error al entrenar el modelo: {e}")` (no se llega a ver con datasets válidos: `fit` no falla).

### 2.3 📈 Evaluación (pestaña 2)

`st.header("📈 Evaluación del Modelo KNN")`. Sin modelo: `st.info("Primero entrena un modelo KNN.")` (igual
en 3, 5 y 6). Con modelo: `show_detailed_evaluation(knn_ytest, model.predict(knn_Xtest), knn_class_names si la
tarea es Clasificación si no None, knn_task_type)` → **C1** o **C2** + **C3** (literal "Tu modelo de regresión
lineal explica…", como en F3). La rama `st.error("No se encontraron los datos necesarios para la
evaluación.")` es inalcanzable.

### 2.4 📉 Visualización (pestaña 3)

`st.header("📉 Visualización de KNN")` + `### Opciones de Visualización` + panel (`create_button_panel`). Sin
modelo: `st.info("Primero entrena un modelo KNN.")`.

| Tipo | Botón (valor, key) | Contenido |
|---|---|---|
| clasificación | `📊 Distribución de Probabilidades` (Probabilidades, `viz_prob`) | `### 📊 Distribución de Probabilidades` + **C5** (+ **C6** en binaria) con `predict_proba(X_test)` y **`class_names=None`** (variable local sin asignar): "Clase 0 (Real)", "Probabilidades para Clase 0" (D61). Probabilidades discretas: múltiplos de 1/K con `uniform` |
| clasificación | `📉 Curva ROC` (ROC, `viz_roc`) | **C4** con `predict_proba(X_test)` **sin** `class_names` ("Clase 0 (AUC = …)"; D61), macro por defecto. Muchas puntuaciones empatadas: curvas con pocos escalones |
| clasificación | `🌈 Frontera de Decisión` (Frontera, `viz_boundary`) | **C8** modo frontera con `KNeighborsClassifier(n_neighbors, weights, metric)` del modelo reentrenado en 2 características de **todo el dataset** (`knn_X`, `knn_y`), no de train, y con los puntos de todo el dataset; el paso de la rejilla usa el tamaño de **todo** el dataset (`knn_step(n)`: Cáncer, 569 → 0,05; con train serían 398 → 0,02) (D63). Sin `try` propio: si falla, el `except` de `plot_decision_boundary` ("Error al crear la visualización de frontera de decisión: {e}") |
| clasificación | `🎮 Visualización Interactiva` (Interactiva, `viz_interactive`) | §2.4.1 (**C18**) |
| clasificación | `📏 Análisis de Distancias` (Distancias, `viz_distances`) | §2.4.2 (**C19**) |
| regresión | `📊 Distribución de Predicciones` (Predicciones, `viz_predictions`) | `### 📊 Gráfico de Predicciones vs Valores Reales` + **C7a**, `### 📈 Análisis de Residuos` + **C7b** (el botón dice "Distribución de Predicciones" pero son residuos) |
| regresión | `🌐 Superficie de Predicción` (Superficie, `viz_surface`) | **C8** modo superficie con `KNeighborsRegressor(n_neighbors, weights, metric)` sobre **todo** el dataset (D63). Sin `try` alrededor del `fit` |

Los avisos "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'." de Frontera y Superficie son
inalcanzables (la pestaña ya comprobó `knn_trained`). Navbar: ("Volver a Evaluación", "Continuar a
Características").

#### 2.4.1 🎮 Visualización Interactiva (`create_interactive_knn_visualization`)

1. `### Visualización Interactiva de KNN` + "Explora cómo el algoritmo KNN toma decisiones en tiempo real".
2. 2 columnas: `Primera característica:` (key `interactive_feature1`, índice 0) y `Segunda característica:`
   (key `interactive_feature2`, índice min(1, n−1)). Si son iguales: `st.warning("Por favor selecciona dos
   características diferentes.")` y nada más.
3. `X_2d = knn_X[:, par]` (**todas** las filas), `y = knn_y`. Un `components.html(html, height=800)` con un
   documento completo (34 KB: CSS, HTML y JS; Qt lo pinta en un QWebEngineView):
   - Caja azul: **"🎯 Visualización Interactiva KNN"** / "Haz click en cualquier parte del área gris para
     agregar puntos de prueba y ver cómo el algoritmo los clasifica."
   - Controles (fila centrada): `Valor K:` `<input type=range min=1 max=15 value=K>` + número; `Métrica:`
     select Euclidean/Manhattan/Minkowski (inicial = la del modelo); `Pesos:` Uniforme/Por distancia;
     `Mostrar confianza:` ☑; `Mostrar distancias:` ☑; botón rojo `🗑️ Limpiar Puntos`. **Con K > 15 el
     deslizador se queda en 15 pero `currentK` empieza valiendo el K del modelo (20)** (D55).
   - Lienzo 3:2 (alto mínimo 400 px), fondo degradado `#f5f7fa → #c3cfe2`, borde gris redondeado, margen
     interior 20 px. **Sin ejes, marcas ni nombres de las características.** Límites `min − 1`, `max + 1`.
   - Puntos de entrenamiento: círculos de radio `clamp(ancho/120, 4, 8)`, borde `#333`, color
     `colors[código % 8]` con `colors = #FF6B6B, #4ECDC4, #45B7D1, #FFA726, #AB47BC, #66BB6A, #EF5350,
     #42A5F5`. Los duplicados se pintan unos encima de otros.
   - Clic dentro del área → punto de prueba: círculo del color de la clase predicha (radio `clamp(ancho/80,
     6, 12)`) con una estrella dorada `#FFD700`; si "Mostrar distancias", líneas discontinuas rojas
     `#FF4444` (`[5, 5]`) a los K vecinos; si "Mostrar confianza", arco verde `#4CAF50` proporcional a la
     confianza y el texto "NN%" debajo; el nombre de la clase en negrita encima. **No hay animación**: el
     parámetro se llama `animate_distances` pero solo activa las líneas.
   - Leyenda "Leyenda:" con un círculo por clase y "Puntos de Prueba" (rombo dorado). Tarjetas: "Puntos de
     Prueba" (nº), "Confianza Promedio" (`Math.round(media·100)%`), "Valor K Actual".
   - **La clasificación la hace el JavaScript, no el modelo** (`classifyPoint`): distancias 2D en bruto
     (minkowski = euclídea), `sort` estable por distancia (empates: orden del dataset), los K primeros, votos 1
     o 1/d, **d = 0 → peso 1 y los demás siguen pesando 1/d** (sklearn: solo cuentan los de distancia 0), gana
     la primera clase (código ascendente) con puntuación estrictamente mayor, confianza = ganadora / total.
   Fixture `interactive.*.samples`: en 5 puntos de muestra por par, la clase del JS coincide con sklearn y
   el conjunto de vecinos difiere en los empates.
4. Debajo: `st.info` "💡 **¿Cómo usar la visualización?** …" (`interactiva-como-usar.md`), porque
   `session_state.test_points` está siempre vacío (§12), y el expander cerrado "📚 Guía de la Visualización
   Interactiva" (`interactiva-guia.md`). Un `st.markdown` con un `<script>` que escucha `postMessage` no hace
   nada (ni en Streamlit ni en Qt).
5. **Inalcanzable**: la tabla "### 📊 Análisis de Puntos de Prueba" (Punto, f1 `{:.3f}`, f2 `{:.3f}`,
   Predicción, Confianza `{:.1%}`, Distancia Promedio `{:.3f}`) con 3 métricas ("Total de Puntos", "Confianza
   Promedio", "Parámetros" = "K={k}, {metric}"), que además fallaría porque llama al modelo completo con 2
   columnas; y el respaldo del `except` ("Error en la visualización interactiva: {e}", "Fallback: Usando
   controles manuales", dos `number_input` "Valor {f}:" de min a max de los ejes con paso 0,1 y "➕ Agregar
   Punto"). Son los "valores editables" que menciona el inventario: en Python el estudiante nunca los ve.

**Sustituto (D54): `KnnPlayground.svelte`, componente nativo sin HTML incrustado** (§3 C18).

#### 2.4.2 📏 Análisis de Distancias

1. `### Análisis de Distancias entre Clases` + expander cerrado "ℹ️ Cómo interpretar este análisis"
   (`distancias-interpretar.md`, con `{metrica}` = `model.metric`).
2. Cálculo: **`train_test_split(knn_X, knn_y, test_size=0.3, random_state=42)` siempre**, ignore el slider
   (D56); `classes = unique(y_train)`; para cada par i ≠ j, `mean(pairwise_distances(X_i, X_j,
   metric=model.metric))`; la diagonal queda a 0. `'euclidean'` usa `euclidean_distances` (fórmula del producto
   escalar con BLAS), `'minkowski'` `scipy.cdist(p=2)`, `'manhattan'` sumas exactas: en los fixtures la
   diferencia con la distancia exacta es ≤ 1e-13 relativa.
3. Heatmap matplotlib 8 × 6 in: `imshow(viridis, aspect='auto')`, texto `{:.2f}` **blanco** fuera de la
   diagonal (poco contraste en las celdas amarillas), etiquetas = nombres de clase (`knn_class_names[c]` o
   "Clase {c}"), x rotadas 45°, título "Distancias Promedio entre Clases\n(Métrica: {metric})", colorbar
   "Distancia Promedio".
4. `### Análisis de Separabilidad`: 3 métricas "Distancia Mínima" (`min` de las celdas > 0, `{:.3f}`),
   "Distancia Máxima" (`{:.3f}`), "Ratio Max/Min" (`{:.2f}`) y un aviso: ratio > 5 `st.success("🟢 Las clases
   están bien separadas (ratio > 5)")`, > 2 `st.warning("🟡 Separación moderada entre clases (2 < ratio ≤
   5)")`, si no `st.error("🔴 Las clases están muy próximas (ratio ≤ 2)")`. Con una sola clase en train:
   `st.info("Se necesitan al menos 2 clases para el análisis de distancias.")`.

### 2.5 🔍 Características (pestaña 4)

`st.header("🔍 Importancia de Características")`. Sin modelo: `st.info("Primero entrena un modelo KNN para ver
la importancia de características.")`. Con modelo: **C10** (`display_feature_importance`). Como KNN no tiene
`feature_importances_`, Python muestra **solo la permutación**: `st.info("🔄 Calculando Permutation
Importance... Esto puede tomar unos segundos.")`, spinner, el texto del método, barras con `xerr = std`
(`capsize 3`; las barras de error quedan negras), tabla con "Desv. Estándar" y el expander con el texto de
permutación + consejos. Ya implementado en F3 (subvista "🔀 Importancia por permutación"); en KNN no hay panel:
la vista es directamente esa. Texto `[importancia] permutacion` corregido en F3 (ejemplo de Diabetes con árbol;
D-revisión: valorar un ejemplo de KNN). Coste en sklearn: California 0,9 s.

### 2.6 🔮 Predicciones (pestaña 5)

`st.header("🔮 Predicciones con Nuevos Datos")` + **C9** (`create_prediction_interface(model, feature_names,
class_names, task_type, knn_X, knn_dataset)`): los rangos, medias y opciones del formulario salen de **todo el
dataset** (`knn_X`), no de `X_train` (D63). Entrada en float32 (D31: Rust en f64). Clasificación: tarjeta
"Clase: {class_names[int(ŷ)]}"; regresión: "Valor: {ŷ:.4f}". Después, en ambos: `### Información del modelo
KNN` + `st.info("**Modelo KNN**\n- K: {k}\n- Pesos: {weights}\n- Métrica: {metric}")`. **Sin probabilidades
ni vecinos** (D58). Navbar: ("Volver a Características", "Continuar a Exportar").

### 2.7 💾 Exportar (pestaña 6)

`st.header("💾 Exportar Modelo")` + `display_model_export_options(model)` de `algorithms/export.py`, **la misma
función que Regresión** (fase2 §2.7): 2 columnas.

| Columna | Botón | key | Al pulsar |
|---|---|---|---|
| Izquierda | `📄 Generar Código` | `generate_code_lr` | `st.code(generate_regression_code())` (`regresion_codigo.py.j2`: guarda y carga `modelo_regresion_logistica.pkl` con un `model` sin definir y sin `import numpy`) + "📥 Descargar código" (`regression_code.py`) |
| Derecha | `📥 Descargar Modelo (Pickle)` | `download_pickle_lr` | "Descargar modelo.pkl" → `pickle.dumps(model)` como **`linear_regression_model.pkl`** |

**No hay ONNX** (el inventario dice "Pickle y ONNX": solo existe en `utils.export_model_onnx`, que usa
`--verify-models`). Rust (D57): "📄 Código de Entrenamiento", "💾 Exportar a ONNX" y "📦 Paquete Python
(.zip)". Navbar: ("Volver a Predicciones", —).

---

## 3. Componentes reutilizados y nuevos

| Comp. | Estado tras F3 | Uso en KNN | Qué falta |
|---|---|---|---|
| Trait `Predictor` | hecho | `Knn` lo implementa | `classes()` = códigos vistos en train; `refit` con K, pesos y métrica; `predict_proba` en clasificación |
| C1/C2 Evaluación, C3 Interpretación | hecho | igual | nada |
| C4 ROC/PR | hecho | probabilidades discretas | nada (D61: pasar los nombres de clase) |
| C5/C6 Probabilidades y umbrales | hecho (F2) | 📊 Distribución de Probabilidades | D61 |
| C7 Residuos | hecho | 📊 Distribución de Predicciones | nada |
| C8 Frontera/superficie | hecho (`grid::knn_step` ya existe) | 2D KNN sobre **todo** el dataset | el comando debe pasar `X`/`y` completos y `knn_step(n_total)` (hoy el comentario dice `n_train`); la evaluación de 40 000 celdas necesita el kd-tree (C17) |
| C9 Formulario | hecho | rangos de todo el dataset (D63) | `PredictionResult::{Class,Value}` + `neighbors` (C20) y `proba` (D58) |
| C10 Importancia | hecho (F3, `importance::permutation` genérica) | solo permutación | nada en core; la vista sin panel |
| C11 Exportación | hecho (lineal, logística, árbol) | ONNX propio + ZIP | rama KNN en `onnx_model`, `package` (D57) |
| C12 Codegen | hecho | `knn_entrenamiento.py.j2`, `paquete_*_knn` | contextos tipados; variante "todos los datos" de `frontera_decision.py.j2` (D63) |
| **C17** Buscador de vecinos | — | todo | **nuevo** (core) |
| **C18** Lienzo interactivo | — | 🎮 Interactiva | **nuevo** (core + Svelte) |
| **C19** Distancias entre clases | — | 📏 | **nuevo** (core + Svelte) |
| **C20** Vecinos de una predicción | — | 🔮 | **nuevo** (DTO + Svelte) |

UI que se reutiliza: `ClassificationEvaluation`, `RegressionEvaluation`, `Interpretation`, `MetricGrid`,
`RocPrCharts`, `ProbabilityDistribution`, `ThresholdAnalysis`, `ResidualsAnalysis`, `DecisionGridView` +
`FeaturePairSelector`, `FeatureImportanceChart`, `PredictionForm`, `PredictionResultCard`, `ExportPanel`,
`ModelCode`, `modelSession.svelte.ts`, `viewLoader.svelte.ts`, y de `lessons/trees/` como modelo `TreeBanner`
(D32/D45), `TrainTab`, `VisualizationTab`, `FeaturesTab`, `ExportTab`.

### C17. Buscador de vecinos (`mltutor-core::knn`)

Especificación ejecutable: **`tools/knn_reference.py`** (D66). Reglas (sklearn 1.7.2):

1. **Motor** (`KNeighborsMixin._fit`, `algorithm='auto'`): `brute` si `n_features > 15` o `K >= n_train // 2`;
   si no, `kd_tree`. `effective_metric`: `minkowski` (p = 2) y `euclidean` → euclídea; `manhattan` → L1. En
   los fixtures: Cáncer (30 características) y los CSV con K grande son `brute`; el resto, `kd_tree` (también
   todos los modelos 2D salvo el CSV de 30 filas con K ≥ 15).
2. **Distancia reducida** (`rdist`): euclídea = `Σ (x_j − y_j)²` en orden de característica; L1 = `Σ |x_j −
   y_j|`; `dist = sqrt(rdist)` o `rdist`. La wheel arm64 de macOS compila la suma euclídea **desenrollada de 8 en
   8 sin FMA y el resto (`n_features % 8`) con FMA** (verificado en 300 vectores × 11 dimensiones). D53:
   reproducirlo con `f64::mul_add` (correctamente redondeado en las 3 plataformas) da distancias bit a bit; con
   la suma simple hay diferencias de 1–4 ulp y un conjunto de vecinos distinto en una celda de la rejilla de
   Iris.
3. **kd-tree** (`_binary_tree.pxi.tp`, `_kd_tree.pyx.tp`), `leaf_size = 30`:
   - `n_levels = int(log2(max(1, (n − 1)/30)) + 1)`, `n_nodes = 2^n_levels − 1`, `idx_array = 0..n`.
   - Nodo `i` (rango `[start, end)`): límites = min/max por característica de sus puntos; hoja si `2i + 1 >=
     n_nodes`; si no, dimensión = la **primera** con mayor `max − min` (estricto, empezando en 0 con spread 0),
     `n_mid = (end − start)/2` y `std::nth_element(idx + start, idx + start + n_mid, idx + end, comp)` con `comp(a,
     b) = (x[a, d], a) < (x[b, d], b)`; hijos `2i + 1` = `[start, start + n_mid)` y `2i + 2` = el resto.
   - **`nth_element` de libc++** (macOS): selección con `__sort3` (mediana de 3), ordenación por selección si
     `len <= 7`, partición de Hoare con guarda y comprobación de "ya ordenado" si no hubo intercambios
     (`libcxx_nth_element` en la referencia: verificado frente al `idx_array` real de sklearn en todos los
     casos). El **conjunto** de cada nodo no depende de la implementación (el orden es total), pero el orden
     **dentro de las hojas** sí, y es el que decide los empates. Una wheel de Linux (libstdc++) o Windows
     (MSVC) probablemente da otro orden: sklearn no es reproducible entre plataformas en los empates; Rust lo
     será (siempre el de macOS, que es el de los fixtures).
   - Consulta en profundidad: `min_rdist(nodo) = Σ (0,5·d)^p` con `d = (lo − x + |lo − x|) + (x − hi + |x −
     hi|)` (`pow(·, 2)` o `pow(·, 1)`); se poda si `min_rdist > mayor del heap` (estricto); en un nodo interno
     se visita primero el hijo con menor `min_rdist` (el izquierdo si empatan); en una hoja se empujan sus
     puntos en el orden de `idx_array`.
   - Heap de tamaño K (`_heap.pyx::heap_push`): max-heap inicializado a +∞; **rechaza `dist >= mayor`** (el
     primero visitado gana los empates) y hunde el nuevo valor con la regla exacta de hijos (`>=` entre
     hermanos).
   - Salida: `simultaneous_sort` (`_sorting.pyx`, quicksort con mediana de 3 de `[0, n/2, n−1]`) sobre las K
     distancias reducidas; después `sqrt`. El orden de los empates en la salida es el que deja este quicksort.
4. **Fuerza bruta**: Rust recorre las filas en orden de índice con el mismo heap (gana el índice menor). sklearn
   ordena la euclídea con el sustituto `||y||² − 2x·y` de BLAS (`EuclideanArgKmin`, y devuelve `sqrt` de ese
   sustituto): en Cáncer las distancias difieren hasta 4,3e-11 y los vecinos coinciden; en Dígitos (no está en
   los fixtures) cambian 3 de 540 filas. La L1 por fuerza bruta es exacta.
5. **Predicción** (`_classification.py`, `_regression.py`, `_base.py::_get_weights`):
   - Pesos: `uniform` → 1; `distance` → `1/d`, y si alguna distancia de la fila es 0 (o `1/d` es infinito),
     **solo** los vecinos con `1/d` infinito pesan 1 y el resto 0.
   - `predict_proba`: votos sumados **en el orden de los vecinos** en un vector por clase y divididos por su
     suma.
   - `predict` (clasificación): `uniform` → moda (la clase menor entre las más votadas); `distance` →
     `weighted_mode` (recorre las clases en orden; `>` estricto: la menor gana los empates exactos). En `brute`
     + `uniform`, sklearn usa `argmax(predict_proba)`: misma regla.
   - Regresión: `uniform` → media de los K valores; `distance` → `Σ w·y / Σ w` (`np.average`). Las sumas de
     numpy son por pares (8 acumuladores a partir de 8 elementos): diferencias de 1 ulp, tolerancia 1e-12.
6. **Errores**: `K > n_train` → `ModelError::TooFewSamples` (D59); NaN → el error común de F3.
7. **Rendimiento**: sklearn tarda 11 ms en predecir las 6 192 filas de prueba de California y 0,9 s en la
   permutación (80 predicciones del test). Presupuesto Rust: construir < 10 ms, consultar < 5 µs/fila en 8D, la
   rejilla de California (33 200 celdas, 20 640 puntos) < 60 ms; la permutación de California < 1 s con progreso
   y cancelación por columna (D-riesgo: paralelizar por filas con `std::thread::scope` si no llega).

API propuesta:

```rust
pub enum KnnWeights { Uniform, Distance }
pub enum KnnMetric { Minkowski, Euclidean, Manhattan }           // Minkowski ≡ Euclidean (p = 2)
pub enum SearchEngine { KdTree, Brute }
pub struct KnnConfig { pub n_neighbors: u32, pub weights: KnnWeights, pub metric: KnnMetric }
pub struct Knn { config, engine: SearchEngine, x: Array2<f64>, y: Target, classes: Option<Vec<usize>>, tree: Option<KdTree> }
pub fn fit(config: KnnConfig, x: ArrayView2<f64>, y: &Target, ctl: &TrainControl) -> Result<Knn, ModelError>;
impl Knn {
    pub fn kneighbors(&self, x: ArrayView2<f64>, ctl: &TrainControl) -> Result<Neighbors, ModelError>; // dist + idx (n × K)
    pub fn engine(&self) -> SearchEngine;
}
impl Predictor for Knn { … }
pub struct KdTree { idx_array: Vec<usize>, nodes: Vec<KdNode>, lower: Array2<f64>, upper: Array2<f64> }  // expone idx_array para el test
```

### C18. Lienzo interactivo (`KnnPlayground.svelte`, D54)

- **Sin HTML incrustado ni JS propio de cálculo**: Plotly (`scattergl`) dentro de la página, como el resto de
  gráficos. Mantiene todo lo que se ve en Python y le añade lo que Python prometía y no llegaba a mostrar.
- **Datos**: los de la 🌈 Frontera (todas las filas, el par elegido); el modelo 2D es `refit` del KNN con el K,
  los pesos y la métrica **de los controles del lienzo** (iniciales = los del modelo entrenado). Lo calcula
  Rust (comando `knn_playground`), con las reglas de sklearn (C17) en lugar de las del JS (§2.4.1).
- **Controles** (fila como en Python): `Valor K:` deslizador **1–20** (D55), `Métrica:` (Euclidean/Manhattan/
  Minkowski), `Pesos:` (Uniforme/Por distancia), `Mostrar confianza:`, `Mostrar distancias:`, `🗑️ Limpiar
  Puntos`. Y dos selects de características encima (los de Python, keys `interactive_feature*`).
- **Gráfico**: fondo con el mismo degradado, puntos con la paleta de Python (`colors[código % 8]`), radio 4–8
  px, borde `#333`. **Mejoras de vistosidad (D54)**: ejes con los nombres de las características (Python no
  tenía) y jitter nulo; los duplicados se indican con el tamaño del punto o el tooltip "×n".
- **Puntos de prueba**: clic en el gráfico (coordenadas de datos de Plotly), **arrastre** para moverlos, y
  "✏️ Añadir un punto con valores exactos" (los dos `number_input` "Valor {f}:" del respaldo de Python, rango
  `min − 1 … max + 1`, paso 0,1, valor inicial el centro; botón "➕ Agregar Punto"). Cada punto: círculo del
  color de su clase con estrella dorada, etiqueta de clase encima, anillo verde de confianza (`max proba`) y
  "NN%".
- **Animación de vecinos** (lo que Python llamaba `animate_distances`): al añadir o mover un punto, las líneas
  discontinuas `#FF4444` a sus K vecinos aparecen **una a una, del más cercano al más lejano** (≈ 60 ms cada
  una, `requestAnimationFrame`; respeta `prefers-reduced-motion`), y los vecinos se resaltan con un borde
  grueso. Tooltip de vecino: "Vecino {orden}: {clase}, distancia {d:.3f}".
- **Estadísticas**: las tres tarjetas de Python (Puntos de Prueba, Confianza Promedio, Valor K Actual) y, debajo,
  la tabla "### 📊 Análisis de Puntos de Prueba" (Punto, f1, f2, Predicción, Confianza, Distancia Promedio) con
  las 3 métricas de Python, ahora alcanzables y calculadas con el modelo 2D (en Python habrían usado el modelo
  completo con 2 columnas y fallado). Los valores de la tabla son editables (equivale a mover el punto).
- Textos: `content/knn/mensajes.toml [interactiva]`, `interactiva-como-usar.md` (se muestra mientras no haya
  puntos) e `interactiva-guia.md` (expander). `nota_modelo_2d` explica que el lienzo no es el modelo completo.
- Estado (por lección, D13): par, K/métrica/pesos del lienzo, toggles y la lista de puntos (se conserva al
  cambiar de subvista; se vacía al reentrenar o cambiar de dataset).

### C19. Distancias entre clases (`mltutor-core::knn::class_distances`)

- `class_distances(x_train, y_train, metric) -> ClassDistances { classes, matrix (k × k, diagonal 0), min, max,
  ratio, verdict: Separated|Moderate|Close, n_samples }`: media de las distancias exactas entre cada par de
  clases (suma en f64 y división por `n_i·n_j`), `min` de las celdas > 0, `ratio = max/min` (0 si no hay
  mínimo), umbrales > 5 y > 2. Coste: Dígitos 90 pares × ~16 000 distancias; < 20 ms.
- Datos: D56 (el split del modelo; Python usa siempre 0,3, que coincide con el valor por defecto).
- D67: además `within: Vec<f64>` (distancia media entre pares distintos de una misma clase, `Σ_{i<j} d /
  (n(n−1)/2)`) para la diagonal, y `verdict = Binary` con 2 clases (texto `[distancias_nuevo] binaria`). El
  `min`/`max`/`ratio` de Python siguen calculándose solo con las celdas fuera de la diagonal (paridad de las
  métricas).
- Vista `ClassDistanceHeatmap.svelte`: Plotly `heatmap` viridis con colorbar "Distancia Promedio", texto
  `{:.2f}` fuera de la diagonal (blanco o negro según la luminosidad de la celda, D-vistosidad), etiquetas de
  clase con x a 45°, título de dos líneas; las 3 métricas con `MetricGrid` y el aviso con `Notice`.

### C20. Vecinos de una predicción (D58)

- `predict_one` devuelve, además de la clase o el valor, `neighbors: Vec<{ train_row, distance, weight,
  class_index|value, features }>` (en el orden de C17) y, en clasificación, `proba` (C9 `ProbabilityTable`).
- Vista `KnnNeighborsTable.svelte`: "#### 👥 Los {k} vecinos más cercanos" (Vecino, Distancia `{:.4f}`,
  Clase/Valor, Peso, y las características), con la nota de distancia 0 cuando aplique. Se mantiene el
  `st.info` "Información del modelo KNN".

---

## 4. Casos límite y comportamientos de Python

| Caso | Python | Rust (propuesta) |
|---|---|---|
| Empate a la distancia del K-ésimo vecino | kd_tree: gana el primero que visita el recorrido; brute: el índice menor (euclídea: con el redondeo de BLAS) | D52: kd-tree portado; brute por índice |
| Puntos duplicados con clases distintas (Iris 2D, Titanic) | Deciden los empates; hasta el 46 % de las celdas de una rejilla | Igual que sklearn (D52) |
| K > n_train (CSV de 30 filas, test 0,5, K = 20) | Entrena ("¡Modelo KNN entrenado correctamente!") y fallan Evaluación, Visualización, Características y Predicciones en inglés ("Expected n_neighbors <= n_samples_fit…") | D59: error en español al entrenar |
| K ≥ n_train // 2 | Cambia a fuerza bruta (otro desempate) | Igual (regla del motor) |
| Muestra a distancia 0 de una de entrenamiento (`distance`) | Solo cuentan los vecinos a distancia 0 | Igual; C20 lo explica |
| Muestra a distancia 0 en el JS del lienzo | Peso 1 para esa y `1/d` para las demás | D54: regla de sklearn |
| K > 15 en el lienzo | El deslizador marca 15 y se usa 20 | D55: 1–20 |
| Cambiar K/pesos/métrica sin reentrenar | Frontera y superficie usan el **modelo** (`model.n_neighbors`…) | Igual (D45) |
| Cambiar el slider de test sin reentrenar | 📏 usa siempre 0,3 | D56: el split del modelo |
| Datos de la frontera, la superficie, el lienzo y el formulario | Todo el dataset (no train) | D63: igual que Python |
| Probabilidades / ROC | Títulos con "Clase 0" | D61: nombres de clase |
| Info del dataset en Entrenamiento | Solo en el primer render tras cargar | D60: siempre |
| 📏 en binaria | Ratio 1,00 y 🔴 siempre | D67 |
| Escalado | No hay (KNN sobre Proline de Vino, 278–1680, domina la distancia) | D62: igual que Python |
| Misma característica en los 2 selects | Frontera/superficie: nada; lienzo: `st.warning` | Igual + aviso de F2 en frontera |
| Una clase ausente del train | `classes_` solo con las vistas | `Predictor::classes()` |
| Formulario en float32 | Distancias con la entrada redondeada a f32 | D31: f64 (`prediction_form.*` guarda ambos) |

---

## 5. Contrato IPC

### 5.1 Convenciones

Las de F1–F3 (§5.1 de F2): comandos `async` + `spawn_blocking`, `build.rs` y `capabilities/default.json`, ts-rs,
`CommandError` en español, `model_id` (`ModelStale`), `seq` en `train_model`, evento `task-progress`. La
cancelación se comprueba entre bloques de 256 filas en `kneighbors`, entre filas de la rejilla y entre columnas
de la permutación.

### 5.2 Estado

Igual que F2/F3 (`LessonModels`, `TrainedModel`, `LessonId::Knn`, que ya existe). `EvaluationCache` añade
`test_neighbors` (para no recalcular). El estado de interfaz (pestaña, K, pesos, métrica, `test_size`, subvista
D13, pares de características de la frontera y del lienzo, controles y puntos del lienzo, valores del
formulario) vive en el store de Svelte de la lección (`lessons/knn/state.svelte.ts`).

### 5.3 Tipos (nuevos o ampliados)

```rust
#[serde(rename_all = "snake_case")] pub enum KnnWeights { Uniform, Distance }
#[serde(rename_all = "snake_case")] pub enum KnnMetric { Minkowski, Euclidean, Manhattan }
#[serde(rename_all = "snake_case")] pub enum SearchEngine { KdTree, Brute }
pub enum ModelConfig {
    Linear, Logistic { max_iter: u32 }, Tree { … },
    Knn { n_neighbors: u8 /* 1..=20 */, weights: KnnWeights, metric: KnnMetric },  // tarea = la del dataset
}
pub struct KnnSummary { pub n_neighbors: u32, pub weights: KnnWeights, pub metric: KnnMetric,
                        pub engine: SearchEngine, pub n_train: u32 }
// TrainedModelInfo añade `knn: Option<KnnSummary>`. CommandError añade TooFewSamples { k, n_train } (D59).

pub struct NeighborRow { pub train_row: u32 /* posición en X_train */, pub dataset_row: u32,
                         pub distance: f64, pub weight: f64,
                         pub class_index: Option<u32>, pub value: Option<f64> }
// PredictionResult::{Class, Value} añaden `neighbors: Option<Vec<NeighborRow>>` (C20, D58); Class.proba = predict_proba.

pub struct ClassDistancesView { pub classes: Vec<u32>, pub class_names: Vec<String>,
                                pub matrix: Vec<Vec<f64>>, pub min: Option<f64>, pub max: Option<f64>,
                                pub ratio: f64, pub verdict: SeparationVerdict /* separated|moderate|close */,
                                pub metric: KnnMetric, pub n_samples: u32 }

pub struct PlaygroundRequest { pub model_id: u32, pub feature_x: u32, pub feature_y: u32,
                               pub n_neighbors: u8, pub weights: KnnWeights, pub metric: KnnMetric,
                               pub points: Vec<[f64; 2]> /* puntos de prueba */ }
pub struct PlaygroundPoint { pub point: [f64; 2], pub class_index: u32, pub confidence: f64 /* max proba */,
                             pub proba: Vec<f64>, pub neighbors: Vec<NeighborRow>, pub mean_distance: f64 }
pub struct PlaygroundView { pub data: Matrix /* todas las filas, 2 col. */, pub classes: Vec<u32>,
                            pub class_names: Vec<String>, pub colors: Vec<String> /* paleta JS */,
                            pub bounds: [f64; 4], pub points: Vec<PlaygroundPoint>,
                            pub mean_confidence: Option<f64>, pub engine: SearchEngine }

pub enum ExportFormat { Onnx, PythonPackage, TreeRules, TreePython }        // KNN usa Onnx y PythonPackage
pub enum CodeTemplate { /* F2–F3… */ KnnTraining }
```

### 5.4 Comandos

| Comando | Entrada | Salida | Errores | Notas |
|---|---|---|---|---|
| `train_model` (F2) | `TrainRequest` con `ModelConfig::Knn` | `TrainedModelInfo` (+ `knn`) | `TooFewSamples` (D59), `NonFinite`, `Cancelled` | Construye el kd-tree; California < 10 ms |
| `lesson_model`, `evaluate_model`, `roc_curves`, `probability_distribution`, `threshold_analysis`, `residuals_view`, `prediction_form` (F2) | — | — | — | Sin cambios; `prediction_form` usa **todas** las filas en KNN (D63) |
| `decision_grid` (F2) | `GridRequest` | `DecisionGrid` | como F2 | En KNN `refit` sobre **todas** las filas y `knn_step(n_total)` (D63); usa el kd-tree |
| `feature_importance(model_id, Permutation, task_id)` (F3) | — | `FeatureImportanceView` | `Cancelled` | Con progreso por columna |
| `class_distances(model_id)` | `u32` | `ClassDistancesView` | `ModelStale`, `NotClassifier` | C19 |
| `knn_playground(req)` | `PlaygroundRequest` | `PlaygroundView` | `ModelStale`, `InvalidSelection` (x = y), `TooFewSamples` | C18; < 10 ms con 20 puntos; la UI lo llama al cambiar el par, un control o los puntos (debounce 50 ms); `data` solo cuando cambia el par |
| `predict_one` (F2) | `PredictRequest` | `PredictionResult` (+ `neighbors`, `proba`) | — | C20 |
| `generate_code(model_id, KnnTraining, extra)` (F2) | — | `String` | — | `knn_entrenamiento.py.j2` |
| `export_model(model_id, format)` (F2) | `Onnx`, `PythonPackage` | `Option<String>` | `Export`, `FileSave` | `knn.onnx`, `modelo_knn.zip` |
| `save_file` (F1) | — | — | — | PNG de cada gráfico, `.py` |

Presupuesto: cada vista < 100 ms con California, salvo la permutación (≈ 1 s, con progreso).

---

## 6. Fixtures de paridad

`uv run python tools/export_golden.py --only knn --only code/knn_training` y `--check`. Código en
`tools/golden_knn.py` (registrado desde `export_golden.py`; reutiliza sus helpers y los de `golden_trees.py`) y
`tools/knn_reference.py`. Listas de escalares en una línea (`dumps_compact`).

| Ruta | Caso | Qué ejercita |
|---|---|---|
| `knn/iris_k5_uniform_minkowski.json` | caso completo | kd_tree, 2 empates en el test, frontera Set3 (2 pares + misma característica), lienzo, distancias, ROC multiclase, formulario |
| `knn/iris_k1_uniform_manhattan.json`, `knn/iris_k5_distance_euclidean.json` | K = 1 L1; pesos por distancia | distancias L1 exactas; peso 1/d |
| `knn/iris_k20_distance_manhattan.json` | K = 20 > 15 del lienzo | 3 filas con empates en el test; lienzo con K = 20 |
| `knn/iris_k5_uniform_minkowski_test02.json` | `test_size = 0,2` | D56: 📏 con 0,3 (Python) y con el split del modelo |
| `knn/wine_k5_uniform_minkowski.json`, `knn/wine_k7_distance_manhattan.json` | 13 características | FMA en las 5 últimas (D53), escala dominada por Proline |
| `knn/breast_cancer_k5_uniform_minkowski.json`, `…_k10_distance_euclidean.json` | 30 características | **brute** (sustituto BLAS), ROC + PR binaria, umbrales, frontera RdBu |
| `knn/titanic_k5_uniform_minkowski.json` | categóricas | 55 filas con empates, formulario con selects |
| `knn/csv_string_target_k5_uniform_minkowski.json`, `…_k15_distance_euclidean.json` | CSV de 30 filas | kd_tree y brute (15 ≥ 21 // 2), modelo 2D brute |
| `knn/csv_string_target_k20_test05_error.json` | K = 20 con 15 de train | D59: lo que hace Python (todas las vistas fallan) |
| `knn/diabetes_k5_uniform_minkowski.json` | regresión completa | residuos, superficie age/sex (5 340 celdas sensibles) y bmi/s5 |
| `knn/diabetes_k10_distance_manhattan.json` | L1 con pesos | `np.average` |
| `knn/california_housing_k5_uniform_minkowski.json`, `…_k8_distance_euclidean.json` | 14 448 de train | tamaño, paso 0,1, permutación, rejilla 200 × 166 |
| `knn/csv_numeric_regression_k5_…`, `…_k20_distance_manhattan.json` | CSV de 40 filas | kd_tree y brute |
| `code/knn_training.json` | 19 renders del script de entrenamiento (17 casos sin Titanic ni el error + 2 con la muestra del formulario) | ejecutado y comparado con los fixtures |

Contenido de cada caso (`expected`): `params` (`get_params()`), `classes`, `train_indices`/`test_indices`,
`y_test`, `y_pred`, `proba`, **`neighbours_test`** (motor, métrica efectiva, índices en orden y distancias de
cada fila de prueba, `ties` con las filas empatadas en el K-ésimo vecino, los casi-empates relativos < 1e-9 y
las filas donde "gana el índice menor" daría otros vecinos, y `reference_impl`: comparación con
`knn_reference.py`, más `kd_tree` con `idx_start`/`idx_end`/`is_leaf` de cada nodo y el `idx_array` completo o
su hash si hay más de 1 000 filas), `evaluation` (como F2), `distance_analysis` (matriz de la app, matriz con
distancias exactas, métricas mostradas, veredicto y, con `test_size ≠ 0,3`, `with_model_split_D56`),
`permutation_importance` (semilla, baseline, matriz 10 × n, media, std; como F3), **`two_feature_models`** (por
par: motor, paso, ejes `x`/`y`, `z` completa, celdas sensibles al desempate y la comprobación de la referencia
en esas celdas más 1 500 al azar), `interactive` (paleta, límites, controles iniciales, reglas del JS y 5
puntos de muestra con el resultado del JS y el de sklearn), `onnx_reference` (skl2onnx: operadores, tamaño y
discrepancias), `onnx_proposal_D57` (el grafo propuesto, construido con `onnx.helper` y ejecutado con
`onnx.reference`), `export_package` (D1 ejecutado de verdad), `prediction_form` (controles, entrada f32,
predicción, probabilidades y vecinos) y `views` (todas las pestañas renderizadas sin Qt; el HTML del lienzo
resumido: tamaño, hash, datos, controles iniciales y textos visibles).

### 6.1 Cómo desempata sklearn (y por qué la paridad puede ser exacta)

Ver C17. Medido (19 casos): la referencia reproduce **en orden y bit a bit** los vecinos de todas las filas de
prueba en los casos `kd_tree` y el `idx_array` de cada árbol; en las rejillas, todas las celdas sensibles al
desempate (de 0 a 5 340 por rejilla) más 1 500 al azar. Con fuerza bruta euclídea (Cáncer, CSV con K grande)
coinciden los conjuntos de vecinos y las distancias difieren ≤ 4,3e-11 por el sustituto de BLAS. La resumen por
caso está en §6.3.

### 6.2 Tolerancias (`meta.tolerance` de cada caso; D6 → D52)

| Qué | Tolerancia | Por qué |
|---|---|---|
| Vecinos `kd_tree` (índices, en orden) | **exactos** | Mismo árbol, recorrido, heap y quicksort (D52) |
| Distancias `kd_tree` | **bit a bit** con D53; 4 ulp sin ella | FMA de la wheel arm64 |
| Vecinos `brute` | mismo conjunto, salvo filas en `ties.rows_with_near_tie_at_kth_rel_1e-9` | sklearn ordena con el sustituto de BLAS |
| Distancias `brute` euclídeas | 1e-9 relativo | sustituto `||y||² − 2x·y` |
| Predicciones | exactas (clases) / 1e-12 relativo (regresión) | sumas de numpy por pares |
| Probabilidades | 1e-15 absoluto | votos en el orden de los vecinos |
| Métricas, curvas ROC/PR | exactas (1e-12) | predicciones exactas |
| Permutation importance | 1e-12 absoluto | RNG reproducido (F3) |
| Análisis de distancias | 1e-10 relativo | BLAS/cdist frente a la suma exacta |
| Rejilla | ejes exactos; Z exacta (clases) / 1e-12 (valores) | el modelo 2D también es exacto |
| Textos | exactos | literales |
| ONNX (D57) | etiquetas exactas salvo empates en el K-ésimo vecino; probabilidades y valores float32 (1e-6 relativo) | `TopK` desempata por índice menor; entrada float32 |

Si Rust no reproduce los vecinos de un caso `kd_tree`, el test debe fallar y el caso anotarse en MEMORY.md: no
se relaja a "misma accuracy".

### 6.3 Resumen de los fixtures

19 casos + `code/knn_training.json`; 12 MB en total (California, 2,0–2,3 MB cada uno, por la rejilla de 33 200
valores). Generación: unos 8 min (la referencia en Python puro sobre las celdas sensibles es lo más lento).
Columnas: motor del modelo; ¿reproduce `knn_reference.py` los vecinos del test? (`sí` = índices en orden y
distancias bit a bit; `conjuntos` = índices en orden, distancias ≤ 4,3e-11 por BLAS); filas del test con
empate en el K-ésimo vecino / con casi-empate (< 1e-9) / donde "gana el índice menor" daría otros vecinos;
modelos 2D (motor, rejilla, celdas cuyos vecinos / cuya predicción cambiarían con "gana el índice menor", y si la
referencia reproduce esas celdas y 1 500 más); discrepancias del ONNX propuesto (D57) frente a sklearn
(etiquetas, o valores con error relativo > 1e-6); las de skl2onnx con entrada double (etiquetas, o error
absoluto máximo en regresión); tamaño en KB.

| Caso | Motor | Vecinos test | Empates test | Modelos 2D | ONNX D57 | skl2onnx | KB |
|---|---|---|---|---|---|---|---|
| `breast_cancer_k10_distance_euclidean` | brute | conjuntos | 0 / 0 / 0 | default: kd_tree 200×200, 0/0 ok | 0 | 32 | 558 |
| `breast_cancer_k5_uniform_minkowski` | brute | conjuntos | 0 / 0 / 0 | default: kd_tree 200×200, 0/0 ok; pair: kd_tree 46×603, 0/0 ok | 0 | 0 | 724 |
| `california_housing_k5_uniform_minkowski` | kd_tree | sí | 0 / 0 / 0 | default: kd_tree 200×166, 930/114 ok; pair: kd_tree 121×115, 355/354 ok | 0 | 0.0 | 2053 |
| `california_housing_k8_distance_euclidean` | kd_tree | sí | 0 / 0 / 0 | default: kd_tree 200×166, 921/145 ok | 0 | 3.377 | 2277 |
| `csv_numeric_regression_k20_distance_manhattan` | brute | sí | 0 / 0 / 0 | default: brute 200×200, 0/0 ok | 0 | 4.344 | 952 |
| `csv_numeric_regression_k5_uniform_minkowski` | kd_tree | sí | 0 / 0 / 0 | default: kd_tree 200×200, 0/0 ok | 0 | 0.0 | 682 |
| `csv_string_target_k15_distance_euclidean` | brute | conjuntos | 0 / 0 / 0 | default: brute 200×200, 0/0 (índices en orden; distancias ≤ 6,4e-14) | 0 | 19 | 336 |
| `csv_string_target_k20_test05_error` | error (K > n_train = 15) | | | | | | 10 |
| `csv_string_target_k5_uniform_minkowski` | kd_tree | sí | 0 / 0 / 0 | default: kd_tree 200×200, 0/0 ok | 0 | 0 | 323 |
| `diabetes_k10_distance_manhattan` | kd_tree | sí | 0 / 0 / 0 | default: kd_tree 105×111, 5779/5779 ok | 0 | 171.1 | 610 |
| `diabetes_k5_uniform_minkowski` | kd_tree | sí | 0 / 0 / 0 | default: kd_tree 105×111, 5340/5340 ok; pair: kd_tree 113×114, 1/1 ok | 0 | 0.0 | 604 |
| `iris_k1_uniform_manhattan` | kd_tree | sí | 1 / 15 / 0 | default: kd_tree 200×200, 1696/964 ok | 0 | 0 | 336 |
| `iris_k20_distance_manhattan` | kd_tree | sí | 3 / 25 / 3 | default: kd_tree 200×200, 16605/54 ok; pair: kd_tree 200×200, 17428/92 ok | 0 | 4 | 583 |
| `iris_k5_distance_euclidean` | kd_tree | sí | 2 / 6 / 0 | default: kd_tree 200×200, 2982/90 ok | 0 | 5 | 346 |
| `iris_k5_uniform_minkowski` | kd_tree | sí | 2 / 6 / 0 | default: kd_tree 200×200, 2982/179 ok; pair: kd_tree 200×200, 8584/0 ok | 0 | 0 | 568 |
| `iris_k5_uniform_minkowski_test02` | kd_tree | sí | 2 / 4 / 0 | default: kd_tree 200×200, 2982/179 ok | 0 | 0 | 342 |
| `titanic_k5_uniform_minkowski` | kd_tree | sí | 55 / 0 / 40 | default: kd_tree 200×80, 5107/503 ok; pair: kd_tree 201×32, 76/3 ok | 2 | 0 | 401 |
| `wine_k5_uniform_minkowski` | kd_tree | sí | 0 / 0 / 0 | default: kd_tree 200×200, 0/0 ok; pair: kd_tree 200×200, 0/0 ok | 0 | 0 | 591 |
| `wine_k7_distance_manhattan` | kd_tree | sí | 0 / 0 / 0 | default: kd_tree 200×200, 681/0 ok | 0 | 29 | 403 |

Lecturas: (1) la referencia reproduce sklearn en **todos** los casos; (2) en los datos de prueba los empates
casi nunca cambian los vecinos (salvo Titanic: 40 de 268 filas), pero en las rejillas 2D sí (Diabetes: el 46 %
de las celdas); (3) el ONNX propuesto solo discrepa en Titanic (2 etiquetas: `TopK` desempata por el índice
menor y el kd-tree no); (4) skl2onnx falla con `weights='distance'` en todos los casos (con `uniform` acierta
con entrada double, y con float acumula diferencias de empates). Los modelos 2D de Cáncer usan kd_tree aunque
el modelo completo use brute (2 características). `code/knn_training.json`: los 19 renders del script se
ejecutan y coinciden en split, `get_params`, vecinos, predicciones, probabilidades, métricas, matriz de
distancias, permutación y vecinos de la muestra (Titanic queda fuera porque seaborn lo descarga; se añaden
dos renders con la muestra del formulario).

---

## 7. Contenido extraído

Literal, con comentario de origen; los textos se han copiado de las vistas renderizadas por el propio Python
(fixture `views`), no del código fuente, para evitar errores de sangría.

```
content/knn/       intro.md (F1), mensajes.toml (NUEVO: entrenamiento, comun, evaluacion, visualizacion,
                   interactiva —con los textos visibles del lienzo HTML—, distancias, caracteristicas,
                   predicciones, exportar; marcas NUEVO/CORREGIDO de D54–D61 y claves inalcanzables marcadas),
                   interactiva-como-usar.md, interactiva-guia.md, distancias-interpretar.md ({metrica})
content/code/      knn_entrenamiento.py.j2 (NUEVO, como D29/F3; verificado en code/knn_training.json; sus
                   bloques de datos son copia literal de arbol_entrenamiento.py.j2),
                   paquete_modelo_knn.py.j2 y paquete_leeme_knn.txt.j2 (NUEVOS, D1/D57; ejecutados en
                   export_package de cada fixture)
```

KNN no muestra código Python propio: el único fragmento es `regresion_codigo.py.j2` (F2) en "📄 Generar
Código" y el de la frontera (`frontera_decision.py.j2`, F2, D30), que para KNN necesita la variante de "todos
los datos" (D63). Falta por escribir (`ml-lesson-author`): `content/knn/exportar.md` ("ℹ️ ¿Qué formato elijo?")
y, si se quiere, un texto corto sobre la escala (D62).

---

## 8. Mapeo a Rust

### 8.1 Cálculos

| Cálculo | Python | Rust | Huecos |
|---|---|---|---|
| Split | `train_test_split(test_size, 42)` | `split` (F1, D7) | — |
| Búsqueda de vecinos | `KDTree` / `ArgKmin` de sklearn | **Propio** `mltutor-core::knn` (C17, `knn_reference.py`); linfa-nn descartado (MEMORY) | `nth_element` de libc++, heap, `simultaneous_sort`, FMA (D53) |
| `predict`, `predict_proba`, regresión | sklearn | Propio (C17.5) | sumas por pares de numpy (1 ulp, tolerado) |
| Métricas, ROC, umbrales, residuos, formulario | F2 | F2 | — |
| Rejilla 2D | `plot_decision_boundary/surface` | `grid::decision_grid` + `Knn::refit` con todas las filas | `knn_step(n_total)` |
| Permutation importance | sklearn 1.7.2 | `importance::permutation` (F3) | rendimiento (California) |
| Distancias entre clases | `pairwise_distances` | `knn::class_distances` (C19) | — |
| Lienzo | JS en el navegador | `knn::playground` (C18) con las reglas de sklearn | — |
| ONNX | — (no existe) | `mltutor-export::onnx::knn` (grafo D57) | `Scan` con subgrafo en prost |
| Paquete Python | — | `mltutor-export::package` con `X_train` | tamaño (California: 2,5 MB de JSON) |
| Código de entrenamiento | — | `codegen::render_named("knn_entrenamiento.py.j2")` | — |

### 8.2 Vistas

| Vista | Comando | Componente Svelte |
|---|---|---|
| Entrenamiento | `train_model`, `lesson_model` | `KnnTrainTab.svelte` (selector F1, `NumberInput` K, 2 `Select`, `Slider`, botón, `Notice`) |
| Evaluación | `evaluate_model` | `ClassificationEvaluation` / `RegressionEvaluation` (F2) |
| 📊 Probabilidades | `probability_distribution`, `threshold_analysis` | `ProbabilityDistribution`, `ThresholdAnalysis` (F2) |
| 📉 ROC | `roc_curves` | `RocPrCharts` (F2) |
| 🌈 Frontera / 🌐 Superficie | `decision_grid` | `FeaturePairSelector` + `DecisionGridView` (F2) |
| 🎮 Interactiva | `knn_playground` | `KnnPlayground.svelte` (nuevo, C18) |
| 📏 Distancias | `class_distances` | `ClassDistanceHeatmap.svelte` (nuevo, C19) + `MetricGrid` + `Notice` |
| 📊 Distribución de Predicciones | `residuals_view` | `ResidualsAnalysis` (F2) |
| 🔍 Características | `feature_importance(Permutation)` | `FeatureImportanceChart` (F3) + `DataTable` |
| 🔮 Predicciones | `prediction_form`, `predict_one` | `PredictionForm`, `PredictionResultCard` (F2) + `KnnNeighborsTable.svelte` (nuevo, C20) |
| 💾 Exportar | `export_model`, `generate_code(KnnTraining)` | `ExportPanel` (F2, 3 botones) + `ModelCode` |

Paletas: la del lienzo de Python (8 colores), viridis (distancias, importancia), Set3/RdBu (frontera, D23).

### 8.3 Detalles numéricos y de formato que Rust debe copiar

- X en **f64** (sklearn no convierte a f32 en KNN). El formulario de Python sí redondea a f32 (D31: no).
- Euclídea: bloques de 8 con `s = s + d*d` y el resto con `s = d.mul_add(d, s)` (D53); `sqrt` al final, solo
  sobre las K distancias. L1: `s += (x − y).abs()`.
- `min_rdist`: `(0.5*d).powi(2)` es exacto (`pow(x, 2.0)` de libm también); con L1, `0.5*d`.
- Poda con `>` estricto; heap con `>=` estricto al rechazar; `simultaneous_sort` exacto (no `sort_by`, que es
  estable y daría otro orden en los empates de la salida).
- `n_mid = (end − start) / 2` entero; la dimensión de corte con `fmax`/`fmin` (los NaN están prohibidos antes).
- `predict_proba`: acumular en el orden de los vecinos y dividir por la suma (no por K) para `distance`; con
  `uniform` la suma es exactamente K.
- Textos: `{:.2f}` y `{:.3f}` = `format!("{:.N}")`; `{:.1%}` = `format!("{:.1}%", x·100)`.

---

## 9. Reparto de tareas y checklist

Orden: core → paridad → comandos → UI → contenido → smoke test. Las casillas de INVENTARIO §5 se marcan al
cerrar la fila de revisión.

### 9.1 `rust-ml-engineer` (`crates/mltutor-core`, `crates/mltutor-export`)

1. [ ] `knn::{KnnConfig, Knn, fit, SearchEngine}`: regla del motor, `KdTree` (libc++ `nth_element`, límites,
   `min_rdist`, recorrido, heap, `simultaneous_sort`) y fuerza bruta. Tests: `neighbours_test.indices` y
   `distances` **exactos** en los casos `kd_tree`, `reference_impl.kd_tree.idx_array`/`idx_start`/`idx_end`,
   conjuntos en los `brute`; un test unitario de `nth_element` con secuencias generadas contra la referencia.
2. [ ] `impl Predictor for Knn` (`predict`, `predict_proba`, `classes`, `refit`) + `kneighbors`. Tests: `y_pred`,
   `proba`, `prediction_form.*.neighbours`.
3. [ ] Rejillas: `decision_grid` con todas las filas para KNN. Tests: `two_feature_models.*.z` exacta en los 19
   casos (Diabetes age/sex es el de control).
4. [ ] `importance::permutation` sobre `Knn` (ya genérica). Test: `permutation_importance` 1e-12. Benchmark de
   California (`#[ignore]`) < 1 s.
5. [ ] `knn::class_distances` (C19) y `knn::playground` (C18). Tests: `distance_analysis.app.matrix` 1e-10 y
   `with_model_split_D56`; `interactive.*.samples[*].sklearn_2d_all_data`.
6. [ ] `mltutor-export`: `onnx::knn_classifier`/`knn_regressor` con el grafo D57 (`Scan` + `TopK` + regla de
   distancia 0), ir 8, opset 15, nombres de nodo únicos (MEMORY). Tests: estructura frente a
   `onnx_proposal_D57.nodes` y salidas frente a `head`; en el job de F6, onnxruntime sobre train + test.
7. [ ] `package` para KNN (`modelo.json` igual a `export_package.model_json` salvo flotantes; `X_train`/`y_train`
   reconstruidos desde `train_indices` cuando el fixture los omite; script = `export_package.script`; LEEME con
   `paquete_leeme_knn.txt.j2`).
8. [ ] `codegen`: contexto tipado de `knn_entrenamiento.py.j2`. Test: `code/knn_training.json` (texto exacto).

### 9.2 `tauri-ui-engineer` (`src-tauri` + `ui`)

Backend:
1. [ ] `ModelConfig::Knn`, `KnnSummary`, DTO de §5.3, `CommandError::TooFewSamples` (D59) con el texto de
   `knn/mensajes.toml`.
2. [ ] Comandos `class_distances`, `knn_playground` (build.rs + capabilities) y ampliaciones de `predict_one`,
   `decision_grid` (todas las filas en KNN), `prediction_form` (todas las filas en KNN), `generate_code`,
   `export_model` (nombres `knn.onnx`, `modelo_knn.zip`, `knn_entrenamiento.py`).
3. [ ] `--smoke-test`: Iris K5, Cáncer K10 distance, Diabetes K5 y California K5 (todas las vistas, el lienzo con 3
   puntos); `--screenshot --scenario` para las capturas de §0.

Frontend:
4. [ ] Store de la lección (`lessons/knn/state.svelte.ts`): K, pesos, métrica, `test_size`, subvista, pares, estado
   del lienzo (D54).
5. [ ] `KnnPlayground.svelte` (C18, D54/D55): Plotly, clic/arrastre/valores exactos, animación de vecinos,
   tarjetas y tabla; tests de vitest con `interactive.*` (paleta, límites, controles iniciales).
6. [ ] `ClassDistanceHeatmap.svelte` (C19), `KnnNeighborsTable.svelte` (C20).
7. [ ] Pestañas 1–6 en `ui/src/lib/lessons/knn/` reutilizando F2/F3 y un `KnnBanner` (D32/D45: `config_modelo`,
   `aviso_config_cambiada`); D60 (info del dataset siempre); D61 (nombres de clase en C4/C5).
8. [ ] Capturas Rust de las pantallas de §0 y comparación (`capturas/comparacion-fase4.md`).

### 9.3 `ml-lesson-author`

1. [ ] Cargar `content/knn/mensajes.toml`, los `.md` y las plantillas sin reescribirlos.
2. [ ] Revisar los textos NUEVO (D54, D58, D59) y escribir `content/knn/exportar.md`; con D55, la línea "Valor K
   (1-15)" de `interactiva-guia.md` pasa a "(1-20)" con marca CORREGIDO.
3. [ ] `frontera_decision.py.j2`: variable nueva (p. ej. `datos = "train" | "todos"`) para que el código de KNN
   use `X`/`y` completos (D63); actualizar `code/regression.json` y los contextos de F2/F3.
4. [ ] Valorar el ejemplo del texto `[importancia] permutacion` (habla de un árbol de Diabetes).

### 9.4 Revisión

`test-runner` → `rust-reviewer` → `ml-correctness-reviewer` (kd-tree exacto, pesos, ONNX, script).

---

## 10. Decisiones para el usuario

Se numeran a continuación de D51 (D52–D67). Si no se dice lo contrario, se aplica la recomendación.

| # | Decisión | Recomendación |
|---|---|---|
| D52 | **Paridad de los vecinos** (concreción de D6): con empates, sklearn elige según el recorrido del kd-tree; "gana el índice menor" cambia hasta 5 340 celdas de una rejilla | **Paridad exacta**: portar el kd-tree de sklearn (con el `nth_element` de libc++, como en los fixtures de macOS) y usar fuerza bruta con índice menor donde sklearn usa brute. Especificación: `tools/knn_reference.py` (verificada en los 19 casos) |
| D53 | **Suma de la distancia euclídea**: la wheel arm64 usa FMA en las `n % 8` últimas características | Reproducirla (`mul_add`) para tener distancias bit a bit; alternativa: suma simple y tolerancia de 4 ulp (1 celda de Iris cambia de vecinos) |
| D54 | **🎮 Visualización Interactiva**: HTML + JS de 34 KB que calcula con sus propias reglas, sin ejes, y cuya tabla y valores editables nunca se ven | Componente nativo `KnnPlayground` (Plotly), calculado en Rust con las reglas de sklearn sobre el modelo 2D de la frontera; mismos controles, colores y textos; clic, arrastre y valores exactos; líneas a los vecinos animadas una a una; ejes con nombres; tabla y métricas de puntos de prueba visibles |
| D55 | **K del lienzo**: deslizador 1–15 aunque el modelo admite 20 (con K = 20 marca 15 y usa 20) | 1–20, como el entrenamiento; corregir "Valor K (1-15)" en la guía |
| D56 | **Split del 📏 Análisis de Distancias**: Python usa siempre `test_size = 0,3` | Usar el split del modelo (igual con el valor por defecto) y decir cuántas muestras se usaron |
| D57 | **Exportar** (concreción de D1/D28): Python ofrece "📄 Generar Código" (el fragmento roto de Regresión) y un pickle llamado `linear_regression_model.pkl`; no hay ONNX. skl2onnx exporta KNN incrustando el train con `Scan` + `TopK`, pero en la 1.19.1 `weights='distance'` sale como `uniform` (error de signo) | Tres botones: "📄 Código de Entrenamiento" (`knn_entrenamiento.py.j2`, verificado), "💾 Exportar a ONNX" con un grafo propio (`float_input` f32 → f64, `Scan` por filas con `Sub`/`Mul`/`ReduceSum`/`TopK`, regla de distancia 0 de sklearn, `OneHot` + votos o media ponderada; verificado con `onnx.reference`: 0 discrepancias en las etiquetas de 18 casos y 2 en Titanic, por empates) y "📦 Paquete Python (.zip)" (`modelo.json` con `X_train` en orden + script que hace `fit`, comprueba vecinos y predicciones y guarda `modelo_knn.pkl`). Sin pickle directo |
| D58 | **Predicción**: solo la clase o el valor y un `st.info` con K, pesos y métrica | Añadir las probabilidades (clasificación) y la tabla de los K vecinos con distancia, peso y clase/valor |
| D59 | **K mayor que las muestras de entrenamiento**: Python entrena y luego fallan todas las vistas en inglés | Bloquear el entrenamiento con un mensaje en español |
| D60 | **"Información del Dataset"** en Entrenamiento: desaparece tras el primer render | Mostrarla siempre, como en Regresión y Árboles |
| D61 | **Nombres de clase** en 📊 Probabilidades y 📉 ROC: Python pasa `None` (variable local) y escribe "Clase 0" | Usar los nombres de clase, como en Regresión |
| D62 | **Escalado**: KNN sin escalar (Proline domina en Vino); los textos recomiendan StandardScaler | Mantener sin escalar (paridad). Opción futura: un control "Escalar características" fuera de esta fase |
| D63 | **Datos de las vistas 2D y del formulario**: la frontera, la superficie, el lienzo y los rangos del formulario usan **todo** el dataset (en Regresión y Árboles, solo train) | Igual que Python (paridad), incluido el paso de la rejilla con `n` total; `frontera_decision.py.j2` gana una variante "todos los datos" |
| D64 | **Capturas Qt de F4** (5,4 MB) y **fixtures** (12 MB; California 2,0–2,3 MB cada uno) | Versionarlos, como en F2/F3 |
| D65 | **Capturas del lienzo**: `capture_qt` no ve WebEngine | Versionar el HTML de referencia y las dos capturas de Chrome headless (`09c`, `09d`) |
| D66 | **`tools/knn_reference.py`** (Python) | Mantenerlo como especificación ejecutable (como `cart_reference.py`, D51) |
| D67 | **Ratio Max/Min en binaria**: con dos clases la matriz solo tiene dos celdas, iguales, y el ratio vale siempre 1,00 → "🔴 Las clases están muy próximas" en Cáncer y Titanic, sea cual sea la separación | Mantener la matriz y las métricas; con 2 clases, sustituir el veredicto por un `st.info` nuevo: "Con dos clases solo hay una distancia entre clases, así que el ratio siempre vale 1 y no mide la separación. Compárala con la distancia media dentro de cada clase." y mostrar en la diagonal la distancia media **dentro** de cada clase (Python la deja a 0), marcada como NUEVO. Alternativa: paridad estricta (veredicto 🔴 siempre) |

---

## 11. Riesgos

| Riesgo | Mitigación |
|---|---|
| Portar `nth_element` de libc++ y el recorrido exactos es delicado | `knn_reference.py` como especificación; el `idx_array` del fixture permite localizar el primer nodo que difiere; tests unitarios de `nth_element` |
| Una versión futura de libc++ o de sklearn cambie el orden dentro de las hojas | Los fixtures fijan sklearn 1.7.2 y el `idx_array`; si se regeneran en otra plataforma (Linux) cambiarán los empates: documentado en MEMORY |
| `f64::mul_add` lento sin FMA por hardware (x86-64 baseline: llamada a `fma` de libm) | Solo en `n % 8` términos; medir la rejilla de California en Windows; plan B: activar `target-feature=+fma` no es portable, mejor aceptar el coste o D53 alternativa |
| Permutación de California (≈ 500 000 consultas) | Cancelable por columna, progreso; paralelizar con `std::thread::scope` si pasa de 1 s |
| Paquete Python de California: 2,5 MB de JSON | Aceptable; comprimido en ZIP baja a ~0,8 MB |
| ONNX con `Scan` poco soportado fuera de onnxruntime | Documentado en el LEEME/expander; onnxruntime es el destino de D1 |
| El lienzo en WebKitGTK con 20 000 puntos (California no tiene lienzo: es regresión; Titanic 891) | `scattergl`; el lienzo solo existe en clasificación |
| Fixtures grandes | `dumps_compact`; `X_train` e `idx_array` como hash por encima de 1 000 filas |

---

## 12. Hallazgos de la versión Python

1. El desempate de los vecinos depende del motor (kd-tree o brute) y, en el kd-tree, del `nth_element` de la
   biblioteca estándar de C++: sklearn da otros vecinos en otra plataforma cuando hay empates.
2. La wheel arm64 de sklearn calcula la distancia euclídea con FMA solo en las `n % 8` últimas características.
3. El lienzo interactivo clasifica en JavaScript con reglas propias (peso 1 a distancia 0 sin anular a los
   demás; sin ejes) y no usa el modelo.
4. `session_state.test_points` nunca recibe puntos: la tabla "📊 Análisis de Puntos de Prueba" y el respaldo con
   valores editables son inalcanzables, y la tabla fallaría con el modelo completo.
5. El deslizador K del lienzo va de 1 a 15; con K = 20 marca 15 y usa 20.
6. 📏 Análisis de Distancias usa siempre `test_size = 0,3`, sea cual sea el slider.
7. 📊 Distribución de Probabilidades y 📉 ROC muestran "Clase 0…" porque `class_names` es una variable local que
   vale `None`.
8. La frontera, la superficie, el lienzo y el formulario usan todo el dataset, no el de entrenamiento; el paso
   de la rejilla también depende del tamaño total.
9. "Información del Dataset" desaparece tras pulsar "🚀 Entrenar Modelo KNN".
10. Con K > n_train el modelo "se entrena" y todas las vistas fallan en inglés.
11. 💾 Exportar es la función de Regresión: el pickle se llama `linear_regression_model.pkl` y el código guarda
    `modelo_regresion_logistica.pkl`. No hay ONNX, aunque el inventario lo diga.
12. skl2onnx 1.19.1 convierte mal `weights='distance'` con opset ≥ 11: niega las distancias de `TopK` y todos
    los pesos quedan en 1e6 (equivale a `uniform`); además sustituye la regla de distancia 0 por `1/max(d,
    1e-6)` y con `FloatTensorType` guarda el train en float32.
13. El botón de regresión "📊 Distribución de Predicciones" muestra predicciones frente a reales y residuos.
14. Código muerto confirmado: `create_animated_neighbors_plot` (sin llamadas), la tabla y el respaldo de puntos
    de prueba, el `<script>` de `postMessage`, el CSS de las pestañas y del iframe, `knn_metrics`,
    `dataset_loaded`, los avisos "Primero debes entrenar…" de Frontera/Superficie, "No se encontraron los datos
    necesarios para la evaluación.", "Primero selecciona y carga un dataset…", y los imports sin usar
    (`px`, `make_subplots`, `sns`, `evaluate_*`; `train_knn_model` en `app.py`).
15. En 📏, con dos clases el ratio Max/Min es siempre 1,00 y el veredicto siempre 🔴 (Cáncer, Titanic): la matriz es simétrica y solo tiene dos celdas fuera de la diagonal.

---

## 13. Cambios propuestos en otros documentos

No se han editado (los aplica quien cierre la fase).

**INVENTARIO.md §5**: "Entrenamiento: … ⚠️ KNN como capa sobre `linfa-nn`" → "KNN propio con kd-tree exacto
(D52)"; "Visualización de clasificación: … 🎮 visualización interactiva nativa (D54) …"; "Características:
permutation importance (C10 de F3)"; "Predicciones con formulario, probabilidades y los K vecinos (D58)";
"Exportar: Pickle y ONNX. ⚠️ Pickle" → "Exportar: código de entrenamiento, ONNX propio y paquete Python ZIP
(D57); Python solo tenía el pickle (`linear_regression_model.pkl`) y el fragmento de Regresión". §8: añadir la
tabla y el respaldo de puntos de prueba de `create_interactive_knn_visualization` (inalcanzables).

**PLAN.md, Fase 4**: enlazar este documento y `capturas/qt/fase4/`; añadir D52–D67 (como D36–D51, con enlace a
§10); cambiar "KNN … sobre `linfa-nn` (kd-tree o fuerza bruta)" por "KNN propio: kd-tree portado de sklearn con
paridad exacta (D52) y fuerza bruta"; añadir el ONNX propio y el paquete Python.

**tests/parity/README.md**: ya actualizado con `knn/` y `code/knn_training.json`.

**MEMORY.md**: ver las líneas propuestas en el informe de `migration-porter` (2026-10-15).
