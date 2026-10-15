# Fase 3 — 🌲 Árboles de Decisión

Plan de port de `migration-porter` (2026-10-14). Cubre INVENTARIO §3 (🌲 Árboles de Decisión), la parte
de §0 (diálogos de exportación), §7 (código Python generado de los árboles) y §9 de estas pantallas.
Referencia: rama `codex/desktop-migration-study`, la versión Qt. Plantilla: [fase2-regresion.md](fase2-regresion.md).

La lección **reutiliza** los componentes compartidos de F2 (C1–C12) y añade cuatro nuevos: el diagrama del
árbol (C13), el texto de reglas (C14), el código Python del árbol (C15) y el camino de decisión (C16).
Además implementa C10 (importancia nativa y por permutación), que F2 solo especificó.

**Hallazgo principal (D36):** la paridad exacta del árbol es posible y necesaria. scikit-learn sortea en
cada nodo el orden de las características con su propio RNG y se queda con la **primera** que alcanza la
mejor mejora; los empates son frecuentes (la raíz de Iris: *petal length* y *petal width* empatan). Con
otro orden sale otro árbol. `tools/cart_reference.py` (transcripción de sklearn 1.7.2) reproduce bit a bit
la estructura de los 18 árboles entrenados de los fixtures y de sus 22 árboles 2D. Rust debe seguirlo.

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

En `docs/migration/capturas/qt/fase3/`, generadas con

```sh
uv run python tools/capture_qt.py docs/migration/capturas/qt/fase3 --phase 3 --quantize
```

(ventana real, no `offscreen`; unos 75 s; 50 PNG, 6,0 MB ya cuantizados). Solo 02, 05 y 31 tienen además
`NNN_ventana.png`. Igual que en F2, los expanders abiertos con `expand_all` siguen abiertos en las capturas
siguientes, y los deslizadores de tamaño (sin key) conservan 20 × 15 desde la 06.

| Archivo | Pantalla |
|---|---|
| `00_evaluacion_sin_entrenar`, `00b_visualizacion_sin_entrenar` | Pestañas sin modelo (el mismo `st.warning` en 2–6) |
| `01_entrenamiento_iris`, `02_entrenado_iris(_ventana)` | 🏋️ Entrenamiento (Clasificación, gini, 3, 2, 0,3) antes y después de "Entrenar Modelo" |
| `03_evaluacion_clasificacion_iris`, `04_…_expandido` | 📈 Evaluación (C1 + C3) |
| `05_arbol_iris(_ventana)`, `06_arbol_iris_20x15`, `07_arbol_iris_codigo` | 🌲 Árbol (`plot_tree`) a 14 × 10 y a 20 × 15, y su código |
| `08_texto_iris` | 📝 Texto (`export_text` con pesos; clases como códigos) |
| `09_frontera_iris`, `10_frontera_petalos_iris` | 🌈 Frontera (C8) por defecto y con los pétalos |
| `11_roc_iris` | 📉 Curva ROC multiclase (C4) con probabilidades de hoja |
| `12_caracteristicas_iris`, `13_…_expandido` | 🔍 Importancia nativa (C10) |
| `14_prediccion_formulario_iris`, `15_prediccion_camino_iris`, `16_…_codigo` | 🔮 Formulario, resultado y camino de decisión (C16) |
| `17_exportar_iris`, `18_…_expandido`, `19_exportar_pickle`, `20_exportar_reglas`, `21_exportar_codigo_python` | 💾 Exportar: estado inicial, vista previa del código, y los tres botones de descarga |
| `22_entrenamiento_cancer_entropy_d10`, `23_arbol_cancer_d10`, `24_roc_binaria_cancer` | Árbol profundo (27 nodos; texto ilegible) y ROC binaria |
| `25_prediccion_formulario_titanic`, `26_prediccion_camino_titanic` | Formulario con selects de metadatos; el camino muestra códigos ("Hombre_Adulto = 0.0000") |
| `27_aviso_clasificacion_en_diabetes` | Diabetes con el tipo por defecto (**Clasificación**): aviso |
| `28_entrenamiento_regresion_diabetes` … `39_exportar_regresion_diabetes` | Regresión: criterios de regresión, evaluación (C2), árbol naranja, texto, residuos (C7), superficie (C8, por defecto y bmi/s5), importancia, predicción y exportar |
| `40_arbol_california_d8` | California, profundidad 8 (407 nodos): el diagrama de matplotlib es ilegible (D41) |
| `41_error_clasificacion_california` | Clasificación sobre California: "Error al entrenar el modelo: Unknown label type: continuous…" (inglés) |
| `42_clasificacion_entrenada_en_diabetes`, `43_error_evaluacion_clasificacion_diabetes` | Clasificación sobre Diabetes **entrena** 192 clases y 📈 Evaluación cae en la página de error (`KeyError: 'accuracy'`) |
| `44_aviso_regresion_en_iris` | 📈 Regresión sobre Iris: aviso |

No se ha capturado: los diálogos nativos de guardar ni el spinner.

---

## 1. Estructura de la lección

`mltutor/apps/decission_tree_app.py::run_decision_trees_app` (el nombre del archivo tiene la errata
"decission").

1. Cabecera: `st.header("🌲 Árboles de Decisión")`, "Aprende sobre los árboles de decisión de forma
   interactiva" y el expander "ℹ️ ¿Qué son los Árboles de Decisión?" (cerrado; `content/arboles/intro.md`).
   Ya está en F1 (`content/inicio/navegacion.toml [lessons.arboles]`).
2. 7 pestañas (botones `tab_0..6`, estado `active_tab`, 0 por defecto) y `---`: 📊 Datos · 🏋️ Entrenamiento ·
   📈 Evaluación · 📉 Visualización · 🔍 Características · 🔮 Predicciones · 💾 Exportar. El bucle que crea
   los botones está **duplicado** (el primero no hace nada) y el CSS `tab-button-dt` no tiene efecto en Qt.
3. `run_dataset_tab(active_tab)` (F1), como en Regresión: pestaña 0 = exploración; 1 = cabecera "📊 Selección
   y Preparación de Datos" + selector; ≥ 2 = `st.info("📊 **Dataset actual:** {selected_dataset}")` + `---`.
4. Contenido de la pestaña (§2) y navbar (F1): (—, "Continuar a Entrenamiento"), ("Volver a Datos",
   "Continuar a Evaluación"), …, ("Volver a Predicciones", —), con **Características** como pestaña 4.

### 1.1 Estado de sesión (Python)

| Clave | Escribe | Contenido |
|---|---|---|
| `active_tab` | pestañas, navbar | 0..6 |
| `tree_type` | botones de tipo | `"Clasificación"` **por defecto, sea cual sea el dataset** (D37), o `"Regresión"` |
| `dataset_option` | Entrenamiento | último dataset cargado (no se usa para nada más) |
| `criterion`, `max_depth`, `min_samples_split`, `test_size` | **cada render** de la pestaña 1 | valores actuales de los controles, aunque no se haya reentrenado (D45) |
| `tree_model`, `X_train`, `X_test`, `y_train`, `y_test` | "Entrenar Modelo" | modelo y split (DataFrames/Series con el índice original) |
| `feature_names`, `class_names`, `dataset_info` | ídem | nombres legibles, clases y descripción |
| `is_trained` | ídem | `True` tras entrenar; **no se borra al cambiar de dataset** (D32 de F2) |
| `viz_type` | `create_button_panel` | global (D13 de F1: en Rust, por lección) |
| `viz_feature1`, `viz_feature2` | selects de frontera/superficie | compartidos con Regresión |
| `feature_{i}` | formulario | compartidos con las otras lecciones |

Flujo de entrenamiento: `load_data(selected_dataset)` → `preprocess_data(X, y, test_size)` (D7, sin
estratificar) → `train_decision_tree(X_train, y_train, criterion, max_depth, min_samples_split, tree_type)` =
`DecisionTreeClassifier` o `DecisionTreeRegressor(max_depth, min_samples_split, criterion, random_state=42)`
con el resto de hiperparámetros por defecto (`splitter="best"`, `max_features=None`, `min_samples_leaf=1`,
`min_impurity_decrease=0`, `ccp_alpha=0`) sobre los datos **sin escalar**. Spinner "Entrenando modelo...".
La rama de las pestañas 2–6 la decide `tree_type` **actual** de la sesión, no el tipo del modelo
entrenado: si se pulsa el otro botón sin reentrenar, la evaluación cambia de rama sobre el mismo modelo
(fixture `iris_gini_d3.views.evaluation_after_switching_tree_type_without_retraining`; D45).

---

## 2. Pestañas

### 2.1 📊 Datos (pestaña 0)

Exactamente la de F1 (`DatasetTab.svelte`).

### 2.2 🏋️ Entrenamiento (pestaña 1)

Orden: selector de F1 → `st.header("Configuración del Modelo")` → `### Información del Dataset` + caja
`info-box` (`dataset_info.description`; CSV como en F2) → `### Tipo de Árbol de Decisión` → 2 botones → aviso
de tipo → `### Parámetros del Modelo` (2 columnas) → botón → resultado → navbar. Si falla la carga:
`st.error("Error al cargar el dataset: {e}")` y no aparecen los botones de tipo (los parámetros sí).

| Control | Tipo | Etiqueta exacta | Rango / opciones | Por defecto | key | Efecto |
|---|---|---|---|---|---|---|
| Tipo clasificación | botón (col. 1, ancho completo) | `🏷️ Clasificación` | — | primary si `tree_type == "Clasificación"` | `btn_classification` | `tree_type = "Clasificación"`; help "Para predecir categorías o clases" |
| Tipo regresión | botón (col. 2) | `📈 Regresión` | — | primary si `"Regresión"` | `btn_regression` | `tree_type = "Regresión"`; help "Para predecir valores numéricos continuos" |
| Criterio | selectbox (col. 1) | `Criterio de División:` | clasificación `gini`, `entropy`; regresión `squared_error`, `friedman_mse`, `absolute_error` (sin traducir) | índice 0 | sin key (por sitio de llamada); si el valor no está en las opciones nuevas vuelve al primero | `criterion`; help "Medida de la calidad de una división. Gini o Entropy para clasificación, MSE para regresión." |
| Profundidad | slider entero (col. 1) | `Profundidad Máxima:` | 1 … 10 | 3 | sin key | `max_depth`; help "La profundidad máxima del árbol. A mayor profundidad, más complejo el modelo." |
| Muestras mínimas | slider entero (col. 2) | `Muestras Mínimas para División:` | 2 … 10 | 2 | sin key | `min_samples_split`; help "Número mínimo de muestras para dividir un nodo." |
| Tamaño de test | slider real (col. 2) | `Tamaño del Conjunto de Prueba:` | 0,1 … 0,5, paso 0,05 | 0,3 | sin key | help "Proporción de datos que se usará para evaluar el modelo." |
| Entrenar | botón primary | `Entrenar Modelo` (sin emoji; en Regresión es "🚀 Entrenar Modelo") | — | — | **sin key** | Entrena y guarda el estado de §1.1 |

Aviso (`st.warning`) si `tree_type` ≠ tarea del dataset: "Este dataset parece ser más adecuado para
{Clasificación|Regresión}. La selección actual podría no ofrecer resultados óptimos." Resultado:
`st.success("¡Modelo entrenado con éxito! Ahora puedes explorar las otras pestañas.")` +
`st.info("👉 Ve a la pestaña '🌲 Visualización' para ver el árbol generado.")` (la pestaña se llama
📉 Visualización; D49) o `st.error("Error al entrenar el modelo: {e}")`. Clasificación sobre un objetivo
continuo: California → error en inglés (captura 41); Diabetes (floats enteros) → **entrena 192 clases**
(captura 42; D37).

### 2.3 📈 Evaluación (pestaña 2)

`st.header("Evaluación del Modelo")`. Sin modelo: `st.warning("Primero debes entrenar un modelo en la
pestaña '🏋️ Entrenamiento'.")` (igual en 2–6). Con modelo: `show_detailed_evaluation(y_test,
model.predict(X_test), class_names si tree_type == "Clasificación" si no None, tree_type)` → **C1** o **C2**
+ **C3** (la interpretación de regresión dice "Tu modelo de regresión lineal explica…" también aquí: literal).
Clasificación sobre Diabetes: `KeyError: 'accuracy'` → página de error (captura 43).

### 2.4 📉 Visualización (pestaña 3)

`st.header("Visualización del Árbol")` + `### Tipo de visualización` + panel (`create_button_panel`):

| Tipo | Botón (valor, key) | Contenido |
|---|---|---|
| ambos | `🌲 Árbol` (Árbol, `viz_tree`) | 2 sliders en 2 columnas, **sin key**: `Ancho de figura:` 8…20 (14) y `Alto de figura:` 6…15 (10) (pulgadas) → `plot_tree` en `[1, 4, 1]` (**C13**) → enlace "📥 Descargar visualización del árbol" (`arbol_decision.png`, 300 dpi) → `show_code_with_download(VIZ_TREE_CODE, "Código para visualizar el árbol", "visualizar_arbol.py")` |
| ambos | `📝 Texto` (Texto, `viz_text`) | `st.text(export_text(model, feature_names, show_weights=True))` (**C14**) → enlace "📥 Descargar texto del árbol" (`arbol_texto.txt`) → código `TEXT_TREE_CODE` ("Código para obtener el texto del árbol", `texto_arbol.py`) |
| clasificación | `🌈 Frontera de Decisión` (Frontera, `viz_boundary`) | **C8** modo frontera con `DecisionTreeClassifier(max_depth, min_samples_split, criterion` **del modelo**`, random_state=42)` reentrenado en 2 características de train. Si falla: `st.error("Error al mostrar la visualización de frontera de decisión: {e}")` + `st.info` de requisitos + `st.exception` |
| clasificación | `📉 Curva ROC` (ROC, `viz_roc`) | **C4** con `predict_proba(X_test)` (probabilidades de hoja: muchas puntuaciones empatadas, curvas con pocos puntos). Si falla: `st.error("Error al mostrar las curvas ROC: {e}")` + `st.info` + `st.exception` (con 192 clases: "Found input variables with inconsistent numbers of samples") |
| regresión | `📈 Análisis de Residuos` (Residuos, `viz_residuals`) | `### 📊 Gráfico de Predicciones vs Valores Reales` + **C7a**, `### 📈 Análisis de Residuos` + **C7b** |
| regresión | `🌐 Superficie de Decisión` (Superficie, `viz_surface`) | **C8** modo superficie con `DecisionTreeRegressor(max_depth, min_samples_split, criterion` **de la sesión**`, random_state=42)` (D45) |

Los avisos "solo está disponible para modelos de clasificación" de Frontera/ROC son inalcanzables (el panel
no ofrece esos botones en regresión). Navbar: ("Volver a Evaluación", "Continuar a Características").

### 2.5 🔍 Características (pestaña 4)

`st.header("Importancia de Características")` + **C10** (`display_feature_importance(model, feature_names,
X_test, y_test, tree_type)`). Como el árbol tiene `feature_importances_`, Python **solo** muestra la
importancia nativa ("Importancia Nativa del Modelo"); la permutación nunca se ve en esta lección (D38):

1. `### Importancia de Características`.
2. Barras horizontales (12 × max(8, 0,4·n) in): orden ascendente por importancia (`sort_values`, quicksort:
   los empates a 0 salen en orden arbitrario), colores `viridis(linspace(0, 1, n))` en ese orden, alpha 0,8
   (el `set_color` lo pisa: color opaco), etiqueta `{:.3f}` a la derecha (x + 1 % del límite), eje x
   "Importancia", título "Importancia de Características\n(Importancia Nativa del Modelo)", rejilla x 0,3.
3. `### Valores Detallados`: tabla Característica/Importancia (`round(4)`), orden descendente, **con
   índice** (el número de columna original).
4. Expander "📖 Cómo interpretar estos resultados": `nativa_clasificacion` o `nativa_regresion` +
   `consejos` (`componentes/mensajes.toml [importancia]`). "La suma de todas las importancias = 1.0" es
   falso si el árbol es una sola hoja (todo 0).

Navbar: ("Volver a Visualización", "Continuar a Predicciones").

### 2.6 🔮 Predicciones (pestaña 5)

`st.header("🔮 Predicciones con Nuevos Datos")` + **C9** (`create_prediction_interface(model, feature_names,
class_names, tree_type, X_train, selected_dataset)`). Clasificación: tarjeta "Clase: {class_names[int(ŷ)]}"
(sin tabla de probabilidades; D46); regresión: "Valor: {ŷ:.4f}". Después, en ambos casos,
`### Camino de decisión` + **C16** (`show_prediction_path`). Navbar: ("Volver a Características",
"Continuar a Exportar").

### 2.7 💾 Exportar (pestaña 6)

`st.header("Exportar Modelo")` + `display_tree_export_options(model, feature_names, class_names, tree_type,
max_depth, min_samples_split, criterion)`:

1. `#### Formatos de Exportación`, 3 columnas con un botón cada una (ancho completo, sin key); al pulsarlo
   aparece un `st.download_button` debajo (en Qt, un botón que abre el diálogo de guardar):

   | Botón | Descarga | Contenido |
   |---|---|---|
   | `📦 Exportar como Pickle` | "Descargar Modelo (.pkl)", `decision_tree_model.pkl` | `pickle.dump({'model', 'feature_names', 'class_names', 'task_type', 'model_info': {task_type, max_depth, min_samples_split, criterion, n_features, n_nodes, n_leaves}})` |
   | `📝 Exportar Reglas como Texto` | "Descargar Reglas (.txt)", `decision_tree_rules.txt` | `export_text(model, feature_names)` **sin** `show_weights` (distinto del de 📝 Texto) |
   | `🐍 Exportar como Código Python` | "Descargar Código Python (.py)", `arbol_decision.py` | `export_tree_as_python_code` (**C15**) |

2. `#### 👀 Vista Previa del Código Python` + expander "Ver código Python del árbol" (cerrado): el mismo
   código + `st.info` "**Información del código generado:**" (nodos, hojas, profundidad, características
   con las 3 primeras, tipo de tarea; "…completamente **autocontenido** y no requiere scikit-learn…").
3. `show_code_with_download(generate_tree_model_export_code(feature_names), "Código para Usar el Modelo",
   "usar_modelo.py")`: carga `decision_tree_model.pkl` y define `predecir` (y `predecir_con_probabilidades`).

No hay ONNX ni imagen en esta pestaña (la imagen se descarga en 🌲 Árbol). Rust: D43. Navbar:
("Volver a Predicciones", —).

---

## 3. Componentes reutilizados y nuevos

| Comp. | Estado tras F2 | Uso en Árboles | Qué falta |
|---|---|---|---|
| Trait `Predictor` (§3.0 de F2) | hecho (`model.rs`) | `DecisionTree` lo implementa | `classes()` = códigos vistos en train; `refit` con los mismos hiperparámetros y `random_state=42`; `predict` convierte X a **f32** antes de comparar |
| C1 Evaluación clasificación | hecho | igual | nada |
| C2 Evaluación regresión | hecho | igual | nada |
| C3 Interpretación | hecho | igual (literal "regresión lineal") | nada |
| C4 ROC/PR | hecho | probabilidades de hoja (empates) | nada: `roc_curve` ya agrupa puntuaciones iguales; tests con `trees/*.evaluation.curves` |
| C5/C6 Probabilidades y umbrales | hecho | **no se usan** en Árboles | — |
| C7 Residuos | hecho | igual | nada |
| C8 Frontera/superficie | hecho (`grid::decision_grid`) | reentrena el árbol 2D | la rejilla se evalúa en f32 (la hace `predict` del árbol); paleta Set3/RdBu (D23) |
| C9 Formulario | hecho | igual | `PredictionOutcome` + camino (C16) y probabilidades de hoja (D46) |
| C10 Importancia | **solo especificado** | nativa + permutación | **implementar aquí** (§3 C10) |
| C11 Exportación | hecho para lineal/logística | ONNX TreeEnsemble*, ZIP, .txt, .py | ramas de árbol en `onnx_model`, `package::estimator_name`, `model_json`; formatos nuevos (D43) |
| C12 Codegen | hecho (`render_named`) | 6 plantillas nuevas | contexto tipado de `arbol_python*.py.j2` y `paquete_*_arbol` |
| C13 Diagrama del árbol | — | 🌲 Árbol | **nuevo** |
| C14 Texto de reglas | — | 📝 Texto, reglas .txt | **nuevo** |
| C15 Código Python del árbol | — | 💾 Exportar | **nuevo** |
| C16 Camino de decisión | — | 🔮 Predicciones | **nuevo** |

UI ya existente que se reutiliza: `ClassificationEvaluation`, `RegressionEvaluation`, `Interpretation`,
`MetricGrid`, `RocPrCharts`, `ResidualsAnalysis`, `DecisionGridView` + `FeaturePairSelector`,
`PredictionForm`, `PredictionResultCard`, `ExportPanel`, `ModelCode`, `ui/src/lib/ml/modelSession.svelte.ts`
(estado por lección) y `viewLoader.svelte.ts`. De `lessons/regression/` sirven de modelo `ModelBanner`
(D32), `TrainTab`, `VisualizationTab` y `ExportTab`.

### C10. Importancia de características (`viz/features.py`)

- **Core** `importance::native(&DecisionTree) -> Vec<f64>`: `compute_feature_importances(normalize=True)`:
  para cada nodo interno, `imp[f] += w·impureza − w_izq·impureza_izq − w_der·impureza_der` en orden de
  nodo; dividir por `w` de la raíz; normalizar por la suma si es > 0 (si no, todo 0).
- **Core** `importance::permutation(model: &dyn Predictor, x_test, y_test, n_repeats = 10, random_state =
  42, &TrainControl) -> PermutationImportance { baseline, importances: Array2 (n_features × n_repeats),
  mean, std (ddof 0) }`. Regla exacta de sklearn 1.7.2 (verificada bit a bit: `reimplementation_bit_exact`):
  1. `seed = RandomState(42).randint(2**31)` — **una sola semilla para todas las columnas** (desde sklearn
     1.5; F2 §C10 decía "una por columna", que era el comportamiento antiguo). Con MT19937 = `next_u32 &
     0x7fffffff` (máscara sin rechazo): en los fixtures, `permutation_importance.seed` = 1608637542.
  2. Para cada columna c: `rs = RandomState(seed)`, `idx = 0..n_test`, `X_perm = X_test`; `n_repeats`
     veces: `rs.shuffle(idx)` **acumulativo** (el mismo Fisher-Yates con rechazo por máscara que
     `permutation`, D7) y `X_perm[:, c] = X_perm[idx, c]` (se permuta la columna **ya permutada**);
     `score = accuracy` (clasificador) o `R²` (regresor) de `predict(X_perm)`.
  3. `importances[c, r] = baseline − score`.
  Coste: California, 8 × 10 × 6 192 predicciones de árbol (< 20 ms). Cancelable entre columnas.
- **Vista** (D38): panel de 2 subvistas, "🌳 Importancia del modelo" (la de Python, por defecto) y
  "🔀 Importancia por permutación" (texto de método, barras con `xerr = std`, `capsize 3`, tabla con
  "Desv. Estándar", expander `permutacion`). Ambas con los textos de `componentes/mensajes.toml [importancia]`.
- **DTO** `FeatureImportanceView { method: Native|Permutation, rows: Vec<{ feature_index, name, importance,
  std: Option<f64> }> /* orden ascendente, empates por índice */, n_repeats, seed }`.
- **Svelte** `FeatureImportanceChart` (Plotly `bar` horizontal, viridis por posición, `error_x`, texto
  `{:.3f}`) + `DataTable`. **Lo reutiliza KNN (F4)** con la permutación sola.

### C13. Diagrama del árbol (`plot_tree` → SVG)

Python dibuja `plot_tree(model, feature_names, class_names (solo si tree_type == "Clasificación"),
filled=True, rounded=True, proportion=True, impurity=True)` (precision 3, `label='all'`) en una figura de
ancho × alto pulgadas, centrada al 66 % del ancho. Reglas (en `trees/*.plot_tree.rules`, con las cajas de
cada nodo en orden de pre-orden):

- **Texto de cada nodo**, líneas separadas por `\n`: nodo interno `"{feature} <= {round(umbral, 3)}"`;
  `"{criterio} = {round(impureza, 3)}"` (`gini`, `entropy`, `squared_error`, `friedman_mse`,
  `absolute_error`); `"samples = {round(100·n/n_raíz, 1)}%"`; `"value = [{np.around(v, 3)}, …]"` (fracciones
  de clase; en regresión sin corchetes); `"class = {class_names[argmax]}"` (solo clasificación). Números:
  `str(round(x, 3))` de Python = `{:.3}` correctamente redondeado y luego la representación mínima
  ("0.664", "0.0", "100.0%"); `np.around` = `rint(x·1000)/1000` (no correctamente redondeado).
- **Color** (`filled`): clasificación, `_color_brew(n_clases)[argmax]` (HSV con s 0,75, v 0,9, tonos desde
  25° cada 360/n; Iris: `#e58139`, `#39e581`, `#8139e5`) mezclado con blanco con
  `alpha = (v1 − v2)/(1 − v2)` (dos mayores fracciones; 0 si empatan: la raíz de Iris sale blanca);
  regresión, `_color_brew(1)[0]` (`#e58139`) con `alpha = (valor − min)/(max − min)` sobre los valores de
  **todos** los nodos; `int(round(alpha·c + (1 − alpha)·255))`. Borde negro, `boxstyle round`.
- **Disposición**: Buchheim/Reingold-Tilford (`sklearn/tree/_reingold_tilford.py`, ~150 líneas):
  `xy = ((x + 0,5)/max_x, (max_y − profundidad − 0,5)/max_y)`. Flechas del padre al hijo; "True  " y
  "  False" solo en las dos aristas de la raíz, en el punto medio. El tamaño de letra se reduce hasta que
  todas las cajas caben: con 27 nodos ya es ilegible, con 407 (California d8) no se ve nada (capturas 23, 40).
- **Sustituto** (D41): `TreeDiagram.svelte`, SVG propio (sin Plotly), con las mismas cajas, textos,
  colores, flechas y etiquetas True/False y la misma disposición (Buchheim portado a Rust, posiciones
  verificadas con `plot_tree.boxes_in_drawing_order[*].xy_axes`). Letra de tamaño fijo legible; zoom con
  la rueda y botones (+, −, ajustar), arrastre para moverse, tooltip por nodo (nodo, regla, impureza,
  muestras en número y %, valores en recuento, clase), resaltado del camino desde 🔮 Predicciones (C16,
  opcional). Los sliders "Ancho de figura:"/"Alto de figura:" se conservan (8–20, 6–15) y fijan el tamaño
  del lienzo (ancho × 96 px · 0,66 del contenedor, como Python). Descarga PNG (`arbol_decision.png`, a 3×)
  y SVG (D42). Leyenda: Python no tiene.
- **Core** `tree::plot::diagram(&DecisionTree, feature_names, class_names: Option) -> TreeDiagram { nodes:
  [{ id, depth, x, y, lines: Vec<String>, fill: "#rrggbb", parent, is_left }], max_x, max_y }`.

### C14. Texto de reglas (`export_text`)

- **Core** `tree::export_text(&DecisionTree, feature_names, ExportTextOptions { class_names: Option,
  show_weights, decimals: 2, spacing: 3, max_depth: 10 })` con el formato exacto de sklearn: `"|" + "   |"·d +
  "--- "`, `"{f} <= {t:.2f}"` / `"{f} >  {t:.2f}"` (dos espacios), hojas `"weights: [31.00, 0.00, 0.00]
  class: 0"` (pesos = fracción × muestras) o `"class: 0"`, regresión `"value: [152.13]"`, rama truncada
  `"truncated branch of depth {n}"` (inalcanzable con profundidad ≤ 10). Sin `class_names`, sklearn usa
  `classes_` (**códigos**: "class: 0"; D40). Dos usos: 📝 Texto (`show_weights=True`) y reglas .txt (sin pesos).
- **Svelte** `TreeRulesText.svelte` (bloque monoespaciado con descarga `arbol_texto.txt`).

### C15. Código Python del árbol (`export_tree_as_python_code`)

- Python genera `def predecir_clase(f1, f2, …)` / `predecir_valor` con el árbol como `if/else`, umbrales
  `{:.6f}` y hojas `return "{classes_[k]}"  # Probabilidad: {p:.3f}` o `return {v:.6f}`. **No es Python
  válido** en ningún fixture (`python_code.app_compiles = false`): los nombres con espacios o paréntesis no
  son identificadores, y aunque lo fueran el cuerpo está en la columna 0, fuera de la función. Además
  devuelve el código de clase ("0"), no el nombre.
- **Plantillas**: `arbol_python.py.j2` (literal, reproduce el texto de Python byte a byte, verificado) y
  `arbol_python_corregido.py.j2` (D44). La recursión del cuerpo es lógica: la genera Rust
  (`codegen::tree_python_body`) y la plantilla recibe `body`.
- **Reglas de la versión corregida** (verificada ejecutándola: mismas predicciones que sklearn en train y
  test en los 16 casos): identificadores `re.sub(r"\W+", "_", nombre).strip("_")` (Unicode permitido),
  `"x"` si queda vacío, prefijo `x_` si empieza por dígito, sufijo `_` si es palabra reservada, `_2`, `_3`… si
  se repite; cuerpo con 4 espacios por nivel desde el nivel 1; hojas `return 'setosa'` (`repr` del nombre)
  `# Probabilidad: {p:.3f}` y `return {repr(valor)}`; umbral `repr(w)`, con `w` el mayor f64 tal que
  `x <= w ⇔ f32(x) <= umbral` (sklearn compara en float32: sin esto, `longitude = -117.9` va por la otra
  rama en California). Algoritmo: `f = mayor f32 ≤ t`, `m = (f + siguiente_f32(f))/2` (exacto en f64);
  `w = m` si `f32(m) == f` (empate hacia par), si no `anterior_f64(m)`.
- **Contexto** (`trees/*.python_code.corrected_context`): `function_name, params, display_names, classifier,
  n_features, max_depth (tree_.max_depth), node_count, body`.

### C16. Camino de decisión (`show_prediction_path`)

- Python convierte la entrada a float32, recorre `decision_path` y escribe una línea por nodo interno:
  `"- {nombre} = {x:.4f} ≤ {t:.4f} ✅"` o `"- {nombre} = {x:.4f} > {t:.4f} ✅"` (el ✅ siempre). La hoja no
  se describe. Las categóricas salen como código ("Hombre_Adulto = 0.0000"). Después,
  `show_code_with_download(DECISION_PATH_CODE, "Código para generar el camino de decisión",
  "camino_decision.py")`.
- **Core** `tree::decision_path(&DecisionTree, x: &[f64]) -> DecisionPath { steps: [{ node, feature, value_f32,
  threshold, goes_left }], leaf: { node, n_samples, value: Vec<f64>, class: Option<usize> } }`.
- **Vista** (D46): las líneas de Python + la línea nueva de la hoja + la etiqueta de las categóricas
  ("Sexo = 0 (Mujer)") + tabla de probabilidades de la hoja (C9 `ProbabilityTable`).
- **Svelte** `DecisionPathList.svelte`.

---

## 4. Casos límite y comportamientos de Python

| Caso | Python | Rust (propuesta) |
|---|---|---|
| Diabetes con el tipo por defecto (Clasificación) | Entrena un clasificador de **192 clases** (floats enteros), "¡Modelo entrenado con éxito!"; 📈 Evaluación cae (`KeyError: 'accuracy'`); ROC falla; la predicción dice "Clase: 131" | D37: preseleccionar la tarea del dataset y bloquear la clasificación con objetivo continuo |
| California con Clasificación | `st.error("Error al entrenar el modelo: Unknown label type: continuous…")` (inglés) | D37 |
| 📈 Regresión sobre Iris | Entrena sobre los códigos 0/1/2, aviso; evaluación de regresión; "Valor: 1.0000" | Permitido, con el aviso (útil para comparar) |
| Cambiar el tipo de árbol sin reentrenar | La evaluación y el panel de visualización cambian de rama sobre el mismo modelo | D45: manda el modelo |
| Mover los sliders sin reentrenar | La superficie usa los valores nuevos (`st.session_state.criterion`…) y la frontera los del modelo | D45: siempre los del modelo |
| Empates entre características | Gana la primera que sortea el RNG (§8.3) | Reproducir el RNG (D36) |
| Empates de clase en una hoja ([0, 0,5, 0,5]) | `argmax` → la primera clase; el color de la caja es blanco | Igual (`leaf_argmax_ties_in_test` en los fixtures) |
| Hoja única (raíz pura o `max_depth` sin división posible) | Importancias todas 0; "La suma… = 1.0" falso | Igual; el texto se mantiene (literal) |
| Árbol profundo (> 30 nodos) | `plot_tree` ilegible | D41 (zoom) |
| Umbral que cae entre dos float32 contiguos | sklearn compara `f32(x) <= t` en f64; ONNX de skl2onnx redondea `t` a f32 al más cercano | D47 (ONNX) y C15 (código) |
| Muestra en la frontera exacta de un umbral | `f32(x) <= t` va a la izquierda | Igual: el árbol de Rust convierte X a f32 |
| Una clase ausente del train (CSV pequeños) | `classes_` solo con las vistas; `predict_proba` con menos columnas | `Predictor::classes()` (F2) |
| Misma característica en los 2 selects | No dibuja nada | Igual + aviso de F2 |
| Formulario con valores categóricos | Se pasan los códigos; el camino los imprime como `{:.4f}` | D46 |

---

## 5. Contrato IPC

### 5.1 Convenciones

Las de F1 y F2 (§5.1 de F2): comandos `async` + `spawn_blocking`, `build.rs` y `capabilities/default.json`,
ts-rs, `CommandError` en español, `model_id` (`ModelStale`), `seq` en `train_model`, evento `task-progress`.
La cancelación se comprueba entre nodos del árbol, entre columnas de la permutación y entre filas de la rejilla.

### 5.2 Estado

Igual que F2 (`LessonModels`, `TrainedModel`, `LessonId::Trees`). `EvaluationCache` añade `test_leaves` y
`proba`. El estado de interfaz (pestaña, tipo de árbol elegido, criterio, profundidad, muestras mínimas,
`test_size`, subvista D13, ancho/alto del diagrama, método de importancia, par de características,
valores del formulario) vive en el store de Svelte de la lección.

### 5.3 Tipos (nuevos o ampliados)

```rust
#[serde(rename_all = "snake_case")] pub enum TreeTask { Classification, Regression }
#[serde(rename_all = "snake_case")]
pub enum TreeCriterion { Gini, Entropy, SquaredError, FriedmanMse, AbsoluteError }   // nombres de sklearn
pub enum ModelConfig {
    Linear, Logistic { max_iter: u32 },
    Tree { task: TreeTask, criterion: TreeCriterion,
           max_depth: u8 /* 1..=10 */, min_samples_split: u8 /* 2..=10 */ },  // random_state = 42 fijo
}
// ModelConfig::task(): Tree { task } → task. Criterio incompatible con la tarea → CommandError::InvalidConfig.
// TrainNotice añade TreeTaskMismatch { message } (aviso de §2.2) — o reutiliza ModelTaskMismatch.

pub struct TreeSummary { pub node_count: u32, pub n_leaves: u32, pub max_depth: u32, pub n_features: u32 }
// TrainedModelInfo añade `tree: Option<TreeSummary>`.

pub struct TreeDiagramNode {
    pub id: u32, pub parent: Option<u32>, pub is_left: bool, pub depth: u32,
    pub x: f64, pub y: f64,                 // Buchheim, sin escalar
    pub lines: Vec<String>,                 // texto de plot_tree (C13), ya formateado
    pub fill: String,                       // "#rrggbb"
    pub leaf: bool, pub feature: Option<u32>, pub threshold: Option<f64>,
    pub impurity: f64, pub n_samples: u32, pub value: Vec<f64>, pub value_counts: Option<Vec<u32>>,
    pub class_index: Option<u32>,
}
pub struct TreeDiagram { pub nodes: Vec<TreeDiagramNode>, pub max_x: f64, pub max_y: f64,
                         pub criterion: String, pub summary: TreeSummary }

#[serde(rename_all = "snake_case")] pub enum RulesKind { View /* show_weights */, Export /* .txt */ }
pub struct RulesRequest { pub model_id: u32, pub kind: RulesKind, pub class_names: bool /* D40 */ }

#[serde(rename_all = "snake_case")] pub enum ImportanceMethod { Native, Permutation }
pub struct ImportanceRow { pub feature_index: u32, pub name: String, pub importance: f64, pub std: Option<f64> }
pub struct FeatureImportanceView { pub method: ImportanceMethod, pub rows: Vec<ImportanceRow> /* ascendente */,
                                   pub n_repeats: Option<u32>, pub baseline: Option<f64> }

pub struct PathStep { pub node: u32, pub feature: u32, pub name: String, pub value: f64 /* f32 */,
                      pub threshold: f64, pub goes_left: bool, pub display: Option<String> /* D46 */ }
pub struct DecisionPath { pub steps: Vec<PathStep>, pub leaf: u32, pub leaf_samples: u32,
                          pub leaf_value: Vec<f64>, pub leaf_class: Option<u32> }
// PredictionResult::{Class, Value} añaden `path: Option<DecisionPath>`; Class.proba = probabilidades de hoja (D46).

pub enum ExportFormat { Onnx, PythonPackage, TreeRules /* .txt */, TreePython /* .py, D44 */ }
pub enum CodeTemplate { /* F2… */ TreeViz, TreeText, DecisionPath, TreeUsage, TreePython, TreePythonLiteral }
```

### 5.4 Comandos

| Comando | Entrada | Salida | Errores | Notas |
|---|---|---|---|---|
| `train_model` (F2) | `TrainRequest` con `ModelConfig::Tree` | `TrainedModelInfo` (+ `tree`) | `InvalidTarget` (D37), `InvalidConfig`, `Cancelled` | CART en `spawn_blocking`; California d10 < 100 ms |
| `lesson_model`, `evaluate_model`, `roc_curves`, `residuals_view`, `decision_grid`, `prediction_form` (F2) | — | — | — | Sin cambios; `decision_grid` usa `refit` del árbol |
| `tree_diagram(model_id)` | `u32` | `TreeDiagram` | `ModelStale`, `NotTree` | C13; 407 nodos ≈ 120 KB |
| `tree_rules(req)` | `RulesRequest` | `String` | `ModelStale`, `NotTree` | C14 |
| `feature_importance(model_id, method, task_id)` | `u32`, `ImportanceMethod`, `u32` | `FeatureImportanceView` | `ModelStale`, `Cancelled`, `NotTree` (nativa) | C10; la permutación con progreso |
| `predict_one` (F2) | `PredictRequest` | `PredictionResult` (+ `path`) | — | C9 + C16 |
| `generate_code(model_id, template, extra)` (F2) | + plantillas de árbol | `String` | — | `TreePython` = C15 corregido |
| `export_model(model_id, format)` (F2) | + `TreeRules`, `TreePython` | `Option<String>` | `Export`, `FileSave` | Nombres: `arbol_decision.onnx`, `modelo_arbol.zip`, `decision_tree_rules.txt`, `arbol_decision.py` |
| `save_file` (F1) | — | — | — | PNG/SVG del diagrama, `arbol_texto.txt`, `.py` |

Presupuesto: cada vista < 100 ms con California (6 192 muestras de test); la más cara es
`feature_importance(Permutation)` (80 recorridos del test).

---

## 6. Fixtures de paridad

`uv run python tools/export_golden.py --only trees --only code/trees` (unos 70 s) y `--check` (84 fixtures,
0 cambios). Código en `tools/golden_trees.py` (registrado desde `export_golden.py`, reutiliza sus
helpers) y `tools/cart_reference.py`. Se escriben con las listas de escalares en una línea
(`dumps_compact`): 7,0 MB en total (California d8, el mayor, 0,9 MB).

| Ruta | Caso | Qué ejercita |
|---|---|---|
| `trees/iris_gini_d3.json` | caso completo (vistas, sliders 20 × 15, cambio de tipo sin reentrenar) | empate en la raíz (petal length/width), frontera Set3 (2 pares + misma característica), ROC multiclase, formulario |
| `trees/iris_entropy_d5_mss5.json`, `trees/iris_gini_d1.json` | entropy; tocón de 3 nodos | FMA en entropy; árbol mínimo |
| `trees/wine_gini_d4.json`, `trees/wine_entropy_d3_mss10.json` | 13 características | `min_samples_split = 10` |
| `trees/breast_cancer_gini_d3.json`, `trees/breast_cancer_entropy_d10.json` | binaria | ROC + PR binaria, árbol hasta pureza (profundidad real 7) |
| `trees/titanic_gini_d4.json` | categóricas | formulario con selects, camino con códigos |
| `trees/csv_string_target_gini_d3.json` | CSV de 30 filas | test de 9 |
| `trees/diabetes_squared_error_d3.json` | regresión completa | residuos, superficie (age/sex y bmi/s5), sliders 20 × 15 |
| `trees/diabetes_friedman_mse_d5_mss10.json`, `trees/diabetes_absolute_error_d4.json` | friedman_mse; MAE (medianas) | proxys distintos |
| `trees/diabetes_squared_error_d10.json` | 351 nodos, 99 nodos con empates | sin vistas |
| `trees/california_housing_squared_error_d3.json`, `…_d8_mss10.json` | 14 448 de train; 407 nodos | tamaño, rejilla 200 × 200, 2 casi-empates de 1 ulp |
| `trees/csv_numeric_regression_squared_error_d3.json` | CSV de 40 filas | un casi-empate de 1 ulp |
| `trees/diabetes_classification_default.json` | tipo por defecto en Diabetes | 192 clases, errores de las vistas |
| `trees/california_housing_classification_default.json` | tipo por defecto en California | error de entrenamiento |
| `trees/iris_regression_mismatch.json` | 📈 Regresión en Iris | aviso, evaluación de regresión |
| `code/trees.json` | 4 plantillas literales | texto exacto frente a jinja2 |

Contenido de cada caso (`expected`): `params` (`get_params()`), `classes`, `train_indices`/`test_indices`,
`y_test`, `y_pred`, `proba`, `tree` (todos los arrays de `tree_` y `value_counts`), `feature_importances`,
`test_leaves`, `decision_paths_test_head` (5 filas: nodos, valor f32, umbral, rama), `reference_impl`
(comparación con `cart_reference.py`, `splitter_seed` y `trace` de cada nodo: orden de sorteo, mejor proxy
por característica, `exact_ties_lost`, `near_ties_lost_rel_1e-9`), `evaluation` (como F2),
`permutation_importance` (semilla, baseline, matriz 10 × n, media, std y, si el test tiene ≤ 60 filas,
los índices barajados de la columna 0 tras cada repetición), `export_text` (vista, .txt y la versión con
nombres de clase de D40), `plot_tree` (texto, posición y color de cada caja, etiquetas True/False,
`_color_brew` y reglas), `python_code` (texto de Python, si compila, contexto literal, contexto y texto
corregidos y su verificación), `two_feature_models` (árbol 2D de cada par y su comparación),
`onnx_reference` y `export_package` (paquete D1 ejecutado de verdad, más el "Código para Usar el Modelo"
ejecutado sobre el pickle que genera), `prediction_form` y `views`.

### 6.1 Cómo desempata sklearn (y por qué la paridad puede ser exacta)

Para cada nodo (en orden de construcción: pila, el hijo **izquierdo** primero), `node_split_best`
(sklearn 1.7.2, `_splitter.pyx`):

1. **Orden de las características**: Fisher-Yates parcial sobre el array `features`, que **persiste entre
   nodos**, con `f_j = rand_int(n_drawn_constants, f_i − n_found_constants)`; `rand_int(a, b) = a +
   our_rand_r() % (b − a)`; `our_rand_r` = xorshift de 32 bits (`s ^= s << 13; s ^= s >> 17; s ^= s << 5`;
   `s = 1` si es 0; devuelve `s % 2^31`), sembrado **una vez por `fit`** con
   `RandomState(random_state).randint(0, 2^31 − 1)` (= 1608637542 con 42; MT19937 con rechazo por máscara,
   como D7). Se sortea aunque `max_features = n` (orden distinto, mismas candidatas). Las características
   constantes en el nodo (`max − min ≤ 1e-7` **en float32**) se apartan y se heredan (`n_constant_features`,
   `constant_features`, `memcpy`), y consumen números del RNG.
2. **Candidatas**: la columna se copia en **float32**, se ordena con el introsort de sklearn (mediana de 3,
   partición en 3, heapsort con `maxd = 2·⌊log2 n⌋`) y se prueban las posiciones donde
   `x[p] > f32(x[p−1] + 1e-7)`.
3. **Criterio**: el *proxy* de mejora (Gini/entropy: `−w_d·imp_d − w_i·imp_i`; MSE: `S_i²/w_i + S_d²/w_d`;
   Friedman: `(w_d·S_i − w_i·S_d)²/(w_i·w_d)`; MAE: el de impurezas con medianas). Las sumas se actualizan
   hacia delante o, si es más corto, desde el final (`reverse_reset`), y la partición final se hace sobre el
   orden que dejó la **última** característica evaluada. Ese orden de sumas decide los casi-empates.
4. **Desempate**: `if proxy > best_proxy` estricto → **gana la primera característica sorteada** (y, dentro
   de ella, el primer umbral). En Iris, la raíz: proxy −37,0 para petal length (2) y petal width (3); el
   sorteo da [0, 2, 3, 1] y gana petal length; con `random_state` 0–3 o 5 ganaría petal width.
5. **Umbral**: `x[p−1]/2 + x[p]/2` en float64 a partir de dos float32 (si sale igual a `x[p]` o infinito,
   `x[p−1]`).
6. **Hoja** si `profundidad ≥ max_depth`, `n < min_samples_split`, `n < 2`, `impureza ≤ ε` (2,2e-16) o no
   hay división válida. La impureza de cada hijo es la que calculó el padre (no se recalcula).

Medido en los fixtures: 13 de los 18 árboles tienen nodos con **empates exactos** (99 nodos en Diabetes d10) y
3 casos (4 nodos) tienen **casi-empates** de 1–2 ulp (California d8, CSV numérico, Iris en regresión), donde solo el
orden exacto de las sumas da el mismo árbol. `cart_reference.py` reproduce **todos** bit a bit en
estructura, umbrales y valores; las impurezas difieren en 1–4 ulp en entropy y MSE porque la wheel de
sklearn para macOS arm64 contrae `a·b + c` en FMA (comprobado: con FMA en Python coinciden las de
entropy). Una wheel de Linux x86-64 no usa FMA.

### 6.2 Tolerancias (`meta.tolerance` de cada caso; D6 → D36)

| Qué | Tolerancia | Por qué |
|---|---|---|
| Estructura (children, feature, n_node_samples, value_counts), umbrales | **exacta** (umbral bit a bit) | Mismo RNG, mismo orden y misma aritmética f32/f64 |
| Impureza, importancias nativas, valores de regresión | 1e-12 relativo (o absoluto si \|x\| < 1) | FMA de la wheel de referencia; sumas en el mismo orden |
| Predicciones, probabilidades, hojas, caminos | exactas | Derivadas de la estructura |
| Métricas, curvas ROC/PR | exactas (1e-12) | Predicciones exactas |
| Permutation importance | 1e-12 absoluto | RNG reproducido (bit a bit en Python); R² en f64 |
| Rejilla de frontera/superficie | ejes exactos; Z exacta | El árbol 2D también es exacto (22/22) |
| Textos (`export_text`, `plot_tree`, camino, código) | exactos | Formatos de Python reproducidos (§8.3) |
| ONNX (D47) | clases exactas; probabilidades 1e-7; regresión 1e-6 relativo | float32 |

Si Rust no alcanza la estructura exacta en algún caso, el test debe fallar y el caso anotarse en MEMORY.md:
no se relaja a "misma accuracy".

---

## 7. Contenido extraído

Literal, con comentario de origen; los textos cortos se han comprobado contra el código fuente con un
script (0 diferencias en las claves sin marca).

```
content/arboles/   intro.md (F1), mensajes.toml (NUEVO: Entrenamiento, avisos, Visualización con los
                   paneles y sliders, Características, Predicciones con el camino, Exportar con el
                   st.info de la vista previa; marcas NUEVO/CORREGIDO de D37, D38, D41–D43, D46, D49)
content/code/      visualizar_arbol.py.j2, texto_arbol.py.j2, camino_decision.py.j2 (literales, sin
                   variables), usar_modelo_arbol.py.j2 ({{ feature_names | pyrepr }}),
                   arbol_python.py.j2 (literal de export_tree_as_python_code, con `body`),
                   arbol_python_corregido.py.j2 (D44), paquete_modelo_arbol.py.j2 y
                   paquete_leeme_arbol.txt.j2 (NUEVOS, D1/D43)
```

Los textos de los componentes (evaluación, ROC, residuos, frontera, formulario, importancia) ya están en
`content/componentes/` (F2) y no se duplican. El expander de la importancia de permutación y sus textos ya
existen (`[importancia] permutacion`, `metodo_perm…`).

---

## 8. Mapeo a Rust

### 8.1 Cálculos

| Cálculo | Python | Rust | Huecos |
|---|---|---|---|
| Split | `train_test_split(0.3, 42)` | `split` (F1, D7) | — |
| CART clasificación y regresión | `DecisionTree{Classifier,Regressor}` | **Propio** `mltutor-core::tree` (linfa-trees descartado, MEMORY) siguiendo `tools/cart_reference.py` | Introsort, xorshift, `features` persistente, `reverse_reset`, partición final; MAE con medianas |
| `predict`, `predict_proba`, `apply`, `decision_path` | sklearn | Propio: X a f32, `x <= t` en f64; proba = `value` de la hoja | — |
| Importancia nativa | `feature_importances_` | `importance::native` | — |
| Permutation importance | sklearn 1.7.2 | `importance::permutation` + `sampling` (MT19937 de F1) | Semilla única (no por columna) |
| Métricas, ROC, residuos, rejilla, formulario | F2 | F2 | — |
| `export_text` | sklearn | `tree::export_text` | `{:.2}` = `"{:.2f}"` de Python |
| `plot_tree` | matplotlib | `tree::plot` (texto, color) + `tree::layout` (Buchheim) | `str(round(x, 3))` y `np.around` (§C13) |
| Código Python del árbol | f-string | `codegen::tree_python_body` + plantillas | `f64_switch_threshold` (§C15) |
| ONNX | — (no existe) | `mltutor-export::onnx::tree_ensemble_{classifier,regressor}` | D47 |
| Paquete Python | — | `mltutor-export::package` con `estimator` de árbol | Plantillas nuevas |

### 8.2 Vistas

| Vista | Comando | Componente Svelte |
|---|---|---|
| Entrenamiento | `train_model`, `lesson_model` | `TreesTrainTab.svelte` (selector F1, `ModelTypeButtons`, `Select` criterio, 3 `Slider`, botón, `Notice`) |
| Evaluación | `evaluate_model` | `ClassificationEvaluation` / `RegressionEvaluation` (F2) |
| 🌲 Árbol | `tree_diagram`, `generate_code(TreeViz)` | `TreeDiagram.svelte` (nuevo, SVG) + 2 `Slider` + `ModelCode` |
| 📝 Texto | `tree_rules(View)`, `generate_code(TreeText)` | `TreeRulesText.svelte` (nuevo) |
| 🌈 Frontera / 🌐 Superficie | `decision_grid` | `FeaturePairSelector` + `DecisionGridView` (F2) |
| 📉 ROC | `roc_curves` | `RocPrCharts` (F2) |
| 📈 Residuos | `residuals_view` | `ResidualsAnalysis` (F2) |
| 🔍 Características | `feature_importance` | `FeatureImportanceChart.svelte` (nuevo, C10) + `DataTable` |
| 🔮 Predicciones | `prediction_form`, `predict_one`, `generate_code(DecisionPath)` | `PredictionForm`, `PredictionResultCard` (F2) + `DecisionPathList.svelte` (nuevo) |
| 💾 Exportar | `export_model`, `generate_code(TreePython, TreeUsage)` | `ExportPanel` (F2, con 4 botones) + `ModelCode` + `Notice` |

Paletas (hex): `_color_brew(n)` (§C13; los colores exactos de cada caja están en los fixtures), viridis
(importancia), y las de F2.

### 8.3 Detalles numéricos y de formato que Rust debe copiar

- **X en float32** al entrenar, al predecir, en la rejilla y en la permutación (sklearn convierte con
  `check_array(dtype=float32)`); `y` en f64. Los umbrales se guardan en f64.
- `f32(x + 1e-7)` se calcula en **float32** (`x[p] + FEATURE_THRESHOLD` en C con dos `float`).
- Gini: `1 − Σc²/w²`; entropy: `−Σ (c/w)·(ln(c/w)/ln 2)`; MSE: `sq/w − (S/w)²` con `sq_izq` recalculado en
  el orden de las muestras y `sq_der = sq_total − sq_izq`; Friedman: mejora `(w_d·S_i − w_i·S_d)²/(w_i·w_d·w)`;
  MAE: `Σ|y − mediana|/w`, con la mediana de pesos unitarios (media de las dos centrales si es par).
- `impurity_improvement` (lo que decide si es hoja con ε) usa `w/w_total`; las importancias usan las
  impurezas de los nodos.
- No activar FMA en Rust (por defecto no la hay); `ln` de libm puede diferir 1 ulp entre plataformas en
  entropy: solo afecta a casi-empates (ninguno en los fixtures).
- Textos: `"{:.2f}"`, `"{:.4f}"`, `"{:.6f}"` = `format!("{:.N}")` de Rust (ambos correctamente
  redondeados); `str(round(x, 3))` = parsear `format!("{:.3}")` e imprimir con ryu; `np.around(x, 3)` =
  `(x·1000).round_ties_even()/1000` impreso con ryu.

### 8.4 Implementación de `rust-ml-engineer` y desviaciones del contrato (2026-10-15)

API estable: tabla "Public API of Phase 3" de `crates/mltutor-core/src/lib.rs` y de
`crates/mltutor-export/src/lib.rs`. Diferencias con §3 y §5:

- **`ModelConfig` no se amplía en core:** `linear::ModelConfig` sigue siendo solo de F2. El árbol se
  entrena con `tree::fit(TreeConfig, x, &Target, &TrainControl)` (o `tree::fit_boxed`). `TreeConfig`
  incluye `task`, `criterion`, `max_depth: u32`, `min_samples_split: u32` y `random_state` (42 en
  `TreeConfig::new`). La capa Tauri traduce su `ModelConfig::Tree` a este tipo.
- **Errores nuevos en `ModelError`:** `ContinuousTarget` (D37: clasificación con objetivo continuo,
  con mensaje en español) e `InvalidConfig { reason }` (un criterio de la otra tarea, `max_depth = 0` o
  `min_samples_split < 2`). La clasificación también exige entre 2 y 50 clases en train
  (`TooFewClasses`/`TooManyClasses`). Para reproducir las 192 clases de Diabetes queda
  `DecisionTree::fit_classifier(config, x, codes, ctl)`, sin límites.
- **Nombres de los tipos de core:** el contrato llama `TreeDiagramNode` a `tree::plot::DiagramNode`, que
  además trae `xy_axes` (la posición de `plot_tree` en `[0, 1]²`), y `TreeDiagram`, que añade
  `edge_labels` ("True  "/"  False" con su `ha`). `class_code` es el código del dataset. Los índices son
  `usize`, y `u32` en los DTO. El `PathStep` de core no lleva `name` ni `display`: los pone la capa Tauri.
- **`class_names` siempre por código del dataset** (`export_text`, `diagram` y código corregido). En
  `plot_tree`, Python indexa los nombres con el índice de la clase en el árbol, así que el nombre sale mal
  si falta una clase en train. Rust usa el código.
- **Impureza negativa por redondeo:** en una hoja de valor constante, la MSE (`Σy²/n − ȳ²`) puede dar
  ±1e-13. Según la FMA, la wheel de sklearn para arm64 escribe `"-0.0"` en unas hojas y `"0.0"` en otras.
  El diagrama de Rust muestra siempre `0.0`: un caso en los fixtures, California d8, caja 392. El test lo
  tolera y la impureza numérica sigue dentro de 1e-12.
- **ONNX binario:** se sigue a skl2onnx (un peso por hoja, P(clase 1) bajo `class_ids = 0`). Si se
  escriben las dos clases, onnxruntime devuelve etiquetas erróneas: 537 de 891 en Titanic.
- **Paquete de CSV con clases de texto:** Python llama a las clases "Clase i" (F1) y Rust conserva los
  nombres reales. El test usa los nombres del fixture para comparar el resto.
- **`py_float_repr` (compartido):** cuando dos cadenas de 17 dígitos vuelven al mismo float, el `{:e}`
  de Rust no siempre elige la más cercana. Ahora se elige como CPython (`1400.5000610351562`, no `…563`).
  Afectaba al código corregido de California.
- **Rendimiento:** California d10, entrenamiento en 48–49 ms (sklearn 1.7.2: 59 ms en el mismo Mac).
  Unos 33 ms son del introsort, que no se puede cambiar sin perder la paridad. El MAE es O(n²) por nodo,
  como en sklearn 1.7.2: California d3 tarda unos 3,6 s y se cancela entre características.

---

## 9. Reparto de tareas y checklist

Orden: core → paridad → comandos → UI → contenido → smoke test. Las casillas de INVENTARIO §3 se marcan al
cerrar la fila de revisión.

### 9.1 `rust-ml-engineer` (`crates/mltutor-core`, `crates/mltutor-export`)

1. [ ] `tree::{TreeConfig, DecisionTree, fit}` siguiendo `tools/cart_reference.py`: RNG (`splitter_seed` +
   xorshift), introsort, `features`/`constant_features` persistentes, criterios Gini, entropy, MSE,
   Friedman y MAE, `reverse_reset`, partición final, pila con el hijo izquierdo primero. Cancelación por
   nodo. Test: arrays de `trees/*.tree` (estructura y umbral exactos, impureza 1e-12) en los 18 casos entrenados y en
   `two_feature_models`; `reference_impl.splitter_seed`; los `trace` para depurar.
2. [ ] `impl Predictor for DecisionTree` (`predict` con X a f32, `predict_proba`, `classes`, `refit`) +
   `apply` y `decision_path`. Tests: `y_pred`, `proba`, `test_leaves`, `decision_paths_test_head`.
3. [ ] `importance::{native, permutation}`. Tests: `feature_importances` (1e-12) y `permutation_importance`
   (matriz completa, 1e-12; índices barajados de la columna 0 en Iris/CSV). Documentar en la API que KNN
   (F4) lo reutiliza.
4. [ ] `tree::export_text` y `tree::plot::diagram` + `tree::layout::buchheim`. Tests: `export_text.*` exactos;
   `plot_tree.boxes_in_drawing_order` (texto exacto, color exacto, `xy_axes` 1e-12).
5. [ ] `codegen::tree_python_body` (literal y corregido) + `f64_switch_threshold` + contextos tipados.
   Tests: `python_code.app_text` (literal) y `python_code.corrected_text` exactos; `code/trees.json`.
6. [ ] `mltutor-export`: `onnx::tree_ensemble_classifier` (salidas `label` int64 + `probabilities`, sin
   ZipMap) y `tree_ensemble_regressor`, opset 15 / ai.onnx.ml 3, umbrales = mayor f32 ≤ t (D47), nombres de
   nodo únicos (MEMORY); `onnx_model` y `onnx_file_name` con la rama de árbol. Tests: estructura frente a
   `onnx_reference.nodes`, salidas frente a `label`/`probabilities`/`variable` (evaluador propio), y en el
   job de F6 con onnxruntime sobre train + test (0 discrepancias).
7. [ ] `package` para árboles (`modelo.json` igual a `export_package.model_json` salvo flotantes 1e-12;
   script = `export_package.script`; LEEME con `paquete_leeme_arbol.txt.j2`).
8. [ ] Benchmark (`#[ignore]`): California d10 entrenar + todas las vistas < 100 ms cada una.

### 9.2 `tauri-ui-engineer` (`src-tauri` + `ui`)

Backend:
1. [ ] `ModelConfig::Tree`, `TreeSummary`, DTO de §5.3, errores `InvalidConfig`, `NotTree`; D37 en
   `train_model` (`InvalidTarget` con los textos de `arboles/mensajes.toml`).
2. [ ] Comandos `tree_diagram`, `tree_rules`, `feature_importance` (build.rs + capabilities) y ampliaciones de
   `predict_one`, `generate_code`, `export_model`.
3. [ ] `--smoke-test`: Iris gini, Diabetes squared_error y Cáncer, todas las vistas; `--screenshot --scenario`
   para las capturas de §0.

Frontend:
4. [ ] Store de la lección (`lessons/trees/state.svelte.ts`): tipo preseleccionado por la tarea del dataset
   (D37), criterio (las opciones cambian con el tipo), profundidad, muestras mínimas, `test_size`, subvista,
   ancho/alto, método de importancia.
5. [ ] `TreeDiagram.svelte` (C13, D41/D42): SVG, zoom/arrastre, tooltip, descarga PNG/SVG; tests de vitest
   (número de cajas, colores y textos desde el fixture).
6. [ ] `TreeRulesText.svelte`, `FeatureImportanceChart.svelte`, `DecisionPathList.svelte`.
7. [ ] Pestañas 1–6 en `ui/src/lib/lessons/trees/` reutilizando los componentes de F2 y `ModelBanner` (D32).
8. [ ] Descargas con los nombres de Python (`arbol_decision.png`, `arbol_texto.txt`,
   `decision_tree_rules.txt`, `arbol_decision.py`, `visualizar_arbol.py`, `texto_arbol.py`,
   `camino_decision.py`, `usar_modelo.py`, `frontera_decision.png`, `matriz_confusion.png`…).
9. [ ] Capturas Rust de las pantallas de §0 y comparación.

### 9.3 `ml-lesson-author`

1. [ ] Cargar `content/arboles/mensajes.toml` y las plantillas sin reescribirlas.
2. [ ] Textos nuevos marcados (D37, D38, D41, D46, D49): revisar redacción; si se aprueba D40, la versión
   corregida de `texto_arbol.py.j2` (pasar `class_names=`).

### 9.4 Revisión

`test-runner` → `rust-reviewer` → `ml-correctness-reviewer` (CART exacto, permutación, ONNX, código
corregido).

---

## 10. Decisiones para el usuario

Se numeran a continuación de D35. Si no se dice lo contrario, se aplica la recomendación.

| # | Decisión | Recomendación |
|---|---|---|
| D36 | **Paridad del CART** (concreción de D6): sklearn desempata con su RNG; con otro orden sale otro árbol | **Paridad exacta**: reproducir el RNG, el introsort y el orden de las sumas (`tools/cart_reference.py`, verificado en 40 árboles). Estructura y umbrales exactos; impureza 1e-12 |
| D37 | **Tipo de árbol por defecto**: Python empieza siempre en Clasificación; en Diabetes entrena 192 clases y la evaluación cae; en California da un error en inglés | Preseleccionar la tarea del dataset al entrar; bloquear la clasificación con objetivo continuo con un mensaje en español (como D26); permitir 📈 Regresión en datasets de clasificación, con el aviso |
| D38 | **Importancia por permutación**: el inventario la pide en Árboles, pero Python solo muestra la nativa | Panel con "🌳 Importancia del modelo" (por defecto, idéntica a Python) y "🔀 Importancia por permutación" (C10, mismos textos que KNN) |
| D39 | **RNG de la permutación** | Reproducir sklearn 1.7.2 exactamente (semilla única, shuffle acumulativo): paridad 1e-12 en lugar de estadística |
| D40 | **"class: 0"** en 📝 Texto: `export_text` sin `class_names` muestra códigos | Mostrar los nombres de clase (`view_text_with_class_names_D40`) y pasar `class_names=` en `texto_arbol.py.j2`; las reglas .txt también |
| D41 | **Diagrama del árbol**: `plot_tree` encoge la letra hasta hacerla ilegible (27 nodos o más) | SVG propio con la misma disposición, textos y colores, letra fija, zoom, arrastre y tooltip; se conservan los sliders de ancho y alto |
| D42 | **Descarga del diagrama** | PNG (`arbol_decision.png`, como Python) y además SVG |
| D43 | **Exportar** (concreción de D1/D28): Python ofrece Pickle, reglas .txt y código .py | Cuatro botones: "💾 Exportar a ONNX", "📦 Paquete Python (.zip)" (reconstruye el árbol y genera `decision_tree_model.pkl` con el diccionario de Python, para que el "Código para Usar el Modelo" siga funcionando; verificado), "📝 Exportar Reglas como Texto" y "🐍 Exportar como Código Python". Sin pickle directo |
| D44 | **Código Python del árbol**: no compila nunca (nombres no válidos, cuerpo fuera de la función), devuelve el código de clase y redondea umbrales a 6 decimales | Versión corregida (`arbol_python_corregido.py.j2`, §C15): verificada, da las mismas predicciones que sklearn en train y test. La vista previa y la descarga usan la corregida |
| D45 | **Hiperparámetros y tarea tras entrenar**: la superficie usa los sliders actuales y la rama de las pestañas 2–6 la decide el botón actual | Mandan el modelo entrenado y su configuración (extiende D32) |
| D46 | **Predicción de un árbol**: sin probabilidades; el camino no dice a qué hoja llega y muestra códigos en las categóricas | Añadir la tabla de probabilidades de la hoja (clasificación), una línea final con la hoja y la etiqueta de las categóricas ("Sexo = 0 (Mujer)") |
| D47 | **Umbrales en ONNX**: skl2onnx redondea al float32 más cercano (de 0 a 52 umbrales por árbol difieren del correcto); en los datos no hay discrepancias, pero puede haberlas con otros valores | Escribir el mayor float32 ≤ umbral (`sklearn_threshold` de skl2onnx, que su conversor de árboles no aplica): equivalencia exacta con sklearn |
| D48 | **Profundidad y criterio en la frontera**: Python la reentrena con 2 características | Igual que Python (paridad; C8 ya lo hace) |
| D49 | **"👉 Ve a la pestaña '🌲 Visualización'"**: la pestaña es 📉 Visualización | Corregir el texto |
| D50 | **Capturas Qt de F3** (6,0 MB) y **fixtures** (7,0 MB) | Versionarlos, como en F2 |
| D51 | **`tools/cart_reference.py`** (Python) | Mantenerlo como especificación ejecutable del CART: lo usan los fixtures y sirve para depurar diferencias de Rust nodo a nodo |

---

## 11. Riesgos

| Riesgo | Mitigación |
|---|---|
| Reproducir el CART bit a bit es delicado (introsort, `features` persistente, `reverse_reset`, f32) | `cart_reference.py` como especificación; tests por array y la `trace` por nodo para localizar la primera divergencia |
| Casi-empates de 1 ulp (3 casos, 4 nodos) dependen del orden de las sumas | Copiar el orden exacto; tests específicos en California d8 y el CSV numérico |
| `ln` de libm distinto entre plataformas (entropy) | Solo afecta a casi-empates; ninguno en los fixtures; CI en 3 sistemas |
| FMA en la referencia (macOS arm64) | Tolerancia 1e-12 en impurezas; si se regeneran los fixtures en x86-64, las impurezas cambian 1–4 ulp (no la estructura) |
| SVG de 400+ nodos en WebKitGTK | Render virtual (solo lo visible) si hace falta; `TreeDiagram` sin Plotly |
| El paquete Python usa `sklearn.tree._tree.Tree.__setstate__` (API privada) | El script comprueba las predicciones (assert) y lo ejecuta el job de F6 con sklearn fijado; si cambia en una versión futura, falla con un mensaje claro |
| Permutación en California (80 × 6 192 recorridos) | < 20 ms en Rust; cancelable por columna |
| Fixtures grandes (7 MB) | `dumps_compact`; California d8 sin vistas |

---

## 12. Hallazgos de la versión Python

1. El tipo de árbol empieza siempre en **Clasificación**: Diabetes entrena 192 clases (y 📈 Evaluación cae con
   `KeyError: 'accuracy'`, la ROC con "inconsistent numbers of samples"); California falla en inglés.
2. sklearn desempata por el orden aleatorio de las características (raíz de Iris). Los empates exactos son
   frecuentes y hay casi-empates de 1 ulp.
3. Las impurezas de la wheel arm64 de sklearn llevan FMA; la estructura no cambia.
4. `export_tree_as_python_code` nunca produce Python válido (nombres, sangría) y devuelve códigos de clase.
5. 📝 Texto muestra "class: 0" (códigos); las reglas .txt no llevan pesos.
6. `plot_tree` es ilegible a partir de unos 25 nodos; el tamaño de letra es automático.
7. La importancia por permutación existe en el código compartido, pero en Árboles nunca se ve.
8. `permutation_importance` (sklearn ≥ 1.5) usa una sola semilla para todas las columnas y baraja de forma
   acumulativa.
9. La superficie de regresión usa los sliders actuales, no los del modelo; la rama de evaluación la decide
   el botón actual.
10. La predicción de un árbol no muestra probabilidades y el camino imprime códigos ("Sexo = 0.0000").
11. skl2onnx redondea los umbrales de los árboles al float32 más cercano (puede desviar muestras).
12. "👉 Ve a la pestaña '🌲 Visualización'": la pestaña se llama 📉 Visualización.
13. El botón es "Entrenar Modelo" (sin emoji ni key), a diferencia de Regresión.
14. El bucle de las pestañas está duplicado y el CSS de las pestañas no tiene efecto en Qt.

---

## 13. Cambios propuestos en otros documentos

No se han editado (los aplica quien cierre la fase).

**INVENTARIO.md §3**: "Características: importancia nativa (y por permutación, D38)"; "Predicciones: …
camino de decisión, probabilidades de la hoja (D46)"; "Exportar: ONNX (TreeEnsemble*), paquete Python ZIP
(D43), reglas .txt, código Python corregido (D44), imagen PNG/SVG del árbol (en 🌲 Árbol) y código de uso".
Quitar "📦 Pickle" y "código de entrenamiento generado" (no existe en la lección de Python). §5: la
permutación de KNN usa C10 de F3. §8: añadir `tree_visualizer.create_tree_visualization`,
`render_tree_visualization`, `tree_visualization.py` entero, `ui.display_feature_importance`,
`ui.display_model_export_options`, `ui.export_model_pickle`, `ui.export_modelo_onnx`,
`utils.generate_model_code`, `model_training.predict_sample` y `DECISION_BOUNDARY_CODE` como código muerto
(confirmado: solo los importa `ui.py` o `app.py` sin llamarlos; `utils.export_model_onnx` solo lo usa
`--verify-models`).

**PLAN.md, Fase 3**: enlazar este documento y `capturas/qt/fase3/`; añadir D36–D51 (como D19–D35, con enlace
a §10); cambiar "Clasificación (linfa-trees o propio)" por "CART propio con paridad exacta (D36)"; añadir
permutation importance (C10) y el paquete Python.

**tests/parity/README.md**: ya actualizado con `trees/` y `code/trees.json`.
