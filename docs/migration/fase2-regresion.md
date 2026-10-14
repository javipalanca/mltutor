# Fase 2 — Lección piloto 📊 Regresión (lineal y logística)

Plan de port de `migration-porter` (2026-10-14). Cubre INVENTARIO §4 (📊 Regresión), la parte de §0
(diálogos de exportación), §1 (descripción del dataset en Entrenamiento), §7 (código Python
generado) y §9 de estas pantallas. Referencia: rama `codex/desktop-migration-study`, la versión Qt.

Esta lección fija los **componentes compartidos** (§3) que reutilizarán Árboles (F3), KNN (F4) y
Redes (F5): métricas, matriz de confusión, informe por clase, ROC/PR, distribución de
probabilidades y umbrales, residuos, frontera de decisión y superficie 2D, formulario de
predicción, importancia, exportación (D1) y código Python generado.

**El contrato IPC (§5) es la frontera.** `rust-ml-engineer` trabaja en `crates/mltutor-core` y
`crates/mltutor-export`; `tauri-ui-engineer`, en `src-tauri` y `ui`. Las listas de tareas están en §9.

Índice: [0 Capturas](#0-capturas-de-referencia-qt) · [1 Estructura](#1-estructura-de-la-lección) ·
[2 Pestañas](#2-pestañas) · [3 Componentes compartidos](#3-componentes-compartidos) ·
[4 Casos límite](#4-casos-límite-y-comportamientos-de-python) · [5 Contrato IPC](#5-contrato-ipc) ·
[6 Fixtures](#6-fixtures-de-paridad) · [7 Contenido](#7-contenido-extraído) ·
[8 Mapeo a Rust](#8-mapeo-a-rust) · [9 Reparto y checklist](#9-reparto-de-tareas-y-checklist) ·
[10 Decisiones](#10-decisiones-para-el-usuario) · [11 Riesgos](#11-riesgos) ·
[12 Hallazgos](#12-hallazgos-de-la-versión-python) · [13 Cambios propuestos](#13-cambios-propuestos-en-otros-documentos)

## 0. Capturas de referencia (Qt)

En `docs/migration/capturas/qt/fase2/`, generadas con

```sh
uv run python tools/capture_qt.py docs/migration/capturas/qt/fase2 --phase 2 --quantize
```

(ventana real, no `offscreen`; unos 75 s). `--quantize` reescribe cada PNG con una paleta de 256
colores (Pillow). `NNN.png` es la página completa; solo 02, 05 y 33 tienen además
`NNN_ventana.png` (con la barra lateral). Total: 6,8 MB. Todas las vistas de esta lección son
matplotlib (`QLabel`), así que salen completas. Ojo: tras un paso `expand_all`, los expanders
siguen abiertos en las capturas siguientes (el estado se recuerda por id).

| Archivo | Pantalla |
|---|---|
| `00_evaluacion_sin_entrenar`, `00b_visualizacion_sin_entrenar` | Pestañas 2 y 3 sin modelo (avisos) |
| `01_entrenamiento_logistica_iris`, `02_entrenado_logistica_iris(_ventana)` | 🏋️ Entrenamiento con Logística seleccionada, antes y después de "🚀 Entrenar Modelo" |
| `03_evaluacion_clasificacion_iris`, `04_…_expandido` | 📈 Evaluación de clasificación multiclase |
| `05_roc_multiclase_iris(_ventana)`, `06_…_expandido`, `07_roc_micro_sin_clases_iris` | 📉 Curva ROC multiclase (por defecto y con solo micro, sin curvas por clase) |
| `08_frontera_multiclase_iris`, `09_frontera_petalos_iris`, `10_frontera_misma_caracteristica` | 🌈 Frontera (par por defecto, pétalos, misma característica → no dibuja nada) |
| `11_probabilidades_multiclase_iris` | 📊 Distribución de probabilidades multiclase |
| `12_coeficientes_logistica_iris`, `13_…_expandido`, `14_coeficientes_clase_versicolor` | 🔍 Coeficientes logística multiclase (clase 0 y clase 1) |
| `15_prediccion_formulario_iris`, `16_prediccion_resultado_iris` | 🔮 Formulario y resultado (con tabla de probabilidades) |
| `17_exportar`, `18_exportar_codigo`, `19_exportar_pickle` | 💾 Exportar: estado inicial, código generado y botón de descarga del pickle |
| `20_entrenado_cancer` … `25_probabilidades_umbrales_cancer_expandido` | Binaria: evaluación, ROC + PR, frontera RdBu, probabilidades y análisis de umbrales |
| `26_prediccion_formulario_titanic`, `27_…_resultado_titanic`, `28_coeficientes_logistica_titanic` | Formulario con selects de metadatos (Sexo, Puerto…) y coeficientes binarios |
| `29_entrenamiento_lineal_diabetes` … `40_prediccion_resultado_diabetes` | Lineal: evaluación de regresión, residuos, superficie (por defecto y bmi/s5), coeficientes, predicción |
| `41_aviso_lineal_en_clasificacion_iris`, `42_aviso_logistica_en_regresion_diabetes` | Avisos de tipo de modelo |
| `43_logistica_entrenada_en_diabetes`, `44_error_roc_logistica_diabetes` | La logística **entrena** en Diabetes (192 clases) y la vista ROC falla con la página de error |

No se ha capturado: los diálogos nativos de guardar (pickle, PNG, código) ni el spinner.

---

## 1. Estructura de la lección

`mltutor/apps/linear_regression_app.py::run_linear_regression_app`.

1. Cabecera: `st.header("📊 Regresión")`, "Aprende sobre regresión lineal y logística de forma
   interactiva" y el expander "ℹ️ ¿Qué es la Regresión?" (cerrado; `content/regresion/intro.md`).
   Ya está en F1 (`content/inicio/navegacion.toml`).
2. 7 pestañas (botones `tab_lr_0..6`, estado `active_tab_lr`, 0 por defecto) y `---`:
   📊 Datos · 🏋️ Entrenamiento · 📈 Evaluación · 📉 Visualización · 🔍 Coeficientes ·
   🔮 Predicciones · 💾 Exportar. El CSS de `tab-button-lr` no tiene efecto en Qt.
3. `run_dataset_tab(active_tab_lr)` (F1): pestaña 0 = exploración completa; pestaña 1 = cabecera
   "📊 Selección y Preparación de Datos" + selector; pestañas ≥ 2 = `st.info("📊 **Dataset actual:**
   {selected_dataset}")` + `---`.
4. Contenido de la pestaña (§2) y navbar (F1, §1.5: "Volver a …"/"Continuar a …", con
   **Coeficientes** en lugar de Características).

### 1.1 Estado de sesión (Python)

| Clave | Escribe | Contenido |
|---|---|---|
| `model_type_lr` | botones de tipo | `"Linear"` (por defecto) o `"Logistic"` |
| `dataset_option_lr` | Entrenamiento | último dataset cargado (no se usa para nada más) |
| `is_trained_lr` | Entrenamiento | siempre `False`, **no se usa** |
| `model_lr`, `X_train_lr`, `X_test_lr`, `y_train_lr`, `y_test_lr` | "🚀 Entrenar Modelo" | modelo y split (DataFrames/Series con el índice original) |
| `feature_names_lr`, `class_names_lr`, `class_names` | ídem | nombres legibles y de clase |
| `max_iter` | ídem (solo logística) | lo lee la frontera de decisión |
| `task_type_lr` | ídem | tarea **del dataset** ("Clasificación"/"Regresión"), no la del modelo |
| `model_trained_lr` | ídem | `True` tras entrenar; **no se borra al cambiar de dataset** |
| `viz_type` | `create_button_panel` | global, compartido entre lecciones (D13 de F1: en Rust, por lección) |
| `viz_feature1`, `viz_feature2` | selects de la frontera | nombres legibles; compartidos por frontera y superficie |
| `feature_{i}` | formulario | valores del formulario (compartidos con las otras lecciones) |

Flujo de entrenamiento: `load_data(selected_dataset)` → `preprocess_data(X, y, test_size)` =
`train_test_split(test_size, random_state=42)` sin estratificar (D7) →
`LinearRegression()` o `LogisticRegression(max_iter, random_state=42)` sobre los datos **sin
escalar**. Spinner "Entrenando modelo...". Nada se reentrena automáticamente: cambiar el dataset,
el tipo, `test_size` o `max_iter` no invalida el modelo anterior (ver D32).

---

## 2. Pestañas

### 2.1 📊 Datos (pestaña 0)

Exactamente la de F1 (`DatasetTab.svelte`). Navbar: (—, "Continuar a Entrenamiento").

### 2.2 🏋️ Entrenamiento (pestaña 1)

Orden: selector de F1 → `st.header("Configuración del Modelo")` → `### Información del Dataset` +
caja `info-box` (fondo #e3f2fd) con `dataset_info.description` (en un CSV: "Dataset personalizado:
{n} muestras, {f} características, {k} clases" o "…, regresión") → `### Tipo de Modelo Lineal` →
2 botones → aviso de tipo → `### Parámetros del Modelo` (2 columnas) → botón de entrenar →
resultado → navbar ("Volver a Datos", "Continuar a Evaluación").

| Control | Tipo | Etiqueta exacta | Rango / opciones | Por defecto | key | Efecto |
|---|---|---|---|---|---|---|
| Tipo lineal | botón (ancho completo, col. 1) | `📈 Regresión Lineal` | — | primary si `model_type_lr == "Linear"` | `btn_linear` | `model_type_lr = "Linear"`; help "Para predecir valores numéricos continuos" |
| Tipo logística | botón (col. 2) | `🏷️ Regresión Logística` | — | primary si `"Logistic"` | `btn_logistic` | `model_type_lr = "Logistic"`; help "Para predecir categorías o clases" |
| Iteraciones | slider entero (col. 1, **solo logística**) | `Máximo de Iteraciones:` | 100 … 2000, paso 100 | 1000 | sin key (por sitio de llamada) | `LogisticRegression(max_iter)`; help "Número máximo de iteraciones para el optimizador de regresión logística." |
| (en lineal) | `st.info` (col. 1) | "La regresión lineal no requiere configuración adicional de iteraciones." | | | | |
| Tamaño de test | slider real (col. 2) | `Tamaño del Conjunto de Prueba:` | 0,1 … 0,5, paso 0,05 | 0,3 | sin key | `test_size`; help "Proporción de datos que se usará para evaluar el modelo." |
| Entrenar | botón primary | `🚀 Entrenar Modelo` | — | — | `train_lr_button` | Entrena y guarda el estado de §1.1 |

Avisos (`st.warning`, debajo de los botones de tipo): dataset de clasificación + Lineal →
"Este dataset parece ser más adecuado para Regresión Logística. La selección actual podría no
ofrecer resultados óptimos."; regresión + Logística → "… más adecuado para Regresión Lineal. …".
Resultado: `st.success("¡Modelo entrenado exitosamente!")`, `st.error("Error al entrenar el
modelo: {e}")` o `st.error("Por favor, carga un dataset válido primero.")`. Si falla la carga:
`st.error("Error al cargar el dataset: {e}")` (y no aparecen los botones de tipo).

### 2.3 📈 Evaluación (pestaña 2)

`st.header("Evaluación del Modelo")`. Sin modelo: `st.warning("Primero debes entrenar un modelo en la
pestaña '🏋️ Entrenamiento'.")`. Con modelo: `show_detailed_evaluation(y_test, model.predict(X_test),
class_names, "Regresión" si Linear, si no "Clasificación")` → componentes **C1** (clasificación) o
**C2** (regresión) + **C3** (interpretación). La tarea de la evaluación la decide el **modelo**,
no el dataset. Navbar: ("Volver a Entrenamiento", "Continuar a Visualización").

### 2.4 📉 Visualización (pestaña 3)

`st.header("Visualizaciones")`. Sin modelo: `st.info("Entrena un modelo primero para ver las
visualizaciones.")`. Panel de subvistas (`create_button_panel`, estado `viz_type`):

| Modelo | Botón (valor, key) | Contenido |
|---|---|---|
| Lineal | `📈 Análisis de Residuos` (Residuos, `viz_residuals`) | `### 📊 Gráfico de Predicciones vs Valores Reales` + **C7a**, `### 📈 Análisis de Residuos` + **C7b** (test) |
| Lineal | `🌐 Superficie de Predicción` (Superficie, `viz_surface`) | **C8** en modo superficie, con `LinearRegression()` reentrenado en 2 características de **train** |
| Logística | `📉 Curva ROC` (ROC, `viz_roc`) | **C4** con `predict_proba(X_test)` |
| Logística | `🌈 Frontera de Decisión` (Frontera, `viz_boundary`) | **C8** en modo frontera, con `LogisticRegression(max_iter, random_state=42)` en 2 características de train. Si falla: `st.error("Error al mostrar la visualización de frontera de decisión: {e}")` + `st.info` con los 3 requisitos + `st.exception` |
| Logística | `📊 Distribución de Probabilidades` (Probs, `viz_prob`) | `### 📊 Distribución de Probabilidades` + **C5** (y **C6** si es binaria) |

Navbar: ("Volver a Evaluación", "Continuar a Coeficientes").

### 2.5 🔍 Coeficientes (pestaña 4) — específica de la lección

`st.header("Coeficientes del Modelo")`. Sin modelo: `st.info("Entrena un modelo primero para ver los
coeficientes.")`.

1. Expander "ℹ️ ¿Cómo interpretar los coeficientes?" (cerrado): `coeficientes-lineal.md` o
   `coeficientes-logistica.md`.
2. Fila mostrada: lineal → `coef_`; logística binaria → `coef_[0]`; multiclase →
   `st.info("**Nota:** Este es un modelo de clasificación multiclase con {k} clases. Mostrando los
   coeficientes para la primera clase ({clase0}).")` + selectbox **sin key** `Selecciona la clase
   para mostrar coeficientes:` (opciones = índices 0..k-1 mostrados con el nombre de clase,
   índice 0) → `coef_[clase]`. Si la longitud no coincide con las características, 3 `st.error` y
   termina.
3. Tabla `coef_df` ordenada por |coef| descendente (`sort_values`, quicksort: sin estabilidad en
   empates). Columnas derivadas:
   - Efecto: "📈 Positivo" si coef > 0, si no "📉 Negativo".
   - Importancia (solo lineal): "🔥 Alta" si |c| > cuantil 0,75 de |coef|, "🔶 Media" si > 0,25, si
     no "🔹 Baja" (cuantiles lineales de pandas).
   - Logística: Odds Ratio = e^c, Cambio = (OR − 1)·100, Fuerza: OR > 7,4 "🔥 Muy Fuerte", > 2,7
     "🔥 Fuerte", > 1,6 **o** < 0,625 "🔶 Moderado", si no "🔹 Débil" (**asimétrica**: un
     coeficiente −2,5 sale "Moderado"; D27).
   - `### 📊 Tabla de Coeficientes`, `hide_index`. Lineal: Característica, Coeficiente (`{:.4f}`),
     Efecto, Importancia. Logística: Característica, Coeficiente (`{:.4f}`), Odds Ratio (`{:.3f}`),
     Cambio en Odds (`{:+.1f}%`), Efecto, Fuerza del Efecto. Después el expander "📖 ¿Cómo leer
     esta tabla?" (`tabla-lineal.md` / `tabla-logistica.md`).
4. `### 🎯 Intercepto del Modelo`: métrica "Intercepto" `{:.4f}` (`intercept_[0]` si es un array:
   **en multiclase siempre la clase 0**, aunque se elija otra; D27) y, en logística, "Odds Ratio
   Base" `{:.3f}`. `st.info` de interpretación (`mensajes.toml [coeficientes]`).
5. `### 📈 Visualización de Coeficientes`: barras horizontales en el orden de la tabla (la primera
   arriba… en matplotlib la fila 0 queda **abajo**; ver captura 12), color #4ecdc4 (positivo) o
   #ff6b6b (resto), alpha 0,7, borde negro; etiqueta `{:.3f}` a la derecha/izquierda de cada barra
   (desplazada 1 % del rango); línea vertical en 0; rejilla en x; leyenda "Efecto Positivo / Efecto
   Negativo" arriba a la derecha; título "Coeficientes del Modelo (ordenados por importancia)",
   eje x "Valor del Coeficiente"; 12 × max(6, 0,4·n) pulgadas, al 80 % del ancho.
6. `### 🔍 Análisis Detallado de Importancia`: hasta 3 positivas y 3 negativas (en el orden de la
   tabla) con su intensidad (logística: OR > 3 / > 1,5; negativas OR < 0,33 / < 0,67; lineal:
   |c| > cuantil 0,8 / 0,6) y el **número = posición original de la característica + 1**, no el
   puesto (D27). Resumen ejecutivo (`st.success`) con el factor de mayor |coef|; en logística,
   balance de signos (`st.warning`/`st.info`).
7. `#### 💡 **Recomendaciones para la Acción:**`: top 3, "**{f}:** {Aumentar|Reducir} esta
   característica para {incrementar|decrementar} la {probabilidad positiva|valor predicho}" (el
   "la valor predicho" es literal).
8. `st.warning` de coeficientes extremos si alguno supera el cuantil 0,95 de |coef| (con n ≥ 2
   siempre hay al menos uno: la advertencia **sale casi siempre**).

Navbar: ("Volver a Visualización", "Continuar a Predicciones").

### 2.6 🔮 Predicciones (pestaña 5)

`st.header("🔮 Predicciones con Nuevos Datos")`. Sin modelo: `st.info("Entrena un modelo primero para
hacer predicciones.")`. Con modelo: **C9** con `X_train`, `class_names`, `task_type_lr` (¡del
dataset!) y `selected_dataset` actual (para los metadatos). Navbar: ("Volver a Coeficientes",
"Continuar a Exportar").

### 2.7 💾 Exportar (pestaña 6)

`st.header("Exportar Modelo")`. Sin modelo: `st.info("Entrena un modelo primero para exportarlo.")`.
`display_model_export_options(model)`: 2 columnas.

| Columna | Botón | key | Al pulsar |
|---|---|---|---|
| Izquierda | `📄 Generar Código` | `generate_code_lr` | `st.code(generate_regression_code())` (`content/code/regresion_codigo.py.j2`: guarda y carga un pickle; no importa numpy y usa `model`, que no está definido; el nombre del archivo es `modelo_regresion_logistica.pkl` también en lineal) + `st.download_button("📥 Descargar código", regression_code.py)` |
| Derecha | `📥 Descargar Modelo (Pickle)` | `download_pickle_lr` | `st.download_button("Descargar modelo.pkl", linear_regression_model.pkl)` (también en logística). En Qt abre el diálogo de guardar |

No hay ONNX en esta lección. En Rust (D1, D28): ONNX + paquete Python. Navbar: ("Volver a
Predicciones", —).

---

## 3. Componentes compartidos

Cada componente tiene: **core** (función pura en `mltutor-core`, sin UI), **DTO** (respuesta del
comando), **Svelte** (componente en `ui/src/lib/components/ml/`) y **quién lo usa**. Las
interfaces son genéricas: reciben vectores de etiquetas, puntuaciones o valores, nunca un
modelo concreto. Los modelos implementan un trait común (§3.0).

### 3.0 Trait de modelo y almacén

```rust
// mltutor-core::model
pub enum Prediction { Classes(Vec<usize>), Values(Vec<f64>) }
pub trait Predictor: Send + Sync {
    fn task(&self) -> TaskType;                       // la del MODELO
    fn n_features(&self) -> usize;
    fn predict(&self, x: ArrayView2<f64>) -> Result<Prediction, ModelError>;
    fn predict_proba(&self, x: ArrayView2<f64>) -> Option<Result<Array2<f64>, ModelError>>; // columnas = clases del modelo
    /// Mismo algoritmo e hiperparámetros, entrenado con otras columnas (frontera/superficie).
    fn refit(&self, x: ArrayView2<f64>, y: &Target) -> Result<Box<dyn Predictor>, ModelError>;
}
```

Linear, Logistic (F2), CART (F3), KNN (F4) y la red (F5, vía `mltutor-nn`) lo implementan.
`src-tauri` guarda un `TrainedModel { id, lesson, dataset, split, model: Arc<dyn Predictor>,
config, features, class_names }` por lección (§5.2).

### C1. Evaluación de clasificación (`show_detailed_evaluation`, "Clasificación")

- **Core** `metrics::classification(y_true: &[usize], y_pred: &[usize], n_classes) ->
  ClassificationEvaluation`: `accuracy`; `per_class: Vec<ClassMetrics { precision, recall, f1,
  support }>` (labels = 0..n_clases, como la app, aunque una clase falte en test; `zero_division`
  = 0 con aviso, como sklearn por defecto); `macro_avg`, `weighted_avg`; `confusion: Matrix`
  (filas = real, columnas = predicción); `correct_head`/`incorrect_head` (las 10 primeras de cada
  grupo, con su posición en el test, real y predicción).
- **Vista** (en este orden): 4 métricas en 4 columnas ("Exactitud (Accuracy)", "Precisión media",
  "Exhaustividad media (Recall)", "F1-Score medio", `{:.4f}`, weighted, con su `help`) + insignia
  bajo Accuracy (≥ 0,9 success "Excelente", ≥ 0,8 info "Muy bueno", ≥ 0,7 warning "Bueno", si no
  warning "Necesita mejora"; Qt descarta el `icon`, D34) y bajo F1 (≥ 0,8 "🌟 Excelente balance",
  ≥ 0,7 "👍 Buen balance", si no "⚠️ Balance mejorable") → `### ⚖️ Matriz de Confusión`, 2 columnas:
  izquierda, heatmap `Blues` con anotaciones enteras, ejes "Predicción"/"Real", título "Matriz de
  Confusión", ticks = nombres de clase, colorbar, 6×5 in en `[1,3,1]`; enlace "📥 Descargar matriz
  de confusión" (PNG) y código "Código para generar la matriz de confusión"
  (`matriz_confusion.py.j2`); derecha, informe por clase (filas = clases; columnas precision,
  recall, f1-score, support; D33) y expander "📊 Explicación de métricas" → `### Visualización de
  Predicciones`: 2 tablas (Real/Predicción/Correcto, alto 300) o `st.info("¡Todas las muestras
  fueron clasificadas correctamente!")` → barras "Precisión por clase" (seaborn, un color
  #3274a1, y de 0 a 1, ejes "Clase"/"Precisión", 8×4 in) + descarga + código
  (`precision_por_clase.py.j2`).
- **Svelte**: `MetricGrid`, `QualityBadge`, `ConfusionMatrix` (Plotly heatmap, escala Blues,
  `texttemplate` entero), `ClassReportTable`, `SamplesTable` ×2, `PrecisionByClassChart`, `CodeBlock`.
- **Usan**: Regresión logística, Árboles (clasificación), KNN (clasificación), Redes (con su propio
  orden de secciones en F5).

### C2. Evaluación de regresión

- **Core** `metrics::regression(y, y_hat) -> RegressionEvaluation { mse, rmse, mae, r2,
  abs_error: Vec<f64>, errors_hist: Histogram (bins "auto" de numpy, F1), errors_kde: KdeCurve }`.
  La KDE del histograma de errores es la de `sns.histplot(kde=True)`: Scott, 200 puntos,
  **cut = 0** (no 3) y escalada por `n · ancho_de_bin` (hist de recuentos). Reutiliza
  `stats::kde` de F1 con un parámetro `cut`.
- **Vista**: 4 métricas ("Error Cuadrático Medio (MSE)", "RMSE", "Error Absoluto Medio (MAE)",
  "R² Score", `{:.4f}`, con `help`) → scatter real vs predicción (alpha 0,5, color = |error| con
  `viridis`, colorbar "Error absoluto", diagonal roja discontinua de min(y) a max(y), ejes "Valores
  reales"/"Predicciones", título "Predicciones vs Valores reales", 6×5 in) + descarga + código
  (`predicciones_vs_reales.py.j2`) → histograma de errores (y − ŷ) con KDE, línea x = 0 roja
  discontinua, título "Distribución de errores", eje x "Error (Real - Predicción)", eje y "Count"
  (seaborn, en inglés) + descarga + código (`distribucion_errores.py.j2`) → expander "📊
  Explicación de métricas de regresión".
- **Svelte**: `MetricGrid`, `PredVsRealChart` (modo `colorByError`), `ErrorHistogram`.
- **Usan**: Regresión lineal, Árboles y KNN (regresión), Redes (regresión).

### C3. Interpretación de resultados

`### 🔍 Interpretación de Resultados` + un `st.info` construido con las plantillas de
`componentes/mensajes.toml [interpretacion]`. Regresión: R² en %, MAE y RMSE `{:.2f}` y la línea
del RMSE > 1,5·MAE ("Tu modelo de regresión lineal explica…", literal también en árboles). Clasificación:
accuracy, precisión y recall weighted en % `{:.1f}`, F1 `{:.3f}` si > 0, balance (|p − r| > 0,1)
y recomendaciones si accuracy < 0,6. **Core**: `interpretation::{regression, classification}`
devuelve el Markdown ya formado (es lógica, no contenido: las frases salen del TOML).

### C4. Curvas ROC y Precision-Recall (`viz/roc.py::plot_roc_curve`)

- **Core** `curves`:
  - `roc_curve(y: &[bool], score: &[f64]) -> Roc { fpr, tpr, thresholds }` con la semántica de
    sklearn 1.7: orden descendente de la puntuación, puntos solo en cambios de valor,
    `drop_intermediate=True` (quita puntos colineales), primer umbral = +∞.
  - `auc(x, y)` (trapecios), `precision_recall_curve` (precision/recall/thresholds de sklearn,
    con el punto final precision = 1, recall = 0) y `average_precision` (Σ (Rₙ − Rₙ₋₁)·Pₙ).
  - `roc_binary(y, p1)` + `baseline = Σy / n` (asume clases 0/1: la positiva es la 1).
  - `roc_multiclass(y, proba, n_classes)`: one-vs-rest por clase; micro (todo aplanado);
    macro = unión de FPR, `np.interp` de cada TPR y media, AUC por trapecios; AP macro = media de
    AP por clase; **weighted** (no se muestra, D20) para Árboles/KNN si se decide.
- **Vista binaria**: `### 📈 Curvas de Rendimiento` + 2 subgráficos (16×6 in, 80 % del ancho):
  ROC (darkorange, ancho 2, "Curva ROC (AUC = {:.3f})"; diagonal navy discontinua "Clasificador
  Aleatorio"; x 0..1, y 0..1,05; ejes "Tasa de Falsos Positivos"/"Tasa de Verdaderos Positivos";
  leyenda abajo a la derecha; rejilla 0,3) y PR (darkgreen, "Curva P-R (AP = {:.3f})", baseline navy
  "Baseline ({:.3f})", ejes "Recall (Sensibilidad)"/"Precision (Precisión)", leyenda abajo a la
  izquierda) → `### 📋 Interpretación de las Curvas` (2 columnas con avisos por umbral, TOML).
- **Vista multiclase**: `### ⚙️ Opciones de Visualización` (checkbox "Mostrar curvas por clase
  individual", True; multiselect "Métodos de promediado:" macro/micro, ambos; vacío → macro) →
  `### 📈 Curvas de Rendimiento Multi-clase` (16×8 in): por clase con colores cíclicos
  aqua, darkorange, cornflowerblue, red, green, purple, brown, pink, gray, olive (ancho 2, alpha 0,7);
  micro deeppink punteado ancho 4; macro navy discontinuo ancho 4 (en PR, línea **horizontal** en el
  AP macro); "Clasificador Aleatorio" `k--` alpha 0,5. → `### 📋 Interpretación de Métricas
  Multi-clase` (por cada método en el orden elegido: `#### Macro-promedio`, avisos de AUC y AP) →
  `#### 📊 Métricas por Clase` (Clase, ROC AUC, Average Precision redondeados a 3; solo si hay
  curvas por clase) → expander "ℹ️ ¿Macro vs Micro promedio?". En ambos casos, al final, el
  expander "ℹ️ ¿Cómo interpretar las Curvas de Rendimiento?".
- **Svelte**: `RocPrCharts` (`mode: binary|multiclass`, `showPerClass`, `averages: ("macro"|"micro")[]`;
  el filtrado es en la UI, sin ida y vuelta) y `CurveInterpretation`.
- **Usan**: Regresión logística, Árboles, KNN.

### C5. Distribución de probabilidades (`plot_threshold_analysis`, parte 1)

- **Core** `probabilities::distribution(y, proba, class_names)`: binaria → para cada clase real, el
  histograma de 20 bins **entre el mínimo y el máximo de su propio grupo** (`ax.hist`) de
  p(clase 1), recuentos de incertidumbre (0,3 < p < 0,7), "confiables" (p < 0,2 o p > 0,8) y el
  porcentaje; multiclase → para cada clase i (las de `unique(y_test)`), 20 bins de `proba[:, i]`.
- **Vista binaria**: expander "ℹ️ ¿Cómo interpretar la Distribución de Probabilidades?" →
  histogramas superpuestos (clase 0 lightcoral, clase 1 lightblue, alpha 0,7, borde negro, leyenda
  "Clase {nombre} (Real)"), umbral rojo discontinuo "Umbral de decisión (0.5)", título
  "Distribución de Probabilidades Predichas por Clase Real", ejes "Probabilidad de Clase
  Positiva"/"Frecuencia", x 0..1, 12×6 in → 3 métricas ("Muestras en zona incierta" n/N,
  "Porcentaje de incertidumbre" `{:.1%}`, "Predicciones confiables" n/N) → aviso por separación
  (< 0,2 success, < 0,4 info, si no warning).
- **Vista multiclase**: `st.info("**Nota:** Clasificación multiclase detectada…")` + expander
  multiclase + rejilla de min(3, k) columnas, un histograma por clase (`Set3(i)`, título
  "Probabilidades para {clase}", ejes "Probabilidad"/"Frecuencia", x 0..1, 5·cols × 4·filas in).
- **Svelte**: `ProbabilityHistograms`. **Usan**: logística, KNN y Árboles ("📊 Distribución de
  Probabilidades" en KNN).

### C6. Análisis de umbrales (binaria)

- **Core** `probabilities::threshold_analysis(y, p1) -> ThresholdAnalysis { rows, best_f1, shown }`:
  umbrales `np.arange(0.1, 1.0, 0.1)` (los valores exactos de numpy, con 0.30000000000000004), se
  omiten los umbrales con una sola clase predicha; accuracy/precision/recall/F1 con
  `zero_division=0`; mejor = primer máximo de F1.
- **Vista**: `### 🎯 Análisis de Umbrales de Decisión` + expander → 2 columnas: métricas "Umbral
  Actual" ("0.50") y "Umbral Óptimo (F1)" (`{:.2f}`) + consejo si |óptimo − 0,5| > 0,1; tabla
  "**Comparación de Umbrales:**" (Umbral, Accuracy, Precision, Recall, F1-Score `{:.3f}`). Por el
  fallo de coma flotante, **Python solo muestra la fila 0,5** (D21).
- **Svelte**: `ThresholdAnalysis`. **Usan**: los mismos que C5.

### C7. Residuos (`viz/residual.py`)

- **Core** `residuals::analysis(y, y_hat) -> ResidualAnalysis { residuals, mean, std (ddof=1),
  skew (scipy, sesgado), linregress_r: Option<f64> (si n > 10), hist20, flags }`.
- **C7a** expander "ℹ️ ¿Cómo interpretar estas visualizaciones?" → scatter real vs predicción
  (alpha 0,6, s 50, borde negro 0,5), diagonal "Predicción Perfecta" `r--` de min(y, ŷ) a max(y, ŷ),
  caja "R² = {:.4f}" (wheat; **siempre 0,0000 en Python**, D22), 12×8 in.
- **C7b** expander "ℹ️ ¿Cómo interpretar el análisis de residuos?" → 2 subgráficos (15×6 in):
  residuos vs predicciones (línea y = 0 "Residuo = 0", caja "Desv. Estándar: {:.3f}") e histograma
  de 20 bins skyblue con "Media: {:.3f}" (rojo discontinuo) e "Ideal (0)" (verde) →
  `### 🔍 Interpretación de los Residuos` con 3 viñetas y la sugerencia si falla alguna. **Ojo:**
  con n ≤ 10 Python lanza `NameError` (`r_value` sin definir) si además la media de residuos está cerca de 0; con `test_size` 0,3 hace falta un CSV de ≤ 33 filas.
- **Svelte**: `PredVsRealChart` (modo simple), `ResidualsCharts`. **Usan**: lineal, KNN regresión
  ("📊 Distribución de predicciones"), Árboles regresión ("📈 residuos").

### C8. Frontera de decisión y superficie 2D (`viz/decision_boundary.py`)

- **Controles**: `### Selección de Características`, "Selecciona 2 características para
  visualizar:", 2 selects en 2 columnas: `Primera característica:` (key `viz_feature1`, índice 0) y
  `Segunda característica:` (key `viz_feature2`, índice min(1, n−1)), con los nombres legibles. Si
  son iguales **no se muestra nada más** (sin aviso).
- **Modelo**: `model.refit(X_train[:, [i, j]], y_train)` (mismos hiperparámetros, `random_state=42`).
  El resto de características **no** se fija en ningún valor: el modelo es otro (el inventario
  §6 dice "el resto fijas en su media"; eso es de Redes, F5).
- **Rejilla (core `grid::decision_axes`, exacta)**: `h = 0.02` (KNN: 0.1 si n > 1000, 0.05 si > 500);
  `x_min = min − 1`, `x_max = max + 1` (ídem y); si max(rango_x, rango_y) > 100, `h = max(h,
  max_rango/200)`; `h = min(h, rango_x/10, rango_y/10)`; ejes = `np.arange(min, max, h)` —
  numpy calcula `n = ceil((max − min)/h)` y el valor i como `min + i·((min + h) − min)`, no
  `min + i·h` — y, si un eje tiene < 2 puntos, `linspace(min, max, 10)`. Si filas·columnas > 50 000:
  `linspace(min, max, min(200, n))` en cada eje (incluye el extremo). Mensajes "Malla creada: {filas}
  x {columnas} = {n:,} puntos" (filas = puntos de y), "⚠️ La malla es muy grande…" y "Nueva
  resolución: …" (frontera: 3 avisos info/warning/info; superficie: uno solo con "- " y
  "\n - "). Si la malla resultante tiene < 2 filas o columnas, error.
- **Frontera (clasificación)**: `contourf(Z clases)`, alpha 0,3; binaria `RdBu` con 50 niveles;
  multiclase `Set3` con `levels=n_clases`; puntos de **train** con `c=y`, misma paleta, borde negro,
  s 50, alpha 0,8; leyenda "Clases" con parches (binaria #d7191c/#2c7bb6; multiclase
  `Set3(linspace(0,1,k))`): los colores del fondo, de los puntos y de la leyenda **no coinciden**
  (D23). Título "Frontera de Decisión para {f1} vs {f2}", ejes = nombres, rejilla 0,3, 10×8 in.
  Después: enlace "📥 Descargar visualización de frontera", `st.warning` de 2D (si hay más de 2
  características) y el código (`frontera_decision.py.j2`, D30).
- **Superficie (regresión)**: `contourf(Z, alpha 0.6, viridis, 20 niveles)`, puntos de train con
  `c=y` viridis, colorbar "Valor Objetivo" (shrink 0,8), título "Superficie de Predicción para {f1}
  vs {f2}". Sin descarga ni código.
- **DTO** `DecisionGrid { x: Vec<f64>, y: Vec<f64>, z: GridValues::{Classes(Vec<u16>),
  Values(Vec<f64>)} (row-major, filas = y), notices: Vec<GridNotice>, points: Matrix (train, 2
  col.), point_target, class_names, feature_names: [String; 2] }`. 40 000 celdas = 80 KB en u16:
  aceptable por IPC.
- **Svelte**: `FeaturePairSelector` + `DecisionBoundaryChart` (Plotly `heatmap` con `zsmooth:false`
  y escala discreta por clase, o `contour` con `coloring:'heatmap'`; `scattergl` para los puntos) y
  `PredictionSurfaceChart` (`contour` viridis + colorbar).
- **Usan**: logística y lineal (F2), Árboles (frontera y superficie), KNN (con su política de
  resolución). Redes usa otra superficie (F5).

### C9. Formulario de predicción (`ui.py::create_prediction_interface`)

- **Core** `form::build(x_train: ArrayView2<f64>, columns, display, metadata: &DatasetMetadata) ->
  Vec<FormControl>`. Reglas por columna i (sobre **X_train**, no sobre el dataset completo):
  - `nunique ≤ 2` → **binaria**: si los metadatos tienen mapa de valores → select con los textos
    (índice 0); si los valores son {0, 1} → checkbox (False); si no → select con los valores
    (índice 0).
  - `nunique ≤ 10` y (todos enteros **o** categórica en metadatos) → **categórica**: select con los
    textos del mapa o los valores, índice `len // 2`.
  - si no → **continua**: slider de min a max de train, paso (max − min)/100 (0,1 si min = max),
    valor = **media de train**.
  - Etiqueta "**{nombre}**\n\n*{descripción}*" si los metadatos tienen descripción; si no, el nombre.
  - Dos columnas: las primeras ⌈n/2⌉ a la izquierda. Keys `feature_{i}`.
  - Los valores de un select con textos se traducen al código original (p. ej. "Mujer" → 0).
- **Predicción** (botón "Realizar predicción", primary): Python construye
  `np.array(valores, dtype=float32)` (D31) y llama a `predict`. Resultado en una tarjeta (#E8F5E9,
  radio 10, centrada): `<h3>Resultado de la predicción</h3>` + "Clase: {nombre}" (24 px, negrita,
  #2E7D32) en clasificación (**`int(predict)`**, que trunca si el modelo es lineal: D25) o
  "Valor: {:.4f}" en regresión. Extras: logística → tabla Clase/Probabilidad (D33); lineal →
  `### Coeficientes` + tabla Característica/Coeficiente; árbol → camino de decisión (F3); KNN →
  info del modelo (F4). La rama de clasificación/regresión la decide `task_type` **del dataset**.
- **DTO** `PredictionForm { controls: Vec<FormControl> }` con `FormControl { index, column, label,
  description: Option<String>, kind: Checkbox{default} | Select{options: Vec<{label, code: f64}>,
  default_index} | Slider{min, max, step, default} }` y `PredictionResult { kind:
  Class{index, name, proba: Option<Vec<(String, f64)>>} | Value{value}, extras: Vec<PredictionExtra> }`.
- **Svelte**: `PredictionForm` (genérico, emite `values: number[]`), `PredictionResultCard`,
  `ProbabilityTable`, `CoefficientMiniTable`.
- **Usan**: las 4 lecciones (Redes con su escalado en F5).

### C10. Importancia de características (`viz/features.py`)

No lo usa Regresión, pero es compartido: lo necesitan Árboles (nativa + permutación) y KNN
(permutación). Se especifica aquí para fijar la interfaz; se implementa en F3.

- **Core** `importance::permutation(model: &dyn Predictor, x_test, y_test, n_repeats = 10, seed =
  42, scorer) -> PermutationImportance { mean, std }`. sklearn usa `check_random_state(42)` y una
  semilla por columna (`random_state.randint(MAX_INT, size=n_features)`) y `rng.shuffle`: hay que
  reproducir el RNG de numpy (rand_mt, como el split D7) o aceptar paridad estadística (decisión en
  F3). Puntuación: accuracy (clasificación) o R² (regresión).
- **Vista**: `### Importancia de Características`, info "🔄 Calculando…", texto del método, barras
  horizontales viridis en orden ascendente con barras de error (std), valores `{:.3f}`, título
  "Importancia de Características\n({método})", tabla "### Valores Detallados" (round 4) y el
  expander "📖 Cómo interpretar estos resultados" (`componentes/mensajes.toml [importancia]`).
- **Svelte**: `FeatureImportanceChart` + `DataTable`.

### C11. Exportación (D1)

- **Core/export** (`mltutor-export`):
  - `onnx::linear_regressor(coef, intercept, n_features)` → `ai.onnx.ml.LinearRegressor`
    (entrada `float_input` [None, n] float32, salida `variable` [None, 1]).
  - `onnx::linear_classifier(coef, intercept, classes, post_transform)` → `LinearClassifier`
    (multinomial: `SOFTMAX`; binaria: skl2onnx usa `LOGISTIC` con 1 fila de coeficientes) +
    salidas `label` (int64) y `probabilities` (sin ZipMap). Estructura de referencia en el
    fixture (`onnx_reference.nodes`; opset 15, ai.onnx.ml 3, como MEMORY).
  - `package::python_zip(model_json, script, readme)` → ZIP con `modelo.json`,
    `reconstruir_modelo.py` (`content/code/paquete_modelo_lineal.py.j2`) y `LEEME.txt`.
  - `modelo.json` (esquema v1): `format = "mltutor-model"`, `format_version`, `estimator`,
    `params` (`get_params()` completo), `feature_names_in`, `display_names`, `n_features_in`,
    `coef_`, `intercept_`, `classes_`, `class_names`, `n_iter_`, `training {dataset, test_size,
    random_state, n_train, n_test}`, `check {X (5 filas de test), prediction, probabilities}`. El
    script reconstruye el estimador sin reentrenar, **comprueba** que predice lo mismo y lo guarda
    en `modelo.pkl`. Verificado ejecutándolo en los 7 casos (`export_package.verified_by_execution`).
- **Vista** (D28): "💾 Exportar a ONNX" y "📦 Paquete Python (.zip)" + "📄 Generar Código".
  Los archivos se guardan con el diálogo nativo **en Rust** (`export_model`): la ruta no pasa por JS.
- **Usan**: todas (Árboles: TreeEnsemble*; KNN: paquete Python, ONNX por decidir en F4; Redes:
  Gemm + activaciones y paquete Keras).

### C12. Código Python generado

- **Core** `codegen::render(template: CodeTemplate, ctx: serde_json::Value) -> String` con minijinja,
  plantillas embebidas (`include_str!("../../../content/code/*.py.j2")`), `keep_trailing_newline` y
  un filtro `pyrepr` que reproduce `repr()` de Python para str, listas, números (`repr(float)` =
  representación mínima de ida y vuelta) y None/True/False.
- **Vista**: `show_code_with_download(code, title, file)` de F1 (`CodeBlock` con descarga).
- El fixture `code/regression.json` da el texto exacto que debe salir de cada plantilla.

---

## 4. Casos límite y comportamientos de Python

| Caso | Python | Rust (propuesta) |
|---|---|---|
| Lineal sobre dataset de clasificación (Iris) | Entrena, aviso; evaluación de regresión; la predicción hace `int(ŷ)` → clase truncada (1,99 → versicolor; < 0 → setosa; ≥ 3 → "3") | D25 |
| Logística sobre regresión con objetivo de floats enteros (Diabetes) | **Entrena** un clasificador de 192 clases, "¡Modelo entrenado exitosamente!"; la ROC falla (página de error); la predicción muestra "Valor:" | D26: bloquear |
| Logística sobre objetivo continuo (California) | `st.error("Error al entrenar el modelo: Unknown label type: continuous…")` (en inglés) | D26 |
| Logística que no converge (Cáncer, 1000 iteraciones sin escalar) | ConvergenceWarning invisible para el estudiante; probabilidades a 0,20 del óptimo | D19 |
| Una clase ausente del test (CSV pequeños) | ROC/probabilidades usan `unique(y_test)`: columnas de `predict_proba` desalineadas; el informe incluye la clase con support 0 y avisos | D20 |
| Test con ≤ 10 muestras (regresión) | `NameError` en el análisis de residuos (si la media de residuos es ≈ 0) | Omitir la viñeta de correlación |
| CSV con clases de texto | "Clase 0…k" (F1, D10: Rust usa los nombres originales) | Los tests comparan por código, no por nombre |
| Cambiar de dataset tras entrenar | Las pestañas 2–6 siguen con el modelo viejo; "Dataset actual" muestra el nuevo; el formulario usa los metadatos del **nuevo** (puede romper) | D32 |
| Misma característica en los dos selects | No dibuja nada, sin mensaje | Igual + `Notice info` "Elige dos características distintas" (texto nuevo, opcional) |
| Muestra casi empatada (CSV `string_target`, margen 2e-5) | — | Una predicción puede cambiar con otro optimizador: el test lo tolera si `min_proba_margin < 2·tol` |
| Rangos enormes (Titanic Tarifa, California) | `h` crece hasta rango/200; mallas de hasta 2 M puntos antes de reducir | Calcular solo los ejes (sin materializar la malla grande) |

---

## 5. Contrato IPC

### 5.1 Convenciones (las de F1, §5.1)

Comandos `async` + `spawn_blocking`; declarados en `build.rs` y `capabilities/default.json`; DTO en
`src-tauri/src/ipc/` con ts-rs; matrices `Matrix { rows, cols, values: Vec<Option<f64>> }`; errores
`CommandError` en español. Nuevo en F2:

- **Progreso**: evento `task-progress { task_id, message, percent: Option<f32> }`. El
  entrenamiento lineal/logístico tarda milisegundos; se emite al empezar y al terminar (la barra
  de estado lo muestra igual que en Python). La cancelación (`cancel_task`, F1) se comprueba entre
  iteraciones de L-BFGS (callback de argmin/linfa) y entre filas de la rejilla.
- **`model_id`**: cada entrenamiento crea un id nuevo por lección. Los comandos de vistas lo
  reciben; si ya no es el actual → `CommandError::ModelStale { message: "El modelo ha cambiado.
  Vuelve a cargar la vista." }` (la UI lo trata como recarga, no como error).
- **`seq`**: `train_model` lleva `seq` (como `select_dataset`): si llegan dos entrenamientos, gana el
  más reciente y el antiguo termina con `Cancelled`.

### 5.2 Estado (`src-tauri/src/session.rs`)

```rust
pub struct LessonModels { slots: HashMap<LessonId, TrainedModel>, next_id: u32, train_seq: HashMap<LessonId, u64> }
pub struct TrainedModel {
    pub id: u32, pub lesson: LessonId, pub dataset: DatasetRef, pub dataset_label: String,
    pub config: ModelConfig, pub split: SplitIndices /* posicionales, D7 */,
    pub model: Arc<dyn Predictor>, pub features: FeatureInfo, pub class_names: Option<Vec<String>>,
    pub metadata: Arc<DatasetMetadata>,          // del dataset de ENTRENAMIENTO (D32)
    pub cache: OnceLock<EvaluationCache>,        // y_pred, proba, métricas
}
```

El estado de interfaz (pestaña activa, tipo de modelo elegido, `max_iter`, `test_size`, subvista,
par de características, opciones de ROC, valores del formulario) vive en el store de Svelte de la
lección, como en F1.

### 5.3 Tipos

```rust
#[serde(rename_all = "snake_case")] pub enum LessonId { Trees, Regression, Knn, Nn }
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelConfig {
    Linear,
    Logistic { max_iter: u32 },                  // 100..=2000, paso 100
    // F3: Tree { … }, F4: Knn { … }
}
pub struct TrainRequest { pub lesson: LessonId, pub dataset: DatasetRef, pub config: ModelConfig,
                          pub test_size: f64 /* 0.1..=0.5 */, pub seq: u64, pub task_id: u32 }
pub struct TrainedModelInfo {
    pub model_id: u32, pub lesson: LessonId, pub dataset: DatasetRef, pub dataset_label: String,
    pub config: ModelConfig, pub model_task: TaskType, pub dataset_task: TaskType,
    pub n_train: u32, pub n_test: u32,
    pub feature_columns: Vec<String>, pub feature_names: Vec<String>, pub class_names: Option<Vec<String>>,
    pub notices: Vec<TrainNotice>,               // NotConverged{max_iter} (D19), ModelTaskMismatch
}
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrainNotice { NotConverged { max_iter: u32, message: String }, ModelTaskMismatch { message: String } }

pub struct ClassMetrics { pub name: String, pub precision: f64, pub recall: f64, pub f1: f64, pub support: u32 }
pub struct SampleRow { pub position: u32, pub real: String, pub predicted: String, pub correct: bool }
pub struct ClassificationEvaluation {
    pub accuracy: f64, pub per_class: Vec<ClassMetrics>, pub macro_avg: ClassMetrics, pub weighted_avg: ClassMetrics,
    pub confusion: Matrix, pub correct_head: Vec<SampleRow>, pub incorrect_head: Vec<SampleRow>,
    pub interpretation_md: String, pub warnings: Vec<String>,
}
pub struct Histogram { pub edges: Vec<f64>, pub counts: Vec<u32> }
pub struct RegressionEvaluation {
    pub mse: f64, pub rmse: f64, pub mae: f64, pub r2: f64,
    pub y_true: Vec<f64>, pub y_pred: Vec<f64>,  // el |error| lo calcula la UI
    pub errors_hist: Histogram, pub errors_kde: KdeCurve /* F1 */, pub interpretation_md: String,
}
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Evaluation { Classification(ClassificationEvaluation), Regression(RegressionEvaluation) }

pub struct Curve { pub x: Vec<f64>, pub y: Vec<f64>, pub thresholds: Option<Vec<Option<f64>>> /* None = +∞ */ }
pub struct ClassCurves { pub class_index: u32, pub name: String, pub roc: Curve, pub auc: f64, pub pr: Curve, pub ap: f64 }
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RocData {
    Binary { roc: Curve, auc: f64, pr: Curve, ap: f64, baseline: f64 },
    Multiclass { per_class: Vec<ClassCurves>, micro: ClassCurves, macro_roc: Curve, macro_auc: f64,
                 macro_ap: f64, weighted_auc: Option<f64>, absent_classes: Vec<String> /* D20 */ },
}
pub struct ThresholdRow { pub threshold: f64, pub accuracy: f64, pub precision: f64, pub recall: f64, pub f1: f64 }
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProbabilityView {
    Binary { per_real_class: Vec<(String, Histogram)>, uncertain: u32, confident: u32, n: u32,
             thresholds: Vec<ThresholdRow>, best_f1_threshold: Option<f64>, shown_thresholds: Vec<f64> /* D21 */ },
    Multiclass { per_class: Vec<(String, Histogram)> },
}
pub struct ResidualsView { pub y_true: Vec<f64>, pub y_pred: Vec<f64>, pub r2: f64 /* D22 */,
    pub mean: f64, pub std: f64, pub skew: f64, pub linregress_r: Option<f64>, pub hist: Histogram,
    pub checks: ResidualChecks { mean_ok: bool, corr_ok: Option<bool>, normal_ok: bool } }
pub struct GridRequest { pub model_id: u32, pub feature_x: u32, pub feature_y: u32, pub task_id: u32 }
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GridNotice { Created { rows: u32, cols: u32 }, TooLarge { points: u64 }, Resized { rows: u32, cols: u32 } }
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GridValues { Classes { z: Vec<u16> }, Values { z: Vec<f64>, min: f64, max: f64 } }
pub struct DecisionGrid { pub x: Vec<f64>, pub y: Vec<f64>, pub values: GridValues, pub notices: Vec<GridNotice>,
    pub points: Matrix /* train, 2 columnas */, pub point_target: Vec<f64>, pub feature_names: [String; 2],
    pub class_names: Option<Vec<String>>, pub show_2d_warning: bool }
pub struct CoefficientRow { pub feature: String, pub feature_position: u32, pub coef: f64, pub abs: f64,
    pub odds_ratio: Option<f64>, pub change_pct: Option<f64>, pub effect: Effect, pub importance: Option<Level>, pub strength: Option<Level> }
pub struct CoefficientsView { pub rows: Vec<CoefficientRow> /* ordenadas por |coef| */, pub class_index: Option<u32>,
    pub n_classes: u32, pub intercept: f64, pub quantiles: AbsQuantiles /* 0.25, 0.6, 0.75, 0.8, 0.95 */,
    pub extreme: Vec<String> }
pub struct PredictionForm { pub controls: Vec<FormControl> }        // §3 C9
pub struct PredictRequest { pub model_id: u32, pub values: Vec<f64> /* códigos, en orden de columnas */ }
pub struct PredictionResult { /* §3 C9 */ }
#[serde(rename_all = "snake_case")] pub enum ExportFormat { Onnx, PythonPackage }
#[serde(rename_all = "snake_case")]
pub enum CodeTemplate { ConfusionMatrix, PrecisionByClass, PredVsReal, ErrorDistribution, DecisionBoundary, RegressionCode /* D29 */ }
```

### 5.4 Comandos

| Comando | Entrada | Salida | Errores | Notas |
|---|---|---|---|---|
| `train_model(req)` | `TrainRequest` | `TrainedModelInfo` | `DatasetNotFound`, `InvalidTarget` (D26), `Training`, `Cancelled` | Split D7 + entrenamiento en `spawn_blocking`; progreso por evento; sustituye el modelo de la lección |
| `lesson_model(lesson)` | `LessonId` | `Option<TrainedModelInfo>` | — | Al montar las pestañas 2–6 (o tras recargar) |
| `evaluate_model(model_id)` | `u32` | `Evaluation` | `ModelStale` | C1/C2/C3; calcula y cachea `y_pred`/`proba` |
| `roc_curves(model_id)` | `u32` | `RocData` | `ModelStale`, `NotClassifier` | C4. Los filtros (por clase, macro/micro) son de la UI |
| `probability_view(model_id)` | `u32` | `ProbabilityView` | ídem | C5 + C6 |
| `residuals_view(model_id)` | `u32` | `ResidualsView` | `ModelStale`, `NotRegressor` | C7 |
| `decision_grid(req)` | `GridRequest` | `DecisionGrid` | `ModelStale`, `InvalidSelection` (x = y), `GridTooSmall`, `Cancelled` | C8. Reentrena en 2D y predice 40 000 puntos (< 50 ms) |
| `coefficients_view(model_id, class_index)` | `u32`, `Option<u32>` | `CoefficientsView` | `ModelStale`, `InvalidSelection` | §2.5 |
| `prediction_form(model_id)` | `u32` | `PredictionForm` | `ModelStale` | C9, metadatos del dataset de entrenamiento |
| `predict_one(req)` | `PredictRequest` | `PredictionResult` | `ModelStale`, `InvalidInput` (longitud, NaN) | C9 |
| `export_model(model_id, format)` | `u32`, `ExportFormat` | `Option<String>` (nombre guardado) | `ModelStale`, `Export`, `FileSave` | Genera ONNX o el ZIP en Rust y abre el diálogo "Guardar" (`regresion_lineal.onnx` / `regresion_logistica.onnx`, `modelo_regresion.zip`); `None` si se cancela |
| `generate_code(model_id, template, extra)` | `u32`, `CodeTemplate`, `Option<CodeExtra { feature_x, feature_y }>` | `String` | `ModelStale` | C12; la descarga del `.py` reutiliza `save_file` (F1) |
| `save_file` (F1) | — | — | — | PNG de cada gráfico (Plotly `toImage`) y `.py` |

Presupuesto: cada comando de vista < 100 ms con California (6 192 muestras de test); el más caro es
`decision_grid` (2D refit + 40 000 predicciones). La UI pide la vista al entrar en la subvista,
no todas a la vez.

---

## 6. Fixtures de paridad

`uv run python tools/export_golden.py --only linear_models --only code` (unos 30 s) y `--check`
(63 fixtures, 0 cambios). Nuevo módulo `tools/golden_render.py`: renderiza la lección real con
`ui.Session` sin Qt, pulsa botones/fija controles como el estudiante y convierte el árbol de nodos
a JSON. De cada `st.pyplot` extrae lo que matplotlib dibujó (tras `canvas.draw()`): líneas, barras,
scatter (puntos, colores resueltos, cmap y norma), heatmaps, rejillas de `contourf` (ejes completos,
Z como dígitos o diezmada, niveles y colores de banda), textos, leyendas, ejes y límites. Las nubes
de más de 600 puntos se resumen (hash, cabeza, min/max) porque los valores ya están en la sección
numérica. Sirve igual para F3–F5.

| Ruta | Caso | Qué ejercita |
|---|---|---|
| `linear_models/iris_logistic.json` | multiclase (3), converge (81 it.) | ROC por clase/micro/macro, opciones de ROC, frontera Set3 (2 pares + misma característica), probabilidades multiclase, coeficientes con selector de clase |
| `linear_models/breast_cancer_logistic.json` | binaria, **no converge** a 1000 | ROC + PR binaria, umbrales, frontera RdBu (50 niveles), D19 |
| `linear_models/titanic_logistic.json` | binaria con categóricas | formulario con selects de metadatos, coeficientes binarios |
| `linear_models/diabetes_linear.json` | OLS | evaluación de regresión, residuos, superficie (age/sex y bmi/s5), formulario de sliders y select binario |
| `linear_models/california_housing_linear.json` | OLS, 20 640 filas | tamaño, rejilla reducida a 200×200 |
| `linear_models/csv_string_target_logistic.json` | CSV `string_target` (30 filas, 3 clases "Clase i") | test de 9 muestras, una casi empatada (margen 2e-5) |
| `linear_models/csv_numeric_regression_linear.json` | CSV `numeric_regression` (40 filas) | test de 12 |
| `linear_models/iris_linear_mismatch.json` | Lineal en clasificación | aviso, evaluación de regresión, truncado `int()` en el formulario |
| `linear_models/diabetes_logistic_mismatch.json` | Logística en regresión | entrena 192 clases; `viz_roc.app_error` |
| `code/regression.json` | 6 plantillas | texto exacto frente a jinja2 |

Contenido de cada caso (`expected`): `train_indices`/`test_indices` (posicionales), `y_test`, `y_pred`,
`model` (coeficientes, intercepto, `n_iter`, `get_params()`), `converged_reference` (logística con
`tol=1e-12`, `max_iter=100000`), `proba`, `decision_function`, `min_proba_margin`, `evaluation`
(informe con `labels`/`target_names` como la app, matriz de confusión, curvas con umbrales —+∞ como
`"inf"`—, AUC/AP por clase/micro/macro, `not_in_app` con weighted, análisis de umbrales con
`comparison_rows_shown_by_app`; o MSE/RMSE/MAE/R² y residuos con skew, `linregress_r` y el
histograma de 20 bins), `two_feature_models` (coeficientes del modelo 2D de cada par),
`coefficients.abs_quantiles_row0`, `prediction_form` (controles, valores por defecto y "custom" y el
vector float32 que recibió `predict`, con su predicción y probabilidades), `onnx_reference` (grafo
de skl2onnx y salidas evaluadas con `onnx.reference.ReferenceEvaluator`; la app no exporta ONNX),
`export_package` (`modelo.json`, script renderizado y su salida al ejecutarlo) y `views`.

**Tolerancias** (en `meta.tolerance` de cada caso):

| Qué | Tolerancia | Por qué |
|---|---|---|
| OLS: coeficientes, intercepto | 1e-8 relativo | Solución cerrada; sklearn usa lstsq sobre datos centrados |
| OLS: predicciones / métricas / residuos | 1e-9 / 1e-8 relativo | Derivados |
| Logística frente a `converged_reference` | 1e-5 en probabilidades, 1e-4 relativo en coeficientes | Mismo óptimo (MEMORY: 5e-6 en Iris) |
| Logística frente al modelo de la app | **2e-3** (Iris, CSV); **3e-3** (Titanic); Cáncer: no normativa | sklearn para en gtol=1e-4; en Cáncer no converge (0,20) |
| Predicciones, métricas, curvas | exactas si coinciden las predicciones/el orden de puntuaciones | Discretas; ver `min_proba_margin` (CSV: 2e-5) |
| Ejes de la rejilla | exactos | Reglas de numpy (`arange` con `delta = (min+h) − min`) |
| Z de la superficie | 1e-9 relativo; frontera: clases exactas salvo celdas en el borde (≤ 0,5 % para la logística 2D) | El modelo 2D logístico hereda la tolerancia del optimizador |
| Formulario | controles exactos; predicción 1e-5 relativo si Rust usa f64 (D31) | Python redondea la entrada a float32 |
| Textos de `views` | exactos tras interpretar cada número con la precisión mostrada (±1 en la última cifra) | Redondeo `{:.4f}` de valores con tolerancia |
| ONNX | 1e-5 (logística) / 1e-4 relativo (lineal) frente a f64 | float32 |

Los fixtures de F0 `regression/iris_logistic.json` (1e-4) y `regression/diabetes_linear.json`
siguen ahí; los nuevos los sustituyen para F2 (ver §13).

---

## 7. Contenido extraído

Literal, con un comentario `<!-- -->` de origen. Los textos largos se han sacado del árbol que
emite la app (ya con `textwrap.dedent`) y los cortos se han comprobado contra el código fuente
(script en el informe de esta fase: 0 diferencias salvo 3 de `export.py`, también literales).

```
content/regresion/   intro.md (F1), coeficientes-lineal.md, coeficientes-logistica.md,
                     tabla-lineal.md, tabla-logistica.md, mensajes.toml (Entrenamiento, avisos,
                     subvistas, Coeficientes con todas las reglas y plantillas, Exportar)
content/componentes/ metricas-clasificacion.md, metricas-regresion.md, roc-interpretar.md,
                     roc-macro-micro.md, probabilidades-binaria.md, probabilidades-multiclase.md,
                     umbrales.md, predicciones-vs-reales.md, residuos-interpretar.md,
                     mensajes.toml (C1–C10: etiquetas, help, umbrales de los avisos, colores,
                     títulos y ejes de cada gráfico)
content/code/        matriz_confusion.py.j2, precision_por_clase.py.j2, predicciones_vs_reales.py.j2,
                     distribucion_errores.py.j2, regresion_codigo.py.j2 (literales),
                     frontera_decision.py.j2 ({{ feature_names | pyrepr }}, {{ class_names | pyrepr }}),
                     paquete_modelo_lineal.py.j2 (NUEVO, D1: script del paquete Python)
```

Notas: `metricas-clasificacion.md` conserva la sangría irregular del original (las viñetas quedan
con 3 espacios); `content/componentes/` es una carpeta nueva porque esos textos son de los
componentes y los reutilizan las demás lecciones. Las plantillas literales siguen la convención de
F1 (sin el salto de línea inicial de las constantes de Python).

---

## 8. Mapeo a Rust

### 8.1 Cálculos

| Cálculo | Python | Rust | Huecos |
|---|---|---|---|
| Split | `train_test_split(0.3, 42)` | `split` (F1, D7) | — |
| OLS | `LinearRegression()` | `linfa-linear` 0.8.1 (`LinearRegression::fit`, = sklearn a 1e-12, MEMORY) | Rango deficiente: comprobar que linfa no falla con columnas colineales (sklearn usa lstsq/pseudoinversa) |
| Logística | `LogisticRegression(lbfgs, C=1, max_iter, multinomial)` | `linfa-logistic` 0.8.1: `LogisticRegression` (binaria) y `MultiLogisticRegression` con `alpha = 1/C`… **ojo a la escala**: sklearn minimiza ½‖W‖² + C·Σ pérdida; linfa, pérdida media o suma + α‖W‖² (verificar el factor ½ y si promedia) | Remapear la clase positiva en binaria (MEMORY); `max_iterations`; detectar la no convergencia (D19); cancelación dentro de argmin |
| Informe por clase, matriz de confusión, accuracy | sklearn | Propio `metrics` (trivial; `zero_division=0`) | — |
| MSE/RMSE/MAE/R² | sklearn | Propio | R² con varianza 0 (sklearn devuelve 0/NaN según versión: replicar 1.7) |
| ROC, PR, AUC, AP | sklearn | Propio `curves` | `drop_intermediate`, empates de puntuación, umbral +∞ |
| Macro (interp) / micro | numpy | Propio | — |
| Umbrales | sklearn | Propio | — |
| Histogramas | `ax.hist(20)`, `histplot('auto')` | `stats::histogram` (F1) | — |
| KDE de errores | seaborn (cut 0) | `stats::kde` (F1) + parámetro `cut` | — |
| Skew, linregress | scipy | Propio (fórmulas cerradas) | — |
| Cuantiles de |coef| | pandas (lineal) | Propio (F1 ya tiene cuantiles lineales) | — |
| Rejilla | numpy arange/linspace | `grid::decision_axes` | Reproducir `arange` exacto |
| Modelo 2D | sklearn refit | `Predictor::refit` | — |
| Formulario | pandas sobre X_train | `form::build` + `dataset::metadata` (F1) | — |
| ONNX | — (no existe en Python) | `mltutor-export::onnx` (prost, F0) | — |
| Paquete Python | — | `mltutor-export::package` (`zip` crate, sin compresión externa) | Añadir dependencia `zip` (deflate puro Rust) |
| Código | f-strings | `codegen` con minijinja | Filtro `pyrepr` |
| Permutation importance (C10) | sklearn | Propio (F3) | RNG de numpy por columna |

### 8.2 Vistas

| Vista | Comando | Componente Svelte |
|---|---|---|
| Pestaña Entrenamiento | `train_model`, `lesson_model` (+ F1 `select_dataset`, `dataset_summary`) | `RegressionTrainTab.svelte` (selector de F1, `ModelTypeButtons`, `Slider` ×2, botón, `Notice`) |
| Evaluación | `evaluate_model` | `EvaluationView.svelte` (C1/C2/C3) |
| ROC | `roc_curves` | `RocPrCharts.svelte`, `CurveInterpretation.svelte` |
| Probabilidades + umbrales | `probability_view` | `ProbabilityHistograms.svelte`, `ThresholdAnalysis.svelte` |
| Residuos | `residuals_view` | `PredVsRealChart.svelte`, `ResidualsCharts.svelte` |
| Frontera / superficie | `decision_grid`, `generate_code(DecisionBoundary)` | `FeaturePairSelector.svelte`, `DecisionBoundaryChart.svelte`, `PredictionSurfaceChart.svelte` |
| Coeficientes | `coefficients_view` | `CoefficientsTab.svelte` (tabla, `CoefficientBarChart`, análisis, recomendaciones) |
| Predicciones | `prediction_form`, `predict_one` | `PredictionForm.svelte`, `PredictionResultCard.svelte`, `ProbabilityTable.svelte` |
| Exportar | `export_model`, `generate_code(RegressionCode)` | `ExportPanel.svelte` + `CodeBlock` (F1) |

Paletas para Plotly (hex): Blues (matriz), viridis (error, superficie), Set3
(`#8dd3c7 #ffffb3 #bebada #fb8072 #80b1d3 #fdb462 #b3de69 #fccde5 #d9d9d9 #bc80bd #ccebc5 #ffed6f`),
RdBu extremos `#67001f`/`#053061`, colores CSS de las curvas (aqua `#00ffff`, darkorange `#ff8c00`,
cornflowerblue `#6495ed`, red, green `#008000`, purple `#800080`, brown `#a52a2a`, pink `#ffc0cb`,
gray `#808080`, olive `#808000`, deeppink `#ff1493`, navy `#000080`, darkgreen `#006400`), lightcoral
`#f08080`, lightblue `#add8e6`, skyblue `#87ceeb`, wheat `#f5deb3`, #4ecdc4/#ff6b6b (coeficientes),
#3274a1 (precisión por clase). Los valores exactos que pintó matplotlib están en cada fixture.

---

## 9. Reparto de tareas y checklist

Orden: core → paridad → comandos → UI → contenido → smoke test. Las casillas de INVENTARIO §4 se
marcan al cerrar la fila 6.

### 9.1 `rust-ml-engineer` (`crates/mltutor-core`, `crates/mltutor-export`)

1. [ ] `model::Predictor`, `ModelError` (thiserror, mensajes en español), `Target`.
2. [ ] `linear::Linear` sobre linfa-linear. Test: `linear_models/{diabetes,california_housing,csv_numeric_regression}_linear.json` (coef/intercepto 1e-8, predicciones 1e-9) usando sus índices.
3. [ ] `linear::Logistic` sobre linfa-logistic (binaria y multinomial), con `max_iter`, mapeo de la
   clase positiva, `converged: bool` y cancelación. Verificar la escala del objetivo (C frente a
   α). Tests: frente a `converged_reference` (1e-5) en los 4 casos y frente al modelo de la app con
   `meta.tolerance.proba_abs_vs_app`. Medir y anotar en MEMORY lo que dé Cáncer.
4. [ ] `metrics::{classification, regression}` y `interpretation`. Tests: informe, matriz, métricas e
   `interpretation_md` frente al `st.info` de `views.evaluation`.
5. [ ] `curves` (ROC/PR/AUC/AP, binaria, multiclase, micro, macro, weighted). Tests: todas las curvas
   y umbrales de `evaluation.curves`, **usando las probabilidades del fixture** (aísla el optimizador).
6. [ ] `probabilities::{distribution, threshold_analysis}`. Tests: histogramas de las barras de
   `views.viz_probs`, `threshold_analysis` (con las probabilidades del fixture).
7. [ ] `residuals::analysis`. Tests: `evaluation.residuals` y los textos de `views.viz_residuals`.
8. [ ] `grid::decision_axes` + `decision_grid(model, i, j)`. Tests: ejes exactos, `notices` y Z
   (superficie 1e-9; frontera ≤ 0,5 % de celdas) de cada `contourf.grid` del fixture.
9. [ ] `form::build` + `predict_one`. Tests: `prediction_form.controls` (kind, opciones, min/max/paso/
   valor por defecto) y las predicciones `default`/`custom` (1e-5).
10. [ ] `coefficients::view` (orden, efectos, niveles, cuantiles, extremos). Tests: tablas y textos de
    `views.coefficients*` (con D27 aplicado, los tests comparan las columnas que no cambian).
11. [ ] `codegen` (minijinja + `pyrepr`). Test: `code/regression.json` exacto.
12. [ ] `mltutor-export`: `onnx::{linear_regressor, linear_classifier}` y `package::python_zip`.
    Tests: estructura frente a `onnx_reference.nodes`; salidas con el evaluador propio o con ORT en
    el job de Python de F6; el `modelo.json` generado igual a `export_package.model_json` (salvo
    coeficientes, con tolerancia) y el script igual a `export_package.script`.
13. [ ] Benchmark (`#[ignore]`): California, entrenamiento + todas las vistas < 100 ms cada una.

### 9.2 `tauri-ui-engineer` (`src-tauri` + `ui`)

Backend:
1. [ ] `session::LessonModels`, DTO de §5.3 y errores nuevos (`ModelStale`, `InvalidTarget`,
   `NotClassifier`, `NotRegressor`, `GridTooSmall`, `InvalidInput`, `Export`).
2. [ ] Comandos de §5.4 (build.rs + capabilities), evento `task-progress`, `seq` en `train_model`.
3. [ ] `export_model` con diálogo nativo (filtros "Modelo ONNX (*.onnx)" y "Paquete Python (*.zip)").
4. [ ] `--smoke-test`: entrenar Iris logística y Diabetes lineal y pedir cada vista; Cáncer para la
   binaria. `--screenshot --scenario` para las capturas de §0.

Frontend:
5. [ ] Store de la lección (tipo, `max_iter`, `test_size`, subvista D13, par de características,
   opciones de ROC, valores del formulario).
6. [ ] Componentes de §3 (C1–C9, C11, C12) en `ui/src/lib/components/ml/`, genéricos y con tests de
   vitest (props → trazas Plotly).
7. [ ] Pestañas 1–6 de la lección en `ui/src/lib/lessons/regression/`.
8. [ ] Descargas PNG de cada gráfico con los nombres de Python (`matriz_confusion.png`,
   `precision_por_clase.png`, `predicciones_vs_reales.png`, `distribucion_errores.png`,
   `frontera_decision.png`).
9. [ ] Capturas Rust de las pantallas de §0 y comparación.

### 9.3 `ml-lesson-author`

1. [ ] Cargar `content/regresion/*` y `content/componentes/*` (Markdown + TOML) sin reescribir.
2. [ ] Si se aprueban D19, D26 y D29: redactar los textos nuevos (aviso de no convergencia, bloqueo
   de la logística, script completo de entrenamiento `regresion_entrenamiento.py.j2`).

### 9.4 Revisión

`test-runner` → `rust-reviewer` → `ml-correctness-reviewer` (paridad logística, curvas, rejilla).

---

## 10. Decisiones para el usuario

Se numeran a continuación de D18. Si no se dice lo contrario, se aplica la recomendación.

| # | Decisión | Recomendación |
|---|---|---|
| D19 | **Logística que no converge** (Cáncer con 1000 iteraciones sobre datos sin escalar). Python lo ignora en silencio y su modelo queda a 0,20 del óptimo; Rust no puede reproducir ese punto intermedio de L-BFGS | Aceptarlo: la paridad se mide frente a la referencia convergida. Mostrar en Entrenamiento un aviso nuevo: "⚠️ El optimizador no ha convergido en {max_iter} iteraciones. Prueba a aumentar el máximo de iteraciones o a escalar las características." |
| D20 | **ROC "weighted"**: el inventario la pide, pero Python solo tiene macro y micro. Además, en multiclase Python usa las clases presentes en el test y desalinea las columnas de `predict_proba` si falta alguna | Mantener macro y micro (paridad); core calcula weighted para F3/F4 sin mostrarlo; corregir el inventario. Usar las clases **del modelo** y omitir las ausentes del test con una nota |
| D21 | **Tabla "Comparación de Umbrales"**: por coma flotante (`0.30000000000000004`) solo muestra 0,5 | Mostrar 0,3, 0,5 y 0,7, como pretendía el código |
| D22 | **"R² = 0.0000"** en Predicciones vs Valores Reales: lee una clave que nunca existe | Mostrar el R² real del test |
| D23 | **Colores de la frontera**: el fondo, los puntos y la leyenda usan colores distintos para la misma clase | Un color por clase (Set3 en multiclase y los extremos de RdBu en binaria), el mismo en fondo, puntos y leyenda |
| D24 | **Rejilla de la frontera/superficie**: ±1 en cada eje y paso 0,02 (con features escaladas, como Diabetes, los datos ocupan el 10 % central) | Paridad exacta, incluidos los avisos "Malla creada…": coincide con `DecisionBoundaryDisplay` (eps = 1), que es lo que enseña el código |
| D25 | **Lineal sobre un dataset de clasificación**: la predicción trunca `int(ŷ)` a una clase | Mostrar "Valor: {ŷ:.4f}" (la evaluación ya es de regresión); conservar el aviso |
| D26 | **Logística sobre un objetivo de regresión**: en Diabetes entrena 192 clases y la ROC falla; en California da un error en inglés | Bloquear el entrenamiento con un mensaje en español ("La regresión logística necesita una variable objetivo con clases. Elige 📈 Regresión Lineal o un dataset de clasificación.") |
| D27 | **Coeficientes**: (a) el intercepto multiclase es siempre el de la clase 0; (b) la lista "MÁS aumentan/disminuyen" numera con la posición original de la característica; (c) "Fuerza del Efecto" es asimétrica (−2,5 → "Moderado") | Corregir las tres: intercepto de la clase elegida, numerar 1, 2, 3 y fuerza simétrica por |β| (2, 1, 0,5, como explica el expander) |
| D28 | **Exportar** (concreción de D1): Python ofrece "📄 Generar Código" y "📥 Descargar Modelo (Pickle)" | Tres botones: "📄 Generar Código", "💾 Exportar a ONNX" y "📦 Paquete Python (.zip)" (modelo.json + reconstruir_modelo.py, que crea el .pkl, + LEEME.txt). Sin pickle directo |
| D29 | **Código de "📄 Generar Código"**: el de Python no funciona (`model` sin definir, sin `import numpy`, nombre "logística" también en lineal) y no reproduce los hiperparámetros (INVENTARIO §7) | Sustituirlo por un script completo y correcto que cargue el dataset, haga el mismo split (`random_state=42`), entrene con los hiperparámetros elegidos y evalúe (`ml-lesson-author`). Hasta entonces, el literal |
| D30 | **Código de la frontera**: usa `X_train[:, [0, 1]]` y todas las características, sea cual sea el par elegido | Rellenar la plantilla con los índices y los nombres del par elegido |
| D31 | **Entrada del formulario**: Python la convierte a float32 antes de predecir | Calcular en f64 (diferencia ≤ 1e-6 relativa, invisible con 4 decimales) |
| D32 | **Modelo y dataset**: tras cambiar de dataset, Python sigue mostrando el modelo anterior con los metadatos del nuevo | El modelo guarda su dataset; las pestañas 2–6 muestran "🏋️ Modelo entrenado con {dataset}" y, si la selección cambió, un aviso para reentrenar. El formulario usa siempre los metadatos del dataset de entrenamiento |
| D33 | **Formato de las tablas**: el informe por clase (`{:.4f}`, support `{:.0f}`) y las probabilidades (`{:.4f}`) usan `Styler`, que Qt descarta (se ven con 6 cifras) | Aplicar el formato que pretendía Python |
| D34 | **Iconos de los avisos**: Qt descarta `icon=` ("Excelente" sin 🌟, "Bueno" sin ⚠️) | Mostrarlos, como hacía la versión Streamlit |
| D35 | **Capturas Qt de F2** (6,8 MB ya cuantizadas) | Versionarlas, como en F1 |

---

## 11. Riesgos

| Riesgo | Mitigación |
|---|---|
| La escala del objetivo de linfa-logistic no es exactamente la de sklearn (½, media o suma) | Test frente a `converged_reference` en 4 datasets; si difiere, ajustar α = 1/(2C) o 1/(C·n) y documentarlo en MEMORY |
| Logística sin escalar mal condicionada (Cáncer, Titanic con Tarifa): linfa/argmin puede dar otro número de iteraciones, otro punto o fallar la búsqueda lineal | D19; tolerancias por caso; el modelo 2D de la frontera hereda el problema (por eso la tolerancia de celdas) |
| Predicciones casi empatadas (CSV, margen 2e-5) cambian una métrica | Los tests de métricas usan las predicciones del fixture cuando `min_proba_margin < 2·tol` |
| Reproducir `np.arange` y los umbrales de `roc_curve` al bit | Tests exactos de ejes y umbrales con los fixtures |
| La rejilla grande (2 M puntos) si se materializa | Calcular solo ejes y predecir la rejilla final (≤ 40 000) |
| Plotly con 40 000 celdas + puntos en WebKitGTK | `heatmap` (una traza) en lugar de `contour`; `scattergl` |
| Paquete Python: versiones de sklearn futuras cambian atributos internos (`n_iter_`, `classes_`) | El script lo comprueba al ejecutarse (asserts); job de F6 con sklearn fijado |
| `pyrepr` de floats distinto de `repr()` de Python | Rust usa ryu (mínima de ida y vuelta, como Python); test con el fixture |
| Fixtures de 3,9 MB más | Aceptable (las vistas son la especificación); California es el mayor (1,1 MB) |

---

## 12. Hallazgos de la versión Python

1. La logística **no converge en Cáncer** (1000 iteraciones, datos sin escalar): ConvergenceWarning
   invisible y probabilidades a 0,20 del óptimo; Titanic converge, pero a 2,6e-3 del óptimo.
2. La logística sobre Diabetes **entrena** (objetivo de floats enteros = 192 clases) y la ROC rompe;
   en California falla con un error en inglés.
3. La tabla de umbrales solo muestra 0,5 (`np.arange(0.1, 1.0, 0.1)` no contiene 0,3 ni 0,7 exactos).
4. "R² = 0.0000" siempre en Predicciones vs Valores Reales (`metrics_lr` no existe).
5. En la frontera, fondo, puntos y leyenda tienen colores distintos para la misma clase (Set3 con
   `levels=n` crea n+1 bandas; la leyenda binaria usa colores fijos que no son los de RdBu).
6. La frontera/superficie **reentrena** un modelo de 2 características sobre train; no fija el resto.
7. La predicción con un modelo lineal en clasificación trunca con `int()`.
8. La rama de clasificación/regresión del formulario la decide la tarea del dataset, y la de la
   evaluación, el tipo de modelo.
9. Coeficientes: intercepto multiclase fijo en la clase 0, numeración por posición original y
   fuerza del efecto asimétrica; la advertencia de "coeficientes extremos" sale casi siempre.
10. Qt descarta los `icon=` de los avisos y los formatos de `Styler`.
11. El análisis de residuos lanza `NameError` con ≤ 10 muestras de test (si la media de los residuos está cerca de 0).
12. En Cáncer la clase positiva (1) es *benign*: la ROC binaria y el baseline miden "benigno".
13. `generate_regression_code` no es ejecutable y `generate_decision_boundary_code` ignora el par
    elegido.
14. Tras cambiar de dataset, el modelo anterior sigue vivo en las pestañas 2–6.
15. `np.arange` calcula cada valor como `min + i·((min + h) − min)`; la KDE de `histplot` usa
    `cut=0` (la del pairplot, `cut=3`).

---

## 13. Cambios propuestos en otros documentos

No se han editado (los aplica quien cierre la fase).

**INVENTARIO.md §4**: precisar "📉 ROC (con curvas por clase y promediados macro y micro)" (no
weighted, D20); "🌐 superficie de predicción con 2 características (modelo reentrenado con esas dos)";
"Exportar: ONNX y paquete Python (D1/D28) y código generado"; añadir "📊 distribución de
probabilidades y 🎯 análisis de umbrales (binaria)". §6: la superficie de Redes es la única que fija
el resto en su media. §0: la casilla de diálogos de guardado se completa con `export_model`.

**PLAN.md, Fase 2**: enlazar este documento y las capturas `capturas/qt/fase2/`; añadir D19–D35 a la
tabla de decisiones (como D9–D17, con enlace a §10); cambiar "ROC/PR (macro, micro y weighted)" por
"macro y micro (weighted solo en core)"; añadir `mltutor-export` (ONNX + ZIP) y `codegen` a las
tareas.

**tests/parity/README.md**: ya actualizado con `linear_models/` y `code/`.
