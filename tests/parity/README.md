# Fixtures golden de paridad

Resultados de referencia generados con la versión Python/Qt de MLTutor. Los tests de
Rust los comparan con lo que calcula `mltutor-core`. Los generó `tools/export_golden.py`
(con `golden_render.py`, `golden_trees.py`, `golden_knn.py` y `golden_nn.py`), que llamaba
a las funciones de la app (`load_builtin_dataset`, `preprocess_data`, `train_linear_model`,
`evaluate_*_model`…) y no a reimplementaciones.

## Fixtures congelados (desde 2026-10-21)

La app Python y esos generadores **ya no están en el árbol**: viven en los tags `python-final` y `v1.0.0` (el mismo commit) y en
la rama `codex/desktop-migration-study`. Estos archivos son referencias congeladas: no se
editan a mano y ningún test los reescribe, salvo los textos de plantillas (abajo). Las
rutas `mltutor/…` que aparecen en ellos (y en los comentarios de `content/`) se refieren
al código de `v1.0.0`.

### Textos de plantillas: `MLTUTOR_BLESS=1`

Algunos tests comparan el texto exacto de una plantilla `content/code/*.j2` renderizada en
Rust con el texto que generó jinja2 en Python: `code/*.json` (`text`, `default_text`,
`script`), `trees/*.json` (`python_code.app_text`, `python_code.corrected_text`,
`export_package.script`), `knn/*.json` y `linear_models/*.json` (`export_package.script`,
`export_package.leeme`) y `nn/exact/*.json` (`export_package.script`, `numpy_script`,
`leeme`). Si cambias una plantilla a propósito:

```sh
MLTUTOR_BLESS=1 cargo test --workspace --all-targets   # reescribe solo esos textos
git diff tests/parity                                  # revisa el cambio
cargo test --workspace --all-targets                   # sin BLESS, en verde
```

Lo hace `crates/mltutor-testkit` (`assert_template_text`): sustituye solo la cadena JSON del
campo comparado, con la codificación de `json.dumps(ensure_ascii=False)`, y deja el resto del
archivo byte a byte. Sin cambios en las plantillas, el modo bless no toca nada. Si un test que
lee esos textos (p. ej. `real_trees_match_the_fixture_facade`) se ejecuta antes que el que los
reescribe, falla en esa pasada: repite el comando. Los fixtures numéricos (datos, modelos,
métricas, vistas) nunca se reescriben.

El modo bless solo garantiza que el texto coincide con lo que genera Rust. Que los scripts
sigan ejecutándose y reproduzcan los resultados de scikit-learn/Keras lo comprueba
`tools/verify_exports.py` en el job `exports` del CI (BUILD_EXECUTABLES.md).

### Regenerar los fixtures numéricos

Hace falta la app Python. Desde un checkout del tag `v1.0.0` (o de la rama
`codex/desktop-migration-study`), por ejemplo en un worktree aparte:

```sh
git worktree add ../mltutor-v1 v1.0.0
cd ../mltutor-v1
uv sync --locked --extra dev
uv run python tools/export_golden.py                    # todos los casos
uv run python tools/export_golden.py --only datasets    # un módulo
uv run python tools/export_golden.py --only split/iris  # un caso (repetible)
uv run python tools/export_golden.py --check            # no escribe; falla si algo cambiaría
uv run python tools/export_golden.py --list
```

y copia después a este árbol los archivos de `tests/parity/` que cambien. Si alguna plantilla
cambió desde la v1.0.0, cópiala antes a ese checkout. La salida es determinista. California
(`fetch_california_housing`), Diamantes y MPG (`seaborn.load_dataset`) necesitan red la
primera vez. Si cambias versiones de sklearn o numpy, revisa el diff y explica en MEMORY.md
cualquier diferencia que no sea trivial.

## Formato

`tests/parity/<módulo>/<caso>.json`, JSON con indentación y UTF-8:

- `meta`: `format_version`, `generated` (fecha), `versions` (python, numpy, pandas,
  scipy, scikit-learn, seaborn, plataforma), `seed`, `tolerance` (por campo) y
  `tolerance_rationale`.
- `input`: dataset, parámetros, función de la app que se ha llamado e hiperparámetros
  completos (`get_params()`).
- `expected`: los resultados.

Los índices de filas son **posicionales** (0..n-1), referidos al CSV del dataset.

| Ruta | Contenido |
|---|---|
| `datasets/<nombre>.json` | Columnas (`columns` = características; `label_columns` = columnas de etiqueta que no son característica, p. ej. `name` en MPG por D8; `csv_columns` = cabecera del CSV), dtypes, forma, nombres legibles (`feature_names`), clases, tipo de tarea, `dataset_info`, resumen del objetivo, primeras filas y hashes |
| `datasets/data/<nombre>.csv` | Datos completos (columnas de `X` y el objetivo al final), listos para `include_bytes!` |
| `split/iris.json` | Permutación, índices de train y test, y recuento por clase con `test_size=0.3`, `random_state=42` |
| `regression/iris_logistic.json` | `LogisticRegression(max_iter=1000, random_state=42)`: coeficientes, intercepto, predicciones, probabilidades, `decision_function` y métricas |
| `regression/diabetes_linear.json` | `LinearRegression()`: coeficientes, intercepto, predicciones y MSE/RMSE/MAE/R² |
| `stats/<nombre>.json` | Pestaña 📊 Datos: vista previa, `describe`, distribución del objetivo, matriz de correlación (row-major, `null` = NaN) y pairplot: selección por defecto, muestreo, KDE de seaborn (Scott, 200 puntos, `common_norm`) e histogramas de la diagonal |
| `csv/<caso>.json` | Cargador de CSV: archivo de entrada (texto y base64), errores, tipos, sugerencia de tarea y salida de `load_dataset_from_file` con `auto`, Clasificación y Regresión |
| `linear_models/<caso>.json` (9) | Fase 2, lección 📊 Regresión: la lección real renderizada sin interfaz (`tools/golden_render.py`). `expected.views` = lo que ve el estudiante en cada pestaña y subvista (textos, números ya redondeados, tablas, controles y datos de cada figura: líneas, barras, puntos, rejillas de contorno, colores, títulos y ejes). Además: modelo (coeficientes, intercepto, `n_iter`), `converged_reference` (logística convergida), predicciones, probabilidades, `decision_function`, informe por clase, matriz de confusión, ROC/PR con umbrales por clase, micro y macro, análisis de umbrales, residuos, modelos de 2 características de la frontera/superficie, formulario de predicción (controles y vector float32 real), `onnx_reference` (skl2onnx + `onnx.reference`) y `export_package` (paquete Python D1 ejecutado de verdad). Casos: Iris, Cáncer y Titanic (logística), Diabetes y California (lineal), dos CSV y dos desajustes de tipo de modelo |
| `code/regression.json` | Plantillas `content/code/*.py.j2` renderizadas con jinja2 (`pyrepr` = `repr`) frente al texto exacto de la app |
| `trees/<caso>.json` (19) | Fase 3, lección 🌲 Árboles de Decisión, renderizada como en F2. `expected.tree` = arrays completos de `tree_` (children, feature, threshold, impurity, n_node_samples, weighted_n_node_samples, value y value_counts); `reference_impl` = comparación con `tools/cart_reference.py` (transcripción de sklearn 1.7.2) y la traza de cada nodo: orden en que el RNG sortea las características, mejor mejora por característica y **empates** (`exact_ties_lost`). Además: predicciones, probabilidades, hojas, caminos de decisión, métricas y curvas, importancia nativa y por permutación (con la semilla y los índices barajados), `export_text`, disposición y colores de `plot_tree`, código Python del árbol (literal y corregido, D44), árboles 2D de la frontera/superficie, `onnx_reference` (TreeEnsemble*) y el paquete Python ejecutado. Gini, entropy, squared_error, friedman_mse y absolute_error; profundidades 1–10; Iris, Vino, Cáncer, Titanic, Diabetes, California y dos CSV; tres desajustes de tipo de árbol. Escritos con las listas de escalares en una línea (`golden_trees.dumps_compact`) |
| `code/trees.json` | Plantillas literales de la lección (`visualizar_arbol`, `texto_arbol`, `camino_decision`, `usar_modelo_arbol`) frente al texto de la app |
| `knn/<caso>.json` (19) | Fase 4, lección 🔍 K-Nearest Neighbors, renderizada como en F2/F3. `expected.neighbours_test` = motor (`kd_tree`/`brute`), índices en orden y distancias de los vecinos de cada fila de prueba, empates en el K-ésimo vecino y `reference_impl` = comparación con `tools/knn_reference.py` (kd-tree de sklearn 1.7.2 con `nth_element` de libc++, heap y `simultaneous_sort`; `idx_array` del árbol). Además: predicciones, probabilidades, métricas y curvas, permutación, análisis de distancias entre clases (D56, D67), modelos 2D de la frontera/superficie con la rejilla completa (todas las filas, D63), el lienzo interactivo (reglas del JS frente a sklearn), `onnx_reference` (skl2onnx, con su error en `weights='distance'`), `onnx_proposal_D57` (grafo propio ejecutado) y el paquete Python ejecutado. Iris, Vino, Cáncer, Titanic, Diabetes, California y dos CSV; K 1–20, uniform/distance, minkowski/euclidean/manhattan; un caso de K > n_train. `X_train` e `idx_array` como hash por encima de 1 000 filas |
| `nn/exact/<caso>.json` (19) | Fase 5, lección 🧠 Redes Neuronales, **nivel (a) de paridad (D69)**: la lección real renderizada sin interfaz con `keras.Model.fit` envuelto para `shuffle=False` y dropout 0. `expected.data` (filas exactas que ve Keras, la validación son las ÚLTIMAS filas), `compile` (pérdida y métrica que resuelve Keras), `initial_weights` (Glorot de Keras), `first_batch` (gradientes), `steps` (pesos tras los pasos 1–3), `history`, `callbacks` (EarlyStopping/ReduceLROnPlateau y márgenes de cada decisión), `final_weights` y `reference` = repetición con `tools/nn_reference.py` (NumPy f32) y su acuerdo época a época (`agree_epochs_rel_1e-3`). Además: predicciones de prueba, pesos, activaciones, superficie (400 celdas con la salida de la red), formulario, `onnx_proposal_D75` (MLP ONNX propio ejecutado), `export_package` (paquete Keras ejecutado en los casos `full`) y el diagnóstico. `iris_dropout_masks.json`: dropout con máscaras dadas. Arrays de más de 1 000 filas en base64 f32 |
| `nn/stat/<caso>.json` (11) | **Nivel (b)**: el entrenamiento real de la app (dropout 0,2, barajado) con 5 semillas; métricas finales, `bands` y `acceptance` para la media de Rust sobre sus semillas |
| `nn/diagnostics.json` | `neural_network_diagnostics` + `show_training_history_tab` sobre 14 historiales sintéticos: lógica pura, salida exacta |
| `nn/controls.json` | 🏗️ Arquitectura y ⚙️ Entrenamiento sin entrenar: valores por defecto por dataset, mensajes de cada combinación de controles, distribución mostrada frente a la real (D73), código de arquitectura literal frente a `red_arquitectura.py.j2` y si compila, pestañas sin modelo |
| `nn/split_stratified.json` | `train_test_split(test_size=0.2, random_state=42, stratify=y)` en 8 datasets y la comprobación de `nn_reference.stratified_split` |
| `nn/errors.json` | Activaciones de salida que Python deja elegir y no puede entrenar (D71) y qué hacen después las otras pestañas |

`data_sha256_f64` es el SHA-256 de los valores en f64 little-endian, fila a fila
(características en el orden de `columns` y después el objetivo; los booleanos valen 0/1).
Se puede recalcular en Rust desde el CSV ya parseado. `csv_sha256` es el hash de los
bytes del archivo.

## Política de tolerancias

- **Datos y splits: exactos.** Los floats del CSV usan la representación mínima de ida
  y vuelta de Python, así que se parsean a f64 sin pérdida.
- **Soluciones cerradas (OLS): relativas de 1e-8 a 1e-9.** Solo cuentan las diferencias
  de BLAS y de factorización.
- **Optimizadores iterativos (L-BFGS de la logística).** `regression/iris_logistic.json`
  (F0) indica 1e-3 en coeficientes y 1e-4 en probabilidades, pero el spike midió 1,6e-3 entre
  linfa y sklearn por defecto: los fixtures de `linear_models/` usan **2e-3** (o más, por caso,
  según `converged_reference.max_abs_proba_diff_vs_app`) frente al modelo de la app y **1e-5**
  frente a `converged_reference`. Si la app no converge (Cáncer), solo es normativa la
  referencia convergida. sklearn para con `tol=1e-4`; con otro solver el óptimo solo coincide
  hasta esa precisión. Las predicciones deben coincidir exactamente si ninguna muestra
  está cerca de un empate (`min_proba_margin`).
- **Árboles: estructura exacta.** sklearn sortea el orden de las características en cada nodo y se queda con la primera que alcanza la mejor mejora, y los empates son frecuentes. Rust reproduce el RNG y el orden de las sumas (`tools/cart_reference.py`), así que la estructura, los umbrales, las predicciones y las probabilidades coinciden bit a bit; las impurezas, con 1e-12 relativo (la wheel arm64 de sklearn usa FMA).
- **Métricas:** exactas (1e-12) cuando las predicciones son exactas. Si no, se derivan
  de la tolerancia de las predicciones.
- **Textos de las vistas:** exactos salvo los números impresos, que se comparan
  interpretándolos con la precisión mostrada (±1 en la última cifra).
- Cada fixture indica su tolerancia en `meta.tolerance`. Si Rust no puede cumplirla (otro
  desempate, otro RNG), se documenta en MEMORY.md. La tolerancia no se relaja sin
  justificarla.
- `code/trees_training.json`: the decision tree training script (`arbol_entrenamiento.py.j2`) rendered and executed for 16 cases, checked against `trees/*.json`.
- `code/knn_training.json`: the KNN training script (`knn_entrenamiento.py.j2`) rendered and executed for 19 cases, checked against `knn/*.json`.
- **Redes neuronales (F5, D69): dos niveles.** (a) Con los pesos iniciales de Keras, sin barajar y sin dropout, el entrenamiento es determinista: gradientes y pesos de los primeros pasos a 1e-6, historial a 1e-4 relativo mientras `reference.relative_diff_by_epoch` lo permita (las trayectorias caóticas, como `wine_rmsprop_lr03_chaotic`, divergen tras la primera época y solo cuentan esas épocas), épocas y mejor época exactas salvo márgenes de `val_loss` < 1e-5, y las vistas calculadas con los pesos finales a precisión f32. (b) Con otro RNG solo se comparan distribuciones: la media de Rust sobre 5 semillas debe caer en `acceptance`.
- `code/nn_training.json`: the Keras training script (`red_entrenamiento.py.j2`) rendered and executed for 6 cases: same X_test/y_test (1e-12) and parameter count as the app, metrics inside the `nn/stat` band.
