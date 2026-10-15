# Fixtures golden de paridad

Resultados de referencia generados con la versión Python/Qt de MLTutor. Los tests de
Rust los comparan con lo que calcula `mltutor-core`. Los genera
`tools/export_golden.py`, que llama a las funciones de la app
(`load_builtin_dataset`, `preprocess_data`, `train_linear_model`,
`evaluate_*_model`…) y no a reimplementaciones.

## Regenerar

```sh
uv sync --locked --extra dev
uv run python tools/export_golden.py                    # todos los casos
uv run python tools/export_golden.py --only datasets    # un módulo
uv run python tools/export_golden.py --only split/iris  # un caso (repetible)
uv run python tools/export_golden.py --check            # no escribe; falla si algo cambiaría
uv run python tools/export_golden.py --list
```

La salida es determinista. Si el contenido de un fixture no cambia, el archivo no se
toca, ni siquiera su fecha. California (`fetch_california_housing`), Diamantes y MPG
(`seaborn.load_dataset`) necesitan red la primera vez y luego quedan en la caché de
sklearn y seaborn. Si cambias versiones de sklearn o numpy, regenera los fixtures,
revisa el diff y explica en MEMORY.md cualquier diferencia que no sea trivial.

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
