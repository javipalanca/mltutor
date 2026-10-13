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
| `datasets/<nombre>.json` | Columnas, dtypes, forma, nombres legibles (`feature_names`), clases, tipo de tarea, `dataset_info`, resumen del objetivo, primeras filas y hashes |
| `datasets/data/<nombre>.csv` | Datos completos (columnas de `X` y el objetivo al final), listos para `include_bytes!` |
| `split/iris.json` | Permutación, índices de train y test, y recuento por clase con `test_size=0.3`, `random_state=42` |
| `regression/iris_logistic.json` | `LogisticRegression(max_iter=1000, random_state=42)`: coeficientes, intercepto, predicciones, probabilidades, `decision_function` y métricas |
| `regression/diabetes_linear.json` | `LinearRegression()`: coeficientes, intercepto, predicciones y MSE/RMSE/MAE/R² |

`data_sha256_f64` es el SHA-256 de los valores en f64 little-endian, fila a fila
(características en el orden de `columns` y después el objetivo; los booleanos valen 0/1).
Se puede recalcular en Rust desde el CSV ya parseado. `csv_sha256` es el hash de los
bytes del archivo.

## Política de tolerancias

- **Datos y splits: exactos.** Los floats del CSV usan la representación mínima de ida
  y vuelta de Python, así que se parsean a f64 sin pérdida.
- **Soluciones cerradas (OLS): relativas de 1e-8 a 1e-9.** Solo cuentan las diferencias
  de BLAS y de factorización.
- **Optimizadores iterativos (L-BFGS de la logística): 1e-3 en coeficientes y 1e-4 en
  probabilidades.** sklearn para con `tol=1e-4`; con otro solver el óptimo solo coincide
  hasta esa precisión. Las predicciones deben coincidir exactamente si ninguna muestra
  está cerca de un empate (`min_proba_margin`).
- **Métricas:** exactas (1e-12) cuando las predicciones son exactas. Si no, se derivan
  de la tolerancia de las predicciones.
- Cada fixture indica su tolerancia en `meta.tolerance`. Si Rust no puede cumplirla (otro
  desempate, otro RNG), se documenta en MEMORY.md. La tolerancia no se relaja sin
  justificarla.
