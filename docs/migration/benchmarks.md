# Benchmarks (Fase 7)

Medidos el 2026-10-20 en el Mac de desarrollo (Apple Silicon, arm64), con `cargo bench`
(perfil release: LTO fat, `codegen-units = 1`). Se da la mediana de criterion. Mientras se
medía, otros agentes compilaban en la misma máquina, así que los números de pocos milisegundos
tienen un ±10–20 % de ruido. Las cifras de Python se tomaron con el `.venv` del repo
(scikit-learn 1.7.2, pandas 2.3.3, NumPy 1.26.4 con Accelerate, TensorFlow 2.16.2 / Keras 3.12 en
CPU). En cada caso se da la mediana de 3–7 ejecuciones tras un calentamiento, con los mismos
datos y el mismo split (`test_size=0.3`, `random_state=42`).

## Cómo ejecutarlos

```sh
cargo bench -p mltutor-core --bench core      # unos 2 min; filtro: -- 'knn/'
cargo bench -p mltutor-nn --bench nn          # unos 1,5 min
cargo bench -p mltutor-export --bench onnx    # unos 30 s
cargo bench --no-run                          # solo compilar (lo que debe comprobar CI)

# Comparar con una referencia guardada
cargo bench -p mltutor-core --bench core -- --save-baseline antes
# … cambio …
cargo bench -p mltutor-core --bench core -- --baseline antes
```

criterion es solo dev-dependency (0.7, sin features por defecto), así que no entra en el binario
de la app. Los presupuestos por comando (con `assert!`) siguen en los tests `#[ignore]`
`tests/bench_*.rs` (`cargo test --release -p mltutor-core --test bench_trees -- --ignored --nocapture`,
y lo mismo con `bench_regression`, `bench_knn`, `bench_california` y `mltutor-nn/tests/bench_nn.rs`).

Todo es determinista: datasets incrustados, el split de la app, semillas fijas en las redes y
datos sintéticos generados con un splitmix64.

## Resultados

### Datos (`benches/core.rs`)

| Benchmark | Rust | Python | Notas |
|---|---:|---:|---|
| `csv/read/100000` (9,3 MB, 10 columnas, 1 de texto) | 50 ms | 43 ms (pandas) | Rust decide el tipo al leer y hace dos pasadas (texto); pandas tiene un lector en C |
| `csv/read/500000` (46 MB) | 255 ms | 204 ms (pandas) | ~180 MB/s |
| `explore/california/load` (inflar + parsear + tabla) | 12,5 ms | 7,9 ms (`read_csv`) | incluye inflar el CSV comprimido (ver abajo) |
| `explore/california/inflate_csv` (1,8 MB) | 4,4 ms | — | coste de la compresión de los datasets |
| `explore/california/describe` | 3,9 ms | 6,3 ms (`describe`) | suma por pares y cuantiles como pandas |
| `explore/california/correlation` | 4,2 ms | 2,0 ms (`corr`) | Welford de pandas (`nancorr`) por paridad |

### Modelos lineales

| Benchmark | Rust | scikit-learn | Notas |
|---|---:|---:|---|
| `linear/ols/Iris` | 4 µs | 161 µs | |
| `linear/ols/BreastCancer` | 173 µs | 309 µs | |
| `linear/ols/Digits` | 2,4 ms | 2,0 ms | de rango deficiente (píxeles constantes): pasa por la SVD propia |
| `linear/ols/CaliforniaHousing` | 0,75 ms | 1,3 ms | |
| `linear/ols_wide/1500x300` | 32 ms | 23 ms | |
| `linear/ols_wide/300x300` (rango 299) | 14 ms | 10 ms | SVD |
| `linear/ols_wide/1500x1500` (rango 1499) | **1,41 s** (antes 14,0 s; 1,3 s en el test de release) | 0,33 s | SVD propia; sklearn usa `gelsd` de Accelerate, multihilo |
| `linear/logistic/Iris` | 223 µs | 4,0 ms | Newton |
| `linear/logistic/BreastCancer` | 0,93 ms | 484 ms | sklearn agota `max_iter=1000` sin converger; Rust converge (9 iteraciones) |
| `linear/logistic/Digits` | **0,54 s** (antes 0,98 s) | 77 ms | no comparable: el lbfgs de sklearn para a las ~139 iteraciones, lejos del óptimo (MEMORY); Rust hace 12 iteraciones de Newton hasta el óptimo |

### Árboles

| Benchmark | Rust | scikit-learn |
|---|---:|---:|
| `tree/fit/california_d10_squared_error` | 51 ms | 58 ms |
| `tree/predict/california_d10_test` (6 192 filas) | 0,20 ms | 0,36 ms |
| `tree/fit/iris_d3_gini` | 16 µs | 241 µs |
| `tree/fit/iris_d10_entropy` | 24 µs | 257 µs |

### KNN (K = 5, uniform, minkowski)

| Benchmark | Rust | scikit-learn | Notas |
|---|---:|---:|---|
| `knn/fit/CaliforniaHousing` | 2,0 ms | 2,3 ms | kd_tree |
| `knn/predict_test/CaliforniaHousing` | 1,6 ms | 9,9 ms | |
| `knn/decision_grid/CaliforniaHousing` (longitud/latitud, todas las filas) | 12 ms | 44 ms | sklearn: reentrenar en 2D y predecir ~38 000 celdas |
| `knn/fit/Digits` | 69 µs | 141 µs | fuerza bruta (64 características) |
| `knn/predict_test/Digits` | 17 ms | 57 ms | |
| `knn/decision_grid/Digits` (píxeles 20 y 28) | 15 ms | 29 ms | |

### Red neuronal (`benches/nn.rs`; ReLU, dropout 0,2, Adam, LR 0,001, `validation_split` 0,2)

| Benchmark | Rust | Keras 3.12 (CPU) |
|---|---:|---:|
| `nn/train/iris_10x10_100_epochs` (batch 64) | 7,5 ms | 2,4 s |
| `nn/train/california_10x10_10_epochs` (batch 64) | 66 ms | 1,37 s |
| `nn/train_large/california_5x256_1_epoch` (batch 32) | 1,08 s | 1,65 s |
| `nn/predict_batch/california_all_rows_10x10` (20 640 filas) | 3,9 ms | 159 ms (`predict`) |
| `nn/predict_batch/california_all_rows_5x256` | 187 ms | 316 ms (`predict`) |

Los tiempos de Rust incluyen todo `train_network` (escalado, split estratificado, Glorot y
evaluación final). En Keras solo se mide `fit`. Según MEMORY.md, California con la red por
defecto y 100 épocas tardaba unos 30 s en la app Python.

### Exportación ONNX (`benches/onnx.rs`: `onnx_model` + `to_bytes`)

| Modelo | Tiempo |
|---|---:|
| lineal (California) | 1,2 µs |
| logística (Dígitos, 10 × 64) | 3,2 µs |
| árbol Iris d3 | 4,5 µs |
| árbol California d10 | 133 µs |
| KNN Iris | 20 µs |
| KNN California (14 448 × 8 filas como initializer) | 0,46 ms |
| MLP Iris 10×10 | 11 µs |
| MLP California 5×256 | 0,59 ms |

La exportación no es un cuello de botella: el tiempo lo marca el diálogo de guardar.

## Cambios de rendimiento hechos en esta tarea

1. **OLS de rango deficiente** (`crates/mltutor-core/src/linear/lstsq.rs`): sustituye la SVD
   completa de linfa-linalg por una SVD propia (bidiagonalización de Golub–Kahan y QR implícito
   sobre la bidiagonal) que no forma U ni V. 1500×1500: de 14,0 s a ≈ 1,3–1,4 s (0,7 s de
   bidiagonalización, 0,07 s de barridos QR y el resto en la QR de los datos). Comprueba la
   cancelación en cada paso: responde en menos de 1 ms (test
   `linear::ols::tests::rank_deficient_fit_cancels_quickly`, que exige menos de 100 ms). La
   paridad con sklearn no cambia: los fixtures de OLS dan los mismos errores (≤ 4,3e-12 en
   predicción, tolerancia 1e-9), y los casos de columnas derivadas o matriz ancha quedan a
   ≤ 1,6e-15 de sklearn (tolerancia 1e-9).
2. **Cholesky de la logística** (`linear/logistic.rs::cholesky_solve`): por columnas, cuatro
   filas a la vez y slices contiguos en lugar de `l[[i, m]]`. Cada elemento conserva su orden de
   suma, así que el resultado es bit a bit igual (comprobado en los 22 624 coeficientes y
   probabilidades de Iris, Cáncer, Dígitos, Vino y Titanic). Dígitos: de 0,98 s a 0,54 s.
3. **Datasets comprimidos**: `build.rs` deflacta los 12 CSV (3,03 MB → 0,97 MB, unos 2 MB menos
   de binario) y `DatasetId::csv_bytes()` los inflacta al cargar. California tarda 4,4 ms más en
   inflarse; la carga pasa de unos 10,8 a 12,5 ms. Los bytes son idénticos: lo comprueba el
   SHA-256 de `parity_datasets`.

## Pendiente

- **Memoria del caso ancho de OLS** (n < p): `lstsq.rs` copia R escalada (m × p) y guarda los
  reflectores de la QR de Rᵀ como vectores aparte, con lo que la memoria transitoria se
  duplica. Se podrían guardar dentro de la propia matriz, pero no compensa mientras los CSV de
  la app tengan pocas columnas (revisión ML de F7, severidad baja).
- Paralelizar de forma determinista la bidiagonalización (las filas del bloque final son
  independientes) para OLS muy anchos. También se podría bajar el coste de la Hessiana de la
  logística (unos 19 ms por iteración de Newton en Dígitos).

## Propuesta de comprobación de regresiones en CI (sin implementar)

Los runners compartidos de GitHub tienen un ruido del 10–30 %, así que no merece la pena
comparar con criterion en cada push. Hay dos opciones baratas:

1. **Presupuestos holgados (recomendada):** un paso en el job de Linux o macOS, solo en push
   a `feature/rust-migration`:

   ```sh
   cargo test --release -p mltutor-core --test bench_trees --test bench_knn \
     --test bench_regression --test bench_california -- --ignored --nocapture
   cargo test --release -p mltutor-nn --test bench_nn -- --ignored --nocapture
   ```

   Estos tests ya comprueban presupuestos de 10 a 100 veces por encima de lo medido aquí (por
   ejemplo, `< 100 ms` por vista en California o `< 1 s` para la permutation importance de KNN):
   solo fallan ante una regresión grave, como un O(n²) accidental o perder la paralelización.
   Cuestan unos 2 o 3 minutos de compilación en release. Con más margen todavía, se podría
   añadir `ols_wide_design_matrices` con un `assert!(t < 10 s)` para el caso 1500×1500.
2. **criterion con referencia:** un workflow manual (`workflow_dispatch`) o semanal que ejecute
   `cargo bench -- --save-baseline main` en `main` y `--baseline main` en la rama, publique la
   salida en el resumen del job y avise (sin fallar) si algo empeora más de un 25 %. Para que la
   comparación sea fiable debe ejecutarse siempre en el mismo tipo de runner (macOS arm64).

En cualquier caso, `cargo bench --no-run` (unos 2 minutos) asegura que los benchmarks siguen
compilando. También se puede cubrir con `cargo clippy --all-targets`, que ya los compila como
targets de bench.
