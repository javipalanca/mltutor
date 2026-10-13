# MEMORY.md — Memoria del proyecto MLTutor

Conocimiento no obvio acumulado durante el desarrollo. Referenciado desde
[AGENTS.md](AGENTS.md). Una entrada por hecho, con fecha absoluta y el porqué.
Actualiza o elimina entradas obsoletas en lugar de duplicarlas.

## Migración a Rust (desde 2026-10-13)

- **Decisión: migrar a Rust con Tauri 2 + Svelte + linfa + burn.** Objetivo:
  velocidad, estabilidad y descargas mucho más pequeñas que con TensorFlow empaquetado.
  Se eligió Tauri en lugar de egui o Slint porque conserva las ventajas visuales de la
  versión actual: gráficos Plotly interactivos, Markdown/HTML enriquecido y CSS.
- **Se calcula en Rust y se enseña scikit-learn/PyTorch.** Los fragmentos de código Python
  que ve el estudiante se mantienen como contenido educativo.
- **Se pierden las exportaciones Pickle, Keras, SavedModel y TFLite**, que son específicas
  de Python/TF. Sustitutos: JSON (serde) para modelos clásicos, records de burn para
  redes y el código Python generado para reentrenar en sklearn/Keras. La exportación ONNX
  desde Rust está por investigar.
- **Huecos de linfa que hay que verificar o implementar:** `linfa-trees` se centra en
  clasificación (los árboles de regresión probablemente sean implementación propia);
  KNN requiere una capa sobre `linfa-nn`; permutation importance, fronteras de
  decisión y curvas ROC/PR se implementan en `mltutor-core`.
- **La paridad es aproximada, no bit a bit:** solvers, desempates y generadores
  aleatorios distintos. Los fixtures de `tests/parity` documentan la tolerancia por
  algoritmo.
- **WebView del sistema:** WebView2 en Windows (ya incluido en Win10/11), WKWebView en macOS
  y WebKitGTK en Linux. Hay que validar en equipos de laboratorio y decidir el formato
  de Linux (AppImage frente a binario).
- **Plan aprobado como borrador (2026-10-13):** `docs/migration/PLAN.md`, con 9 fases (F0–F8)
  y las decisiones D1–D6, que tienen una opción recomendada por defecto.
- **Toolchain local (2026-10-13):** el default de rustup en este Mac es x86_64 y falla
  ("Bad CPU type"). `rust-toolchain.toml` fuerza `stable` = aarch64. `gh` no está
  instalado (hace falta para release.sh).
- **Datasets que hoy se descargan:** California (`fetch_california_housing`) y Diamantes
  (seaborn) dependen de la red en la versión Python. En Rust se incluyen en el binario.
- **Código muerto en Python:** varias pestañas de `viz/nn_extra.py` y funciones de
  `ui.py`, `tree_visualization.py` y `knn_app.py` no son alcanzables; no se portan
  (INVENTARIO §8).
- Rama de trabajo: `feature/rust-migration`, creada el 2026-10-13 a partir de `codex/desktop-migration-study`, que tiene la versión Qt de referencia.

## Decisiones de diseño (versión Python/Qt)

- **2026-09 — Migración de Streamlit a Qt nativo.** La versión Streamlit con
  ventana nativa (pywebview/pythonnet, instalador Inno Setup, v0.2.x) daba
  problemas recurrentes en Windows y en laboratorios sin permisos. Se sustituye
  por PySide6 con ejecutable portable sin instalador. `main` conserva la versión
  web hasta terminar la migración.
- **API `ui` compatible con Streamlit.** Se mantuvo la forma `st.xxx(...)` para no
  reescribir miles de líneas de lecciones; `ui.py` construye un árbol declarativo
  y `window.py` lo renderiza. Mantener esta separación.
- **macOS usa onedir (`.app`), Windows/Linux onefile.** El bundle evita
  descomprimir TensorFlow en cada arranque y permite firma de código.
- **Versión única en `pyproject.toml`.** `pyinstaller.spec` y `release.sh` la leen
  de ahí. Las versiones `-rc.N` pueden publicarse desde la rama de migración; las
  estables solo desde `main`.
- **Firma Developer ID y notarización de macOS: pendiente** (faltan credenciales).

## Trampas conocidas (versión Python/Qt)

- **NumPy/Accelerate en macOS:** NumPy debe importarse en el hilo principal antes
  de crear el `QThread`; si se importa primero dentro del trabajador puede crashear
  en la comprobación BLAS. Ver `mltutor/desktop/__main__.py`.
- **Pila del QThread en macOS:** es mucho menor que la del hilo principal;
  `Worker` fija `setStackSize(16 MiB)` para LAPACK y TensorFlow.
- **joblib/multiprocessing en el ejecutable congelado (2026-09-23):**
  `permutation_importance(n_jobs=-1)` relanzaba la GUI y fallaba. Se usa
  `compute_permutation_importance` con `n_jobs=1` (`mltutor/viz/features.py`), y
  `--verify-models` lo comprueba.
- **Altura de bloques HTML (2026-09-23):** las vistas WebEngine deben esperar a que
  Chromium termine el layout antes de medir la altura; medir en `loadFinished`
  devuelve tamaños incompletos (grafo de red neuronal recortado).
- **TensorFlow en Windows:** el wheel `tensorflow` es un metapaquete; hay que
  declarar `tensorflow-intel` explícitamente o uv no lo instala.
- **onnx < 1.18:** onnx 1.18+ exige `ml_dtypes>=0.5`, incompatible con TF 2.16.
- **Keras 3 y TFLite:** la exportación TFLite cambió con Keras 3; `--verify-models`
  comprueba Pickle, ONNX, Keras, SavedModel y TFLite con equivalencia numérica.
- **HTTPS en el ejecutable:** se fija `SSL_CERT_FILE` a certifi cuando `sys.frozen`.
- **Datasets de seaborn** se empaquetan en `mltutor/dataset/data` y se cargan en
  local primero para no depender de la red.
- **Logs del ejecutable sin consola:** stdout/stderr van a `~/.mltutor/mltutor.log`;
  es lo primero que hay que mirar ante un fallo en Windows.
- **Alertas pedagógicas vs. errores:** algunas redes pequeñas generan avisos de
  error intencionados ("Modelo no convergió", "Accuracy bajo"…); `tests/test_lessons.py`
  los filtra. No los confundas con fallos reales.

## Pendientes / ideas

- Firma y notarización de macOS.
- Validar descargas en equipos limpios con cuentas sin privilegios antes de
  publicar a estudiantes.
- Distribuir los datos de los datasets que aún requieren descarga.
