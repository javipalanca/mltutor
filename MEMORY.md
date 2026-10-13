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
- **Spike de crates (2026-10-13). Versiones fijadas:** linfa, linfa-linear, linfa-logistic
  y linfa-nn 0.8.1 (exigen ndarray 0.16 y rand 0.8); burn `=0.22.0` con backend `flex` +
  `autodiff` (el backend `ndarray` está deprecado en 0.22); rand y rand_chacha 0.10;
  rand_mt 6.1; prost 0.14. El ndarray de linfa (0.16) y el de burn-ndarray (0.17) son
  incompatibles: mltutor-core usa ndarray 0.16 y mltutor-nn intercambia `Vec<f32>` o
  slices. Sin BLAS, openssl, cc ni protoc: compila para mac arm64, win x64 y linux x64.
- **linfa-trees no sirve:** solo clasifica, no guarda n_samples/impurity/value por nodo, no
  tiene predict_proba, poda subárboles y calcula en f32. En Iris (depth 3) da 5 nodos
  frente a 9 de sklearn y falla 6/150 predicciones. **CART propio** para clasificación y
  regresión, con los umbrales como punto medio en f32 (sklearn convierte X a float32) e
  importancia ponderada por muestras.
- **linfa-logistic ≈ sklearn:** misma función objetivo con α = 1/C (sin penalizar el
  intercepto). sklearn por defecto se detiene antes del óptimo: 1.6e-3 de diferencia en
  las probabilidades de Iris, y 5e-6 frente a sklearn convergido. Tolerancia de paridad:
  2e-3. **Trampa:** en binaria, la clase "positiva" de linfa es la más frecuente, no la
  clase 1; hay que remapear las probabilidades y el signo de los coeficientes.
- **linfa-linear OLS = sklearn** (1e-12 en Diabetes).
- **KNN:** linfa-nn no trae clasificador ni regresor y `k_nearest` no devuelve distancias.
  KdTree y BallTree con L1 dan vecinos incorrectos, y KdTree hace panic con NaN. **Usar
  fuerza bruta propia**, que coincide con sklearn (error de 0 a 2e-16). Con distancia 0
  se aplica la regla de sklearn: si algún vecino está a distancia 0, solo cuentan esos.
- **burn 0.22:** la API ya no es genérica (`Tensor<D>`, `Device::flex().autodiff()`); la
  documentación antigua con `B: Backend` no vale. El RNG del backend es un static global:
  hay que serializar los entrenamientos con un Mutex para que sean deterministas. Valores
  por defecto distintos de Keras que hay que fijar a mano: Adam eps 1e-7; RmsProp rho .9,
  momentum 0 y eps 1e-7; bias a cero (XavierUniform también lo inicializa). El LR va en
  cada `step`, así que ReduceLROnPlateau se implementa a mano. Iris, 100 épocas: unos
  90 ms. Añade unos 4 MB al binario. En `[profile.dev.package."*"]` hace falta opt-level
  2–3 o `tauri dev` entrenará muy lento.
- **ONNX desde Rust:** prost con los structs de onnx.proto escritos a mano (sin protoc),
  ir_version 8, opset 15 y ai.onnx.ml 3. Pasan el checker y onnxruntime con
  LinearRegressor, LinearClassifier, TreeEnsemble* y Gemm+activaciones, con errores
  ≤ 3e-7. onnxruntime rechaza nombres de nodo repetidos y el checker no lo detecta:
  validar siempre con ORT.
- **Splits iguales que sklearn:** rand_mt `Mt::new(42)` con Fisher-Yates y rechazo por
  máscara reproduce exactamente `RandomState(42).permutation(n)` y, por tanto,
  `train_test_split` sin estratificar.
- **uv en este Mac:** el Python gestionado por defecto es x86_64 y falla con `--no-project`.
  Hay que usar `--python .venv/bin/python`. onnxruntime no está en el entorno del repo;
  para validar: `uv run --no-project --python .venv/bin/python --with onnx==1.17.0 --with onnxruntime ...`.
- **Decisiones D7 y D8 (2026-10-13, usuario):** el split se reproduce exactamente como en
  Python (sin estratificar, con rand_mt), y en MPG `name` deja de ser característica pero
  se mantiene para visualizar (tooltips y vista previa).
- **Esqueleto Tauri (2026-10-13):** la CLI de Tauri es una devDependency de `ui/`;
  `npm --prefix ui run tauri <cmd>` ejecuta `cd .. && tauri`. `beforeDevCommand` y
  `beforeBuildCommand` son `npm run dev`/`build`, sin `--prefix`, porque Tauri ya ejecuta en
  `ui/`. Flags al binario en dev: `npm --prefix ui run tauri dev -- -- -- --smoke-test`.
  `cargo build --release` a secas no empaqueta el frontend: hay que usar `tauri build`.
- **Panics:** los comandos Tauri síncronos se ejecutan en el hilo principal, y un panic ahí
  aborta la app. Todos los comandos son `async` y el trabajo pesado va en `spawn_blocking`.
  `smoke_panic_probe` comprueba en el smoke test que la app sobrevive a un panic de un worker.
- **Comandos permitidos:** cada comando nuevo se declara en `src-tauri/build.rs`
  (`AppManifest::commands`) y en `capabilities/default.json` (`allow-<cmd>`).
- **Tipos compartidos:** se usa ts-rs. `cargo test` regenera `ui/src/lib/bindings/*.ts`, que
  van versionados; la CI comprueba `git diff --exit-code`.
- **CSP y estilos en línea:** con `style-src 'self'` Tauri inyecta nonces y bloquea los
  atributos `style=`. Para Plotly y KaTeX (F1/F2) habrá que poner `'unsafe-inline'` en
  `style-src` y `dangerousDisableAssetCspModification: ["style-src"]`. Hasta entonces, nada de
  `style=` estáticos.
- **Captura:** `--screenshot` es nativa solo en macOS (`WKWebView.takeSnapshot`, recortando
  la barra de título). En Windows (`CapturePreview`) y Linux (`webkit_web_view_get_snapshot`)
  sale con 1 por ahora.
- **Exit codes:** `App::run_return` + `AppHandle::exit(code)`; el watchdog usa
  `process::exit(2)`. En Windows el binario es `windows_subsystem="windows"`, así que el CI
  necesita `Start-Process -Wait -PassThru` para leer el código.
- **`.gitignore`:** la regla Python `lib/` ignoraba `ui/src/lib/`; se ha añadido `!ui/src/lib/`.
- **Medidas F0 (Mac arm64):** `.app` de 6,65 MiB (zip de 3,8 MB) y unos 0,6 s hasta la
  interfaz lista. TypeScript fijado a ~6.0 (svelte-check 4.7 no admite el 7).
- **CI Rust (2026-10-13):** `.github/workflows/rust.yml` es el único workflow que publica
  desde esta rama. A `build-executables.yml` se le quitó el disparador de tags, y su job
  release exige `event_name == 'push'`. El job release comprueba que el tag es `v` + la
  versión de Cargo.toml. En Windows se fija `core.autocrlf false` para que el diff de los
  bindings de ts-rs no falle. Validado con actionlint y shellcheck (instalados con brew).
  Solo el job de macOS está probado en local; Windows y Linux, pendientes de la primera
  ejecución en CI.
- **AppImage:** Ubuntu 22.04 no trae libfuse2. En CI se usa `APPIMAGE_EXTRACT_AND_RUN=1`,
  y los estudiantes pueden usar `--appimage-extract-and-run`, que no necesita admin. El
  smoke test en Linux corre bajo `xvfb-run dbus-run-session`, con las variables WEBKIT_*
  de render por software.
- **release.sh:** sincroniza la versión en Cargo.toml (`[workspace.package]`),
  tauri.conf.json, ui/package.json y los dos lockfiles. Tiene `--dry-run` y pide
  confirmación (y/N) antes del commit, el tag y el push; si algo falla, restaura los
  archivos.
- **macOS:** el `.app` lleva firma ad-hoc; el zip hecho con ditto arranca y pasa el
  smoke test.
- **Firma de macOS (2026-10-13):** sin `bundle.macOS.signingIdentity`, el `.app` solo lleva la
  firma del linker del binario ("linker-signed", sin recursos sellados). Al descargarlo,
  Gatekeeper dice que **está dañado**. Con `"signingIdentity": "-"` en
  `tauri.macos.conf.json`, Tauri firma ad-hoc el bundle completo y queda sellado. Al
  abrirlo, el aviso pasa a ser el normal de "no se puede verificar" (se resuelve con
  *Abrir igualmente* o con `xattr -dr com.apple.quarantine MLTutor.app`). El CI
  descomprime el zip y ejecuta `codesign --verify --deep --strict`. La solución
  definitiva sigue siendo Developer ID + notarización.
- **D3 sin implementar:** todavía no existe el aviso con enlace si falta WebView2.
- **Sin releases intermedias (2026-10-13, usuario):** no se publica ninguna versión hasta que
  la funcionalidad esté completa o casi. Durante la migración se valida con los artefactos
  del CI. `rust.yml` se ejecuta en cada push a `feature/rust-migration`.
- **Fase 1, hallazgos de la versión Python (2026-10-13; detalle en `docs/migration/fase1-shell-datos.md` §11):**
  - `selected_dataset` es global en session_state y lo comparten las lecciones y la página
    CSV. Al cargar un CSV, pasa a ser `"📄 {archivo}"`.
  - **Pingüinos se ve mal en la exploración de Python:** `X` conserva el índice de
    `dropna()` e `y` no, así que el concat da 342 filas con 9 NaN y etiquetas desplazadas.
    No afecta al entrenamiento. Rust alinea por posición.
  - Moons se regenera con la hora como semilla en cada renderizado; en Rust queda fijo por
    sesión (D9).
  - `create_button_panel` usa una clave global `viz_type`; en Rust hay estado por lección (D13).
  - Correlación: usa `X.columns` y el objetivo como códigos; la máscara `triu(k=0)` también
    oculta la diagonal. En Rust la escala va de -1 a 1 (D12).
  - Colores: el pairplot usa tab10 por orden de aparición; las barras del objetivo ordenan
    por clase y usan `#3274a1`. `describe()` excluye las columnas bool.
  - CSV: dos heurísticas de tarea incoherentes (el selector usa `nunique ≤ 20` y `< 10 %`;
    "auto", `nunique < max(5, 5 %)`). Las clases de texto pasan a llamarse "Clase i", y
    los `;` o Latin-1 fallan. Rust es más tolerante (D10).
  - `dataset_metadata.py` etiqueta mal los días de Propinas (0=Fri, 1=Sat, 2=Sun, 3=Thur);
    corregido en el TOML (D15). Los metadatos de Iris nunca se veían (nombres de seaborn).
  - KDE del pairplot: `gaussian_kde` con Scott, rejilla `linspace(min-3bw, max+3bw, 200)` por
    nivel, `common_norm` y omite los niveles con n<2 o varianza 0. Se muestrea con más de
    2000 filas (`sample(2000, random_state=42)`; en clasificación, por grupo).
- **mltutor-core F1 (2026-10-13), paridad casi exacta con NumPy, pandas y SciPy copiando sus
  algoritmos:** la suma por pares de NumPy (8 acumuladores, bloques de 128) en
  mean/std/describe; Welford de pandas `nancorr`; `np.percentile` lineal con el `_lerp` de
  dos lados; `linspace` como `i*step+start` con el último punto en `stop`; KDE de SciPy con
  Cholesky `sqrt(cov)·factor`. Errores ≤ 1.4e-15 en describe y correlación y 3.8e-13 en la
  KDE; histogramas y muestras exactos. El pipeline de California completo tarda 22 ms.
- **`make_moons` es reproducible bit a bit** con el `RandomState` legacy: `gauss` polar con
  el segundo valor en caché, `random_sample` = (a>>5, b>>6)/2^53, más el shuffle. Aun así,
  para la semilla 42 se embebe `moons.csv`, por si la libm de otro sistema difiere.
- **Muestreo por grupo del pairplot:** `groupby().apply(sample(random_state=42))` usa un
  `RandomState(42)` nuevo por grupo, en orden de clave. Con uno compartido salen filas distintas.
- **serde_json necesita la feature `float_roundtrip`** en los tests contra fixtures; sin ella
  algunos floats salen a 1 ulp.
- **Titanic:** `adult_male` y `alone` son bool en pandas, pero el CSV guarda 0/1, así que
  van fijos en `DatasetId::bool_columns()` (describe los excluye).
- **CSV en Rust (D10):** los bytes NUL dan `CsvError::Encoding` (UTF-16 o binario); si no es
  UTF-8 se lee como Windows-1252, que nunca falla. `;` y tabulador se detectan en la primera
  línea, fuera de comillas. La coma decimal solo se acepta con `;`. Una fila con menos
  campos que la cabecera es `Malformed`, mientras pandas la rellena con NaN. Fuzzing con
  proptest sin panics.
- **Plotly y CSP (2026-10-13):** se usa `plotly.js-strict-dist-min`. El bundle normal usa
  `Function` en regl (scattergl) y obligaría a poner `unsafe-eval`; el strict trae los shaders
  precompilados, así que basta con `script-src 'self'`. Se carga con `import()` dinámico.
  **`freezePrototype: true` rompe Plotly** ("Attempted to assign to readonly property"), así
  que está a `false` (D18). Si no hay WebGL, el pairplot se dibuja en SVG.
- **Svelte 5:** no leer y escribir `$state` en el mismo `$effect` (`effect_update_depth_exceeded`);
  el contador `busy` del IPC usa `untrack`.
- **IPC:** Tauri pasa los argumentos de snake_case a camelCase en JS (`draftId`).
  `CoreFacade` tiene dos implementaciones: `RealFacade` (core) en la app y `FixtureFacade`
  en los tests (o en debug con `MLTUTOR_FACADE=fixture`).
- **Capturas Rust:** `--scenario <nombre>[:ruta.csv]` y `--full-page` (macOS: agranda el
  WKWebView más allá de la pantalla; `takeSnapshot` renderiza también lo que queda fuera).
  Salida en `docs/migration/capturas/rust/`; comparación en `capturas/comparacion-fase1.md`.
- **Iterar rápido:** `npm --prefix ui run tauri build -- --debug --no-bundle` tarda unos 25 s,
  frente a varios minutos de la release con LTO fat.
- **Formato de números:** `toFixed` de JS no redondea los empates exactos como Python;
  `ui/src/lib/fmt.ts` lo corrige.
- **Medidas F1:** `.app` de 13,4 MiB (zip de 7,4 MB) y unos 0,6 s hasta la interfaz lista.
  Lo que más ocupa es Plotly (4,7 MB), seguido de los CSV (3 MB) y KaTeX (1 MB; sobran
  los .ttf/.woff).
- **CSV en streaming (2026-10-13, revisión F1):** el tipo de cada columna se decide al leer
  (vacía → int → float → bool → texto) y el texto se lee en una segunda pasada para conservar
  la cadena original. Con un CSV numérico de 95 MiB, el pico pasa de 1,5 GiB a 324 MiB
  (`tests/csv_memory.rs`, ignorado, en release). Trampas: `HashSet` con `collect()` reserva
  según el `size_hint` (144 MiB con 900 valores distintos); a `sorted_unique` le hace falta
  `shrink_to_fit`; una copia filtrada sin `with_capacity` llega a 3× por el crecimiento del Vec.
- **No finitos:** `inf`/`-Infinity` son float64 como en pandas, pero las estadísticas y
  gráficos del core los excluyen (aviso `CsvNotice::NonFinite`). `"NAN"`/`"Nan"` son texto,
  como en pandas. Diferencia conocida: `1e999` es texto en pandas e inf en Rust.
- **Cabeceras repetidas:** se sigue el algoritmo exacto de pandas 2 (`a,a,a.1` → `a, a.2, a.1`).
- **Histograma "auto":** máximo de 1000 intervalos, con Sturges si se supera; numpy no
  limita. El máximo en los fixtures es 542 (California). La correlación usa como máximo
  100 columnas (`truncated`). -0.0 y 0.0 son el mismo valor (`float_key`).
- **Cancelación en el core:** variantes `*_cancellable` con `&AtomicBool`, comprobadas cada
  16.384 filas al leer el CSV y por fila en la correlación.
- **Rutas de archivo nunca desde JS (2026-10-13, revisión F1):** `pick_and_load_csv` y
  `save_file` abren el diálogo desde Rust (`blocking_*` dentro de `spawn_blocking`). Sin
  `tauri-plugin-fs` y sin permisos `dialog:`/`fs:` para JS. `--scenario csv_…:<ruta>` solo lo
  lee `load_scenario_csv` en modo captura. Un FIFO bloquea `File::open`: se comprueba
  `is_file()` antes y después de abrir, y se lee con `take(MAX+1)`.
- **Binario por IPC:** `invoke(cmd, Uint8Array, { headers })` llega como `InvokeBody::Raw`; las
  cabeceras deben ser ASCII (el nombre va con `encodeURIComponent`).
- **Inyección desde datos:** `fmt` escapa HTML y Markdown en los valores interpolados;
  `fmtText` es solo para nodos de texto de Svelte. Plotly interpreta pseudo-HTML (`<a href>`,
  `<b>`) en títulos, leyendas, ticks y hover: hay que escapar y usar `%{fullData.name}`.
  `on_navigation` se instala como plugin (`plugin::Builder::on_navigation`) para que cubra
  las ventanas de `tauri.conf.json`; permite solo el origen propio de la app.
- **Orden de peticiones:** las selecciones y los registros de CSV llevan un `seq` de la UI (que
  empieza en `Date.now()*1000`), y gana el más alto aunque termine antes. Los comandos async no
  se ejecutan en orden de llegada.
- **Memoria de sesión:** se guarda un solo CSV registrado. El borrador vive mientras la página
  CSV lo usa y se libera al cargar otro archivo o con `discard_csv`.
- **Cancelación:** `tasks.rs` guarda un `Arc<AtomicBool>` por `task_id`; `cancel_task` y el
  botón "Cancelar" de la barra de estado cubren la lectura del CSV, `set_csv_target` y la
  correlación. Un `Cancelled` no se muestra como error.
- **Doctests en Windows (2026-10-13):** con `cargo test --workspace`, cargo pasa a rustdoc las
  rutas nativas de TODOS los build scripts. En Windows, el `msvcrt.lib` que deja el build de
  Tauri rompe el enlace de los doctests de mltutor-core (`__CxxFrameHandler3`). En CI se usa
  `cargo test --workspace --all-targets`, más `cargo test --doc -p mltutor-core -p mltutor-nn -p mltutor-export`.
- **Capturas Qt:** `uv run python tools/capture_qt.py` genera las de todas las pantallas en
  `docs/migration/capturas/qt/`. No funciona en modo offscreen y no captura las vistas
  Plotly/WebEngine. Se cuantizan a 256 colores con Pillow antes del commit (de 9,7 MB a 3,9 MB).
- **Fixtures golden (2026-10-13):** se generan con `uv run python tools/export_golden.py` y
  se verifican con `--check` (exit 1 si algo cambiaría). Son deterministas y pesan unos
  3 MB de CSV en `tests/parity/datasets/data/` (California ocupa 1,9 MB). Tolerancias:
  datos y splits exactos, OLS 1e-8 relativo, logística 1e-3 en coeficientes y 1e-4 en
  probabilidades.
- **El split de Python NO estratifica (2026-10-13):** `preprocess_data` es
  `train_test_split(test_size, random_state=42)`, con `test = perm[:ceil(t·n)]` de
  `RandomState(42)` (MT19937). En Iris el test queda 19/13/13. Los tests de paridad de
  modelos deben usar los índices del fixture, no reproducir el RNG de numpy.
- **Moons usa el timestamp como semilla** en la app Python, así que no es reproducible.
  El fixture usa `random_state=42`.
- **MPG también necesita red en Python:** se carga con `sns.load_dataset("mpg")`, no con
  el CSV incluido, y codifica `name` (el modelo de coche, 301 valores) como característica.
  Diamantes codifica cut/color/clarity en orden alfabético, no ordinal, y hace
  `sample(8000, random_state=42)`.
- **`X.columns` frente a `feature_names`:** difieren en Titanic, Propinas, Pingüinos,
  California y Dígitos (p0..p63). Los CSV de parity usan `X.columns`; la UI muestra
  `feature_names`.
- **Cáncer:** 0 = malignant y 1 = benign, en el orden de sklearn. Titanic rellena la edad y
  la tarifa con la mediana y embarked con 'S'. Pingüinos: 333 filas tras dropna; MPG: 392.
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
