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
- **Fase 2, fixtures y hallazgos (2026-10-14; detalle en `docs/migration/fase2-regresion.md` §12):**
  - `tools/golden_render.py` renderiza las lecciones Python sin Qt y extrae lo que dibuja
    matplotlib (líneas, puntos con color, rejillas de contourf, textos). `expected.views` en
    los fixtures es la especificación de lo que se ve. Sirve para F3–F5.
  - La logística de Python no converge en Cáncer (`max_iter=1000`; queda a 0,20 del óptimo) y en
    Titanic queda a 2,6e-3. La paridad se mide frente a `converged_reference` (tol=1e-12) con
    1e-5 en probabilidades (D19). El 1e-4 del fixture de F0 era optimista.
  - Logística sobre un objetivo de regresión: en Diabetes entrena 192 clases y en California
    falla. En Rust se bloquea (D26).
  - Fallos visibles en Python: `np.arange(0.1,1,0.1)` no contiene 0,3 ni 0,7 exactos, así que la
    tabla de umbrales solo muestra 0,5; "R² = 0.0000" siempre; el fondo de la frontera (Set3,
    n+1 bandas) no casa con los colores de los puntos. Corregidos en Rust (D21–D23).
  - Rejilla de la frontera: min−1..max+1 con h=0,02 (rango/200 si supera 100), con
    `np.arange` = `min + i·((min+h)−min)`. Si pasa de 50 000 puntos, `linspace(min(200,n))`. La
    frontera y la superficie REENTRENAN un modelo 2D sobre train; no fijan el resto.
  - El formulario de predicción de Python usa float32; Rust usa f64 (D31).
    `onnx.reference.ReferenceEvaluator` evalúa LinearClassifier/LinearRegressor sin onnxruntime.
  - Capturas de F2: `uv run python tools/capture_qt.py --phase 2 --quantize`, unos 75 s.
- **CART de sklearn y empates (2026-10-14, F3):** en cada nodo, sklearn sortea el orden de las
  características (xorshift `our_rand_r` sembrado con `RandomState(rs).randint(0, 2^31−1)`, que da
  1608637542 con 42, y Fisher-Yates sobre un array `features` que persiste entre nodos). Gana la
  PRIMERA que alcanza la mejor mejora (`>` estricto). Los empates son frecuentes (raíz de Iris:
  petal length/width) y hay casi-empates de 1 ulp que solo coinciden sumando en el mismo orden
  (introsort, `reverse_reset`, partición). `tools/cart_reference.py` lo reproduce bit a bit en los 40
  árboles de los fixtures y es la especificación del CART de Rust (D36, D51).
  - El umbral es `x[p−1]/2 + x[p]/2` en f64 de valores f32. `predict` compara `f32(x) <= t`.
  - **CORREGIDO (2026-10-15, revisión ML de F3):** `FEATURE_THRESHOLD` se comporta como **0** en sklearn
    1.7.2, no como 1e-7: sklearn divide incluso entre dos float32 contiguos. Con 1e-7, una columna en
    escala 1e-6 se fundía y clasificaba mal. Hay que comparar `x[p+1] <= x[p]` (tanto en cart.rs como
    en cart_reference.py).
  - La wheel macOS arm64 de sklearn usa FMA: las impurezas difieren 1–4 ulp y, en algunos empates
    exactos (particiones espejo), la FMA rompe el empate y **la estructura SÍ puede cambiar** (3 de
    36 árboles en la revisión). Rust reproduce el algoritmo sin FMA; hay que validar los fixtures
    también en x86-64. El script en Apple Silicon puede dar otro árbol en esos casos.
- **Revisión F3, correcciones del core (2026-10-15):**
  - `FEATURE_THRESHOLD = 0` verificado con el fixture `trees/tiny_scale_features` (5 casos que con 1e-7
    darían otro árbol). Causa probable: el inicializador de un `cdef` del `.pxd` no se ejecuta en los
    módulos que hacen `cimport`.
  - El CART comprueba la cancelación cada 256 posiciones candidatas: un MAE con n=60 000 se cancela en
    20–40 ms (antes ~15 s).
  - `Predictor::predict_f32_rows`: la permutation importance convierte a f32 una sola vez (3000×300×10:
    de 2,93 s a 0,24 s); KNN usa `predict` sobre una copia f64.
  - Identificadores Python: NFKC + XID_Start/XID_Continue (`unicode-ident`, `unicode-normalization`); "m²"
    es alfanumérico pero no XID_Continue. `display_label` sustituye los caracteres de control de nombres
    de columna o clase por espacios.
  - `tools/check_trees_platform.py` reentrena sklearn con los 45 árboles (sin la app ni red) y se ejecuta
    en el job de Linux del CI (x86-64, sin FMA). En arm64, `cart_reference` coincide bit a bit con
    sklearn en 30/45 árboles por la FMA, con la estructura igual en todos.
  - Primera ejecución en x86-64 (CI Linux, 2026-10-15): los 45 árboles de los fixtures coinciden con
    sklearn (estructura y umbrales; impurezas 31/45 bit a bit, error ≤ 1,14e-13), así que los fixtures
    generados en arm64 valen también en x86-64. `cart_reference` coincide bit a bit con el sklearn de
    x86-64 en 43/45 (en arm64, 30/45).
  - `export_golden.py --check` terminó dos veces con el código 144 y sin salida al ejecutarse en primer
    plano desde el agente; en segundo plano pasa. Parece un límite del entorno, no un fallo.
- **Métricas con clases ausentes del train (2026-10-15):** los scripts de entrenamiento usan
  `labels=sorted(nombre_clase)` (todas las clases del dataset), como `metrics.rs`. Con
  `labels=modelo.classes_` una clase ausente del train desaparece del promedio ponderado. Las curvas
  ROC/PR siguen con `classes_` (columnas de `predict_proba`). `camino_decision.py.j2` ya no es
  literal: lee la rama de `children_left`, muestra `float32(x)` y tiene entrada en `golden_trees.CODE_CHANGED`.
- **NaN y MAE en el entrenamiento (2026-10-15, revisión F3):** `real_models::fit` rechaza ANTES del split
  cualquier característica no finita con `CommandError::MissingValues { columns }` (`[entrenamiento]
  error_nan`), en todas las lecciones. sklearn 1.7 sí entrena árboles con NaN: es una diferencia
  documentada. `absolute_error` se limita a 20 000 filas de entrenamiento (`check_mae_limit`), también en
  el reentrenamiento 2D de `decision_grid`.
- **PNG del árbol:** `pngScale` no tiene suelo, porque rompería el límite de 16 000 px de lado de
  WebKit/WebView2. Por debajo de una escala de 0,5 el botón PNG se desactiva y se ofrece el SVG. Un botón
  `disabled` no muestra `title` en WebKit, así que el tooltip va en un `span` que lo envuelve.
- **testing-library/svelte 5 `rerender`:** sustituye todo el objeto de props, así que los `$derived`
  que crean objetos cambian de identidad. Los efectos de "reiniciar al cambiar X" deben comparar la
  identidad de la prop (`diagram`), no la de un derivado.
- **Split: tamaño de train (2026-10-15):** sklearn con `train_size=None` usa `n_train = n − ceil(t·n)`.
  `floor((1−t)·n)` perdía una fila por redondeo (California: 0.7·20640 = 14447.999…) y afectaba a F2 y F3.
  Los tests de paridad no lo detectaban porque usan los índices del fixture: ahora hay un test explícito.
- **TreeDiagram (2026-10-15):** la unidad horizontal de Buchheim es la caja más ancha (más un margen), así
  que con letra fija las cajas no se solapan. Si ajustar todo deja la escala < 0,45, la primera vista va a
  la raíz al 80 %. PNG en canvas a 3× (límite de 16 000 px por lado). California d8 (407 nodos): unos 65 ms.
  En ventanas ocultas `requestAnimationFrame` no se ejecuta, así que se mide con `getBoundingClientRect()`.
- **vitest y fixtures:** `new URL(…, import.meta.url)` choca con `fs.allow` de Vite; hay que usar
  `resolve(process.cwd(), "../tests/…")`. `fmt` también escapa dentro de los bloques de código, así que
  los identificadores entre comillas invertidas se sustituyen antes.
- **Smoke test:** en 1 de 5 ejecuciones locales acabó en timeout (2) tras `predict_one` de Cáncer,
  posiblemente por los timers congelados con la ventana oculta. Vigilarlo en CI.
- **CART de Rust (2026-10-15):** `tree/cart.rs` transcribe `cart_reference.py` y reproduce bit a bit la
  estructura y los umbrales de los 40 árboles de los fixtures. Las impurezas difieren ≤ 1,1e-13 por la FMA
  de la wheel arm64. California d10 tarda unos 48 ms (sklearn 59 ms), de los que 33 son del introsort. No
  se puede sustituir por `sort_unstable`: el orden en que deja los empates decide las sumas. MAE es
  O(n²) por nodo, como en sklearn (California d3: 3,6 s, cancelable).
- **ONNX binario de árbol:** `TreeEnsembleClassifier` debe llevar UN peso por hoja (P(clase 1) con
  `class_ids = 0`), como skl2onnx; con las dos clases, onnxruntime equivoca 537 de 891 etiquetas en Titanic.
- **`py_float_repr`:** `format!("{:e}")` da la representación más corta, pero no siempre la más cercana
  (1400.50006103515625: Rust `…563`, CPython `…562`); se usa `{:.Ne}` con el mismo nº de dígitos.
- **OLS:** se resuelve por sustitución hacia atrás si σ_min estimado supera en 1e6 el umbral de rango. Un
  1500×1500 centrado pierde rango y sigue en 12,7 s en la SVD de linfa-linalg, sin cancelación (pendiente).
- **friedman_mse ≈ squared_error en un árbol único (2026-10-15):** la puntuación de Friedman es la reducción
  del SSE, así que los árboles salen idénticos salvo en empates numéricos (en sklearn difieren en 10 de
  60 árboles con muchos empates; en Diabetes e Iris coinciden en las 450 combinaciones).
- **Plantillas de F3:** `arbol_entrenamiento.py.j2` copia literalmente los bloques de imports y carga de
  datos de `regresion_entrenamiento.py.j2`, y `golden_trees._training_shared_blocks` falla si divergen.
  El camino de decisión en Python se lee de `children_left` y no comparando `np.float32(x) <= umbral`:
  con NEP 50 (numpy ≥ 2) esa comparación se hace en float32 y contradice a sklearn.
- **Verificar scripts sin red:** `sns.load_dataset("titanic", data_home="mltutor/dataset/data/sample_datasets")`.
  minijinja de Python: `uv run --with minijinja python …` con `undefined_behavior="strict"` y
  `keep_trailing_newline=True` reproduce `codegen.rs`.
- **permutation_importance (sklearn ≥ 1.5):** usa UNA sola semilla `randint(2^31)` para todas las
  columnas y baraja de forma acumulativa (`shuffle(idx)` y `X[:, c] = X[idx, c]`).
- **Umbrales de árbol en ONNX y Python:** skl2onnx redondea los umbrales al float32 más cercano; hay
  que escribir el mayor float32 ≤ t (D47). El código Python que recibe floats f64 usa el "switch
  point" w (`x <= w ⇔ f32(x) <= t`); si no, `longitude = -117.9` toma la otra rama en California.
- **Fallos del árbol en Python:**
  - el tipo de árbol empieza siempre en Clasificación (Diabetes entrena 192 clases y la Evaluación cae);
  - `export_tree_as_python_code` nunca compila;
  - la superficie usa los sliders actuales;
  - en Árboles nunca se ve la importancia por permutación.
- **OLS (2026-10-14, revisión F2):** sklearn usa `scipy.linalg.lstsq` con `cond = max(n,p)·eps`, no
  eps. Con `eps·σ_max`, las columnas derivadas (`0.1·x0 + 0.7·x2`, `x0/3`) conservan valores
  singulares de ruido y dan coeficientes de unos 1e12 (y un ONNX que falla en millones). Rust usa una
  QR de Householder propia más la SVD de R (O(n·p²), cancelable): 100k×200 pasa de 36 s a 1,2 s. Ya no
  se usa linfa-linear. La SVD de linfa-linalg no ordena los valores singulares.
- **Logística en Dígitos:** está tan mal condicionada que L-BFGS sigue a 2,8e-3 del óptimo tras 1000
  iteraciones, y el lbfgs de sklearn se para a las 139, a 0,25. Newton la resuelve en 12 iteraciones
  (unos 0,6 s). Newton se usa si d ≤ 700 y n·d² ≤ 2e9; si no, L-BFGS con el criterio de sklearn y un
  ftol relativo de 1e-10. Se estandariza sobre x/max|x| para que valores de 1e300 no desborden.
- **Límites:** `MAX_CLASSES = 50` (`ModelError::TooManyClasses`). El CSV "auto" elige Regresión con más
  de 50 valores distintos (diferencia deliberada con Python). `onnx_class_labels` usa los valores de
  clase solo si todos son enteros distintos; si no, los códigos. `CsvDialect` se deduce de los avisos
  del lector y se pasa al script generado.
- **Script con newton-cholesky (2026-10-14, revisión ML de F2):** `regresion_entrenamiento.py.j2`
  usa `LogisticRegression(solver="newton-cholesky", tol=1e-8)`. lbfgs sin escalar no llega al óptimo
  de Rust (Cáncer queda a 0,20) y con el tol por defecto Iris queda a 7,4e-5. `export_golden`
  compara la logística del script con el óptimo (newton-cholesky, tol=1e-14), no con la app
  Python. Requiere `scikit-learn>=1.6`.
- **Escalar NO es equivalente con L2:** `StandardScaler` + `LogisticRegression` es otro modelo (en
  Cáncer, hasta 0,907 de diferencia en probabilidades). Rust estandariza solo como cambio de
  variables, con la penalización sobre los coeficientes originales.
- **`pycomment` (filtro de plantillas):** quita CR/LF, elimina `"""` en bucle (`""\n"` volvería a
  formar comillas triples) y duplica `\`.
- **Titanic:** `class_encoded` = `pclass` − 1 (colinealidad exacta); la L2 reparte el efecto (−0,416
  cada una).
- **`seq` de entrenamiento (2026-10-14):** `LessonModels.last_seq` es por lección y no se borra
  nunca; si se borrara, una petición antigua que llegara después de terminar la nueva la
  sustituiría. `task-progress` se limita con `ProgressThrottle` (solo si cambia el % entero o el
  mensaje, o cada ≥100 ms), y la UI solo acepta eventos del `task_id` activo.
- **Clases del modelo frente a las del dataset:** `class_names` se indexa por el código del
  dataset; las filas de `coef_` y las columnas de `proba`, por `model_classes` (`classes_`). Si
  falta una clase en el train, el modelo puede ser binario (positiva = `classes_[1]`). Para
  colorear la frontera hay que usar `point_codes`, nunca el valor del objetivo. Los textos de
  avisos del backend se leen de `content/` con `mltutor_core::content::Messages`; no se duplican.
- **UI de F2 (2026-10-14):** WKWebView no avanza las transiciones CSS en ventanas ocultas o
  con `--full-page`, y las capturas salían con botones a medio cambiar. En modo automatizado,
  `startup.ts` pone `data-automation` en `<html>` y `theme.css` desactiva las transiciones. En
  el smoke test, los timers se congelan con la ventana oculta y `waitFor` puede no expirar:
  los selectores deben apuntar a elementos visibles (el `pre` de un expander cerrado colgaba la
  prueba). `listen()` necesita `core:event:allow-listen/unlisten`. Los textos propios en
  Markdown se insertan con un marcador después de `fmt` para que no se escapen.
  `modelSession.svelte.ts` y `viewLoader.svelte.ts` (`ui/src/lib/ml/`) son reutilizables en
  F3–F5; la frontera de los modelos es `ModelView` + `DatasetView::train`.
- **Medidas F2:** `.app` de 14,9 MiB (zip de 8,1 MB); entrenar y cada vista ≤ 5 ms en California;
  el smoke test completo tarda unos 13 s.
- **Logística en Rust (2026-10-14):** linfa minimiza Σ pérdida + (α/2)‖W‖²: α = 1/C exacto,
  sin ½ ni media (`tests/linfa_logistic_scale.rs`). Aun así NO se usa linfa-logistic: no
  cancela, no informa de iteraciones ni convergencia y con L-BFGS no converge en Cáncer. El
  solver es propio:
  - Newton amortiguado (Hessiana exacta + Cholesky + Armijo) si `k·(p+1) ≤ 300`;
  - L-BFGS de argmin con bucle propio en los demás casos (Dígitos: unos 0,5 s);
  - estandarización interna, buscando el óptimo y no el punto donde para sklearn;
  - Cáncer converge en 9 iteraciones.
  La `converged_reference` de Cáncer no está en el óptimo (scipy para por ftol, max|∇| 1,2e-5),
  así que el test comprueba gradiente < 1e-6 y un objetivo ≤ al de la referencia. Consecuencia:
  el aviso D19 casi nunca saldrá. Los coeficientes multinomiales se centran (Σₖ = 0).
- **minijinja 3:** `keep_trailing_newline` se activa con `SyntaxConfig::builder()`; serde con
  `Value::from(Serde(x))`; en modo strict, un `undefined` pasado a un filtro NO falla, así que
  `pyrepr` lo rechaza a mano. `crates/mltutor-core/build.rs` incrusta todos los
  `content/code/*.j2`, que se cargan por nombre.
- **zip 9 determinista:** `default-features = false`, `deflate-flate2-zlib-rs` y
  `last_modified_time(DateTime::default())`.
- **ONNX binario de skl2onnx:** un `LinearClassifier` con filas `[−w, w]` y `LOGISTIC`, sin
  Normalizer. onnxruntime acepta `ai.onnx.ml` 3.
- **CSV:** `"-0"` en una columna entera que pasa a float debe dar -0.0 (lo encontró la
  proptest; el caso está en `proptest-regressions/`, versionado).
- **Contenido de F2 (2026-10-14):** `content/code/regresion_entrenamiento.py.j2` (D29) se ejecuta
  en `export_golden.py` (`code/regression_training`) y reproduce la app a 1e-12. Rust debe pasar
  SIEMPRE todas las variables (null si no aplican), porque las plantillas se probaron con
  minijinja en modo strict. Los textos de `content/` marcados CORREGIDO/NUEVO difieren a
  propósito de Python, y los tests de paridad de textos deben excluirlos. Correcciones
  conceptuales:
  - ejemplos de umbrales (spam, diagnóstico) que estaban al revés;
  - el R² va de −∞ a 1;
  - en la softmax, e^β no es el odds ratio de una clase frente al resto;
  - el micro-promedio lo dominan las clases frecuentes;
  - la prueba de Pearson sobre los residuos no detecta curvas;
  - con L2, escalar cambia el modelo y no solo la convergencia.
- **AUC macro de la app:** interpola las curvas ROC por clase y las promedia, así que no es la
  media de los AUC por clase (0,561 frente a 0,502 en el CSV `string_target`); el AP macro sí
  lo es. Cáncer sin escalar necesita 2278 iteraciones (el máximo del deslizador es 2000);
  escalado, 19. En la multinomial con L2 de sklearn, los coeficientes de cada característica
  suman 0 entre clases.
- **seaborn-data:** `sns.load_dataset` reproduce los CSV golden. Pingüinos pasa `sex` a formato
  título (el CSV original trae MALE/FEMALE).
- **CSV de texto con diccionario (2026-10-14):** `ColumnData::Text` es un `TextColumn`: un código
  `u32` por fila (`u32::MAX` = faltante) y los valores distintos una sola vez, en orden de
  primera aparición, que es el desempate de `value_counts` en pandas. Con 94 MiB de texto, el
  pico pasa de 1253 a 515 MiB (5,5×) y el RSS de 1,8 a 0,58 GB. Una columna de ids todos
  distintos sigue en unas 4,7×. El `"nan"` de los faltantes comparte clase con el texto literal
  `"nan"`, como `astype(str)`. El asignador contador mide bytes pedidos, no RSS: un `String`
  de 2 bytes ocupa 16 en malloc.
- **`linspace` con rango que desborda:** si `stop - start` no es finito, se usa
  `start·(1−t) + stop·t`; los rangos representables siguen la fórmula de numpy bit a bit.
  `describe_in_place` calcula la media y la desviación ANTES de ordenar, para conservar la suma
  por parejas de pandas.
- **Liberar memoria fuera del lock (2026-10-14):** los métodos de `Session` que sueltan un
  borrador o un CSV lo devuelven, y el que llama lo libera ya sin el lock. Trampa:
  `self.session().clear_draft();` como sentencia suelta libera con el guard vivo; hay que
  hacer `let old = …; drop(old);`.
- **`save_file` y GTK (2026-10-14):** GTK solo pregunta si se reemplaza el nombre tal como se
  escribió. Si se añade la extensión después, hay que volver a confirmar
  (`dialog().message().blocking_show()`). Los diálogos que se abren desde Rust no necesitan
  capabilities. `Request<'_>` solo presta el cuerpo, así que hace falta una copia para
  `spawn_blocking`.
- **`seq` tras recargar:** `get_selected_dataset` devuelve `last_seq` y la UI sigue con
  `max(Date.now()*1000, last_seq+1)`. Así funciona aunque el reloj retroceda.
- **`tauri.localhost` solo en Windows;** en macOS y Linux el origen es `tauri://localhost`.
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
