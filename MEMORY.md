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
  ("Bad CPU type"). `rust-toolchain.toml` fuerza `stable` = aarch64. `gh` ya está instalado
  (2026-10-20: `gh run view <id> --log --job <id>` lee los logs del CI).
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
  KdTree y BallTree con L1 dan vecinos incorrectos, y KdTree hace panic con NaN. KNN propio.
  **CORREGIDO en F4:** la fuerza bruta con "gana el índice menor" NO basta (ver la entrada de F4). Con distancia 0
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
- **KNN y empates (2026-10-15, F4):** con `algorithm='auto'`, sklearn usa kd_tree si hay ≤ 15
  características y K < n_train//2, y brute en otro caso.
  - kd_tree: gana el PRIMER punto visitado. El orden dentro de las hojas sale del `std::nth_element`
    de libc++ (comparador `(valor, índice)`) y el heap rechaza `dist >= mayor`. Depende de la
    plataforma: libstdc++ en Linux da otros empates.
  - brute (**CORREGIDO, revisión ML de F4**): NO es "gana el índice menor". El montículo, al insertar
    un valor menor, expulsa la raíz, y entre máximos empatados cuál es la raíz depende de la
    disposición del montículo. Rust lo reproduce (idéntico a sklearn en un hilo). Además, con varios
    hilos sklearn reparte por filas de train (`parallel_on_Y` si n_test < n_train) y fusiona los
    montículos, así que **los empates cambian con el número de núcleos** (Dígitos K=20 manhattan: 104 de
    540 filas con otros vecinos). Los scripts deben usar `threadpoolctl.threadpool_limits(1)`.
  - En la euclídea por fuerza bruta, sklearn usa `‖x‖²−2x·y+‖y‖²`: una consulta idéntica a una fila de
    train da ~4e-5, no 0, y la regla de distancia 0 no se activa (Rust da 0 exacto).
  - Orden dentro de los empates (revisión F4, reproducido): con un hilo, `ArgKmin` usa `parallel_on_Y`
    si n_query ≤ n_train y n_query ≤ 1024, y vuelve a meter el montículo del hilo en otro nuevo antes
    de ordenar (mismo conjunto, otro orden: 493/540 filas en Dígitos K20 manhattan).
    `knn_reference.brute_parallel_on_y` y `NeighborHeap::repush` lo reproducen
    (`knn/brute_tie_order.json`). El orden depende del tamaño de CADA llamada, así que no hay que
    trocear las consultas de otra forma que Python: `decision_grid` predice bloques de ≥ 1025 celdas.
  - Fixtures de KNN: `case_knn` y los probes van dentro de `threadpool_limits(1)`.
  - onnx.reference 1.17 no resuelve nombres del grafo exterior dentro de un Scan: se evalúa con el grafo
    equivalente con estado. onnxruntime y `full_check` sí los resuelven.
  - Tabla de umbrales: t = i/10 exactos; la fila 0,5 usa p > 0,5 y coincide siempre con `predict`.
  - Una separación J2 de +∞ (clase sin dispersión, lejos de las demás) viaja como `separation: None` +
    `separation_unbounded` (serde enviaría `null`) y se muestra como "∞".
  - `class_distances`: hasta 8 hilos deterministas y un tope de 5000 filas por clase (semilla 42,
    `sampled`). Con 70 000 filas: de 14,6 s a 20–36 ms (2 clases). Las consultas cuya distancia a la caja
    de entrenamiento se desborda devuelven `NonFinite`.
  - El ONNX (TopK estable) desempata siempre por el índice menor, así que con datos discretos puede
    discrepar de la app con frecuencia.
  - La regla "índice menor" cambia hasta el 46 % de una rejilla (Diabetes age/sex).
    `tools/knn_reference.py` lo reproduce bit a bit y es la especificación (D52, D66).
  - La wheel arm64 suma la distancia euclídea en bloques de 8 sin FMA, y el resto (`n % 8`) con FMA
    (D53, `mul_add`).
- **Fase 6, validación cruzada de exportaciones (2026-10-20):**
  - El corpus lo escribe `src-tauri/src/export_corpus.rs` (test `#[ignore]`, 48 casos) por la capa de dominio:
    `DataState` (selección o `load_csv_bytes` + `set_csv_target`), `ModelState::train` y las funciones que
    ahora usan los comandos: `models::predict_with`, `code_of` y `export_bytes` (con sus `validate_*`).
    Antes de escribir comprueba el split FILA A FILA: las filas `check` de los paquetes, la matriz de
    confusión o `y_true` de 📈 Evaluación y las filas de entrenamiento de los vecinos KNN. Determinista;
    unos 3 s. `DatasetView::train` con una red devuelve `Internal`: hay que usar `train_observed`.
  - `ipc::CodeTemplate::source()` es la única correspondencia plantilla de la UI → `content/code/*.j2`; el
    corpus guarda ese nombre y la cobertura del verificador sale de ahí. `ClassProbability` lleva `code`.
  - `tools/verify_exports.py` + `tools/verify_exports.lock.txt` (`uv pip compile --universal
    --generate-hashes` de `verify_exports.requirements.txt`; en Windows hace falta `tensorflow-intel`).
    En este Mac: `uv run --no-project --python .venv/bin/python --with-requirements
    tools/verify_exports.lock.txt python tools/verify_exports.py <corpus>`; ~398 comprobaciones en unos 50 s
    con 4 procesos. Cada script corre con `python -I -X utf8` en una carpeta nueva, sockets bloqueados,
    `SEABORN_DATA` en una copia de `mltutor/dataset/data/sample_datasets` y `SCIKIT_LEARN_DATA` vacío
    (California no carga sin red: solo su ONNX y su paquete; Diamantes no está en `sample_datasets`).
    `python -I` no ve `PYTHON*`, pero sí `MPLBACKEND`, `SEABORN_DATA` y `OMP_NUM_THREADS`.
  - Nada pasa en vacío: 0 filas, otro número de filas que Rust o una fila sin nada comparado es FAIL;
    `label_gap ≤ 2·proba` y una etiqueta tolerada sigue comparando probabilidades.
  - **KNN:** en cada fila las K distancias ordenadas de Rust, del `.pkl` y del script deben coincidir
    (1e-9) y con las de numpy. En un empate en la K-ésima distancia el verificador enumera todas las
    respuestas válidas (cada subconjunto compatible, con la regla de distancia 0) y la del artefacto debe
    ser una; para ONNX, en el espacio de la entrada redondeada a float32 y con 1e-6 (sus salidas son
    float32). Filas toleradas como mucho el 5 % (mín. 3); observado: 7/268 en Titanic (ONNX) en mac y en
    Linux, 4/268 en scikit-learn en Linux. Un duplicado exacto de train deja de estar a distancia 0 al
    pasar a float32 ("distancia 0" se cuenta aparte): lo explica el LEEME del KNN.
    Un empate de VOTOS sin empate de vecinos (K=2, 1–1) tiene UNA etiqueta válida: sklearn, Rust y
    el ArgMax del ONNX toman el PRIMER máximo (con pesos distance solo se admite ruido relativo de
    1e-12). Una fila ONNX cuenta como "float32" solo si su respuesta no existe en float64.
    `verify_exports.self_test()` lo comprueba antes de cada ejecución.
  - `vinos_raros.csv` (cultivares 2/3/4 y un 1 solo en la prueba): valor ≠ código ≠ posición en
    `classes_`; con la lista `clases`, `texto_arbol`/`camino_decision` deben dar ValueError, y lo
    comprueba el verificador (también en `raras.csv`, donde `classes_` = [0, 2, 3, 4]).
  - **ONNX de una red con salida tanh (binaria):** `probabilities` = `[1−p, p]` con p ∈ (−1, 1), así que
    salen valores negativos o > 1 (observado: −0,043..1,043). Lo avisa el LEEME de la red (D87).
  - Errores máximos observados: ONNX lineal 5,1e-6 relativo (California), logística 8e-7, árboles 5e-8,
    MLP 2,4e-7; `.keras`/`.h5`/SavedModel/`modelo_completo.pkl` 2,4e-7, TFLite 2e-7, NumPy 1,3e-7; `.pkl`
    lineal/logística ≤ 4,4e-16 (tolerancia 1e-12: `coef_` se fija, no se reentrena); script logístico
    (newton-cholesky reentrenado) 6,4e-7 en Dígitos.
  - **Fallos encontrados en plantillas** (clases numéricas de un CSV como 1/2/3, o una clase ausente):
    `usar_modelo_arbol` hacía `class_names[prediction]` con el VALOR; `texto_arbol` y `camino_decision`
    indexaban la lista por valor (nombre de otra clase en silencio); `matriz_confusion` y
    `precision_por_clase` no pasaban `labels=` (etiquetas desplazadas y ValueError);
    `frontera_decision` emparejaba `legend_elements()` (solo clases presentes, ~9 como mucho) con todos
    los nombres. Convención única: el diccionario `nombre_clase` {valor: nombre} que definen todos los
    scripts de entrenamiento (`red_entrenamiento` lo añade; `arbol_entrenamiento` pone `None` en
    regresión). Una lista solo se acepta si `classes_ == 0..n−1`; si no, ValueError claro.
  - `analisis_exploratorio.py.j2` tiene huecos (`df`, `selected_features = ['feature1', …]`): el
    verificador los rellena con Iris. Las plantillas literales que la app no ofrece están en `NOT_RUN`.
  - El paquete Keras atrapa los fallos al guardar cada formato (`⚠️ No se pudo guardar`): el verificador
    exige los 6 archivos y que no aparezca el aviso.
  - El bug de Python del binario 1/2 sin LabelEncoder hace fallar `red_entrenamiento` (ValueError en
    `confusion_matrix`); además el verificador exige pérdidas ≥ 0 y exactitud > clase mayoritaria.
- **Contenido de la revisión F5c/F5d (2026-10-16):**
  - "Tu modelo de regresión lineal explica" pasa a "Tu modelo explica" en `[interpretacion] reg`. Los
    fixtures de Árboles y KNN conservan el texto de Python, y `parity_components.rs` lo normaliza en el
    valor esperado.
  - `nota_validacion_pequena` usa `{margen}` = 2·sqrt(p(1−p)/n) en lugar de un 0,2 fijo (Iris: 24 filas,
    p=0,9 → 0,12).
  - R² negativo: con LR < 0,1, `aviso_r2_negativo` aconseja subirlo; con LR ≥ 0,1,
    `aviso_r2_negativo_lr_alto` aconseja bajarlo.
  - Una neurona "muerta" con tanh o sigmoid solo significa que no supera el umbral en las muestras
    analizadas (`muertas_no_relu`). `fuera_rango` es genérico; `fuera_rango_relu` es para ocultas ReLU.
- **F5d, interfaz (2026-10-16):**
  - En `--smoke-test`/`--screenshot` no se abre ningún diálogo: `export_model` y `nn_save_batch`
    escriben en `$TMPDIR/mltutor-automatizacion-XXXXXX/` (nombre aleatorio, 0700, sale en el log; archivos
    con `create_new` y sufijo `-2` si ya existen). El lote lee el CSV de `--scenario nombre:ruta` o escribe
    uno con 25 filas. **tempfile 3.27:** `Builder::tempdir()` crea con 0777 & umask (0755), no 0700: hay
    que pasar `.permissions(Permissions::from_mode(0o700))`.
  - El CSV de resultados del lote se queda en Rust (`NnBatches`) y a la UI solo viajan 100 filas.
  - Las columnas repetidas del lote se detectan con `CsvNotice::RenamedHeaders` (solo bloquean si son
    características del modelo) y el lote admite 1 columna (`min_columns: 1`).
  - Fecha de los metadatos: el backend valida `AAAA-MM-DD HH:MM` enviado por la UI; si no encaja usa
    `trained_at_ms` en UTC con el sufijo " UTC".
  - CSV de resultados: BOM UTF-8 y los campos que empiezan por `= + @ \t \r` o `-`+no dígito llevan `'`
    delante. Con salida tanh no hay columna "Confianza". Una fila no finita no tiene clase (regresión: celda
    vacía).
  - **Divergencia recuperada (K1):** las vistas se bloquean por `network_valid` (`is_valid()`), no por
    `diverged_epoch`. Caso: California, [64,64], SGD, LR 0,3, dropout 0, batch 64, paciencia 10 → diverge
    en la época 3 y restaura la 1 (base 1); escenario `redes_evaluacion_divergencia_recuperada`.
  - Superficie de regresión: escala compartida salvo que el rango predicho sea < 20 % del real
    (`OWN_SCALE_FRACTION`, el caso de Diabetes por defecto).
  - En Plotly, un eje derecho con `overlaying: "y"` toma las marcas del izquierdo; se fija con
    `tickmode: "linear"`.
  - Corregido por el coordinador: el predictor NumPy se guardaba con el filtro y la extensión de un `.py`
    aunque el contenido era un ZIP.
  - Medidas: `.app` de 16,8 MiB; smoke test de 55–80 s con 23 comprobaciones.
- **F5d, cálculo (2026-10-16):**
  - C29 vive en `mltutor_core::nn_predict` sobre el trait `NetworkOutput`, porque el core no ve
    `NeuralNetwork`.
  - `show_neural_network_predictions` (código muerto en Python) se puede renderizar sin interfaz con
    `LessonRenderer`, inyectando `state["nn_batch_predictions"] = (nombre, bytes)`; así se generó
    `nn/predictions.json`.
  - Python formatea las salidas f32 de Keras con `{:.6f}`, y un ulp cambia el último dígito: los tests
    de textos numéricos comparan con tolerancia.
  - `predecir_numpy.py` necesita `modelo.json`, así que se descarga como `red_predictor_numpy.zip`.
  - Metadatos (D75): `num_classes` = 2 en binaria, la fecha del entrenamiento y el `target_column` real.
    Se pueden generar con una red divergida (`valid: false`).
  - Con 5×256 en California: exploración 2 ms; lote de 20 640 filas en 204 ms (431 KB).
- **Revisión F5c/F5d, core (2026-10-16):**
  - Lote 🧠: una fila con salida NaN/∞ queda con `finite[i] = false` y clase `None`; antes, `NaN > 0.5`
    era falso y le daba la clase 0. Las entradas NaN, ±∞ o con `|v| > f32::MAX` se rechazan con
    `ModelError::NonFiniteCell { row, column }` (índices desde 0; `NonFinite` sigue sin campos).
  - CSV: `read_csv_cancellable_with(…, ReadOptions { min_columns })` (2 por defecto) permite el lote de una
    red de una sola característica. `CsvNotice::RenamedHeaders` informa de las cabeceras repetidas (sin las
    `Unnamed: i`) y también sale en 📁 Cargar CSV.
  - Casi divergencia = `History.near_divergence`: alguna `val_loss` finita > 1e6 × `val_loss[0]`, sin
    `diverged_epoch`. Con divergencia real solo se muestran los avisos de divergencia.
  - `r2_negative_notice(r2, lr)` usa el LR configurado: con LR ≥ 0,1 (también el 0,1 en f32) → `_lr_alto`.
  - Metadatos: `num_classes` es null en regresión; `weights_epoch` "best"/"last". Con EarlyStopping los
    `best_*` describen los pesos del archivo y los `final_*`, la última época.
  - El umbral binario `> 0.5` y el criterio de neurona muerta `all(a <= umbral)` no dependen de la
    activación (tanh incluida).
- **Contenido de F5d (2026-10-16):**
  - `red_entrenamiento.py.j2` recibe `seed` (obligatoria en modo strict): con Keras 3.12 en CPU, la
    misma semilla reproduce el historial bit a bit.
  - Extrapolación: con los pesos de `iris_default`, un pétalo de 20 cm da virginica con 0,996 (la
    confianza máxima en las flores reales es ~0,8). La red ReLU es lineal a trozos y se vuelve más segura
    fuera de los datos.
  - La exploración (D81) son curvas ICE de una sola muestra, con 50 valores entre el mínimo y el máximo
    del DATASET ENTERO. El "impacto relativo" es rango / |media de las predicciones| (depende del cero
    del objetivo).
  - Los avisos de fuera de rango usan el rango de ENTRENAMIENTO sin escalar.
  - En la predicción por lotes, las columnas se buscan por nombre (sobrar columnas o venir en otro orden
    no es un error) y se escala con el escalador del entrenamiento.
- **Revisión Rust F5a, correcciones (2026-10-16):**
  - `f32::max(NaN, 0.0)` devuelve 0, así que una ReLU hecha con max OCULTA la divergencia; se usa
    `if v < 0 { 0 }`.
  - Divergir no es un error: se sigue entrenando como Keras, se marca `History.diverged_epoch` y
    `NeuralNetwork::is_valid()` pasa a false. Las vistas devuelven `Numerical` y las exportaciones
    `InvalidModel`. Los f64 del historial se serializan como "nan"/"inf" (`serde_nonfinite`).
  - Keras 3.12: ReduceLROnPlateau y EarlyStopping empiezan con `best = None` (la primera época siempre
    es la mejor, también con NaN); `best = inf` era de Keras 2. Hay test en `engine.rs`.
  - Productos en paralelo por filas de salida (rayon, umbral 2^19), bit a bit iguales con cualquier
    número de hilos. 5×256 en California pasa de 3,18 a 0,92 s por época. Con un umbral de 2^15,
    [64,64] iba el doble de lento.
  - `histogram_plotly` reproduce `go.Histogram(nbinsx)` de plotly.js: ancho redondeado a {2,5,10}·10^k,
    desplazamiento de medio bin y recorte de los extremos vacíos. Validado con plotly.js en jsdom.
  - `Scaler`: aplica la regla `_is_constant_feature` de sklearn. `modelo.json` usa el decimal f32 más
    corto (de 10 a 3,2 MB con 5×256).
  - ONNX MLP: `label` = código del dataset (`Gather`), y en binaria `probabilities` = `Concat(1−p, p)`.
  - `training_cost` (K6): 2e9 de work por segundo; < 10 s → Low, 10–60 s → High, más → VeryHigh.
  - zsh no parte `$args`: los `--only` de export_golden.py se escriben explícitamente.
- **F5c (2026-10-16):**
  - El hash del mapa de neuronas del canvas de Python, `((h^(h>>13))*1274126177)^…` sobre dobles de JS,
    nunca pasa de 0,5: con "% conexiones" ≥ 50 dibujaba todo y la dispersión era asimétrica. Rust usa
    `Math.imul`.
  - El mapa de 5×256 tiene 281 000 conexiones: va en un canvas HiDPI (`NetworkMap.svelte`), no en SVG.
  - `nn_evaluation` es un comando aparte (`evaluate_model` sigue igual), y las vistas que necesitan los
    pesos pasan por el trait `NnEngine`.
  - La superficie de regresión usa escalas de color separadas (Diabetes por defecto predice ~0,8 con
    objetivos de 25–346).
  - Divergencia reproducible: Diabetes, tanh, SGD, LR 0,3, sin callbacks, en la época 7.
  - Los bins de `nbinsx` de Plotly pueden ser menos de los pedidos (39 en lugar de 50 en Iris).
  - Medidas con 5×256: `nn_weights` 36 ms (9 KB), `nn_activations` 39 ms (89 KB). Smoke test de 52 s
    con 20 comprobaciones; `.app` de 16,5 MiB.
- **Canvas en blanco (2026-10-20, mapa 🗺️ de 📉 Activaciones):** CUALQUIER asignación a `canvas.width`
  o `canvas.height`, aunque sea del mismo valor, borra el bitmap. `NetworkMap.svelte` tenía `width={width}`
  en el `<canvas>` y `draw()` lo redimensionaba desde un `$effect`. Al montar, el efecto del
  ResizeObserver cambia `width` (900 → ancho real) y Svelte 5.57 ejecuta el `$effect` de `draw` ANTES
  que el efecto de plantilla que vuelve a aplicar el atributo: el mapa salía vacío en la ventana visible.
  Con `--full-page` no se veía, porque el redimensionado posterior de la ventana provocaba otro dibujo
  en el orden correcto. Regla: un `<canvas>` que se dibuja a mano no lleva atributos `width`/`height` en
  la plantilla; el bitmap es solo de `draw()`. `networkMap.test.ts` lo comprueba con un contexto simulado.
- **Plantillas de redes (2026-10-16, revisión ML F5a):**
  - `reconstruir_modelo.py` usa `np.allclose(rtol=1e-5, atol=1e-4)`: con salidas de ~3e5, el error
    absoluto de Keras f32 frente a la app ronda 3e-2, aunque el relativo sea de ~1e-7.
  - `red_entrenamiento.py.j2` recodifica SIEMPRE con LabelEncoder un objetivo CSV de clasificación.
    Sin eso, con clases 1/2/3 falla `to_categorical`, y con un binario 1/2 se entrena sin error pero
    con BCE negativa y un 37 % de acierto.
  - El diagnóstico de regresión usa los textos `*_regresion` (`convergio`, `salud_bueno`,
    `estado_bueno`, `rec_bueno`, `rec_bueno_otro`), porque "bueno" solo significa que bajó el MAE.
  - Los `nn/exact/*.json` guardan como texto exacto el script y el LEEME del paquete: si cambias esas
    plantillas, regenera `nn/exact`.
- **`nn-epoch` (2026-10-16, F5b):** cada evento lleva todas las épocas desde el anterior (como mucho 20
  eventos por segundo; la última siempre se envía). Al terminar, la UI pide `nn_history`, que es la
  fuente fiable, porque el último lote puede llegar después de la respuesta de `train_model`.
  `DatasetView::train_observed` y `script_base` NO tienen implementación por defecto: con una por defecto,
  los envoltorios (`CachedViews`) se tragaban los eventos.
  - `ModelConfig` ya no es `Copy`.
  - El smoke test falla con cualquier `.notice.error`, salvo dentro de `[data-pedagogical]`.
  - `bind:clientWidth` en un SVG con `height: auto` produce "ResizeObserver loop…" en WebKit.
  - El diagrama resume las capas de más de 24 neuronas (12 + ⋮ + 12).
  - Medidas: `.app` de 16,2 MiB; smoke test de 17 s con 17 comprobaciones.
- **D68 resuelto (2026-10-16): MLP propio en f32, SIN burn.** Las entradas de burn de más arriba quedan
  obsoletas.
  - Los productos se acumulan en f64 en orden fijo y se redondean una vez a f32. Acumulando en f32, el
    gradiente de Diabetes se salía de la tolerancia.
  - Iris, 100 épocas: 5 ms (burn daba unos 90 ms); California, 100 épocas: 0,66 s (Python unos 30 s).
  - Sin Mutex global y 4 MB menos.
  - RNG (D70): `RandomState(seed)` de NumPy (MT19937), en el orden Glorot capa a capa → por época, la
    permutación → por batch, las máscaras.
  - Los floats f32 de la UI (`lr_factor`, `dropout`, `learning_rate`) se pasan a f64 por su
    representación decimal (`f32_as_decimal`), como en Python.
  - `EpochLog.epoch` empieza en 0 (la UI muestra `epoch + 1`). Los errores van en `nn_data::NnError`.
  - Paridad (a): gradientes ≤ 3e-5, 17/18 historiales en todas las épocas, pesos finales ≤ 3,3e-5.
  - Paridad (b): las 11 medias dentro de la banda.
  - ONNX MLP: ≤ 1,1e-6 frente a Keras, sin etiquetas distintas.
- **Errores de los fixtures de redes:**
  - `iris_dropout_masks.json` guarda los pesos POSTERIORES a los 5 pasos (golden_nn muta `w0` en sitio);
    hay que usar `iris_dropout_masks_w0.json`.
  - `np.polyfit` en tercios planos o empatados da signos de ruido de LAPACK; esas filas del diagnóstico
    no son normativas.
  - En California, la referencia NumPy ya difiere de Keras en más de 1e-5 desde la época 0, así que se
    compara a 1e-3.
- **Paridad de redes en dos niveles (D69, 2026-10-15):** con los pesos iniciales de Keras,
  `shuffle=False` y dropout 0, `tools/nn_reference.py` (NumPy f32) reproduce el `fit` de Keras 3.12 en
  17 de 18 casos durante todas las épocas (pesos finales ≤ 5e-6 casi siempre). Las trayectorias caóticas
  (RMSprop con LR 0,3) divergen en la época 2. Sin esas condiciones solo cabe paridad estadística
  (`nn/stat`, 5 semillas). Fixtures generados con TF 2.16.2 (tensorflow-macos) y Keras 3.12 en CPU, con
  `enable_op_determinism` y un hilo.
- **Keras 3.12 frente a burn (no usar los optimizadores ni el RNG de burn):**
  - BCE y CCE se calculan desde los logits si la salida es sigmoid o softmax, sin recortar con ε.
  - Adam aplica ε sin corregir: `w −= m·α/(sqrt(v)+ε)` con `α = lr·sqrt(1−β2ᵗ)/(1−β1ᵗ)`.
  - RMSprop pone ε dentro de la raíz: `lr·g/sqrt(v+ε)`.
  - Los logs de época son medias ponderadas por el tamaño del batch, con las predicciones de antes de
    cada paso.
- **Callbacks de Keras 3.12:**
  - `validation_split` toma las ÚLTIMAS `n − floor(n·(1−v))` filas, sin barajar.
  - EarlyStopping con `restore_best_weights` restaura los mejores pesos aunque no llegue a parar.
  - ReduceLROnPlateau usa `min_delta` 1e-4 y registra `learning_rate` antes de reducir. La app usa
    paciencia `max(1, paciencia // 2)`, o 5 sin early stopping.
- **Split y escalado de la red:** la lección usa `train_test_split(test_size=0.2, random_state=42,
  stratify=y)` (distinto de D7) y escala con TODAS las filas antes de dividir. Es una fuga de datos que se
  mantiene por paridad (D72). `nn_reference.stratified_split` lo reproduce con `RandomState.permutation`.
- **tensorflow-metal 1.2.0:** `model.predict` tras `set_weights` da salidas erróneas en la GPU de Apple,
  mientras que `model(x)` es correcto. La app oculta la GPU (`USE_GPU=0`); los fixtures y las plantillas
  fuerzan la CPU (D93).
- **Fallos de la red en Python:**
  - el selector de salida no tiene `key` y conserva su valor entre datasets ("softmax" de Iris rompe
    Cáncer);
  - tras un entrenamiento fallido, `model_trained_nn=True` con `nn_model=None` rompe 📈 y 🎯;
  - el diagnóstico de regresión compara solo con las 5 primeras épocas (Diabetes con R² −3,4 sale
    "👍 Bueno");
  - `viz/nn_extra.py` está muerto entero.
  - En `golden_render`, Moons hay que sembrarlo (`moons_X`/`moons_y` en el `state`).
- **Contenido de redes (2026-10-16):**
  - El diagnóstico de Python dice "inestable (CV 0,3 %)" con la pérdida estable y un rendimiento bajo,
    porque `is_stable` incluye la calidad. Rust usa las claves `*_rendimiento_bajo` en ese caso
    (aprobado).
  - Las tendencias usan la pérdida de ENTRENAMIENTO y los umbrales de pendiente son absolutos (con el MSE
    de Diabetes sale "Excelente"). "Convergió" solo mide que la pérdida bajó. Los textos están corregidos.
  - Keras 3.12 en CPU con `set_random_seed` es determinista entre ejecuciones (reproduce `nn/stat` bit a
    bit).
  - Con los valores por defecto (D88): Iris 0,63–0,77, y con LR 0,01, 0,93–0,97; Diabetes con R² de −3,8 a
    −2,8, y con LR 0,01, de 0,19 a 0,43.
  - minijinja de Python 2.16 pasa a los filtros los flotantes enteros como `int` (`0.0` → `0`); el
    `py_repr` de Rust no lo sufre.
  - `StandardScaler` ajustado con un DataFrame avisa si luego se transforma un array, así que en las
    plantillas hay que pasar DataFrames.
- **Separación J2 (2026-10-15):** separación = min M[i,j]/max(W_i,W_j). Con el split de la app, euclídea
  y K=5: Iris 1,58, Dígitos 1,11 (moderada, aunque acierta el 99,3 %), Cáncer 1,26, Vino 1,04 y CSV
  string_target 0,88. Es una medida global y no predice el acierto del KNN.
- **`threadpool_limits(limits=1)` sin `with`** deja el límite activo hasta el final del proceso. Solo
  afecta a las bibliotecas ya cargadas, así que hay que llamarlo después de importar sklearn.
  `threadpoolctl` viene como dependencia de sklearn.
- **Plotly y `%{` (2026-10-15, revisión F4):** Plotly no tiene escape para `%{…}` en
  `hovertemplate`/`texttemplate`: `%%` no sirve, porque escanea con `/%{([^\s%{}:]*)…}/`. Los datos
  interpolados en la plantilla pasan por `escapePlotlyTemplate` (`%` → `&#37;`). Lo que va por
  `customdata`/`text` solo necesita `escapePlotly`.
- **`cancel_task` antes que el comando:** los dos comandos son async y concurrentes, así que el cancel
  puede llegar antes de `tasks.start(id)`. `TaskRegistry` guarda esos ids pendientes durante 10 s y
  recuerda los últimos 256 terminados, porque el contador de la ventana se reinicia al recargar.
- **KNN, evaluación y lienzo:** los vecinos del test se buscan una sola vez dentro de `train_model`
  (cancelable) y se cachean (`TestOutputs.test_neighbors`). `CoreModel.playground` cachea el KNN 2D por
  (par, K, pesos, métrica), de modo que arrastrar no reentrena. La UI corta en 200 puntos, con el mismo
  texto que el backend (lo comprueba vitest).
- **Veredicto de distancias (J2):** se basa en la separación también en binaria (`SeparationVerdict` sin
  `Binary`). Sin separación (una clase con 1 muestra o con todas sus muestras idénticas) se vuelve a los
  textos de Python por el ratio.
- **Lienzo KNN (2026-10-15):** Plotly no tiene marcadores arrastrables, así que se usa `dragmode: false`
  con `pointerdown/move/up` en fase de captura y la conversión de píxeles con `_fullLayout.{x,y}axis`
  (`_offset`, `_length`, `p2l`, `l2p`). `setPointerCapture` lanza una excepción con punteros sintéticos
  (va en un `try`). En ventanas ocultas `requestAnimationFrame` no corre, de modo que se usa rAF o
  `setTimeout(32)`, lo que llegue antes. `knn_playground` tarda ≤ 0,1 ms en el backend, y el arrastre
  22–31 ms de puntero a redibujado. `performance.now()` en WKWebView tiene una resolución de 1 ms.
  `ModelConfig::task()` pasa a ser `task_on(dataset_task)`/`fixed_task()`, porque la tarea del KNN depende
  del dataset. La FixtureFacade de KNN desempata por el índice menor (solo para desarrollo); la paridad
  exacta se prueba en `real_knn.rs` y en el core.
- **Medidas F4:** `.app` de 15,8 MiB; smoke test de unos 37 s con 13 comprobaciones.
- **KNN en Rust (2026-10-15, F4):** `mltutor-core::knn` transcribe `knn_reference.py`.
  - Con kd_tree, los vecinos del test salen exactos en orden y con distancias bit a bit (`mul_add` en el
    resto `n % 8`). Las 806 677 celdas de rejilla de los fixtures coinciden y el `nth_element` de libc++
    está verificado con 260 secuencias.
  - Con fuerza bruta euclídea, las distancias difieren ≤ 9e-12 por BLAS; con `weights='distance'`, las
    probabilidades, ≤ 4,3e-13.
  - `kneighbors` usa hasta 8 hilos deterministas (`std::thread::scope`) a partir de 2048 filas y comprueba
    la cancelación cada 256.
  - Tiempos en California: fit 2 ms, predecir el test 4 ms, permutación 0,17 s (sklearn 0,9 s) y
    rejillas ≤ 26 ms.
  - `mul_add` en x86-64 es una llamada a `fma` de libm (FMA3 por hardware con glibc/UCRT). Falta medirlo
    en Windows con `bench_knn`.
- **ONNX de KNN (D57):** prost necesita `TensorProto` (`raw_data` LE), `Attribute.t`/`g` (con `boxed`) y
  `Graph.initializer`. Solo usa el opset `ai.onnx` 15. Pasa `full_check` y onnxruntime en 18 casos; frente
  a la app solo discrepa 1 etiqueta de Titanic, por un empate.
- **Contenido de KNN (2026-10-15):**
  - Con `weights='distance'`, la exactitud o el R² de entrenamiento vale 1 con CUALQUIER K, porque cada
    muestra está a distancia 0 de sí misma: no sirve para detectar sobreajuste.
  - El ratio Max/Min de 📏 no mide la separabilidad (Iris da 2,53 aunque versicolor y virginica se
    solapan); D67 compara con la distancia dentro de cada clase.
  - Vino sin escalar: K=5 da 0,741, y con StandardScaler 0,963 (proline decide los vecinos).
  - `frontera_decision.py.j2` tiene una variable opcional `datos` ("todos" en KNN, D63) que se comprueba
    con `is defined`, de modo que los contextos de F2/F3 no cambian.
  - El script del paquete KNN debe tolerar empates de plataforma: compara las distancias, no solo los
    índices.
- **skl2onnx 1.19.1 y KNN:** con opset ≥ 11, `weights='distance'` sale como `uniform` (distancias de
  `TopK` negadas), y usa `1/max(d,1e-6)` en vez de la regla de distancia 0. onnxruntime no tiene
  `OneHot` con salida double. Se usa un grafo propio (Scan + TopK, `Equal`+`Cast`).
- **Fallos del KNN en Python:**
  - el lienzo interactivo calcula en JS con reglas propias, y `test_points` nunca se llena;
  - el análisis de distancias usa siempre test 0,3, y en binaria su ratio vale siempre 1;
  - Probabilidades y ROC muestran "Clase i";
  - con K > n_train entrena y luego fallan las vistas;
  - la exportación es la de Regresión (`linear_regression_model.pkl`).
  - Frontera, superficie, lienzo y formulario usan todo el dataset (D63 lo mantiene).
- **Capturas de WebEngine:** el lienzo de Qt WebEngine se captura aparte, con Chrome headless
  (`--headless=new --screenshot`) sobre el HTML guardado. `--only knn` tarda unos 8 min (12 MB).
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
