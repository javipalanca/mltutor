# AGENTS.md — Guía para agentes de código en MLTutor

MLTutor es una aplicación educativa de escritorio para aprender Machine Learning
en español. **El proyecto está en migración de Python/Qt a Rust.** La versión
Python (PySide6 + scikit-learn + TensorFlow) es la referencia funcional y visual;
la versión Rust debe igualarla en contenido, resultados y aspecto, y superarla en
velocidad, tamaño y estabilidad.

> **Plan de migración:** [docs/migration/PLAN.md](docs/migration/PLAN.md) (fases,
> decisiones y criterios de aceptación) e [INVENTARIO.md](docs/migration/INVENTARIO.md)
> (lista de funciones que hay que mantener). Toda tarea de migración pertenece a una fase
> del plan y a casillas del inventario. Al terminarla, marca las casillas.
>
> **Memoria del proyecto:** antes de empezar una tarea lee [MEMORY.md](MEMORY.md).
> Contiene decisiones, trampas conocidas y lecciones aprendidas que no se deducen
> del código. Cuando descubras algo no obvio (una diferencia numérica con
> scikit-learn, una limitación de un crate, un fallo de plataforma), añádelo allí
> con fecha absoluta.

## Ramas

- `feature/rust-migration`: migración a Rust. **Aquí se trabaja.**
- `codex/desktop-migration-study`: versión Python/Qt de referencia (solo bugs).
- `main`: versión original Streamlit y su despliegue web (Docker, Swarm, nginx).

## Stack objetivo (Rust)

| Capa | Elección | Motivo |
|---|---|---|
| Aplicación | **Tauri 2** | Binario pequeño y rápido; interfaz web local que permite reutilizar Plotly, Markdown y el estilo actual |
| Interfaz | **Svelte 5 + TypeScript + Vite**, Plotly.js y KaTeX empaquetados en local | Componentes reactivos ligeros; gráficos interactivos y fórmulas sin CDN |
| ML clásico | **linfa 0.8** (`linfa-linear`, `linfa-logistic`) sobre `ndarray 0.16`; CART y KNN propios | API tipo scikit-learn; linfa-trees y KdTree descartados (ver MEMORY.md) |
| Redes neuronales | **burn 0.22** con backend `flex` + autodiff (CPU, Rust puro) | Entrenamiento nativo, portátiles de laboratorio sin GPU |
| Lo que falte | Implementación propia en `mltutor-core` | Árboles de regresión, permutation importance, fronteras de decisión, curvas ROC, etc. |

Principio: **se calcula en Rust y se enseña scikit-learn/PyTorch/Keras.** El código
de ejemplo que se muestra al estudiante sigue siendo Python, porque es lo que usará
fuera de la app. Ese código debe ser correcto y reproducir lo que la app calcula.

## Estructura (objetivo)

```
Cargo.toml                 workspace
crates/mltutor-core/       algoritmos, métricas, preprocesado, datasets embebidos.
                           Sin Tauri ni UI; 100 % testeable con `cargo test`
crates/mltutor-nn/         redes neuronales con burn (arquitectura, entrenamiento, historial)
crates/mltutor-export/     ONNX nativo y "paquetes Python" (JSON + script sklearn/Keras)
src-tauri/                 app Tauri: comandos IPC, estado, tareas en segundo plano,
                           diálogos de archivo, exportaciones
ui/                        frontend Svelte + TS: lecciones, controles, gráficos
content/                   textos pedagógicos en Markdown (+KaTeX), uno por lección/pestaña,
                           y plantillas de código Python (code/*.py.j2, minijinja)
docs/migration/            PLAN.md, INVENTARIO.md y un plan por módulo
tests/parity/              fixtures "golden" generados desde la versión Python
tools/export_golden.py     genera esos fixtures con la app Python
mltutor/, tests/*.py       versión Python de referencia: congelada salvo bugs graves
```

## Comandos

```sh
# Rust
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace           # también regenera ui/src/lib/bindings (ts-rs)

# Frontend (la CLI de Tauri es una devDependency de ui/; no hace falta cargo-tauri)
npm --prefix ui ci
npm --prefix ui run check        # svelte-check + tsc
npm --prefix ui run test -- --run  # vitest (sin --run se queda en modo watch)

# App
npm --prefix ui run tauri dev    # app en desarrollo (Vite con recarga en :5173)
npm --prefix ui run tauri build  # release: .app (macOS), AppImage (Linux), mltutor.exe sin instalador (Windows)
target/release/mltutor --smoke-test          # recorre las rutas y prueba 📊 Datos (correlación y pairplot de Iris); 0 ok, 1 fallo, 2 timeout (180 s)
target/release/mltutor --screenshot out.png  # captura de la ventana (solo macOS por ahora)
target/release/mltutor --screenshot out.png --scenario datos_pairplot_kde_iris --full-page  # pantalla concreta, página entera
#   escenarios en ui/src/lib/scenarios.ts (nombres de docs/migration/capturas/qt/); los CSV llevan la ruta: csv_cargado_clasificacion:<ruta.csv>
npm --prefix ui run tauri build -- --debug --no-bundle  # binario de depuración con el frontend embebido (target/debug/mltutor), rápido para iterar
MLTUTOR_FACADE=fixture target/debug/mltutor  # solo en debug: datos de tests/parity en lugar de mltutor-core
# Ojo: `cargo build --release` a secas no empaqueta el frontend (carga devUrl); usa `tauri build`.
# Registro: ~/.mltutor/mltutor.log (MLTUTOR_LOG_DIR lo cambia, útil en CI)

# Referencia Python (para comparar y generar fixtures)
uv sync --locked --extra dev
uv run python launcher_qt.py
uv run pytest -q
uv run python tools/export_golden.py      # --check en CI
uv run python tools/capture_qt.py         # capturas de referencia Qt
```

No ejecutes `release.sh`, no crees tags ni hagas push sin permiso explícito.

## Reglas que no se deben romper

1. **`mltutor-core` no depende de la UI.** Funciones puras sobre `ndarray`/structs;
   Tauri solo adapta tipos (serde) y orquesta.
2. **Nunca bloquees el hilo principal.** El entrenamiento y el cálculo pesado van en
   `tauri::async_runtime::spawn_blocking`. El progreso se envía como eventos y la
   cancelación es cooperativa (un `Arc<AtomicBool>` comprobado en los bucles).
3. **Sin `unwrap()`/`expect()` en código de producción**, salvo invariantes
   documentados. Errores con `thiserror` en core; la capa Tauri los traduce a
   mensajes en español para el estudiante. Un panic no puede cerrar la app.
4. **Determinismo:** todo lo aleatorio recibe semilla explícita (`StdRng::seed_from_u64`),
   con `42` por defecto, como `random_state=42` en la versión Python.
5. **`f64` en todo el core.** Comparaciones con tolerancia, nunca `==` en flotantes.
6. **Sin red en tiempo de ejecución.** CSP estricta en `tauri.conf.json`; Plotly,
   KaTeX, fuentes y datasets van empaquetados (`include_bytes!` en el caso de los datasets).
7. **Paridad antes que novedad.** Cada algoritmo portado tiene un test que lo
   compara con el fixture de scikit-learn/TensorFlow, con tolerancia documentada. Si
   no puede coincidir (otro solver, otro desempate), se explica en MEMORY.md.
8. **Contenido como datos.** Los textos pedagógicos van en `content/*.md`, no
   incrustados en Rust ni en componentes. Se conserva el orden de pestañas, emojis,
   colores y explicaciones de la versión Python.
9. **Español** en toda la interfaz y en los mensajes de error. Identificadores,
   comentarios y commits en inglés.

## Definición de terminado (cada tarea de migración)

1. Las casillas correspondientes de `docs/migration/INVENTARIO.md` están marcadas.
2. `cargo fmt`, clippy sin avisos, `cargo test --workspace` y `npm --prefix ui run check/test`
   en verde. Los tests de paridad cubren el cálculo nuevo.
3. Hay una captura Rust comparada con la captura Qt equivalente, sin pérdidas de
   contenido ni de vistosidad.
4. Lo aprendido que no sea obvio está en MEMORY.md.
5. Han revisado `rust-reviewer` y, si hay ML, `ml-correctness-reviewer`.

## Convenciones

- Rust edition 2021+, `rustfmt` por defecto, clippy sin avisos.
- Tipos de dominio explícitos (`Dataset`, `TrainConfig`, `Metrics`…) serializables
  con `serde` para el IPC; mantén los tipos TS sincronizados (p. ej. `ts-rs` o `specta`).
- Un comando Tauri = una acción de usuario; respuestas pequeñas. Las imágenes
  grandes no se transfieren: se envían datos y la UI dibuja con Plotly.
- Pruebas: unitarias en core, de paridad en `tests/parity`, de componentes con
  vitest y un smoke test de arranque en CI para cada plataforma.

## Subagentes disponibles (`.claude/agents/`)

| Agente | Úsalo para |
|---|---|
| `migration-porter` | Inventariar un módulo Python, planificar su port y generar fixtures de paridad |
| `rust-ml-engineer` | Algoritmos en `mltutor-core`/`mltutor-nn` con linfa, burn o implementación propia |
| `tauri-ui-engineer` | Comandos Tauri, frontend Svelte, Plotly, estilos, IPC y tareas en segundo plano |
| `ml-lesson-author` | Contenido pedagógico en `content/` y su presentación en lecciones |
| `ml-correctness-reviewer` | Revisión de rigor de ML y de la paridad numérica (solo lectura) |
| `rust-reviewer` | Revisión de código Rust: errores, panics, concurrencia, rendimiento (solo lectura) |
| `packaging-release-engineer` | Bundler de Tauri, CI en tres plataformas, releases; también el PyInstaller heredado |
| `test-runner` | Ejecutar cargo/npm/pytest y resumir fallos |
| `qt-desktop-engineer` | Solo mantenimiento de la versión Python/Qt heredada |
