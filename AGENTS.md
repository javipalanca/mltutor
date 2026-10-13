# AGENTS.md — Guía para agentes de código en MLTutor

MLTutor es una aplicación educativa de escritorio para aprender Machine Learning
en español. **El proyecto está en migración de Python/Qt a Rust.** La versión
Python (PySide6 + scikit-learn + TensorFlow) es la referencia funcional y visual;
la versión Rust debe igualarla en contenido, resultados y aspecto, y superarla en
velocidad, tamaño y estabilidad.

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
| ML clásico | **linfa** (`linfa-trees`, `linfa-linear`, `linfa-logistic`, `linfa-nn`) sobre `ndarray` | API tipo scikit-learn |
| Redes neuronales | **burn** con backend `ndarray` (CPU) | Entrenamiento nativo, portátiles de laboratorio sin GPU |
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
src-tauri/                 app Tauri: comandos IPC, estado, tareas en segundo plano,
                           diálogos de archivo, exportaciones
ui/                        frontend Svelte + TS: lecciones, controles, gráficos
content/                   textos pedagógicos en Markdown (+KaTeX), uno por lección/pestaña
tests/parity/              fixtures "golden" generados desde la versión Python
tools/export_golden.py     genera esos fixtures con la app Python
mltutor/, tests/*.py       versión Python de referencia: congelada salvo bugs graves
```

## Comandos

```sh
# Rust
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo tauri dev                  # app en desarrollo (frontend con recarga)
cargo tauri build                # instalables/portables de la plataforma actual

# Frontend
npm --prefix ui ci
npm --prefix ui run check        # svelte-check + tsc
npm --prefix ui run test         # vitest

# Referencia Python (para comparar y generar fixtures)
uv sync --locked --extra dev
uv run python launcher_qt.py
uv run pytest -q
uv run python tools/export_golden.py
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
