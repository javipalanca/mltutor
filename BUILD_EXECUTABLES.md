# Compilar, probar y publicar MLTutor (versión Rust)

Guía para desarrolladores. La guía para estudiantes y profesores (descargas y
primer arranque) está en el [README](README.md). Las reglas del proyecto y la lista
completa de comandos están en [AGENTS.md](AGENTS.md), y las trampas conocidas en
[MEMORY.md](MEMORY.md).

MLTutor es una app **Tauri 2**: el cálculo se hace en Rust y la interfaz es
**Svelte 5 + TypeScript + Vite**, con Plotly.js y KaTeX empaquetados en local, que
se muestra en el WebView del sistema (WebView2 en Windows, WKWebView en macOS y
WebKitGTK en Linux).

```
Cargo.toml                 workspace (versión en [workspace.package])
crates/mltutor-core/       algoritmos, métricas, preprocesado y datasets incrustados (sin UI)
crates/mltutor-nn/         MLP propio (entrenamiento con las fórmulas de Keras)
crates/mltutor-export/     ONNX nativo y paquetes Python
crates/mltutor-testkit/    solo para tests: MLTUTOR_BLESS (textos de plantillas de los fixtures)
src-tauri/                 app Tauri: comandos IPC, tareas, diálogos, registro, --smoke-test, --screenshot
ui/                        frontend Svelte + TS
content/                   textos pedagógicos (.md/.toml) y plantillas de código Python (code/*.j2)
tests/parity/              fixtures generados con la versión Python (tag v1.0.0), congelados
tools/                     verify_exports.py, check_trees_platform.py, transcripciones en Python de
                           sklearn/Keras (cart/knn/nn_reference.py) y data/seaborn/ (CSV sin red)
```

## Requisitos

En todos los sistemas:

- **Rust** con [rustup](https://rustup.rs). No hace falta elegir versión:
  `rust-toolchain.toml` fija `stable` con `rustfmt` y `clippy`, y rustup lo instala
  al ejecutar el primer `cargo` en el repositorio. La versión mínima del workspace
  es la 1.85 (`rust-version` en `Cargo.toml`).
- **Node.js LTS** con npm (el CI usa `lts/*`; en local se usa Node 24). La CLI de
  Tauri es una devDependency de `ui/`, así que no hace falta instalar `cargo-tauri`.
- Para la validación de exportaciones y la comprobación de árboles (opcional):
  [uv](https://docs.astral.sh/uv/); uv descarga Python 3.11 si no lo hay.

Además, por sistema:

| Sistema | Paquetes |
|---|---|
| macOS (arm64) | Xcode Command Line Tools (`xcode-select --install`) |
| Windows (x86_64) | Microsoft C++ Build Tools (carga de trabajo «Desarrollo para el escritorio con C++») y WebView2, que viene de serie con Windows 10/11 |
| Linux (x86_64) | Los paquetes del CI, en Ubuntu 22.04 (ver abajo) |

Linux (los mismos que instala `rust.yml`):

```sh
sudo apt-get install -y \
  build-essential curl wget file xdg-utils \
  libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev \
  libxdo-dev libssl-dev patchelf libfuse2
# solo para el smoke test sin pantalla: xvfb xauth dbus-x11 libgl1-mesa-dri
```

Compila el AppImage en Ubuntu 22.04: enlaza con la glibc del sistema donde se
compila, y uno hecho en una distribución más nueva no arrancará en 22.04.

> **Mac con Apple Silicon:** si el toolchain por defecto de rustup es x86_64,
> `cargo` falla fuera del repo con «Bad CPU type». Dentro del repo
> `rust-toolchain.toml` lo resuelve; fuera, `rustup default stable-aarch64-apple-darwin`.

## Compilar y ejecutar

```sh
npm --prefix ui ci                  # dependencias del frontend (respeta ui/package-lock.json)

npm --prefix ui run tauri dev       # app en desarrollo, Vite con recarga en :5173
npm --prefix ui run tauri dev -- -- -- --smoke-test   # pasar flags al binario en dev

npm --prefix ui run tauri build     # release (LTO fat: varios minutos)
npm --prefix ui run tauri build -- --debug --no-bundle  # target/debug/mltutor con el frontend embebido (~25 s)
```

`cargo build --release` a secas **no** empaqueta el frontend (el binario cargaría la
URL de desarrollo): usa siempre `tauri build`.

Salida de `tauri build`:

| Sistema | Resultado | Descarga que publica el CI |
|---|---|---|
| macOS | `target/release/bundle/macos/MLTutor.app` (firma ad-hoc) | `mltutor-macos-arm64.zip` |
| Windows | `target/release/mltutor.exe` (sin instalador: `bundle.active: false`) | `mltutor-windows-x86_64.exe` |
| Linux | `target/release/bundle/appimage/*.AppImage` | `mltutor-linux-x86_64.AppImage` |

El zip de macOS se hace con `ditto`, que conserva la firma del bundle:

```sh
ditto -c -k --sequesterRsrc --keepParent target/release/bundle/macos/MLTutor.app mltutor-macos-arm64.zip
```

La configuración está en `src-tauri/tauri.conf.json` y en los archivos por
sistema `tauri.macos.conf.json`, `tauri.windows.conf.json` y
`tauri.linux.conf.json`. El perfil `release` del workspace usa `opt-level = 3`,
`lto = "fat"`, `codegen-units = 1` y `strip = true`.

Variables útiles al ejecutar:

| Variable | Efecto |
|---|---|
| `MLTUTOR_LOG_DIR` | Carpeta del registro (por defecto `~/.mltutor`, archivo `mltutor.log`) |
| `MLTUTOR_FACADE=fixture` | Solo en debug: datos de `tests/parity` en lugar de `mltutor-core` |
| `MLTUTOR_FIXTURE_EPOCH_MS=40` | Con la fachada de fixtures, ralentiza la red reproducida (gráfica en vivo, «Cancelar») |
| `WEBVIEW2_BROWSER_EXECUTABLE_FOLDER` | Windows: runtime WebView2 «Fixed Version». La app la fija sola si hay una carpeta `WebView2` junto al `.exe` |

## Pruebas y calidad

```sh
cargo fmt --all                                    # el CI usa --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                             # también regenera ui/src/lib/bindings (ts-rs)

npm --prefix ui run check                          # svelte-check + tsc
npm --prefix ui run test -- --run                  # vitest (sin --run se queda en modo watch)
```

- Los tipos TypeScript de `ui/src/lib/bindings/` los genera ts-rs al ejecutar
  `cargo test` y van versionados: el CI falla si cambian o aparecen archivos nuevos
  sin commitear.
- El CI ejecuta `cargo test --workspace --all-targets` y, aparte,
  `cargo test --doc -p mltutor-core -p mltutor-nn -p mltutor-export`: en Windows,
  los doctests con `--workspace` no enlazan por la salida del build de Tauri.
- `Cargo.lock` y `ui/package-lock.json` siempre van versionados.

### Paridad con la versión Python

Los tests de paridad (`cargo test`) comparan con los fixtures de `tests/parity/`,
generados con la versión Python de referencia. Esa versión ya no está en el árbol
(está en el tag `v1.0.0`), así que los fixtures están **congelados**. La única
excepción son los textos de las plantillas `content/code/*.j2`: si cambias una a
propósito, se actualizan desde Rust y se revisa el diff:

```sh
MLTUTOR_BLESS=1 cargo test --workspace --all-targets   # reescribe solo los textos de plantilla
git diff tests/parity
```

Que esos scripts sigan funcionando y coincidan con scikit-learn/Keras lo comprueba
`tools/verify_exports.py` (abajo). Para regenerar los fixtures numéricos hay que
usar la app Python del tag `v1.0.0`; los pasos están en
[tests/parity/README.md](tests/parity/README.md).

El CI de Linux comprueba además que scikit-learn en x86-64 construye los mismos
árboles que los fixtures (generados en arm64):

```sh
uv run --no-project --python 3.11 \
  --with numpy==1.26.4 --with pandas==2.3.3 --with scikit-learn==1.7.2 \
  python tools/check_trees_platform.py
```

### Validación de exportaciones y plantillas de código

Cada ONNX, paquete Python y plantilla de `content/code/` que ofrece la app se
comprueba con onnxruntime, scikit-learn y Keras. Los scripts leen los datasets de
seaborn de `tools/data/seaborn/` (`SEABORN_DATA`), sin red. Primero se escribe el corpus con el código de la propia app
(`src-tauri/src/export_corpus.rs`) y después se verifica sin red:

```sh
MLTUTOR_EXPORT_CORPUS=/tmp/corpus \
  cargo test -p mltutor --bin mltutor export_corpus::export_corpus -- --ignored --exact

uv run --no-project --python 3.11 --with-requirements tools/verify_exports.lock.txt \
  python tools/verify_exports.py /tmp/corpus     # en CI, además, --timeout 300
# opciones: --only <subcadena>, --jobs N, --timeout S, --keep
```

El entorno está fijado con hashes en `tools/verify_exports.lock.txt`. Si cambias
`tools/verify_exports.requirements.txt`, regenera el lock:

```sh
uv pip compile --universal --generate-hashes --python-version 3.11 \
  tools/verify_exports.requirements.txt -o tools/verify_exports.lock.txt
```

### Smoke test

```sh
target/release/mltutor --smoke-test
# macOS, sobre el bundle: target/release/bundle/macos/MLTutor.app/Contents/MacOS/mltutor --smoke-test
```

Recorre todas las rutas y, en cada lección, entrena modelos reales y abre sus
vistas (📊 Datos, Regresión, Árboles, KNN y Redes, incluidas la cancelación de un
entrenamiento y las exportaciones). Códigos de salida: **0** correcto, **1** fallo,
**2** tiempo agotado (180 s). Registra los ms de cada ruta y comprobación, y no
abre diálogos: lo que exporta va a `$TMPDIR/mltutor-automatizacion-XXXXXX/` (la
ruta sale en el registro).

- **Linux sin pantalla:** `xvfb-run -a dbus-run-session -- ./MLTutor.AppImage --smoke-test`,
  con `APPIMAGE_EXTRACT_AND_RUN=1`, `WEBKIT_DISABLE_DMABUF_RENDERER=1`,
  `WEBKIT_DISABLE_COMPOSITING_MODE=1` y `LIBGL_ALWAYS_SOFTWARE=1`, como en el CI.
- **Windows:** el binario es de subsistema GUI, así que el código de salida solo se
  ve con `Start-Process -Wait -PassThru` en PowerShell.

### Capturas

```sh
target/release/mltutor --screenshot out.png                    # ventana inicial (macOS, Windows y Linux)
target/release/mltutor --screenshot out.png --scenario datos_pairplot_kde_iris --full-page
MLTUTOR_FACADE=fixture target/debug/mltutor --screenshot out.png --scenario arboles_arbol_iris
python3 -I .github/scripts/check_png.py out.png                # rechaza capturas vacías o de un solo color
```

Los escenarios están en `ui/src/lib/scenarios.ts` y en
`ui/src/lib/lessons/*/scenarios.ts`; los de CSV llevan la ruta
(`csv_cargado_clasificacion:<ruta.csv>`). `--full-page` es fiable en macOS; en
Windows y Linux es de mejor esfuerzo (en Linux, solo si el gestor de ventanas deja
crecer la ventana). Las capturas de referencia de la versión Qt están en
`docs/migration/capturas/qt/` (congeladas; las generaba `tools/capture_qt.py` del tag
`v1.0.0`) y las de la versión Rust en `docs/migration/capturas/rust/`.

### Benchmarks y presupuestos de tiempo

```sh
cargo bench -p mltutor-core --bench core       # unos 2 min; filtro: -- 'knn/'
cargo bench -p mltutor-nn --bench nn
cargo bench -p mltutor-export --bench onnx

# presupuestos por comando (tests #[ignore], en release)
cargo test --release -p mltutor-core \
  --test bench_trees --test bench_knn --test bench_regression --test bench_california \
  -- --ignored --nocapture --test-threads=1
cargo test --release -p mltutor-nn --test bench_nn -- --ignored --nocapture --test-threads=1
```

Los resultados y la comparación con scikit-learn y Keras están en
[docs/migration/benchmarks.md](docs/migration/benchmarks.md). Los presupuestos
están calibrados en un Mac con Apple Silicon; `MLTUTOR_BUDGET_FACTOR` los
multiplica (el CI usa 3).

## Integración continua (`.github/workflows/rust.yml`)

Se ejecuta en cada push a `main`, `develop`, `release/**`, `hotfix/**` y `feature/**`, en las pull requests que tocan
el código Rust, la UI, el contenido o los fixtures, con `workflow_dispatch` y con
los tags `v*`. Caché de cargo (`Swatinem/rust-cache`) y de npm.

| Job | Qué hace |
|---|---|
| `build` (Ubuntu 22.04, Windows, macOS arm64) | fmt, clippy, tests (y los doctests aparte), árboles en x86-64 (Linux), bindings de ts-rs al día, `npm ci`/`check`/`test`, `tauri build`, smoke test del binario empaquetado (Xvfb en Linux), aviso de WebView2 ausente (Windows), capturas validadas con `check_png.py`, empaquetado de la descarga (en macOS además `codesign --verify --deep --strict` y smoke test del zip descomprimido) y presupuesto de tamaño |
| `exports` (Ubuntu 22.04) | Corpus de exportaciones y verificación con `tools/verify_exports.py` (Python 3.11) |
| `benchmarks` (Ubuntu 22.04) | `cargo bench --no-run` de los tres crates puros y los presupuestos de tiempo `bench_*` en release |
| `release` (solo tags `v*`) | Necesita los tres jobs anteriores; ver [Releases](#releases) |

Artefactos (14 días): `mltutor-<plataforma>` con la descarga y
`screenshots-<plataforma>` con las capturas y los registros; `export-corpus` (7
días) solo si falla la verificación. Así se prueban las versiones en equipos
reales sin publicar una release.

Presupuestos:

- **Tamaño de la descarga** (`matrix.size_budget`, en bytes): Linux 99 000 000,
  Windows 21 900 000 y macOS 10 020 000, unos +15 % sobre lo medido. El resumen del
  job muestra el tamaño y el porcentaje usado. Solo se suben a propósito,
  explicando el motivo en MEMORY.md.
- **Tiempo:** el smoke test tiene un vigilante interno de 180 s (en CI tarda
  25–46 s) y un `timeout-minutes` de 8 por paso. Los jobs tienen un límite de 90
  (`build`), 45 (`exports`) y 40 minutos (`benchmarks`).

## Releases

**Solo se publica una release con la aprobación explícita del responsable del
proyecto.** Durante la migración no se publican versiones intermedias: se
validan con los artefactos del CI.

La versión vive en `Cargo.toml` (`[workspace.package]`) y debe coincidir con
`src-tauri/tauri.conf.json`, `ui/package.json` y los dos lockfiles (lo comprueba
el test `versions_are_in_sync`). `release.sh` se encarga de todo:

```sh
./release.sh 1.1.0-rc.1 --dry-run   # comprueba, enseña el diff de versiones y restaura los archivos
./release.sh 1.1.0-rc.1            # prerelease (desde cualquier rama)
./release.sh 1.1.0                  # estable (solo desde main, tras fusionar release/1.1.0)
```

1. Comprueba la rama (las estables solo desde `main`), que el árbol esté limpio y
   que el tag no exista en local ni en `origin`.
2. Actualiza la versión en los cinco archivos y verifica que coinciden.
3. **Pide confirmación (y/N)** en una terminal interactiva. Si se responde que no,
   si se interrumpe o si algo falla, restaura los archivos de versión.
4. Hace el commit `Release vX.Y.Z`, crea el tag anotado y hace push de ambos.
5. Espera a que `rust.yml` termine y comprueba que la release tiene las tres
   descargas y `SHA256SUMS.txt`, y que el estado de prerelease es el esperado.

Necesita `git`, `cargo`, `npm`, `python3` y GitHub CLI (`gh`) autenticado; con
`--dry-run` no hace falta `gh`.

El job `release` comprueba que el tag es `v` + la versión de `Cargo.toml`, exige
las tres descargas, genera `SHA256SUMS.txt` y publica la release con notas para
los estudiantes. Los tags con `-` (por ejemplo `v1.1.0-rc.1`) se publican como
prerelease y no pasan a ser la última versión.

## Firma

- **macOS:** firma *ad-hoc* del bundle completo (`"signingIdentity": "-"` en
  `tauri.macos.conf.json`). Sin ella, Gatekeeper diría que la app descargada «está
  dañada»; con ella, muestra el aviso normal de «no se puede verificar». El CI lo
  comprueba con `codesign --verify --deep --strict`. **Pendiente:** Developer ID y
  notarización, a falta de credenciales.
- **Windows:** el `.exe` no está firmado y SmartScreen avisa la primera vez.
  **Pendiente:** firma con SignPath Foundation (gratis para proyectos de código
  abierto). La licencia MIT del repositorio cumple el requisito de licencia OSI;
  falta la solicitud en signpath.org. Después, en CI: firmar `mltutor.exe` antes
  de empaquetarlo y comprobarlo con `signtool verify`.
- **Linux:** el AppImage no se firma.

## Versiones anteriores

La versión Python/Qt (PySide6, scikit-learn y TensorFlow, empaquetada con
PyInstaller), con `pyinstaller.spec`, `build-executables.yml`, la versión Streamlit
y su despliegue web (Docker, Swarm, nginx), se retiró del árbol tras la v1.0.0
(2026-10-21). Está en el tag
[`v1.0.0`](https://github.com/javipalanca/mltutor/tree/v1.0.0) y en la rama
[`codex/desktop-migration-study`](https://github.com/javipalanca/mltutor/tree/codex/desktop-migration-study),
que es donde se generan los fixtures y las capturas de referencia.
