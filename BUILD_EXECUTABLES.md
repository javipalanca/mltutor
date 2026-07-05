# Generar ejecutables de MLTutor

MLTutor se empaqueta con [PyInstaller](https://pyinstaller.org) en un
ejecutable autocontenido (modo *onedir*) que incluye Python, Streamlit,
TensorFlow y todas las dependencias. Los estudiantes solo tienen que
descargar, descomprimir y hacer doble clic — no necesitan instalar nada.

Desde la versión de escritorio, MLTutor se abre como una **app con ventana
propia** (no en el navegador): la UI de Streamlit se empotra en una ventana
nativa mediante [pywebview](https://pywebview.flowrl.com/):

| Plataforma | Motor de la ventana |
|---|---|
| macOS | WKWebView (WebKit del sistema, vía pyobjc) |
| Windows | WebView2 (Edge/Chromium, preinstalado en Windows 10/11) |
| Linux | Qt WebEngine (PySide6, empaquetado dentro del ejecutable) |

Si no hay backend gráfico disponible (p. ej. WebView2 ausente o un Linux
sin display), el launcher cae automáticamente al modo clásico: arranca el
servidor y abre el navegador. El flag `--browser` fuerza ese modo.

## Cómo funciona

- [launcher_rich.py](launcher_rich.py) es el punto de entrada: arranca el
  servidor Streamlit y abre la ventana nativa (o el navegador como
  fallback). En modo congelado el launcher se relanza a sí mismo con
  `--server-mode` y ejecuta Streamlit *in-process* (en un ejecutable
  PyInstaller no existe un intérprete Python externo).
- En Windows el ejecutable es *windowed* (sin consola); la salida se
  registra en `~/.mltutor/mltutor.log`. En macOS se genera un bundle
  `MLTutor.app` con doble clic nativo.
- Para pruebas automatizadas, `MLTUTOR_WINDOW_TIMEOUT=N` cierra la ventana
  a los N segundos.
- [pyinstaller.spec](pyinstaller.spec) define el empaquetado. El paquete
  `mltutor/` se incluye como **datos** (Streamlit necesita el `.py` real
  para ejecutarlo), por lo que todas las librerías que usa la app se
  fuerzan con `collect_all`.
- Si añades una dependencia nueva a la app, añádela también a la lista de
  `collect_all` del spec.

## Build local

Requiere Python 3.12 (fijado en `.python-version`; TensorFlow 2.16 no
soporta 3.13+) y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pyinstaller pyinstaller.spec --noconfirm
```

El resultado queda en `dist/mltutor/` (el binario es `dist/mltutor/mltutor`,
o `mltutor.exe` en Windows). El directorio completo es lo que se distribuye.

PyInstaller **no cross-compila**: cada ejecutable debe generarse en su
propio sistema operativo.

## Build automática (GitHub Actions)

El workflow [build-executables.yml](.github/workflows/build-executables.yml)
compila en paralelo para:

| Plataforma | Runner | Artefacto | Contenido |
|---|---|---|---|
| Linux x86_64 | ubuntu-22.04 | `mltutor-linux-x86_64.tar.gz` | carpeta `mltutor/` |
| Windows x86_64 | windows-latest | `mltutor-windows-x86_64.zip` | carpeta `mltutor/` |
| macOS arm64 (Apple Silicon) | macos-latest | `mltutor-macos-arm64.tar.gz` | bundle `MLTutor.app` |

Se dispara:

- **Manualmente**: pestaña *Actions* → *Build executables* → *Run workflow*.
  Los ejecutables quedan como artefactos del workflow (14 días).
- **Con un tag** `v*` (p. ej. `git tag v1.0.0 && git push --tags`):
  además publica una **GitHub Release** con los tres ficheros adjuntos,
  lista para compartir el enlace de descarga con los estudiantes.

Incluye un *smoke test* en Linux/macOS que arranca el binario y comprueba
que el servidor responde.

Nota: macOS Intel no está soportado (no hay wheels de TensorFlow recientes
para esa plataforma).

## Instrucciones para estudiantes

1. Descargar el fichero de su sistema operativo desde la página de Releases.
2. Descomprimirlo.
3. Ejecutar:
   - **Windows**: doble clic en `mltutor.exe` (dentro de la carpeta
     `mltutor`). Si SmartScreen avisa, pulsar *Más información* →
     *Ejecutar de todas formas*.
   - **macOS**: la primera vez, clic derecho sobre `MLTutor.app` → *Abrir*
     (el bundle no está firmado y Gatekeeper lo bloquea con doble clic
     normal). Alternativa por terminal: `xattr -cr MLTutor.app`.
   - **Linux**: `./mltutor/mltutor` desde una terminal.
4. Se abrirá MLTutor en su propia ventana. Al cerrar la ventana, la app
   se detiene sola. Con `--browser` se usa el modo clásico (navegador).

El ejecutable ocupa en torno a 1–2 GB descomprimido (TensorFlow incluido).
