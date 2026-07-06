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
| Windows x86_64 | windows-latest | `mltutor-windows-x86_64-setup.exe` | instalador (Inno Setup) |
| Windows x86_64 | windows-latest | `mltutor-windows-x86_64-portable.zip` | carpeta `mltutor/` (portable) |
| macOS arm64 (Apple Silicon) | macos-latest | `mltutor-macos-arm64.tar.gz` | bundle `MLTutor.app` |

En Windows se distribuye un **instalador** además de un zip portable. El
instalador **no requiere permisos de administrador**: instala por-usuario
en `%LOCALAPPDATA%\Programs\MLTutor` (sin UAC; un admin puede forzar la
instalación global con `/ALLUSERS`), crea el acceso directo del Menú
Inicio con el icono y añade desinstalador (el usuario nunca ve el
directorio `_internal` de PyInstaller). El script del instalador es
[installers/windows/mltutor.iss](installers/windows/mltutor.iss).
El **zip portable** no instala nada: descomprimir y ejecutar
`mltutor\mltutor.exe` — útil en equipos con políticas restrictivas
(laboratorios docentes).

Además del test de servidor, el CI comprueba en Windows y Linux (con
xvfb) que la **ventana nativa** abre de verdad (`MLTUTOR_REQUIRE_WINDOW=1`
desactiva el fallback a navegador y hace fallar el build si no hay
ventana).

Se dispara:

- **Manualmente**: pestaña *Actions* → *Build executables* → *Run workflow*.
  Los ejecutables quedan como artefactos del workflow (14 días).
- **Con un tag** `v*` (p. ej. `git tag v1.0.0 && git push --tags`):
  además publica una **GitHub Release** con los tres ficheros adjuntos,
  lista para compartir el enlace de descarga con los estudiantes.

## Publicar una release (script)

Todo el ciclo está automatizado en [release.sh](release.sh):

```bash
./release.sh 0.3.0
```

El script actualiza la versión en `pyproject.toml` (única fuente de
versión: el bundle de macOS y el instalador de Windows la leen de ahí),
hace commit, crea y pushea el tag `v0.3.0`, espera a que el CI compile
las tres plataformas y verifica que la release queda publicada con todos
los artefactos. Requiere `gh` (GitHub CLI) autenticado.

Incluye un *smoke test* en Linux/macOS que arranca el binario y comprueba
que el servidor responde.

Nota: macOS Intel no está soportado (no hay wheels de TensorFlow recientes
para esa plataforma).

### Firma y notarización en macOS (pendiente)

El bundle va firmado *ad-hoc* (lo hace PyInstaller), pero **no notarizado**:
macOS moderno lo bloquea en la primera apertura y el estudiante tiene que
autorizarlo en Ajustes (ver instrucciones más abajo). La solución
definitiva es firmar con un certificado *Developer ID Application* y
notarizar con `notarytool`, lo que requiere una cuenta de pago del Apple
Developer Program. Cuando se disponga de ella, se puede automatizar en el
workflow con los secrets del certificado y `xcrun notarytool submit`.

## Instrucciones para estudiantes

1. Descargar el fichero de su sistema operativo desde la página de Releases.
2. Instalar / ejecutar:
   - **Windows**: doble clic en `mltutor-windows-x86_64-setup.exe` y seguir
     el asistente; después, abrir *MLTutor* desde el Menú Inicio. Si
     SmartScreen avisa, pulsar *Más información* → *Ejecutar de todas
     formas*.
   - **macOS**: descomprimir e intentar abrir `MLTutor.app` (fallará con
     *"no se puede abrir"* o error -47: el bundle no está notarizado y
     Gatekeeper lo bloquea; desde macOS Sequoia el clic derecho → *Abrir*
     ya no sirve). Después, ir a **Ajustes del Sistema → Privacidad y
     seguridad**, bajar hasta el aviso sobre MLTutor y pulsar **Abrir de
     todos modos**. Alternativa por terminal (sin pasar por Ajustes):
     `xattr -dr com.apple.quarantine MLTutor.app` y abrir normalmente.
   - **Linux**: descomprimir y ejecutar `./mltutor/mltutor` desde una
     terminal.
3. Se abrirá MLTutor en su propia ventana. Al cerrar la ventana, la app
   se detiene sola. Con `--browser` se usa el modo clásico (navegador).

El ejecutable ocupa en torno a 1–2 GB descomprimido (TensorFlow incluido).
