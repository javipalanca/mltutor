# Generar ejecutables de MLTutor

MLTutor se empaqueta con [PyInstaller](https://pyinstaller.org) en un
ejecutable autocontenido (modo *onedir*) que incluye Python, Streamlit,
TensorFlow y todas las dependencias. Los estudiantes **solo tienen que hacer
doble clic** — no necesitan instalar Python ni ningún otro programa.

## Artefactos por plataforma

| Plataforma | Artefacto nativo | Archivo crudo |
|---|---|---|
| macOS arm64 | `MLTutor.dmg` | `mltutor-macos-arm64.tar.gz` |
| Windows x86_64 | `mltutor-windows-x86_64-installer.exe` | `mltutor-windows-x86_64.zip` |
| Linux x86_64 | `mltutor-linux-x86_64.AppImage` | `mltutor-linux-x86_64.tar.gz` |

El artefacto **nativo** es el instalador recomendado para estudiantes. El
archivo crudo está disponible para usuarios técnicos o para despliegues sin
instalación.

## Cómo funciona

### Núcleo (PyInstaller)

- [launcher_rich.py](launcher_rich.py) es el punto de entrada: muestra un
  panel en la terminal, arranca Streamlit y abre el navegador.
  En modo congelado el launcher se relanza a sí mismo con `--server-mode`
  y ejecuta Streamlit *in-process*.
- [pyinstaller.spec](pyinstaller.spec) define el empaquetado. El paquete
  `mltutor/` se incluye como **datos** (Streamlit necesita el `.py` real
  para ejecutarlo), por lo que todas las librerías que usa la app se
  fuerzan con `collect_all`.
  En Windows, el `.exe` lleva el icono embebido (generado por
  `installers/create_icon.py`).

### Instaladores nativos

| Plataforma | Script | Herramienta |
|---|---|---|
| macOS | `installers/macos/build_dmg.sh` | `hdiutil` (incluido en macOS) |
| Windows | `installers/windows/mltutor.iss` | [Inno Setup 6](https://jrsoftware.org/isinfo.php) |
| Linux | `installers/linux/build_appimage.sh` | [`appimagetool`](https://github.com/AppImage/AppImageKit) (descarga automática) |

**macOS**: `build_dmg.sh` construye un `.app` bundle (con launcher
`osascript` que abre Terminal.app) y lo empaqueta en un DMG con enlace a
`/Applications` para arrastrar e instalar.

**Windows**: `mltutor.iss` define un instalador que copia el bundle en
`%ProgramFiles%\MLTutor`, crea accesos directos en el Menú Inicio (y
opcionalmente el Escritorio) e incluye desinstalador.

**Linux**: `build_appimage.sh` construye un AppImage de fichero único.
El usuario solo necesita `chmod +x` y ejecutarlo; no se instala nada en el
sistema.

## Build local

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev

# 1. Generar icono
uv run python installers/create_icon.py installers/

# 2. Empaquetar con PyInstaller
uv run pyinstaller pyinstaller.spec --noconfirm

# 3. Construir el instalador nativo según la plataforma:
bash installers/macos/build_dmg.sh            # macOS
bash installers/linux/build_appimage.sh       # Linux
iscc installers\windows\mltutor.iss           # Windows (Inno Setup instalado)
```

PyInstaller **no cross-compila**: cada ejecutable debe generarse en su
propio sistema operativo.

## Build automática (GitHub Actions)

El workflow [build-executables.yml](.github/workflows/build-executables.yml)
compila en paralelo para:

| Plataforma | Runner | Artefactos |
|---|---|---|
| Linux x86_64 | ubuntu-22.04 | `.tar.gz` + `.AppImage` |
| Windows x86_64 | windows-latest | `.zip` + `-installer.exe` |
| macOS arm64 (Apple Silicon) | macos-latest | `.tar.gz` + `.dmg` |

Se dispara:

- **Manualmente**: pestaña *Actions* → *Build executables* → *Run workflow*.
  Los artefactos quedan disponibles 14 días.
- **Con un tag** `v*` (p. ej. `git tag v1.0.0 && git push --tags`):
  además publica una **GitHub Release** con los seis ficheros adjuntos,
  lista para compartir el enlace de descarga con los estudiantes.

Incluye un *smoke test* en Linux/macOS que arranca el binario y comprueba
que el servidor responde.

Nota: macOS Intel no está soportado (no hay wheels de TensorFlow recientes
para esa plataforma).

## Instrucciones para estudiantes

1. Descargar el fichero de su sistema operativo desde la página de Releases.

2. **macOS**:
   - Abrir el `.dmg` y arrastrar `MLTutor.app` a la carpeta `Aplicaciones`.
   - Doble clic en `MLTutor.app`.
   - La primera vez, macOS puede bloquear la app porque no está firmada:
     hacer **clic derecho → Abrir** y confirmar en el diálogo.
   - Se abrirá una ventana de Terminal con el lanzador y el navegador con
     MLTutor.

3. **Windows**:
   - Ejecutar `mltutor-windows-x86_64-installer.exe`.
   - Si SmartScreen avisa, pulsar *Más información* → *Ejecutar de todas
     formas*.
   - El instalador copia los ficheros y crea un acceso directo.
   - Doble clic en el acceso directo de MLTutor para arrancar.

4. **Linux**:
   - Dar permisos de ejecución: `chmod +x mltutor-linux-x86_64.AppImage`
   - Doble clic (si el gestor de ficheros lo permite) o ejecutar:
     `./mltutor-linux-x86_64.AppImage`
   - Requiere FUSE: `sudo apt-get install libfuse2` (Ubuntu/Debian).

El ejecutable ocupa en torno a 1–2 GB descomprimido (TensorFlow incluido).

