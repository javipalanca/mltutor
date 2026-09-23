# MLTutor portable con Qt

La aplicación de escritorio utiliza PySide6 (Qt Widgets). Los controles son
nativos y los gráficos interactivos se muestran en Qt WebEngine con JavaScript
local. No hay servidor HTTP, navegador externo, Streamlit ni instalador.

## Para estudiantes

- **Windows x86_64:** descargar `mltutor-windows-x86_64.exe` y abrirlo.
- **macOS Apple Silicon:** descargar `mltutor-macos-arm64.zip`, descomprimir y
  abrir `MLTutor.app` desde esa misma carpeta. No hace falta copiar a Aplicaciones.
- **Linux x86_64:** descargar `mltutor-linux-x86_64`, marcarlo como ejecutable en
  las propiedades del archivo y abrirlo. La compatibilidad se valida sobre
  Ubuntu 22.04 y sistemas compatibles; no es un binario universal para toda
  distribución de Linux.

No se necesita instalar Python, Qt, TensorFlow ni usar permisos de administrador.
Cada descarga contiene su propio entorno. Windows y Linux extraen componentes a
una carpeta temporal del usuario al abrirse: el primer arranque puede tardar y
necesita espacio libre. macOS utiliza un bundle de aplicación autocontenido.

Los equipos deben permitir ejecutar aplicaciones descargadas. El empaquetado
portable no evita políticas institucionales que bloqueen ejecutables, ni las
comprobaciones de confianza del sistema. La firma Developer ID y notarización de
macOS siguen pendientes de credenciales; no se promete apertura sin avisos hasta
completar esa fase. TensorFlow sigue siendo la parte más pesada de la descarga.

## Desarrollo

```sh
uv sync --locked --extra dev
uv run python launcher_qt.py
uv run pytest -q
uv run python launcher_qt.py --smoke-test
uv run python launcher_qt.py --verify-models
```

También se puede iniciar con `uv run mltutor` o `python -m mltutor.desktop`.
Python 3.11 o 3.12. La versión original Streamlit permanece en `main`; la
migración se desarrolla en `codex/desktop-migration-study`.

## Empaquetado

```sh
uv run pyinstaller pyinstaller.spec --noconfirm
```

Salida: `dist/MLTutor.exe` en Windows, `dist/MLTutor` en Linux y
`dist/MLTutor.app` en macOS. Construir por separado en cada plataforma.
Para trabajar en paralelo con builds anteriores se puede usar
`--distpath dist/qt --workpath build/qt`.

El workflow `.github/workflows/build-executables.yml` ejecuta pruebas de las
lecciones, construye y abre el ejecutable portable en las tres plataformas. Se
puede lanzar manualmente, desde una PR o mediante un tag `v*`. Solo los tags
publican una release. Los tags con sufijo (por ejemplo `v0.3.0-rc.1`) publican
una versión preliminar sin reemplazar la estable. Se exige la presencia de las
tres descargas y se adjunta `SHA256SUMS.txt`.

Para publicar desde la rama de migración: `./release.sh 0.3.0-rc.1`.
El script necesita GitHub CLI autenticado y mantiene `main` intacta. No utiliza Inno Setup ni genera instaladores.

## Arquitectura

- `mltutor/desktop/ui.py`: descripción de controles y estado independiente de Qt.
- `mltutor/desktop/window.py`: controles Qt, formularios, tablas, gráficos y
  diálogos de archivos. Los cálculos se ejecutan en un QThread y la ventana solo
  recibe resultados mediante señales. La cancelación es cooperativa.
- `mltutor/desktop/__main__.py`: arranque y prueba de apertura.
- Las lecciones de `apps`, `dataset`, `viz` y `algorithms` conservan sus textos,
  cálculos y orden de navegación, utilizando ahora la capa de presentación local.
  El alias `st` en esos módulos es únicamente un nombre histórico; importa
  `mltutor.desktop.ui`, nunca Streamlit.
- Matplotlib se convierte en imágenes en el trabajador. Plotly y las animaciones
  HTML usan archivos locales, incluyendo Plotly JS; no necesitan CDN.
- Los datasets ya incluidos funcionan sin conexión. Los datasets que requieren
  descarga mantienen esa necesidad hasta que se distribuyan también sus datos.

Las pruebas cubren navegación, estado, CSV, entrenamiento real de los cuatro
algoritmos, evaluación y renderizado de las pestañas. La prueba del binario
comprueba la apertura de las secciones sin depender de Python instalado fuera
del paquete. `--verify-models` comprueba dentro del ejecutable entrenamiento real,
predicción y exportaciones Pickle, ONNX, Keras, SavedModel y TensorFlow Lite,
incluyendo equivalencia numérica de predicciones. Es recomendable
validar también las descargas en equipos limpios
con cuentas sin privilegios antes de publicarlas a estudiantes.
