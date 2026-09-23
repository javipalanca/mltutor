# 🎓 MLTutor

Aplicación educativa de escritorio para aprender Machine Learning mediante
experimentos, visualizaciones y explicaciones en español. Implementada con
Python y **Qt (PySide6)**; no utiliza un servidor Streamlit.

## Descargar y abrir

Las distribuciones portables se publican en las Releases del repositorio:

| Sistema | Archivo | Uso |
|---|---|---|
| Windows x86_64 | `mltutor-windows-x86_64.exe` | Descargar y abrir |
| macOS Apple Silicon | `mltutor-macos-arm64.zip` | Descomprimir y abrir `MLTutor.app` |
| Linux x86_64 | `mltutor-linux-x86_64` | Dar permiso de ejecución y abrir |

**Sin instalador y sin instalar Python, Qt ni dependencias.** Se ejecuta con una
cuenta de usuario normal. Las políticas de ejecución y confianza de cada equipo
siguen aplicándose. Los nuevos binarios Qt deben generarse y validarse antes de
publicarlos; las releases anteriores corresponden a la versión Streamlit.

## Contenido

- 🌲 **Árboles de decisión:** clasificación y regresión, reglas, estructura,
  importancia de características, fronteras de decisión y predicciones.
- 📊 **Regresión:** lineal y logística, coeficientes, residuos, probabilidades,
  evaluación y generación de código.
- 🔍 **KNN:** clasificación y regresión, vecinos, distancias, visualizaciones
  interactivas, ajuste de K y predicciones.
- 🧠 **Redes neuronales:** arquitectura, activaciones, entrenamiento,
  regularización, historial, pesos, superficies y exportación TensorFlow.
- 📁 **Datos:** datasets incluidos, importación de CSV, exploración, correlación
  y matriz de dispersión.
- 💾 **Exportación:** código Python, imágenes, reglas, modelos y formatos de
  intercambio ofrecidos por cada lección, mediante diálogos de guardado nativos.

Se conserva el orden de las pestañas, las explicaciones, los colores y las
visualizaciones de la aplicación original. Los controles se presentan como
widgets Qt y las gráficas HTML se ejecutan localmente dentro de la aplicación.
Los cálculos se ejecutan en segundo plano, con progreso y cancelación cooperativa.

## Desarrollo

```sh
uv sync --locked --extra dev
uv run python launcher_qt.py
```

```sh
uv run pytest -q
uv run python launcher_qt.py --smoke-test
uv run pyinstaller pyinstaller.spec --noconfirm
```

Consultar [BUILD_EXECUTABLES.md](BUILD_EXECUTABLES.md) para detalles de
empaquetado, límites de compatibilidad, pruebas y arquitectura.

La versión original y su despliegue web están conservados en `main` durante la
migración, que se desarrolla en `codex/desktop-migration-study`. Los antiguos
archivos Docker y scripts de despliegue web son referencias de esa versión y no
son el método de ejecución de la aplicación Qt.
