"""Capture reference screenshots of the Qt app for phase 1 (shell + data).

Reuses MainWindow like ``--smoke-test`` does, driving navigation through the
session state and dispatching control actions on the rendered node tree.
Usage: uv run python tools/capture_qt.py docs/migration/capturas/qt
(QT_QPA_PLATFORM must NOT be offscreen: WebEngine and fonts need the real platform.)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy  # noqa: E402,F401  (main thread BLAS init, see MEMORY.md)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mltutor.desktop.window import MainWindow  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
CSV = REPO / "mltutor" / "dataset" / "data" / "sample_datasets" / "tips.csv"

app = QApplication(sys.argv[:1])
app.setStyle("Fusion")
window = MainWindow(autostart=False)
window.show()
state = window.session.state


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def find(kind, label=None, key=None):
    for node in walk(window.root_node):
        if node.kind != kind:
            continue
        if label is not None and node.props.get("label") != label:
            continue
        if key is not None and node.props.get("key") != key:
            continue
        return node
    raise LookupError(f"{kind} {label or key} not found")


def go(page, **extra):
    def run():
        state["navigation"] = page
        state.update(extra)
        window.evaluate()
    return run


def click(label):
    return lambda: window.action(find("button", label))


def choose(value, label=None, key=None, kind="select"):
    return lambda: window.action(find(kind, label, key), value)


def expand_all():
    for node in walk(window.root_node):
        if node.kind == "expander":
            window.view_state[node.props["id"]] = True
    window.evaluate()


def upload():
    window.action(find("file"), (CSV.name, CSV.read_bytes()))


# (action, screenshot name or None). A name means: capture once the page is ready.
STEPS = [
    (go("🏠 Inicio"), "01_inicio"),
    (go("📊 Regresión", active_tab_lr=0), "02_datos_iris"),
    (expand_all, "03_datos_iris_expandido"),
    (click("🔗 Matriz de Correlación"), "04_datos_correlacion_iris"),
    (click("📈 Matriz de Dispersión (Pairplot)"), "05_datos_pairplot_opciones"),
    (click("🚀 Generar Matriz de Dispersión"), "06_datos_pairplot_kde_iris"),
    (choose("Histograma", label="Tipo de gráfico en la diagonal:", kind="radio"), None),
    (click("🚀 Generar Matriz de Dispersión"), "07_datos_pairplot_hist_iris"),
    (lambda: (state.__setitem__("active_analysis", None),
              choose("🩺 Diabetes - Progresión (regresión)", key="unified_dataset_selector")()),
     "08_datos_diabetes"),
    (expand_all, "09_datos_diabetes_expandido"),
    (click("📈 Matriz de Dispersión (Pairplot)"), None),
    (click("🚀 Generar Matriz de Dispersión"), "10_datos_pairplot_diabetes"),
    (lambda: (state.__setitem__("active_analysis", "correlation"),
              choose("🔢 Dígitos - Clasificación de dígitos", key="unified_dataset_selector")()),
     "11_datos_correlacion_digitos"),
    (lambda: (state.__setitem__("active_analysis", None),
              choose("🚢 Titanic - Supervivencia", key="unified_dataset_selector")()),
     "12_datos_titanic_desbalance"),
    (lambda: (state.__setitem__("active_analysis", None),
              choose("🌙 Moons - Clasificación sintética", key="unified_dataset_selector")()),
     "13_datos_moons"),
    (go("🧠 Redes Neuronales", active_tab_nn=0, active_analysis=None,
        selected_dataset="🌸 Iris - Clasificación de flores",
        unified_dataset_selector="🌸 Iris - Clasificación de flores"), "14_datos_redes_tips"),
    (go("📊 Regresión", active_tab_lr=2), "15_regresion_pestana_evaluacion_dataset_actual"),
    (go("📁 Cargar CSV Personalizado"), "16_csv_vacio"),
    (upload, "17_csv_cargado_clasificacion"),
    (expand_all, "18_csv_cargado_clasificacion_expandido"),
    (choose("tip", label="Columna objetivo:"), None),
    (expand_all, "19_csv_cargado_regresion_expandido"),
]

steps = iter(STEPS)
pending = [None]


def capture(name):
    window.grab().save(str(OUT / f"{name}_ventana.png"))
    widget = window.content.widget()
    if widget is not None:
        widget.grab().save(str(OUT / f"{name}.png"))
    print("captured", name, flush=True)


def ready():
    if window.failed_message:
        print("FAILED", pending[0], window.failed_message, file=sys.stderr, flush=True)
    name = pending[0]
    if name:
        pending[0] = None
        QTimer.singleShot(900, lambda: (capture(name), advance()))
        return
    QTimer.singleShot(50, advance)


def advance():
    step = next(steps, None)
    if step is None:
        app.exit(0)
        return
    action, name = step
    pending[0] = name
    try:
        action()
    except Exception as exc:  # keep going, report
        print("STEP ERROR", name, repr(exc), file=sys.stderr, flush=True)
        pending[0] = None
        QTimer.singleShot(50, advance)


window.page_ready.connect(ready)
QTimer.singleShot(0, advance)
QTimer.singleShot(600000, lambda: app.exit(2))
code = app.exec()
if window.worker is not None:
    window.session.cancel.set()
    window.worker.wait()
window.close()
sys.exit(code)
