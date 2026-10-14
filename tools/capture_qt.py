"""Capture reference screenshots of the Qt app, per migration phase.

Reuses MainWindow like ``--smoke-test`` does, driving navigation through the
session state and dispatching control actions on the rendered node tree.

Usage:
    uv run python tools/capture_qt.py docs/migration/capturas/qt             # phase 1
    uv run python tools/capture_qt.py docs/migration/capturas/qt/fase2 --phase 2
    uv run python tools/capture_qt.py docs/migration/capturas/qt/fase3 --phase 3
    ... --quantize   # rewrite every PNG with a 256-colour palette (Pillow) before commit
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

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
PHASE = int(sys.argv[sys.argv.index("--phase") + 1]) if "--phase" in sys.argv else 1
if "--phase" in sys.argv:
    ARGS.remove(str(PHASE))
QUANTIZE = "--quantize" in sys.argv
OUT = Path(ARGS[0])
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


def click_key(key):
    return lambda: window.action(find("button", key=key))


def set_state(**values):
    def run():
        state.update(values)
        window.evaluate()
    return run


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

IRIS = "🌸 Iris - Clasificación de flores"
CANCER = "🔬 Cáncer - Diagnóstico binario"
TITANIC = "🚢 Titanic - Supervivencia"
DIABETES = "🩺 Diabetes - Progresión (regresión)"
LR = "📊 Regresión"


def lr(tab, **extra):
    return go(LR, active_tab_lr=tab, **extra)


# Phase 2: 📊 Regresión lesson (linear and logistic). See docs/migration/fase2-regresion.md §0.
STEPS_F2 = [
    (go(LR, active_tab_lr=2, model_trained_lr=False, selected_dataset=IRIS,
        unified_dataset_selector=IRIS, active_analysis=None), "00_evaluacion_sin_entrenar"),
    (lr(3), "00b_visualizacion_sin_entrenar"),
    (lr(1, model_type_lr="Logistic"), "01_entrenamiento_logistica_iris"),
    (click_key("train_lr_button"), "02_entrenado_logistica_iris"),
    (lr(2), "03_evaluacion_clasificacion_iris"),
    (expand_all, "04_evaluacion_clasificacion_iris_expandido"),
    (lr(3, viz_type="ROC"), "05_roc_multiclase_iris"),
    (expand_all, "06_roc_multiclase_iris_expandido"),
    (lambda: (window.action(find("checkbox", "Mostrar curvas por clase individual"), False)),
     None),
    (choose(["micro"], label="Métodos de promediado:", kind="multiselect"),
     "07_roc_micro_sin_clases_iris"),
    (lambda: (window.action(find("checkbox", "Mostrar curvas por clase individual"), True)),
     None),
    (choose(["macro", "micro"], label="Métodos de promediado:", kind="multiselect"), None),
    (lr(3, viz_type="Frontera"), "08_frontera_multiclase_iris"),
    (choose("petal width (cm)", key="viz_feature2"), None),
    (choose("petal length (cm)", key="viz_feature1"), "09_frontera_petalos_iris"),
    (choose("petal length (cm)", key="viz_feature2"), "10_frontera_misma_caracteristica"),
    (lr(3, viz_type="Probs", viz_feature1="sepal length (cm)",
        viz_feature2="sepal width (cm)"), "11_probabilidades_multiclase_iris"),
    (lr(4), "12_coeficientes_logistica_iris"),
    (expand_all, "13_coeficientes_logistica_iris_expandido"),
    (choose(1, label="Selecciona la clase para mostrar coeficientes:"),
     "14_coeficientes_clase_versicolor"),
    (lr(5), "15_prediccion_formulario_iris"),
    (click("Realizar predicción"), "16_prediccion_resultado_iris"),
    (lr(6), "17_exportar"),
    (click_key("generate_code_lr"), "18_exportar_codigo"),
    (click_key("download_pickle_lr"), "19_exportar_pickle"),
    (lr(1, selected_dataset=CANCER, unified_dataset_selector=CANCER), None),
    (click_key("train_lr_button"), "20_entrenado_cancer"),
    (lr(2), "21_evaluacion_binaria_cancer"),
    (lr(3, viz_type="ROC"), "22_roc_binaria_cancer"),
    (lr(3, viz_type="Frontera"), "23_frontera_binaria_cancer"),
    (lr(3, viz_type="Probs"), "24_probabilidades_umbrales_cancer"),
    (expand_all, "25_probabilidades_umbrales_cancer_expandido"),
    (lr(1, selected_dataset=TITANIC, unified_dataset_selector=TITANIC), None),
    (click_key("train_lr_button"), None),
    (lr(5), "26_prediccion_formulario_titanic"),
    (click("Realizar predicción"), "27_prediccion_resultado_titanic"),
    (lr(4), "28_coeficientes_logistica_titanic"),
    (lr(1, selected_dataset=DIABETES, unified_dataset_selector=DIABETES), None),
    (click_key("btn_linear"), "29_entrenamiento_lineal_diabetes"),
    (click_key("train_lr_button"), "30_entrenado_lineal_diabetes"),
    (lr(2), "31_evaluacion_regresion_diabetes"),
    (expand_all, "32_evaluacion_regresion_diabetes_expandido"),
    (lr(3, viz_type="Residuos"), "33_residuos_diabetes"),
    (expand_all, "34_residuos_diabetes_expandido"),
    (lr(3, viz_type="Superficie"), "35_superficie_diabetes"),
    (choose("s5", key="viz_feature2"), None),
    (choose("bmi", key="viz_feature1"), "36_superficie_bmi_s5_diabetes"),
    (lr(4), "37_coeficientes_lineal_diabetes"),
    (expand_all, "38_coeficientes_lineal_diabetes_expandido"),
    (lr(5), "39_prediccion_formulario_diabetes"),
    (click("Realizar predicción"), "40_prediccion_resultado_diabetes"),
    (lr(1, selected_dataset=IRIS, unified_dataset_selector=IRIS), None),
    (click_key("btn_linear"), "41_aviso_lineal_en_clasificacion_iris"),
    (lr(1, selected_dataset=DIABETES, unified_dataset_selector=DIABETES), None),
    (click_key("btn_logistic"), "42_aviso_logistica_en_regresion_diabetes"),
    (click_key("train_lr_button"), "43_logistica_entrenada_en_diabetes"),
    (lr(3, viz_type="ROC"), "44_error_roc_logistica_diabetes"),
]

TREES = "🌲 Árboles de Decisión"
WINE = "🍷 Vino - Clasificación de vinos"
CALIFORNIA = "🏠 Viviendas California - Precios"


def dt(tab, **extra):
    return go(TREES, active_tab=tab, **extra)


def slide(label, value):
    return lambda: window.action(find("slider", label), value)


# Phase 3: 🌲 Árboles de Decisión. See docs/migration/fase3-arboles.md §0.
STEPS_F3 = [
    (go(TREES, active_tab=2, is_trained=False, selected_dataset=IRIS,
        unified_dataset_selector=IRIS, active_analysis=None), "00_evaluacion_sin_entrenar"),
    (dt(3), "00b_visualizacion_sin_entrenar"),
    (dt(1), "01_entrenamiento_iris"),
    (click("Entrenar Modelo"), "02_entrenado_iris"),
    (dt(2), "03_evaluacion_clasificacion_iris"),
    (expand_all, "04_evaluacion_clasificacion_iris_expandido"),
    (dt(3, viz_type="Árbol"), "05_arbol_iris"),
    (slide("Ancho de figura:", 20), None),
    (slide("Alto de figura:", 15), "06_arbol_iris_20x15"),
    (expand_all, "07_arbol_iris_codigo"),
    (dt(3, viz_type="Texto"), "08_texto_iris"),
    (dt(3, viz_type="Frontera"), "09_frontera_iris"),
    (choose("petal width (cm)", key="viz_feature2"), None),
    (choose("petal length (cm)", key="viz_feature1"), "10_frontera_petalos_iris"),
    (dt(3, viz_type="ROC", viz_feature1="sepal length (cm)", viz_feature2="sepal width (cm)"),
     "11_roc_iris"),
    (dt(4), "12_caracteristicas_iris"),
    (expand_all, "13_caracteristicas_iris_expandido"),
    (dt(5), "14_prediccion_formulario_iris"),
    (click("Realizar predicción"), "15_prediccion_camino_iris"),
    (expand_all, "16_prediccion_camino_iris_codigo"),
    (dt(6), "17_exportar_iris"),
    (expand_all, "18_exportar_iris_expandido"),
    (click("📦 Exportar como Pickle"), "19_exportar_pickle"),
    (click("📝 Exportar Reglas como Texto"), "20_exportar_reglas"),
    (click("🐍 Exportar como Código Python"), "21_exportar_codigo_python"),
    (dt(1, selected_dataset=CANCER, unified_dataset_selector=CANCER), None),
    (choose("entropy", label="Criterio de División:"), None),
    (slide("Profundidad Máxima:", 10), "22_entrenamiento_cancer_entropy_d10"),
    (click("Entrenar Modelo"), None),
    (dt(3, viz_type="Árbol"), "23_arbol_cancer_d10"),
    (dt(3, viz_type="ROC"), "24_roc_binaria_cancer"),
    (dt(1, selected_dataset=TITANIC, unified_dataset_selector=TITANIC), None),
    (choose("gini", label="Criterio de División:"), None),
    (slide("Profundidad Máxima:", 4), None),
    (click("Entrenar Modelo"), None),
    (dt(5), "25_prediccion_formulario_titanic"),
    (click("Realizar predicción"), "26_prediccion_camino_titanic"),
    (dt(1, selected_dataset=DIABETES, unified_dataset_selector=DIABETES), "27_aviso_clasificacion_en_diabetes"),
    (click_key("btn_regression"), "28_entrenamiento_regresion_diabetes"),
    (slide("Profundidad Máxima:", 3), None),
    (click("Entrenar Modelo"), "29_entrenado_regresion_diabetes"),
    (dt(2), "30_evaluacion_regresion_diabetes"),
    (dt(3, viz_type="Árbol"), "31_arbol_regresion_diabetes"),
    (dt(3, viz_type="Texto"), "32_texto_regresion_diabetes"),
    (dt(3, viz_type="Residuos"), "33_residuos_diabetes"),
    (dt(3, viz_type="Superficie"), "34_superficie_diabetes"),
    (choose("s5", key="viz_feature2"), None),
    (choose("bmi", key="viz_feature1"), "35_superficie_bmi_s5_diabetes"),
    (dt(4), "36_caracteristicas_regresion_diabetes"),
    (expand_all, "37_caracteristicas_regresion_diabetes_expandido"),
    (dt(5, viz_feature1="age", viz_feature2="sex"), None),
    (click("Realizar predicción"), "38_prediccion_regresion_diabetes"),
    (dt(6), "39_exportar_regresion_diabetes"),
    (dt(1, selected_dataset=CALIFORNIA, unified_dataset_selector=CALIFORNIA), None),
    (slide("Profundidad Máxima:", 8), None),
    (slide("Muestras Mínimas para División:", 10), None),
    (click("Entrenar Modelo"), None),
    (dt(3, viz_type="Árbol"), "40_arbol_california_d8"),
    (dt(1), None),
    (click_key("btn_classification"), None),
    (slide("Profundidad Máxima:", 3), None),
    (slide("Muestras Mínimas para División:", 2), None),
    (click("Entrenar Modelo"), "41_error_clasificacion_california"),
    (dt(1, selected_dataset=DIABETES, unified_dataset_selector=DIABETES), None),
    (click("Entrenar Modelo"), "42_clasificacion_entrenada_en_diabetes"),
    (dt(2), "43_error_evaluacion_clasificacion_diabetes"),
    (dt(1, selected_dataset=IRIS, unified_dataset_selector=IRIS), None),
    (click_key("btn_regression"), "44_aviso_regresion_en_iris"),
]

if PHASE == 2:
    STEPS = STEPS_F2
elif PHASE == 3:
    STEPS = STEPS_F3

steps = iter(STEPS)
pending = [None]


# Phase 2 keeps the window capture (with the sidebar) only for a few screens.
WINDOW_F2 = {"02_entrenado_logistica_iris", "05_roc_multiclase_iris", "33_residuos_diabetes"}
WINDOW_F3 = {"02_entrenado_iris", "05_arbol_iris", "31_arbol_regresion_diabetes"}


def capture(name):
    if PHASE == 1 or name in WINDOW_F2 or name in WINDOW_F3:
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


def quantize_all():
    from PIL import Image

    for png in sorted(OUT.glob("*.png")):
        img = Image.open(png).convert("RGB").quantize(colors=256)
        img.save(png, optimize=True)


def advance():
    step = next(steps, None)
    if step is None:
        if QUANTIZE:
            quantize_all()
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
