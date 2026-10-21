"""Exercise the same page actions as the GUI, with real models and datasets."""

import pickle
import sys
import pytest
from mltutor.desktop import ui


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def render(session, allow_diagnostics=False):
    from mltutor.app import main

    root, side = session.render(main)
    import matplotlib.pyplot as plt

    plt.close("all")
    errors = [
        n.props["text"]
        for n in walk(root)
        if n.kind == "notice" and n.props["severity"] == "error"
    ]
    if allow_diagnostics:
        # Undertrained small networks intentionally produce educational alerts;
        # those are content, whereas computation/rendering errors must fail.
        expected = (
            "Accuracy bajo",
            "Crítico:",
            "Modelo no convergió",
            "CRÍTICO:",
            "Complejidad Alta",
            "Patrón de sobreajuste detectado",
        )
        errors = [
            message
            for message in errors
            if not any(term in message for term in expected)
        ]
    assert not errors, "\n".join(errors)
    return root


def click(session, root, label):
    button = next(
        n for n in walk(root) if n.kind == "button" and n.props["label"] == label
    )
    session.dispatch(button)
    return render(session)


@pytest.mark.parametrize(
    "navigation,tab_key,train,model_key,dataset,extra",
    [
        (
            "🌲 Árboles de Decisión",
            "active_tab",
            "Entrenar Modelo",
            "tree_model",
            "🌸 Iris - Clasificación de flores",
            {},
        ),
        (
            "🌲 Árboles de Decisión",
            "active_tab",
            "Entrenar Modelo",
            "tree_model",
            "🩺 Diabetes - Progresión (regresión)",
            {"tree_type": "Regresión"},
        ),
        (
            "📊 Regresión",
            "active_tab_lr",
            "🚀 Entrenar Modelo",
            "model_lr",
            "🌸 Iris - Clasificación de flores",
            {"model_type_lr": "Logistic"},
        ),
        (
            "📊 Regresión",
            "active_tab_lr",
            "🚀 Entrenar Modelo",
            "model_lr",
            "🩺 Diabetes - Progresión (regresión)",
            {"model_type_lr": "Linear"},
        ),
        (
            "🔍 K-Nearest Neighbors",
            "active_tab_knn",
            "🚀 Entrenar Modelo KNN",
            "knn_model",
            "🌸 Iris - Clasificación de flores",
            {},
        ),
        (
            "🔍 K-Nearest Neighbors",
            "active_tab_knn",
            "🚀 Entrenar Modelo KNN",
            "knn_model",
            "🩺 Diabetes - Progresión (regresión)",
            {},
        ),
    ],
)
def test_full_classical_lesson(navigation, tab_key, train, model_key, dataset, extra):
    session = ui.Session()
    session.state.update(navigation=navigation, selected_dataset=dataset, **extra)
    render(session)
    session.state[tab_key] = 1
    root = render(session)
    click(session, root, train)
    assert session.state[model_key] is not None
    restored = pickle.loads(pickle.dumps(session.state[model_key]))
    assert type(restored) is type(session.state[model_key])
    for tab in range(2, 7):
        session.state[tab_key] = tab
        root = render(session)
        assert root.children
        if tab == 3:
            # Every visualization button, including JS KNN and Plotly surfaces.
            labels = [
                n.props["label"]
                for n in walk(root)
                if n.kind == "button" and str(n.props.get("key", "")).startswith("viz_")
            ]
            for label in labels:
                root = click(session, root, label)
        if tab == 5:
            click(session, root, "Realizar predicción")
    if model_key in ("model_lr", "knn_model"):
        root = click(session, root, "📥 Descargar Modelo (Pickle)")
        download = next(n for n in walk(root) if n.kind == "download")
        assert type(pickle.loads(download.props["data"])) is type(
            session.state[model_key]
        )
    assert "streamlit" not in sys.modules


def test_neural_network_lesson():
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    session = ui.Session()
    session.state.navigation = "🧠 Redes Neuronales"
    render(session)
    session.state.active_tab_nn = 1
    session.state.layer_0 = 4
    session.state.layer_1 = 2
    render(session)
    session.state.active_tab_nn = 2
    root = render(session)
    epochs = next(
        n for n in walk(root) if n.kind == "slider" and n.props["label"] == "Épocas"
    )
    session.dispatch(epochs, 10)
    root = render(session)
    root = click(session, root, "🧠 Entrenar Red Neuronal")
    assert session.state.nn_model is not None
    assert len(session.state.nn_history.history["loss"]) > 0
    for tab in range(3, 7):
        session.state.active_tab_nn = tab
        root = render(session, allow_diagnostics=True)
        if tab == 4:
            for name in ("Pesos", "Superficie", "Activaciones"):
                session.state.viz_type = name
                render(session, allow_diagnostics=True)
        if tab == 5:
            click(session, root, "Realizar predicción")
    session.state.viz_type = "Metadatos"
    root = render(session)
    assert any(
        n.kind == "download" and n.props["file_name"].endswith(".json")
        for n in walk(root)
    )
    session.state.viz_type = "Tensorflow"
    session.state.nn_export_format = "HDF5 (.h5)"
    root = click(session, render(session), "💾 Exportar Modelo TensorFlow")
    download = next(n for n in walk(root) if n.kind == "download")
    assert download.props["data"].startswith(b"\x89HDF")


def test_custom_csv_enters_shared_dataset_selector():
    session = ui.Session()
    session.state.navigation = "📁 Cargar CSV Personalizado"
    root = render(session)
    uploader = next(n for n in walk(root) if n.kind == "file")
    data = b"x,y,target\n" + b"".join(
        f"{i},{i * 2},{i % 2}\n".encode() for i in range(30)
    )
    session.dispatch(uploader, ("classroom.csv", data))
    render(session)
    assert "📄 classroom.csv" in session.state.csv_datasets
    session.state.navigation = "🌲 Árboles de Decisión"
    root = render(session)
    selector = next(
        n
        for n in walk(root)
        if n.kind == "select" and n.props.get("key") == "unified_dataset_selector"
    )
    assert selector.props["value"] == "📄 classroom.csv"
