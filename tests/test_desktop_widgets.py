"""Check actual Qt controls, action delivery and offline interactive plots."""

import time
import numpy  # noqa: F401 -- initialise native BLAS on the main thread
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox
from mltutor.desktop import ui
from mltutor.desktop.window import MainWindow


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


@pytest.fixture
def window(application):
    win = MainWindow(autostart=False)
    win.show()
    yield win
    if win.worker is not None:
        win.session.cancel.set()
        while win.worker is not None:
            application.processEvents()
            time.sleep(0.01)
    win.close()
    win.deleteLater()
    application.processEvents()


def wait_ready(win, app, timeout=60):
    deadline = time.monotonic() + timeout
    while win.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert win.worker is None, "El trabajador no terminó"
    assert win.failed_message is None, win.failed_message


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def test_native_navigation_training_and_dataset_change(window, application):
    window.evaluate()
    wait_ready(window, application)
    QTest.mouseClick(window.controls["key:nav_trees"], Qt.MouseButton.LeftButton)
    wait_ready(window, application)
    assert window.session.state.navigation == "🌲 Árboles de Decisión"
    QTest.mouseClick(window.controls["key:tab_1"], Qt.MouseButton.LeftButton)
    wait_ready(window, application)
    assert window.session.state.active_tab == 1
    train = next(
        n
        for n in walk(window.root_node)
        if n.kind == "button" and n.props["label"] == "Entrenar Modelo"
    )
    QTest.mouseClick(window.controls[train.props["id"]], Qt.MouseButton.LeftButton)
    wait_ready(window, application)
    assert window.session.state.tree_model is not None
    selector = window.controls["key:unified_dataset_selector"]
    assert isinstance(selector, QComboBox)
    selector.setCurrentIndex(1)
    wait_ready(window, application)
    assert "Vino" in window.session.state.selected_dataset


def test_file_dialog_download_writes_exact_bytes(window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    path = tmp_path / "model.pkl"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
    payload = b"\x80\x04MLTutor\x00\xff"
    window.save_data(payload, "model.pkl")
    assert path.read_bytes() == payload


def test_plotly_is_local_and_javascript_renders(window, application):
    import plotly.graph_objects as go

    session = ui.Session()
    root, _ = session.render(
        lambda: ui.plotly_chart(go.Figure(go.Scatter(x=[1, 2], y=[3, 4])))
    )
    widget = window.build(root)
    window.content.setWidget(widget)
    view = window.webviews[0]
    ready = []
    view.loadFinished.connect(ready.append)
    deadline = time.monotonic() + 30
    while not ready and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.02)
    assert ready == [True]
    results = []
    view.page().runJavaScript(
        "typeof Plotly !== 'undefined' && document.querySelectorAll('.plotly-graph-div .main-svg').length > 0",
        results.append,
    )
    while not results and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.02)
    assert results == [True]
    assert view.url().isLocalFile()
