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


def test_table_headers_values_and_resizing(window, application):
    import pandas as pd
    from mltutor.desktop.widgets import DataTable

    frame = pd.DataFrame(
        {
            name: [0.123456789, 2300.25]
            for name in [
                "carat",
                "cut",
                "color",
                "clarity",
                "depth",
                "table",
                "x",
                "y",
                "z",
                "target",
            ]
        }
    )
    panel = window.build(ui.Node("table", {"data": frame}))
    window.content.setWidget(panel)
    application.processEvents()
    table = panel.findChild(DataTable)
    header = table.horizontalHeader()
    for column in range(10):
        assert (
            table.columnWidth(column)
            >= header.fontMetrics().horizontalAdvance(frame.columns[column]) + 24
        )
    assert table.columnWidth(9) < table.columnWidth(0) * 1.5
    window.resize(1850, 920)
    application.processEvents()
    assert (
        abs(sum(table.columnWidth(c) for c in range(10)) - table.viewport().width())
        <= 10
    )
    model = table.model()
    assert model.data(model.index(0, 0), Qt.ItemDataRole.ToolTipRole) == "0.123456789"
    table.selectAll()
    from PySide6.QtGui import QKeySequence

    table.setFocus()
    QTest.keySequence(table, QKeySequence(QKeySequence.StandardKey.Copy))
    assert "0.123456789" in application.clipboard().text()
    assert application.clipboard().text().startswith("carat\tcut\tcolor")


def test_python_syntax_multiline_and_copy(window, application):
    from mltutor.desktop.widgets import CodeEditor
    from PySide6.QtWidgets import QPushButton

    source = 'def entrenar(x):\n    """Primera línea\n    segunda línea"""\n    # Comentario\n    return len(x) + 42\n'
    panel = window.build(ui.Node("code", {"text": source, "language": "python"}))
    window.content.setWidget(panel)
    application.processEvents()
    editor = panel.findChild(CodeEditor)
    document = editor.document()
    assert document.firstBlock().layout().formats()
    assert document.findBlockByNumber(2).layout().formats()  # multiline string
    assert document.findBlockByNumber(3).layout().formats()[0].format.fontItalic()
    colors = {
        f.format.foreground().color().name()
        for f in document.findBlockByNumber(4).layout().formats()
    }
    assert len(colors) >= 3  # keyword, builtin and numeric literal
    QTest.mouseClick(
        panel.findChild(QPushButton, "copyCode"), Qt.MouseButton.LeftButton
    )
    assert application.clipboard().text() == source


def test_html_fits_dynamic_content_without_inner_scroll(window, application):
    view = window.html_view(
        '<html><body style="margin:8px"><div id="content" style="height:900px">Gráfico</div><footer>Leyenda y estadísticas</footer></body></html>',
        200,
    )
    window.content.setWidget(view)

    def until(predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.02)
        assert predicate()

    until(lambda: view.height() > 930)
    first = view.height()
    view.page().runJavaScript(
        "document.getElementById('content').style.height = '1400px'"
    )
    until(lambda: view.height() > 1430)
    view.page().runJavaScript(
        "document.getElementById('content').style.height = '300px'"
    )
    until(lambda: 330 < view.height() < first)
    sizes = []
    view.page().runJavaScript(
        "document.documentElement.scrollHeight <= window.innerHeight", sizes.append
    )
    until(lambda: bool(sizes))
    assert sizes == [True]


def test_long_code_and_table_expand_to_full_height(window, application):
    import pandas as pd
    from mltutor.desktop.widgets import CodeEditor, DataTable

    code = CodeEditor("\n".join(f"print({i})" for i in range(100)), "python")
    window.content.setWidget(code)
    application.processEvents()
    assert code.verticalScrollBar().maximum() == 0
    table_box = window.build(
        ui.Node("table", {"data": pd.DataFrame({"dato": range(40)}), "height": 100})
    )
    window.content.setWidget(table_box)
    application.processEvents()
    table = table_box.findChild(DataTable)
    assert table.verticalScrollBar().maximum() == 0


def test_multiple_choices_wrap_and_enforce_limit(window, application):
    from mltutor.desktop.widgets import MultiSelect

    choices = MultiSelect(
        [
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
        ],
        ["alcohol"],
        2,
    )
    window.content.setWidget(choices)
    changes = []
    choices.changed.connect(changes.append)
    application.processEvents()
    QTest.mouseClick(choices.buttons[1], Qt.MouseButton.LeftButton)
    assert changes[-1] == ["alcohol", "malic_acid"]
    assert not choices.buttons[2].isEnabled()
    QTest.mouseClick(choices.buttons[0], Qt.MouseButton.LeftButton)
    assert changes[-1] == ["malic_acid"]
    assert choices.buttons[2].isEnabled()
    assert "1 de 6" in choices.summary.text()
    assert choices.columns > 1
    assert all(button.isVisible() for button in choices.buttons)
