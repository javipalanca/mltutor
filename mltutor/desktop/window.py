"""Qt widgets and event loop for MLTutor's declarative lesson pages."""

from __future__ import annotations

import base64
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re
import tempfile
import traceback

import markdown
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableView,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
)
from . import ui


STYLE = """
QWidget { font-family: "Arial"; font-size: 14px; color: #263238; }
QMainWindow, QScrollArea, QWidget#page { background: #ffffff; }
QWidget#sidebar { background: #f0f2f6; }
QLabel#title { color: #1e88e5; font-size: 30px; font-weight: bold; }
QLabel#heading { color: #0d47a1; font-size: 23px; font-weight: bold; }
QLabel#subheading { color: #0d47a1; font-size: 19px; font-weight: bold; }
QPushButton { background: white; border: 1px solid #d5dce5; border-radius: 6px;
             padding: 9px 12px; min-height: 20px; }
QPushButton:hover { background: #e3f2fd; border-color: #90caf9; }
QPushButton[primary="true"] { background: #1e88e5; color: white; border-color: #1e88e5; }
QPushButton:disabled { color: #9aa3ac; background: #f4f5f7; }
QComboBox, QSpinBox, QDoubleSpinBox, QListWidget { background: #f6f8fb;
    border: 1px solid #d5dce5; border-radius: 5px; padding: 7px; min-height: 22px; }
QComboBox QAbstractItemView { background: white; selection-background-color: #e3f2fd; }
QTableView { background: white; alternate-background-color: #f5f8fc;
    gridline-color: #e4e8ed; border: 1px solid #dce2ea; }
QHeaderView::section { background: #eef3f9; padding: 7px; border: none; }
QTabWidget::pane { border: 1px solid #e0e6ed; }
QTabBar::tab { background: #f0f2f6; padding: 12px; }
QTabBar::tab:selected { background: #e3f2fd; color: #0d47a1; }
QProgressBar { border: 1px solid #dce2ea; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #1e88e5; }
QToolTip { color: #263238; background: #fff8e1; border: 1px solid #ffe082; }
"""

TEXT_CSS = """
body { font-family: Arial; color: #263238; font-size: 14px; }
h1 { color: #1e88e5; font-size: 30px; }
h2,h3,h4 { color: #0d47a1; }
p,li { line-height: 145%; }
.main-header { color: #1e88e5; text-align: center; font-size: 32px; }
.sub-header { color: #0d47a1; font-size: 20px; }
.info-box { background-color: #e3f2fd; }
.footer { color: #888888; font-size: 10px; text-align: center; }
a { color: #1976d2; }
code { background-color: #eceff1; }
"""


class Downloads(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.active = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href", "").startswith("data:"):
            self.active = [attrs["href"], attrs.get("download", "descarga"), ""]

    def handle_data(self, data):
        if self.active is not None:
            self.active[2] += data

    def handle_endtag(self, tag):
        if tag == "a" and self.active is not None:
            self.items.append(self.active)
            self.active = None


class RichText(QTextBrowser):
    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self.open_link)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.document().setDefaultStyleSheet(TEXT_CSS)
        self.setHtml(source)
        self.setStyleSheet("background: transparent; border: none;")

    def open_link(self, url):
        if url.scheme() in ("https", "http", "mailto"):
            QDesktopServices.openUrl(url)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.document().setTextWidth(max(40, self.viewport().width()))
        self.setFixedHeight(max(28, int(self.document().size().height()) + 8))


class Picture(QLabel):
    def __init__(self, data):
        super().__init__()
        self.original = QPixmap()
        self.original.loadFromData(data)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.original.isNull():
            scaled = self.original.scaledToWidth(
                max(1, self.width()), Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)
            self.setFixedHeight(scaled.height())


class TableModel(QAbstractTableModel):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.frame = frame

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            value = self.frame.iat[index.row(), index.column()]
            return f"{value:.5g}" if isinstance(value, float) else str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            source = (
                self.frame.columns
                if orientation == Qt.Orientation.Horizontal
                else self.frame.index
            )
            return str(source[section])
        return None


class Worker(QThread):
    result = Signal(object, object)
    progress = Signal(str, object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, session, action=None, parent=None):
        super().__init__(parent)
        # macOS QThread defaults to a much smaller stack than Python's main
        # thread. Accelerate/LAPACK and TensorFlow need room for native frames.
        self.setStackSize(16 * 1024 * 1024)
        self.session, self.action = session, action

    def run(self):
        self.session.notify = self.progress.emit
        try:
            with ui.bind(self.session):
                if self.action:
                    self.session.dispatch(*self.action)
                from mltutor.app import main

                root, side = self.session.render(main)
            self.result.emit(root, side)
        except ui.Cancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.session.notify = lambda *args: None
            # Figures have already been converted to bytes, so workers never hand
            # mutable matplotlib objects across threads.
            import matplotlib.pyplot as plt

            plt.close("all")


class MainWindow(QMainWindow):
    page_ready = Signal()

    def __init__(self, autostart=True):
        super().__init__()
        self.setWindowTitle("MLTutor — Aprende Machine Learning")
        self.resize(1400, 920)
        self.setMinimumSize(1000, 700)
        icon = Path(__file__).parents[1] / "assets" / "icon.png"
        self.setWindowIcon(QIcon(str(icon)))
        self.session = ui.Session()
        self.worker = None
        self.closed_requested = False
        self.view_state = {}
        self.controls = {}
        self.webviews = []
        self.web_profile = None
        self.temp = tempfile.TemporaryDirectory(prefix="mltutor-")
        self.html_dir = Path(self.temp.name)
        self.root_node = None
        self.failed_message = None
        self.previous_scope = None
        self.central = QWidget()
        self.setCentralWidget(self.central)
        layout = QVBoxLayout(self.central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter()
        self.sidebar = self._scroll()
        self.content = self._scroll()
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.content)
        self.splitter.setSizes([250, 1150])
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        layout.addWidget(self.splitter)
        self.status = QLabel("Preparando MLTutor…")
        self.bar = QProgressBar()
        self.bar.setFixedWidth(170)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.cancel_work)
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.bar)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.setStyleSheet(STYLE)
        if autostart:
            QTimer.singleShot(0, self.evaluate)

    @staticmethod
    def _scroll():
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        return scroll

    def evaluate(self, action=None):
        if self.worker is not None:
            return
        self.failed_message = None
        self.session.cancel.clear()
        self.splitter.setEnabled(False)
        self.bar.setRange(0, 0)
        self.bar.show()
        self.cancel_button.show()
        self.status.setText("Actualizando…")
        self.worker = Worker(self.session, action, self)
        self.worker.result.connect(self.present)
        self.worker.progress.connect(self.update_progress)
        self.worker.failed.connect(self.show_failure)
        self.worker.cancelled.connect(
            lambda: self.status.setText("Operación cancelada")
        )
        self.worker.finished.connect(self.work_finished)
        self.worker.start()

    def update_progress(self, message, percent):
        self.status.setText(message)
        if percent is not None:
            self.bar.setRange(0, 100)
            self.bar.setValue(max(0, min(100, int(percent))))

    def cancel_work(self):
        self.session.cancel.set()
        self.status.setText("Cancelando al terminar el paso actual…")
        self.cancel_button.setEnabled(False)

    def work_finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.splitter.setEnabled(True)
        self.bar.hide()
        self.cancel_button.hide()
        self.cancel_button.setEnabled(True)
        if self.closed_requested:
            self.close()
        else:
            self.page_ready.emit()

    def show_failure(self, message):
        self.failed_message = message
        self.status.setText("No se pudo completar la operación")
        body = ui.Node(
            "page",
            children=[
                ui.Node(
                    "notice",
                    {
                        "severity": "error",
                        "text": "No se pudo completar la operación. Puedes volver a intentarlo o elegir otra sección.",
                    },
                ),
                ui.Node("code", {"text": message}),
            ],
        )
        self.content.setWidget(self.build(body))

    def present(self, root, side):
        self.root_node = root
        scope = (
            self.session.state.get("navigation"),
            tuple(
                (k, v)
                for k, v in self.session.state.items()
                if k.startswith("active_tab")
            ),
        )
        scroll = (
            self.content.verticalScrollBar().value()
            if scope == self.previous_scope
            else 0
        )
        self.previous_scope = scope
        self.controls.clear()
        for view in self.webviews:
            view.stop()
        self.webviews.clear()
        content = self.build(root)
        content.setObjectName("page")
        sidebar = self.build(side)
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(215)
        self.sidebar.setWidget(sidebar)
        self.content.setWidget(content)
        QTimer.singleShot(0, lambda: self.content.verticalScrollBar().setValue(scroll))
        self.status.setText("Listo")

    def action(self, node, value=None):
        self.evaluate((node, value))

    def save_data(self, data, name):
        destination, _ = QFileDialog.getSaveFileName(
            self, "Guardar archivo", str(Path.home() / Path(name).name)
        )
        if destination:
            try:
                Path(destination).write_bytes(data)
                self.status.setText(f"Guardado: {Path(destination).name}")
            except OSError as exc:
                QMessageBox.warning(self, "No se pudo guardar", str(exc))

    def choose_file(self, node):
        extensions = node.props.get("extensions", [])
        pattern = " ".join(f"*.{x}" for x in extensions) or "*"
        path, _ = QFileDialog.getOpenFileName(
            self, node.props["label"], str(Path.home()), f"Archivos ({pattern})"
        )
        if path:
            try:
                self.action(node, (Path(path).name, Path(path).read_bytes()))
            except OSError as exc:
                QMessageBox.warning(self, "No se pudo abrir", str(exc))

    def _box(self, horizontal=False):
        widget = QWidget()
        layout = QHBoxLayout(widget) if horizontal else QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        widget.setMinimumWidth(0)
        return widget, layout

    def _labelled(self, node):
        box, layout = self._box()
        label = QLabel(node.props.get("label", ""))
        label.setWordWrap(True)
        layout.addWidget(label)
        box.setToolTip(node.props.get("help") or "")
        return box, layout

    def _register(self, node, control):
        identity = node.props.get("id")
        if identity:
            control.setObjectName(identity)
            self.controls[identity] = control
        control.setToolTip(node.props.get("help") or "")
        control.setEnabled(not node.props.get("disabled", False))
        return control

    def build(self, node):
        kind, p = node.kind, node.props
        if kind in ("page", "column", "row"):
            box, layout = self._box(kind == "row")
            if kind == "page":
                layout.setContentsMargins(30, 20, 30, 24)
            for index, child in enumerate(node.children):
                widget = self.build(child)
                if kind == "row":
                    layout.addWidget(widget, max(1, int(p["weights"][index] * 10)))
                else:
                    layout.addWidget(widget)
            return box
        if kind == "expander":
            box, layout = self._box()
            toggle = QPushButton(p["label"])
            toggle.setCheckable(True)
            opened = self.view_state.get(p["id"], p["expanded"])
            toggle.setChecked(opened)
            layout.addWidget(toggle)
            body = self.build(ui.Node("column", children=node.children))
            body.setVisible(opened)
            layout.addWidget(body)
            toggle.toggled.connect(body.setVisible)
            toggle.toggled.connect(
                lambda value: self.view_state.__setitem__(p["id"], value)
            )
            return box
        if kind == "tabs":
            tabs = QTabWidget()
            for label, child in zip(p["labels"], node.children):
                tabs.addTab(self.build(child), label)
            tabs.setCurrentIndex(self.view_state.get(p["id"], 0))
            tabs.currentChanged.connect(
                lambda value: self.view_state.__setitem__(p["id"], value)
            )
            return tabs
        if kind in ("heading", "caption", "plain"):
            label = QLabel(p["text"])
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            if kind == "heading":
                label.setObjectName("heading" if p["level"] == 1 else "subheading")
            elif kind == "caption":
                label.setStyleSheet("color: #65748b; font-size: 12px;")
            return label
        if kind in ("text", "notice"):
            body = p["text"]
            parser = Downloads()
            parser.feed(body)
            box, layout = self._box()
            for data_url, name, label in parser.items:
                try:
                    data = base64.b64decode(data_url.split(",", 1)[1])
                    button = QPushButton(label or "Guardar archivo")
                    button.clicked.connect(
                        lambda checked=False, d=data, n=name: self.save_data(d, n)
                    )
                    layout.addWidget(button)
                except (ValueError, IndexError):
                    pass
            body = re.sub(r'<a\b[^>]*href=["\']data:.*?</a>', "", body, flags=re.S)
            # QTextDocument paints a div background behind individual text
            # lines. Turn the original HTML cards into actual Qt panels.
            card = re.match(r'\s*<div\s+style=["\']([^"\']+)["\']\s*>', body)
            if card:
                color = re.search(r"background-color:\s*(#[0-9a-fA-F]+)", card[1])
                if color:
                    box.setObjectName("lessonCard")
                    box.setStyleSheet(
                        f"QWidget#lessonCard {{ background: {color[1]}; border-radius: 9px; }}"
                    )
                    layout.setContentsMargins(16, 12, 16, 12)
                    body = body[card.end() :]
                    body = re.sub(r"</div>\s*$", "", body)
                    if "height: 200px" in card[1]:
                        box.setMinimumHeight(180)
            if body.strip():
                # HTML blocks in the lessons are already formatted. Markdown
                # handles the remaining explanations, tables, links and lists.
                source = (
                    body
                    if p.get("html")
                    else markdown.markdown(body, extensions=["tables", "fenced_code"])
                )
                layout.addWidget(RichText(source))
            if kind == "notice":
                colors = {
                    "info": "#e3f2fd",
                    "warning": "#fff8e1",
                    "error": "#ffebee",
                    "success": "#e8f5e9",
                }
                box.setStyleSheet(
                    f"background: {colors[p['severity']]}; border-radius: 6px;"
                )
            return box
        if kind == "metric":
            box, layout = self._box()
            box.setStyleSheet("background: #f3f7fc; border-radius: 7px;")
            label = QLabel(p["label"])
            label.setWordWrap(True)
            value = QLabel(p["value"])
            value.setWordWrap(True)
            value.setStyleSheet("font-size: 26px; font-weight: bold; color: #0d47a1;")
            layout.addWidget(label)
            layout.addWidget(value)
            if p.get("delta") is not None:
                layout.addWidget(QLabel(str(p["delta"])))
            box.setToolTip(p.get("help") or "")
            return box
        if kind == "code":
            box, layout = self._box()
            editor = QPlainTextEdit(p["text"])
            editor.setReadOnly(True)
            editor.setFont(QFont("Menlo", 11))
            editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            editor.setFixedHeight(min(440, max(100, (p["text"].count("\n") + 2) * 18)))
            copy = QPushButton("Copiar código")
            copy.clicked.connect(lambda: QApplication.clipboard().setText(p["text"]))
            layout.addWidget(copy)
            layout.addWidget(editor)
            return box
        if kind == "button":
            button = self._register(node, QPushButton(p["label"]))
            button.setProperty("primary", p.get("type") == "primary")
            button.clicked.connect(lambda: self.action(node))
            return button
        if kind == "download":
            button = QPushButton(p["label"])
            button.clicked.connect(lambda: self.save_data(p["data"], p["file_name"]))
            return button
        if kind == "file":
            box, layout = self._labelled(node)
            name = p["value"][0] if p["value"] else "Seleccionar archivo CSV…"
            button = self._register(node, QPushButton(name))
            button.clicked.connect(lambda: self.choose_file(node))
            layout.addWidget(button)
            if p["value"]:
                clear = QPushButton("Quitar archivo")
                clear.clicked.connect(lambda: self.action(node, None))
                layout.addWidget(clear)
            return box
        if kind == "checkbox":
            control = self._register(node, QCheckBox(p["label"]))
            control.setChecked(bool(p["value"]))
            control.toggled.connect(lambda value: self.action(node, value))
            return control
        if kind in ("select", "radio"):
            box, layout = self._labelled(node)
            if kind == "select":
                combo = self._register(node, QComboBox())
                combo.addItems(p["display"])
                combo.setCurrentIndex(
                    p["options"].index(p["value"]) if p["value"] in p["options"] else -1
                )
                combo.currentIndexChanged.connect(
                    lambda i: self.action(node, p["options"][i]) if i >= 0 else None
                )
                layout.addWidget(combo)
            else:
                row, buttons = self._box(p.get("horizontal", False))
                for option, label in zip(p["options"], p["display"]):
                    button = QRadioButton(label)
                    button.setChecked(option == p["value"])
                    button.toggled.connect(
                        lambda checked, value=option: self.action(node, value)
                        if checked
                        else None
                    )
                    buttons.addWidget(button)
                self._register(node, row)
                layout.addWidget(row)
            return box
        if kind == "multiselect":
            box, layout = self._labelled(node)
            listing = self._register(node, QListWidget())
            listing.setFixedHeight(min(220, max(80, len(p["options"]) * 28)))
            for option in p["options"]:
                item = QListWidgetItem(str(option), listing)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if option in p["value"]
                    else Qt.CheckState.Unchecked
                )

            def changed(item):
                selected = [
                    p["options"][i]
                    for i in range(listing.count())
                    if listing.item(i).checkState() == Qt.CheckState.Checked
                ]
                limit = p.get("max_selections")
                if limit and len(selected) > limit:
                    listing.blockSignals(True)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    listing.blockSignals(False)
                    return
                self.action(node, selected)

            listing.itemChanged.connect(changed)
            layout.addWidget(listing)
            return box
        if kind in ("slider", "number"):
            box, layout = self._labelled(node)
            integer = isinstance(p["value"], int)
            low = p.get("min_value") if p.get("min_value") is not None else -1e9
            high = p.get("max_value") if p.get("max_value") is not None else 1e9
            step = p.get("step") or (1 if integer else 0.01)
            spin = QSpinBox() if integer else QDoubleSpinBox()
            if not integer:
                spin.setDecimals(6)
            spin.setRange(
                int(low) if integer else float(low),
                int(high) if integer else float(high),
            )
            spin.setSingleStep(step)
            spin.setValue(p["value"])
            spin.setKeyboardTracking(False)
            self._register(node, spin)
            if kind == "slider":
                slider = QSlider(Qt.Orientation.Horizontal)
                ticks = min(100000, max(1, round((high - low) / step)))
                slider.setRange(0, ticks)
                slider.setValue(
                    round((p["value"] - low) / (high - low) * ticks)
                    if high > low
                    else 0
                )

                def slide(value):
                    numeric = low + (high - low) * value / ticks
                    spin.blockSignals(True)
                    spin.setValue(round(numeric) if integer else numeric)
                    spin.blockSignals(False)

                slider.valueChanged.connect(slide)
                slider.sliderReleased.connect(lambda: self.action(node, spin.value()))
                layout.addWidget(slider)
            spin.valueChanged.connect(lambda value: self.action(node, value))
            layout.addWidget(spin)
            return box
        if kind == "table":
            table = QTableView()
            table.setModel(TableModel(p["data"], table))
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.verticalHeader().setVisible(not p.get("hide_index", False))
            table.setFixedHeight(p.get("height", min(350, 55 + len(p["data"]) * 30)))
            table.resizeColumnsToContents()
            table.horizontalHeader().setStretchLastSection(True)
            return table
        if kind == "image":
            box, layout = self._box()
            layout.addWidget(Picture(p["data"]))
            if p.get("caption"):
                layout.addWidget(QLabel(p["caption"]))
            button = QPushButton("Guardar imagen…")
            button.clicked.connect(
                lambda: self.save_data(p["data"], "visualizacion.png")
            )
            layout.addWidget(button)
            return box
        if kind == "html":
            return self.html_view(p["html"], p["height"])
        if kind == "progress":
            bar = QProgressBar()
            bar.setRange(0, 100)
            value = p["value"]
            bar.setValue(int(value if isinstance(value, int) else value * 100))
            return bar
        raise ValueError(f"Componente de interfaz desconocido: {kind}")

    def html_view(self, body, height):
        from PySide6.QtWebEngineCore import (
            QWebEnginePage,
            QWebEngineProfile,
            QWebEngineUrlRequestInterceptor,
        )
        from PySide6.QtWebEngineWidgets import QWebEngineView

        if self.web_profile is None:

            class LocalOnly(QWebEngineUrlRequestInterceptor):
                def interceptRequest(self, request):
                    if request.requestUrl().scheme() not in (
                        "file",
                        "data",
                        "blob",
                        "qrc",
                        "about",
                    ):
                        request.block(True)

            self.web_profile = QWebEngineProfile(self)
            self.interceptor = LocalOnly(self.web_profile)
            self.web_profile.setUrlRequestInterceptor(self.interceptor)
            self.web_profile.downloadRequested.connect(self.web_download)
        js = self.html_dir / "plotly.min.js"
        if not js.exists():
            from plotly.offline import get_plotlyjs

            js.write_text(get_plotlyjs(), encoding="utf-8")
        path = self.html_dir / (hashlib.sha256(body.encode()).hexdigest() + ".html")
        path.write_text(body, encoding="utf-8")
        view = QWebEngineView()
        view.setPage(QWebEnginePage(self.web_profile, view))
        view.setFixedHeight(int(height))
        view.load(QUrl.fromLocalFile(str(path)))
        self.webviews.append(view)
        return view

    def web_download(self, request):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar gráfico", str(Path.home() / request.downloadFileName())
        )
        if path:
            request.setDownloadDirectory(str(Path(path).parent))
            request.setDownloadFileName(Path(path).name)
            request.accept()
        else:
            request.cancel()

    def closeEvent(self, event):
        if self.worker is not None:
            self.closed_requested = True
            self.cancel_work()
            event.ignore()
            return
        for view in self.webviews:
            view.stop()
            from shiboken6 import delete

            delete(view)
        self.webviews.clear()
        if self.web_profile is not None:
            from shiboken6 import delete

            delete(self.web_profile)
            self.web_profile = None
        self.temp.cleanup()
        if self.session.upload_dir is not None:
            self.session.upload_dir.cleanup()
        event.accept()
