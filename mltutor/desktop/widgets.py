"""Readable native data tables and syntax-highlighted code panels."""

import builtins
import io
import keyword
import tokenize

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFontDatabase,
    QKeySequence,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QHeaderView,
    QPlainTextEdit,
    QTableView,
    QAbstractItemView,
)


class DataTable(QTableView):
    """Share spare space across columns without sacrificing readable minimums."""

    def __init__(self, model, hide_index=False, height=None):
        super().__init__()
        self.setModel(model)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setMinimumWidth(0)
        self.verticalHeader().setVisible(not hide_index)
        self.verticalHeader().setDefaultSectionSize(36)
        self.verticalHeader().setMinimumSectionSize(36)
        header = self.horizontalHeader()
        header.setMinimumHeight(44)
        header.setMinimumSectionSize(100)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        self._user_sized = False
        self._resizing = False
        header.sectionResized.connect(self._manual_resize)
        self._minimums = []
        # Bounded sampling keeps even very large uploaded datasets responsive.
        metrics = self.fontMetrics()
        header_font = header.font()
        header_font.setBold(True)
        header.setFont(header_font)
        header_metrics = header.fontMetrics()
        for col in range(model.columnCount()):
            label = str(model.headerData(col, Qt.Orientation.Horizontal))
            samples = [
                str(model.data(model.index(row, col)))
                for row in range(min(100, model.rowCount()))
            ]
            width = max(
                [100, header_metrics.horizontalAdvance(label) + 36]
                + [min(320, metrics.horizontalAdvance(value) + 32) for value in samples]
            )
            self._minimums.append(width)
        rows = max(1, model.rowCount())
        self.setFixedHeight(
            max(height if isinstance(height, int) else 0, 46 + rows * 36 + 18)
        )
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setToolTip(
            "Selecciona celdas y usa Ctrl+C (⌘C en Mac) para copiarlas. Arrastra los bordes de las columnas para ajustar su ancho."
        )

    def _manual_resize(self, *_):
        if not self._resizing:
            self._user_sized = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_sized and self._minimums:
            self._resizing = True
            spare = max(0, self.viewport().width() - sum(self._minimums))
            extra, remainder = divmod(spare, len(self._minimums))
            for col, width in enumerate(self._minimums):
                self.setColumnWidth(col, width + extra + (col < remainder))
            self._resizing = False

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            selected = self.selectionModel().selectedIndexes()
            if selected:
                rows = sorted({i.row() for i in selected})
                cols = sorted({i.column() for i in selected})
                cells = {(i.row(), i.column()) for i in selected}
                model = self.model()
                # Export original values, not rounded display strings.
                text = "\t".join(str(model.frame.columns[c]) for c in cols) + "\n"
                text += "\n".join(
                    "\t".join(
                        str(model.frame.iat[r, c]) if (r, c) in cells else ""
                        for c in cols
                    )
                    for r in rows
                )
                QApplication.clipboard().setText(text)
            event.accept()
        else:
            super().keyPressEvent(event)


class PythonHighlighter(QSyntaxHighlighter):
    """Tokenize the entire read-only snippet, including multiline strings."""

    COLORS = {
        "keyword": "#c4a7ff",
        "builtin": "#7dd3fc",
        "name": "#82e2bf",
        "string": "#a8db8f",
        "comment": "#98a8c2",
        "number": "#f6bd79",
        "operator": "#89c9ef",
    }

    def __init__(self, document, source):
        super().__init__(document)
        self.spans = {}
        lines = source.splitlines()
        previous = None
        try:
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                kind = None
                if token.type == tokenize.NAME:
                    if keyword.iskeyword(token.string):
                        kind = "keyword"
                    elif token.string in vars(builtins):
                        kind = "builtin"
                    elif previous in ("def", "class"):
                        kind = "name"
                else:
                    kind = {
                        tokenize.STRING: "string",
                        tokenize.COMMENT: "comment",
                        tokenize.NUMBER: "number",
                        tokenize.OP: "operator",
                    }.get(token.type)
                if kind:
                    for row in range(token.start[0], token.end[0] + 1):
                        if row > len(lines):
                            continue
                        line = lines[row - 1]
                        start = token.start[1] if row == token.start[0] else 0
                        end = token.end[1] if row == token.end[0] else len(line)
                        # Qt positions count UTF-16 units, Python counts Unicode characters.
                        offset = len(line[:start].encode("utf-16-le")) // 2
                        length = len(line[start:end].encode("utf-16-le")) // 2
                        self.spans.setdefault(row - 1, []).append(
                            (offset, length, kind)
                        )
                previous = token.string
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass  # Preserve highlighting up to an unfinished example.
        self.rehighlight()

    def highlightBlock(self, text):
        for start, length, kind in self.spans.get(
            self.currentBlock().blockNumber(), []
        ):
            style = QTextCharFormat()
            style.setForeground(QColor(self.COLORS[kind]))
            if kind == "comment":
                style.setFontItalic(True)
            self.setFormat(start, length, style)


class CodeEditor(QPlainTextEdit):
    def __init__(self, source, language=None):
        super().__init__()
        self.setObjectName("codeEditor")
        self.setReadOnly(True)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(12)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)
        self.document().setDocumentMargin(18)
        self.setPlainText(source)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(
            max(140, (source.count("\n") + 2) * self.fontMetrics().lineSpacing() + 44)
        )
        if language in (None, "python", "py", "python3"):
            self.highlighter = PythonHighlighter(self.document(), source)


class DecimalSpinBox(QDoubleSpinBox):
    """Keep precise numeric input without displaying distracting trailing zeros."""

    def textFromValue(self, value):
        text = super().textFromValue(value)
        separator = self.locale().decimalPoint()
        if separator in text:
            return text.rstrip("0").removesuffix(separator)
        return text


class MultiSelect(QWidget):
    """Full-height, wrapping choice buttons with explicit selected states."""

    changed = Signal(object)

    def __init__(self, options, selected, limit=None):
        super().__init__()
        self.setObjectName("multiSelect")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.options = list(options)
        self.limit = limit
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(12)
        self.summary = QLabel()
        self.summary.setObjectName("selectionSummary")
        layout.addWidget(self.summary)
        self.grid = QGridLayout()
        self.grid.setSpacing(8)
        layout.addLayout(self.grid)
        self.buttons = []
        self.columns = 0
        for option in self.options:
            button = QPushButton(str(option))
            button.setObjectName("choiceChip")
            button.setCheckable(True)
            button.setChecked(option in selected)
            button.setMinimumWidth(0)
            button.setToolTip(str(option))
            button.toggled.connect(self._changed)
            self.buttons.append(button)
        self._refresh()
        self._reflow()

    def _refresh(self):
        count = sum(button.isChecked() for button in self.buttons)
        limit = f" · Máximo {self.limit}" if self.limit else ""
        self.summary.setText(f"{count} de {len(self.options)} seleccionadas{limit}")
        for option, button in zip(self.options, self.buttons):
            button.setText(("✓  " if button.isChecked() else "+  ") + str(option))
            button.setEnabled(
                not self.limit or count < self.limit or button.isChecked()
            )

    def _changed(self):
        self._refresh()
        self.changed.emit(
            [
                option
                for option, button in zip(self.options, self.buttons)
                if button.isChecked()
            ]
        )

    def _reflow(self):
        width = max(
            [190]
            + [
                min(340, self.fontMetrics().horizontalAdvance(str(x)) + 64)
                for x in self.options
            ]
        )
        columns = max(1, (self.width() - 28 + 8) // (width + 8))
        columns = min(columns, max(1, len(self.buttons)))
        if columns == self.columns:
            return
        old_columns = self.columns
        self.columns = columns
        while self.grid.count():
            self.grid.takeAt(0)
        for col in range(max(old_columns, columns)):
            self.grid.setColumnStretch(col, 1 if col < columns else 0)
        for index, button in enumerate(self.buttons):
            self.grid.addWidget(button, index // columns, index % columns)
        self.grid.invalidate()
        self.layout().activate()
        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()
