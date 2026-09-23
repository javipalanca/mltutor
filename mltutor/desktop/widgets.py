"""Readable native data tables and syntax-highlighted code panels."""

import builtins
import io
import keyword
import tokenize

from PySide6.QtCore import Qt
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
        rows = min(10, max(1, model.rowCount()))
        self.setFixedHeight(height if isinstance(height, int) else 46 + rows * 36 + 18)
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
        self.setFixedHeight(
            min(
                500,
                max(
                    140,
                    (source.count("\n") + 2) * self.fontMetrics().lineSpacing() + 44,
                ),
            )
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
