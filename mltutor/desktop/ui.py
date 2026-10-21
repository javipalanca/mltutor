"""Declarative presentation model for the Qt desktop application.

Lesson functions describe a tree of controls; the Qt renderer owns the widgets.
Calculations run on a worker, never touching Qt objects. Keeping this small API
lets all existing lessons, explanations and plots share the native interface.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import inspect
import io
from pathlib import Path
import re
import threading
import tempfile
import textwrap
import traceback


class State(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


@dataclass
class Node:
    kind: str
    props: dict = field(default_factory=dict)
    children: list = field(default_factory=list)


class Rerun(BaseException):
    """Restart presentation after a navigation action, without repeating it."""


class Cancelled(BaseException):
    """Cooperative cancellation of a page or training operation."""


class Session:
    def __init__(self):
        self.state = State()
        self.values = {}
        self.control_types = {}
        self.clicked = None
        self.cancel = threading.Event()
        self.notify = lambda message, percent=None: None
        self.root = Node("page")
        self.side = Node("column")
        self.stack = [self.root]
        self.counts = Counter()
        self.scope = ""
        self.upload_dir = None

    def begin(self):
        self.root = Node("page")
        self.side = Node("column")
        self.stack = [self.root]
        self.counts.clear()
        self.scope = str(self.state.get("navigation", "🏠 Inicio"))

    def dispatch(self, node, value=None):
        if node.kind == "button":
            self.clicked = node.props["id"]
        else:
            self.values[node.props["id"]] = value
            key = node.props.get("key")
            if key is not None:
                self.state[key] = value
        callback = node.props.get("on_change") or node.props.get("on_click")
        if callback:
            with bind(self):
                callback()

    def render(self, page):
        with bind(self):
            try:
                for _ in range(20):
                    self.begin()
                    try:
                        page()
                        break
                    except Rerun:
                        continue
                else:
                    raise RuntimeError("La navegación no ha podido estabilizarse.")
            finally:
                self.clicked = None
        return self.root, self.side


_local = threading.local()


def current():
    if not hasattr(_local, "session"):
        _local.session = Session()
    return _local.session


@contextmanager
def bind(session):
    previous = getattr(_local, "session", None)
    _local.session = session
    try:
        yield session
    finally:
        if previous is None:
            del _local.session
        else:
            _local.session = previous


class StateProxy:
    def __getattr__(self, name):
        return getattr(current().state, name)

    def __setattr__(self, name, value):
        setattr(current().state, name, value)

    def __getitem__(self, key):
        return current().state[key]

    def __setitem__(self, key, value):
        current().state[key] = value

    def __delitem__(self, key):
        del current().state[key]

    def __contains__(self, key):
        return key in current().state

    def __iter__(self):
        return iter(current().state)


session_state = StateProxy()


def check_cancelled():
    if current().cancel.is_set():
        raise Cancelled()


def report_progress(message, percent=None):
    check_cancelled()
    current().notify(str(message), percent)


def _id(kind, label="", key=None):
    if key is not None:
        return f"key:{key}"
    # The call site and occurrence keep controls stable when conditional content
    # elsewhere in the page changes. Labels distinguish controls inside loops.
    frame = inspect.currentframe().f_back
    while frame and frame.f_code.co_filename == __file__:
        frame = frame.f_back
    site = (Path(frame.f_code.co_filename).name, frame.f_lineno) if frame else ("", 0)
    base = f"{current().scope}:{site}:{kind}:{label}"
    occurrence = current().counts[base]
    current().counts[base] += 1
    return hashlib.sha256(f"{base}:{occurrence}".encode()).hexdigest()[:20]


def _emit(kind, **props):
    check_cancelled()
    node = Node(kind, props)
    current().stack[-1].children.append(node)
    return node


class Block:
    def __init__(self, node, replace=False):
        self.node, self.replace = node, replace

    def __enter__(self):
        if self.replace:
            self.node.children.clear()
        current().stack.append(self.node)
        return self

    def __exit__(self, *args):
        current().stack.pop()

    def __getattr__(self, name):
        fn = globals().get(name)
        if not callable(fn):
            raise AttributeError(name)

        def call(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)

        return call

    def empty(self):
        self.node.kind = "column"
        self.node.props.clear()
        self.node.children.clear()

    def container(self, **kwargs):
        return Block(self.node, replace=self.replace)

    def progress(self, value, text=None):
        self.node.props["value"] = value
        report_progress(
            text or "Trabajando…", value if isinstance(value, int) else value * 100
        )
        return self


class Sidebar:
    def __getattr__(self, name):
        return getattr(Block(current().side), name)


sidebar = Sidebar()


def container(**kwargs):
    return Block(_emit("column"))


def empty():
    return Block(_emit("column"), replace=True)


def columns(spec, **kwargs):
    weights = [1] * spec if isinstance(spec, int) else list(spec)
    row = _emit("row", weights=weights)
    row.children = [Node("column") for _ in weights]
    return [Block(node) for node in row.children]


def expander(label, expanded=False, **kwargs):
    return Block(
        _emit("expander", label=label, expanded=expanded, id=_id("expander", label))
    )


def tabs(labels):
    node = _emit("tabs", labels=list(labels), id=_id("tabs", str(labels)))
    node.children = [Node("column") for _ in labels]
    return [Block(child) for child in node.children]


def set_page_config(**kwargs):
    current().root.props.update(kwargs)


def rerun():
    raise Rerun()


def markdown(body, unsafe_allow_html=False, **kwargs):
    body = re.sub(
        r"<style\b[^>]*>.*?</style>", "", textwrap.dedent(str(body)), flags=re.S | re.I
    ).strip()
    # Styling-only wrappers were meaningful in the browser, not in Qt layouts.
    if not body or re.fullmatch(r"</?div[^>]*>", body):
        return
    return _emit("text", text=body, html=unsafe_allow_html)


def header(body, **kwargs):
    return _emit("heading", text=str(body), level=1)


def subheader(body, **kwargs):
    return _emit("heading", text=str(body), level=2)


def caption(body, **kwargs):
    return _emit("caption", text=str(body))


def text(body, **kwargs):
    return _emit("plain", text=str(body))


def write(*args, **kwargs):
    for value in args:
        if hasattr(value, "columns") or isinstance(value, dict):
            dataframe(value)
        else:
            markdown(value)


def _notice(kind, body, **kwargs):
    body = textwrap.dedent(str(body)).strip()
    report_progress(re.sub(r"<[^>]+>", "", str(body))[:250])
    return _emit("notice", text=str(body), severity=kind)


def info(body, **kwargs):
    return _notice("info", body, **kwargs)


def warning(body, **kwargs):
    return _notice("warning", body, **kwargs)


def error(body, **kwargs):
    return _notice("error", body, **kwargs)


def success(body, **kwargs):
    return _notice("success", body, **kwargs)


def exception(exc):
    error(str(exc))
    code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def metric(label, value, delta=None, **kwargs):
    return _emit("metric", label=label, value=str(value), delta=delta, **kwargs)


def code(body, language=None, **kwargs):
    return _emit("code", text=str(body), language=language)


def json(body, **kwargs):
    import json as encoder

    code(encoder.dumps(body, indent=2, ensure_ascii=False, default=str), "json")


@contextmanager
def spinner(text="Trabajando…", **kwargs):
    report_progress(text)
    yield
    check_cancelled()


def progress(value, text=None):
    block = Block(_emit("progress", value=value, text=text))
    block.progress(value, text)
    return block


def _input(kind, label, default, key=None, **props):
    identity = _id(kind, label, key)
    state = current().state
    if current().control_types.get(identity, kind) != kind:
        current().values.pop(identity, None)
        if key is not None:
            state.pop(key, None)
    current().control_types[identity] = kind
    value = (
        state.get(key, default)
        if key is not None
        else current().values.get(identity, default)
    )
    options = props.get("options")
    if options is not None:
        if kind == "multiselect":
            value = [x for x in (value or []) if x in options][
                : props.get("max_selections") or len(options)
            ]
        elif value not in options:
            value = default
    if kind in ("slider", "number"):
        low, high = props.get("min_value"), props.get("max_value")
        if low is not None:
            value = max(low, value)
        if high is not None:
            value = min(high, value)
    current().values[identity] = value
    if key is not None:
        state[key] = value
    _emit(kind, id=identity, label=label, key=key, value=value, **props)
    return value


def button(label, key=None, **kwargs):
    identity = _id("button", label, key)
    _emit("button", id=identity, key=key, label=label, **kwargs)
    if current().clicked == identity and not kwargs.get("disabled"):
        current().clicked = None
        return True
    return False


def selectbox(label, options, index=0, key=None, format_func=str, **kwargs):
    options = list(options)
    default = options[index] if options and index is not None else None
    return _input(
        "select",
        label,
        default,
        key,
        options=options,
        display=[str(format_func(x)) for x in options],
        **kwargs,
    )


def radio(label, options, index=0, key=None, format_func=str, **kwargs):
    options = list(options)
    return _input(
        "radio",
        label,
        options[index] if options else None,
        key,
        options=options,
        display=[str(format_func(x)) for x in options],
        **kwargs,
    )


def multiselect(label, options, default=None, key=None, **kwargs):
    return _input(
        "multiselect", label, list(default or []), key, options=list(options), **kwargs
    )


def checkbox(label, value=False, key=None, **kwargs):
    return _input("checkbox", label, value, key, **kwargs)


def slider(
    label, min_value=0, max_value=100, value=None, step=None, key=None, **kwargs
):
    return _input(
        "slider",
        label,
        min_value if value is None else value,
        key,
        min_value=min_value,
        max_value=max_value,
        step=step,
        **kwargs,
    )


def number_input(
    label, min_value=None, max_value=None, value=0, step=None, key=None, **kwargs
):
    return _input(
        "number",
        label,
        value,
        key,
        min_value=min_value,
        max_value=max_value,
        step=step,
        **kwargs,
    )


class UploadedFile(io.BytesIO):
    def __init__(self, name, data):
        super().__init__(data)
        self.name = name
        self.size = len(data)


def file_uploader(label, type=None, key=None, **kwargs):
    value = _input("file", label, None, key, extensions=type or [], **kwargs)
    # A fresh stream on each evaluation prevents read_csv seeing an exhausted file.
    return UploadedFile(value[0], value[1]) if value else None


def store_csv(data):
    """Keep imported datasets for this session without leaking temporary files."""
    session = current()
    if session.upload_dir is None:
        session.upload_dir = tempfile.TemporaryDirectory(prefix="mltutor-csv-")
    payload = data.to_csv(index=False).encode("utf-8")
    path = Path(session.upload_dir.name) / (
        hashlib.sha256(payload).hexdigest() + ".csv"
    )
    if not path.exists():
        path.write_bytes(payload)
    return str(path)


def download_button(label, data, file_name="descarga", mime=None, **kwargs):
    if hasattr(data, "getvalue"):
        data = data.getvalue()
    if isinstance(data, str):
        data = data.encode("utf-8")
    _emit(
        "download",
        label=label,
        data=bytes(data),
        file_name=file_name,
        mime=mime,
        **kwargs,
    )
    return False


def dataframe(data, **kwargs):
    import pandas as pd
    from pandas.io.formats.style import Styler

    if isinstance(data, Styler):
        data = data.data
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    return _emit("table", data=data.copy(), **kwargs)


def pyplot(fig=None, **kwargs):
    import matplotlib.pyplot as plt

    fig = fig if fig is not None else plt.gcf()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    return _emit("image", data=buffer.getvalue(), caption="", **kwargs)


def image(image, caption=None, **kwargs):
    if isinstance(image, str) and image.startswith(("https://", "http://")):
        # Supplementary illustrations are replaced by bundled ones by the lesson.
        return globals()["caption"](caption or "")
    data = Path(image).read_bytes() if isinstance(image, (str, Path)) else image
    if hasattr(data, "getvalue"):
        data = data.getvalue()
    return _emit("image", data=data, caption=caption or "", **kwargs)


def plotly_chart(figure, **kwargs):
    # Local Plotly JS is written once by the renderer; never use a CDN.
    html = figure.to_html(
        include_plotlyjs="plotly.min.js",
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    return _emit("html", html=html, height=figure.layout.height or 520)


def html(body, height=400, scrolling=False, **kwargs):
    return _emit("html", html=str(body), height=height)
