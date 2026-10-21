"""Headless rendering of reference lessons for golden fixtures.

The Qt app builds a declarative node tree (``mltutor.desktop.ui``) that the Qt
window renders. This module renders a lesson page with a plain ``ui.Session``
(no Qt), presses buttons or sets controls like the user would, and converts the
resulting tree into JSON: texts, notices, metrics, tables, code blocks, controls
and, for every ``st.pyplot`` figure, the data matplotlib actually drew (lines,
bars, scatter points, heatmaps, contour grids, colours, titles, axes, legends).

Used by ``tools/export_golden.py`` (phase 2 onwards). Reusable for the tree, KNN
and neural-network lessons.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

# Point clouds larger than this are summarised (count, hash, head, min/max):
# the raw values are already in the fixture's numeric section.
MAX_POINTS = 600
# Contour grids: keep the axis vectors and a decimated Z (every k-th row/col).
GRID_DECIMATE_TARGET = 40


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(v) -> Any:
    """JSON-safe float (NaN/inf -> None)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    return x if math.isfinite(x) else None


def _flist(values) -> list:
    return [_f(v) for v in np.asarray(values, dtype=float).ravel().tolist()]


def _hex(color) -> str | None:
    from matplotlib import colors as mcolors

    try:
        return mcolors.to_hex(color, keep_alpha=True)
    except (ValueError, TypeError):
        return None


def _sha(values) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _points(arr: np.ndarray) -> dict:
    """Full values when small, a summary otherwise."""
    arr = np.asarray(arr, dtype=float)
    if arr.size <= MAX_POINTS * (arr.shape[1] if arr.ndim == 2 else 1):
        return {"n": int(arr.shape[0]) if arr.ndim else 0,
                "values": [_flist(r) for r in arr] if arr.ndim == 2 else _flist(arr)}
    return {
        "n": int(arr.shape[0]),
        "summarised": True,
        "sha256_f64": _sha(arr),
        "head": [_flist(r) for r in arr[:5]] if arr.ndim == 2 else _flist(arr[:5]),
        "min": _flist(np.nanmin(arr, axis=0)),
        "max": _flist(np.nanmax(arr, axis=0)),
    }


def _unique_colors(rgba: np.ndarray) -> list:
    if rgba is None or len(rgba) == 0:
        return []
    seen: dict[str, int] = {}
    for c in rgba:
        h = _hex(c)
        seen[h] = seen.get(h, 0) + 1
    return [{"color": k, "count": v} for k, v in seen.items()]


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------
_CONTOUR_RECORDS: list[dict] = []


@contextmanager
def record_contourf():
    """Record the (x, y, Z) arguments of every ``Axes.contourf`` call."""
    from matplotlib.axes import Axes

    original = Axes.contourf

    def patched(self, *args, **kwargs):
        cs = original(self, *args, **kwargs)
        if len(args) >= 3:
            xx, yy, zz = (np.asarray(a) for a in args[:3])
            _CONTOUR_RECORDS.append({"ax": self, "xx": xx, "yy": yy, "Z": zz,
                                     "kwargs": {k: v for k, v in kwargs.items()
                                                if isinstance(v, (int, float, str))},
                                     "levels": np.asarray(cs.levels)})
        return cs

    Axes.contourf = patched
    try:
        yield
    finally:
        Axes.contourf = original


def _grid(rec: dict) -> dict:
    xx, yy, Z = rec["xx"], rec["yy"], rec["Z"]
    x_axis, y_axis = xx[0, :], yy[:, 0]
    step = max(1, int(math.ceil(max(Z.shape) / GRID_DECIMATE_TARGET)))
    out = {
        "shape": list(Z.shape),
        "x": _flist(x_axis),
        "y": _flist(y_axis),
        "x_min": _f(x_axis[0]), "x_max": _f(x_axis[-1]),
        "y_min": _f(y_axis[0]), "y_max": _f(y_axis[-1]),
        "levels": _flist(rec["levels"]),
        "kwargs": rec["kwargs"],
        "z_min": _f(np.min(Z)), "z_max": _f(np.max(Z)),
        "z_sha256_f64": _sha(Z),
    }
    if np.issubdtype(Z.dtype, np.integer) and Z.max() < 10 and Z.min() >= 0:
        out["z_rows_digits"] = ["".join(str(int(v)) for v in row) for row in Z]
    else:
        out["z_decimation"] = step
        out["z_decimated"] = [_flist(row[::step]) for row in Z[::step]]
    return out


def figure_to_dict(fig) -> dict:
    from matplotlib.collections import PathCollection, PolyCollection, QuadMesh
    from matplotlib.contour import ContourSet
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    fig.canvas.draw()  # resolves colormapped colours (scatter c=..., contour bands)
    axes_out = []
    for ax in fig.axes:
        if not ax.get_visible():
            axes_out.append({"visible": False})
            continue
        d: dict[str, Any] = {
            "title": ax.get_title(),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "xlim": _flist(ax.get_xlim()),
            "ylim": _flist(ax.get_ylim()),
            "is_colorbar": bool(getattr(ax, "_colorbar", None) is not None
                                or ax.get_label() == "<colorbar>"),
        }
        xt = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        yt = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        if xt and not all(_looks_numeric(t) for t in xt):
            d["xticklabels"] = xt
        if yt and not all(_looks_numeric(t) for t in yt):
            d["yticklabels"] = yt
        leg = ax.get_legend()
        if leg is not None:
            d["legend"] = {"title": leg.get_title().get_text(),
                           "labels": [t.get_text() for t in leg.get_texts()]}
        lines = []
        for ln in ax.get_lines():
            if not isinstance(ln, Line2D):
                continue
            x, y = np.asarray(ln.get_xdata(), dtype=float), np.asarray(ln.get_ydata(), dtype=float)
            item = {"label": ln.get_label(), "color": _hex(ln.get_color()),
                    "linestyle": ln.get_linestyle(), "linewidth": _f(ln.get_linewidth()),
                    "alpha": _f(ln.get_alpha()) if ln.get_alpha() is not None else None}
            item["xy"] = _points(np.column_stack([x, y]))
            lines.append(item)
        if lines:
            d["lines"] = lines
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        if rects:
            d["bars"] = {
                "n": len(rects),
                "x": _flist([r.get_x() for r in rects]),
                "y": _flist([r.get_y() for r in rects]),
                "width": _flist([r.get_width() for r in rects]),
                "height": _flist([r.get_height() for r in rects]),
                "facecolor": [_hex(r.get_facecolor()) for r in rects],
                "edgecolor": sorted({_hex(r.get_edgecolor()) for r in rects}),
            }
        cols = []
        for col in ax.collections:
            if isinstance(col, ContourSet):
                rec = next((r for r in _CONTOUR_RECORDS if r["ax"] is ax), None)
                item = {"type": "contourf", "cmap": col.get_cmap().name,
                        "alpha": _f(col.get_alpha()) if col.get_alpha() is not None else None,
                        "band_colors": [_hex(c) for c in col.get_facecolor()]}
                if rec is not None:
                    item["grid"] = _grid(rec)
                cols.append(item)
            elif isinstance(col, PathCollection):
                offs = np.asarray(col.get_offsets(), dtype=float)
                item = {"type": "scatter", "offsets": _points(offs),
                        "sizes": _flist(np.unique(col.get_sizes())),
                        "alpha": _f(col.get_alpha()) if col.get_alpha() is not None else None,
                        "facecolors": _unique_colors(col.get_facecolors()),
                        "edgecolors": _unique_colors(col.get_edgecolors())}
                arr = col.get_array()
                if arr is not None:
                    item["c"] = _points(np.asarray(arr, dtype=float))
                    item["cmap"] = col.get_cmap().name
                    item["norm"] = [_f(col.norm.vmin), _f(col.norm.vmax)]
                cols.append(item)
            elif isinstance(col, QuadMesh):
                arr = np.ma.filled(np.asarray(col.get_array(), dtype=float), np.nan)
                item = {"type": "heatmap", "cmap": col.get_cmap().name,
                        "norm": [_f(col.norm.vmin), _f(col.norm.vmax)],
                        "values": _flist(arr)}
                cols.append(item)
            elif isinstance(col, PolyCollection):
                paths = col.get_paths()
                verts = np.concatenate([p.vertices for p in paths]) if paths else np.zeros((0, 2))
                cols.append({"type": "poly", "n_paths": len(paths),
                             "facecolors": _unique_colors(col.get_facecolors()),
                             "vertices": _points(verts)})
        if cols:
            d["collections"] = cols
        texts = [{"text": t.get_text(), "x": _f(t.get_position()[0]), "y": _f(t.get_position()[1]),
                  "axes_coords": t.get_transform() == ax.transAxes}
                 for t in ax.texts if t.get_text()]
        if texts:
            d["texts"] = texts
        axes_out.append(d)
    w, h = fig.get_size_inches()
    out = {"size_inches": [_f(w), _f(h)], "axes": axes_out}
    if fig._suptitle is not None:
        out["suptitle"] = fig._suptitle.get_text()
    return out


def _looks_numeric(text: str) -> bool:
    try:
        float(text.replace("−", "-"))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Node tree -> JSON
# ---------------------------------------------------------------------------
_DROP_PROPS = {"id", "on_change", "on_click", "data"}
_DATA_LINK = __import__("re").compile(
    r'<a href="data:([^;]+);base64,([^"]+)" download="([^"]+)">(.*?)</a>', __import__("re").S)


def _jsonable(v):
    if isinstance(v, (str, bool)) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return _f(v)
    if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
        return [_jsonable(x) for x in list(v)]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def table_to_dict(df: pd.DataFrame, hide_index: bool) -> dict:
    out = {"columns": [str(c) for c in df.columns],
           "shape": list(df.shape),
           "rows": [[_jsonable(v) for v in row] for row in df.itertuples(index=False, name=None)]}
    if not hide_index:
        out["index"] = [_jsonable(i) for i in df.index.tolist()]
    return out


def node_to_dict(node) -> dict:
    out: dict[str, Any] = {"kind": node.kind}
    props = node.props
    if node.kind == "image":
        if "figure" in props:
            out["figure"] = props["figure"]
    elif node.kind == "table":
        out["table"] = table_to_dict(props["data"], bool(props.get("hide_index")))
        for k in ("height", "hide_index"):
            if props.get(k) is not None:
                out[k] = _jsonable(props[k])
    elif node.kind == "download":
        data = props.get("data") or b""
        out.update({"label": props.get("label"), "file_name": props.get("file_name"),
                    "mime": props.get("mime"), "bytes": len(data)})
        if props.get("mime", "").startswith("text") or str(props.get("file_name", "")).endswith(".py"):
            out["text"] = data.decode("utf-8")
    elif node.kind == "text" and "base64," in str(props.get("text", "")):
        # get_image_download_link / get_code_download_link: <a href="data:...">.
        m = _DATA_LINK.search(str(props["text"]))
        if m is None:
            raise ValueError("unrecognised data: link")
        mime, payload, file_name, label = m.groups()
        out = {"kind": "download_link", "label": label, "file_name": file_name, "mime": mime}
        if mime.startswith("text"):
            import base64

            out["text"] = base64.b64decode(payload).decode("utf-8")
        else:
            out["note"] = "PNG generated by matplotlib at 300 dpi (savefig bbox_inches='tight')"
    else:
        for k, v in props.items():
            if k in _DROP_PROPS or k == "figure":
                continue
            out[k] = _jsonable(v)
    if node.children:
        out["children"] = [node_to_dict(c) for c in node.children]
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
class LessonRenderer:
    """Render a lesson page headlessly and drive its controls.

    ``page`` is the lesson entry point (e.g. ``run_linear_regression_app``).
    Figures passed to ``st.pyplot`` are converted with ``figure_to_dict`` and
    attached to the image node (no PNG is produced).
    """

    def __init__(self, page: Callable[[], None], state: dict | None = None):
        from mltutor.desktop import ui

        self.ui = ui
        self.page = page
        self.session = ui.Session()
        if state:
            self.session.state.update(state)
        self.root = None

    @contextmanager
    def _patched(self):
        import matplotlib.pyplot as plt

        ui = self.ui
        original = ui.pyplot

        def pyplot(fig=None, **kwargs):
            fig = fig if fig is not None else plt.gcf()
            data = figure_to_dict(fig)
            return ui._emit("image", figure=data, caption="", **kwargs)

        ui.pyplot = pyplot
        _CONTOUR_RECORDS.clear()
        try:
            with record_contourf():
                yield
        finally:
            ui.pyplot = original
            plt.close("all")
            _CONTOUR_RECORDS.clear()

    def render(self) -> dict:
        with self._patched():
            self.root, _ = self.session.render(self.page)
        return node_to_dict(self.root)

    # -- finding and driving controls ---------------------------------------
    def walk(self, node=None) -> Iterable:
        node = node if node is not None else self.root
        yield node
        for c in node.children:
            yield from self.walk(c)

    def find(self, kind: str, label: str | None = None, key: str | None = None):
        for n in self.walk():
            if n.kind != kind:
                continue
            if label is not None and n.props.get("label") != label:
                continue
            if key is not None and n.props.get("key") != key:
                continue
            return n
        raise LookupError(f"{kind} label={label!r} key={key!r} not found")

    def click(self, label: str | None = None, key: str | None = None) -> dict:
        if self.root is None:
            self.render()
        node = self.find("button", label, key)
        self.session.clicked = node.props["id"]
        return self.render()

    def set(self, kind: str, value, label: str | None = None, key: str | None = None,
            render: bool = True):
        if self.root is None:
            self.render()
        node = self.find(kind, label, key)
        self.session.values[node.props["id"]] = value
        if node.props.get("key") is not None:
            self.session.state[node.props["key"]] = value
        return self.render() if render else None


def strip_chrome(tree: dict, lesson_tabs_kind: str = "row") -> dict:
    """Drop the lesson header, tab row and navbar, which are covered in phase 1.

    Keeps everything between the first ``---`` after the tab row and the
    ``### 🧭 Navegación`` block.
    """
    children = tree.get("children", [])
    start = 0
    for i, c in enumerate(children):
        if c["kind"] == "row" and len(c.get("children", [])) == 7:
            start = i + 2  # skip the tab row and the '---' separator
            break
    end = len(children)
    for i in range(len(children) - 1, -1, -1):
        c = children[i]
        if c["kind"] == "text" and c.get("text") == "### 🧭 Navegación":
            end = i - 1  # also drop the '---' before it
            break
    return {"kind": "page", "children": children[start:end]}


def texts(tree: dict) -> list[str]:
    """All user-visible strings of a tree, in order (for quick assertions)."""
    out: list[str] = []

    def rec(n):
        for k in ("text", "label", "value"):
            v = n.get(k)
            if isinstance(v, str) and k != "value":
                out.append(v)
        for c in n.get("children", []):
            rec(c)

    rec(tree)
    return out
