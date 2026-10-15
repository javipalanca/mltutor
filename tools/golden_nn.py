"""Phase 5 golden fixtures: the 🧠 Redes Neuronales lesson (fase5-redes.md §6).

Registered by ``tools/export_golden.py`` (``build_cases``), which passes itself as ``eg``.
Everything is produced by the reference app's own code (``run_neural_networks_app`` rendered
headlessly by ``tools/golden_render.py``, ``train_neural_network``, ``neural_network_diagnostics``,
``show_*_tab``), on TensorFlow 2.16 CPU with ``enable_op_determinism`` and one thread.

Parity strategy (fase5-redes.md §6.1, D69):

* ``nn/exact/<case>.json`` — level (a), deterministic. The app trains with ``fit`` wrapped so that
  ``shuffle=False`` (and dropout 0). The fixture stores the initial Glorot weights Keras drew, the
  exact batches (no shuffling: ``validation_split`` keeps the LAST rows), the gradients of the
  first batch, the weights after the first 3 steps, every epoch log, EarlyStopping /
  ReduceLROnPlateau decisions and the final weights. ``tools/nn_reference.py`` (NumPy, float32)
  replays the training from the initial weights and the agreement with Keras is stored epoch by
  epoch: Rust must match within float32 tolerance while the reference does.
  The views (weights, activations, surface, prediction, export) are rendered from the trained
  model, so Rust can load the final weights and reproduce them exactly.
* ``nn/exact/iris_dropout_masks.json`` — dropout with explicit Bernoulli masks (Keras' own dropout
  RNG cannot be reproduced; the formula can).
* ``nn/stat/<case>.json`` — level (b), statistical: the app's real training (dropout 0.2, shuffle)
  with 5 seeds; final metrics and their band.
* ``nn/diagnostics.json`` (pure logic: exact), ``nn/controls.json`` (🏗️/⚙️ texts and defaults,
  architecture code), ``nn/split_stratified.json`` and ``nn/errors.json``.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

MODULE = "nn"
NAV = "🧠 Redes Neuronales"
SEEDS = [0, 1, 2, 3, 4]
BIG = 1000           # above this many rows, arrays go as base64 float32 / hashes
SURFACE_SAMPLE = 400  # grid cells stored with their raw network output
RECORD_STEPS = 3

# Level (a): dropout 0 and shuffle=False. "layers" None = the app's suggested sizes.
EXACT_CASES: dict[str, dict[str, Any]] = {
    "iris_default": {"dataset": "iris", "full": True},
    "iris_sgd_tanh_lr01_b16": {"dataset": "iris", "act": "tanh", "opt": "sgd", "lr": 0.1, "bs": 16,
                               "epochs": 30, "es": False, "rlr": False},
    "iris_rmsprop_sigmoid_1layer": {"dataset": "iris", "layers": [8], "act": "sigmoid",
                                    "opt": "rmsprop", "lr": 0.01, "bs": 32, "epochs": 60,
                                    "patience": 5, "factor": 0.3},
    "iris_tanh_output": {"dataset": "iris", "out": "tanh", "lr": 0.01, "bs": 16, "epochs": 40},
    "iris_relu_3layers_lr03": {"dataset": "iris", "layers": [16, 12, 8], "lr": 0.3, "bs": 16,
                               "epochs": 40, "patience": 5, "views": True},
    "breast_cancer_default": {"dataset": "breast_cancer", "full": True},
    "breast_cancer_sgd_tanh_output": {"dataset": "breast_cancer", "out": "tanh", "opt": "sgd",
                                      "lr": 0.1, "bs": 32, "epochs": 30, "es": False, "rlr": False},
    "wine_default": {"dataset": "wine", "views": True},
    "wine_rmsprop_lr03_chaotic": {"dataset": "wine", "opt": "rmsprop", "lr": 0.3, "bs": 16,
                                  "epochs": 40},
    "diabetes_default": {"dataset": "diabetes", "full": True},
    "diabetes_adam_lr01_b32": {"dataset": "diabetes", "lr": 0.1, "bs": 32, "epochs": 50},
    "diabetes_sgd_lr001_tanh": {"dataset": "diabetes", "act": "tanh", "opt": "sgd", "lr": 0.001,
                                "epochs": 20, "es": False, "rlr": False},
    "diabetes_softmax_output": {"dataset": "diabetes", "out": "softmax", "epochs": 10},
    "diabetes_sigmoid_output": {"dataset": "diabetes", "out": "sigmoid", "lr": 0.01, "epochs": 20},
    "california_housing_default_e20": {"dataset": "california_housing", "epochs": 20, "views": True},
    "moons_default": {"dataset": "moons", "views": True},
    "csv_string_target_default": {"csv": "string_target", "views": True},
    "csv_numeric_regression_default": {"csv": "numeric_regression", "views": True},
}
# Level (b): the app's defaults (dropout 0.2), 5 seeds.
STAT_CASES: dict[str, dict[str, Any]] = {
    "iris_default": {"dataset": "iris"},
    "iris_sgd_lr001": {"dataset": "iris", "opt": "sgd", "lr": 0.01},
    "iris_rmsprop": {"dataset": "iris", "opt": "rmsprop"},
    "breast_cancer_default": {"dataset": "breast_cancer"},
    "wine_default": {"dataset": "wine"},
    "diabetes_default": {"dataset": "diabetes"},
    "diabetes_adam_lr01": {"dataset": "diabetes", "lr": 0.1},
    "california_housing_default": {"dataset": "california_housing"},
    "moons_default": {"dataset": "moons"},
    "csv_string_target_default": {"csv": "string_target"},
    "csv_numeric_regression_default": {"csv": "numeric_regression"},
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_TF_READY = False


def _tf():
    """Import TensorFlow on CPU, deterministic, one thread (and patch the app's pauses)."""
    global _TF_READY
    import keras
    import tensorflow as tf

    if not _TF_READY:
        tf.config.set_visible_devices([], "GPU")
        tf.config.experimental.enable_op_determinism()
        from mltutor.algorithms import model_training as mt

        mt.time.sleep = lambda _s: None  # 0.8 + 0.8 + 1.0 s of artificial pauses
        _TF_READY = True
    return tf, keras


def tf_versions() -> dict[str, str]:
    tf, keras = _tf()
    import importlib.metadata as md

    out = {"tensorflow": tf.__version__, "keras": keras.__version__,
           "tf_devices": [d.device_type for d in tf.config.get_visible_devices()],
           "op_determinism": True, "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads()}
    for pkg in ("tensorflow-macos", "tensorflow-metal", "onnx"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _fl(values) -> list:
    return [_f(v) for v in np.asarray(values, dtype=float).ravel().tolist()]


def _mat(a) -> list:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return _fl(a)
    return [_fl(r) for r in a]


def _sha(a, dtype="<f4") -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=dtype)).tobytes()).hexdigest()


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dtype="<f4")).tobytes()).decode()


def _array_block(a, dtype="<f4") -> dict:
    a = np.asarray(a)
    out = {"shape": list(a.shape), f"sha256_{dtype[1:]}": _sha(a, dtype)}
    if a.shape[0] <= BIG:
        out["values"] = _mat(a.astype(float))
    else:
        out["base64_f32_le"] = _b64(a)
    return out


def _weights(ws) -> list[dict]:
    return [{"kernel": _mat(ws[i]), "bias": _fl(ws[i + 1])} for i in range(0, len(ws), 2)]


def _pairs(ws):
    return [(np.asarray(ws[i], np.float32), np.asarray(ws[i + 1], np.float32))
            for i in range(0, len(ws), 2)]


def _maxdiff(a, b) -> float:
    return float(max(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float))) for x, y in zip(a, b)))


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?")
_TMP = re.compile(r"/(private/)?(var|tmp)/[^\s'\"]+")


def _scrub(tree):
    """Remove what changes between runs: dates, temp paths, sizes of binary downloads."""
    if isinstance(tree, dict):
        out = {}
        for k, v in tree.items():
            if k == "bytes" and tree.get("kind") == "download" and "text" not in tree:
                out["bytes_nonzero"] = bool(v)
                continue
            out[k] = _scrub(v)
        return out
    if isinstance(tree, list):
        return [_scrub(v) for v in tree]
    if isinstance(tree, str):
        if "Graph execution error" in tree:
            # TF call stacks depend on the caller; keep the first and the last message lines.
            lines = [ln for ln in tree.splitlines() if ln.strip() and not ln.startswith("  File ")]
            tree = "\n".join([lines[0], "<pila de llamadas>"] + lines[-2:])
            tree = re.sub(r"\[Op:__inference_\w+?_\d+\]", "[Op:__inference_<n>]", tree)
        return _TMP.sub("<tmp>", _DATE.sub("<fecha>", tree))
    return tree


# ---------------------------------------------------------------------------
# Plotly and HTML canvases: compact summaries instead of 30 KB of HTML
# ---------------------------------------------------------------------------
def _plotly_summary(fig) -> dict:
    from golden_render import _points

    js = fig.to_plotly_json()
    traces = []
    for t in js.get("data", []):
        item = {k: _jsonable(t[k]) for k in ("type", "name", "mode", "showlegend", "nbinsx", "opacity",
                                              "xaxis", "yaxis", "colorscale", "showscale")
                if k in t}
        for k in ("marker", "line"):
            if k in t:
                item[k] = {kk: _jsonable(vv) for kk, vv in t[k].items()
                           if not isinstance(vv, (list, np.ndarray)) or kk == "color"}
        for axis in ("x", "y", "z", "text"):
            if axis in t and t[axis] is not None:
                arr = np.asarray(t[axis])
                if arr.dtype.kind in "fiub":
                    item[axis] = _points(arr.astype(float))
                else:
                    item[axis] = _jsonable(arr.tolist())
        if "annotation_text" in t:
            item["annotation_text"] = _jsonable(t["annotation_text"])
        traces.append(item)
    lay = js.get("layout", {})
    layout: dict[str, Any] = {}
    for k, v in lay.items():
        if k == "template":
            continue
        if k.startswith(("xaxis", "yaxis")):
            layout[k] = {kk: _jsonable(vv) for kk, vv in v.items()
                         if kk in ("title", "type", "side", "overlaying", "anchor", "domain")}
        elif k in ("title", "height", "showlegend", "bargap", "barmode", "margin", "annotations",
                   "shapes"):
            layout[k] = _jsonable(v)
    return {"traces": traces, "layout": layout}


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in list(v)]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return _f(v)
    if isinstance(v, (str, bool, int)) or v is None:
        return v
    return str(v)


_JS_CONST = re.compile(r"const (\w+) = (.+?);\s*(?://.*)?$", re.M)


def _summarise_html(tree):
    """Canvases of 🏗️ (create_neural_network_visualization) and 📉 (dead-neuron map)."""
    if isinstance(tree, dict):
        if tree.get("kind") == "html":
            html = tree.get("html", "")
            out = {"kind": "html", "height": tree.get("height"), "bytes": len(html.encode()),
                   "sha256": hashlib.sha256(html.encode()).hexdigest()}
            consts = {}
            for name, expr in _JS_CONST.findall(html):
                if name in ("architecture", "arch", "dead", "constantN", "jitterStrength",
                            "connectionSample", "adaptiveRadius", "groupLarge", "minSpacing",
                            "maxCols", "neuronInfo"):
                    try:
                        consts[name] = json.loads(expr.replace("true", "true"))
                    except json.JSONDecodeError:
                        consts[name] = expr.strip()
            out["js_constants"] = consts
            colors = sorted(set(re.findall(r"#[0-9A-Fa-f]{6}\b", html)))
            out["colors"] = colors
            vis = re.sub(r"<script>.*?</script>|<style>.*?</style>", "", html, flags=re.S)
            out["visible_texts"] = [t for t in (s.strip() for s in re.sub(
                r"<[^>]+>", "\n", vis).split("\n")) if t]
            return out
        return {k: _summarise_html(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_summarise_html(v) for v in tree]
    return tree


@contextlib.contextmanager
def plotly_as_data():
    from mltutor.desktop import ui

    original = ui.plotly_chart

    def plotly_chart(figure, **kwargs):
        return ui._emit("plotly", chart=_plotly_summary(figure))

    ui.plotly_chart = plotly_chart
    try:
        yield
    finally:
        ui.plotly_chart = original


def view(tree) -> dict:
    from golden_render import strip_chrome

    return _scrub(_summarise_html(strip_chrome(tree)))


# ---------------------------------------------------------------------------
# Training probe: wraps keras.Model.fit inside the app's train_neural_network
# ---------------------------------------------------------------------------
class _Probe:
    def __init__(self, exact: bool):
        self.exact = exact
        self.calls: list[dict] = []

    @contextlib.contextmanager
    def patch(self):
        tf, keras = _tf()
        probe = self
        original = keras.Model.fit

        class Recorder(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.steps, self.batch_logs, self.epoch = [], [], 0

            def on_train_begin(self, logs=None):
                self.w0 = [w.copy() for w in self.model.get_weights()]

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch

            def on_train_batch_end(self, batch, logs=None):
                if self.epoch == 0:
                    self.batch_logs.append({k: float(v) for k, v in (logs or {}).items()})
                    if batch < RECORD_STEPS:
                        self.steps.append([w.copy() for w in self.model.get_weights()])

        def fit(model, *args, **kwargs):
            rec = Recorder()
            cbs = list(kwargs.get("callbacks") or [])
            if probe.exact:
                kwargs["shuffle"] = False
            kwargs["callbacks"] = cbs + [rec]
            call = {"x": np.asarray(args[0]), "y": np.asarray(args[1]), "kwargs": {
                k: v for k, v in kwargs.items() if k in ("epochs", "batch_size", "validation_split",
                                                         "shuffle")},
                "callbacks": cbs, "recorder": rec, "model": model}
            probe.calls.append(call)
            hist = original(model, *args, **kwargs)
            call["history"] = hist
            return hist

        keras.Model.fit = fit
        try:
            yield self
        finally:
            keras.Model.fit = original


# ---------------------------------------------------------------------------
# Driving the lesson like the student
# ---------------------------------------------------------------------------
def _setup(eg, spec: dict, tmp: Path):
    state, label, ref = eg._lr_setup({**spec, "model": "Linear"}, tmp)
    for k in ("navigation", "active_tab_lr", "model_type_lr"):
        state.pop(k, None)
    state.update({"navigation": NAV, "active_tab_nn": 1, "selected_dataset": label,
                  "unified_dataset_selector": label})
    if spec.get("dataset") == "moons":
        # The app caches Moons in session_state, drawn with a timestamp seed; seed it (D9).
        from sklearn.datasets import make_moons

        state["moons_X"], state["moons_y"] = make_moons(**eg.MOONS_PARAMS)
    return state, label, ref


def _load(eg, label: str):
    """load_data with Moons seeded like the fixtures."""
    from mltutor.dataset.dataset_manager import load_data
    from mltutor.desktop import ui

    s = ui.Session()
    if "Moons" in label:
        from sklearn.datasets import make_moons

        s.state["moons_X"], s.state["moons_y"] = make_moons(**eg.MOONS_PARAMS)
    with ui.bind(s):
        return load_data(label)


def _page():
    from mltutor.apps.neural_network_app import run_neural_networks_app

    return run_neural_networks_app


def _defaults(spec: dict) -> dict:
    return {"act": spec.get("act", "relu"), "out": spec.get("out"), "dropout": spec.get("dropout", 0.0),
            "bs": spec.get("bs", 64), "opt": spec.get("opt", "adam"), "lr": spec.get("lr", 0.001),
            "epochs": spec.get("epochs"), "val": spec.get("val", 20), "es": spec.get("es", True),
            "patience": spec.get("patience", 10), "rlr": spec.get("rlr", True),
            "factor": spec.get("factor", 0.5), "layers": spec.get("layers")}


def configure_and_train(eg, spec: dict, tmp: Path, probe: _Probe, views: dict | None,
                        dropout_default: bool = False):
    """Arquitectura → Entrenamiento → train, as the student does. Returns the renderer."""
    from golden_render import LessonRenderer

    c = _defaults(spec)
    state, label, ref = _setup(eg, spec, tmp)
    r = LessonRenderer(_page(), state)
    first = r.render()
    if views is not None:
        views["arquitectura_default"] = view(first)
    if c["layers"] is not None:
        r.set("slider", len(c["layers"]), label="Número de capas ocultas")
        for i, n in enumerate(c["layers"]):
            r.set("slider", n, key=f"layer_{i}", render=False)
    r.set("select", c["act"], label="Función de activación", render=False)
    if c["out"] is not None:
        r.set("select", c["out"], label="Función de activación de salida", render=False)
    if not dropout_default:
        r.set("slider", c["dropout"], label="Tasa de Dropout", render=False)
    r.set("select", c["bs"], label="Tamaño de Batch", render=False)
    arch = r.set("select", c["opt"], label="Optimizador")
    if views is not None:
        views["arquitectura"] = view(arch)
        views["arquitectura_codigo"] = view(r.click("📝 Generar Código de la Arquitectura"))
    r.session.state["active_tab_nn"] = 2
    train_tab = r.render()
    if views is not None:
        views["entrenamiento_default"] = view(train_tab)
    r.set("select", c["lr"], label="Tasa de Aprendizaje", render=False)
    if c["epochs"] is not None:
        r.set("slider", c["epochs"], label="Épocas", render=False)
    r.set("slider", c["val"], label="% Datos de Validación", render=False)
    r.set("checkbox", c["es"], label="Activar Parada Temprana")
    if c["es"]:
        r.set("slider", c["patience"], label="Paciencia (épocas)", render=False)
    r.set("checkbox", c["rlr"], label="Reducir Learning Rate Automáticamente")
    if c["rlr"]:
        r.set("slider", c["factor"], label="Factor de Reducción", render=False)
    before = r.render()
    with probe.patch():
        after = r.click("🧠 Entrenar Red Neuronal")
    if views is not None:
        views["entrenamiento_antes"] = view(before)
        views["entrenamiento_despues"] = view(after)
    return r, label, ref, after


def _controls_used(r) -> dict:
    st = r.session.state
    cfg = dict(st["nn_config"])
    return {k: _jsonable(v) for k, v in cfg.items()}


# ---------------------------------------------------------------------------
# Level (a) numbers
# ---------------------------------------------------------------------------
def _split_block(y_codes: np.ndarray, task: str) -> dict:
    """Split of the app (train_test_split on the positions) and the reference check."""
    import nn_reference as ref
    from sklearn.model_selection import train_test_split

    n = len(y_codes)
    pos = np.arange(n)
    if task == "Clasificación":
        tr, te = train_test_split(pos, test_size=0.2, random_state=42, stratify=y_codes)
        rtr, rte = ref.stratified_split(y_codes, 0.2, 42)
        kind = "stratified"
    else:
        tr, te = train_test_split(pos, test_size=0.2, random_state=42)
        rtr, rte = ref.plain_split(n, 0.2, 42)
        kind = "plain (D7, split.rs)"
    out = {"kind": kind, "test_size": 0.2, "random_state": 42, "n_train": int(len(tr)),
           "n_test": int(len(te)),
           "reference_matches": bool(np.array_equal(tr, rtr) and np.array_equal(te, rte))}
    if n <= BIG:
        out["train_indices"] = tr.tolist()
        out["test_indices"] = te.tolist()
    else:
        out["train_indices_sha256_i64"] = _sha(tr, "<i8")
        out["test_indices_sha256_i64"] = _sha(te, "<i8")
        out["train_indices_head"] = tr[:20].tolist()
        out["test_indices_head"] = te[:20].tolist()
    return out, tr, te


def _keras_gradients(model, w0, xb, yb):
    tf, keras = _tf()
    m2 = keras.models.clone_model(model)
    m2.set_weights(w0)
    xb = tf.constant(xb.astype(np.float32))
    yb = tf.constant(np.asarray(yb, np.float32))

    @tf.function
    def g():
        with tf.GradientTape() as tape:
            pred = m2(xb, training=True)
            loss = model.compute_loss(x=xb, y=yb, y_pred=pred)
        return loss, tape.gradient(loss, m2.trainable_weights)

    loss, grads = g()
    return float(loss), [np.asarray(gr) for gr in grads]


def exact_numbers(call: dict, task: str, spec_c: dict, cfg: dict) -> dict:
    import nn_reference as ref

    model, rec, hist = call["model"], call["recorder"], call["history"]
    x, y = call["x"].astype(np.float32), np.asarray(call["y"], np.float32)
    bs = call["kwargs"]["batch_size"]
    v = call["kwargs"]["validation_split"]
    split_at = ref.keras_validation_split(len(x), v)
    res = ref.resolve_training(task, cfg["output_activation"], cfg["output_size"])
    metric_obj = [m for m in model.metrics if type(m).__name__ == "CompileMetrics"]
    resolved = []
    if metric_obj:
        resolved = [type(m).__name__ for m in metric_obj[0].metrics]
    w0 = rec.w0
    yb = y[:bs] if y.ndim > 1 else y[:bs, None]
    kl, kg = _keras_gradients(model, w0, x[:min(bs, split_at)], yb[:min(bs, split_at)])
    rl, rg = ref.gradients(_pairs(w0), x[:min(bs, split_at)], yb[:min(bs, split_at)],
                           cfg["activation"], cfg["output_activation"], res["loss"])
    rg_flat = [g for pair in rg for g in pair]
    es = next((c for c in call["callbacks"] if type(c).__name__ == "EarlyStopping"), None)
    rlr = next((c for c in call["callbacks"] if type(c).__name__ == "ReduceLROnPlateau"), None)
    r = ref.train(_pairs(w0), x, y, hidden=cfg["activation"], output=cfg["output_activation"],
                  loss=res["loss"], metric=res["metric"], optimizer=cfg["optimizer"],
                  lr=spec_c["lr"], batch_size=bs, epochs=call["kwargs"]["epochs"],
                  validation_split=v, early_stopping=spec_c["es"],
                  patience=spec_c["patience"] if spec_c["es"] else None,
                  reduce_lr=spec_c["rlr"], lr_factor=spec_c["factor"] if spec_c["rlr"] else None)
    hk = {k: [float(a) for a in vals] for k, vals in hist.history.items()}
    hr = r["history"]
    agree = {}
    agree_epochs = None
    for k, vals in hk.items():
        a = np.asarray(vals, float)
        b = np.asarray(hr.get(k, [math.nan] * len(a))[:len(a)], float)
        if len(b) < len(a):
            b = np.concatenate([b, np.full(len(a) - len(b), math.nan)])
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-6)
        agree[k] = _fl(rel)
        if k in ("loss", "val_loss"):
            bad = np.where(~(rel <= 1e-3))[0]
            first_bad = int(bad[0]) if len(bad) else len(a)
            agree_epochs = first_bad if agree_epochs is None else min(agree_epochs, first_bad)
    wf = model.get_weights()
    rf = [w for pair in r["params"] for w in pair]
    steps_ref = [[w for pair in s["params"] for w in pair] for s in r["steps"]]
    # Validation margins of the callbacks (how close each decision was).
    vl = np.asarray(hk["val_loss"], float)
    best_so_far = np.minimum.accumulate(vl)
    margins = _fl(np.abs(vl[1:] - best_so_far[:-1]) / np.maximum(np.abs(best_so_far[:-1]), 1e-12)) \
        if len(vl) > 1 else []
    return {
        "data": {"x_fit": _array_block(x), "y_fit": _array_block(y),
                 "validation_split": v, "split_at": split_at, "n_val": int(len(x) - split_at),
                 "batch_size": bs, "n_batches": int(math.ceil(split_at / bs)),
                 "last_batch": int(split_at - (math.ceil(split_at / bs) - 1) * bs),
                 "shuffle": False},
        "compile": {"loss": model.loss, "metrics_resolved": resolved, "reference": res,
                    "optimizer": type(model.optimizer).__name__,
                    "optimizer_config": {k: _jsonable(v) for k, v in model.optimizer.get_config().items()
                                         if k in ("learning_rate", "beta_1", "beta_2", "epsilon", "rho",
                                                  "momentum", "nesterov", "centered", "amsgrad")}},
        "initial_weights": _weights(w0),
        "first_batch": {"rows": int(min(bs, split_at)), "keras_loss": kl, "reference_loss": rl,
                        "keras_gradients": _weights(kg), "max_abs_diff_reference": _maxdiff(kg, rg_flat)},
        "steps": [{"step": i + 1, "weights": _weights(s),
                   "max_abs_diff_reference": _maxdiff(s, steps_ref[i]) if i < len(steps_ref) else None}
                  for i, s in enumerate(rec.steps)],
        "epoch0_batch_logs_running_mean": rec.batch_logs[:10],
        "history": hk,
        "callbacks": {
            "early_stopping": None if es is None else {
                "patience": es.patience, "min_delta": float(abs(es.min_delta)), "best": _f(es.best),
                "best_epoch": int(es.best_epoch), "stopped_epoch": int(es.stopped_epoch),
                "restore_best_weights": es.restore_best_weights,
                "restored_at_train_end": True},
            "reduce_lr": None if rlr is None else {
                "patience": rlr.patience, "factor": rlr.factor, "min_lr": rlr.min_lr,
                "min_delta": float(abs(rlr.min_delta)), "cooldown": rlr.cooldown},
            "val_loss_relative_margins": margins,
        },
        "final_weights": _weights(wf),
        "reference": {"epochs": len(hr["loss"]), "best_epoch": r["best_epoch"],
                      "stopped_epoch": r["stopped_epoch"], "history": {k: _fl(vv) for k, vv in hr.items()},
                      "relative_diff_by_epoch": agree, "agree_epochs_rel_1e-3": agree_epochs,
                      "final_weights_max_abs_diff": _maxdiff(wf, rf) if len(wf) == len(rf) else None,
                      "same_epochs": len(hr["loss"]) == len(hk["loss"])},
    }


def test_block(r, task: str) -> dict:
    """Predictions on X_test (what 📈 Evaluación uses)."""
    st = r.session.state
    model = st["nn_model"]
    X_test, y_test = st["nn_test_data"]
    raw = model.predict(X_test, verbose=0)
    ev = model.evaluate(X_test, y_test, verbose=0)
    ev = ev if isinstance(ev, list) else [ev]
    out = {"x_test": _array_block(np.asarray(X_test, np.float32)), "y_test": _mat(np.asarray(y_test, float)),
           "raw_output": _mat(raw), "evaluate": _fl(ev),
           "evaluate_names": list(model.metrics_names)}
    if task == "Clasificación":
        yt = np.asarray(y_test)
        out["y_pred"] = (np.argmax(raw, 1) if raw.shape[1] > 1 else (raw.ravel() > 0.5).astype(int)).tolist()
        out["y_true"] = (np.argmax(yt, 1) if yt.ndim > 1 and yt.shape[1] > 1 else yt.ravel().astype(int)).tolist()
    return out


# ---------------------------------------------------------------------------
# Views of the trained network (and the numbers behind them)
# ---------------------------------------------------------------------------
def weights_numbers(model) -> dict:
    import nn_reference as ref

    ws = model.get_weights()
    names = [ly.name for ly in model.layers]
    dense_idx = [i for i, ly in enumerate(model.layers) if ly.get_weights()]
    out = ref.weight_stats(_pairs(ws))
    out["model_layers"] = [{"index": i, "name": ly.name, "class": type(ly).__name__} for i, ly in enumerate(model.layers)]
    out["dense_layer_indices"] = dense_idx
    out["labels_python"] = {"caption_and_table": [f"Capa {i + 1}" for i in dense_idx],
                            "sections": [f"Capa {k + 1}" for k in range(len(dense_idx))]}
    out["names"] = names
    return out


def activations_numbers(model, X_test, act: str, sample_size=50, dead=0.0, const=1e-3) -> dict:
    import nn_reference as ref
    tf, keras = _tf()

    n = min(sample_size, len(X_test))
    xs = np.asarray(X_test[:n], np.float32)
    hidden = [(i, ly) for i, ly in enumerate(model.layers[:-1]) if hasattr(ly, "units") and i > 0]
    keras_stats = []
    for i, ly in hidden:
        acts = keras.Model(inputs=model.input, outputs=ly.output).predict(xs, verbose=0)
        deadm = (acts <= dead).all(axis=0)
        constm = (~deadm) & (acts.std(axis=0) < const)
        keras_stats.append({"layer_index": i, "label_python": f"Capa {i + 1}", "name": ly.name,
                            "units": int(ly.units), "dead": np.where(deadm)[0].tolist(),
                            "constant": np.where(constm)[0].tolist(),
                            "saturation_pct": float((acts >= 0.99).mean() * 100) if acts.max() <= 1.2 else 0.0,
                            "mean_act": float(acts.mean(axis=0).mean()),
                            "std_act": float(acts.std(axis=0).mean()),
                            "act_max": float(acts.max()), "histogram_values": _array_block(acts.ravel()[:, None])})
    refs = ref.activation_stats(_pairs(model.get_weights()), xs, act, dead, const)
    same = all(a["dead"] == b["dead"] and a["constant"] == b["constant"] for a, b in zip(keras_stats, refs))
    total = sum(s["units"] for s in keras_stats)
    dead_n = sum(len(s["dead"]) for s in keras_stats)
    return {"params": {"sample_size": sample_size, "rows_used": n, "dead_threshold": dead,
                       "constant_std": const}, "layers": keras_stats, "reference_same_sets": same,
            "reference_mean_act_max_abs_diff": float(max(abs(a["mean_act"] - b["mean_act"])
                                                         for a, b in zip(keras_stats, refs))) if refs else 0.0,
            "dead_pct": dead_n / total * 100 if total else 0.0}


def surface_numbers(model, X_test, y_test, task: str, pair=(0, 1)) -> dict:
    """show_decision_surface_tab: grid in the SCALED space, the other features at their mean."""
    X_test = np.asarray(X_test)
    X2 = X_test[:, list(pair)]
    h = 0.02
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xs, ys = np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    mean = np.mean(X_test, axis=0)
    mesh = np.tile(mean, (xx.size, 1))
    mesh[:, pair[0]] = xx.ravel()
    mesh[:, pair[1]] = yy.ravel()
    raw = model.predict(mesh, verbose=0)
    if task == "Clasificación":
        z = np.argmax(raw, 1) if raw.shape[1] > 1 else (raw > 0.5).astype(int).ravel()
    else:
        z = raw.ravel() if raw.shape[1] == 1 else raw[:, 0]
    rng = np.random.RandomState(42)
    cells = np.sort(rng.choice(xx.size, size=min(SURFACE_SAMPLE, xx.size), replace=False))
    out = {"pair": list(pair), "h": h, "x_axis": {"start": float(xs[0]), "count": int(len(xs)),
                                                    "last": float(xs[-1])},
           "y_axis": {"start": float(ys[0]), "count": int(len(ys)), "last": float(ys[-1])},
           "cells": int(xx.size), "mean_values": _fl(mean),
           "sample": {"index": cells.tolist(), "x": _fl(xx.ravel()[cells]), "y": _fl(yy.ravel()[cells]),
                      "raw_output": _mat(raw[cells]), "z": _fl(z[cells])},
           "z_sha256": _sha(z, "<i8" if task == "Clasificación" else "<f4")}
    if task == "Clasificación":
        vals, counts = np.unique(z, return_counts=True)
        out["z_counts"] = {str(int(a)): int(b) for a, b in zip(vals, counts)}
        out["y_plot"] = (np.argmax(y_test, 1) if np.asarray(y_test).ndim > 1 else np.asarray(y_test)).tolist() \
            if len(X_test) <= BIG else None
    return out


def prediction_numbers(r) -> dict:
    """The 🔮 form with its default values: float32 input -> scaler -> network."""
    st = r.session.state
    model, scaler = st["nn_model"], st["nn_scaler"]
    feats = [k for k in st["nn_df"].columns if k != st["nn_target_col"]]
    vals = []
    for i in range(len(feats)):
        v = st.get(f"feature_{i}")
        vals.append(v)
    x = np.array([vals], dtype=np.float32)
    xs = scaler.transform(x)
    raw = model.predict(xs, verbose=0)
    return {"input_float32": _fl(x[0]), "scaled": _fl(xs[0]), "raw_output": _mat(raw),
            "scaler_mean": _fl(scaler.mean_), "scaler_scale": _fl(scaler.scale_)}


def render_views(r, task: str, full: bool, views: dict, numbers: dict) -> None:
    st = r.session.state
    model = st["nn_model"]
    X_test, y_test = st["nn_test_data"]
    cfg = st["nn_config"]
    st["active_tab_nn"] = 3
    views["evaluacion"] = view(r.render())
    st["active_tab_nn"] = 4
    for vt in ("Historial", "Pesos", "Superficie", "Activaciones"):
        st["viz_type"] = vt
        views[f"visualizacion_{vt.lower()}"] = view(r.render())
    numbers["weights"] = weights_numbers(model)
    numbers["activations_default"] = activations_numbers(model, X_test, cfg["activation"])
    numbers["surface_default"] = surface_numbers(model, X_test, y_test, task)
    if full:
        st["viz_type"] = "Pesos"
        r.render()
        r.set("slider", 20, key="bins_weights", render=False)
        r.set("slider", 10, key="bins_biases", render=False)
        views["visualizacion_pesos_bins20_log"] = view(r.set("checkbox", True, label="Escala log en frecuencia"))
        st["viz_type"] = "Activaciones"
        r.render()
        r.set("slider", 200, label="Muestras", render=False)
        r.set("number", 0.05, label="Umbral muerta (<=)", render=False)
        r.set("number", 0.01, label="Std constante <", render=False)
        r.set("slider", 0, label="Dispersión nodos", render=False)
        r.set("slider", 100, label="% conexiones dibujadas", render=False)
        r.set("checkbox", False, label="Radio adaptativo", render=False)
        r.set("checkbox", False, label="Agrupar capas grandes", render=False)
        r.set("slider", 30, label="Espaciado mínimo vertical", render=False)
        r.set("slider", 2, label="Máx subcolumnas capa")
        for node in list(r.walk()):
            if node.kind == "checkbox" and str(node.props.get("key", "")).startswith(("hist_act_", "means_")):
                r.set("checkbox", True, key=node.props["key"], render=False)
        views["visualizacion_activaciones_opciones"] = view(r.render())
        numbers["activations_options"] = activations_numbers(model, X_test, cfg["activation"],
                                                             200, 0.05, 0.01)
        if cfg["input_size"] > 2:
            st["viz_type"] = "Superficie"
            r.render()
            k1, k2 = ("viz_f1", "viz_f2") if task == "Clasificación" else ("viz_f1_reg", "viz_f2_reg")
            names = st["nn_feature_names"]
            pair = (2, 3) if len(names) > 3 else (1, 0)
            r.set("select", names[pair[0]], key=k1, render=False)
            views["visualizacion_superficie_par"] = view(r.set("select", names[pair[1]], key=k2))
            numbers["surface_pair"] = surface_numbers(model, X_test, y_test, task, pair)
            views["visualizacion_superficie_misma"] = view(r.set("select", names[pair[0]], key=k2))
    st["active_tab_nn"] = 5
    views["predicciones_formulario"] = view(r.render())
    views["predicciones_resultado"] = view(r.click("Realizar predicción"))
    numbers["prediction_form"] = prediction_numbers(r)
    st["active_tab_nn"] = 6
    for vt in ("Tensorflow", "Completo", "Python", "Metadatos"):
        st["viz_type"] = vt
        views[f"exportar_{vt.lower()}"] = view(r.render())
    if full:
        st["viz_type"] = "Tensorflow"
        r.render()
        for fmt, name in (("SavedModel (.pb)", "savedmodel"), ("HDF5 (.h5)", "h5"),
                          ("TensorFlow Lite (.tflite)", "tflite")):
            r.set("radio", fmt, key="nn_export_format")
            views[f"exportar_tf_{name}"] = view(r.click("💾 Exportar Modelo TensorFlow"))
        st["viz_type"] = "Completo"
        r.render()
        views["exportar_completo_click"] = view(r.click("💾 Exportar Modelo Completo"))
        st["viz_type"] = "Python"
        r.render()
        views["exportar_python_click"] = view(r.click("📝 Generar Código"))


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def _meta(eg, tolerance: dict, rationale: str) -> dict:
    return eg.make_meta(tolerance, rationale, {"tensorflow": tf_versions()})


EXACT_TOL = {
    "split_indices": "exact",
    "scaler_mean_scale": "1e-12 relative",
    "x_fit": "f32: 1 ulp (sklearn scales in f64, Keras casts to f32)",
    "first_batch_gradients": "1e-6 absolute (f32 matmul order)",
    "weights_after_steps_1_3": "1e-6 absolute",
    "history_epochs_before_divergence": "1e-4 relative while reference.relative_diff_by_epoch <= 1e-5; "
                                        "only the first reference.agree_epochs_rel_1e-3 epochs are comparable",
    "epochs_trained_and_best_epoch": "exact unless callbacks.val_loss_relative_margins[i] < 1e-5",
    "final_weights": "1e-4 absolute when reference.final_weights_max_abs_diff < 1e-4 (else trajectory is chaotic: skip)",
    "views_from_final_weights": "f32 forward: raw outputs 1e-5 absolute, classes exact except |p - 0.5| < 1e-5 or argmax ties",
    "texts": "exact (dates replaced by <fecha>)",
}


def case_exact(eg, name: str, spec: dict, writer) -> None:
    import golden_trees as gt

    full = spec.get("full", False)
    want_views = full or spec.get("views", False)
    views: dict[str, Any] = {}
    numbers: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(), plotly_as_data():
        warnings.simplefilter("ignore")
        tf, keras = _tf()
        keras.utils.set_random_seed(42)
        probe = _Probe(exact=True)
        r, label, ref_in, _after = configure_and_train(eg, spec, Path(tmpdir), probe,
                                                       views if want_views else None)
        st = r.session.state
        if not probe.calls or st.get("nn_model") is None:
            raise RuntimeError(f"training failed: {[n for n in r.walk() if n.kind == 'error']}")
        cfg = st["nn_config"]
        task = cfg["task_type"]
        call = probe.calls[-1]
        c = _defaults(spec)
        y_codes = np.asarray(st["nn_df"][st["nn_target_col"]])
        split, tr, te = _split_block(y_codes, task)
        X_all = st["nn_df"].drop(columns=[st["nn_target_col"]]).to_numpy(float)
        import nn_reference as nref
        _xs, mean, scale = nref.standard_scaler(X_all)
        sc = st["nn_scaler"]
        numbers["split"] = split
        numbers["scaler"] = {"fit_on": "all rows (before the split)", "mean": _fl(sc.mean_),
                             "scale": _fl(sc.scale_),
                             "reference_max_rel_diff": float(max(
                                 np.max(np.abs(mean - sc.mean_) / np.maximum(np.abs(sc.mean_), 1e-300)),
                                 np.max(np.abs(scale - sc.scale_) / sc.scale_)))}
        x_fit_ok = np.array_equal(call["x"], (X_all[tr] - sc.mean_) / sc.scale_) or \
            np.allclose(call["x"], (X_all[tr] - sc.mean_) / sc.scale_, rtol=0, atol=1e-12)
        numbers["split"]["x_fit_equals_scaled_train_rows"] = bool(x_fit_ok)
        numbers.update(exact_numbers(call, task, c, cfg))
        numbers["test"] = test_block(r, task)
        if want_views:
            render_views(r, task, full, views, numbers)
        X_raw_test = X_all[te]
        numbers["onnx_proposal_D75"] = onnx_proposal(st["nn_model"], sc, task, X_raw_test, cfg)
        if full or spec.get("views"):
            numbers["export_package"] = export_package(eg, st["nn_model"], sc, cfg, st, X_raw_test[:5],
                                                       len(X_all), run=full)
        numbers["diagnostics"] = diagnostics_numbers(SimpleNamespace(history=call["history"].history), cfg)
    doc = {
        "meta": _meta(eg, EXACT_TOL,
                      "Level (a) of D69: same initial weights, no shuffling, no dropout. Keras and "
                      "tools/nn_reference.py (NumPy f32) agree within float32 rounding while the "
                      "trajectory is not chaotic; reference.* says how long they agree."),
        "input": {**ref_in, "app_label": label, "controls": c, "nn_config": _controls_used(r),
                  "fit_kwargs": {k: _jsonable(v) for k, v in call["kwargs"].items()},
                  "function": "mltutor.apps.neural_network_app.run_neural_networks_app (rendered by "
                              "tools/golden_render.py), keras.Model.fit wrapped with shuffle=False",
                  "seed": "keras.utils.set_random_seed(42) before training (the app sets none)"},
        "expected": {**numbers, "views": views},
    }
    writer.write_json(eg.PARITY_DIR / MODULE / "exact" / f"{name}.json", doc, dump=gt.dumps_compact)


def case_dropout_masks(eg, writer) -> None:
    """Dropout with explicit masks: Keras layers + optimizer, masks from RandomState(42)."""
    import golden_trees as gt
    import nn_reference as nref
    tf, keras = _tf()
    from sklearn.datasets import load_iris

    keras.utils.set_random_seed(42)
    X, y = load_iris(return_X_y=True)
    Xs, mean, scale = nref.standard_scaler(X)
    tr, te = nref.stratified_split(y, 0.2, 42)
    rate, bs, steps, lr = 0.5, 16, 5, 0.01
    xb_all, yb_all = Xs[tr].astype(np.float32), np.eye(3, dtype=np.float32)[y[tr]]
    inp = keras.layers.Input(shape=(4,))
    d1 = keras.layers.Dense(10, activation="relu")
    d2 = keras.layers.Dense(10, activation="relu")
    do = keras.layers.Dense(3, activation="softmax")
    model = keras.Model(inp, do(d2(d1(inp))))
    w0 = [w.copy() for w in model.get_weights()]
    opt = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.CategoricalCrossentropy()
    rng = np.random.RandomState(42)
    log = []
    params = _pairs(w0)
    ref_opt = nref.Optimizer("adam", np.float32(lr))
    for s in range(steps):
        xb, yb = xb_all[s * bs:(s + 1) * bs], yb_all[s * bs:(s + 1) * bs]
        m1 = (rng.uniform(size=(len(xb), 10)) >= rate).astype(np.float32)
        m2 = (rng.uniform(size=(len(xb), 10)) >= rate).astype(np.float32)

        @tf.function
        def step(xb, yb, m1, m2):
            with tf.GradientTape() as tape:
                h = d1(xb) * m1 / (1 - rate)
                h = d2(h) * m2 / (1 - rate)
                p = do(h)
                loss = loss_fn(yb, p)
            g = tape.gradient(loss, model.trainable_weights)
            opt.apply(g, model.trainable_weights)
            return loss

        kl = float(step(tf.constant(xb), tf.constant(yb), tf.constant(m1), tf.constant(m2)))
        rl, rg = nref.gradients(params, xb, yb, "relu", "softmax", "categorical_crossentropy",
                                masks=[m1, m2], rate=rate)
        flat = [w for pair in params for w in pair]
        ref_opt.step(flat, [g for pair in rg for g in pair])
        params = [(flat[2 * i], flat[2 * i + 1]) for i in range(len(params))]
        log.append({"step": s + 1, "rows": [s * bs, (s + 1) * bs], "mask_1": m1.astype(int).tolist(),
                    "mask_2": m2.astype(int).tolist(), "keras_loss": kl, "reference_loss": rl,
                    "weights": _weights(model.get_weights()),
                    "max_abs_diff_reference": _maxdiff(model.get_weights(), flat)})
    probe = tf.ones((2000, 10))
    out = np.asarray(keras.layers.Dropout(rate)(probe, training=True))
    doc = {"meta": _meta(eg, {"weights": "1e-6 absolute", "loss": "1e-6 relative"},
                         "Inverted dropout (x·mask/(1−rate)) after each hidden activation; the mask "
                         "is Bernoulli(1−rate) per sample and neuron. Keras' own masks come from its "
                         "SeedGenerator and cannot be reproduced, so the masks are given."),
           "input": {"dataset": "datasets/iris.json", "architecture": [4, 10, 10, 3], "hidden": "relu",
                     "output": "softmax", "loss": "categorical_crossentropy", "optimizer": "adam",
                     "lr": lr, "rate": rate, "batch_size": bs, "rows": "X_train (stratified split, "
                     "scaled with all rows) in order", "masks": "RandomState(42).uniform >= rate, mask_1 then mask_2"},
           "expected": {"initial_weights": _weights(w0), "steps": log,
                        "keras_dropout_semantics": {
                            "values": sorted({float(v) for v in np.unique(out)}),
                            "kept_fraction": float((out != 0).mean()),
                            "inference_is_identity": bool(np.all(np.asarray(
                                keras.layers.Dropout(rate)(probe, training=False)) == 1.0))}}}
    writer.write_json(eg.PARITY_DIR / MODULE / "exact" / "iris_dropout_masks.json", doc, dump=gt.dumps_compact)


def _stat_metrics(model, X_test, y_test, task: str) -> dict:
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

    raw = model.predict(X_test, verbose=0)
    ev = model.evaluate(X_test, y_test, verbose=0)
    ev = ev if isinstance(ev, list) else [ev]
    if task == "Clasificación":
        yt = np.asarray(y_test)
        yt = np.argmax(yt, 1) if yt.ndim > 1 and yt.shape[1] > 1 else yt.ravel()
        yp = np.argmax(raw, 1) if raw.shape[1] > 1 else (raw.ravel() > 0.5).astype(int)
        return {"accuracy": float(accuracy_score(yt, yp)), "test_loss": float(ev[0])}
    yt = np.asarray(y_test, float).ravel()
    yp = raw.ravel()
    mse = float(mean_squared_error(yt, yp))
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)), "test_loss": float(ev[0])}


def case_stat(eg, name: str, spec: dict, writer) -> None:
    import golden_trees as gt
    from golden_render import LessonRenderer
    from mltutor.algorithms import model_training as mt
    from mltutor.desktop import ui

    tf, keras = _tf()
    runs = []
    with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(), plotly_as_data():
        warnings.simplefilter("ignore")
        state, label, ref_in = _setup(eg, spec, Path(tmpdir))
        r = LessonRenderer(_page(), state)
        r.render()
        cfg = dict(r.session.state["nn_config"])
        if "opt" in spec:
            cfg["optimizer"] = spec["opt"]
        r.session.state["active_tab_nn"] = 2
        r.render()
        epochs = r.find("slider", label="Épocas").props["value"]
        df, target = r.session.state["nn_df"], r.session.state["nn_target_col"]
        lr = spec.get("lr", 0.001)
        for seed in SEEDS:
            keras.utils.set_random_seed(seed)
            probe = _Probe(exact=False)
            with probe.patch(), ui.bind(ui.Session()):
                model, hist, X_test, y_test, _sc, _le = mt.train_neural_network(
                    df, target, cfg, lr, epochs, 0.2, True, 10, True, 0.5)
            h = hist.history
            runs.append({"seed": seed, "epochs_trained": len(h["loss"]),
                         "final": {k: float(v[-1]) for k, v in h.items()},
                         "best_val_loss": float(min(h["val_loss"])),
                         "best_epoch": int(np.argmin(h["val_loss"])),
                         "test": _stat_metrics(model, X_test, y_test, cfg["task_type"])})
    keys = list(runs[0]["test"].keys())
    bands = {}
    for k in keys + ["epochs_trained"]:
        vals = np.array([rr["test"][k] if k in rr["test"] else rr[k] for rr in runs], float)
        bands[k] = {"min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1))}
    acceptance = {}
    for k, b in bands.items():
        if k == "epochs_trained":
            acceptance[k] = {"lo": max(1, int(b["min"] * 0.5)), "hi": int(epochs)}
            continue
        floor = 0.03 if k == "accuracy" else (0.05 if k == "r2" else 0.0)
        margin = max(3 * b["std"], floor, 0.1 * abs(b["mean"]) if k in ("mse", "rmse", "mae", "test_loss") else 0)
        acceptance[k] = {"lo": b["min"] - margin, "hi": b["max"] + margin}
    doc = {"meta": _meta(eg, {"rust_mean_over_5_seeds": "inside acceptance[metric] = [min − m, max + m], "
                                                       "m = max(3·std, floor) (accuracy 0.03, r2 0.05, "
                                                       "errors 10 % of the mean)",
                              "each_rust_seed": "inside [min − 2m, max + 2m]"},
                         "Level (b) of D69: Keras and Rust draw different initial weights, dropout "
                         "masks and batch orders, so only the distribution of the final metrics can "
                         "match. The app sets no seed (every Python run differs); the fixture fixes "
                         "keras.utils.set_random_seed(seed) to be reproducible on this machine."),
           "input": {**ref_in, "app_label": label, "nn_config": {k: _jsonable(v) for k, v in cfg.items()},
                     "lr": lr, "epochs": epochs, "validation_split": 0.2, "early_stopping": True,
                     "patience": 10, "reduce_lr": True, "lr_factor": 0.5, "seeds": SEEDS,
                     "shuffle": True},
           "expected": {"runs": runs, "bands": bands, "acceptance": acceptance}}
    writer.write_json(eg.PARITY_DIR / MODULE / "stat" / f"{name}.json", doc, dump=gt.dumps_compact)


# ---------------------------------------------------------------------------
# Training diagnostics (model_evaluation.neural_network_diagnostics): pure logic, exact
# ---------------------------------------------------------------------------
def _render_fn(fn, state: dict) -> dict:
    from golden_render import LessonRenderer

    with plotly_as_data():
        r = LessonRenderer(fn, state)
        tree = r.render()
    return _scrub(_summarise_html(tree))


def diagnostics_numbers(history, cfg: dict) -> dict:
    from mltutor.algorithms.model_evaluation import (analyze_training_diagnostics,
                                                     generate_training_recommendations,
                                                     neural_network_diagnostics)
    from mltutor.viz.nn import show_training_history_tab

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = analyze_training_diagnostics(history, cfg)
        rec = generate_training_recommendations(d, history, cfg)
        loss = history.history["loss"]
        n = len(loss)
        trends = None
        if n >= 10:
            segs = [loss[:n // 3], loss[n // 3:2 * n // 3], loss[2 * n // 3:]]
            slopes = [float(np.polyfit(range(len(s)), s, 1)[0]) for s in segs]
            trends = {"segments": [[0, n // 3], [n // 3, 2 * n // 3], [2 * n // 3, n]],
                      "slopes": slopes, "consistency_std": float(np.std(slopes))}
        state = {"nn_history": history, "nn_config": cfg}
        tree = _render_fn(lambda: neural_network_diagnostics(history, cfg), state)
        try:
            hist_tab = _render_fn(show_training_history_tab, state)
        except Exception as exc:  # noqa: BLE001 - e.g. a history without val_loss
            hist_tab = {"error": f"{type(exc).__name__}: {exc}"}
    windows = {"convergence": {"early": "loss[:n//4]", "late": "loss[-n//4:] == loss[(-n)//4:] "
                               "(Python floor division of a negative number: the last ceil(n/4) epochs)"}}
    return {"diagnostics": {k: _jsonable(v) for k, v in d.items()},
            "recommendations": _jsonable(rec), "trends": trends, "windows": windows,
            "view": tree, "history_tab": hist_tab}


def _synthetic_histories() -> dict[str, tuple[dict, str]]:
    e = np.arange(60, dtype=float)
    smooth = 1.1 * np.exp(-e / 12) + 0.08
    out: dict[str, tuple[dict, str]] = {}

    def cls(loss, val_loss, acc, val_acc):
        return {"accuracy": list(acc), "loss": list(loss), "val_accuracy": list(val_acc),
                "val_loss": list(val_loss)}

    out["cls_ideal_60"] = (cls(smooth, smooth * 1.05, 1 - smooth / 1.5, 0.97 - smooth / 1.4), "Clasificación")
    out["cls_overfit_60"] = (cls(smooth * 0.5, 0.6 + 0.01 * e, np.minimum(0.99, 0.6 + e / 60),
                                 0.8 - 0.001 * e), "Clasificación")
    out["cls_poor_30"] = (cls(0.69 + 0.001 * np.sin(e[:30]), 0.70 + 0.002 * np.cos(e[:30]),
                              0.5 + 0.01 * np.sin(e[:30]), 0.48 + 0.0 * e[:30]), "Clasificación")
    out["cls_moderate_40"] = (cls(smooth[:40] + 0.2, smooth[:40] + 0.3, 0.75 + 0.0 * e[:40] + e[:40] / 400,
                                  0.62 + 0.0 * e[:40]), "Clasificación")
    out["cls_unstable_40"] = (cls(0.5 + 0.3 * np.abs(np.sin(e[:40])), 0.55 + 0.3 * np.abs(np.cos(e[:40])),
                                  0.8 + 0.05 * np.sin(e[:40]), 0.78 + 0.05 * np.cos(e[:40])), "Clasificación")
    out["cls_rising_late_30"] = (cls(np.concatenate([np.linspace(1, 0.3, 20), np.linspace(0.3, 0.5, 10)]),
                                     np.concatenate([np.linspace(1.1, 0.4, 20), np.linspace(0.4, 0.8, 10)]),
                                     np.linspace(0.6, 0.9, 30), np.linspace(0.55, 0.86, 30)), "Clasificación")
    out["cls_short_3"] = (cls([1.0, 0.8, 0.7], [1.0, 0.85, 0.75], [0.5, 0.6, 0.7], [0.5, 0.6, 0.72]), "Clasificación")
    out["cls_8_epochs"] = (cls(smooth[:8], smooth[:8] * 1.1, 1 - smooth[:8] / 2, 0.95 - smooth[:8] / 2), "Clasificación")
    out["cls_flat_20"] = (cls([0.5] * 20, [0.52] * 20, [0.9] * 20, [0.88] * 20), "Clasificación")
    out["cls_10_epochs_boundary"] = (cls(smooth[:10], smooth[:10] * 1.02, 1 - smooth[:10] / 2,
                                         0.95 - smooth[:10] / 2), "Clasificación")
    out["cls_11_epochs_window"] = (cls(smooth[:11], smooth[:11] * 1.02, 1 - smooth[:11] / 2,
                                       0.95 - smooth[:11] / 2), "Clasificación")
    reg_loss = 3000 * np.exp(-e / 10) + 50
    out["reg_good_60"] = ({"loss": list(reg_loss), "mae": list(np.sqrt(reg_loss)), "val_loss": list(reg_loss * 1.1),
                           "val_mae": list(np.sqrt(reg_loss * 1.1))}, "Regresión")
    out["reg_bad_30"] = ({"loss": list(29000 - 10 * e[:30]), "mae": list(150 - 0.01 * e[:30]),
                          "val_loss": list(31000 + 0.0 * e[:30]), "val_mae": list(160 + 0.5 * e[:30])}, "Regresión")
    out["reg_no_val_mae_20"] = ({"loss": list(reg_loss[:20]), "mae": list(np.sqrt(reg_loss[:20])),
                                 "val_loss": list(reg_loss[:20] * 1.2)}, "Regresión")
    return out


def case_diagnostics(eg, writer) -> None:
    import golden_trees as gt

    cases = {}
    for name, (h, task) in _synthetic_histories().items():
        h = {k: [float(x) for x in v] for k, v in h.items()}
        cfg = {"task_type": task, "architecture": [4, 10, 10, 3 if task == "Clasificación" else 1],
               "activation": "relu", "output_activation": "softmax" if task == "Clasificación" else "linear",
               "dropout_rate": 0.2, "batch_size": 64, "optimizer": "adam", "input_size": 4,
               "output_size": 3 if task == "Clasificación" else 1}
        out = diagnostics_numbers(SimpleNamespace(history=h), cfg)
        cases[name] = {"history": h, "config": cfg, **out}
    doc = {"meta": _meta(eg, {"numbers": "1e-12 relative (np.polyfit slope = least squares)",
                              "texts": "exact"},
                         "Pure logic over the history: same inputs, same output. The real histories "
                         "of every nn/exact case are also in their expected.diagnostics."),
           "input": {"function": "mltutor.algorithms.model_evaluation.neural_network_diagnostics + "
                                 "mltutor.viz.nn.show_training_history_tab"},
           "expected": {"cases": cases}}
    writer.write_json(eg.PARITY_DIR / MODULE / "diagnostics.json", doc, dump=gt.dumps_compact)


# ---------------------------------------------------------------------------
# 🏗️ Arquitectura and ⚙️ Entrenamiento controls (no training)
# ---------------------------------------------------------------------------
CONTROL_DATASETS = ["iris", "wine", "breast_cancer", "digits", "diabetes", "california_housing",
                    "moons", "titanic", "penguins", "mpg"]


def case_controls(eg, writer) -> None:
    import golden_trees as gt
    from golden_render import LessonRenderer

    out: dict[str, Any] = {"datasets": {}, "variations": {}, "training": {}, "code": {}, "tabs_without_model": {}}
    with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(), plotly_as_data():
        warnings.simplefilter("ignore")
        tmp = Path(tmpdir)
        specs = [{"dataset": d} for d in CONTROL_DATASETS] + [{"csv": "string_target"},
                                                               {"csv": "numeric_regression"}]
        for spec in specs:
            key = spec.get("dataset") or f"csv_{spec['csv']}"
            state, label, _ = _setup(eg, spec, tmp)
            r = LessonRenderer(_page(), state)
            tree = r.render()
            st = r.session.state
            item = {"view": view(tree), "nn_config": _controls_used(r), "n_rows": int(st["nn_df"].shape[0])}
            st["active_tab_nn"] = 2
            item["entrenamiento"] = view(r.render())
            item["epochs_default"] = r.find("slider", label="Épocas").props["value"]
            out["datasets"][key] = item
        # Variations on Iris (multiclass), Cancer (binary) and Diabetes (regression).
        for ds in ("iris", "breast_cancer", "diabetes"):
            state, label, _ = _setup(eg, {"dataset": ds}, tmp)
            r = LessonRenderer(_page(), state)
            r.render()
            var: dict[str, Any] = {}
            opts = r.find("select", label="Función de activación de salida").props["options"]
            for o in opts:
                var[f"salida_{o}"] = view(r.set("select", o, label="Función de activación de salida"))
            r.set("select", opts[0], label="Función de activación de salida")
            if ds == "iris":
                for a in ("relu", "tanh", "sigmoid"):
                    var[f"oculta_{a}"] = view(r.set("select", a, label="Función de activación"))
                r.set("select", "relu", label="Función de activación")
                for n in (1, 3, 5):
                    var[f"capas_{n}"] = view(r.set("slider", n, label="Número de capas ocultas"))
                r.set("slider", 2, label="Número de capas ocultas")
                for d in (0.0, 0.2, 0.3, 0.6):
                    var[f"dropout_{d}"] = view(r.set("slider", d, label="Tasa de Dropout"))
                for b in (16, 32, 64, 128, 256):
                    var[f"batch_{b}"] = view(r.set("select", b, label="Tamaño de Batch"))
                for o in ("adam", "sgd", "rmsprop"):
                    var[f"optimizador_{o}"] = view(r.set("select", o, label="Optimizador"))
                var["neuronas_256_x2"] = view(r.set("slider", 256, key="layer_1"))
            out["variations"][ds] = var
            # Training tab controls
            r.session.state["active_tab_nn"] = 2
            r.render()
            tr: dict[str, Any] = {}
            for lr in (0.001, 0.01, 0.1, 0.3):
                tr[f"lr_{lr}"] = view(r.set("select", lr, label="Tasa de Aprendizaje"))
            for ep in (10, 50, 160, 500):
                tr[f"epocas_{ep}"] = view(r.set("slider", ep, label="Épocas"))
            for v in (10, 20, 40):
                tr[f"validacion_{v}"] = view(r.set("slider", v, label="% Datos de Validación"))
            for p in (5, 10, 20):
                tr[f"paciencia_{p}"] = view(r.set("slider", p, label="Paciencia (épocas)"))
            for f in (0.2, 0.5, 0.8):
                tr[f"factor_{f}"] = view(r.set("slider", f, label="Factor de Reducción"))
            tr["sin_parada_temprana"] = view(r.set("checkbox", False, label="Activar Parada Temprana"))
            tr["sin_reduccion_lr"] = view(r.set("checkbox", False, label="Reducir Learning Rate Automáticamente"))
            n = r.session.state["nn_df"].shape[0]
            tr["distribucion_real_vs_caption"] = {
                str(v): {"caption": [int(n * (100 - v) / 100 * 0.8), int(n * v / 100),
                                     n - int(n * (100 - v) / 100 * 0.8) - int(n * v / 100)],
                         "real": [math.floor((n - math.ceil(0.2 * n)) * (1 - v / 100)),
                                  (n - math.ceil(0.2 * n)) - math.floor((n - math.ceil(0.2 * n)) * (1 - v / 100)),
                                  math.ceil(0.2 * n)]}
                for v in range(10, 41)}
            out["training"][ds] = tr
        # Architecture code (literal Python and the template)
        from mltutor.algorithms.code_examples import generate_neural_network_architecture_code as gen

        code_cases = {
            "iris_default": ([4, 10, 10, 3], "relu", "softmax", 0.2, "adam", 64, "Clasificación", "iris"),
            "breast_cancer_default": ([30, 30, 15, 1], "relu", "sigmoid", 0.2, "adam", 64, "Clasificación",
                                      "breast_cancer"),
            "diabetes_tanh_sgd": ([10, 10, 1], "tanh", "linear", 0.0, "sgd", 32, "Regresión", "diabetes"),
            "iris_1layer_tanh_output": ([4, 8, 3], "sigmoid", "tanh", 0.3, "rmsprop", 16, "Clasificación", "iris"),
            "wine_5layers_linear": ([13, 13, 10, 10, 10, 10, 3], "relu", "linear", 0.8, "sgd", 256,
                                    "Clasificación", "wine"),
            "csv_empty_names": ([2, 10, 10, 1], "relu", "linear", 0.2, "adam", 64, "Regresión", None),
        }
        for name, (arch, act, oact, drop, opt, bs, task, ds) in code_cases.items():
            names = []
            if ds:
                from mltutor.dataset.dataset_manager import load_data

                names = list(load_data(eg.DATASETS[ds])[2])
            py = gen(arch, act, oact, drop, opt, bs, task, names)
            tpl = eg._render_code_template("red_arquitectura.py.j2", {
                "architecture": arch, "activation": act, "output_activation": oact, "dropout_rate": drop,
                "optimizer": opt, "batch_size": bs, "task_type": task, "feature_names": names})
            try:
                compile(py, "red.py", "exec")
                compiles = True
                compile_error = None
            except SyntaxError as exc:
                compiles, compile_error = False, f"SyntaxError: {exc.msg} (línea {exc.lineno})"
            out["code"][name] = {"context": {"architecture": arch, "activation": act, "output_activation": oact,
                                             "dropout_rate": drop, "optimizer": opt, "batch_size": bs,
                                             "task_type": task, "feature_names": names},
                                 "python": py, "template_equal": py == tpl, "compiles": compiles,
                                 "compile_error": compile_error,
                                 "file_name": f"red_neuronal_arquitectura_{task.lower()}.py"}
        # Tabs 2..6 without configuration / model
        for tab in range(2, 7):
            state, label, _ = _setup(eg, {"dataset": "iris"}, tmp)
            state["active_tab_nn"] = tab
            out["tabs_without_model"][str(tab)] = view(LessonRenderer(_page(), state).render())
    doc = {"meta": _meta(eg, {"texts": "exact", "numbers": "exact"},
                         "Controls, defaults, captions and notices of 🏗️ and ⚙️; nothing is trained."),
           "input": {"function": "run_neural_networks_app rendered headlessly"}, "expected": out}
    writer.write_json(eg.PARITY_DIR / MODULE / "controls.json", doc, dump=gt.dumps_compact)


def case_split(eg, writer) -> None:
    import golden_trees as gt
    from mltutor.dataset.dataset_manager import load_data

    out = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for spec in [{"dataset": d} for d in ("iris", "wine", "breast_cancer", "digits", "titanic", "penguins",
                                              "moons")] + [{"csv": "string_target"}]:
            key = spec.get("dataset") or f"csv_{spec['csv']}"
            _state, label, _ = _setup(eg, spec, Path(tmpdir))
            if "csv" in spec:
                from mltutor.dataset.dataset_manager import load_dataset_from_file

                p = Path(tmpdir) / f"{spec['csv']}.csv"
                p.write_bytes(eg._csv_cases()[spec["csv"]])
                y = load_dataset_from_file(str(p), None, "auto")[1]
            else:
                y = _load(eg, label)[1]
            block, _tr, _te = _split_block(np.asarray(y.reset_index(drop=True)), "Clasificación")
            block["class_counts"] = {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()}
            out[key] = block
    doc = {"meta": _meta(eg, {"indices": "exact"},
                         "train_test_split(test_size=0.2, random_state=42, stratify=y) as the lesson calls "
                         "it; tools/nn_reference.stratified_split reproduces it with RandomState.permutation only."),
           "input": {"function": "sklearn.model_selection.train_test_split (StratifiedShuffleSplit)"},
           "expected": out}
    writer.write_json(eg.PARITY_DIR / MODULE / "split_stratified.json", doc, dump=gt.dumps_compact)


ERROR_CASES = {
    "iris_salida_sigmoid": {"dataset": "iris", "out": "sigmoid", "epochs": 10},
    "iris_salida_linear": {"dataset": "iris", "out": "linear", "epochs": 10},
    "breast_cancer_salida_softmax": {"dataset": "breast_cancer", "out": "softmax", "epochs": 10},
    "breast_cancer_salida_linear": {"dataset": "breast_cancer", "out": "linear", "epochs": 10},
    "diabetes_salida_softmax": {"dataset": "diabetes", "out": "softmax", "epochs": 10},
    "csv_una_muestra_por_clase": {"csv": "one_class_per_row", "epochs": 10},
}


def case_errors(eg, writer) -> None:
    import golden_trees as gt

    out = {}
    with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(), plotly_as_data():
        warnings.simplefilter("ignore")
        tf, keras = _tf()
        for name, spec in ERROR_CASES.items():
            keras.utils.set_random_seed(42)
            item: dict[str, Any] = {"spec": spec}
            try:
                r, _label, _ref, after = configure_and_train(eg, spec, Path(tmpdir), _Probe(exact=False),
                                                             None, dropout_default=True)
            except Exception as exc:  # noqa: BLE001
                item["exception"] = _scrub(f"{type(exc).__name__}: {exc}")
                out[name] = item
                continue
            st = r.session.state
            item["training_view"] = view(after)
            item["state_after"] = {"model_trained_nn": st.get("model_trained_nn"),
                                   "nn_model_is_none": st.get("nn_model") is None}
            for tab in (3, 4, 5, 6):
                st["active_tab_nn"] = tab
                try:
                    item[f"tab_{tab}"] = view(r.render())
                except Exception as exc:  # noqa: BLE001
                    item[f"tab_{tab}"] = {"exception": _scrub(f"{type(exc).__name__}: {str(exc)[:300]}")}
            out[name] = item
    doc = {"meta": _meta(eg, {"texts": "exact (TF messages may change with the TF version)"},
                         "Output activations that Python lets the student choose but cannot train, and "
                         "what the other tabs do afterwards (D71)."),
           "input": {"function": "run_neural_networks_app rendered headlessly"}, "expected": out}
    writer.write_json(eg.PARITY_DIR / MODULE / "errors.json", doc, dump=gt.dumps_compact)


def register(cases: dict, eg) -> None:
    for name, spec in EXACT_CASES.items():
        cases[f"{MODULE}/exact/{name}"] = (lambda n, s: (lambda w: case_exact(eg, n, s, w)))(name, spec)
    cases[f"{MODULE}/exact/iris_dropout_masks"] = lambda w: case_dropout_masks(eg, w)
    for name, spec in STAT_CASES.items():
        cases[f"{MODULE}/stat/{name}"] = (lambda n, s: (lambda w: case_stat(eg, n, s, w)))(name, spec)
    cases[f"{MODULE}/diagnostics"] = lambda w: case_diagnostics(eg, w)
    cases[f"{MODULE}/controls"] = lambda w: case_controls(eg, w)
    cases[f"{MODULE}/split_stratified"] = lambda w: case_split(eg, w)
    cases[f"{MODULE}/errors"] = lambda w: case_errors(eg, w)
    cases["code/nn_training"] = lambda w: case_code_nn_training(eg, w)


# ---------------------------------------------------------------------------
# 💾 Exportar in Rust (D75): ONNX MLP, "paquete Keras" and the NumPy predictor
# ---------------------------------------------------------------------------
ONNX_ACT = {"relu": "Relu", "tanh": "Tanh", "sigmoid": "Sigmoid", "softmax": "Softmax", "linear": "Identity"}


def onnx_proposal_model(ws, hidden: str, output: str, mean, scale, task: str):
    """float_input (raw features) -> (x − mean)/scale -> Gemm + activation per layer.

    Classification: ``probabilities`` (the network output) and ``label`` (ArgMax, or output > 0.5
    with one neuron). Regression: ``variable``. ir_version 8, opset 15 (MEMORY.md)."""
    from onnx import TensorProto, helper, numpy_helper

    f = ws[0].shape[0]
    inits = [numpy_helper.from_array(np.asarray(mean, np.float32), "scaler_mean"),
             numpy_helper.from_array(np.asarray(scale, np.float32), "scaler_scale")]
    nodes = [helper.make_node("Sub", ["float_input", "scaler_mean"], ["centred"], name="scaler_sub"),
             helper.make_node("Div", ["centred", "scaler_scale"], ["scaled"], name="scaler_div")]
    prev = "scaled"
    n_layers = len(ws) // 2
    for i in range(n_layers):
        W, b = np.asarray(ws[2 * i], np.float32), np.asarray(ws[2 * i + 1], np.float32)
        inits += [numpy_helper.from_array(W, f"W{i}"), numpy_helper.from_array(b, f"B{i}")]
        nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"B{i}"], [f"z{i}"], name=f"gemm_{i}"))
        act = output if i == n_layers - 1 else hidden
        out_name = (("probabilities" if task == "Clasificación" else "variable")
                    if i == n_layers - 1 else f"a{i}")
        kw = {"axis": 1} if act == "softmax" else {}
        nodes.append(helper.make_node(ONNX_ACT[act], [f"z{i}"], [out_name], name=f"act_{i}", **kw))
        prev = out_name
    k = ws[-1].shape[0]
    outputs = [helper.make_tensor_value_info(prev, TensorProto.FLOAT, [None, k])]
    if task == "Clasificación":
        if k > 1:
            nodes.append(helper.make_node("ArgMax", [prev], ["label"], name="argmax", axis=1, keepdims=0))
        else:
            inits.append(numpy_helper.from_array(np.array(0.5, np.float32), "threshold"))
            inits.append(numpy_helper.from_array(np.array([1], np.int64), "squeeze_axes"))
            nodes += [helper.make_node("Greater", [prev, "threshold"], ["is_positive"], name="greater"),
                      helper.make_node("Cast", ["is_positive"], ["label_2d"], name="cast", to=TensorProto.INT64),
                      helper.make_node("Squeeze", ["label_2d", "squeeze_axes"], ["label"], name="squeeze")]
        outputs.insert(0, helper.make_tensor_value_info("label", TensorProto.INT64, [None]))
    graph = helper.make_graph(nodes, "mltutor_mlp",
                              [helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [None, f])],
                              outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)],
                              producer_name="mltutor-golden")
    model.ir_version = 8
    return model


def onnx_proposal(model, scaler, task: str, X_raw: np.ndarray, cfg: dict) -> dict:
    import onnx
    from onnx.reference import ReferenceEvaluator

    ws = model.get_weights()
    m = onnx_proposal_model(ws, cfg["activation"], cfg["output_activation"], scaler.mean_, scaler.scale_, task)
    onnx.checker.check_model(m, full_check=True)
    x32 = np.asarray(X_raw, np.float32)
    got = ReferenceEvaluator(m).run(None, {"float_input": x32})
    want = model.predict(scaler.transform(x32), verbose=0)
    out_idx = 1 if task == "Clasificación" else 0
    err = float(np.max(np.abs(got[out_idx] - want)))
    res = {"nodes": [{"op": n.op_type, "name": n.name, "inputs": list(n.input), "outputs": list(n.output),
                      "attributes": {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}}
                     for n in m.graph.node],
           "inputs": [i.name for i in m.graph.input], "outputs": [o.name for o in m.graph.output],
           "ir_version": m.ir_version, "opset": 15, "bytes": len(m.SerializeToString()),
           "rows_checked": int(len(x32)), "max_abs_error_vs_keras": err,
           "evaluator": "onnx.reference.ReferenceEvaluator (onnx " + onnx.__version__ + ")"}
    if task == "Clasificación":
        lab = (np.argmax(want, 1) if want.shape[1] > 1 else (want.ravel() > 0.5).astype(int))
        res["label_mismatches"] = int(np.sum(got[0] != lab))
    return res


def package_model_json(model, scaler, cfg: dict, st, X_check, n_samples: int) -> dict:
    task = "classification" if cfg["task_type"] == "Clasificación" else "regression"
    ws = model.get_weights()
    dense = [ly for ly in model.layers if ly.get_weights()]
    layers = [{"name": ly.name, "units": int(ws[2 * i].shape[1]),
               "activation": cfg["output_activation"] if i == len(dense) - 1 else cfg["activation"],
               "kernel": _mat(ws[2 * i]), "bias": _fl(ws[2 * i + 1])} for i, ly in enumerate(dense)]
    res = __import__("nn_reference").resolve_training(cfg["task_type"], cfg["output_activation"],
                                                      cfg["output_size"])
    lr = float(model.optimizer.learning_rate.numpy())
    return {"format": "mltutor-red-neuronal/1", "task": task,
            "feature_names": list(st["nn_feature_names"]),
            "class_names": list(st["nn_class_names"]) if task == "classification" else None,
            "architecture": list(cfg["architecture"]), "activation": cfg["activation"],
            "output_activation": cfg["output_activation"], "dropout_rate": cfg["dropout_rate"],
            "loss": res["loss"], "metric": "accuracy" if task == "classification" else "mae",
            "optimizer": {"name": cfg["optimizer"], "learning_rate": lr},
            "scaler": {"mean": _fl(scaler.mean_), "scale": _fl(scaler.scale_), "n_samples": n_samples},
            "layers": layers,
            "config": {k: _jsonable(v) for k, v in cfg.items()},
            "check": {"X": _mat(X_check), "output": _mat(model.predict(scaler.transform(
                np.asarray(X_check, np.float32)), verbose=0))}}


def export_package(eg, model, scaler, cfg, st, X_check, n_samples, run: bool) -> dict:
    task = "classification" if cfg["task_type"] == "Clasificación" else "regression"
    data = package_model_json(model, scaler, cfg, st, X_check, n_samples)
    dataset = st["selected_dataset"]
    script = eg._render_code_template("paquete_modelo_red.py.j2", {"dataset": dataset, "task": task,
                                                                     "json_name": "modelo.json"})
    numpy_script = eg._render_code_template("red_predictor_numpy.py.j2", {"dataset": dataset, "task": task,
                                                                           "json_name": "modelo.json"})
    classes = ([{"value": i, "name": n} for i, n in enumerate(data["class_names"])]
               if task == "classification" else None)
    leeme = eg._render_code_template("paquete_leeme_red.txt.j2", {
        "dataset": dataset, "task": task, "architecture": data["architecture"], "activation": cfg["activation"],
        "output_activation": cfg["output_activation"], "dropout_rate": cfg["dropout_rate"],
        "n_params": int(model.count_params()), "feature_names": data["feature_names"], "classes": classes,
        "n_train": int(len(st["nn_df"]) - len(st["nn_test_data"][0])), "n_test": int(len(st["nn_test_data"][0]))})
    out = {"model_json": data, "script": script, "numpy_script": numpy_script, "leeme": leeme}
    if run:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "modelo.json").write_text(json.dumps(data, ensure_ascii=False), "utf-8")
            (Path(tmp) / "reconstruir_modelo.py").write_text(script, "utf-8")
            (Path(tmp) / "predecir_numpy.py").write_text(numpy_script, "utf-8")
            env = {**__import__("os").environ, "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": ""}
            p = subprocess.run([sys.executable, "reconstruir_modelo.py"], cwd=tmp, capture_output=True,
                               text=True, env=env, timeout=600)
            q = subprocess.run([sys.executable, "predecir_numpy.py"], cwd=tmp, capture_output=True,
                               text=True, env=env, timeout=120)
            files = sorted(str(f.relative_to(tmp)) for f in Path(tmp).iterdir())
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith(("✅", "💾", "⚠️"))]
            out["run"] = {"returncode": p.returncode, "stdout_lines": [re.sub(r"\(error máximo .*\)",
                                                                              "(error máximo …)", ln)
                                                                       for ln in lines],
                          "files": files, "stderr_tail": _scrub(p.stderr[-300:]) if p.returncode else ""}
            diff = [ln for ln in q.stdout.splitlines() if ln.startswith("Diferencia máxima")]
            val = float(diff[0].split(":")[1]) if diff else math.nan
            out["numpy"] = {"returncode": q.returncode, "max_abs_diff_vs_app_lt_1e-5": bool(val < 1e-5)}
    return out


# ---------------------------------------------------------------------------
# code/nn_training.json: red_entrenamiento.py.j2 executed (D74)
# ---------------------------------------------------------------------------
TRAINING_TEMPLATE = "red_entrenamiento.py.j2"
TRAINING_CASES = ["iris_default", "breast_cancer_default", "diabetes_adam_lr01", "moons_default",
                  "csv_string_target_default", "csv_numeric_regression_default"]
_TRAINING_PROBE = """\
import json, runpy, sys
import numpy as np
g = runpy.run_path(sys.argv[1], run_name="__main__")
out = {k: float(g[k]) for k in ("accuracy", "mse", "rmse", "mae", "r2") if k in g}
out["x_test"] = np.asarray(g["X_test"], dtype=float).tolist()
out["y_test"] = np.asarray(g["y_test"]).tolist()
out["params"] = int(g["modelo"].count_params())
out["epochs"] = len(g["historial"].history["loss"])
out["history_keys"] = sorted(g["historial"].history)
out["prediccion"] = float(g["prediccion"])
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f)
"""


def _training_shared_blocks(eg) -> None:
    code = eg.REPO_ROOT / "content" / "code"
    knn = (code / "knn_entrenamiento.py.j2").read_text("utf-8")
    red = (code / TRAINING_TEMPLATE).read_text("utf-8")

    def block(text: str, start: str, end: str) -> str:
        i = text.index(start)
        return text[i:text.index(end, i) + len(end)]

    for start, end in (("import numpy as np\n", "make_moons\n{%- endif %}\n"),
                       ("# 1. Cargar los datos", "{%- endif %}\n{%- endif %}\n")):
        if block(knn, start, end) != block(red, start, end):
            raise AssertionError(f"{TRAINING_TEMPLATE}: the block '{start.strip()}' differs from "
                                 "knn_entrenamiento.py.j2")


def training_context(eg, fixture: dict) -> tuple[dict, dict[str, bytes]]:
    import nn_reference as nref

    inp = fixture["input"]
    cfg = inp["nn_config"]
    files: dict[str, bytes] = {}
    ctx: dict[str, Any] = dict.fromkeys(["dataset", "dataset_label", "csv_file", "csv_target", "csv_sep",
                                         "csv_decimal", "csv_encoding"])
    task = "classification" if cfg["task_type"] == "Clasificación" else "regression"
    if "csv_fixture" in inp:
        name = Path(inp["csv_fixture"]).stem
        files[f"{name}.csv"] = eg._csv_cases()[name]
        ctx.update(dataset="csv", csv_file=f"{name}.csv", csv_target=inp["target"], csv_sep=",",
                   csv_decimal=".", csv_encoding="utf-8")
    else:
        ctx["dataset"] = Path(inp["dataset"]).stem
    from mltutor.dataset.dataset_manager import load_data

    if ctx["dataset"] == "csv":
        feature_names, class_names = None, None
    else:
        _X, _y, feature_names, class_names, _i, _t = _load(eg, inp["app_label"])
    res = nref.resolve_training(cfg["task_type"], cfg["output_activation"], cfg["output_size"])
    es, rlr = inp["early_stopping"], inp["reduce_lr"]
    ctx.update(dataset_label=inp["app_label"], task=task, architecture=cfg["architecture"],
               activation=cfg["activation"], output_activation=cfg["output_activation"], loss=res["loss"],
               one_hot=res["one_hot"], dropout_rate=cfg["dropout_rate"], optimizer=cfg["optimizer"],
               learning_rate=inp["lr"], batch_size=cfg["batch_size"], epochs=inp["epochs"],
               validation_split=inp["validation_split"], early_stopping=es,
               patience=inp["patience"] if es else None, reduce_lr=rlr,
               lr_factor=inp["lr_factor"] if rlr else None,
               reduce_lr_patience=max(1, (inp["patience"] if es else 10) // 2) if rlr else None,
               feature_names=list(feature_names) if feature_names is not None else None,
               class_names=list(class_names) if (class_names is not None and task == "classification") else None,
               sample=None)
    return ctx, files


def case_code_nn_training(eg, writer) -> None:
    import golden_trees as gt
    import nn_reference as nref

    _training_shared_blocks(eg)
    out = {}
    for name in TRAINING_CASES:
        fixture = json.loads((eg.PARITY_DIR / MODULE / "stat" / f"{name}.json").read_text("utf-8"))
        ctx, files = training_context(eg, fixture)
        with tempfile.TemporaryDirectory() as tmp:
            # CSV datasets: the app reads the stored copy; names/classes come from that load.
            if ctx["dataset"] == "csv":
                from mltutor.dataset.dataset_manager import load_dataset_from_file

                p = Path(tmp) / ctx["csv_file"]
                p.write_bytes(files[ctx["csv_file"]])
                _X, _y, fn, cn, _i, _t = load_dataset_from_file(str(p), ctx["csv_target"], "auto")
                ctx["feature_names"] = list(fn)
                ctx["class_names"] = list(cn) if ctx["task"] == "classification" and cn is not None else None
                if ctx["task"] == "classification":
                    raw = pd.read_csv(p)[ctx["csv_target"]]
                    if raw.dtype == object:  # D10: Rust keeps the original class names
                        ctx["class_names"] = sorted(raw.astype(str).unique().tolist())
            script = eg._render_code_template(TRAINING_TEMPLATE, ctx)
            for fname, data in files.items():
                (Path(tmp) / fname).write_bytes(data)
            (Path(tmp) / "script.py").write_text(script, "utf-8")
            (Path(tmp) / "probe.py").write_text(_TRAINING_PROBE, "utf-8")
            env = {**__import__("os").environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
            proc = subprocess.run([sys.executable, "probe.py", "script.py", "numbers.json"], cwd=tmp,
                                  capture_output=True, text=True, env=env, timeout=900)
            if proc.returncode:
                raise RuntimeError(f"{name}: the script failed:\n{proc.stderr[-1500:]}")
            got = json.loads((Path(tmp) / "numbers.json").read_text("utf-8"))
        # What the app computes for the same data
        from mltutor.dataset.dataset_manager import load_data
        from sklearn.model_selection import train_test_split

        if ctx["dataset"] == "csv":
            with tempfile.TemporaryDirectory() as tmp2:
                from mltutor.dataset.dataset_manager import load_dataset_from_file

                p = Path(tmp2) / ctx["csv_file"]
                p.write_bytes(files[ctx["csv_file"]])
                X, y = load_dataset_from_file(str(p), ctx["csv_target"], "auto")[:2]
        else:
            X, y = _load(eg, fixture["input"]["app_label"])[:2]
        Xs, _m, _s = nref.standard_scaler(X.to_numpy(float))
        from sklearn.preprocessing import StandardScaler

        Xs = StandardScaler().fit_transform(X)
        strat = y if ctx["task"] == "classification" else None
        _a, X_test, _b, y_test = train_test_split(Xs, y.reset_index(drop=True), test_size=0.2,
                                                  random_state=42, stratify=strat)
        acc = fixture["expected"]["acceptance"]
        metric = "accuracy" if ctx["task"] == "classification" else "r2"
        lo, hi = acc[metric]["lo"], acc[metric]["hi"]
        margin = hi - lo
        x_diff = float(np.max(np.abs(np.asarray(got["x_test"]) - np.asarray(X_test, float))))
        checks = {"x_test_equal_app": x_diff < 1e-12, "x_test_max_abs_diff": x_diff,
                  "y_test_equal_app": got["y_test"] == np.asarray(y_test).tolist(),
                  "params_equal_app": got["params"] == sum(a * b + b for a, b in zip(
                      ctx["architecture"][:-1], ctx["architecture"][1:])),
                  f"{metric}_in_band": bool(lo <= got[metric] <= hi),
                  f"{metric}_in_band_2x": bool(lo - margin / 2 <= got[metric] <= hi + margin / 2)}
        if not (checks["x_test_equal_app"] and checks["y_test_equal_app"] and checks["params_equal_app"]):
            raise RuntimeError(f"{name}: the script does not reproduce the app: {checks}")
        out[name] = {"context": ctx, "script": script, "got": {k: v for k, v in got.items() if k not in ("y_test", "x_test")},
                     "checks": checks, "band": acc[metric]}
    doc = {"meta": _meta(eg, {"script_text": "exact", "x_test_y_test_params": "exact",
                              "metrics": "inside the nn/stat acceptance band (other RNG)"},
                         "red_entrenamiento.py.j2 rendered with the context Rust builds and executed: "
                         "same data, scaling, split and network as the app; the metrics fall in the band "
                         "of level (b)."),
           "input": {"template": f"content/code/{TRAINING_TEMPLATE}", "cases": TRAINING_CASES},
           "expected": out}
    writer.write_json(eg.PARITY_DIR / "code" / "nn_training.json", doc, dump=gt.dumps_compact)
