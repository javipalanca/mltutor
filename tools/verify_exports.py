#!/usr/bin/env python3
"""Cross-check every export and code template of the Rust app against Python (Phase 6).

Replaces the old ``--verify-models`` of the Python app. The corpus comes from the Rust
side, through the app's domain layer (``src-tauri/src/export_corpus.rs``):

    MLTUTOR_EXPORT_CORPUS=/tmp/corpus cargo test -p mltutor --bin mltutor \\
        export_corpus::export_corpus -- --ignored --exact
    uv run --no-project --python 3.11 --with-requirements tools/verify_exports.lock.txt \\
        python tools/verify_exports.py /tmp/corpus

(on the Apple Silicon Mac of the project: ``--python .venv/bin/python``, see MEMORY.md).

For every case of the corpus (every test row of the split is an input):

* ``onnx``: ``modelo.onnx`` in onnxruntime (float32 input, like skl2onnx) against Rust.
* ``paquete``: the ZIP is unpacked in a fresh temporary folder and ``reconstruir_modelo.py``
  runs with ``python -I -X utf8`` and without network. It must pass its own check and
  write every file D1 promises (``.pkl``; networks: ``.keras``, ``.h5``, SavedModel,
  ``.tflite``, ``scaler.pkl`` and ``modelo_completo.pkl``). Every artefact is loaded again
  and must reproduce Rust on every input. Trees: the rules ``.txt`` equal ``export_text``
  of the rebuilt model, and ``usar_modelo.py`` (``predecir`` and
  ``predecir_con_probabilidades``) gives Rust's classes, values and probabilities.
* ``numpy``: 🧠 ``red_predictor_numpy.zip`` against Rust; ``metadatos``: the 🧠 JSON.
* ``plantillas``: every template the lesson offers, as the app rendered it, executed
  offline. The training script first: its split must be the app's row by row, and the
  scikit-learn models must predict like Rust. The 🧠 Keras script retrains: its split and
  test size must be the app's, its losses ≥ 0 and its accuracy above the majority class
  (R² > 0 in regression). Then each snippet in the namespace the training script leaves,
  with the names its comment says it assumes (``GLUE``), and its result is checked: the
  confusion matrix and the precision of every class (all classes, also the missing ones),
  the decision boundary's columns and legend, the tree's text with weights (= 📝 Texto),
  every line of the decision path (= Rust's path), and the tree's Python function.

Tolerances are in ``TOL``. Nothing passes vacuously: a comparison with no rows, a different
number of rows than Rust, or a row where nothing could be compared is a failure.

KNN: on every row the sorted K distances of Rust, of the rebuilt scikit-learn model and of
the training script must agree (``TOL["knn_distance"]``), and Rust's must agree with the
distances computed here in float64. A row is a tie when the K-th distance is shared by
more training rows than places remain (``KNN_TIE_REL``). The verifier enumerates every
answer valid under that tie (every K-subset compatible with it; for ONNX, in the space of
the float32-rounded input, with scikit-learn's distance-0 rule) and the artefact's answer
must be one of them. When it differs from Rust's the row is "tolerated" and counted by
cause (tie, float32 input, lost distance 0); tolerated rows may not exceed
``KNN_TOLERATED_MAX``.

Templates that are not runnable on purpose (Python's literal text kept for the text parity
tests and never offered by the Rust app) are listed in ``NOT_RUN`` with the reason; the
coverage is driven by the template names Rust recorded in ``case.json``. Exit status 1 if
any check fails.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CODE_DIR = REPO / "content" / "code"
# Offline copies of the seaborn datasets (sns.load_dataset reads them from SEABORN_DATA).
SEABORN_DATA = REPO / "mltutor" / "dataset" / "data" / "sample_datasets"

# --- tolerances --------------------------------------------------------------------------
# Values: |artefact − Rust| ≤ abs + rel·|Rust|. A label may differ only when Rust's two
# most likely classes are within `label_gap` (≤ 2× the probability tolerance, so a tolerated
# label still needs probabilities within `proba`).
TOL: dict[str, dict[str, float]] = {
    # onnxruntime computes in float32 from float32 inputs; Rust in float64.
    "onnx_linear": {"rel": 1e-5, "abs": 1e-4, "proba": 1e-5, "label_gap": 2e-5},
    # Trees: float32 comparisons on both sides (sklearn casts X to float32), so the leaf is
    # the same; leaf values/weights are stored as float32 in the ONNX.
    "onnx_tree": {"rel": 1e-6, "abs": 1e-6, "proba": 1e-6, "label_gap": 0.0},
    # MLP: float32 on both sides.
    "onnx_nn": {"rel": 1e-5, "abs": 1e-5, "proba": 1e-5, "label_gap": 2e-5},
    # Rebuilt LinearRegression / LogisticRegression: coef_ and intercept_ are set from
    # modelo.json, so only the summation order differs.
    "sk_package": {"rel": 1e-12, "abs": 1e-12, "proba": 1e-12, "label_gap": 2e-12},
    # Training script, OLS refitted by scipy's lstsq.
    "script_linear": {"rel": 1e-9, "abs": 1e-9, "proba": 1e-9, "label_gap": 0.0},
    # Training script, LogisticRegression(newton-cholesky, tol=1e-8) REFITTED, against Rust's
    # optimum (export_golden: ≤ 1e-6 vs the optimum; Rust ≤ 1e-5 vs the converged reference).
    "script_logistic": {"rel": 1e-9, "abs": 1e-9, "proba": 1e-5, "label_gap": 2e-5},
    "sk_tree": {"rel": 1e-12, "abs": 1e-12, "proba": 1e-12, "label_gap": 0.0},
    # Keras artefacts (float32): the package's own check uses rtol=1e-5, atol=1e-4.
    "keras": {"rel": 1e-5, "abs": 1e-4, "proba": 1e-4, "label_gap": 2e-4},
    # TFLite may fuse operations.
    "tflite": {"rel": 1e-4, "abs": 1e-4, "proba": 1e-4, "label_gap": 2e-4},
    "numpy": {"rel": 1e-5, "abs": 1e-5, "proba": 1e-5, "label_gap": 2e-5},
    # KNN: an artefact against an answer computed here (same algorithm, float64).
    "knn_exact": {"rel": 1e-9, "abs": 1e-9, "proba": 1e-9, "label_gap": 0.0},
    # KNN ONNX against the answer computed here in the float32-input space: the graph
    # computes in float64 but its outputs are float32.
    "knn_onnx_exact": {"rel": 1e-6, "abs": 1e-6, "proba": 1e-6, "label_gap": 0.0},
    # KNN ONNX against Rust: the input is rounded to float32 (relative error ≤ 6e-8 per value)
    # and the weights 1/d move by about that much over d. Beyond this the row must match the
    # float32-space answer exactly and is counted as tolerated ("float32").
    "onnx_knn": {"rel": 1e-6, "abs": 1e-6, "proba": 1e-5, "label_gap": 0.0},
    # KNN distances (Rust, scikit-learn's kneighbors, numpy here).
    "knn_distance": {"rel": 1e-9, "abs": 1e-12, "proba": 0.0, "label_gap": 0.0},
}
for _k, _t in TOL.items():
    assert _t["label_gap"] <= 2 * _t["proba"] + 1e-300 or _t["label_gap"] == 0.0, _k
# Two distances are tied when they differ by at most this, relative to max(1, d).
KNN_TIE_REL = 1e-9
# At most this fraction of the rows (and at least KNN_TOLERATED_MIN rows) may differ from
# Rust because of a tie or the float32 input (MEMORY.md: kd-tree ties depend on the C++
# library; Titanic has many identical passengers). Observed (2026-10-20): at most 7 of 268
# rows (Titanic K=5, ONNX) on macOS arm64, and 7 (ONNX) / 4 (scikit-learn) on Linux x86-64.
KNN_TOLERATED_MAX = 0.05
KNN_TOLERATED_MIN = 3
# Relative rounding noise of a sum of weights 1/d: two classes this close count as tied.
KNN_VOTE_NOISE = 1e-12
# More valid answers than this under a tie is a failure (not expected in the corpus).
KNN_MAX_ANSWERS = 20000

# Snippets: the names their comments say they assume, taken from the training script.
TREE_GLUE = "tree_model = modelo\nfeature_names = nombres\n"
GLUE: dict[str, str] = {
    "tree_viz": TREE_GLUE,
    "tree_text": TREE_GLUE,
    "decision_path": TREE_GLUE,
}
# The list of names (`clases`, by code) is accepted only if classes_ == 0..n-1; otherwise
# the snippet must raise ValueError instead of naming the wrong class.
_LIST_PATH = (
    "def _with_list(fn):\n"
    "    _l = globals().get('clases')\n"
    "    if _l is None:\n"
    "        return None\n"
    "    try:\n"
    "        return ('ok', fn(_l))\n"
    "    except ValueError as _e:\n"
    "        return ('ValueError', str(_e))\n")
# Code run after each snippet; it leaves what is compared in RESULT.
AFTER: dict[str, str] = {
    "confusion_matrix": "RESULT = {'cm': cm.tolist(), "
                        "'labels': [t.get_text() for t in ax.get_xticklabels()]}\n",
    "precision_by_class": "RESULT = {k: float(v) for k, v in prec_by_class.items()}\n",
    "decision_boundary": (
        "import numpy as _np\n"
        "_X = X if SPEC['lesson'] == 'knn' else X_train\n"
        "RESULT = {'same_columns': bool(_np.array_equal(\n"
        "    X_2d, _np.asarray(_X, dtype=float)[:, SPEC['pair']])),\n"
        "    'legend': [t.get_text() for t in ax.get_legend().get_texts()]}\n"),
    "tree_text": _LIST_PATH + (
        "RESULT = {'text': tree_text,\n"
        "          'list': _with_list(lambda l: get_tree_text(tree_model, feature_names, l))}\n"),
    "decision_path": _LIST_PATH + (
        "import contextlib, io\nRESULT = {'paths': []}\n"
        "def _path(i, names):\n"
        "    _buf = io.StringIO()\n"
        "    with contextlib.redirect_stdout(_buf):\n"
        "        mostrar_camino_decision(tree_model, X_test.iloc[[i]], feature_names, names)\n"
        "    return _buf.getvalue()\n"
        "for _i in range(len(X_test)):\n"
        "    RESULT['paths'].append(_path(_i, nombre_clase))\n"
        "RESULT['list'] = _with_list(lambda l: _path(0, l))\n"),
}
# Templates of content/code that the Rust app never offers as runnable code.
NOT_RUN: dict[str, str] = {
    "arbol_python.py.j2": "literal text of Python's export_tree_as_python_code (does not compile; "
                          "the app exports arbol_python_corregido.py.j2, D44)",
    "red_arquitectura.py.j2": "literal text of Python (NameError/SyntaxError); replaced by "
                              "red_entrenamiento.py.j2 (D74)",
    "red_cargar_completo.py.j2": "literal text of Python (safe_get_output_size undefined); "
                                 "replaced by the Keras package (D75)",
    "red_predictor_plantilla.py.j2": "literal text of Python (predict() returns zeros); "
                                     "replaced by red_predictor_numpy.py.j2 (D75)",
    "regresion_codigo.py.j2": "literal pickle snippet of Python (valor1, valor2…); 📄 Generar "
                              "Código renders regresion_entrenamiento.py.j2 instead (D29)",
}
TRAINING_KEYS = {"regression_code", "tree_training", "knn_training", "nn_training"}

# --- the child process -------------------------------------------------------------------
# Runs a script with `python -I -X utf8` in its folder: no network (socket blocked),
# matplotlib Agg, then optional probe code and snippets in the namespace the script leaves.
RUNNER = r'''
import json, runpy, socket, sys, traceback

SPEC = json.load(open(sys.argv[1], encoding="utf-8"))


def _offline(*args, **kwargs):
    raise OSError("verify_exports: sin red")


socket.socket.connect = _offline
socket.socket.connect_ex = _offline
socket.create_connection = _offline
socket.getaddrinfo = _offline

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

result = {"ok": True, "snippets": {}}


def dump():
    with open(SPEC["out"], "w", encoding="utf-8") as f:
        json.dump(result, f)


def run_main():
    if not SPEC.get("script"):
        return {"__name__": "__main__"}
    try:
        return runpy.run_path(SPEC["script"], run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
        raise RuntimeError("the script called sys.exit(0) before finishing")


try:
    g = run_main()
    plt.close("all")
except BaseException:
    result.update(ok=False, error=traceback.format_exc(limit=6))
    dump()
    sys.exit(1)

g["SPEC"] = SPEC
if SPEC.get("probe"):
    ns = dict(g)
    ns["g"] = g
    try:
        exec(compile(SPEC["probe"], "<probe>", "exec"), ns)
        result["probe"] = ns.get("RESULT")
    except BaseException:
        result.update(ok=False, error="probe:\n" + traceback.format_exc(limit=6))
for snip in SPEC.get("snippets", []):
    ns = dict(g)
    entry = {"ok": True}
    try:
        exec(compile(snip.get("glue", ""), "<glue>", "exec"), ns)
        with open(snip["file"], encoding="utf-8") as f:
            exec(compile(f.read(), snip["file"], "exec"), ns)
        if snip.get("after"):
            exec(compile(snip["after"], "<after>", "exec"), ns)
        entry["probe"] = ns.get("RESULT")
    except BaseException:
        entry = {"ok": False, "error": traceback.format_exc(limit=6)}
    plt.close("all")
    result["snippets"][snip["name"]] = entry
dump()
'''

PROBE_TRAINING = r'''
import numpy as np
from threadpoolctl import threadpool_limits
m = g["modelo"]
X_test = g["X_test"]
RESULT = {"y_pred": np.asarray(g["y_pred"]).ravel().tolist()}
if hasattr(X_test, "index"):
    RESULT["test_positions"] = [int(g["X"].index.get_loc(i)) for i in X_test.index]
if hasattr(m, "classes_"):
    RESULT["classes"] = np.asarray(m.classes_).tolist()
    if hasattr(m, "predict_proba"):
        RESULT["proba"] = m.predict_proba(X_test).tolist()
if hasattr(m, "kneighbors"):
    with threadpool_limits(1):
        RESULT["distances"] = m.kneighbors(X_test)[0].tolist()
'''

PROBE_SKLEARN_PACKAGE = r'''
import json
import pickle
import numpy as np
from threadpoolctl import threadpool_limits
data = json.load(open(SPEC["data"], encoding="utf-8"))
with open(SPEC["pkl"], "rb") as f:
    obj = pickle.load(f)
model = obj["model"] if isinstance(obj, dict) else obj
X = np.asarray(data["inputs"], dtype=float)
with threadpool_limits(1):
    RESULT = {"y_pred": np.asarray(model.predict(X)).ravel().tolist()}
    if hasattr(model, "classes_"):
        RESULT["classes"] = np.asarray(model.classes_).tolist()
        if hasattr(model, "predict_proba"):
            RESULT["proba"] = model.predict_proba(X).tolist()
    if hasattr(model, "kneighbors"):
        RESULT["distances"] = model.kneighbors(X)[0].tolist()
if SPEC.get("rules"):
    from sklearn.tree import export_text
    names = data["class_names"]
    class_names = None if names is None else [str(names[int(c)]) for c in data["model_classes"]]
    RESULT["rules"] = export_text(model, feature_names=data["feature_names"],
                                  class_names=class_names, show_weights=False)
'''

PROBE_TREE_USAGE = r'''
import json
data = json.load(open(SPEC["data"], encoding="utf-8"))
def _plain(v):
    return v.item() if hasattr(v, "item") else v
RESULT = {"pred": [_plain(predecir(list(row))) for row in data["inputs"]]}
if task_type == "Clasificación":
    _p = [predecir_con_probabilidades(list(row)) for row in data["inputs"]]
    RESULT["proba_pred"] = [_plain(r["prediccion"]) for r in _p]
    RESULT["proba_keys"] = [list(r["probabilidades"]) for r in _p]
    RESULT["proba"] = [[float(v) for v in r["probabilidades"].values()] for r in _p]
'''

PROBE_TREE_FUNCTION = r'''
import json, re
data = json.load(open(SPEC["data"], encoding="utf-8"))
_src = open(SPEC["tree_python"], encoding="utf-8").read()
_name = re.search(r"^def (\w+)\(", _src, re.M).group(1)
RESULT = [globals()[_name](*row) for row in data["inputs"]]
'''

PROBE_KERAS = r'''
import json
import pickle
import numpy as np
import tensorflow as tf
import keras
data = json.load(open(SPEC["data"], encoding="utf-8"))
X = np.asarray(data["inputs"], dtype=float)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
Xs = scaler.transform(X).astype(np.float32)
RESULT = {}
RESULT["keras"] = keras.models.load_model("modelo_red.keras").predict(Xs, verbose=0).tolist()
RESULT["h5"] = keras.models.load_model("modelo_red.h5", compile=False).predict(Xs, verbose=0).tolist()
saved = tf.saved_model.load("modelo_red_savedmodel")
RESULT["savedmodel"] = np.asarray(saved.serve(tf.constant(Xs))).tolist()
interp = tf.lite.Interpreter(model_path="modelo_red.tflite")
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
interp.resize_tensor_input(inp["index"], list(Xs.shape))
interp.allocate_tensors()
interp.set_tensor(inp["index"], Xs)
interp.invoke()
RESULT["tflite"] = interp.get_tensor(out["index"]).tolist()
with open("modelo_completo.pkl", "rb") as f:
    full = pickle.load(f)
Xf = full["scaler"].transform(X).astype(np.float32)
RESULT["modelo_completo"] = full["model"].predict(Xf, verbose=0).tolist()
'''

PROBE_NUMPY = r'''
import json
import numpy as np
data = json.load(open(SPEC["data"], encoding="utf-8"))
RESULT = np.asarray(g["predecir"](data["inputs"])).tolist()
'''

PROBE_NN_TRAINING = r'''
import numpy as np
RESULT = {k: float(g[k]) for k in ("accuracy", "r2", "mse", "mae") if k in g}
y_test = g["y_test"]
RESULT["n_test"] = int(len(y_test))
RESULT["test_positions"] = [int(i) for i in y_test.index]
RESULT["y_test"] = np.asarray(y_test).ravel().tolist()
RESULT["y_pred"] = np.asarray(g["y_pred"]).ravel().tolist()
h = g["historial"].history
RESULT["min_loss"] = float(min(min(h[k]) for k in ("loss", "val_loss") if k in h))
'''


@dataclass
class Row:
    case: str
    check: str
    ok: bool
    detail: str
    seconds: float = 0.0


@dataclass
class Ctx:
    root: Path
    tools: Path
    work: Path
    timeout: int
    env: dict[str, str] = field(default_factory=dict)


# --- helpers -----------------------------------------------------------------------------
def load_case(root: Path, name: str) -> dict:
    return json.loads((root / name / "case.json").read_text("utf-8"))


def fresh_dir(ctx: Ctx, prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=ctx.work))


def run_child(ctx: Ctx, cwd: Path, spec: dict) -> tuple[dict, str]:
    """Runs RUNNER with `python -I -X utf8` in `cwd`; returns its result and output."""
    spec_dir = Path(tempfile.mkdtemp(prefix="spec-", dir=ctx.tools))
    spec = dict(spec, out=str(spec_dir / "result.json"))
    spec_path = spec_dir / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", str(ctx.tools / "runner.py"), str(spec_path)],
            cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=ctx.env, timeout=ctx.timeout)
        output = (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout ({ctx.timeout} s)"}, ""
    out = spec_dir / "result.json"
    if not out.exists():
        return {"ok": False, "error": f"exit {proc.returncode} without result\n{output[-3000:]}"}, output
    return json.loads(out.read_text("utf-8")), output


def data_file(ctx: Ctx, case: dict) -> Path:
    path = Path(tempfile.mkdtemp(prefix="data-", dir=ctx.tools)) / "data.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    return path


def unpack(ctx: Ctx, archive: Path, prefix: str) -> Path:
    """The ZIP in a new, empty folder (only plain names: no paths, drives or '..')."""
    target = fresh_dir(ctx, prefix)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if (any(ch in name for ch in "/\\:") or name.startswith(".") or info.is_dir()
                    or not name):
                raise ValueError(f"unexpected entry {name!r} in {archive.name}")
            (target / name).write_bytes(zf.read(info))
    return target


def first_error(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " | ".join(lines[-3:])[:400]


def template_of(case: dict, key: str) -> str:
    return case["code"][key]["template"]


def package_model_json(ctx: Ctx, name: str, case: dict) -> dict:
    with zipfile.ZipFile(ctx.root / name / case["files"]["package"]) as zf:
        return json.loads(zf.read("modelo.json"))


# --- comparisons -------------------------------------------------------------------------
def code_of_label(case: dict, label: Any, classes: list) -> int | None:
    """Dataset class code of a label predicted by a scikit-learn/Keras model: the label's
    position in `classes` (the model's classes_, sorted like model_classes) indexes
    model_classes. None if the label is not a class of the model."""
    try:
        return case["model_classes"][classes.index(label)]
    except ValueError:
        return None


def onnx_label_of_code(case: dict, code: int) -> int:
    """`classlabels_ints` rule of mltutor-export (onnx_class_labels): the class values when
    all of them are distinct integers; the codes otherwise. Networks: always the code."""
    if case["lesson"] == "redes":
        return code
    values = [case["class_values"][c] for c in case["model_classes"]]
    ints = all(math.isfinite(v) and float(v).is_integer() and abs(v) < 2.0**63 for v in values)
    if ints and len(set(values)) == len(values):
        return int(case["class_values"][code])
    return code


def code_of_onnx_label(case: dict, label: int) -> int | None:
    for code in case["model_classes"]:
        if onnx_label_of_code(case, code) == int(label):
            return code
    return None


def top_gap(proba: list[float] | None) -> float:
    if not proba or len(proba) < 2:
        return math.inf
    s = sorted(proba, reverse=True)
    return s[0] - s[1]


def close(a: float, b: float, tol: dict) -> bool:
    return abs(a - b) <= tol["abs"] + tol["rel"] * abs(b)


def compare(case: dict, kind: str, *, codes: list | None = None, proba: list | None = None,
            values: list | None = None, names: list | None = None,
            rust: list | None = None) -> tuple[bool, str]:
    """Row-by-row comparison with Rust (`case["rust"]` or `rust`). `codes` are the
    artefact's labels as dataset class codes; `names` its class names; `proba` in the order
    of model_classes; `values` for regressors. Fails on 0 rows, on a different number of
    rows, or on a row where nothing was compared."""
    tol = TOL[kind]
    rust = case["rust"] if rust is None else rust
    n = len(rust)
    if n == 0:
        return False, "no rows to compare"
    for label, given in (("labels", codes), ("names", names), ("probabilities", proba),
                         ("values", values)):
        if given is not None and len(given) != n:
            return False, f"{len(given)} {label} for {n} Rust rows"
    bad: list[str] = []
    gap_tolerated = 0
    max_err = max_perr = 0.0
    for i, r in enumerate(rust):
        compared = False
        if "value" in r:
            if values is not None:
                compared = True
                got = float(np.ravel(values[i])[0])
                if close(got, r["value"], tol):
                    max_err = max(max_err, abs(got - r["value"]) / max(1.0, abs(r["value"])))
                else:
                    bad.append(f"row {i}: {got!r} vs {r['value']!r}")
            if not compared:
                bad.append(f"row {i}: nothing compared")
            continue
        if codes is not None or names is not None:
            compared = True
            ok_label = (codes[i] == r["code"]) if codes is not None else (names[i] == r["name"])
            if not ok_label:
                if top_gap(r.get("proba")) <= tol["label_gap"]:
                    gap_tolerated += 1
                else:
                    got = codes[i] if codes is not None else names[i]
                    bad.append(f"row {i}: label {got!r} vs {r['name']!r} (code {r['code']})")
        if proba is not None and r.get("proba") is not None:
            compared = True
            got = np.asarray(proba[i], dtype=float).ravel()
            exp = np.asarray(r["proba"], dtype=float)
            if got.shape != exp.shape:
                bad.append(f"row {i}: {got.size} probabilities vs {exp.size}")
            else:
                err = float(np.max(np.abs(got - exp)))
                if err <= tol["proba"]:
                    max_perr = max(max_perr, err)
                else:
                    bad.append(f"row {i}: probabilities differ by {err:.2g}")
        if not compared:
            bad.append(f"row {i}: nothing compared")
    detail = f"{n} rows"
    if max_err:
        detail += f", max rel err {max_err:.1e}"
    if max_perr:
        detail += f", max proba err {max_perr:.1e}"
    if gap_tolerated:
        detail += f", {gap_tolerated} labels within label_gap"
    if bad:
        return False, f"{len(bad)} rows differ: " + "; ".join(bad[:3])
    return True, detail


def sklearn_codes(case: dict, labels: list, classes: list) -> list:
    return [code_of_label(case, lab, classes) for lab in labels]


def nn_outputs(case: dict, out: list) -> dict:
    """Network outputs → keyword arguments of `compare` (codes, proba or values)."""
    arr = np.asarray(out, dtype=float)
    if case["task"] == "regression":
        return {"values": arr.reshape(len(arr), -1)[:, 0].tolist()}
    codes = case["model_classes"]
    if arr.ndim == 1 or arr.shape[1] == 1:
        p1 = arr.reshape(-1)
        return {"codes": [codes[int(p > 0.5)] for p in p1],
                "proba": np.stack([1 - p1, p1], axis=1).tolist()}
    return {"codes": [codes[int(j)] for j in np.argmax(arr, axis=1)], "proba": arr.tolist()}


# --- KNN under ties ----------------------------------------------------------------------
@dataclass
class KnnRow:
    distances: np.ndarray      # the K smallest, sorted
    tie: bool                  # more rows at the K-th distance than places
    zero: bool                 # a training row at distance exactly 0
    answers: list[dict]        # every answer valid under the tie


class KnnReference:
    """The KNN of the package (X_train, y_train, params), recomputed in numpy."""

    def __init__(self, case: dict, model_json: dict):
        cfg = case["config"]
        self.case = case
        self.k = int(cfg["n_neighbors"])
        self.weights = cfg["weights"]
        self.manhattan = cfg["metric"] == "manhattan"
        self.X = np.asarray(model_json["X_train"], dtype=float)
        y = model_json["y_train"]
        if case["task"] == "classification":
            classes = model_json["classes_"]
            self.y = np.array([classes.index(v) for v in y])  # position in model_classes
            self.n_classes = len(classes)
        else:
            self.y = np.asarray(y, dtype=float)
            self.n_classes = 0

    def _weights(self, d: np.ndarray, zero_rule: bool) -> np.ndarray:
        if self.weights == "uniform":
            return np.ones_like(d)
        if zero_rule:
            return (d == 0.0).astype(float)
        return 1.0 / d

    def rows(self, Q: np.ndarray) -> list[KnnRow]:
        out = []
        for q in Q:
            diff = self.X - q
            d = np.abs(diff).sum(axis=1) if self.manhattan else np.sqrt((diff * diff).sum(axis=1))
            order = np.argsort(d, kind="stable")
            dk = d[order[self.k - 1]]
            eps = KNN_TIE_REL * max(1.0, dk)
            definite = order[d[order] < dk - eps]
            tied = np.flatnonzero(np.abs(d - dk) <= eps)
            slots = self.k - len(definite)
            zero = bool(d[order[0]] == 0.0)
            out.append(KnnRow(distances=d[order[:self.k]], tie=len(tied) > slots, zero=zero,
                              answers=self._answers(d, definite, tied, slots, zero)))
        return out

    def _labels(self, proba: np.ndarray) -> set[int]:
        """The predicted class: scikit-learn, Rust and the ONNX ArgMax all take the FIRST
        maximum, so a vote tie (1–1 with K=2) has ONE valid label. With distance weights
        the sums 1/d carry rounding noise, so a class within KNN_VOTE_NOISE of the best
        (relative) is also accepted; the first maximum always is."""
        codes = self.case["model_classes"]
        first = int(np.argmax(proba))
        labels = {codes[first]}
        if self.weights == "distance":
            best = proba[first]
            labels |= {codes[i] for i in np.flatnonzero(proba >= best * (1 - KNN_VOTE_NOISE))}
        return labels

    def _answers(self, d, definite, tied, slots, zero) -> list[dict]:
        zero_rule = self.weights == "distance" and zero
        groups: dict[Any, list[int]] = {}
        for j in tied:
            groups.setdefault(self.y[j].item(), []).append(int(j))
        keys = list(groups)
        answers = []
        for counts in compositions([len(groups[k]) for k in keys], slots):
            chosen = list(definite) + [j for key, c in zip(keys, counts) for j in groups[key][:c]]
            chosen = np.asarray(chosen, dtype=int)
            w = self._weights(d[chosen], zero_rule)
            if self.n_classes:
                votes = np.bincount(self.y[chosen], weights=w, minlength=self.n_classes)
                proba = votes / votes.sum()
                answers.append({"proba": proba, "codes": self._labels(proba)})
            else:
                answers.append({"value": float((w * self.y[chosen]).sum() / w.sum())})
            if len(answers) > KNN_MAX_ANSWERS:
                raise RuntimeError("too many answers valid under a tie")
        return answers


def compositions(limits: list[int], total: int):
    """Every (c_1..c_n) with 0 ≤ c_i ≤ limits[i] and Σ c_i = total."""
    if not limits:
        if total == 0:
            yield ()
        return
    first, rest = limits[0], limits[1:]
    for c in range(min(first, total), -1, -1):
        if total - c <= sum(rest):
            for tail in compositions(rest, total - c):
                yield (c, *tail)


def knn_matches(answer: dict, *, code=None, proba=None, value=None, tol: dict) -> bool:
    if "value" in answer:
        return value is not None and close(value, answer["value"], tol)
    if code is not None and code not in answer["codes"]:
        return False
    if proba is not None:
        return bool(np.max(np.abs(np.asarray(proba, dtype=float) - answer["proba"])) <= tol["proba"])
    return code is not None


def compare_knn(ctx: Ctx, name: str, case: dict, *, onnx: bool, codes: list | None = None,
                proba: list | None = None, values: list | None = None,
                distances: list | None = None) -> tuple[bool, str]:
    """A KNN artefact against Rust, allowing only the answers valid under a tie (or, for
    ONNX, under the float32 input), within KNN_TOLERATED_MAX."""
    rust = case["rust"]
    n = len(rust)
    if n == 0:
        return False, "no rows to compare"
    for label, given in (("labels", codes), ("probabilities", proba), ("values", values),
                         ("distance rows", distances)):
        if given is not None and len(given) != n:
            return False, f"{len(given)} {label} for {n} Rust rows"
    ref = KnnReference(case, package_model_json(ctx, name, case))
    X = np.asarray(case["inputs"], dtype=float)
    rows64 = ref.rows(X)
    rows32 = ref.rows(X.astype(np.float32).astype(float)) if onnx else rows64
    exact, dtol = TOL["knn_exact"], TOL["knn_distance"]
    art_exact = TOL["knn_onnx_exact"] if onnx else exact
    vs_rust = TOL["onnx_knn"] if onnx else exact
    bad: list[str] = []
    causes = {"empate": 0, "float32": 0, "distancia 0": 0}
    for i, r in enumerate(rust):
        r64, r32 = rows64[i], rows32[i]
        rd = np.asarray(r["knn"]["distances"], dtype=float)
        if rd.shape != r64.distances.shape or not np.all(
                np.abs(rd - r64.distances) <= dtol["abs"] + dtol["rel"] * np.abs(r64.distances)):
            bad.append(f"row {i}: Rust's distances differ from numpy's")
            continue
        if r["knn"]["tie_at_kth"] is not None and not r64.tie:
            bad.append(f"row {i}: Rust reports a tie that numpy does not see")
            continue
        rcode, rproba, rvalue = r.get("code"), r.get("proba"), r.get("value")
        if not any(knn_matches(a, code=rcode, proba=rproba, value=rvalue, tol=exact)
                   for a in r64.answers):
            bad.append(f"row {i}: Rust's answer is not valid under its own neighbours")
            continue
        if distances is not None:
            got = np.sort(np.asarray(distances[i], dtype=float))
            if got.shape != rd.shape or not np.all(
                    np.abs(got - rd) <= dtol["abs"] + dtol["rel"] * np.abs(rd)):
                bad.append(f"row {i}: kneighbors distances differ from Rust's")
                continue
        code = None if codes is None else codes[i]
        p = None if proba is None else proba[i]
        v = None if values is None else float(np.ravel(values[i])[0])
        if code is None and p is None and v is None:
            bad.append(f"row {i}: nothing compared")
            continue
        if not any(knn_matches(a, code=code, proba=p, value=v, tol=art_exact)
                   for a in r32.answers):
            bad.append(f"row {i}: {code if v is None else v!r} is not a valid answer")
            continue
        if knn_matches({"codes": {rcode}, "proba": np.asarray(rproba, dtype=float)}
                       if rvalue is None and rproba is not None else
                       ({"codes": {rcode}} if rvalue is None else {"value": rvalue}),
                       code=code, proba=p, value=v, tol=vs_rust):
            continue
        # Why it differs from Rust. A float64 answer means a choice under a float64 tie; only
        # an answer that exists solely for the float32 input is a float32 effect.
        in64 = any(knn_matches(a, code=code, proba=p, value=v, tol=art_exact)
                   for a in r64.answers)
        if onnx and r64.zero and not r32.zero:
            causes["distancia 0"] += 1
        elif in64 and r64.tie:
            causes["empate"] += 1
        elif onnx and not in64:
            causes["float32"] += 1
        else:
            bad.append(f"row {i}: differs from Rust without a tie")
    tolerated = sum(causes.values())
    cap = max(KNN_TOLERATED_MIN, math.floor(KNN_TOLERATED_MAX * n))
    detail = f"{n} rows, distances = Rust's" if distances is not None else f"{n} rows"
    n_ties = sum(r.tie for r in rows64)
    detail += f", {n_ties} with a tie at K"
    if tolerated:
        detail += ", tolerated: " + ", ".join(f"{v} {k}" for k, v in causes.items() if v)
    if tolerated > cap:
        bad.append(f"{tolerated} tolerated rows > cap {cap}")
    if bad:
        return False, f"{len(bad)} problems: " + "; ".join(bad[:3])
    return True, detail


# --- checks ------------------------------------------------------------------------------
def check_onnx(ctx: Ctx, name: str, case: dict) -> list[Row]:
    import onnxruntime as ort

    t0 = time.perf_counter()
    path = ctx.root / name / case["files"]["onnx"]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    X = np.asarray(case["inputs"], dtype=np.float32)
    outs = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, {"float_input": X}))}
    regression = case["task"] == "regression"
    kw: dict[str, Any] = {}
    if regression:
        kw["values"] = outs["variable"].tolist()
    else:
        kw["codes"] = [code_of_onnx_label(case, lab) for lab in outs["label"].tolist()]
        kw["proba"] = outs["probabilities"].tolist()
    if case["lesson"] == "knn":
        ok, detail = compare_knn(ctx, name, case, onnx=True, **kw)
    else:
        kind = {"regresion": "onnx_linear", "arboles": "onnx_tree", "redes": "onnx_nn"}[
            case["lesson"]]
        ok, detail = compare(case, kind, **kw)
        if case["lesson"] == "redes" and not regression and \
                case["config"]["arch"]["output_activation"] == "tanh":
            p = np.asarray(kw["proba"])
            detail += (f"; tanh: 'probabilities' in [{p.min():.3f}, {p.max():.3f}] "
                       "(not a probability, LEEME says so)")
    return [Row(name, "onnx (onnxruntime)", ok, detail, time.perf_counter() - t0)]


def check_sklearn_package(ctx: Ctx, name: str, case: dict) -> list[Row]:
    t0 = time.perf_counter()
    rows: list[Row] = []
    pkg = unpack(ctx, ctx.root / name / case["files"]["package"], f"{name}-paquete-")
    script = (pkg / "reconstruir_modelo.py").read_text("utf-8")
    match = re.search(r"open\([\"']([\w.-]+\.pkl)[\"'], \"wb\"\)", script)
    if match is None:
        return [Row(name, "paquete", False, "no .pkl written by reconstruir_modelo.py")]
    pkl = match.group(1)
    data = data_file(ctx, case)
    spec: dict[str, Any] = {"script": "reconstruir_modelo.py", "probe": PROBE_SKLEARN_PACKAGE,
                            "data": str(data), "pkl": pkl, "rules": case["lesson"] == "arboles"}
    if case["lesson"] == "arboles":
        usage = ctx.root / name / case["code"]["tree_usage"]["file"]
        spec["snippets"] = [{"name": "tree_usage", "file": str(usage), "after": PROBE_TREE_USAGE}]
    result, output = run_child(ctx, pkg, spec)
    label = f"paquete ({case['package_templates'][0]})"
    if not result["ok"]:
        return [Row(name, label, False, first_error(result.get("error", output)),
                    time.perf_counter() - t0)]
    if not (pkg / pkl).is_file() or "✅" not in output:
        return [Row(name, label, False, f"missing {pkl} or no ✅ in the output",
                    time.perf_counter() - t0)]
    got = result["probe"]
    regression = case["task"] == "regression"
    if case["lesson"] == "knn":
        kw: dict[str, Any] = {"distances": got["distances"]}
        if regression:
            kw["values"] = got["y_pred"]
        else:
            kw["codes"] = sklearn_codes(case, got["y_pred"], got["classes"])
            kw["proba"] = got["proba"]
        ok, detail = compare_knn(ctx, name, case, onnx=False, **kw)
    else:
        kind = "sk_tree" if case["lesson"] == "arboles" else "sk_package"
        if regression:
            ok, detail = compare(case, kind, values=got["y_pred"])
        else:
            ok, detail = compare(case, kind, codes=sklearn_codes(case, got["y_pred"],
                                                                 got["classes"]),
                                 proba=got.get("proba"))
    rows.append(Row(name, label + f" → {pkl}", ok, detail, time.perf_counter() - t0))
    if case["lesson"] == "arboles":
        rules = (ctx.root / name / case["files"]["rules"]).read_text("utf-8")
        same = rules.strip() == got["rules"].strip()
        rows.append(Row(name, "reglas .txt = export_text", same,
                        f"{len(rules.splitlines())} lines" if same else "text differs"))
        usage = result["snippets"].get("tree_usage", {})
        tlabel = f"plantilla {template_of(case, 'tree_usage')}"
        if not usage.get("ok"):
            rows.append(Row(name, tlabel, False, first_error(usage.get("error", ""))))
        else:
            u = usage["probe"]
            if regression:
                ok, detail = compare(case, "sk_tree", values=u["pred"])
            else:
                ok, detail = compare(case, "sk_tree", names=u["pred"])
                ok2, detail2 = compare(case, "sk_tree", names=u["proba_pred"], proba=u["proba"])
                want_keys = [case["class_names"][c] for c in case["model_classes"]]
                keys_ok = all(k == want_keys for k in u["proba_keys"])
                ok = ok and ok2 and keys_ok
                detail = (f"predecir: {detail}; predecir_con_probabilidades: {detail2}"
                          + ("" if keys_ok else "; wrong class names in the probabilities"))
            rows.append(Row(name, tlabel, ok, detail))
    return rows


def check_keras_package(ctx: Ctx, name: str, case: dict) -> list[Row]:
    t0 = time.perf_counter()
    pkg = unpack(ctx, ctx.root / name / case["files"]["package"], f"{name}-keras-")
    data = data_file(ctx, case)
    result, output = run_child(ctx, pkg, {"script": "reconstruir_modelo.py",
                                          "probe": PROBE_KERAS, "data": str(data)})
    label = f"paquete Keras ({case['package_templates'][0]})"
    if not result["ok"]:
        return [Row(name, label, False, first_error(result.get("error", output)),
                    time.perf_counter() - t0)]
    expected = ["modelo.json", "modelo_red.keras", "modelo_red.h5", "modelo_red_savedmodel",
                "modelo_red.tflite", "scaler.pkl", "modelo_completo.pkl"]
    missing = [f for f in expected if not (pkg / f).exists()]
    if missing or "⚠️" in output or "✅" not in output:
        return [Row(name, label, False, f"missing {missing}; output: {first_error(output)}",
                    time.perf_counter() - t0)]
    rows = [Row(name, label + ": guarda los 6 formatos", True, ", ".join(expected[1:]),
                time.perf_counter() - t0)]
    for fmt, outs in result["probe"].items():
        ok, detail = compare(case, "tflite" if fmt == "tflite" else "keras",
                             **nn_outputs(case, outs))
        rows.append(Row(name, f"  {fmt}", ok, detail))
    return rows


def check_numpy_predictor(ctx: Ctx, name: str, case: dict) -> list[Row]:
    t0 = time.perf_counter()
    pkg = unpack(ctx, ctx.root / name / case["files"]["numpy_predictor"], f"{name}-numpy-")
    text = (pkg / "predecir_numpy.py").read_text("utf-8")
    shown = (ctx.root / name / case["code"]["nn_numpy_predictor"]["file"]).read_text("utf-8")
    data = data_file(ctx, case)
    result, output = run_child(ctx, pkg, {"script": "predecir_numpy.py", "probe": PROBE_NUMPY,
                                          "data": str(data)})
    label = f"predictor NumPy ({template_of(case, 'nn_numpy_predictor')})"
    if not result["ok"]:
        return [Row(name, label, False, first_error(result.get("error", output)))]
    ok, detail = compare(case, "numpy", **nn_outputs(case, result["probe"]))
    if text != shown:
        ok, detail = False, "predecir_numpy.py differs from the code shown in the app"
    return [Row(name, label, ok, detail, time.perf_counter() - t0)]


def check_metadata(ctx: Ctx, name: str, case: dict) -> list[Row]:
    meta = json.loads((ctx.root / name / case["files"]["metadata"]).read_text("utf-8"))
    problems = []
    mi, di = meta.get("model_info", {}), meta.get("data_info", {})
    if mi.get("valid") is not True:
        problems.append("model_info.valid is not true")
    # D75: num_classes = 2 in binary (one sigmoid neuron) and null in regression.
    expected_classes = None if case["task"] == "regression" else len(case["class_names"])
    if di.get("num_classes") != expected_classes:
        problems.append(f"num_classes {di.get('num_classes')} vs {expected_classes}")
    arch = case["config"]["arch"]
    binary_one = case["task"] == "classification" and len(case["model_classes"]) == 2
    n_out = 1 if case["task"] == "regression" or binary_one else len(case["model_classes"])
    expected_arch = [len(case["feature_names"]), *arch["hidden"], n_out]
    if mi.get("architecture") != expected_arch:
        problems.append(f"architecture {mi.get('architecture')} vs {expected_arch}")
    for key, value in (("target_column", case["target_column"]),
                       ("feature_names", case["feature_names"]),
                       ("feature_columns", case["feature_columns"]),
                       ("dataset", case["dataset_label"]),
                       ("n_test", case["n_test"])):
        if di.get(key) != value:
            problems.append(f"data_info.{key} = {di.get(key)!r}")
    return [Row(name, "metadatos .json", not problems, "; ".join(problems) or
                "valid JSON, architecture, classes, columns and test size of the case")]


def stage_script_dir(ctx: Ctx, name: str, case: dict) -> Path:
    folder = fresh_dir(ctx, f"{name}-plantillas-")
    if case["csv_file"]:
        shutil.copy(ctx.root / name / case["csv_file"], folder / case["csv_file"])
    return folder


def confusion(case: dict, real: list, predicted: list) -> list[list[int]]:
    k = len(case["class_names"])
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(real, predicted):
        cm[a][b] += 1
    return cm


def check_training_script(case: dict, got: dict, label: str, seconds: float) -> tuple[Row, list]:
    """The scikit-learn training script against Rust; returns the row and the script's
    predictions as class codes (classification) for the snippets."""
    name = case["name"]
    positions = got.get("test_positions")
    if positions is None:
        return Row(name, label, False, "the script has no X_test with an index"), []
    if positions != case["input_rows"]:
        return Row(name, label, False, "the script's split differs from the app's"), []
    regression = case["task"] == "regression"
    lesson = case["lesson"]
    if lesson == "knn":
        kw: dict[str, Any] = {"distances": got["distances"]}
        if regression:
            kw["values"] = got["y_pred"]
        else:
            kw["codes"] = sklearn_codes(case, got["y_pred"], got["classes"])
            kw["proba"] = got["proba"]
        return None, kw  # compared by the caller (needs the package's X_train)
    kind = {"arboles": "sk_tree", "regresion": "script_linear" if regression
            else "script_logistic"}[lesson]
    if regression:
        ok, detail = compare(case, kind, values=got["y_pred"])
        return Row(name, label + " = Rust", ok, "split = app's; " + detail, seconds), []
    codes = sklearn_codes(case, got["y_pred"], got["classes"])
    ok, detail = compare(case, kind, codes=codes, proba=got.get("proba"))
    return Row(name, label + " = Rust", ok, "split = app's; " + detail, seconds), codes


def list_accepted(case: dict) -> bool:
    """Whether the tree snippets accept the list of names by code: the model's classes_
    (the class values the script's y holds) must be 0..n-1."""
    values = [case["class_values"][c] for c in case["model_classes"]]
    return values == list(range(len(values)))


def check_list_path(case: dict, got, with_dict) -> tuple[bool, str]:
    """`got` = the snippet called with the list `clases`: ('ok', output) or
    ('ValueError', message); None in regression."""
    if case["task"] == "regression":
        return got is None, ""
    if list_accepted(case):
        ok = got is not None and got[0] == "ok" and got[1] == with_dict
        return ok, "; list = dict" if ok else f"; the list gave {got!r}"
    ok = got is not None and got[0] == "ValueError" and "nombre_clase" in got[1]
    return ok, ("; list → ValueError (classes_ = values)" if ok
                else f"; the list should raise ValueError, got {got!r}")


def check_snippet(case: dict, key: str, entry: dict, files: dict, predicted: list,
                  root: Path) -> Row:
    name = case["name"]
    tlabel = f"  {template_of(case, key)}"
    if not entry["ok"]:
        return Row(name, tlabel, False, first_error(entry["error"]))
    res = entry.get("probe")
    if key == "confusion_matrix":
        want_labels = list(case["class_names"])
        if res["labels"] != want_labels:
            return Row(name, tlabel, False, f"labels {res['labels']} vs {want_labels}")
        k = len(want_labels)
        if not predicted:
            ok = np.asarray(res["cm"]).shape == (k, k) and sum(map(sum, res["cm"])) == case["n_test"]
            return Row(name, tlabel, ok, f"{k}×{k} with every class (Keras predictions)")
        want = confusion(case, case["targets"], predicted)
        return Row(name, tlabel, res["cm"] == want,
                   f"{k}×{k}, every class, = script's predictions" if res["cm"] == want
                   else f"cm {res['cm']} vs {want}")
    if key == "precision_by_class":
        want_names = list(case["class_names"])
        if list(res) != want_names:
            return Row(name, tlabel, False, f"classes {list(res)} vs {want_names}")
        if not predicted:
            return Row(name, tlabel, True, f"{len(res)} classes (Keras predictions)")
        cm = np.asarray(confusion(case, case["targets"], predicted), dtype=float)
        col = cm.sum(axis=0)
        want = np.divide(np.diag(cm), col, out=np.zeros_like(col), where=col > 0)
        ok = bool(np.allclose(list(res.values()), want, rtol=0, atol=1e-12))
        return Row(name, tlabel, ok, "every class (zero_division=0)" if ok
                   else f"{list(res.values())} vs {want.tolist()}")
    if key == "decision_boundary":
        if case["lesson"] == "knn":
            want = list(case["class_names"])  # D63: every row of the dataset
        else:
            want = [case["class_names"][c] for c in case["model_classes"]]
        ok = res["same_columns"] and res["legend"] == want
        return Row(name, tlabel, ok, f"columns {case['boundary_pair']}, legend {res['legend']}"
                   if ok else f"same_columns={res['same_columns']}, legend {res['legend']} vs {want}")
    if key == "tree_text":
        rules = (root / files["rules_with_weights"]).read_text("utf-8").strip()
        ok = res["text"].strip() == rules
        list_ok, list_detail = check_list_path(case, res["list"], res["text"])
        return Row(name, tlabel + " = 📝 Texto", ok and list_ok,
                   ("same text, weights and classes" if ok
                    else "differs from the app's text with weights") + list_detail)
    if key == "decision_path":
        bad = []
        list_ok, list_detail = check_list_path(case, res["list"], res["paths"][0]
                                               if res["paths"] else None)
        res = res["paths"]
        if not list_ok:
            return Row(name, tlabel, False, list_detail.lstrip("; "))
        if len(res) != len(case["rust"]):
            return Row(name, tlabel, False, f"{len(res)} paths for {len(case['rust'])} rows")
        for i, (text, r) in enumerate(zip(res, case["rust"])):
            want = ["Camino de decisión:"] + [
                f"- {s['name']} = {s['value']:.4f} {'≤' if s['goes_left'] else '>'} "
                f"{s['threshold']:.4f}" for s in r["path"]["steps"]]
            want.append(f"Predicción final: {r['name']}" if "name" in r
                        else f"Predicción final: {r['value']:.4f}")
            if text.strip().splitlines() != want:
                bad.append(i)
        return Row(name, tlabel + " = Rust", not bad,
                   (f"{len(res)} rows, every step" if not bad else f"rows {bad[:5]} differ")
                   + list_detail)
    if key == "tree_python":
        regression = case["task"] == "regression"
        ok, detail = (compare(case, "sk_tree", values=res) if regression
                      else compare(case, "sk_tree", names=res))
        exported = (root / files["tree_python"]).read_text("utf-8")
        shown = (root / case["code"]["tree_python"]["file"]).read_text("utf-8")
        if exported != shown:
            ok, detail = False, "arbol_decision.py differs from the code shown"
        return Row(name, tlabel + " (arbol_decision.py) = Rust", ok, detail)
    return Row(name, tlabel, True, "runs")


def check_templates(ctx: Ctx, name: str, case: dict) -> list[Row]:
    t0 = time.perf_counter()
    code = case["code"]
    training = next(k for k in code if k in TRAINING_KEYS)
    folder = stage_script_dir(ctx, name, case)
    data = data_file(ctx, case)
    snippets = []
    for key, entry in code.items():
        if key in TRAINING_KEYS or key in ("tree_usage", "nn_numpy_predictor"):
            continue
        snippets.append({"name": key, "file": str(ctx.root / name / entry["file"]),
                         "glue": GLUE.get(key, ""),
                         "after": PROBE_TREE_FUNCTION if key == "tree_python" else AFTER.get(key, "")})
    spec = {"script": str(ctx.root / name / code[training]["file"]),
            "probe": PROBE_NN_TRAINING if case["lesson"] == "redes" else PROBE_TRAINING,
            "data": str(data), "snippets": snippets, "lesson": case["lesson"],
            "pair": case["boundary_pair"],
            "tree_python": str(ctx.root / name / code["tree_python"]["file"])
            if "tree_python" in code else ""}
    result, output = run_child(ctx, folder, spec)
    label = f"plantilla {template_of(case, training)}"
    seconds = time.perf_counter() - t0
    if not result["ok"]:
        return [Row(name, label, False, first_error(result.get("error", output)), seconds)]
    got = result["probe"]
    rows: list[Row] = []
    predicted: list = []
    if case["lesson"] == "redes":
        rows.append(check_nn_training(case, got, label, seconds))
        if case["task"] == "classification":
            predicted = []  # Keras retrains: its predictions are not Rust's
    else:
        row, extra = check_training_script(case, got, label, seconds)
        if row is None:  # KNN
            ok, detail = compare_knn(ctx, name, case, onnx=False, **extra)
            row = Row(name, label + " = Rust", ok, "split = app's; " + detail, seconds)
            predicted = extra.get("codes") or []
        else:
            predicted = extra
        rows.append(row)
    for key, entry in result["snippets"].items():
        rows.append(check_snippet(case, key, entry, case["files"], predicted, ctx.root / name))
    return rows


def check_nn_training(case: dict, got: dict, label: str, seconds: float) -> Row:
    name = case["name"]
    problems = []
    if got["n_test"] != case["n_test"]:
        problems.append(f"n_test {got['n_test']} vs {case['n_test']}")
    if got["test_positions"] != case["input_rows"]:
        problems.append("the split differs from the app's (D72)")
    if not got["min_loss"] >= 0:
        problems.append(f"negative loss {got['min_loss']:.4g}")
    rust = case["rust_metrics"]
    if case["task"] == "classification":
        counts = np.bincount(np.asarray(got["y_test"], dtype=int))
        majority = counts.max() / counts.sum()
        acc = got.get("accuracy", float("nan"))
        if not acc > majority:
            problems.append(f"accuracy {acc:.3f} ≤ majority class {majority:.3f}")
        detail = (f"accuracy {acc:.3f} (majority {majority:.3f}, Rust "
                  f"{rust.get('accuracy', float('nan')):.3f}), losses ≥ 0, split = app's")
    else:
        r2 = got.get("r2", float("nan"))
        if not r2 > 0:
            problems.append(f"R² {r2:.3f} ≤ 0 (worse than the mean)")
        detail = f"R² {r2:.3f} (Rust {rust.get('r2', float('nan')):.3f}), losses ≥ 0, split = app's"
    return Row(name, label + " (Keras)", not problems, "; ".join(problems) or detail, seconds)


def check_static_templates(ctx: Ctx) -> list[Row]:
    """📊 Datos' exploration snippet (the UI shows the file as is), run after loading Iris
    into `df` and choosing 3 columns, the two lines it asks the student to replace."""
    import jinja2

    t0 = time.perf_counter()
    text = jinja2.Environment(keep_trailing_newline=True, undefined=jinja2.StrictUndefined
                              ).from_string((CODE_DIR / "analisis_exploratorio.py.j2")
                                            .read_text("utf-8")).render()
    placeholder = "selected_features = ['feature1', 'feature2', 'feature3']"
    if placeholder not in text:
        return [Row("datos", "plantilla analisis_exploratorio.py.j2", False,
                    "the placeholder line changed: update check_static_templates")]
    text = text.replace(placeholder, "selected_features = list(X.columns[:3])")
    folder = fresh_dir(ctx, "datos-")
    (folder / "analisis_exploratorio.py").write_text(text, encoding="utf-8")
    glue = ("from sklearn.datasets import load_iris\n"
            "df = load_iris(as_frame=True).frame\n")
    result, output = run_child(ctx, folder, {
        "script": None,
        "snippets": [{"name": "analisis", "glue": glue,
                      "file": str(folder / "analisis_exploratorio.py")}]})
    entry = result.get("snippets", {}).get("analisis", {})
    ok = bool(result["ok"] and entry.get("ok"))
    detail = "runs (df = Iris, selected_features = 3 columns)" if ok else first_error(
        entry.get("error") or result.get("error", output))
    return [Row("datos", "plantilla analisis_exploratorio.py.j2", ok, detail,
                time.perf_counter() - t0)]


def coverage(cases: dict[str, dict]) -> list[Row]:
    """Every template of content/code is run here, shipped as text in a package, or listed
    in NOT_RUN. Driven by the template names Rust recorded in case.json."""
    run = {"analisis_exploratorio.py.j2"}
    shipped: set[str] = set()
    for case in cases.values():
        run |= {entry["template"] for entry in case["code"].values()}
        for t in case["package_templates"]:
            (shipped if t.endswith(".txt.j2") else run).add(t)
    rows = []
    for path in sorted(CODE_DIR.glob("*.j2")):
        if path.name in run:
            continue
        if path.name in shipped:
            rows.append(Row("-", f"texto del paquete: {path.name}", True,
                            "LEEME.txt inside the ZIP (not code)"))
        elif path.name in NOT_RUN:
            rows.append(Row("-", f"no se ejecuta: {path.name}", True, NOT_RUN[path.name]))
        else:
            rows.append(Row("-", f"plantilla sin cubrir: {path.name}", False,
                            "add a corpus case or a NOT_RUN entry"))
    return rows


def case_checks(case: dict) -> list[Callable[[Ctx, str, dict], list[Row]]]:
    checks: list[Callable[[Ctx, str, dict], list[Row]]] = [check_onnx]
    if case["lesson"] == "redes":
        checks += [check_keras_package, check_numpy_predictor, check_metadata]
    else:
        checks.append(check_sklearn_package)
    if case["templates_offline"]:
        checks.append(check_templates)
    return checks


def self_test() -> None:
    """Unit checks of the KNN reference (run before every verification)."""
    # K=2 uniform, query 0: class 0 at distance 1, class 1 at distance 2. No neighbour tie,
    # but a 1–1 vote tie: scikit-learn, Rust and ONNX predict the FIRST class, 0, only.
    case = {"config": {"n_neighbors": 2, "weights": "uniform", "metric": "euclidean"},
            "task": "classification", "model_classes": [0, 1]}
    ref = KnnReference(case, {"X_train": [[1.0], [2.0], [10.0]], "y_train": [0, 1, 1],
                              "classes_": [0, 1]})
    row = ref.rows(np.array([[0.0]]))[0]
    tol = TOL["knn_onnx_exact"]
    assert not row.tie and len(row.answers) == 1, row
    assert any(knn_matches(a, code=0, proba=[.5, .5], tol=tol) for a in row.answers)
    assert not any(knn_matches(a, code=1, proba=[.5, .5], tol=tol) for a in row.answers)
    # A real tie at the K-th distance: both choices are valid.
    tie = KnnReference(case, {"X_train": [[1.0], [-2.0], [2.0]], "y_train": [0, 0, 1],
                              "classes_": [0, 1]}).rows(np.array([[0.0]]))[0]
    assert tie.tie and len(tie.answers) == 2, tie
    assert any(knn_matches(a, code=0, proba=[1.0, 0.0], tol=tol) for a in tie.answers)
    assert any(knn_matches(a, code=0, proba=[.5, .5], tol=tol) for a in tie.answers)
    assert not any(knn_matches(a, code=1, proba=[.5, .5], tol=tol) for a in tie.answers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, help="folder written by export_corpus")
    parser.add_argument("--only", action="append", default=[], help="case name substring")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=int, default=300, help="seconds per script")
    parser.add_argument("--keep", action="store_true", help="keep the temporary folders")
    args = parser.parse_args(argv)
    self_test()

    root = args.corpus.resolve()
    index = json.loads((root / "index.json").read_text("utf-8"))
    names = [n for n in index["cases"] if not args.only or any(o in n for o in args.only)]
    if not names:
        print("no case matches", file=sys.stderr)
        return 1
    cases = {n: load_case(root, n) for n in names}
    base = Path(tempfile.mkdtemp(prefix="verify-exports-"))
    tools, work = base / "tools", base / "work"
    tools.mkdir()
    work.mkdir()
    (tools / "runner.py").write_text(RUNNER, encoding="utf-8")
    seaborn = base / "seaborn-data"
    shutil.copytree(SEABORN_DATA, seaborn)
    (base / "sklearn-data").mkdir()
    (base / "keras-home").mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({
        "MPLBACKEND": "Agg",
        "SEABORN_DATA": str(seaborn),
        # An empty scikit-learn cache: fetch_* would need the network, which is blocked.
        "SCIKIT_LEARN_DATA": str(base / "sklearn-data"),
        "KERAS_HOME": str(base / "keras-home"),
        # One thread: scikit-learn's KNN ties depend on the number of threads (MEMORY.md).
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # D93: always on the CPU (the templates hide the GPU too).
        "CUDA_VISIBLE_DEVICES": "",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_ENABLE_ONEDNN_OPTS": "0",
    })
    ctx = Ctx(root=root, tools=tools, work=work, timeout=args.timeout, env=env)

    started = time.perf_counter()
    jobs: list[tuple[str, Callable]] = [("datos", lambda: check_static_templates(ctx))]
    for name, case in cases.items():
        for check in case_checks(case):
            jobs.append((name, (lambda c, n, d: (lambda: c(ctx, n, d)))(check, name, case)))
    results: dict[int, list[Row]] = {}
    with futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        pending = {pool.submit(fn): i for i, (_, fn) in enumerate(jobs)}
        for fut in futures.as_completed(pending):
            i = pending[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # a crash in a check is a failure, not an abort
                results[i] = [Row(jobs[i][0], "check", False, f"{type(exc).__name__}: {exc}")]
    rows = [row for i in sorted(results) for row in results[i]]
    if not args.only:
        rows += coverage(cases)

    width = max(len(r.case) for r in rows)
    cwidth = max(len(r.check) for r in rows)
    for r in rows:
        status = "ok  " if r.ok else "FAIL"
        secs = f"{r.seconds:6.1f}s" if r.seconds else "       "
        print(f"{status} {r.case:<{width}}  {r.check:<{cwidth}} {secs}  {r.detail}")
    failed = [r for r in rows if not r.ok]
    print(f"\n{len(rows) - len(failed)} ok, {len(failed)} failed, {len(cases)} cases, "
          f"{time.perf_counter() - started:.0f} s (python {sys.version.split()[0]})")
    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)
    else:
        print(f"temporary folders kept in {base}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
