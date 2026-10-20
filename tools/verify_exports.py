#!/usr/bin/env python3
"""Cross-check every export and code template of the Rust app against Python (Phase 6).

Replaces the old ``--verify-models`` of the Python app. The corpus comes from the Rust
side, through the same code paths the app uses (``src-tauri/src/export_corpus.rs``):

    MLTUTOR_EXPORT_CORPUS=/tmp/corpus cargo test -p mltutor export_corpus -- --ignored
    uv run --no-project --python 3.11 --with-requirements tools/verify_exports.requirements.txt \\
        python tools/verify_exports.py /tmp/corpus

(on the Apple Silicon Mac of the project: ``--python .venv/bin/python``, see MEMORY.md).

For every case of the corpus it checks:

* ``onnx``: ``modelo.onnx`` in onnxruntime (float32 input, like skl2onnx) against the
  predictions Rust computed for the same rows (``case.json``, the 🔮 form path).
* ``paquete``: the ZIP is unpacked in a fresh temporary folder and its
  ``reconstruir_modelo.py`` runs with ``python -I`` and without network. It must rebuild the
  model, pass its own check and write every file D1 promises (``.pkl``; for networks
  ``.keras``, ``.h5``, SavedModel, ``.tflite``, ``scaler.pkl`` and ``modelo_completo.pkl``).
  Every saved artefact is then loaded again and must reproduce Rust's predictions on all
  the inputs. Trees: the rules ``.txt`` must equal ``export_text`` of the rebuilt model and
  ``usar_modelo.py`` (``usar_modelo_arbol.py.j2``) must give Rust's class names/values.
* ``numpy``: 🧠 ``red_predictor_numpy.zip`` (``predecir_numpy.py``) against Rust.
* ``metadatos``: 🧠 metadata JSON (valid JSON, consistent with the case).
* ``plantillas``: every template the lesson offers, as the app rendered it (``code/*.py``),
  executed offline: the training script first (its split must be the app's and, for the
  scikit-learn models, its predictions must be Rust's), then each snippet in the namespace
  the training script leaves, with the few names the snippet says it assumes
  (``GLUE``). The tree's Python function (``arbol_decision.py``) must predict like Rust.

Tolerances (``TOL``) are documented next to each comparison. KNN ties: a row is
"ambiguous" when, computed here in float64 from the package's ``X_train``, the distance of
the K-th and (K+1)-th neighbours are equal within 1e-9 relative, or when rounding the input
to float32 (ONNX) changes the set of K neighbours. Only ambiguous rows may differ.

Templates that are not runnable on purpose (Python's literal text kept for the text parity
tests and never offered by the Rust app) are listed in ``NOT_RUN`` with the reason.
Exit status 1 if any check fails.
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
TOL: dict[str, dict[str, float]] = {
    # onnxruntime computes in float32 from float32 inputs; Rust in float64 from the original
    # values. Values: |ort − rust| ≤ abs + rel·|rust|.
    "onnx_linear": {"rel": 1e-5, "abs": 1e-4, "proba": 1e-5, "label_gap": 1e-4},
    # Trees: float32 comparisons on both sides (sklearn casts X to float32), so the leaf is
    # the same; leaf values/weights are stored as float32 in the ONNX.
    "onnx_tree": {"rel": 1e-6, "abs": 1e-6, "proba": 1e-6, "label_gap": 0.0},
    # KNN: float64 inside the graph, but the input is rounded to float32 first.
    "onnx_knn": {"rel": 1e-6, "abs": 1e-6, "proba": 1e-5, "label_gap": 0.0},
    # MLP: float32 on both sides.
    "onnx_nn": {"rel": 1e-5, "abs": 1e-5, "proba": 1e-5, "label_gap": 1e-5},
    # Rebuilt scikit-learn objects (float64, the same algorithm as Rust).
    "sk_linear": {"rel": 1e-9, "abs": 1e-9, "proba": 1e-6, "label_gap": 1e-6},
    "sk_tree": {"rel": 1e-12, "abs": 1e-12, "proba": 1e-12, "label_gap": 0.0},
    "sk_knn": {"rel": 1e-9, "abs": 1e-9, "proba": 1e-9, "label_gap": 0.0},
    # Keras artefacts (float32): the package's own check uses rtol=1e-5, atol=1e-4.
    "keras": {"rel": 1e-5, "abs": 1e-4, "proba": 1e-4, "label_gap": 1e-4},
    # TFLite may fuse operations: a little looser.
    "tflite": {"rel": 1e-4, "abs": 1e-4, "proba": 1e-4, "label_gap": 1e-4},
    "numpy": {"rel": 1e-5, "abs": 1e-5, "proba": 1e-5, "label_gap": 1e-5},
    # Training scripts: LogisticRegression(newton-cholesky, tol=1e-8) vs Rust's optimum
    # (export_golden: ≤ 1e-6 vs the optimum; Rust ≤ 1e-5 vs the converged reference).
    "script_logistic": {"rel": 1e-9, "abs": 1e-9, "proba": 1e-5, "label_gap": 1e-5},
}
KNN_TIE_REL = 1e-9
# Snippets: the names their comments say they assume, taken from the training script.
TREE_GLUE = ("tree_model = modelo\nfeature_names = nombres\n"
             "class_names = globals().get('clases')\n")
# texto_arbol/camino_decision take the names as a list by code (0, 1, 2…) or, when the class
# values are other numbers (a CSV with classes 1, 2, 3), as the dict {value: name} of the
# training script (nombre_clase): the student's choice, both paths are exercised.
TREE_GLUE_BY_VALUE = TREE_GLUE + (
    "if class_names is not None and not set(int(c) for c in modelo.classes_) <= "
    "set(range(len(class_names))):\n    class_names = nombre_clase\n")
GLUE: dict[str, str] = {
    "confusion_matrix": "class_names = clases\n",
    "precision_by_class": "class_names = clases\n",
    "pred_vs_real": "",
    "error_distribution": "",
    "decision_boundary": "",
    "tree_viz": TREE_GLUE,
    "tree_text": TREE_GLUE_BY_VALUE,
    "decision_path": TREE_GLUE_BY_VALUE,
}
# Code run after a snippet that only defines a function: the example call of its comment.
# Its output is kept (RESULT) and compared with Rust in check_templates.
AFTER: dict[str, str] = {
    "decision_path": ("import contextlib, io\n_buf = io.StringIO()\n"
                      "with contextlib.redirect_stdout(_buf):\n"
                      "    mostrar_camino_decision(tree_model, X_test.iloc[[0]], feature_names, "
                      "class_names)\nRESULT = _buf.getvalue()\n"),
    "tree_text": "RESULT = tree_text\n",
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
    "paquete_leeme.txt.j2": "text (LEEME.txt), not code",
    "paquete_leeme_arbol.txt.j2": "text (LEEME.txt), not code",
    "paquete_leeme_knn.txt.j2": "text (LEEME.txt), not code",
    "paquete_leeme_red.txt.j2": "text (LEEME.txt), not code",
}
# Template file of each CodeTemplate key of the corpus (code/<key>.py).
TEMPLATE_FILES: dict[str, str] = {
    "regression_code": "regresion_entrenamiento.py.j2",
    "tree_training": "arbol_entrenamiento.py.j2",
    "knn_training": "knn_entrenamiento.py.j2",
    "nn_training": "red_entrenamiento.py.j2",
    "nn_numpy_predictor": "red_predictor_numpy.py.j2",
    "confusion_matrix": "matriz_confusion.py.j2",
    "precision_by_class": "precision_por_clase.py.j2",
    "pred_vs_real": "predicciones_vs_reales.py.j2",
    "error_distribution": "distribucion_errores.py.j2",
    "decision_boundary": "frontera_decision.py.j2",
    "tree_viz": "visualizar_arbol.py.j2",
    "tree_text": "texto_arbol.py.j2",
    "decision_path": "camino_decision.py.j2",
    "tree_usage": "usar_modelo_arbol.py.j2",
    "tree_python": "arbol_python_corregido.py.j2",
}
PACKAGE_TEMPLATES = {
    "regresion": "paquete_modelo_lineal.py.j2",
    "arboles": "paquete_modelo_arbol.py.j2",
    "knn": "paquete_modelo_knn.py.j2",
    "redes": "paquete_modelo_red.py.j2",
}
TRAINING_KEYS = {"regression_code", "tree_training", "knn_training", "nn_training"}

# --- the child process -------------------------------------------------------------------
# Runs a script with `python -I` in its folder: no network (socket blocked), matplotlib Agg,
# then optional probe code and snippets in the namespace the script leaves.
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
m = g["modelo"]
RESULT = {"y_pred": np.asarray(g["y_pred"]).ravel().tolist()}
if hasattr(g.get("X_test"), "index"):
    RESULT["test_positions"] = [int(g["X"].index.get_loc(i)) for i in g["X_test"].index]
if hasattr(m, "classes_"):
    RESULT["classes"] = np.asarray(m.classes_).tolist()
    if hasattr(m, "predict_proba"):
        RESULT["proba"] = m.predict_proba(g["X_test"]).tolist()
for key in ("accuracy", "r2", "mse"):
    if key in g:
        RESULT[key] = float(g[key])
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
if SPEC.get("rules"):
    from sklearn.tree import export_text
    names = data["class_names"]
    class_names = None if names is None else [str(names[int(c)]) for c in data["model_classes"]]
    RESULT["rules"] = export_text(model, feature_names=data["feature_names"],
                                  class_names=class_names, show_weights=False)
'''

PROBE_TREE_USAGE = r'''
import json
import numpy as np
data = json.load(open(SPEC["data"], encoding="utf-8"))
RESULT = [predecir(list(row)) for row in data["inputs"]]
RESULT = [v.item() if hasattr(v, "item") else v for v in RESULT]
'''

PROBE_TREE_FUNCTION = r'''
import json
import re
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
RESULT["n_test"] = int(len(g["y_test"]))
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
    keep: bool
    env: dict[str, str] = field(default_factory=dict)


# --- helpers -----------------------------------------------------------------------------
def load_case(root: Path, name: str) -> dict:
    return json.loads((root / name / "case.json").read_text("utf-8"))


def fresh_dir(ctx: Ctx, prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=ctx.work))


def run_child(ctx: Ctx, cwd: Path, spec: dict, timeout: int | None = None) -> tuple[dict, str]:
    """Runs RUNNER with `python -I` in `cwd`; returns its result and the captured output."""
    spec_dir = Path(tempfile.mkdtemp(prefix="spec-", dir=ctx.tools))
    spec = dict(spec, out=str(spec_dir / "result.json"))
    spec_path = spec_dir / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    try:
        proc = subprocess.run([sys.executable, "-I", str(ctx.tools / "runner.py"), str(spec_path)],
                              cwd=cwd, capture_output=True, text=True, env=ctx.env,
                              timeout=timeout or ctx.timeout)
        output = (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout ({timeout or ctx.timeout} s)"}, ""
    out = spec_dir / "result.json"
    if not out.exists():
        return {"ok": False, "error": f"exit {proc.returncode} without result\n{output[-3000:]}"}, output
    return json.loads(out.read_text("utf-8")), output


def data_file(ctx: Ctx, case: dict) -> Path:
    path = Path(tempfile.mkdtemp(prefix="data-", dir=ctx.tools)) / "data.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    return path


def unpack(ctx: Ctx, archive: Path, prefix: str) -> Path:
    """The ZIP in a new, empty folder (only plain names: no paths, no '..')."""
    target = fresh_dir(ctx, prefix)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if "/" in name or "\\" in name or name.startswith(".") or info.is_dir():
                raise ValueError(f"unexpected entry {name!r} in {archive.name}")
            (target / name).write_bytes(zf.read(info))
    return target


def first_error(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " | ".join(lines[-3:])[:400]


# --- comparisons -------------------------------------------------------------------------
def code_of_label(case: dict, label: Any, classes: list | None) -> int | None:
    """Dataset class code of a predicted label: by position in `classes` (classes_, in the
    order of model_classes) when given, else the label itself (ONNX labels)."""
    codes = case["model_classes"]
    if classes is not None:
        try:
            return codes[classes.index(label)]
        except ValueError:
            return None
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


def top_gap(proba: list[float] | None) -> float:
    if not proba or len(proba) < 2:
        return math.inf
    s = sorted(proba, reverse=True)
    return s[0] - s[1]


def knn_ambiguous(case: dict, x_train: np.ndarray, *, onnx: bool) -> np.ndarray:
    cfg = case["config"]
    k = cfg["n_neighbors"]
    X = np.asarray(case["inputs"], dtype=float)
    manhattan = cfg["metric"] == "manhattan"

    def neighbours(Q: np.ndarray):
        diff = Q[:, None, :] - x_train[None, :, :]
        d = np.abs(diff).sum(axis=2) if manhattan else np.sqrt((diff * diff).sum(axis=2))
        order = np.argsort(d, axis=1, kind="stable")
        return d, order

    d64, o64 = neighbours(X)
    sorted_d = np.take_along_axis(d64, o64, axis=1)
    amb = np.zeros(len(X), dtype=bool)
    if sorted_d.shape[1] > k:
        dk, dk1 = sorted_d[:, k - 1], sorted_d[:, k]
        amb |= (dk1 - dk) <= KNN_TIE_REL * np.maximum(1.0, dk)
    if onnx:
        X32 = X.astype(np.float32).astype(float)
        _, o32 = neighbours(X32)
        same = np.all(np.sort(o64[:, :k], axis=1) == np.sort(o32[:, :k], axis=1), axis=1)
        amb |= ~same
        # Exact copies of a training row: float32 rounding moves them off distance 0, where
        # scikit-learn's rule (only the neighbours at distance 0 count) applies.
        amb |= sorted_d[:, 0] == 0.0
    return amb


def compare(case: dict, kind: str, *, labels: list | None = None, classes: list | None = None,
            proba: list | None = None, values: list | None = None,
            onnx_labels: bool = False, allowed: np.ndarray | None = None,
            names: bool = False) -> tuple[bool, str]:
    """Predictions of an artefact against Rust's (`case["rust"]`, row by row)."""
    tol = TOL[kind]
    rust = case["rust"]
    n = len(rust)
    bad: list[str] = []
    tolerated = 0
    max_err = 0.0
    max_perr = 0.0
    for i in range(n):
        r = rust[i]
        skip = allowed is not None and bool(allowed[i])
        if "value" in r:
            got = float(np.ravel(values[i])[0]) if values is not None else None
            if got is None:
                continue
            err = abs(got - r["value"])
            limit = tol["abs"] + tol["rel"] * abs(r["value"])
            if not err <= limit:
                if skip:
                    tolerated += 1
                else:
                    bad.append(f"row {i}: {got!r} vs {r['value']!r}")
            else:
                max_err = max(max_err, err / max(1.0, abs(r["value"])))
            continue
        # classification
        if labels is not None:
            lab = labels[i]
            if names:
                ok_label = lab == r["name"]
            elif onnx_labels:
                ok_label = int(lab) == onnx_label_of_code(case, r["code"])
            else:
                ok_label = code_of_label(case, lab, classes) == r["code"]
            if not ok_label:
                if skip or top_gap(r.get("proba")) <= tol["label_gap"]:
                    tolerated += 1
                    continue
                bad.append(f"row {i}: label {lab!r} vs {r['name']!r} (code {r['code']})")
                continue
        if proba is not None and r.get("proba") is not None:
            got = np.asarray(proba[i], dtype=float).ravel()
            exp = np.asarray(r["proba"], dtype=float)
            if got.shape != exp.shape:
                bad.append(f"row {i}: {got.size} probabilities vs {exp.size}")
                continue
            err = float(np.max(np.abs(got - exp)))
            if not err <= tol["proba"]:
                if skip:
                    tolerated += 1
                else:
                    bad.append(f"row {i}: probabilities differ by {err:.2g}")
            else:
                max_perr = max(max_perr, err)
    detail = f"{n} rows"
    if max_err:
        detail += f", max rel err {max_err:.1e}"
    if max_perr:
        detail += f", max proba err {max_perr:.1e}"
    if tolerated:
        detail += f", {tolerated} tie rows tolerated"
    if bad:
        return False, f"{len(bad)} rows differ: " + "; ".join(bad[:3])
    return True, detail


def nn_outputs_to_rust_shape(case: dict, out: list) -> tuple[list | None, list | None, list | None]:
    """Network outputs → (labels as codes, probabilities, values) in Rust's layout."""
    arr = np.asarray(out, dtype=float)
    if case["task"] == "regression":
        return None, None, arr.reshape(len(arr), -1)[:, 0].tolist()
    codes = case["model_classes"]
    if arr.ndim == 1 or arr.shape[1] == 1:
        p1 = arr.reshape(-1)
        proba = np.stack([1 - p1, p1], axis=1)
        labels = [codes[int(p > 0.5)] for p in p1]
    else:
        proba = arr
        labels = [codes[int(j)] for j in np.argmax(arr, axis=1)]
    return labels, proba.tolist(), None


# --- checks ------------------------------------------------------------------------------
def check_onnx(ctx: Ctx, name: str, case: dict) -> list[Row]:
    import onnxruntime as ort

    t0 = time.perf_counter()
    path = ctx.root / name / case["files"]["onnx"]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    X = np.asarray(case["inputs"], dtype=np.float32)
    outs = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, {"float_input": X}))}
    kind = {"regresion": "onnx_linear", "arboles": "onnx_tree", "knn": "onnx_knn",
            "redes": "onnx_nn"}[case["lesson"]]
    allowed = None
    if case["lesson"] == "knn":
        model = json.loads(zipfile.ZipFile(ctx.root / name / case["files"]["package"])
                           .read("modelo.json"))
        allowed = knn_ambiguous(case, np.asarray(model["X_train"], dtype=float), onnx=True)
    if case["task"] == "regression":
        ok, detail = compare(case, kind, values=outs["variable"].tolist(), allowed=allowed)
    else:
        ok, detail = compare(case, kind, labels=outs["label"].tolist(),
                             proba=outs["probabilities"].tolist(), onnx_labels=True,
                             allowed=allowed)
    return [Row(name, "onnx (onnxruntime)", ok, detail, time.perf_counter() - t0)]


def check_sklearn_package(ctx: Ctx, name: str, case: dict) -> list[Row]:
    t0 = time.perf_counter()
    rows: list[Row] = []
    pkg = unpack(ctx, ctx.root / name / case["files"]["package"], f"{name}-paquete-")
    script = (pkg / "reconstruir_modelo.py").read_text("utf-8")
    pkl = {"regresion": None, "arboles": "decision_tree_model.pkl", "knn": "modelo_knn.pkl"}[
        case["lesson"]]
    if pkl is None:  # the linear package names the pickle after the model
        pkl = re.search(r"open\([\"']([\w.-]+\.pkl)[\"'], \"wb\"\)", script).group(1)
    data = data_file(ctx, case)
    spec: dict[str, Any] = {"script": "reconstruir_modelo.py", "probe": PROBE_SKLEARN_PACKAGE,
                            "data": str(data), "pkl": pkl,
                            "rules": case["lesson"] == "arboles"}
    if case["lesson"] == "arboles":
        usage = ctx.root / name / case["code"]["tree_usage"]
        spec["snippets"] = [{"name": "tree_usage", "file": str(usage), "after": PROBE_TREE_USAGE}]
    result, output = run_child(ctx, pkg, spec)
    label = f"paquete ({PACKAGE_TEMPLATES[case['lesson']]})"
    if not result["ok"]:
        return [Row(name, label, False, first_error(result.get("error", output)),
                    time.perf_counter() - t0)]
    missing = [] if (pkg / pkl).is_file() else [pkl]
    if missing or "✅" not in output:
        return [Row(name, label, False, f"missing {missing} or no ✅ in the output",
                    time.perf_counter() - t0)]
    got = result["probe"]
    kind = {"regresion": "sk_linear", "arboles": "sk_tree", "knn": "sk_knn"}[case["lesson"]]
    if case["lesson"] == "regresion" and case["task"] == "classification":
        kind = "script_logistic"  # same solver as the training script (newton-cholesky)
    allowed = None
    if case["lesson"] == "knn":
        model = json.loads((pkg / "modelo.json").read_text("utf-8"))
        allowed = knn_ambiguous(case, np.asarray(model["X_train"], dtype=float), onnx=False)
    if case["task"] == "regression":
        ok, detail = compare(case, kind, values=got["y_pred"], allowed=allowed)
    else:
        ok, detail = compare(case, kind, labels=got["y_pred"], classes=got["classes"],
                             proba=got.get("proba"), allowed=allowed)
    rows.append(Row(name, label + f" → {pkl}", ok, detail, time.perf_counter() - t0))
    if case["lesson"] == "arboles":
        rules = (ctx.root / name / case["files"]["rules"]).read_text("utf-8")
        same = rules.strip() == got["rules"].strip()
        rows.append(Row(name, "reglas .txt = export_text", same,
                        f"{len(rules.splitlines())} lines" if same else "text differs"))
        usage = result["snippets"].get("tree_usage", {})
        if not usage.get("ok"):
            rows.append(Row(name, "plantilla usar_modelo_arbol.py.j2", False,
                            first_error(usage.get("error", ""))))
        else:
            vals = usage["probe"]
            if case["task"] == "regression":
                ok, detail = compare(case, "sk_tree", values=vals)
            else:
                ok, detail = compare(case, "sk_tree", labels=vals, names=True)
            rows.append(Row(name, "plantilla usar_modelo_arbol.py.j2", ok, detail))
    return rows


def check_keras_package(ctx: Ctx, name: str, case: dict) -> list[Row]:
    t0 = time.perf_counter()
    pkg = unpack(ctx, ctx.root / name / case["files"]["package"], f"{name}-keras-")
    data = data_file(ctx, case)
    result, output = run_child(ctx, pkg, {"script": "reconstruir_modelo.py",
                                          "probe": PROBE_KERAS, "data": str(data)})
    label = "paquete Keras (paquete_modelo_red.py.j2)"
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
        labels, proba, values = nn_outputs_to_rust_shape(case, outs)
        kind = "tflite" if fmt == "tflite" else "keras"
        ok, detail = compare(case, kind, labels=labels, proba=proba, values=values,
                             classes=case["model_classes"])
        rows.append(Row(name, f"  {fmt}", ok, detail))
    return rows


def check_numpy_predictor(ctx: Ctx, name: str, case: dict) -> list[Row]:
    t0 = time.perf_counter()
    pkg = unpack(ctx, ctx.root / name / case["files"]["numpy_predictor"], f"{name}-numpy-")
    text = (pkg / "predecir_numpy.py").read_text("utf-8")
    shown = (ctx.root / name / case["code"]["nn_numpy_predictor"]).read_text("utf-8")
    data = data_file(ctx, case)
    result, output = run_child(ctx, pkg, {"script": "predecir_numpy.py", "probe": PROBE_NUMPY,
                                          "data": str(data)})
    label = "predictor NumPy (red_predictor_numpy.py.j2)"
    if not result["ok"]:
        return [Row(name, label, False, first_error(result.get("error", output)))]
    labels, proba, values = nn_outputs_to_rust_shape(case, result["probe"])
    ok, detail = compare(case, "numpy", labels=labels, proba=proba, values=values,
                         classes=case["model_classes"])
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
    n_out = 1 if case["task"] == "regression" or arch["output_activation"] == "sigmoid" \
        else len(case["model_classes"])
    expected_arch = [len(case["feature_names"]), *arch["hidden"], n_out]
    if mi.get("architecture") != expected_arch:
        problems.append(f"architecture {mi.get('architecture')} vs {expected_arch}")
    for key, value in (("target_column", case["target_column"]),
                       ("feature_names", case["feature_names"]),
                       ("feature_columns", case["feature_columns"]),
                       ("dataset", case["dataset_label"])):
        if di.get(key) != value:
            problems.append(f"data_info.{key} = {di.get(key)!r}")
    return [Row(name, "metadatos .json", not problems, "; ".join(problems) or
                "valid JSON, architecture, classes and columns of the case")]


def stage_script_dir(ctx: Ctx, name: str, case: dict) -> Path:
    folder = fresh_dir(ctx, f"{name}-plantillas-")
    if case["csv_file"]:
        shutil.copy(ctx.root / name / case["csv_file"], folder / case["csv_file"])
    return folder


def check_templates(ctx: Ctx, name: str, case: dict) -> list[Row]:
    t0 = time.perf_counter()
    code = case["code"]
    training = next(k for k in code if k in TRAINING_KEYS)
    folder = stage_script_dir(ctx, name, case)
    data = data_file(ctx, case)
    snippets = []
    for key, rel in code.items():
        if key in TRAINING_KEYS or key in ("tree_usage", "nn_numpy_predictor"):
            continue
        entry = {"name": key, "file": str(ctx.root / name / rel), "glue": GLUE.get(key, ""),
                 "after": AFTER.get(key, "")}
        if key == "tree_python":
            entry["after"] = PROBE_TREE_FUNCTION
        snippets.append(entry)
    spec = {"script": str(ctx.root / name / code[training]),
            "probe": PROBE_NN_TRAINING if case["lesson"] == "redes" else PROBE_TRAINING,
            "data": str(data), "snippets": snippets,
            "tree_python": str(ctx.root / name / code.get("tree_python", ""))}
    result, output = run_child(ctx, folder, spec)
    label = f"plantilla {TEMPLATE_FILES[training]}"
    if not result["ok"]:
        return [Row(name, label, False, first_error(result.get("error", output)),
                    time.perf_counter() - t0)]
    got = result["probe"]
    rows = []
    if case["lesson"] == "redes":
        metric = {k: round(v, 4) for k, v in got.items() if k != "n_test"}
        rows.append(Row(name, label + " (Keras)", True, f"runs; {metric}",
                        time.perf_counter() - t0))
    else:
        positions = got.get("test_positions")
        n = len(case["input_rows"])
        if positions is not None and positions[:n] != case["input_rows"]:
            rows.append(Row(name, label, False, "the script's split differs from the app's"))
        else:
            kind = {"regresion": "script_logistic", "arboles": "sk_tree", "knn": "sk_knn"}[
                case["lesson"]]
            if case["lesson"] == "regresion" and case["task"] == "regression":
                kind = "sk_linear"
            allowed = None
            if case["lesson"] == "knn":
                model = json.loads(zipfile.ZipFile(ctx.root / name / case["files"]["package"])
                                   .read("modelo.json"))
                allowed = knn_ambiguous(case, np.asarray(model["X_train"], dtype=float),
                                        onnx=False)
            sub = dict(case, rust=case["rust"][:len(got["y_pred"])])
            if case["task"] == "regression":
                ok, detail = compare(sub, kind, values=got["y_pred"], allowed=allowed)
            else:
                ok, detail = compare(sub, kind, labels=got["y_pred"], classes=got["classes"],
                                     proba=got.get("proba"), allowed=allowed)
            rows.append(Row(name, label + " = Rust", ok, "split = app's; " + detail,
                            time.perf_counter() - t0))
    for key, entry in result["snippets"].items():
        tlabel = f"  {TEMPLATE_FILES[key]}"
        if not entry["ok"]:
            rows.append(Row(name, tlabel, False, first_error(entry["error"])))
            continue
        if key == "tree_python":
            vals = entry["probe"]
            if case["task"] == "regression":
                ok, detail = compare(case, "sk_tree", values=vals)
            else:
                ok, detail = compare(case, "sk_tree", labels=vals, names=True)
            exported = (ctx.root / name / case["files"]["tree_python"]).read_text("utf-8")
            shown = (ctx.root / name / code["tree_python"]).read_text("utf-8")
            if exported != shown:
                ok, detail = False, "arbol_decision.py differs from the code shown"
            rows.append(Row(name, tlabel + " (arbol_decision.py) = Rust", ok, detail))
        elif key == "tree_text":
            # With show_weights=True; without the weights it must be the exported rules .txt.
            text = re.sub(r"weights: \[[^\]]*\] ", "", entry["probe"]).strip()
            rules = (ctx.root / name / case["files"]["rules"]).read_text("utf-8").strip()
            same = text == rules
            rows.append(Row(name, tlabel + " = reglas .txt", same,
                            "same rules and leaves (without the weights)" if same
                            else "the text differs from the exported rules"))
        elif key == "decision_path":
            last = entry["probe"].strip().splitlines()[-1]
            r = case["rust"][0]
            want = (f"Predicción final: {r['name']}" if "name" in r
                    else f"Predicción final: {r['value']:.4f}")
            rows.append(Row(name, tlabel, last == want, f"first test row: {last!r}"
                            + ("" if last == want else f", Rust: {want!r}")))
        else:
            rows.append(Row(name, tlabel, True, "runs"))
    return rows


def check_static_templates(ctx: Ctx) -> list[Row]:
    """Templates without a model: 📊 Datos' exploration snippet (the UI shows the file as
    is), run after loading Iris into `df`, as its first comment asks."""
    import jinja2

    t0 = time.perf_counter()
    text = jinja2.Environment(keep_trailing_newline=True, undefined=jinja2.StrictUndefined
                              ).from_string((CODE_DIR / "analisis_exploratorio.py.j2")
                                            .read_text("utf-8")).render()
    # The two lines the snippet asks the student to replace ("reemplaza esto"): the
    # DataFrame and the features of the pairplot.
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
    detail = "runs (df = Iris, selected_features = 3 columns)" if ok else first_error(entry.get("error") or
                                                      result.get("error", output))
    return [Row("datos", "plantilla analisis_exploratorio.py.j2", ok, detail,
                time.perf_counter() - t0)]


def coverage(ctx: Ctx, cases: dict[str, dict]) -> list[Row]:
    """Every template of content/code is either run here or listed in NOT_RUN."""
    run = {"analisis_exploratorio.py.j2"} | set(PACKAGE_TEMPLATES.values())
    for case in cases.values():
        run |= {TEMPLATE_FILES[k] for k in case["code"]}
    rows = []
    for path in sorted(CODE_DIR.glob("*.j2")):
        if path.name in run:
            continue
        if path.name in NOT_RUN:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, help="folder written by export_corpus")
    parser.add_argument("--only", action="append", default=[], help="case name substring")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=int, default=900, help="seconds per script")
    parser.add_argument("--keep", action="store_true", help="keep the temporary folders")
    args = parser.parse_args(argv)

    root = args.corpus.resolve()
    index = json.loads((root / "index.json").read_text("utf-8"))
    names = [n for n in index["cases"] if not args.only or any(o in n for o in args.only)]
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
    ctx = Ctx(root=root, tools=tools, work=work, timeout=args.timeout, keep=args.keep, env=env)

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
        rows += coverage(ctx, cases)

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
