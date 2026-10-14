"""Phase 3 golden fixtures: the 🌲 Árboles de Decisión lesson (fase3-arboles.md §6).

Registered by ``tools/export_golden.py`` (``build_cases``), which passes itself as
``eg`` so the shared helpers (Writer, make_meta, datasets, CSV cases, classification
and regression numbers, ONNX and code rendering) are reused, not copied.

Each ``trees/<case>.json`` renders the real lesson headlessly (``golden_render``),
trains like the student (criterion, depth, min samples split, test size 0.3,
``random_state=42``), visits every tab and sub-view and stores:

* the full tree (``tree_`` arrays, bit-exact) and the per-node split ``trace`` of
  ``tools/cart_reference.py`` (draw order of features, ties), checked against sklearn;
* predictions, probabilities, leaves, decision paths, metrics and curves;
* native (Gini/entropy/MSE) and permutation importances (``n_repeats=10``,
  ``random_state=42``) with the RNG values needed to reproduce them exactly;
* ``export_text`` (the 📝 Texto view and the rules .txt), ``plot_tree`` layout and
  colours, ``export_tree_as_python_code`` (literal and corrected, D44), the 2-feature
  trees of the boundary/surface, ``onnx_reference`` (skl2onnx TreeEnsemble*) and the
  "paquete Python" (D1) executed for real.
"""

from __future__ import annotations

import json
import keyword
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODULE = "trees"
NAV = "🌲 Árboles de Decisión"
TREE_CASES: dict[str, dict[str, Any]] = {
    # Classification (the default tree type)
    "iris_gini_d3": {"dataset": "iris", "criterion": "gini", "max_depth": 3, "mss": 2,
                     "pair": (2, 3), "same_pair": True, "full": True},
    "iris_entropy_d5_mss5": {"dataset": "iris", "criterion": "entropy", "max_depth": 5, "mss": 5},
    "iris_gini_d1": {"dataset": "iris", "criterion": "gini", "max_depth": 1, "mss": 2},
    "wine_gini_d4": {"dataset": "wine", "criterion": "gini", "max_depth": 4, "mss": 2,
                     "pair": (0, 9)},
    "wine_entropy_d3_mss10": {"dataset": "wine", "criterion": "entropy", "max_depth": 3,
                              "mss": 10},
    "breast_cancer_gini_d3": {"dataset": "breast_cancer", "criterion": "gini", "max_depth": 3,
                              "mss": 2, "pair": (20, 27)},
    "breast_cancer_entropy_d10": {"dataset": "breast_cancer", "criterion": "entropy",
                                  "max_depth": 10, "mss": 2},
    "titanic_gini_d4": {"dataset": "titanic", "criterion": "gini", "max_depth": 4, "mss": 2,
                        "pair": (1, 4)},
    "csv_string_target_gini_d3": {"csv": "string_target", "criterion": "gini", "max_depth": 3,
                                  "mss": 2},
    # Regression (button 📈 Regresión)
    "diabetes_squared_error_d3": {"dataset": "diabetes", "tree": "Regresión",
                                  "criterion": "squared_error", "max_depth": 3, "mss": 2,
                                  "pair": (2, 8), "full": True},
    "diabetes_friedman_mse_d5_mss10": {"dataset": "diabetes", "tree": "Regresión",
                                       "criterion": "friedman_mse", "max_depth": 5, "mss": 10},
    "diabetes_absolute_error_d4": {"dataset": "diabetes", "tree": "Regresión",
                                   "criterion": "absolute_error", "max_depth": 4, "mss": 2},
    "diabetes_squared_error_d10": {"dataset": "diabetes", "tree": "Regresión",
                                   "criterion": "squared_error", "max_depth": 10, "mss": 2,
                                   "views": False},
    "california_housing_squared_error_d3": {"dataset": "california_housing", "tree": "Regresión",
                                            "criterion": "squared_error", "max_depth": 3,
                                            "mss": 2, "pair": (0, 6)},
    "california_housing_squared_error_d8_mss10": {"dataset": "california_housing",
                                                  "tree": "Regresión",
                                                  "criterion": "squared_error", "max_depth": 8,
                                                  "mss": 10, "views": False},
    "csv_numeric_regression_squared_error_d3": {"csv": "numeric_regression", "tree": "Regresión",
                                                "criterion": "squared_error", "max_depth": 3,
                                                "mss": 2},
    # Tree type that does not match the dataset (what the app does)
    "diabetes_classification_default": {"dataset": "diabetes", "criterion": "gini",
                                        "max_depth": 3, "mss": 2, "mismatch": True},
    "california_housing_classification_default": {"dataset": "california_housing",
                                                  "criterion": "gini", "max_depth": 3, "mss": 2,
                                                  "mismatch": True},
    "iris_regression_mismatch": {"dataset": "iris", "tree": "Regresión",
                                 "criterion": "squared_error", "max_depth": 3, "mss": 2,
                                 "mismatch": True},
}
PERM_TRACE_MAX_TEST = 60  # store the shuffled indices only for small test sets
ONNX_HEAD = 200  # ONNX outputs stored for the first rows of X_test


def dumps_compact(doc: dict) -> str:
    """Like export_golden.dumps (indent 2, UTF-8), but lists of scalars on one line."""

    def enc(v, ind):
        pad, pad2 = "  " * ind, "  " * (ind + 1)
        if isinstance(v, dict):
            if not v:
                return "{}"
            items = [f"{pad2}{json.dumps(k, ensure_ascii=False)}: {enc(x, ind + 1)}"
                     for k, x in v.items()]
            return "{\n" + ",\n".join(items) + "\n" + pad + "}"
        if isinstance(v, list):
            if not v:
                return "[]"
            if all(not isinstance(x, (dict, list)) for x in v):
                return json.dumps(v, ensure_ascii=False, allow_nan=False)
            return "[\n" + ",\n".join(pad2 + enc(x, ind + 1) for x in v) + "\n" + pad + "]"
        return json.dumps(v, ensure_ascii=False, allow_nan=False)

    return enc(doc, 0) + "\n"


def _page():
    from mltutor.apps.decission_tree_app import run_decision_trees_app

    return run_decision_trees_app


# ---------------------------------------------------------------------------
# Tree data
# ---------------------------------------------------------------------------
def tree_arrays(model) -> dict:
    t = model.tree_
    classifier = hasattr(model, "classes_")
    value = np.asarray(t.value, dtype=float)[:, 0, :]
    out = {
        "node_count": int(t.node_count),
        "max_depth": int(t.max_depth),
        "n_leaves": int(t.n_leaves),
        "n_features": int(t.n_features),
        "n_classes": int(t.n_classes[0]),
        "children_left": t.children_left,
        "children_right": t.children_right,
        "feature": t.feature,
        "threshold": t.threshold,
        "threshold_float32_inputs": "midpoint x[p-1]/2 + x[p]/2 in float64 of two consecutive "
                                    "DISTINCT float32 training values (X is cast to float32); "
                                    "leaves store -2",
        "impurity": t.impurity,
        "n_node_samples": t.n_node_samples,
        "weighted_n_node_samples": t.weighted_n_node_samples,
        "missing_go_to_left": np.asarray(t.missing_go_to_left, dtype=int),
        "value": value,
        "value_meaning": ("class fractions (sklearn >= 1.4); counts = value * "
                          "weighted_n_node_samples" if classifier else
                          "mean of y (median for absolute_error)"),
    }
    if classifier:
        out["value_counts"] = np.rint(value * t.weighted_n_node_samples[:, None]).astype(int)
    return out


def reference_check(X_train, y_train, model, classification: bool) -> dict:
    """tools/cart_reference.py against the fitted sklearn tree, plus its split trace."""
    import cart_reference as cr

    p = model.get_params()
    ref = cr.fit(X_train, y_train, classification=classification, criterion=p["criterion"],
                 max_depth=p["max_depth"], min_samples_split=p["min_samples_split"],
                 random_state=p["random_state"])
    cmp = cr.compare_with_sklearn(ref, model.tree_)
    exact = {k: v for k, v in cmp.items() if v is True}
    trace = []
    for n in ref["trace"]:
        n = dict(n)
        n["visited"] = [[v["feature"], "constant"] if v["constant"] else
                        [v["feature"], v["best_proxy"], v["best_threshold"], v["best_n_left"]]
                        for v in n["visited"]]
        trace.append(n)
    return {
        "implementation": "tools/cart_reference.py (transcription of sklearn 1.7.2 "
                          "_tree/_splitter/_partitioner/_criterion.pyx)",
        "splitter_seed": ref["splitter_seed"],
        "splitter_seed_rule": "RandomState(random_state).randint(0, 2147483647); then "
                              "our_rand_r xorshift, rand_int(low, high) = low + r % (high - low)",
        "comparison": cmp,
        "arrays_bit_exact": sorted(exact),
        "nodes_with_exact_ties": sum(1 for n in trace if n["exact_ties_lost"]),
        "nodes_with_near_ties": sum(1 for n in trace if n["near_ties_lost_rel_1e-9"]),
        "trace_format": "one entry per split node, in build order: visit_order = features in "
                        "the order the RNG drew them; visited = [feature, best proxy, its "
                        "threshold, n_left] or [feature, 'constant'] (max - min <= 1e-7 in the "
                        "node); exact_ties_lost = features with the same best proxy that lost "
                        "because they were drawn later",
        "trace": trace,
    }


def decision_path_rows(model, X: np.ndarray) -> list[dict]:
    X32 = np.asarray(X, dtype=np.float32)
    ind = model.decision_path(X32)
    leaves = model.apply(X32)
    t = model.tree_
    rows = []
    for r in range(X32.shape[0]):
        nodes = ind.indices[ind.indptr[r]:ind.indptr[r + 1]].tolist()
        steps = []
        for nid in nodes:
            if nid == leaves[r]:
                continue
            f = int(t.feature[nid])
            steps.append({"node": nid, "feature": f, "value_float32": float(X32[r, f]),
                          "threshold": float(t.threshold[nid]),
                          "goes_left": bool(X32[r, f] <= t.threshold[nid])})
        rows.append({"nodes": nodes, "leaf": int(leaves[r]), "steps": steps})
    return rows


def plot_tree_layout(model, feature_names, class_names, figsize=(14, 10)) -> dict:
    """What plot_tree draws in the 🌲 Árbol view (default 14x10 in): text, colours, layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    from sklearn.tree._export import _color_brew

    fig, ax = plt.subplots(figsize=figsize)
    anns = plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True,
                     rounded=True, ax=ax, proportion=True, impurity=True)
    fig.canvas.draw()
    boxes, labels = [], []
    for a in anns:
        bp = a.get_bbox_patch()
        if bp is None:
            labels.append({"text": a.get_text(), "xy_axes": list(a.xyann), "ha": a.get_ha()})
            continue
        item = {"text": a.get_text(), "xy_axes": list(a.xyann),
                "facecolor": __import__("matplotlib").colors.to_hex(bp.get_facecolor()),
                "boxstyle": type(bp.get_boxstyle()).__name__}
        if a.arrow_patch is not None:
            item["arrow_from_axes"] = list(a.xy)
        boxes.append(item)
    out = {
        "figsize_inches": list(figsize),
        "fontsize_auto": float(anns[0].get_fontsize()) if anns else None,
        "n_boxes": len(boxes),
        "boxes_in_drawing_order": boxes,
        "edge_labels": labels,
        "palette_color_brew": _color_brew(int(model.tree_.n_classes[0])),
        "rules": {
            "order": "pre-order (node, left subtree, right subtree); box i = node i of tree_",
            "position": "Buchheim/Reingold-Tilford (sklearn/tree/_reingold_tilford.py): "
                        "xy = ((x + 0.5) / max_x, (max_y - depth - 0.5) / max_y), "
                        "max_x, max_y = max_extents() + 1",
            "text": "'{feature} <= {round(threshold, 3)}' (internal nodes), '{criterion} = "
                    "{round(impurity, 3)}', 'samples = {round(100*n/n_root, 1)}%', 'value = "
                    "[np.around(value, 3)...]' (regression without brackets), 'class = "
                    "{class_names[argmax]}' (classification only); numbers are Python str() "
                    "of round()",
            "colour": "classification: _color_brew(n_classes)[argmax(value)] blended with white "
                      "by alpha = (v1 - v2) / (1 - v2) (two largest fractions; 0 if they tie); "
                      "regression: _color_brew(1)[0] with alpha = (value - min) / (max - min) "
                      "over the values of ALL nodes; int(round(alpha*c + (1-alpha)*255))",
            "edges": "arrow from the parent's centre to the child's; 'True  ' (left, ha=right) "
                     "and '  False' (right, ha=left) only on the two edges of the root, at the "
                     "midpoint",
            "fontsize": "auto: shrunk so that the widest/tallest box fits scale_x/scale_y "
                        "(unreadable for deep trees; the SVG uses zoom instead, D41)",
        },
    }
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Permutation importance (C10)
# ---------------------------------------------------------------------------
def permutation_reference(model, X_test: pd.DataFrame, y_test) -> dict:
    from mltutor.viz.features import compute_permutation_importance

    res = compute_permutation_importance(model, X_test, y_test)
    seed = int(np.random.RandomState(42).randint(np.iinfo(np.int32).max + 1))
    # Re-implementation of _calculate_permutation_scores (sklearn 1.7.2) to prove the rule.
    X = X_test.to_numpy(dtype=float)
    y = np.asarray(y_test)
    baseline = model.score(X_test, y_test)
    imps = np.zeros((X.shape[1], 10))
    first_col_idx = []
    for c in range(X.shape[1]):
        rs = np.random.RandomState(seed)
        Xp = X.copy()
        idx = np.arange(X.shape[0])
        for r in range(10):
            rs.shuffle(idx)
            Xp[:, c] = Xp[idx, c]
            imps[c, r] = baseline - model.score(pd.DataFrame(Xp, columns=X_test.columns), y)
            if c == 0 and X.shape[0] <= PERM_TRACE_MAX_TEST:
                first_col_idx.append(idx.copy())
    exact = bool(np.array_equal(imps, res.importances))
    out = {
        "n_repeats": 10, "random_state": 42, "scorer": "estimator.score: accuracy "
        "(classifier) or R² (regressor), on X_test",
        "seed_rule": "check_random_state(42).randint(np.iinfo(np.int32).max + 1): ONE seed "
                     "shared by every column (sklearn >= 1.5; older versions drew one per "
                     "column). Each column: RandomState(seed), idx = arange(n_test), then "
                     "n_repeats times: shuffle(idx) IN PLACE (cumulative) and "
                     "X_perm[:, col] = X_perm[idx, col] (the already permuted column)",
        "seed": seed,
        "baseline_score": baseline,
        "importances": res.importances,
        "importances_mean": res.importances_mean,
        "importances_std": res.importances_std,
        "std_rule": "np.std (ddof=0)",
        "reimplementation_bit_exact": exact,
    }
    if first_col_idx:
        out["column0_shuffling_idx_after_each_repeat"] = first_col_idx
    return out


# ---------------------------------------------------------------------------
# Python code of the tree (export_tree_as_python_code) and the corrected one (D44)
# ---------------------------------------------------------------------------
def py_identifier(name: str, used: set[str]) -> str:
    """D44: a valid, unique Python identifier for a feature name."""
    s = re.sub(r"\W+", "_", str(name), flags=re.UNICODE).strip("_")
    if not s:
        s = "x"
    if s[0].isdigit():
        s = "x_" + s
    if keyword.iskeyword(s) or s in ("True", "False", "None"):
        s += "_"
    base, k = s, 2
    while s in used:
        s = f"{base}_{k}"
        k += 1
    used.add(s)
    return s


def tree_python_body(model, names: list[str], class_labels: list[str] | None,
                     threshold_fmt, base_depth: int = 0, value_fmt=None) -> str:
    t = model.tree_

    def rec(node, depth):
        ind = "    " * depth
        if t.children_left[node] == t.children_right[node]:
            v = t.value[node][0]
            if class_labels is not None:
                k = int(v.argmax())
                return f'{ind}return {class_labels[k]}  # Probabilidad: {v[k] / v.sum():.3f}'
            return f"{ind}return {value_fmt(v[0]) if value_fmt else format(v[0], '.6f')}"
        f = int(t.feature[node])
        code = f"{ind}if {names[f]} <= {threshold_fmt(t.threshold[node])}:\n"
        code += rec(t.children_left[node], depth + 1) + "\n"
        code += f"{ind}else:\n"
        code += rec(t.children_right[node], depth + 1)
        return code

    return rec(0, base_depth)


def python_code(eg, model, feature_names: list[str], class_names: list | None,
                X_check: np.ndarray) -> dict:
    from mltutor.algorithms.export import export_tree_as_python_code

    classifier = hasattr(model, "classes_")
    app_text = export_tree_as_python_code(model, list(feature_names))
    try:
        compile(app_text, "arbol_decision.py", "exec")
        app_ok, app_err = True, None
    except SyntaxError as exc:
        app_ok, app_err = False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
    # The literal template: same text from content/code/arbol_python.py.j2 + the body.
    literal_ctx = {
        "function_name": "predecir_clase" if classifier else "predecir_valor",
        "params": list(feature_names),
        "classifier": classifier,
        "n_features": len(feature_names),
        "max_depth": int(model.tree_.max_depth),
        "node_count": int(model.tree_.node_count),
        "body": tree_python_body(model, list(feature_names),
                                 [f'"{c}"' for c in model.classes_] if classifier else None,
                                 lambda v: f"{v:.6f}"),
    }
    literal = eg._render_code_template("arbol_python.py.j2", literal_ctx)
    if literal != app_text:
        raise AssertionError("arbol_python.py.j2 does not reproduce export_tree_as_python_code")
    if model.tree_.node_count > 31:  # the body is already inside app_text
        literal_ctx["body"] = "<omitted: app_text between the docstring and '# Ejemplo de uso'>"
    # D44: valid identifiers, class names, round-trip thresholds.
    used: set[str] = set()
    idents = [py_identifier(n, used) for n in feature_names]
    labels = None
    if classifier:
        names = [str(class_names[int(c)]) if class_names is not None and int(c) < len(class_names)
                 else str(c) for c in model.classes_]
        labels = [repr(n) for n in names]
    corrected_ctx = {
        "function_name": "predecir_clase" if classifier else "predecir_valor",
        "params": idents,
        "display_names": list(feature_names),
        "classifier": classifier,
        "n_features": len(feature_names),
        "max_depth": int(model.tree_.max_depth),
        "node_count": int(model.tree_.node_count),
        "body": tree_python_body(model, idents, labels,
                                 lambda v: repr(f64_switch_threshold(float(v))), base_depth=1,
                                 value_fmt=lambda v: repr(float(v))),
    }
    corrected = eg._render_code_template("arbol_python_corregido.py.j2", corrected_ctx)
    ns: dict[str, Any] = {}
    exec(compile(corrected, "arbol_decision.py", "exec"), ns)
    fn = ns[corrected_ctx["function_name"]]
    got = [fn(*map(float, row)) for row in X_check]
    want = model.predict(np.asarray(X_check, dtype=np.float32))
    if classifier:
        want_l = [names[list(model.classes_).index(w)] for w in want]
        same = got == want_l
    else:
        same = bool(np.array_equal(np.asarray(got, dtype=float), want))
    return {
        "app_text": app_text,
        "app_compiles": app_ok,
        "app_compile_error": app_err,
        "literal_template": "content/code/arbol_python.py.j2",
        "literal_context": literal_ctx,
        "corrected_template": "content/code/arbol_python_corregido.py.j2 (D44)",
        "corrected_context": corrected_ctx,
        "corrected_text": corrected,
        "corrected_identifier_rule": "re.sub(r'\\W+', '_', name).strip('_'); 'x' if empty; "
                                     "'x_' prefix if it starts with a digit; '_' suffix for "
                                     "keywords; '_2', '_3'... if repeated",
        "corrected_threshold_rule": "repr(w), w = largest float64 with x <= w <=> float32(x) "
                                    "<= threshold (f64_switch_threshold): the function takes "
                                    "Python floats and needs no numpy, yet routes every value "
                                    "like scikit-learn, which casts X to float32",
        "corrected_check": {"rows": "X_train and X_test (all), as Python floats (f64)",
                            "same_predictions_as_sklearn": same,
                            "regression": "exact (leaf values written with repr)"},
    }


# ---------------------------------------------------------------------------
# ONNX (skl2onnx reference) and the Python package (D1)
# ---------------------------------------------------------------------------
def onnx_reference(eg, model, X_test: pd.DataFrame, X_all: np.ndarray) -> dict:
    from onnx.reference import ReferenceEvaluator
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    classifier = hasattr(model, "classes_")
    n = X_test.shape[1]
    options = {id(model): {"zipmap": False}} if classifier else None
    onx = convert_sklearn(model, initial_types=[("float_input", FloatTensorType([None, n]))],
                          options=options, target_opset={"": 15, "ai.onnx.ml": 3})
    nodes = []
    for node in onx.graph.node:
        attrs = {}
        for a in node.attribute:
            if a.type == a.FLOATS:
                attrs[a.name] = f"<{len(a.floats)} floats>"
            elif a.type == a.INTS:
                attrs[a.name] = list(a.ints) if len(a.ints) <= 64 else f"<{len(a.ints)} ints>"
            elif a.type == a.STRINGS:
                vals = [s.decode() for s in a.strings]
                attrs[a.name] = sorted(set(vals)) if len(vals) > 8 else vals
            elif a.type == a.STRING:
                attrs[a.name] = a.s.decode()
            elif a.type == a.INT:
                attrs[a.name] = a.i
            elif a.type == a.FLOAT:
                attrs[a.name] = a.f
            else:
                attrs[a.name] = f"<type {a.type}>"
        nodes.append({"op_type": node.op_type, "domain": node.domain,
                      "inputs": list(node.input), "outputs": list(node.output),
                      "attributes": attrs})
    sess = ReferenceEvaluator(onx)
    names = [o.name for o in onx.graph.output]

    def run(X):
        return dict(zip(names, sess.run(None, {"float_input": np.asarray(X, dtype=np.float32)})))

    head = min(len(X_test), ONNX_HEAD)
    X_head = X_test.iloc[:head]
    out_test = run(X_head.to_numpy())
    out_all = run(X_all)
    result = {"producer": "skl2onnx (reference only; the Python app has no ONNX for trees)",
              "opset_import": {o.domain or "ai.onnx": o.version for o in onx.opset_import},
              "inputs": [i.name for i in onx.graph.input], "outputs": names, "nodes": nodes,
              "outputs_rows": f"first {head} rows of X_test (label/probabilities/variable); "
                              "the mismatch counts use every train and test row",
              "threshold_note": "skl2onnx writes float32(threshold) (round to nearest) and "
                                "BRANCH_LEQ without sklearn_threshold(); a sample whose float32 "
                                "value lies between the f64 threshold and its f32 rounding is "
                                "misrouted. Rust should write the largest float32 <= threshold "
                                "(D47)."}
    sk_all = model.predict(np.asarray(X_all, dtype=np.float32))
    if classifier:
        result["label"] = np.asarray(out_test["label"]).astype(int)
        result["probabilities"] = np.asarray(out_test["probabilities"], dtype=float)
        result["max_abs_diff_vs_sklearn_proba"] = float(np.max(np.abs(
            result["probabilities"] - model.predict_proba(X_head))))
        result["label_mismatches_train_and_test"] = int(np.sum(
            np.asarray(out_all["label"]).astype(sk_all.dtype) != sk_all))
    else:
        var = np.asarray(out_test["variable"], dtype=float).ravel()
        result["variable"] = var
        result["max_abs_diff_vs_sklearn"] = float(np.max(np.abs(var - model.predict(X_head))))
        allv = np.asarray(out_all["variable"], dtype=float).ravel()
        result["mismatches_train_and_test_rel_1e-6"] = int(np.sum(
            ~np.isclose(allv, sk_all, rtol=1e-6, atol=1e-9)))
    # Same check with the D47 thresholds (largest float32 <= threshold).
    t = model.tree_
    thr = t.threshold[t.feature >= 0]
    down = np.array([_f32_floor(v) for v in thr], dtype=np.float32)
    rounded = thr.astype(np.float32)
    result["thresholds_where_round_to_nearest_differs_from_floor"] = int(np.sum(down != rounded))
    return result


def f64_switch_threshold(t: float) -> float:
    """D44: the largest float64 w such that ``x <= w`` <=> ``float32(x) <= t`` for any float64 x.

    scikit-learn compares float32(x) with the float64 threshold. f = largest float32 <= t;
    every float64 below the rounding boundary between f and the next float32 rounds to a
    value <= t. At the boundary itself (a tie) round-half-even decides.
    """
    f = _f32_floor(t)
    nxt = np.nextafter(f, np.float32(np.inf), dtype=np.float32)
    m = (float(f) + float(nxt)) / 2.0  # exact in float64
    if np.float32(m) == f:
        return m
    return float(np.nextafter(m, -np.inf))


def _f32_floor(v: float) -> np.float32:
    f = np.float32(v)
    if float(f) > v:
        f = np.nextafter(f, np.float32(-np.inf), dtype=np.float32)
    return f


def export_package(eg, model, feature_columns, feature_names, class_names, X_test,
                   dataset_label, n_train, n_test) -> dict:
    import subprocess
    import sys
    import tempfile

    classifier = hasattr(model, "classes_")
    t = model.tree_
    X_check = X_test.head(5)
    doc = {
        "format": "mltutor-model",
        "format_version": 1,
        "estimator": "sklearn.tree.DecisionTreeClassifier" if classifier
        else "sklearn.tree.DecisionTreeRegressor",
        "params": model.get_params(),
        "feature_names_in": list(feature_columns),
        "display_names": list(feature_names),
        "n_features_in": int(X_test.shape[1]),
        "n_outputs": 1,
        "max_features_": int(model.max_features_),
        "tree": {
            "max_depth": int(t.max_depth), "node_count": int(t.node_count),
            "n_leaves": int(t.n_leaves),
            "children_left": t.children_left, "children_right": t.children_right,
            "feature": t.feature, "threshold": t.threshold, "impurity": t.impurity,
            "n_node_samples": t.n_node_samples,
            "weighted_n_node_samples": t.weighted_n_node_samples,
            "missing_go_to_left": np.asarray(t.missing_go_to_left, dtype=int),
            "value": np.asarray(t.value, dtype=float)[:, 0, :],
        },
        "training": {"dataset": dataset_label, "test_size": 0.3, "random_state": 42,
                     "n_train": n_train, "n_test": n_test},
        "check": {"X": X_check.to_numpy(dtype=float), "prediction": model.predict(X_check)},
    }
    if classifier:
        doc.update({"classes_": model.classes_, "n_classes_": int(model.n_classes_),
                    "class_names": None if class_names is None else list(class_names)})
        doc["check"]["probabilities"] = model.predict_proba(X_check)
    doc = eg.to_jsonable(doc)
    ctx = {"estimator": doc["estimator"], "dataset": dataset_label, "classifier": classifier,
           "json_name": "modelo.json", "pkl_name": "decision_tree_model.pkl",
           "task_type": "Clasificación" if classifier else "Regresión"}
    script = eg._render_code_template("paquete_modelo_arbol.py.j2", ctx)
    usage = eg._render_code_template("usar_modelo_arbol.py.j2",
                                     {"feature_names": list(feature_names)})
    probe = ("import runpy, numpy as np\nns = runpy.run_path('usar_modelo.py')\n"
             "import json\nd = json.load(open('modelo.json', encoding='utf-8'))\n"
             "print('USO', ns['predecir'](d['check']['X'][0]))\n")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "modelo.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        (Path(tmp) / "reconstruir_modelo.py").write_text(script, encoding="utf-8")
        proc = subprocess.run([sys.executable, "reconstruir_modelo.py"], cwd=tmp,
                              capture_output=True, text=True, timeout=120)
        ok = proc.returncode == 0 and (Path(tmp) / "decision_tree_model.pkl").exists()
        if not ok:
            raise RuntimeError(f"tree package script failed: {proc.stdout}\n{proc.stderr}")
        (Path(tmp) / "usar_modelo.py").write_text(usage, encoding="utf-8")
        (Path(tmp) / "probe.py").write_text(probe, encoding="utf-8")
        use = subprocess.run([sys.executable, "probe.py"], cwd=tmp, capture_output=True,
                             text=True, timeout=120)
    return {"zip_name": "modelo_arbol.zip",
            "files": ["modelo.json", "reconstruir_modelo.py", "LEEME.txt"],
            "model_json": doc, "script_context": ctx, "script": script,
            "script_stdout": proc.stdout, "verified_by_execution": ok,
            "usage_code_runs_on_the_pickle": use.returncode == 0,
            "usage_code_stdout": use.stdout.strip(),
            "usage_code_stderr_tail": use.stderr.strip().splitlines()[-1:] if use.returncode else []}


# ---------------------------------------------------------------------------
# The case
# ---------------------------------------------------------------------------
def _setup(eg, spec: dict, tmp: Path):
    state, label, ref = eg._lr_setup({**spec, "model": "Linear"}, tmp)
    for k in ("navigation", "active_tab_lr", "model_type_lr"):
        state.pop(k, None)
    state.update({"navigation": NAV, "active_tab": 1, "selected_dataset": label,
                  "unified_dataset_selector": label,
                  "tree_type": spec.get("tree", "Clasificación")})
    return state, label, ref


def case_tree(eg, name: str, spec: dict, writer) -> None:
    import tempfile

    from golden_render import LessonRenderer, strip_chrome

    full = spec.get("full", False)
    want_views = spec.get("views", True)
    views: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        state, label, ref = _setup(eg, spec, Path(tmpdir))
        r = LessonRenderer(_page(), state)
        first = r.render()
        if full:
            views["train_before"] = strip_chrome(first)
        r.set("select", spec["criterion"], label="Criterio de División:", render=False)
        r.set("slider", spec["max_depth"], label="Profundidad Máxima:", render=False)
        r.set("slider", spec["mss"], label="Muestras Mínimas para División:", render=False)
        views["train"] = strip_chrome(r.click(label="Entrenar Modelo"))
        st_state = r.session.state
        inp = {**ref, "app_label": label, "tree_type": spec.get("tree", "Clasificación"),
               "criterion": spec["criterion"], "max_depth": spec["max_depth"],
               "min_samples_split": spec["mss"], "test_size": 0.3, "random_state": 42,
               "function": "mltutor.apps.decission_tree_app.run_decision_trees_app, rendered "
                           "headlessly by tools/golden_render.py"}
        if not st_state.get("is_trained"):
            doc = {"meta": eg.make_meta({"texts": "exact"}, "Training fails in the app; the "
                                        "fixture records what the student sees."),
                   "input": inp, "expected": {"trained": False, "views": views}}
            writer.write_json(eg.PARITY_DIR / MODULE / f"{name}.json", doc, dumps_compact)
            return
        model = st_state["tree_model"]
        X_train, X_test = st_state["X_train"], st_state["X_test"]
        y_train, y_test = st_state["y_train"], st_state["y_test"]
        feature_names = list(st_state["feature_names"])
        class_names = st_state.get("class_names")
        tree_type = st_state["tree_type"]
        classifier = hasattr(model, "classes_")

        def view(key):
            try:
                views[key] = strip_chrome(r.render())
            except Exception as exc:
                views[key] = {"app_error": f"{type(exc).__name__}: {exc}"}

        pairs = {"default": [0, 1 if len(feature_names) > 1 else 0]}
        form_default, custom, default_x, custom_x = [], [], None, None
        if want_views:
            st_state["active_tab"] = 2
            view("evaluation")
            st_state["active_tab"] = 3
            st_state["viz_type"] = "Árbol"
            view("viz_tree")
            if full:
                r.set("slider", 20, label="Ancho de figura:", render=False)
                r.set("slider", 15, label="Alto de figura:", render=False)
                view("viz_tree_20x15")
            st_state["viz_type"] = "Texto"
            view("viz_text")
            if tree_type == "Clasificación":
                st_state["viz_type"] = "Frontera"
                bkey = "viz_boundary"
            else:
                st_state["viz_type"] = "Residuos"
                view("viz_residuals")
                st_state["viz_type"] = "Superficie"
                bkey = "viz_surface"
            view(f"{bkey}_default")
            if spec.get("pair"):
                a, b = spec["pair"]
                st_state["viz_feature1"], st_state["viz_feature2"] = feature_names[a], feature_names[b]
                view(f"{bkey}_pair")
                pairs["pair"] = [a, b]
            if spec.get("same_pair"):
                st_state["viz_feature1"] = st_state["viz_feature2"] = feature_names[0]
                view(f"{bkey}_same_feature")
            st_state.pop("viz_feature1", None)
            st_state.pop("viz_feature2", None)
            if tree_type == "Clasificación":
                st_state["viz_type"] = "ROC"
                view("viz_roc")
            st_state["active_tab"] = 4
            view("features")
            st_state["active_tab"] = 5
            view("prediction_form")
            form_default = eg._form_values(r.root)
            recorded: list[np.ndarray] = []
            with eg._record_predict(type(model), recorded):
                r.session.clicked = r.find("button", "Realizar predicción").props["id"]
                view("prediction_default")
                default_x = recorded[0] if recorded else None
                recorded.clear()
                for i, kind, value, props in form_default:
                    if kind == "slider":
                        v = props["max_value"]
                    elif kind == "select":
                        v = props["options"][-1]
                    elif kind == "checkbox":
                        v = True
                    else:
                        v = value
                    st_state[f"feature_{i}"] = v
                    custom.append(v)
                r.session.clicked = r.find("button", "Realizar predicción").props["id"]
                view("prediction_custom")
                custom_x = recorded[0] if recorded else None
            st_state["active_tab"] = 6
            view("export")
            for key, lab in (("export_pickle", "📦 Exportar como Pickle"),
                             ("export_rules", "📝 Exportar Reglas como Texto"),
                             ("export_python", "🐍 Exportar como Código Python")):
                r.session.clicked = r.find("button", lab).props["id"]
                view(key)
        if full:
            st_state["active_tab"] = 2
            st_state["tree_type"] = "Regresión" if tree_type == "Clasificación" else "Clasificación"
            view("evaluation_after_switching_tree_type_without_retraining")

    from sklearn.tree import export_text

    train_idx, test_idx = X_train.index.to_numpy(), X_test.index.to_numpy()
    y_pred = model.predict(X_test)
    expected: dict[str, Any] = {
        "trained": True,
        "tree_type": tree_type,
        "estimator": type(model).__name__,
        "params": model.get_params(),
        "feature_columns": list(X_train.columns),
        "feature_names": feature_names,
        "class_names": class_names,
        "classes": getattr(model, "classes_", None),
        "train_indices": train_idx,
        "test_indices": test_idx,
        "y_test": np.asarray(y_test),
        "y_pred": y_pred,
        "tree": tree_arrays(model),
        "feature_importances": model.feature_importances_,
        "test_leaves": model.apply(X_test.to_numpy(dtype=np.float32)),
        "decision_paths_test_head": decision_path_rows(model, X_test.head(5).to_numpy()),
        "reference_impl": reference_check(X_train, y_train, model, classifier),
    }
    if classifier:
        proba = model.predict_proba(X_test)
        top2 = np.sort(proba, axis=1)[:, -2:]
        expected["proba"] = proba
        expected["leaf_argmax_ties_in_test"] = int(np.sum(np.isclose(top2[:, 1], top2[:, 0])))
        expected["argmax_rule"] = "predict = classes_[argmax(proba)]: first maximum on ties"
    if spec.get("mismatch"):
        expected.pop("proba", None)
        expected["reference_impl"].pop("trace", None)
        expected["views"] = {k: v for k, v in views.items()
                             if len(json.dumps(v, ensure_ascii=False, default=str)) < 300_000}
        expected["views_omitted"] = sorted(set(views) - set(expected["views"]))
        if classifier:
            expected["n_classes"] = int(len(model.classes_))
        doc = {"meta": eg.make_meta({"tree": "structure exact (see reference_impl)",
                                     "texts": "exact"},
                                    "Documents what the app does when the tree type does not "
                                    "match the task (default 'Clasificación' on a regression "
                                    "dataset, or 📈 Regresión on class codes)."),
               "input": inp, "expected": expected}
        writer.write_json(eg.PARITY_DIR / MODULE / f"{name}.json", doc, dumps_compact)
        return

    if classifier:
        expected["evaluation"] = eg._lr_classification_numbers(
            model, X_train, X_test, y_train, y_test, class_names)
    else:
        expected["evaluation"] = eg._lr_regression_numbers(y_test, y_pred)
    expected["permutation_importance"] = permutation_reference(model, X_test, y_test)
    expected["export_text"] = {
        "view_text": export_text(model, feature_names=feature_names, show_weights=True),
        "view_rule": "export_text(model, feature_names=feature_names, show_weights=True): "
                     "decimals=2, spacing=3, max_depth=10, class shown as classes_ (codes)",
        "rules_txt": export_text(model, feature_names=feature_names),
        "rules_txt_rule": "export_text(model, feature_names=feature_names) (💾 Exportar)",
    }
    if classifier and class_names is not None:
        expected["export_text"]["view_text_with_class_names_D40"] = export_text(
            model, feature_names=feature_names, show_weights=True,
            class_names=[str(class_names[int(c)]) for c in model.classes_])
    expected["plot_tree"] = plot_tree_layout(model, feature_names,
                                             class_names if tree_type == "Clasificación" else None)
    expected["python_code"] = python_code(eg, model, feature_names,
                                          class_names if classifier else None,
                                          np.vstack([X_train.to_numpy(dtype=float),
                                                     X_test.to_numpy(dtype=float)]))
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    two = {}
    for k, (a, b) in pairs.items():
        if a == b:
            continue
        cls = DecisionTreeClassifier if classifier else DecisionTreeRegressor
        p = model.get_params()
        m2 = cls(max_depth=p["max_depth"], min_samples_split=p["min_samples_split"],
                 criterion=p["criterion"], random_state=42)
        X2 = X_train.iloc[:, [a, b]].values
        m2.fit(X2, y_train)
        two[k] = {"features": [a, b], "tree": tree_arrays(m2),
                  "reference_impl_comparison": reference_check(
                      X2, y_train, m2, classifier)["comparison"]}
    expected["two_feature_models"] = two
    X_all = np.vstack([X_train.to_numpy(dtype=float), X_test.to_numpy(dtype=float)])
    expected["onnx_reference"] = onnx_reference(eg, model, X_test, X_all)
    expected["export_package"] = export_package(eg, model, list(X_train.columns), feature_names,
                                                class_names if classifier else None, X_test,
                                                label, len(train_idx), len(test_idx))
    if want_views:
        expected["prediction_form"] = {
            "controls": [{"index": i, "kind": k, "value": eg.to_jsonable(v),
                          **{p: eg.to_jsonable(props[p]) for p in
                             ("label", "min_value", "max_value", "step", "options", "display")
                             if p in props}}
                         for i, k, v, props in form_default],
            "custom_choice": "sliders at max_value, selects at their last option, checkboxes on",
            "custom_controls": [eg.to_jsonable(v) for v in custom],
            "default": _predict_vector(model, default_x),
            "custom": _predict_vector(model, custom_x),
            "note": "The app builds np.array(values, dtype=np.float32); predict compares the "
                    "float32 value with the float64 threshold. It shows 'Clase: "
                    "{class_names[int(predict)]}' or 'Valor: {:.4f}' and the decision path "
                    "(no probabilities for trees).",
        }
        expected["views"] = views
    tol = {
        "tree_structure": "exact: children_left/right, feature, n_node_samples, "
                          "weighted_n_node_samples, value_counts, missing_go_to_left",
        "threshold": "exact (bit): same float32 midpoint rule",
        "impurity_rel": 1e-12,
        "impurity_note": "sklearn's macOS arm64 wheel contracts a*b+c into FMA (entropy, MSE), "
                         "so its impurities differ from a non-FMA build by a few ulps; the tree "
                         "itself is identical (reference_impl)",
        "value": "classification fractions exact (counts / n); regression mean/median rel 1e-12",
        "feature_importances_rel": 1e-12,
        "predictions": "exact (classes) / rel 1e-12 (regression)",
        "proba": "exact",
        "metrics": "exact when predictions are exact (1e-12)",
        "curves": "exact",
        "permutation_importance_abs": 1e-12,
        "grid_axes": "exact", "grid_z": "exact (2-D tree is exact too)",
        "texts": "exact",
        "onnx_float32": "classes exact, proba 1e-7, regression rel 1e-6, after D47",
    }
    rationale = ("Exact parity is feasible and required (D36): scikit-learn draws the features "
                 "of every node in a random order (Fisher-Yates with its xorshift RNG seeded by "
                 "RandomState(42)) and keeps the FIRST feature that reaches the best proxy "
                 "improvement (strict '>'). Ties are frequent (see "
                 "reference_impl.nodes_with_exact_ties: the root of Iris, petal length vs petal "
                 "width), so a different draw order gives a different tree. "
                 "tools/cart_reference.py reproduces the arrays bit for bit; Rust must follow it.")
    doc = {"meta": eg.make_meta(tol, rationale),
           "input": {**inp, "boundary_pairs": pairs}, "expected": expected}
    writer.write_json(eg.PARITY_DIR / MODULE / f"{name}.json", doc, dumps_compact)


def _predict_vector(model, x):
    if x is None:
        return None
    out = {"input_float32": np.asarray(x[0], dtype=float).tolist(),
           "predict": model.predict(x)[0].item(),
           "leaf": int(model.apply(np.asarray(x, dtype=np.float32))[0]),
           "path": decision_path_rows(model, np.asarray(x, dtype=np.float32))[0]}
    if hasattr(model, "classes_"):
        out["predict_proba"] = model.predict_proba(x)[0].tolist()
    return out


# ---------------------------------------------------------------------------
# code/trees.json: literal templates of the lesson
# ---------------------------------------------------------------------------
CODE_LITERALS = {
    "visualizar_arbol.py.j2": ("mltutor.algorithms.code_examples.VIZ_TREE_CODE", None),
    "texto_arbol.py.j2": ("mltutor.algorithms.code_examples.TEXT_TREE_CODE", None),
    "camino_decision.py.j2": ("mltutor.algorithms.code_examples.DECISION_PATH_CODE", None),
    "usar_modelo_arbol.py.j2": ("mltutor.algorithms.code_examples.generate_tree_model_export_code",
                                {"feature_names": ["sepal length (cm)", "sepal width (cm)",
                                                   "petal length (cm)", "petal width (cm)"]}),
}


def case_code_trees(eg, writer) -> None:
    import importlib

    out = {}
    for template, (dotted, ctx) in CODE_LITERALS.items():
        mod, attr = dotted.rsplit(".", 1)
        obj = getattr(importlib.import_module(mod), attr)
        app_text = obj(**ctx) if callable(obj) else obj
        tctx = dict(ctx or {})
        rendered = eg._render_code_template(template, tctx)
        if rendered.strip("\n") != app_text.strip("\n"):
            raise AssertionError(f"{template} does not reproduce {dotted}")
        out[template] = {"source": dotted, "context": tctx, "text": app_text,
                         "compare": "text.strip('\\n') == render(template, context).strip('\\n')"}
    doc = {
        "meta": eg.make_meta({"text": "exact (ignoring leading/trailing newlines)"},
                             "Verbatim copies of the lesson's code strings. arbol_python.py.j2 "
                             "and arbol_python_corregido.py.j2 (export_tree_as_python_code) and "
                             "paquete_modelo_arbol.py.j2 are checked inside each trees/*.json "
                             "(python_code, export_package), because their context comes from "
                             "the trained tree."),
        "input": {"engine": "jinja2 with keep_trailing_newline=True, filter pyrepr = repr()"},
        "expected": out,
    }
    writer.write_json(eg.PARITY_DIR / "code" / "trees.json", doc, dumps_compact)


def register(cases: dict, eg) -> None:
    for name, spec in TREE_CASES.items():
        cases[f"{MODULE}/{name}"] = (lambda n, s: (lambda w: case_tree(eg, n, s, w)))(name, spec)
    cases["code/trees"] = lambda w: case_code_trees(eg, w)
