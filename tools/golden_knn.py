"""Phase 4 golden fixtures: the 🔍 K-Nearest Neighbors lesson (fase4-knn.md §6).

Registered by ``tools/export_golden.py`` (``build_cases``), which passes itself as ``eg``
so the shared helpers (Writer, make_meta, datasets, CSV cases, classification and
regression numbers, code rendering) are reused, not copied. ``golden_trees`` provides
``dumps_compact`` and the permutation-importance reference.

Each ``knn/<case>.json`` renders the real lesson headlessly (``golden_render``), trains
like the student (K, weights, metric, test size, no scaling), visits every tab and
sub-view and stores:

* the neighbours of every test row (indices in X_train order, distances), the engine
  scikit-learn chose (``kd_tree`` or ``brute``), the ties at the K-th neighbour and the
  check of ``tools/knn_reference.py`` (kd-tree with libc++ ``nth_element``, heap and
  ``simultaneous_sort``: same neighbours in the same order, bit for bit);
* predictions, probabilities, metrics and curves; permutation importance with its seed;
* the class-distance analysis (📏), the 2-D models of the boundary/surface (all data,
  as the app does) with their full grids, and the interactive view (its JS rules
  transcribed, compared with scikit-learn);
* the prediction form, what 💾 Exportar does today (Pickle + the regression snippet), the
  skl2onnx reference (and its ``weights='distance'`` bug), the ONNX graph proposed for
  Rust (D57) evaluated for real, and the "paquete Python" (D1) executed for real.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODULE = "knn"
NAV = "🔍 K-Nearest Neighbors"
KNN_CASES: dict[str, dict[str, Any]] = {
    # Classification
    "iris_k5_uniform_minkowski": {"dataset": "iris", "k": 5, "weights": "uniform",
                                  "metric": "minkowski", "pair": (2, 3), "same_pair": True,
                                  "full": True},
    "iris_k1_uniform_manhattan": {"dataset": "iris", "k": 1, "weights": "uniform",
                                  "metric": "manhattan"},
    "iris_k20_distance_manhattan": {"dataset": "iris", "k": 20, "weights": "distance",
                                    "metric": "manhattan", "pair": (2, 3)},
    "iris_k5_distance_euclidean": {"dataset": "iris", "k": 5, "weights": "distance",
                                   "metric": "euclidean"},
    "iris_k5_uniform_minkowski_test02": {"dataset": "iris", "k": 5, "weights": "uniform",
                                         "metric": "minkowski", "test_size": 0.2},
    "wine_k5_uniform_minkowski": {"dataset": "wine", "k": 5, "weights": "uniform",
                                  "metric": "minkowski", "pair": (0, 9)},
    "wine_k7_distance_manhattan": {"dataset": "wine", "k": 7, "weights": "distance",
                                   "metric": "manhattan"},
    "breast_cancer_k5_uniform_minkowski": {"dataset": "breast_cancer", "k": 5,
                                           "weights": "uniform", "metric": "minkowski",
                                           "pair": (20, 27)},
    "breast_cancer_k10_distance_euclidean": {"dataset": "breast_cancer", "k": 10,
                                             "weights": "distance", "metric": "euclidean"},
    "titanic_k5_uniform_minkowski": {"dataset": "titanic", "k": 5, "weights": "uniform",
                                     "metric": "minkowski", "pair": (1, 4)},
    "csv_string_target_k5_uniform_minkowski": {"csv": "string_target", "k": 5,
                                               "weights": "uniform", "metric": "minkowski"},
    "csv_string_target_k15_distance_euclidean": {"csv": "string_target", "k": 15,
                                                 "weights": "distance", "metric": "euclidean"},
    "csv_string_target_k20_test05_error": {"csv": "string_target", "k": 20,
                                           "weights": "uniform", "metric": "minkowski",
                                           "test_size": 0.5, "error": True},
    # Regression (the lesson takes the task of the dataset; there is no type button)
    "diabetes_k5_uniform_minkowski": {"dataset": "diabetes", "k": 5, "weights": "uniform",
                                      "metric": "minkowski", "pair": (2, 8), "full": True},
    "diabetes_k10_distance_manhattan": {"dataset": "diabetes", "k": 10, "weights": "distance",
                                        "metric": "manhattan"},
    "california_housing_k5_uniform_minkowski": {"dataset": "california_housing", "k": 5,
                                                "weights": "uniform", "metric": "minkowski",
                                                "pair": (6, 7)},
    "california_housing_k8_distance_euclidean": {"dataset": "california_housing", "k": 8,
                                                 "weights": "distance", "metric": "euclidean",
                                                 "views": False},
    "csv_numeric_regression_k5_uniform_minkowski": {"csv": "numeric_regression", "k": 5,
                                                    "weights": "uniform", "metric": "minkowski"},
    "csv_numeric_regression_k20_distance_manhattan": {"csv": "numeric_regression", "k": 20,
                                                      "weights": "distance",
                                                      "metric": "manhattan"},
}
JS_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA726", "#AB47BC", "#66BB6A", "#EF5350",
             "#42A5F5"]
BIG = 1000          # above this many training rows, X_train/y_train/idx_array go as a hash
GRID_REF_SAMPLE = 1500  # grid cells re-checked with knn_reference.py besides the tie cells
ONNX_HEAD = 200
NEAR_TIE_REL = 1e-9


def _page():
    from mltutor.apps.knn_app import run_knn_app

    return run_knn_app


def _sha(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    return hashlib.sha256(a.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Neighbours, ties and the reference implementation
# ---------------------------------------------------------------------------
def _exact_dist(Q: np.ndarray, B: np.ndarray, metric: str) -> np.ndarray:
    """Plain sequential distances (no FMA), all pairs; for tie detection only."""
    D = np.zeros((Q.shape[0], B.shape[0]))
    for j in range(B.shape[1]):
        d = Q[:, j][:, None] - B[:, j][None, :]
        D += np.abs(d) if metric == "manhattan" else d * d
    return D if metric == "manhattan" else np.sqrt(D)


def _tie_report(Q, B, ind_sk, k, metric) -> dict:
    """Where the K-th neighbour is tied, and whether 'lowest index wins' would differ."""
    import knn_reference as kr

    em = kr.effective_metric(metric)
    rows_tied, rows_near, rows_canon_diff = [], [], []
    for s in range(0, len(Q), 2000):
        D = _exact_dist(Q[s:s + 2000], B, em)
        srt = np.sort(D, axis=1)
        kth = srt[:, k - 1]
        nxt = srt[:, k] if B.shape[0] > k else np.full(len(D), np.inf)
        less = np.sum(D < kth[:, None], axis=1)
        eq = np.sum(D == kth[:, None], axis=1)
        canon = np.argsort(D, axis=1, kind="stable")[:, :k]
        for r in range(len(D)):
            g = s + r
            if eq[r] > k - less[r]:
                rows_tied.append({"row": g, "kth_distance": float(kth[r]),
                                  "n_at_kth_distance": int(eq[r]), "slots": int(k - less[r])})
            elif math.isfinite(nxt[r]) and nxt[r] - kth[r] <= NEAR_TIE_REL * max(kth[r], 1e-300):
                rows_near.append(g)
            if set(canon[r].tolist()) != set(ind_sk[g].tolist()):
                rows_canon_diff.append(g)
    return {"rows_with_tie_at_kth": rows_tied,
            "rows_with_near_tie_at_kth_rel_1e-9": rows_near,
            "rows_where_lowest_index_rule_gives_other_neighbours": rows_canon_diff,
            "note": "A tie at the K-th distance means scikit-learn had to choose; the kd_tree "
                    "keeps the FIRST point its traversal visits (not the lowest index). "
                    "knn_reference.py reproduces that choice."}


def _reference(model, X_fit: np.ndarray, Q: np.ndarray, store_tree: bool) -> dict:
    import knn_reference as kr

    out = kr.compare_with_sklearn(model, X_fit, Q)
    if model._fit_method == "kd_tree":
        _, idx_array, node_data, _ = model._tree.get_arrays()
        tree = {"leaf_size": 30, "n_nodes": int(len(node_data)),
                "idx_start": [int(n["idx_start"]) for n in node_data],
                "idx_end": [int(n["idx_end"]) for n in node_data],
                "is_leaf": [bool(n["is_leaf"]) for n in node_data]}
        if store_tree:
            tree["idx_array"] = np.asarray(idx_array).tolist()
        else:
            tree["idx_array_sha256_int64"] = hashlib.sha256(
                np.ascontiguousarray(np.asarray(idx_array, dtype="<i8")).tobytes()).hexdigest()
        out["kd_tree"] = tree
    out["implementation"] = ("tools/knn_reference.py (transcription of sklearn 1.7.2 "
                             "_binary_tree.pxi.tp/_kd_tree.pyx.tp/_heap.pyx/_sorting.pyx and "
                             "libc++ std::nth_element)")
    return out


def neighbours_block(model, X_fit: np.ndarray, Q: np.ndarray, store_tree: bool) -> dict:
    dist, ind = model.kneighbors(Q)
    return {
        "fit_method": model._fit_method,
        "effective_metric": model.effective_metric_,
        "n_samples_fit": int(model.n_samples_fit_),
        "indices": ind,
        "indices_are": "positions in the training matrix (X_train rows in train_indices order)",
        "distances": dist,
        "order": "ascending distance; ties in the order of the kd-tree heap after "
                 "simultaneous_sort (kd_tree) or of the index (brute)",
        "ties": _tie_report(Q, X_fit, ind, model.n_neighbors, model.metric),
        "reference_impl": _reference(model, X_fit, Q, store_tree),
    }


# ---------------------------------------------------------------------------
# 📏 Análisis de distancias
# ---------------------------------------------------------------------------
def distance_analysis(X: pd.DataFrame, y: pd.Series, metric: str, class_names,
                      test_size: float) -> dict:
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.model_selection import train_test_split

    def matrix(Xtr, ytr):
        classes = np.unique(ytr)
        n = len(classes)
        M = np.zeros((n, n))
        Mx = np.zeros((n, n))
        for i, c1 in enumerate(classes):
            for j, c2 in enumerate(classes):
                if i != j:
                    a, b = Xtr[ytr == c1], Xtr[ytr == c2]
                    M[i, j] = np.mean(pairwise_distances(a, b, metric=metric))
                    import knn_reference as kr
                    Mx[i, j] = np.mean(_exact_dist(a, b, kr.effective_metric(metric)))
        # D67 (not in the app): mean distance between distinct pairs of the same class
        within = []
        for c in classes:
            a = Xtr[ytr == c]
            import knn_reference as kr
            D = _exact_dist(a, a, kr.effective_metric(metric))
            n = len(a)
            within.append(float(D[np.triu_indices(n, 1)].sum() / (n * (n - 1) / 2)) if n > 1 else None)
        pos = M[M > 0]
        mn, mx = (float(np.min(pos)), float(np.max(M))) if pos.size else (None, None)
        ratio = mx / mn if mn and mn > 0 else 0
        verdict = ("success" if ratio > 5 else "warning" if ratio > 2 else "error")
        labels = ([class_names[int(c)] for c in classes] if class_names
                  else [f"Clase {int(c)}" for c in classes])
        return {"classes": classes, "class_labels": labels, "matrix": M,
                "matrix_exact_distances": Mx,
                "max_rel_diff_vs_exact": float(np.max(np.abs(M - Mx) / np.where(Mx > 0, Mx, 1))),
                "min_distance": mn, "max_distance": mx, "ratio": ratio,
                "metrics_shown": {"Distancia Mínima": f"{mn:.3f}" if mn is not None else None,
                                  "Distancia Máxima": f"{mx:.3f}" if mx is not None else None,
                                  "Ratio Max/Min": f"{ratio:.2f}"},
                "verdict": verdict,
                "within_class_mean_D67": within,
                "within_rule_D67": "mean exact distance over the n(n-1)/2 distinct pairs of each "
                                   "class (upper triangle); not shown by the Python app"}

    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    out = {"app": matrix(np.asarray(Xtr, dtype=float), np.asarray(ytr)),
           "app_rule": "train_test_split(knn_X, knn_y, test_size=0.3, random_state=42) ALWAYS "
                       "(whatever the slider says); class means of pairwise_distances(metric="
                       "model.metric): 'euclidean' = sklearn euclidean_distances (dot-product "
                       "formula, BLAS), 'minkowski' = scipy cdist p=2, 'manhattan' = exact sums; "
                       "diagonal left at 0; min over the cells > 0; ratio = max/min; verdict "
                       "ratio > 5 success, > 2 warning, else error"}
    if abs(test_size - 0.3) > 1e-12:
        Xtr2, _, ytr2, _ = train_test_split(X, y, test_size=test_size, random_state=42)
        out["with_model_split_D56"] = matrix(np.asarray(Xtr2, dtype=float), np.asarray(ytr2))
    return out


# ---------------------------------------------------------------------------
# 2-D models (🌈 Frontera / 🌐 Superficie / 🎮 Interactiva): ALL the data, as the app
# ---------------------------------------------------------------------------
def grid_axes(X2: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(X2)
    h = 0.1 if n > 1000 else 0.05 if n > 500 else 0.02
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xr, yr = x_max - x_min, y_max - y_min
    if max(xr, yr) > 100:
        h = max(h, max(xr, yr) / 200)
    h = min(h, xr / 10, yr / 10)
    xp, yp = np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    if len(xp) < 2:
        xp = np.linspace(x_min, x_max, 10)
    if len(yp) < 2:
        yp = np.linspace(y_min, y_max, 10)
    if len(xp) * len(yp) > 50000:
        xp = np.linspace(x_min, x_max, min(200, len(xp)))
        yp = np.linspace(y_min, y_max, min(200, len(yp)))
    return xp, yp, h


def two_feature_model(model, X: pd.DataFrame, y: pd.Series, pair, classifier: bool) -> dict:
    import knn_reference as kr
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    a, b = pair
    X2 = X.iloc[:, [a, b]].to_numpy(dtype=float)
    yv = np.asarray(y)
    cls = KNeighborsClassifier if classifier else KNeighborsRegressor
    m2 = cls(n_neighbors=model.n_neighbors, weights=model.weights, metric=model.metric)
    m2.fit(X2, yv)
    xp, yp, h = grid_axes(X2)
    xx, yy = np.meshgrid(xp, yp)
    Q = np.c_[xx.ravel(), yy.ravel()]
    Z = m2.predict(Q)
    # Which cells depend on the tie rule (lowest index would choose other neighbours)?
    _, ind = m2.kneighbors(Q)
    sens = _tie_report(Q, X2, ind, m2.n_neighbors,
                       m2.metric)["rows_where_lowest_index_rule_gives_other_neighbours"]
    rng = np.random.default_rng(42)
    sample = sorted(set(sens) | set(rng.choice(len(Q), min(GRID_REF_SAMPLE, len(Q)),
                                               replace=False).tolist()))
    ref = kr.compare_with_sklearn(m2, X2, Q[sample])
    # Z that 'lowest index wins' would give (to show what the kd-tree port buys)
    D_c, I_c = [], []
    for s in range(0, len(Q), 2000):
        D = _exact_dist(Q[s:s + 2000], X2, kr.effective_metric(model.metric))
        idx = np.argsort(D, axis=1, kind="stable")[:, :m2.n_neighbors]
        D_c.append(np.take_along_axis(D, idx, 1))
        I_c.append(idx)
    D_c, I_c = np.vstack(D_c), np.vstack(I_c)
    if classifier:
        Zc = kr.predict_class(D_c, I_c, yv.astype(int), len(m2.classes_), model.weights)
        z_diff = int(np.sum(m2.classes_[Zc] != Z))
    else:
        Zc = kr.predict_value(D_c, I_c, yv, model.weights)
        z_diff = int(np.sum(~np.isclose(Zc, Z, rtol=1e-12, atol=1e-12)))
    out = {
        "features": [a, b],
        "fit_on": "ALL rows of the dataset (knn_X, knn_y), not X_train",
        "fit_method": m2._fit_method,
        "n_samples": int(len(X2)),
        "step_h": h,
        "x": xp, "y": yp,
        "z_shape": [len(yp), len(xp)],
        "z": Z.reshape(len(yp), len(xp)) if classifier else Z.reshape(len(yp), len(xp)),
        "z_layout": "rows = y values, columns = x values (np.meshgrid)",
        "cells_whose_neighbours_change_with_lowest_index_rule": len(sens),
        "cells_whose_prediction_changes_with_lowest_index_rule": z_diff,
        "reference_impl_on_tie_cells_plus_sample": {**ref, "cells_checked": len(sample)},
    }
    if not classifier:
        out["z_sha256_f64"] = _sha(Z)
    return out


# The JS of create_interactive_knn_visualization, transcribed (classifyPoint).
def js_classify(point, data_xy, labels, k, metric, weights) -> dict:
    px, py = point
    d = []
    for i, ((x, yv), lab) in enumerate(zip(data_xy, labels)):
        if metric == "manhattan":
            dist = abs(px - x) + abs(py - yv)
        else:  # euclidean and minkowski (p=2)
            dist = math.sqrt(math.pow(px - x, 2) + math.pow(py - yv, 2))
        d.append((dist, i, int(lab)))
    d.sort(key=lambda t: t[0])  # Array.prototype.sort is stable (ES2019)
    nb = d[:k]
    votes: dict[int, float] = {}
    for dist, _, lab in nb:
        w = (1.0 if dist == 0 else 1.0 / dist) if weights == "distance" else 1.0
        votes[lab] = votes.get(lab, 0.0) + w
    best, best_score = None, 0.0
    for lab in sorted(votes):  # Object.entries: integer keys in ascending order
        if votes[lab] > best_score:
            best, best_score = lab, votes[lab]
    total = sum(votes.values())
    return {"predicted_class": best, "confidence": best_score / total if total > 0 else 0,
            "neighbours": [i for _, i, _ in nb], "distances": [t[0] for t in nb]}


def interactive_view(model, X: pd.DataFrame, y: pd.Series, pair, class_names) -> dict:
    from sklearn.neighbors import KNeighborsClassifier

    a, b = pair
    X2 = X.iloc[:, [a, b]].to_numpy(dtype=float)
    yv = np.asarray(y).astype(int)
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    m2 = KNeighborsClassifier(n_neighbors=model.n_neighbors, weights=model.weights,
                              metric=model.metric).fit(X2, yv)
    pts = [((x_min + x_max) / 2, (y_min + y_max) / 2), tuple(X2[0]),
           (X2[:, 0].min(), X2[:, 1].min()), (X2[:, 0].max(), X2[:, 1].max()),
           (float(np.median(X2[:, 0])), float(np.median(X2[:, 1])))]
    samples = []
    for p in pts:
        js = js_classify(p, X2, yv, model.n_neighbors, model.metric, model.weights)
        q = np.array([p], dtype=float)
        dist, ind = m2.kneighbors(q)
        proba = m2.predict_proba(q)[0]
        pred = int(m2.predict(q)[0])
        samples.append({"point": list(map(float, p)), "js": js,
                        "sklearn_2d_all_data": {"predict": pred, "proba": proba,
                                                "confidence_max_proba": float(np.max(proba)),
                                                "neighbours": ind[0], "distances": dist[0]},
                        "same_prediction": js["predicted_class"] == pred,
                        "same_neighbour_set": set(js["neighbours"]) == set(ind[0].tolist())})
    classes = np.unique(yv)
    return {
        "features": [a, b],
        "data": "ALL rows of knn_X[:, pair] with knn_y (not the training split)",
        "bounds": {"xMin": x_min, "xMax": x_max, "yMin": y_min, "yMax": y_max},
        "palette": JS_COLORS,
        "colour_rule": "colors[class_code % 8]",
        "class_info": [{"id": int(c), "name": class_names[int(c)] if class_names else
                        f"Clase {int(c)}", "color": JS_COLORS[int(c) % 8]} for c in classes],
        "training_points_head": [{"x": float(X2[i, 0]), "y": float(X2[i, 1]), "class": int(yv[i]),
                                  "color": JS_COLORS[int(yv[i]) % 8]} for i in range(5)],
        "n_points": int(len(X2)),
        "initial_controls": {"k": int(model.n_neighbors), "k_slider": [1, 15],
                             "k_note": "the <input type=range> clamps the thumb to 15 but "
                                       "currentK starts at the model's K (up to 20)",
                             "metric": model.metric, "weights": model.weights,
                             "show_confidence": True, "show_distances": True},
        "js_rules": "distances on the raw 2-D values (minkowski = euclidean); stable sort by "
                    "distance (ties: dataset order); K first; votes 1 (uniform) or 1/d with "
                    "d = 0 -> weight 1 (the others keep 1/d: NOT sklearn's rule); winner = "
                    "first class (ascending code) with the strictly largest score; confidence "
                    "= winner score / total. Computed in the browser, never by the model.",
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# ONNX: skl2onnx reference and the graph proposed for Rust (D57)
# ---------------------------------------------------------------------------
def onnx_reference(model, X_test: pd.DataFrame, X_all: np.ndarray, classifier: bool) -> dict:
    from onnx.reference import ReferenceEvaluator
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType

    out: dict[str, Any] = {"producer": "skl2onnx 1.19.1 (reference only; the Python app has no "
                                       "ONNX for KNN)"}
    n = X_test.shape[1]
    head = min(len(X_test), ONNX_HEAD)
    for tname, T, dt in (("float_input", FloatTensorType, np.float32),
                         ("double_input", DoubleTensorType, np.float64)):
        opts = {id(model): {"zipmap": False}} if classifier else None
        onx = convert_sklearn(model, initial_types=[("X", T([None, n]))], options=opts,
                              target_opset={"": 15, "ai.onnx.ml": 3})
        sess = ReferenceEvaluator(onx)
        sel = X_all if len(X_all) <= 3000 else X_all[np.r_[0:1500, -1500:0]]
        res = sess.run(None, {"X": sel.astype(dt)})
        entry = {"op_types": sorted({nd.op_type for nd in onx.graph.node}),
                 "n_nodes": len(onx.graph.node),
                 "bytes": len(onx.SerializeToString()),
                 "rows_compared": int(len(sel))}
        if classifier:
            lab = np.asarray(res[0]).ravel()
            entry["label_mismatches"] = int(np.sum(lab != model.predict(sel)))
            entry["proba_max_abs_diff"] = float(np.max(np.abs(np.asarray(res[1], dtype=float)
                                                              - model.predict_proba(sel))))
        else:
            v = np.asarray(res[0], dtype=float).ravel()
            entry["max_abs_diff"] = float(np.max(np.abs(v - model.predict(sel))))
        out[tname] = entry
    out["head_rows"] = head
    out["how_it_works"] = (
        "The whole training set goes into the graph as an initializer. A Scan computes, row by "
        "row, the distances to every training row (Sub, Mul/Abs, ReduceSum, Sqrt), TopK(k, "
        "largest=0) picks the nearest (ties: lower index first, ONNX spec), "
        "ArrayFeatureExtractor/Gather fetch their labels, and the votes are summed per class "
        "(Equal + Cast + ReduceSum) and divided by k (or by the weights).")
    out["known_bug_weights_distance"] = (
        "skl2onnx 1.19.1 with opset >= 11 returns TopK's distances multiplied by -1 "
        "(onnx_nearest_neighbors_indices_k keeps the opset-10 negation), so Max(d, 1e-6) is "
        "1e-6 for every neighbour and weights='distance' behaves as 'uniform' (confirmed with "
        "onnxruntime 1.31 and onnx.reference). It also replaces sklearn's zero-distance rule "
        "by 1/max(d, 1e-6). With float_input the training set is stored as float32, so ties "
        "and near ties change.")
    return out


def _onnx_proposal_model(model, X_train: np.ndarray, y_train, classifier: bool):
    """D57: the graph Rust (mltutor-export) should write. Built with onnx.helper."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    k = int(model.n_neighbors)
    em = "manhattan" if model.metric == "manhattan" else "euclidean"
    n_feat = X_train.shape[1]
    inits = [numpy_helper.from_array(np.ascontiguousarray(X_train, dtype=np.float64),
                                     "train_X"),
             numpy_helper.from_array(np.array([1], dtype=np.int64), "axes1")]
    # Scan body: one query row -> (k,) distances (squared/abs sums) and (k,) indices
    body_nodes = [
        helper.make_node("Constant", [], ["k_in"], name="body_k",
                         value=numpy_helper.from_array(np.array([k], dtype=np.int64), "kc")),
        helper.make_node("Constant", [], ["axes1_in"], name="body_axes",
                         value=numpy_helper.from_array(np.array([1], dtype=np.int64), "ac")),
        helper.make_node("Sub", ["x_row", "train_X_in"], ["diff"], name="body_sub"),
        (helper.make_node("Abs", ["diff"], ["term"], name="body_abs") if em == "manhattan"
         else helper.make_node("Mul", ["diff", "diff"], ["term"], name="body_sq")),
        helper.make_node("ReduceSum", ["term", "axes1_in"], ["r"], keepdims=0, name="body_sum"),
        helper.make_node("TopK", ["r", "k_in"], ["r_k", "i_k"], axis=0, largest=0, sorted=1,
                         name="body_topk"),
        helper.make_node("Identity", ["train_X_in"], ["train_X_out"], name="body_id_x"),
    ]
    body = helper.make_graph(
        body_nodes, "knn_row",
        [helper.make_tensor_value_info("train_X_in", TensorProto.DOUBLE, ["N", n_feat]),
         helper.make_tensor_value_info("x_row", TensorProto.DOUBLE, [n_feat])],
        [helper.make_tensor_value_info("train_X_out", TensorProto.DOUBLE, ["N", n_feat]),
         helper.make_tensor_value_info("r_k", TensorProto.DOUBLE, [k]),
         helper.make_tensor_value_info("i_k", TensorProto.INT64, [k])])
    nodes = [
        helper.make_node("Cast", ["float_input"], ["X64"], to=TensorProto.DOUBLE, name="cast_in"),
        helper.make_node("Scan", ["train_X", "X64"], ["train_X_final", "rdist_k", "idx_k"],
                         body=body, num_scan_inputs=1, name="scan_rows"),
    ]
    nodes.append(helper.make_node("Sqrt", ["rdist_k"], ["dist_k"], name="sqrt")
                 if em == "euclidean" else
                 helper.make_node("Identity", ["rdist_k"], ["dist_k"], name="dist_id"))
    # Weights: uniform -> 1; distance -> 1/d, or (d == 0) on the rows with a zero distance
    if model.weights == "distance":
        inits.append(numpy_helper.from_array(np.array([0.0], dtype=np.float64), "zero"))
        nodes += [
            helper.make_node("Equal", ["dist_k", "zero"], ["is_zero"], name="eq0"),
            helper.make_node("Cast", ["is_zero"], ["zero_w"], to=TensorProto.DOUBLE, name="c0"),
            helper.make_node("ReduceMax", ["zero_w"], ["row_has_zero_f"], axes=[1], keepdims=1,
                             name="rowmax"),
            helper.make_node("Cast", ["row_has_zero_f"], ["row_has_zero"], to=TensorProto.BOOL,
                             name="c1"),
            helper.make_node("Reciprocal", ["dist_k"], ["inv_d"], name="inv"),
            helper.make_node("Where", ["row_has_zero", "zero_w", "inv_d"], ["w"], name="where"),
        ]
    else:
        nodes.append(helper.make_node("Shape", ["dist_k"], ["dk_shape"], name="shape_w"))
        nodes.append(helper.make_node("ConstantOfShape", ["dk_shape"], ["w"],
                                      value=helper.make_tensor("one", TensorProto.DOUBLE, [1],
                                                               [1.0]), name="ones"))
    outputs = []
    if classifier:
        classes = np.asarray(model.classes_)
        codes = np.searchsorted(classes, np.asarray(y_train)).astype(np.int64)
        inits += [numpy_helper.from_array(codes, "train_y_code"),
                  numpy_helper.from_array(classes.astype(np.int64), "classes"),
                  numpy_helper.from_array(np.arange(len(classes), dtype=np.int64), "class_range"),
                  numpy_helper.from_array(np.array([2], dtype=np.int64), "axes2")]
        nodes += [
            helper.make_node("Gather", ["train_y_code", "idx_k"], ["lab_k"], axis=0, name="g_lab"),
            # one-hot without OneHot (onnxruntime has no double OneHot): Equal against 0..C-1
            helper.make_node("Unsqueeze", ["lab_k", "axes2"], ["lab3"], name="unsq_lab"),
            helper.make_node("Equal", ["lab3", "class_range"], ["oh_b"], name="eq_cls"),
            helper.make_node("Cast", ["oh_b"], ["oh"], to=TensorProto.DOUBLE, name="cast_oh"),
            helper.make_node("Unsqueeze", ["w", "axes2"], ["w3"], name="unsq"),
            helper.make_node("Mul", ["oh", "w3"], ["votes3"], name="mulv"),
            helper.make_node("ReduceSum", ["votes3", "axes1"], ["votes"], keepdims=0, name="sumv"),
            helper.make_node("ReduceSum", ["votes", "axes1"], ["norm"], keepdims=1, name="sumn"),
            helper.make_node("Div", ["votes", "norm"], ["proba64"], name="div"),
            helper.make_node("ArgMax", ["votes"], ["arg"], axis=1, keepdims=0, name="argmax"),
            helper.make_node("Gather", ["classes", "arg"], ["label"], axis=0, name="g_cls"),
            helper.make_node("Cast", ["proba64"], ["probabilities"], to=TensorProto.FLOAT,
                             name="cast_p"),
        ]
        outputs = [helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
                   helper.make_tensor_value_info("probabilities", TensorProto.FLOAT,
                                                 [None, len(classes)])]
    else:
        inits.append(numpy_helper.from_array(np.asarray(y_train, dtype=np.float64), "train_y"))
        nodes += [
            helper.make_node("Gather", ["train_y", "idx_k"], ["y_k"], axis=0, name="g_y"),
            helper.make_node("Mul", ["y_k", "w"], ["yw"], name="mul_yw"),
            helper.make_node("ReduceSum", ["yw", "axes1"], ["num"], keepdims=1, name="sum_num"),
            helper.make_node("ReduceSum", ["w", "axes1"], ["den"], keepdims=1, name="sum_den"),
            helper.make_node("Div", ["num", "den"], ["pred64"], name="div_y"),
            helper.make_node("Cast", ["pred64"], ["variable"], to=TensorProto.FLOAT, name="cast_y"),
        ]
        outputs = [helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1])]
    graph = helper.make_graph(
        nodes, "mltutor_knn",
        [helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [None, n_feat])],
        outputs, initializer=inits)
    m = helper.make_model(graph, producer_name="mltutor-proposal",
                          opset_imports=[helper.make_opsetid("", 15)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def onnx_proposal(model, X_train: np.ndarray, y_train, X_test: np.ndarray, classifier: bool) -> dict:
    from onnx.reference import ReferenceEvaluator

    m = _onnx_proposal_model(model, X_train, y_train, classifier)
    sess = ReferenceEvaluator(m)
    X_all = np.vstack([X_train, X_test])
    sel = X_all if len(X_all) <= 3000 else X_all[np.r_[0:1500, -1500:0]]
    f32 = sel.astype(np.float32)
    res = sess.run(None, {"float_input": f32})
    out = {"producer": "tools/golden_knn.py::_onnx_proposal_model (onnx.helper), checked with "
                       "onnx.checker and run with onnx.reference (also validated with "
                       "onnxruntime 1.31: same labels except ties at the K-th distance)",
           "opset": 15, "ir_version": 8, "input": "float_input float32 [N, n_features]",
           "op_types": sorted({nd.op_type for nd in m.graph.node}),
           "nodes": [{"op_type": nd.op_type, "name": nd.name, "inputs": list(nd.input),
                      "outputs": list(nd.output)} for nd in m.graph.node],
           "bytes": len(m.SerializeToString()), "rows_compared": int(len(sel)),
           "head_input_float32": f32[:5].astype(float)}
    # sklearn on the SAME float32-rounded inputs (what the ONNX sees) and on the f64 originals
    sk32 = model.predict(f32.astype(np.float64))
    if classifier:
        lab = np.asarray(res[0])
        out["label_mismatches_vs_sklearn_on_float32_inputs"] = int(np.sum(lab != sk32))
        out["label_mismatches_vs_sklearn_on_float64_inputs"] = int(np.sum(lab != model.predict(sel)))
        out["proba_max_abs_diff_float32_inputs"] = float(np.max(np.abs(
            np.asarray(res[1], dtype=float) - model.predict_proba(f32.astype(np.float64)))))
        out["head"] = {"label": lab[:5], "probabilities": np.asarray(res[1], dtype=float)[:5]}
    else:
        v = np.asarray(res[0], dtype=float).ravel()
        rel = np.abs(v - sk32) / np.maximum(np.abs(sk32), 1e-12)
        out["max_rel_diff_vs_sklearn_on_float32_inputs"] = float(np.max(rel))
        out["n_rel_diff_gt_1e-6"] = int(np.sum(rel > 1e-6))
        out["head"] = {"variable": v[:5]}
    out["note"] = ("Differences can only come from ties at the K-th distance (TopK keeps the "
                   "lower index, the kd_tree the first visited) and from the float32 input "
                   "rounding; probabilities and values are cast to float32 at the end.")
    return out


# ---------------------------------------------------------------------------
# Python package (D1/D57): modelo.json + reconstruir_modelo.py (refits: KNN stores data)
# ---------------------------------------------------------------------------
def export_package(eg, model, X_train: pd.DataFrame, y_train, feature_names, class_names,
                   X_test: pd.DataFrame, dataset_label: str, test_size: float) -> dict:
    import subprocess
    import sys
    import tempfile

    classifier = hasattr(model, "classes_")
    X_check = X_test.head(5)
    d_chk, i_chk = model.kneighbors(X_check)
    doc = {
        "format": "mltutor-model",
        "format_version": 1,
        "estimator": ("sklearn.neighbors.KNeighborsClassifier" if classifier
                      else "sklearn.neighbors.KNeighborsRegressor"),
        "params": model.get_params(),
        "feature_names_in": list(X_train.columns),
        "display_names": list(feature_names),
        "n_features_in": int(X_train.shape[1]),
        "X_train": X_train.to_numpy(dtype=float),
        "y_train": np.asarray(y_train),
        "training": {"dataset": dataset_label, "test_size": test_size, "random_state": 42,
                     "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                     "scaling": None, "fit_method": model._fit_method},
        "check": {"X": X_check.to_numpy(dtype=float), "prediction": model.predict(X_check),
                  "neighbours": i_chk, "distances": d_chk},
    }
    if classifier:
        doc.update({"classes_": model.classes_,
                    "class_names": None if class_names is None else list(class_names)})
        doc["check"]["probabilities"] = model.predict_proba(X_check)
    doc = eg.to_jsonable(doc)
    ctx = {"estimator": doc["estimator"], "dataset": dataset_label, "classifier": classifier,
           "json_name": "modelo.json", "pkl_name": "modelo_knn.pkl"}
    script = eg._render_code_template("paquete_modelo_knn.py.j2", ctx)
    leeme_ctx = {"estimator": doc["estimator"], "dataset": dataset_label,
                 "feature_names": list(feature_names),
                 "classes": ([{"value": int(c), "name": str(class_names[int(c)])}
                              for c in model.classes_] if classifier else None),
                 "n_train": int(len(X_train)), "n_test": int(len(X_test)), "test_size": test_size,
                 "n_neighbors": int(model.n_neighbors), "weights": model.weights,
                 "metric": model.metric, "pkl_name": "modelo_knn.pkl"}
    leeme = eg._render_code_template("paquete_leeme_knn.txt.j2", leeme_ctx)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "modelo.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        (Path(tmp) / "reconstruir_modelo.py").write_text(script, encoding="utf-8")
        proc = subprocess.run([sys.executable, "reconstruir_modelo.py"], cwd=tmp,
                              capture_output=True, text=True, timeout=300)
        ok = proc.returncode == 0 and (Path(tmp) / "modelo_knn.pkl").exists()
        if not ok:
            raise RuntimeError(f"knn package script failed: {proc.stdout}\n{proc.stderr}")
    if len(X_train) > BIG:  # the rows are those of train_indices; keep the fixture small
        doc["X_train"] = {"omitted": "X_train rows in train_indices order",
                          "sha256_f64": _sha(X_train.to_numpy(dtype=float))}
        doc["y_train"] = {"omitted": "y_train in train_indices order",
                          "sha256_f64": _sha(np.asarray(y_train, dtype=float))}
    return {"zip_name": "modelo_knn.zip",
            "files": ["modelo.json", "reconstruir_modelo.py", "LEEME.txt"],
            "model_json": doc, "script_context": ctx, "script": script,
            "leeme_context": leeme_ctx, "leeme": leeme,
            "script_stdout": proc.stdout, "verified_by_execution": ok}


# ---------------------------------------------------------------------------
# The case
# ---------------------------------------------------------------------------
_HTML_RE = {
    "training": re.compile(r"const trainingData = (\[.*?\]);", re.S),
    "class_info": re.compile(r"const classInfo = (\[.*?\]);", re.S),
}


def _summarise_html(tree):
    """Replace the 34 KB HTML of the interactive view by what it contains."""
    if isinstance(tree, dict):
        if tree.get("kind") == "html":
            html = tree.get("html", "")
            out = {"kind": "html", "height": tree.get("height"), "bytes": len(html.encode()),
                   "sha256": hashlib.sha256(html.encode()).hexdigest()}
            m = _HTML_RE["training"].search(html)
            if m:
                data = json.loads(m.group(1))
                out["training_data_n"] = len(data)
                out["training_data_head"] = data[:3]
            m = _HTML_RE["class_info"].search(html)
            if m:
                out["class_info"] = json.loads(m.group(1))
            for key in ("currentK", "currentMetric", "currentWeights", "showConfidence",
                        "showDistances"):
                mm = re.search(rf"let {key} = ([^;]+);", html)
                if mm:
                    out[key] = mm.group(1).strip().strip('"')
            mm = re.search(r"const bounds = \{(.*?)\};", html, re.S)
            if mm:
                out["bounds"] = {k: float(v) for k, v in
                                 re.findall(r"(\w+): ([-0-9.e]+)", mm.group(1))}
            vis = re.sub(r"<script>.*?</script>|<style>.*?</style>", "", html, flags=re.S)
            out["visible_texts"] = [t for t in (s.strip() for s in re.sub(
                r"<[^>]+>", "\n", vis).split("\n")) if t]
            return out
        return {k: _summarise_html(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_summarise_html(v) for v in tree]
    return tree


def _setup(eg, spec: dict, tmp: Path):
    state, label, ref = eg._lr_setup({**spec, "model": "Linear"}, tmp)
    for k in ("navigation", "active_tab_lr", "model_type_lr"):
        state.pop(k, None)
    state.update({"navigation": NAV, "active_tab_knn": 1, "selected_dataset": label,
                  "unified_dataset_selector": label})
    return state, label, ref


def _predict_vector(model, x):
    if x is None:
        return None
    x = np.asarray(x)
    d, i = model.kneighbors(x.astype(np.float64))
    out = {"input_float32": x[0].astype(float).tolist(),
           "predict": model.predict(x)[0].item(),
           "neighbours": i[0].tolist(), "distances": d[0].tolist()}
    if hasattr(model, "classes_"):
        out["predict_proba"] = model.predict_proba(x)[0].tolist()
    return out


def case_knn(eg, name: str, spec: dict, writer) -> None:
    import tempfile

    import golden_trees as gt
    from golden_render import LessonRenderer, strip_chrome

    full = spec.get("full", False)
    want_views = spec.get("views", True)
    test_size = spec.get("test_size", 0.3)
    views: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        state, label, ref = _setup(eg, spec, Path(tmpdir))
        r = LessonRenderer(_page(), state)
        first = r.render()
        if full:
            views["train_before"] = strip_chrome(first)
        r.set("number", spec["k"], label="Vecinos (K)", render=False)
        r.set("select", spec["weights"], label="Pesos", render=False)
        r.set("select", spec["metric"], label="Métrica", render=False)
        r.set("slider", test_size, label="Tamaño del Conjunto de Prueba:", render=False)
        views["train"] = strip_chrome(r.click(key="train_knn_button"))
        st = r.session.state
        inp = {**ref, "app_label": label, "n_neighbors": spec["k"], "weights": spec["weights"],
               "metric": spec["metric"], "test_size": test_size, "scaling": None,
               "function": "mltutor.apps.knn_app.run_knn_app, rendered headlessly by "
                           "tools/golden_render.py"}
        model = st["knn_model"]
        X, y = st["knn_X"], st["knn_y"]
        X_train, X_test = st["knn_Xtrain"], st["knn_Xtest"]
        y_train, y_test = st["knn_ytrain"], st["knn_ytest"]
        feature_names = list(st["knn_feature_names"])
        class_names = st.get("knn_class_names")
        task = st["knn_task_type"]
        classifier = task == "Clasificación"

        def view(key):
            try:
                views[key] = _summarise_html(strip_chrome(r.render()))
            except Exception as exc:
                views[key] = {"app_error": f"{type(exc).__name__}: {exc}"}

        if spec.get("error"):
            for tab in (2, 3, 4, 5, 6):
                st["active_tab_knn"] = tab
                view(f"tab_{tab}")
            doc = {"meta": eg.make_meta({"texts": "exact"},
                                        "K larger than the training set: the app trains (fit does "
                                        "not check) and every view that predicts fails."),
                   "input": inp,
                   "expected": {"trained": bool(st.get("knn_trained")),
                                "n_train": int(len(X_train)), "views": views}}
            writer.write_json(eg.PARITY_DIR / MODULE / f"{name}.json", doc, gt.dumps_compact)
            return

        pairs = {"default": [0, 1 if len(feature_names) > 1 else 0]}
        if spec.get("pair"):
            pairs["pair"] = list(spec["pair"])
        form_default, custom, default_x, custom_x = [], [], None, None
        if want_views:
            st["active_tab_knn"] = 2
            view("evaluation")
            st["active_tab_knn"] = 3
            if classifier:
                for vt in ("Probabilidades", "ROC", "Distancias"):
                    st["viz_type"] = vt
                    view(f"viz_{vt.lower()}")
                st["viz_type"] = "Frontera"
                bkey = "viz_boundary"
            else:
                st["viz_type"] = "Predicciones"
                view("viz_predictions")
                st["viz_type"] = "Superficie"
                bkey = "viz_surface"
            view(f"{bkey}_default")
            if spec.get("pair"):
                a, b = spec["pair"]
                st["viz_feature1"], st["viz_feature2"] = feature_names[a], feature_names[b]
                view(f"{bkey}_pair")
            if spec.get("same_pair"):
                st["viz_feature1"] = st["viz_feature2"] = feature_names[0]
                view(f"{bkey}_same_feature")
            st.pop("viz_feature1", None)
            st.pop("viz_feature2", None)
            if classifier:
                st["viz_type"] = "Interactiva"
                view("viz_interactive_default")
                if spec.get("pair"):
                    a, b = spec["pair"]
                    st["interactive_feature1"] = feature_names[a]
                    st["interactive_feature2"] = feature_names[b]
                    view("viz_interactive_pair")
                if spec.get("same_pair"):
                    st["interactive_feature1"] = st["interactive_feature2"] = feature_names[0]
                    view("viz_interactive_same_feature")
                st.pop("interactive_feature1", None)
                st.pop("interactive_feature2", None)
            st["active_tab_knn"] = 4
            view("features")
            st["active_tab_knn"] = 5
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
                    st[f"feature_{i}"] = v
                    custom.append(v)
                r.session.clicked = r.find("button", "Realizar predicción").props["id"]
                view("prediction_custom")
                custom_x = recorded[0] if recorded else None
            st["active_tab_knn"] = 6
            view("export")
            for key, k in (("export_code", "generate_code_lr"),
                           ("export_pickle", "download_pickle_lr")):
                r.session.clicked = r.find("button", key=k).props["id"]
                view(key)

    pos = lambda idx: X.index.get_indexer(idx).astype(int)  # noqa: E731
    train_idx, test_idx = pos(X_train.index), pos(X_test.index)
    Xtr = X_train.to_numpy(dtype=float)
    Xte = X_test.to_numpy(dtype=float)
    y_pred = model.predict(X_test)
    expected: dict[str, Any] = {
        "trained": True,
        "task": task,
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
        "neighbours_test": neighbours_block(model, Xtr, Xte, len(Xtr) <= BIG),
    }
    if classifier:
        expected["proba"] = model.predict_proba(X_test)
        expected["evaluation"] = eg._lr_classification_numbers(
            model, X_train, X_test, y_train, y_test, class_names)
        expected["distance_analysis"] = distance_analysis(X, y, model.metric, class_names,
                                                          test_size)
    else:
        expected["evaluation"] = eg._lr_regression_numbers(y_test, y_pred)
    expected["permutation_importance"] = gt.permutation_reference(model, X_test, y_test)
    expected["two_feature_models"] = {
        k: two_feature_model(model, X, y, p, classifier)
        for k, p in pairs.items() if p[0] != p[1]}
    if classifier:
        expected["interactive"] = {k: interactive_view(model, X, y, p, class_names)
                                   for k, p in pairs.items() if p[0] != p[1]}
    X_all = np.vstack([Xtr, Xte])
    expected["onnx_reference"] = onnx_reference(model, X_test, X_all, classifier)
    expected["onnx_proposal_D57"] = onnx_proposal(model, Xtr, np.asarray(y_train), Xte, classifier)
    expected["export_package"] = export_package(eg, model, X_train, y_train, feature_names,
                                                class_names if classifier else None, X_test,
                                                label, test_size)
    if want_views:
        expected["prediction_form"] = {
            "controls": [{"index": i, "kind": k, "value": eg.to_jsonable(v),
                          **{p: eg.to_jsonable(props[p]) for p in
                             ("label", "min_value", "max_value", "step", "options", "display")
                             if p in props}}
                         for i, k, v, props in form_default],
            "ranges_from": "knn_X: ALL rows of the dataset (not X_train)",
            "custom_choice": "sliders at max_value, selects at their last option, checkboxes on",
            "custom_controls": [eg.to_jsonable(v) for v in custom],
            "default": _predict_vector(model, default_x),
            "custom": _predict_vector(model, custom_x),
            "note": "The app builds np.array(values, dtype=np.float32) and predicts; KNN computes "
                    "the distances in float64 from those float32 values (D31: Rust uses the "
                    "float64 values). It shows 'Clase: {class_names[int(predict)]}' or 'Valor: "
                    "{:.4f}' and an st.info with K, weights and metric (no probabilities, no "
                    "neighbours: D58).",
        }
        expected["views"] = views
    tol = {
        "neighbours_kd_tree": "exact: same indices in the same order (knn_reference.py)",
        "neighbours_brute": "same set except rows listed in ties.rows_with_near_tie_at_kth "
                            "(euclidean brute ranks with sklearn's BLAS dot-product formula)",
        "distances_kd_tree": "bit-exact if the Euclidean sum is computed like the macOS arm64 "
                             "wheel (8-wide blocks without FMA, remainder with FMA; D53); "
                             "otherwise 4 ulp",
        "distances_brute_euclidean_rel": 1e-9,
        "predictions": "exact (classes) / rel 1e-12 (regression)",
        "proba": "abs 1e-15 (votes added in neighbour order)",
        "metrics_and_curves": "exact when predictions are exact (1e-12)",
        "permutation_importance_abs": 1e-12,
        "distance_analysis_rel": 1e-10,
        "grid_axes": "exact", "grid_z": "exact (classes) / rel 1e-12 (values)",
        "texts": "exact",
        "onnx_proposal": "labels exact except K-th ties; proba/values float32 (rel 1e-6)",
    }
    rationale = ("KNN has no training: parity is about WHICH neighbours are chosen. With "
                 "algorithm='auto' scikit-learn uses a kd-tree (n_features <= 15, K < n_train/2) "
                 "or brute force, and resolves ties at the K-th distance differently in each. "
                 "Ties are frequent with duplicated points (Titanic, Diabetes' sex, Iris in 2-D, "
                 "California's coordinates): with 'lowest index wins' up to 46 % of the cells of "
                 "a grid change. tools/knn_reference.py reproduces the kd-tree (libc++ "
                 "nth_element, heap, simultaneous_sort) bit for bit; Rust must follow it (D52).")
    doc = {"meta": eg.make_meta(tol, rationale),
           "input": {**inp, "boundary_pairs": pairs}, "expected": expected}
    writer.write_json(eg.PARITY_DIR / MODULE / f"{name}.json", doc, gt.dumps_compact)


def register(cases: dict, eg) -> None:
    for name, spec in KNN_CASES.items():
        cases[f"{MODULE}/{name}"] = (lambda n, s: (lambda w: case_knn(eg, n, s, w)))(name, spec)
    register_code(cases, eg)


# ---------------------------------------------------------------------------
# code/knn_training.json: the complete training script (knn_entrenamiento.py.j2)
# ---------------------------------------------------------------------------
TRAINING_TEMPLATE = "knn_entrenamiento.py.j2"
TRAINING_CASES: dict[str, dict[str, Any]] = {
    name: {"fixture": name, "sample": None}
    for name in KNN_CASES if not KNN_CASES[name].get("error")
    and KNN_CASES[name].get("dataset") not in ("titanic",)  # seaborn download (network)
}
TRAINING_CASES["iris_k5_uniform_minkowski_form_sample"] = {
    "fixture": "iris_k5_uniform_minkowski", "sample": "custom"}
TRAINING_CASES["diabetes_k5_uniform_minkowski_form_sample"] = {
    "fixture": "diabetes_k5_uniform_minkowski", "sample": "custom"}
TRAINING_CONTEXT_KEYS = ["dataset", "dataset_label", "csv_file", "csv_target", "csv_sep",
                         "csv_decimal", "csv_encoding", "task", "n_neighbors", "weights",
                         "metric", "test_size", "feature_names", "class_names", "sample"]
_TRAINING_PROBE = """\
import json, runpy, sys
import numpy as np
g = runpy.run_path(sys.argv[1], run_name="__main__")
m = g["modelo"]
out = {k: float(g[k]) for k in ("accuracy", "precision", "recall", "f1", "auc", "ap", "auc_micro",
                                "auc_macro", "ap_micro", "ap_macro", "mse", "rmse", "mae", "r2")
       if k in g}
out["params"] = m.get_params()
out["y_pred"] = np.asarray(g["y_pred"]).tolist()
if hasattr(m, "predict_proba"):
    out["proba"] = m.predict_proba(g["X_test"]).tolist()
    out["distancias_clases"] = g["distancias_clases"].tolist()
d, i = m.kneighbors(g["X_test"])
out["neighbours"] = i.tolist()
out["perm_mean"] = g["importancia_perm"].importances_mean.tolist()
out["perm_std"] = g["importancia_perm"].importances_std.tolist()
out["test_positions"] = [int(g["X"].index.get_loc(i)) for i in g["X_test"].index]
out["prediction"] = np.asarray(g["prediccion"]).item()
out["sample_neighbours"] = [int(v) for v in g["vecinos"]]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f)
"""


def _training_shared_blocks(eg) -> None:
    code = eg.REPO_ROOT / "content" / "code"
    tree = (code / "arbol_entrenamiento.py.j2").read_text("utf-8")
    knn = (code / TRAINING_TEMPLATE).read_text("utf-8")

    def block(text: str, start: str, end: str) -> str:
        i = text.index(start)
        return text[i:text.index(end, i) + len(end)]

    for start, end in (("import numpy as np\n", "make_moons\n{%- endif %}\n"),
                       ("# 1. Cargar los datos", "{%- endif %}\n{%- endif %}\n")):
        if block(tree, start, end) != block(knn, start, end):
            raise AssertionError(f"{TRAINING_TEMPLATE}: the block '{start.strip()}' differs "
                                 "from arbol_entrenamiento.py.j2")


def case_code_knn_training(eg, writer) -> None:
    import io
    import os
    import subprocess
    import sys
    import tempfile

    _training_shared_blocks(eg)
    out = {}
    for name, spec in TRAINING_CASES.items():
        fixture = json.loads((eg.PARITY_DIR / MODULE / f"{spec['fixture']}.json").read_text("utf-8"))
        inp, exp = fixture["input"], fixture["expected"]
        classification = exp["estimator"] == "KNeighborsClassifier"
        files: dict[str, bytes] = {}
        ctx = dict.fromkeys(TRAINING_CONTEXT_KEYS)
        class_names = exp["class_names"] if classification else None
        if "csv_fixture" in inp:
            cname = Path(inp["csv_fixture"]).stem
            raw = eg._csv_cases()[cname]
            files[f"{cname}.csv"] = raw
            ctx.update(dataset="csv", csv_file=f"{cname}.csv", csv_target=inp["target"],
                       csv_sep=",", csv_decimal=".", csv_encoding="utf-8")
            target = pd.read_csv(io.BytesIO(raw))[inp["target"]]
            if classification and target.dtype == object:  # D10: Rust keeps the class names
                class_names = sorted(target.astype(str).unique().tolist())
        else:
            ctx["dataset"] = Path(inp["dataset"]).stem
        sample = exp["prediction_form"]["custom"]["input_float32"] if spec["sample"] else None
        ctx.update(dataset_label=inp["app_label"],
                   task="classification" if classification else "regression",
                   n_neighbors=inp["n_neighbors"], weights=inp["weights"], metric=inp["metric"],
                   test_size=inp["test_size"], feature_names=exp["feature_names"],
                   class_names=class_names, sample=sample)
        script = eg._render_code_template(TRAINING_TEMPLATE, ctx)
        compile(script, f"{name}.py", "exec")
        with tempfile.TemporaryDirectory() as tmp:
            for fname, data in files.items():
                (Path(tmp) / fname).write_bytes(data)
            (Path(tmp) / "script.py").write_text(script, encoding="utf-8")
            (Path(tmp) / "probe.py").write_text(_TRAINING_PROBE, encoding="utf-8")
            proc = subprocess.run([sys.executable, "probe.py", "script.py", "numbers.json"],
                                  cwd=tmp, capture_output=True, text=True, timeout=600,
                                  env={**os.environ, "PYTHONWARNINGS": "ignore"})
            if proc.returncode != 0:
                raise RuntimeError(f"{name}: script failed:\n{proc.stdout}\n{proc.stderr}")
            got = json.loads((Path(tmp) / "numbers.json").read_text("utf-8"))

        def fail(what):
            return AssertionError(f"{name}: {what} differs from knn/{spec['fixture']}")

        if got["test_positions"] != exp["test_indices"]:
            raise fail("the split")
        if eg.to_jsonable(got["params"]) != eg.to_jsonable(exp["params"]):
            raise fail("get_params()")
        if got["neighbours"] != exp["neighbours_test"]["indices"]:
            raise fail("the neighbours of X_test")
        if classification:
            if got["y_pred"] != exp["y_pred"] or not np.allclose(got["proba"], exp["proba"],
                                                                  rtol=0, atol=1e-15):
                raise fail("predict/predict_proba")
            ref = eg._classification_reference(exp["evaluation"])
            dm = np.asarray(got["distancias_clases"])
            app_dm = exp["distance_analysis"].get("with_model_split_D56",
                                                  exp["distance_analysis"]["app"])["matrix"]
            if not np.allclose(dm, app_dm, rtol=1e-12, atol=1e-12):
                raise fail("the class distance matrix")
        else:
            if not np.allclose(got["y_pred"], exp["y_pred"], rtol=1e-12, atol=1e-12):
                raise fail("predict")
            ref = {k: exp["evaluation"][k] for k in ("mse", "rmse", "mae", "r2")}
        for key, val in ref.items():
            if not math.isclose(got[key], val, rel_tol=1e-12, abs_tol=1e-12):
                raise fail(f"{key} ({got[key]!r} vs {val!r})")
        perm = exp["permutation_importance"]
        if not (np.allclose(got["perm_mean"], perm["importances_mean"], rtol=0, atol=1e-12)
                and np.allclose(got["perm_std"], perm["importances_std"], rtol=0, atol=1e-12)):
            raise fail("permutation_importance")
        if spec["sample"]:
            want = exp["prediction_form"]["custom"]
            if got["sample_neighbours"] != want["neighbours"]:
                raise fail("the neighbours of the form sample")
        else:
            want = {"predict": exp["y_pred"][0]}
            if got["sample_neighbours"] != exp["neighbours_test"]["indices"][0]:
                raise fail("the neighbours of the sample")
        if not math.isclose(float(got["prediction"]), float(want["predict"]), rel_tol=1e-12,
                            abs_tol=1e-12):
            raise fail("the prediction of the sample")
        out[name] = {"context": ctx, "text": script, "stdout": proc.stdout,
                     "checked_against": f"{MODULE}/{spec['fixture']}.json",
                     "checked_numbers": {k: got[k] for k in ref},
                     "sample": {"prediction": got["prediction"],
                                "neighbours": got["sample_neighbours"]}}
    doc = {
        "meta": eg.make_meta(
            {"text": "exact", "stdout": "informational (depends on the scikit-learn version)",
             "numbers": "split, params and neighbours exact; predictions exact (classes) or "
                        "rel 1e-12; metrics, distance matrix rel 1e-12; permutation 1e-12 abs"},
            "The training script of 🔍 K-Nearest Neighbors (like D29 of F2 and the tree script "
            "of F3) is rendered with jinja2 and executed. Its split, get_params(), neighbours, "
            "predictions, probabilities, metrics, class-distance matrix (model split, D56), "
            "permutation importances and the neighbours of the sample equal the app's "
            "(knn/*.json). Rust must render 'text' exactly from 'context'. The dataset blocks "
            "are checked to be copies of arbol_entrenamiento.py.j2 (itself checked against "
            "regresion_entrenamiento.py.j2). Titanic is left out (seaborn downloads it)."),
        "input": {"template": f"content/code/{TRAINING_TEMPLATE}",
                  "engine": "jinja2 with keep_trailing_newline=True, StrictUndefined, "
                            "filters pyrepr = repr() and pycomment",
                  "context_keys": TRAINING_CONTEXT_KEYS,
                  "file_name": "knn_entrenamiento.py"},
        "expected": out,
    }
    writer.write_json(eg.PARITY_DIR / "code" / "knn_training.json", doc)


# ---------------------------------------------------------------------------
# code/knn_boundary.json: frontera_decision.py.j2 with datos="todos" (D63)
# ---------------------------------------------------------------------------
BOUNDARY_TEMPLATE = "frontera_decision.py.j2"
BOUNDARY_CASES: dict[str, str] = {
    # training-script case of code/knn_training.json -> the fixture with its 2-D models
    "iris_k5_uniform_minkowski": "iris_k5_uniform_minkowski",
    "wine_k5_uniform_minkowski": "wine_k5_uniform_minkowski",
    "breast_cancer_k5_uniform_minkowski": "breast_cancer_k5_uniform_minkowski",
    "csv_string_target_k15_distance_euclidean": "csv_string_target_k15_distance_euclidean",
}
_BOUNDARY_PROBE = """\
import json, runpy, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
g = runpy.run_path(sys.argv[1], run_name="__main__")
m2 = g["model_2d"]
Q = np.array(json.loads(sys.argv[3]), dtype=float)
out = {"n_fit": int(m2.n_samples_fit_), "n_X": int(len(g["X"])), "fit_method": m2._fit_method,
       "params": m2.get_params(), "predict": m2.predict(Q).tolist()}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f)
"""


def case_code_knn_boundary(eg, writer) -> None:
    """D63: the boundary code of 🔍 KNN fits the 2-D model on ALL the rows. Each case is the
    training script (knn_entrenamiento.py.j2) followed by frontera_decision.py.j2 with
    datos="todos", executed; its model_2d must be the app's 2-D model (same rows, engine,
    params and predictions on 200 cells of the fixture's grid)."""
    import os
    import subprocess
    import sys
    import tempfile

    training = json.loads((eg.PARITY_DIR / "code" / "knn_training.json").read_text("utf-8"))
    out = {}
    for name, fixture_name in BOUNDARY_CASES.items():
        fixture = json.loads((eg.PARITY_DIR / MODULE / f"{fixture_name}.json").read_text("utf-8"))
        exp = fixture["expected"]
        tfm = exp["two_feature_models"].get("pair") or exp["two_feature_models"]["default"]
        a, b = tfm["features"]
        classification = exp["estimator"] == "KNeighborsClassifier"
        tctx = training["expected"][name]["context"]
        ctx = {"feature_indices": [a, b],
               "feature_names": [exp["feature_names"][a], exp["feature_names"][b]],
               "class_names": tctx["class_names"] if classification else None,
               "datos": "todos"}
        text = eg._render_code_template(BOUNDARY_TEMPLATE, ctx)
        default_text = eg._render_code_template(
            BOUNDARY_TEMPLATE, {k: v for k, v in ctx.items() if k != "datos"})
        if default_text != eg._render_code_template(BOUNDARY_TEMPLATE,
                                                    {**ctx, "datos": "entrenamiento"}):
            raise AssertionError("datos='entrenamiento' must render like the default")
        script = training["expected"][name]["text"] + "\n" + text
        rng = np.random.default_rng(42)
        xs, ys = np.asarray(tfm["x"]), np.asarray(tfm["y"])
        cells = sorted(rng.choice(len(xs) * len(ys), 200, replace=False).tolist())
        Q = [[float(xs[c % len(xs)]), float(ys[c // len(xs)])] for c in cells]
        z = np.asarray(tfm["z"]).ravel()
        want = [z[c].item() for c in cells]
        files = {}
        if "csv_fixture" in fixture["input"]:
            cname = Path(fixture["input"]["csv_fixture"]).stem
            files[f"{cname}.csv"] = eg._csv_cases()[cname]
        with tempfile.TemporaryDirectory() as tmp:
            for fname, data in files.items():
                (Path(tmp) / fname).write_bytes(data)
            (Path(tmp) / "script.py").write_text(script, encoding="utf-8")
            (Path(tmp) / "probe.py").write_text(_BOUNDARY_PROBE, encoding="utf-8")
            proc = subprocess.run([sys.executable, "probe.py", "script.py", "out.json",
                                   json.dumps(Q)], cwd=tmp, capture_output=True, text=True,
                                  timeout=600, env={**os.environ, "PYTHONWARNINGS": "ignore",
                                                    "MPLBACKEND": "Agg"})
            if proc.returncode != 0:
                raise RuntimeError(f"{name}: boundary script failed:\n{proc.stdout}\n{proc.stderr}")
            got = json.loads((Path(tmp) / "out.json").read_text("utf-8"))
        if got["n_fit"] != got["n_X"] or got["n_fit"] != tfm["n_samples"]:
            raise AssertionError(f"{name}: model_2d is not fitted on all the rows")
        if got["fit_method"] != tfm["fit_method"]:
            raise AssertionError(f"{name}: engine of model_2d")
        if not np.allclose(got["predict"], want, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"{name}: model_2d predicts differently from the app's grid")
        out[name] = {"context": ctx, "text": text, "default_text": default_text,
                     "checked_against": f"{MODULE}/{fixture_name}.json two_feature_models",
                     "n_fit": got["n_fit"], "cells_checked": len(cells)}
    doc = {
        "meta": eg.make_meta(
            {"text": "exact"},
            "D63: frontera_decision.py.j2 takes an optional 'datos'. With 'todos' (🔍 KNN) the "
            "2-D model is fitted on every row of the dataset (X, y), like the app; the script "
            "is run after knn_entrenamiento.py.j2 and its model_2d equals the app's 2-D model "
            "(rows, engine and 200 grid cells). Without 'datos' (or with 'entrenamiento') the "
            "text is the one of F2/F3 ('default_text', also in code/regression.json)."),
        "input": {"template": f"content/code/{BOUNDARY_TEMPLATE}",
                  "engine": "jinja2 with keep_trailing_newline=True, StrictUndefined, "
                            "filters pyrepr = repr() and pycomment"},
        "expected": out,
    }
    writer.write_json(eg.PARITY_DIR / "code" / "knn_boundary.json", doc)


def register_code(cases: dict, eg) -> None:
    cases["code/knn_training"] = lambda w: case_code_knn_training(eg, w)
    cases["code/knn_boundary"] = lambda w: case_code_knn_boundary(eg, w)
