"""Check the tree fixtures against the scikit-learn of THIS machine (phase 3).

The fixtures in ``tests/parity/trees`` were generated on macOS arm64, whose
scikit-learn wheel contracts ``a*b + c`` into FMA. Most of the time that only moves
impurities by a few ulps, but in an exact tie (mirror partitions) the FMA can break
the tie and change the tree (MEMORY.md, 2026-10-15). The Rust CART has no FMA and
matches the fixtures bit for bit, so this script asks the question that matters on
x86-64 (no FMA in the manylinux wheels): does scikit-learn grow the SAME trees here?

For every trained tree of the fixtures (the lesson's model, the 2-feature models of
the boundary/surface and ``tiny_scale_features``) it refits sklearn on the stored
training rows and compares with ``expected.tree``:

* structure: children, feature, n_node_samples, value counts / values — exact;
* thresholds — bit for bit;
* impurities — relative 1e-12 (reported; ulps of FMA are expected).

It also runs ``tools/cart_reference.py`` (pure Python, no FMA) on the same data and
checks it against this sklearn. Exit 1 if any structure or threshold differs.

Only numpy, pandas and scikit-learn are needed (no app, no network):

    uv run --no-project --python 3.11 --with numpy==1.26.4 --with pandas==2.3.3 \\
        --with scikit-learn==1.7.2 python tools/check_trees_platform.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent
PARITY = ROOT / "tests" / "parity"
sys.path.insert(0, str(ROOT / "tools"))

import cart_reference as cr  # noqa: E402

ESTIMATORS = {"DecisionTreeClassifier": DecisionTreeClassifier,
              "DecisionTreeRegressor": DecisionTreeRegressor}
FIT_PARAMS = ("criterion", "max_depth", "min_samples_split", "random_state")


def _dataset(rel: str) -> tuple[pd.DataFrame, pd.Series]:
    meta = json.loads((PARITY / rel).read_text(encoding="utf-8"))["expected"]
    df = pd.read_csv(PARITY / "datasets" / meta["csv"], float_precision="round_trip")
    X = df[meta["columns"]].astype(np.float64)
    y = df[meta["target_name"]]
    rows = np.column_stack([X.to_numpy(np.float64), y.to_numpy(np.float64)])
    digest = hashlib.sha256(np.ascontiguousarray(rows, dtype="<f8").tobytes()).hexdigest()
    if digest != meta["data_sha256_f64"]:
        raise SystemExit(f"{rel}: the CSV does not read back exactly (sha256 differs)")
    return X, y


def _csv(rel: str, task: str) -> tuple[pd.DataFrame, pd.Series]:
    fx = json.loads((PARITY / rel).read_text(encoding="utf-8"))
    key = {"auto": "auto", "classification": "Clasificación", "regression": "Regresión"}
    loaded = fx["expected"]["load_dataset_from_file"][key.get(task, task)]
    p = len(loaded["feature_names"])
    X = pd.DataFrame(np.asarray(loaded["X"], dtype=np.float64).reshape(-1, p))  # row-major
    return X, pd.Series(loaded["y"])


def _compare(label: str, model, tree: dict, report: dict) -> None:
    t = model.tree_
    problems = []
    for key in ("children_left", "children_right", "feature", "n_node_samples"):
        if list(map(int, getattr(t, key))) != list(map(int, tree[key])):
            problems.append(key)
    a = np.asarray(t.threshold, dtype=np.float64)
    b = np.asarray(tree["threshold"], dtype=np.float64)
    if a.shape != b.shape or not np.array_equal(a.view(np.uint64), b.view(np.uint64)):
        problems.append("threshold")
    value = np.asarray(t.value, dtype=np.float64)[:, 0, :]
    ev = np.asarray(tree["value"], dtype=np.float64)
    if hasattr(model, "classes_"):
        counts = np.rint(value * t.weighted_n_node_samples[:, None]).astype(int)
        if "value_counts" in tree and (counts.shape != np.shape(tree["value_counts"]) or
                                       not np.array_equal(counts, tree["value_counts"])):
            problems.append("value_counts")
    elif value.shape != ev.shape or np.max(np.abs(value - ev) / np.maximum(1, np.abs(ev))) > 1e-12:
        problems.append("value")
    if not problems:
        ia, ib = np.asarray(t.impurity), np.asarray(tree["impurity"])
        err = float(np.max(np.abs(ia - ib) / np.maximum(1.0, np.abs(ib))))
        report["impurity_max_rel"] = max(report["impurity_max_rel"], err)
        report["impurity_bit_exact"] += int(np.array_equal(ia, ib))
        if err > 1e-12:
            problems.append(f"impurity ({err:.1e})")
    report["trees"] += 1
    if problems:
        report["failures"].append(f"{label}: {', '.join(problems)}")


def _reference(label: str, X, y, model, report: dict) -> None:
    p = model.get_params()
    ref = cr.fit(X, y, classification=hasattr(model, "classes_"), criterion=p["criterion"],
                 max_depth=p["max_depth"], min_samples_split=p["min_samples_split"],
                 random_state=p["random_state"])
    cmp = cr.compare_with_sklearn(ref, model.tree_)
    report["reference_bit_exact"] += int(cmp["bit_exact"])
    if not cmp["structure_exact"]:
        report["failures"].append(f"{label}: cart_reference.py differs from this sklearn")


def _fit(estimator: str, params: dict, X: np.ndarray, y) -> object:
    cls = ESTIMATORS[estimator]
    return cls(**{k: params[k] for k in FIT_PARAMS}).fit(np.asarray(X, dtype=np.float32), y)


def main() -> int:
    # Diabetes trained as a classifier (192 classes) warns; that is the app's case.
    warnings.filterwarnings("ignore", message="The number of unique classes")
    report = {"trees": 0, "impurity_max_rel": 0.0, "impurity_bit_exact": 0,
              "reference_bit_exact": 0, "failures": []}
    for path in sorted((PARITY / "trees").glob("*.json")):
        fx = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem
        if "cases" in fx:  # tiny_scale_features: X stored as float32 values
            for case, c in fx["cases"].items():
                inp = c["input"]
                X = np.asarray(inp["X"], dtype=np.float32)
                y = np.asarray(inp["y"])
                est = ("DecisionTreeClassifier" if inp["tree_type"] == "Clasificación"
                       else "DecisionTreeRegressor")
                model = _fit(est, inp, X, y)
                _compare(f"{name}/{case}", model, c["expected"]["tree"], report)
                _reference(f"{name}/{case}", X, y, model, report)
            continue
        e = fx["expected"]
        if not e.get("trained") or "tree" not in e:
            continue
        inp = fx["input"]
        if "dataset" in inp:
            X, y = _dataset(inp["dataset"])
        else:
            X, y = _csv(inp["csv_fixture"], inp.get("task", "auto"))
        Xt = X.to_numpy(np.float64)[e["train_indices"]]
        yt = y.to_numpy()[e["train_indices"]]
        model = _fit(e["estimator"], e["params"], Xt, yt)
        _compare(name, model, e["tree"], report)
        _reference(name, Xt, yt, model, report)
        for key, m in (e.get("two_feature_models") or {}).items():
            X2 = Xt[:, m["features"]]
            m2 = _fit(e["estimator"], e["params"], X2, yt)
            _compare(f"{name}/{key}", m2, m["tree"], report)
            _reference(f"{name}/{key}", X2, yt, m2, report)
    print(f"platform {platform.platform()} ({platform.machine()}), scikit-learn "
          f"{sklearn.__version__}, numpy {np.__version__}")
    print(f"trees compared: {report['trees']}; impurities bit-exact: "
          f"{report['impurity_bit_exact']}; impurity max rel err "
          f"{report['impurity_max_rel']:.2e} (tol 1e-12); cart_reference bit-exact with "
          f"this sklearn: {report['reference_bit_exact']}/{report['trees']}")
    for f in report["failures"]:
        print(f"DIFFERS  {f}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
