"""Line-by-line Python transcription of scikit-learn 1.7.2's dense CART (phase 3).

Reference specification for the Rust port (``mltutor-core``, fase3-arboles.md §8) and
the tool that documents how scikit-learn breaks ties. It reproduces, for the case the
app uses (dense ``X`` without NaN, no sample weights, ``splitter="best"``,
``max_features=None``, ``min_samples_leaf=1``, ``min_impurity_decrease=0``, no
monotonic constraints):

* ``DepthFirstTreeBuilder.build`` (``sklearn/tree/_tree.pyx``): stack, right child
  pushed before the left one (so the left subtree is built first), leaf rules.
* ``node_split_best`` (``_splitter.pyx``): the Fisher-Yates feature draw with the
  splitter's xorshift RNG (``our_rand_r``, seeded with
  ``RandomState(random_state).randint(0, RAND_R_MAX)``), the ``features`` and
  ``constant_features`` arrays that persist across nodes, constant features
  (``max <= min``: see ``FEATURE_THRESHOLD``), candidate positions (``next_p``), the
  strict ``>`` comparison of the proxy improvement (the first feature drawn wins a
  tie) and the threshold ``x[p-1]/2 + x[p]/2`` computed in float64 from float32
  values.
* ``DensePartitioner`` (``_partitioner.pyx``): float32 copy of the column,
  introsort with median of 3 and 3-way partition, heapsort fallback, and
  ``partition_samples_final`` (which partitions the order left by the **last**
  feature evaluated, not the best one).
* Criteria (``_criterion.pyx``): Gini, Entropy, MSE (``squared_error``),
  FriedmanMSE and MAE (``absolute_error``), with the same incremental ``update``
  (forwards or, when shorter, ``reverse_reset`` and backwards) and the same
  floating-point formulas, so sums are accumulated in the same order.

``fit`` returns the tree arrays (as ``sklearn.tree._tree.Tree``) plus a per-node
``trace``: features in the order they were drawn, best proxy improvement and
threshold of each, and the exact ties with the chosen split. ``tools/export_golden.py``
checks that the arrays equal scikit-learn's bit for bit and stores the trace in the
fixtures (``trees/*.json`` → ``expected.reference_impl``).

Not covered (the app never uses them): sample weights, NaN, sparse X, random
splitter, ``max_features``, ``min_samples_leaf > 1``, ``min_impurity_decrease``,
monotonic constraints, multi-output, Poisson.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

RAND_R_MAX = 2147483647  # sklearn/utils/_random.pxd
# ``_partitioner.pxd`` declares ``cdef float32_t FEATURE_THRESHOLD = 1e-7``, but in the
# compiled 1.7.2 wheels the splitter behaves as if it were 0 (presumably because the
# initialiser of a module-level ``cdef`` in a .pxd does not run in the modules that
# cimport it). sklearn splits even two contiguous float32 values (fixture
# ``trees/tiny_scale_features``), so the comparisons are ``x[p+1] <= x[p]`` and a
# feature is constant iff ``max <= min``.
FEATURE_THRESHOLD = np.float32(0.0)
EPSILON = float(np.finfo("double").eps)  # sklearn/tree/_tree.pyx
TREE_LEAF = -1
TREE_UNDEFINED = -2
LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------
class RandR:
    """``our_rand_r`` (32-bit xorshift) and ``rand_int`` of sklearn/tree/_utils.pyx."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next(self) -> int:
        s = self.state
        if s == 0:
            s = 1  # DEFAULT_SEED
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.state = s
        return s % (RAND_R_MAX + 1)

    def rand_int(self, low: int, high: int) -> int:
        return low + self.next() % (high - low)


def splitter_seed(random_state: int = 42) -> int:
    """``Splitter.init``: ``self.random_state.randint(0, RAND_R_MAX)``."""
    return int(np.random.RandomState(random_state).randint(0, RAND_R_MAX))


# ---------------------------------------------------------------------------
# Sorting (introsort of _partitioner.pyx), on parallel lists from offset `lo`
# ---------------------------------------------------------------------------
def _swap(v, s, i, j):
    v[i], v[j] = v[j], v[i]
    s[i], s[j] = s[j], s[i]


def _median3(v, lo, n):
    a, b, c = v[lo], v[lo + n // 2], v[lo + n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        return a
    elif b < c:
        if a < c:
            return a
        return c
    return b


def _sift_down(v, s, lo, start, end):
    root = start
    while True:
        child = root * 2 + 1
        maxind = root
        if child < end and v[lo + maxind] < v[lo + child]:
            maxind = child
        if child + 1 < end and v[lo + maxind] < v[lo + child + 1]:
            maxind = child + 1
        if maxind == root:
            break
        _swap(v, s, lo + root, lo + maxind)
        root = maxind


def _heapsort(v, s, lo, n):
    start = (n - 2) // 2
    end = n
    while True:
        _sift_down(v, s, lo, start, end)
        if start == 0:
            break
        start -= 1
    end = n - 1
    while end > 0:
        _swap(v, s, lo, lo + end)
        _sift_down(v, s, lo, 0, end)
        end -= 1


def _introsort(v, s, lo, n, maxd):
    while n > 1:
        if maxd <= 0:
            _heapsort(v, s, lo, n)
            return
        maxd -= 1
        pivot = _median3(v, lo, n)
        i = l = 0
        r = n
        while i < r:
            x = v[lo + i]
            if x < pivot:
                _swap(v, s, lo + i, lo + l)
                i += 1
                l += 1
            elif x > pivot:
                r -= 1
                _swap(v, s, lo + i, lo + r)
            else:
                i += 1
        _introsort(v, s, lo, l, maxd)
        lo += r
        n -= r


def sort(v, s, lo, n):
    if n == 0:
        return
    maxd = 2 * int(math.log2(n))
    _introsort(v, s, lo, n, maxd)


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
class _Classification:
    def __init__(self, y_codes, n_classes, entropy):
        self.y = y_codes
        self.k = n_classes
        self.entropy = entropy

    def init(self, samples, start, end):
        self.samples, self.start, self.end = samples, start, end
        tot = [0.0] * self.k
        w = 0.0
        for p in range(start, end):
            tot[self.y[samples[p]]] += 1.0
            w += 1.0
        self.sum_total, self.w_node = tot, w
        self.reset()

    def reset(self):
        self.pos = self.start
        self.sum_left = [0.0] * self.k
        self.sum_right = list(self.sum_total)
        self.w_left, self.w_right = 0.0, self.w_node

    def reverse_reset(self):
        self.pos = self.end
        self.sum_right = [0.0] * self.k
        self.sum_left = list(self.sum_total)
        self.w_right, self.w_left = 0.0, self.w_node

    def update(self, new_pos):
        pos, end, s = self.pos, self.end, self.samples
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                self.sum_left[self.y[s[p]]] += 1.0
                self.w_left += 1.0
        else:
            self.reverse_reset()
            for p in range(end - 1, new_pos - 1, -1):
                self.sum_left[self.y[s[p]]] -= 1.0
                self.w_left -= 1.0
        self.w_right = self.w_node - self.w_left
        self.sum_right = [t - l for t, l in zip(self.sum_total, self.sum_left)]
        self.pos = new_pos

    def _imp(self, counts, w):
        if self.entropy:
            e = 0.0
            for c in counts:
                if c > 0.0:
                    c = c / w
                    e -= c * (math.log(c) / LN2)
            return e
        sq = 0.0
        for c in counts:
            sq += c * c
        return 1.0 - sq / (w * w)

    def node_impurity(self):
        return self._imp(self.sum_total, self.w_node)

    def children_impurity(self):
        return self._imp(self.sum_left, self.w_left), self._imp(self.sum_right, self.w_right)

    def proxy(self):
        il, ir = self.children_impurity()
        return -self.w_right * ir - self.w_left * il

    def improvement(self, w_samples, parent, il, ir):
        return ((self.w_node / w_samples) *
                (parent - (self.w_right / self.w_node * ir) - (self.w_left / self.w_node * il)))

    def node_value(self):
        return [c / self.w_node for c in self.sum_total]


class _Regression:
    def __init__(self, y, kind):
        self.y = y
        self.kind = kind  # "squared_error" | "friedman_mse" | "absolute_error"

    def init(self, samples, start, end):
        self.samples, self.start, self.end = samples, start, end
        tot = sq = w = 0.0
        for p in range(start, end):
            yi = self.y[samples[p]]
            wy = 1.0 * yi
            tot += wy
            sq += wy * yi
            w += 1.0
        self.sum_total, self.sq_sum_total, self.w_node = tot, sq, w
        if self.kind == "absolute_error":
            self.node_median = _median([self.y[samples[p]] for p in range(start, end)])
        self.reset()

    def reset(self):
        self.pos = self.start
        self.sum_left, self.sum_right = 0.0, self.sum_total
        self.w_left, self.w_right = 0.0, self.w_node

    def reverse_reset(self):
        self.pos = self.end
        self.sum_right, self.sum_left = 0.0, self.sum_total
        self.w_right, self.w_left = 0.0, self.w_node

    def update(self, new_pos):
        pos, end, s = self.pos, self.end, self.samples
        if self.kind == "absolute_error":
            # MAE.update keeps median heaps; only the weights matter here and the
            # medians are recomputed exactly (they are order statistics).
            self.w_left = float(new_pos - self.start)
            self.w_right = self.w_node - self.w_left
            self.pos = new_pos
            return
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                self.sum_left += 1.0 * self.y[s[p]]
                self.w_left += 1.0
        else:
            self.reverse_reset()
            for p in range(end - 1, new_pos - 1, -1):
                self.sum_left -= 1.0 * self.y[s[p]]
                self.w_left -= 1.0
        self.w_right = self.w_node - self.w_left
        self.sum_right = self.sum_total - self.sum_left
        self.pos = new_pos

    def node_impurity(self):
        if self.kind == "absolute_error":
            imp = 0.0
            for p in range(self.start, self.end):
                imp += abs(self.y[self.samples[p]] - self.node_median) * 1.0
            return imp / (self.w_node * 1)
        imp = self.sq_sum_total / self.w_node
        imp -= (self.sum_total / self.w_node) ** 2.0
        return imp / 1

    def children_impurity(self):
        s, start, pos, end = self.samples, self.start, self.pos, self.end
        if self.kind == "absolute_error":
            ml = _median([self.y[s[p]] for p in range(start, pos)])
            il = 0.0
            for p in range(start, pos):
                il += abs(self.y[s[p]] - ml) * 1.0
            mr = _median([self.y[s[p]] for p in range(pos, end)])
            ir = 0.0
            for p in range(pos, end):
                ir += abs(self.y[s[p]] - mr) * 1.0
            return il / (self.w_left * 1), ir / (self.w_right * 1)
        sq_left = 0.0
        for p in range(start, pos):
            yi = self.y[s[p]]
            sq_left += 1.0 * yi * yi
        sq_right = self.sq_sum_total - sq_left
        il = sq_left / self.w_left
        ir = sq_right / self.w_right
        il -= (self.sum_left / self.w_left) ** 2.0
        ir -= (self.sum_right / self.w_right) ** 2.0
        return il / 1, ir / 1

    def proxy(self):
        if self.kind == "squared_error":
            return (self.sum_left * self.sum_left) / self.w_left + \
                (self.sum_right * self.sum_right) / self.w_right
        if self.kind == "friedman_mse":
            diff = self.w_right * self.sum_left - self.w_left * self.sum_right
            return diff * diff / (self.w_left * self.w_right)
        il, ir = self.children_impurity()
        return -self.w_right * ir - self.w_left * il

    def improvement(self, w_samples, parent, il, ir):
        if self.kind == "friedman_mse":
            diff = (self.w_right * self.sum_left - self.w_left * self.sum_right) / 1
            return diff * diff / (self.w_left * self.w_right * self.w_node)
        return ((self.w_node / w_samples) *
                (parent - (self.w_right / self.w_node * ir) - (self.w_left / self.w_node * il)))

    def node_value(self):
        if self.kind == "absolute_error":
            return [self.node_median]
        return [self.sum_total / self.w_node]


def _median(values):
    """WeightedMedianCalculator.get_median with unit weights."""
    v = sorted(values)
    n = len(v)
    if n % 2 == 0:
        return (v[n // 2 - 1] + v[n // 2]) / 2.0
    return v[n // 2]


# ---------------------------------------------------------------------------
# Builder + splitter
# ---------------------------------------------------------------------------
def fit(X, y, *, classification: bool, criterion: str, max_depth: int,
        min_samples_split: int, random_state: int = 42) -> dict[str, Any]:
    """Grow the tree like ``DecisionTree{Classifier,Regressor}(...).fit(X, y)``.

    ``X`` is converted to float32 (``check_array(dtype=DTYPE)``); ``y`` are the
    original labels (classification, encoded with ``np.unique``) or values.
    """
    X32 = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    Xv = X32.astype(np.float64).tolist()  # exact float32 values as Python floats
    n, n_features = X32.shape
    if classification:
        classes, y_enc = np.unique(np.asarray(y), return_inverse=True)
        crit = _Classification([int(c) for c in y_enc], len(classes), criterion == "entropy")
    else:
        classes = None
        crit = _Regression([float(v) for v in np.asarray(y, dtype=np.float64)], criterion)

    rng = RandR(splitter_seed(random_state))
    samples = list(range(n))
    fvals = [0.0] * n
    features = list(range(n_features))
    constant_features = [0] * n_features
    w_samples = float(n)

    nodes: list[dict] = []
    values: list[list[float]] = []
    trace: list[dict] = []
    stack = [dict(start=0, end=n, depth=0, parent=TREE_UNDEFINED, is_left=False,
                  impurity=math.inf, n_constant=0)]
    first = True
    while stack:
        rec = stack.pop()
        start, end, depth = rec["start"], rec["end"], rec["depth"]
        n_node = end - start
        crit.init(samples, start, end)
        is_leaf = depth >= max_depth or n_node < min_samples_split or n_node < 2
        impurity = rec["impurity"]
        if first:
            impurity = crit.node_impurity()
            first = False
        is_leaf = is_leaf or impurity <= EPSILON
        split = None
        n_constant = rec["n_constant"]
        node_trace = None
        if not is_leaf:
            split, n_constant, node_trace = _node_split_best(
                crit, rng, samples, fvals, Xv, features, constant_features,
                start, end, n_constant, impurity, w_samples)
            is_leaf = split["pos"] >= end or (split["improvement"] + EPSILON < 0.0)
        node_id = len(nodes)
        nodes.append(dict(left=TREE_LEAF, right=TREE_LEAF,
                          feature=TREE_UNDEFINED if is_leaf else split["feature"],
                          threshold=float(TREE_UNDEFINED) if is_leaf else split["threshold"],
                          impurity=impurity, n_node_samples=n_node,
                          weighted_n_node_samples=float(n_node),
                          missing_go_to_left=0 if is_leaf else int(split["missing_go_to_left"])))
        if rec["parent"] != TREE_UNDEFINED:
            nodes[rec["parent"]]["left" if rec["is_left"] else "right"] = node_id
        values.append(crit.node_value())
        if node_trace is not None:
            node_trace["node"] = node_id
            trace.append(node_trace)
        if not is_leaf:
            stack.append(dict(start=split["pos"], end=end, depth=depth + 1, parent=node_id,
                              is_left=False, impurity=split["impurity_right"],
                              n_constant=n_constant))
            stack.append(dict(start=start, end=split["pos"], depth=depth + 1, parent=node_id,
                              is_left=True, impurity=split["impurity_left"],
                              n_constant=n_constant))
    return {
        "classes": classes,
        "children_left": [d["left"] for d in nodes],
        "children_right": [d["right"] for d in nodes],
        "feature": [d["feature"] for d in nodes],
        "threshold": [d["threshold"] for d in nodes],
        "impurity": [d["impurity"] for d in nodes],
        "n_node_samples": [d["n_node_samples"] for d in nodes],
        "weighted_n_node_samples": [d["weighted_n_node_samples"] for d in nodes],
        "missing_go_to_left": [d["missing_go_to_left"] for d in nodes],
        "value": values,
        "trace": trace,
        "splitter_seed": splitter_seed(random_state),
    }


def _node_split_best(crit, rng, samples, fvals, Xv, features, constant_features,
                     start, end, n_known_constants, impurity, w_samples):
    n_features = len(features)
    max_features = n_features
    f_i = n_features
    n_visited = n_found_constants = n_drawn_constants = 0
    n_total_constants = n_known_constants
    best = dict(pos=end, feature=0, threshold=0.0, improvement=-math.inf,
                impurity_left=math.inf, impurity_right=math.inf, missing_go_to_left=False)
    best_proxy = -math.inf
    current_feature = None
    visited: list[dict] = []
    while (f_i > n_total_constants and
           (n_visited < max_features or n_visited <= n_found_constants + n_drawn_constants)):
        n_visited += 1
        f_j = rng.rand_int(n_drawn_constants, f_i - n_found_constants)
        if f_j < n_known_constants:
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
            n_drawn_constants += 1
            continue
        f_j += n_found_constants
        current_feature = features[f_j]
        for p in range(start, end):
            fvals[p] = Xv[samples[p]][current_feature]
        sort(fvals, samples, start, end - start)
        if fvals[end - 1] <= fvals[start] + FEATURE_THRESHOLD:  # max <= min
            features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
            n_found_constants += 1
            n_total_constants += 1
            visited.append({"feature": current_feature, "constant": True})
            continue
        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        crit.reset()
        feat_best_proxy, feat_best_thr, feat_best_pos = -math.inf, None, None
        p = start
        while p < end:
            while p + 1 < end and fvals[p + 1] <= fvals[p] + FEATURE_THRESHOLD:
                p += 1
            p_prev = p
            p += 1
            if p >= end:
                continue
            n_left, n_right = p - start, end - p
            if n_left < 1 or n_right < 1:
                continue
            crit.update(p)
            proxy = crit.proxy()
            thr = fvals[p_prev] / 2.0 + fvals[p] / 2.0
            if thr == fvals[p] or math.isinf(thr):
                thr = fvals[p_prev]
            if proxy > feat_best_proxy:
                feat_best_proxy, feat_best_thr, feat_best_pos = proxy, thr, p
            if proxy > best_proxy:
                best_proxy = proxy
                best.update(pos=p, feature=current_feature, threshold=thr,
                            missing_go_to_left=n_left > n_right)
        visited.append({"feature": current_feature, "constant": False,
                        "best_proxy": feat_best_proxy, "best_threshold": feat_best_thr,
                        "best_n_left": None if feat_best_pos is None else feat_best_pos - start})
    if best["pos"] < end:
        # partition_samples_final on the order left by the last evaluated feature
        thr, feat = best["threshold"], best["feature"]
        p, partition_end = start, end - 1
        while p < partition_end:
            if Xv[samples[p]][feat] <= thr:
                p += 1
            else:
                samples[p], samples[partition_end] = samples[partition_end], samples[p]
                partition_end -= 1
        crit.reset()
        crit.update(best["pos"])
        il, ir = crit.children_impurity()
        best["impurity_left"], best["impurity_right"] = il, ir
        best["improvement"] = crit.improvement(w_samples, impurity, il, ir)
    features[:n_known_constants] = constant_features[:n_known_constants]
    constant_features[n_known_constants:n_known_constants + n_found_constants] = \
        features[n_known_constants:n_known_constants + n_found_constants]
    ties = [v["feature"] for v in visited if not v["constant"] and v["feature"] != best["feature"]
            and v["best_proxy"] == best_proxy]
    near = []
    for v in visited:
        if v["constant"] or v["feature"] == best["feature"] or v["best_proxy"] == best_proxy:
            continue
        scale = max(abs(best_proxy), 1e-300)
        rel = (best_proxy - v["best_proxy"]) / scale
        if rel < 1e-9:
            near.append({"feature": v["feature"], "relative_gap": rel})
    node_trace = {"visit_order": [v["feature"] for v in visited],
                  "visited": visited,
                  "chosen_feature": best["feature"] if best["pos"] < end else None,
                  "chosen_proxy": best_proxy if best["pos"] < end else None,
                  "exact_ties_lost": ties,
                  "near_ties_lost_rel_1e-9": near}
    return best, n_total_constants, node_trace


def compare_with_sklearn(ref: dict, tree) -> dict[str, Any]:
    """Bit-for-bit comparison with a fitted ``sklearn.tree._tree.Tree``."""
    out: dict[str, Any] = {}
    exact = True
    for key in ("children_left", "children_right", "feature", "n_node_samples"):
        same = list(map(int, getattr(tree, key))) == list(map(int, ref[key]))
        out[key] = same
        exact &= same
    for key in ("threshold", "impurity", "weighted_n_node_samples"):
        a = np.asarray(getattr(tree, key), dtype=float)
        b = np.asarray(ref[key], dtype=float)
        same = a.shape == b.shape and bool(np.array_equal(a, b))
        out[key] = same if same else (float(np.max(np.abs(a - b))) if a.shape == b.shape else "shape")
        exact &= same
    val = np.asarray(tree.value, dtype=float)[:, 0, :]
    rv = np.asarray(ref["value"], dtype=float)
    same = val.shape == rv.shape and bool(np.array_equal(val, rv))
    out["value"] = same if same else (float(np.max(np.abs(val - rv))) if val.shape == rv.shape else "shape")
    exact &= same
    out["bit_exact"] = bool(exact)
    out["structure_exact"] = all(out[k] is True for k in (
        "children_left", "children_right", "feature", "n_node_samples", "threshold",
        "weighted_n_node_samples", "value"))
    return out
