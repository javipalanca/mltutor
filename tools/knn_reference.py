"""Executable specification of scikit-learn 1.7.2's k-nearest neighbours (fase4-knn.md §6.1).

``KNeighborsClassifier``/``KNeighborsRegressor`` with ``algorithm='auto'`` pick one of two
engines, and each one breaks distance ties its own way. This module transcribes both so
that Rust can follow it and the golden fixtures can prove, case by case, that the rules
reproduce scikit-learn's neighbours **in order** and bit for bit:

* ``kd_tree`` (n_features <= 15 and n_neighbors < n_train // 2): ``KDTree(leaf_size=30)``.
  The tree is built by recursive median splits with ``std::nth_element`` and a comparator
  ``(value, index)``; the macOS wheel links libc++, whose algorithm is transcribed in
  :func:`libcxx_nth_element` (the *order* of the indices inside each leaf depends on it).
  The query is depth-first (closest child first, prune when ``min_rdist > largest``), the
  candidates of a leaf are pushed in ``idx_array`` order into a fixed-size max-heap that
  rejects ``dist >= largest`` (so among equal distances the FIRST visited stays), and the
  k results are ordered with sklearn's own ``simultaneous_sort`` quicksort.
* ``brute`` (otherwise): every training row in index order; among equal distances the
  lowest index stays. For the Euclidean metric scikit-learn ranks with the GEMM surrogate
  ``||y||^2 - 2 x.y`` (BLAS dependent), so near-ties may differ by rounding; Manhattan is
  exact.

Distances: the Euclidean reduced distance is ``sum((x_j - y_j)^2)`` in feature order. The
macOS arm64 wheel compiles that loop unrolled by 8 with separate multiply and add, and the
remaining ``n_features % 8`` iterations with fused multiply-add (FMA); :func:`rdist`
reproduces it exactly (``plain=True`` gives the portable sum without FMA). Manhattan is
``sum(|x_j - y_j|)``. ``dist = sqrt(rdist)`` (Euclidean) or ``rdist`` (Manhattan).
``metric='minkowski'`` (p=2, the default) IS ``'euclidean'`` in every neighbour search.

Used by ``v1.0.0:tools/golden_knn.py``; run it directly to self-check against scikit-learn::

    uv run --no-project --python 3.11 --with numpy==1.26.4 --with scikit-learn==1.7.2 \
        python tools/knn_reference.py
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

LEAF_SIZE = 30  # KNeighbors* default


# ---------------------------------------------------------------------------
# libc++ std::nth_element (include/__algorithm/nth_element.h, LLVM 15-19)
# ---------------------------------------------------------------------------
def _sort3(a: list, x: int, y: int, z: int, less: Callable) -> int:
    r = 0
    if not less(a[y], a[x]):
        if not less(a[z], a[y]):
            return r
        a[y], a[z] = a[z], a[y]
        r = 1
        if less(a[y], a[x]):
            a[x], a[y] = a[y], a[x]
            r = 2
        return r
    if less(a[z], a[y]):
        a[x], a[z] = a[z], a[x]
        return 1
    a[x], a[y] = a[y], a[x]
    r = 1
    if less(a[z], a[y]):
        a[y], a[z] = a[z], a[y]
        r = 2
    return r


def _selection_sort(a: list, first: int, last: int, less: Callable) -> None:
    lm1 = last - 1
    while first != lm1:
        m = first
        for k in range(first + 1, last):
            if less(a[k], a[m]):
                m = k
        if m != first:
            a[first], a[m] = a[m], a[first]
        first += 1


def libcxx_nth_element(a: list, first: int, nth: int, last: int, less: Callable) -> None:
    """In-place ``std::nth_element(a+first, a+nth, a+last, less)`` exactly as libc++.

    Only the branches reachable with a STRICT TOTAL order are needed (sklearn's comparator
    is ``(value, index)``, so no two elements are equivalent); the equivalence branches of
    libc++ are kept for completeness.
    """
    limit = 7
    while True:
        if nth == last:
            return
        n = last - first
        if n <= 1:
            return
        if n == 2:
            if less(a[last - 1], a[first]):
                a[first], a[last - 1] = a[last - 1], a[first]
            return
        if n == 3:
            _sort3(a, first, first + 1, last - 1, less)
            return
        if n <= limit:
            _selection_sort(a, first, last, less)
            return
        m = first + n // 2
        lm1 = last - 1
        n_swaps = _sort3(a, first, m, lm1, less)
        i, j = first, lm1
        if not less(a[i], a[m]):  # *first == *m (unreachable with a strict total order)
            found = False
            while True:
                j -= 1
                if i == j:
                    break
                if less(a[j], a[m]):
                    found = True
                    break
            if found:
                a[i], a[j] = a[j], a[i]
                n_swaps += 1
            else:
                i += 1
                j = last
                j -= 1
                if not less(a[first], a[j]):
                    while True:
                        if i == j:
                            return
                        if less(a[first], a[i]):
                            a[i], a[j] = a[j], a[i]
                            n_swaps += 1
                            i += 1
                            break
                        i += 1
                if i == j:
                    return
                while True:
                    while not less(a[first], a[i]):
                        i += 1
                    j -= 1
                    while less(a[first], a[j]):
                        j -= 1
                    if i >= j:
                        break
                    a[i], a[j] = a[j], a[i]
                    n_swaps += 1
                    i += 1
                if nth < i:
                    return
                first = i
                continue
        i += 1
        if i < j:
            while True:
                while less(a[i], a[m]):
                    i += 1
                j -= 1
                while not less(a[j], a[m]):
                    j -= 1
                if i >= j:
                    break
                a[i], a[j] = a[j], a[i]
                n_swaps += 1
                if m == i:
                    m = j
                i += 1
        if i != m and less(a[m], a[i]):
            a[i], a[m] = a[m], a[i]
            n_swaps += 1
        if nth == i:
            return
        if n_swaps == 0:
            if nth < i:
                j = m = first
                sorted_part = False
                while True:
                    j += 1
                    if j == i:
                        sorted_part = True
                        break
                    if less(a[j], a[m]):
                        break
                    m = j
                if sorted_part:
                    return
            else:
                j = m = i
                sorted_part = False
                while True:
                    j += 1
                    if j == last:
                        sorted_part = True
                        break
                    if less(a[j], a[m]):
                        break
                    m = j
                if sorted_part:
                    return
        if nth < i:
            last = i
        else:
            i += 1
            first = i


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def _fma(a: float, b: float, c: float) -> float:
    """Correctly rounded a*b + c (math.fsum of the exact product split and c)."""
    p = a * b
    # Veltkamp/Dekker exact split of the product (no overflow for the data used here).
    split = 134217729.0  # 2**27 + 1
    t = split * a
    ah = t - (t - a)
    al = a - ah
    t = split * b
    bh = t - (t - b)
    bl = b - bh
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return math.fsum((p, err, c))


def rdist(x: np.ndarray, y: np.ndarray, metric: str, plain: bool = False) -> float:
    """Reduced distance of scikit-learn's macOS arm64 wheel (see the module docstring)."""
    n = len(x)
    if metric == "manhattan":
        s = 0.0
        for j in range(n):
            s += abs(x[j] - y[j])
        return s
    main = n if plain else n - n % 8
    s = 0.0
    for j in range(n):
        d = x[j] - y[j]
        s = s + d * d if j < main else _fma(d, d, s)
    return s


def rdist_to_dist(r: float, metric: str) -> float:
    return r if metric == "manhattan" else math.sqrt(r)


# ---------------------------------------------------------------------------
# KDTree (sklearn/neighbors/_binary_tree.pxi.tp, _kd_tree.pyx.tp)
# ---------------------------------------------------------------------------
class KDTree:
    def __init__(self, data: np.ndarray, leaf_size: int = LEAF_SIZE, metric: str = "euclidean"):
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.metric = metric
        self.p = 1.0 if metric == "manhattan" else 2.0
        n, f = self.data.shape
        self.n_levels = int(np.log2(max(1.0, (n - 1) / leaf_size)) + 1)
        self.n_nodes = 2 ** self.n_levels - 1
        self.idx_array = list(range(n))
        self.idx_start = [0] * self.n_nodes
        self.idx_end = [0] * self.n_nodes
        self.is_leaf = [False] * self.n_nodes
        self.lower = np.zeros((self.n_nodes, f))
        self.upper = np.zeros((self.n_nodes, f))
        self._build(0, 0, n)

    def _build(self, i_node: int, start: int, end: int) -> None:
        d = self.data
        rows = d[self.idx_array[start:end]]
        self.lower[i_node] = rows.min(axis=0)
        self.upper[i_node] = rows.max(axis=0)
        self.idx_start[i_node], self.idx_end[i_node] = start, end
        if 2 * i_node + 1 >= self.n_nodes:
            self.is_leaf[i_node] = True
            return
        if end - start < 2:
            self.is_leaf[i_node] = True
            return
        # find_node_split_dim: first feature with the strictly largest spread (> 0)
        j_max, max_spread = 0, 0.0
        for j in range(d.shape[1]):
            spread = float(self.upper[i_node, j] - self.lower[i_node, j])
            if spread > max_spread:
                max_spread, j_max = spread, j
        n_mid = (end - start) // 2
        col = d[:, j_max]

        def less(a, b):
            va, vb = col[a], col[b]
            return a < b if va == vb else va < vb

        libcxx_nth_element(self.idx_array, start, start + n_mid, end, less)
        self._build(2 * i_node + 1, start, start + n_mid)
        self._build(2 * i_node + 2, start + n_mid, end)

    def min_rdist(self, i_node: int, pt: np.ndarray) -> float:
        r = 0.0
        for j in range(len(pt)):
            d_lo = self.lower[i_node, j] - pt[j]
            d_hi = pt[j] - self.upper[i_node, j]
            d = (d_lo + abs(d_lo)) + (d_hi + abs(d_hi))
            r += math.pow(0.5 * d, self.p)
        return r

    def query(self, pt: np.ndarray, k: int, plain: bool = False) -> tuple[list, list]:
        heap_d = [math.inf] * k
        heap_i = [0] * k

        def visit(i_node: int, lb: float) -> None:
            if lb > heap_d[0]:
                return
            if self.is_leaf[i_node]:
                for i in range(self.idx_start[i_node], self.idx_end[i_node]):
                    idx = self.idx_array[i]
                    heap_push(heap_d, heap_i, rdist(pt, self.data[idx], self.metric, plain), idx)
                return
            i1, i2 = 2 * i_node + 1, 2 * i_node + 2
            lb1, lb2 = self.min_rdist(i1, pt), self.min_rdist(i2, pt)
            if lb1 <= lb2:
                visit(i1, lb1)
                visit(i2, lb2)
            else:
                visit(i2, lb2)
                visit(i1, lb1)

        visit(0, self.min_rdist(0, pt))
        simultaneous_sort(heap_d, heap_i, 0, k)
        return [rdist_to_dist(v, self.metric) for v in heap_d], heap_i


def heap_push(values: list, indices: list, val: float, idx: int) -> None:
    """sklearn/utils/_heap.pyx: fixed-size max-heap; rejects val >= largest."""
    size = len(values)
    if val >= values[0]:
        return
    values[0], indices[0] = val, idx
    cur = 0
    while True:
        lc = 2 * cur + 1
        rc = lc + 1
        if lc >= size:
            break
        if rc >= size:
            if values[lc] > val:
                sw = lc
            else:
                break
        elif values[lc] >= values[rc]:
            if val < values[lc]:
                sw = lc
            else:
                break
        else:
            if val < values[rc]:
                sw = rc
            else:
                break
        values[cur], indices[cur] = values[sw], indices[sw]
        cur = sw
    values[cur], indices[cur] = val, idx


def simultaneous_sort(v: list, ix: list, lo: int, size: int) -> None:
    """sklearn/utils/_sorting.pyx (in place on v[lo:lo+size], ix[lo:lo+size])."""

    def sw(a, b):
        v[a], v[b] = v[b], v[a]
        ix[a], ix[b] = ix[b], ix[a]

    if size <= 1:
        return
    if size == 2:
        if v[lo] > v[lo + 1]:
            sw(lo, lo + 1)
        return
    if size == 3:
        if v[lo] > v[lo + 1]:
            sw(lo, lo + 1)
        if v[lo + 1] > v[lo + 2]:
            sw(lo + 1, lo + 2)
            if v[lo] > v[lo + 1]:
                sw(lo, lo + 1)
        return
    piv = lo + size // 2
    last = lo + size - 1
    if v[lo] > v[last]:
        sw(lo, last)
    if v[last] > v[piv]:
        sw(last, piv)
        if v[lo] > v[last]:
            sw(lo, last)
    pivot_val = v[last]
    store = lo
    for i in range(lo, last):
        if v[i] < pivot_val:
            sw(i, store)
            store += 1
    sw(store, last)
    p = store - lo
    if p > 1:
        simultaneous_sort(v, ix, lo, p)
    if p + 2 < size:
        simultaneous_sort(v, ix, store + 1, size - p - 1)


# ---------------------------------------------------------------------------
# Brute force and the engine choice
# ---------------------------------------------------------------------------
def fit_method(n_train: int, n_features: int, k: int) -> str:
    """KNeighborsMixin._fit with algorithm='auto' (minkowski p=2 / euclidean / manhattan)."""
    if n_features > 15 or k >= n_train // 2:
        return "brute"
    return "kd_tree"


def effective_metric(metric: str) -> str:
    return "manhattan" if metric == "manhattan" else "euclidean"


# ArgKmin (sklearn/metrics/_pairwise_distances_reduction/_base.pyx.tp), one thread
# (threadpool_limits(1)): chunk_size = 256, so parallel_on_X when n_train < n_query or
# 4 * 256 * 1 < n_query, else parallel_on_Y.
ARGKMIN_CHUNK_SIZE = 256


def brute_parallel_on_y(n_train: int, n_query: int, n_threads: int = 1) -> bool:
    """The strategy ArgKmin picks for ONE call of kneighbors with n_query rows."""
    if n_train < n_query:
        return False
    return not (4 * ARGKMIN_CHUNK_SIZE * n_threads < n_query)


def brute_query(train: np.ndarray, pt: np.ndarray, k: int, metric: str,
                plain: bool = False, parallel_on_y: bool = False) -> tuple[list, list]:
    """Exact distances pushed in index order into sklearn's heap (NOT "lowest index wins":
    a smaller value evicts the root, and which tied maximum is the root depends on the
    heap layout), then simultaneous_sort.

    parallel_on_y: with one thread the thread-local heap is the same, but
    _parallel_on_Y_synchronize pushes its k entries (in heap-array order) into a fresh
    main heap before the final sort, which reorders the tied neighbours (same set)."""
    heap_d = [math.inf] * k
    heap_i = [0] * k
    for idx in range(train.shape[0]):
        heap_push(heap_d, heap_i, rdist(pt, train[idx], metric, plain), idx)
    if parallel_on_y:
        main_d = [math.inf] * k
        main_i = [0] * k
        for v, i in zip(heap_d, heap_i):
            heap_push(main_d, main_i, v, i)
        heap_d, heap_i = main_d, main_i
    simultaneous_sort(heap_d, heap_i, 0, k)
    return [rdist_to_dist(v, metric) for v in heap_d], heap_i


class KNN:
    """The neighbour search of a fitted KNeighbors* model (algorithm='auto')."""

    def __init__(self, X_train: np.ndarray, k: int, metric: str):
        self.X = np.ascontiguousarray(X_train, dtype=np.float64)
        self.k = k
        self.metric = effective_metric(metric)
        self.method = fit_method(self.X.shape[0], self.X.shape[1], k)
        self.tree = KDTree(self.X, LEAF_SIZE, self.metric) if self.method == "kd_tree" else None

    def kneighbors(self, Q: np.ndarray, plain: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """sklearn's kneighbors(Q) in ONE call with threadpool_limits(1): with brute force
        the order of tied neighbours depends on len(Q) (ArgKmin strategy)."""
        D, I = [], []
        Q = np.asarray(Q, dtype=np.float64)
        on_y = brute_parallel_on_y(self.X.shape[0], len(Q))
        for pt in Q:
            if self.tree is not None:
                d, i = self.tree.query(pt, self.k, plain)
            else:
                d, i = brute_query(self.X, pt, self.k, self.metric, plain, on_y)
            D.append(d)
            I.append(i)
        return np.asarray(D), np.asarray(I, dtype=np.int64)


# ---------------------------------------------------------------------------
# Prediction rules (sklearn/neighbors/_classification.py, _regression.py, _base.py)
# ---------------------------------------------------------------------------
def get_weights(dist: np.ndarray, weights: str) -> np.ndarray | None:
    if weights == "uniform":
        return None
    with np.errstate(divide="ignore"):
        w = 1.0 / dist
    inf_mask = np.isinf(w)
    inf_row = np.any(inf_mask, axis=1)
    w[inf_row] = inf_mask[inf_row]
    return w


def predict_proba(dist, ind, y_codes, n_classes, weights) -> np.ndarray:
    """Votes added in neighbour order, then divided by their (numpy) sum."""
    w = get_weights(dist, weights)
    if w is None:
        w = np.ones_like(ind, dtype=float)
    labels = np.asarray(y_codes)[ind]
    proba = np.zeros((ind.shape[0], n_classes))
    rows = np.arange(ind.shape[0])
    for c in range(ind.shape[1]):
        proba[rows, labels[:, c]] += w[:, c]
    return proba / proba.sum(axis=1)[:, None]


def predict_class(dist, ind, y_codes, n_classes, weights) -> np.ndarray:
    """uniform: mode (smallest class among the most frequent); distance: weighted_mode
    (class order, strict '>': the smallest class wins exact ties)."""
    labels = np.asarray(y_codes)[ind]
    w = get_weights(dist, weights)
    if w is None:
        w = np.ones_like(ind, dtype=float)
    best = np.zeros(ind.shape[0], dtype=np.int64)
    best_score = np.zeros(ind.shape[0])
    for c in np.unique(labels):
        s = np.where(labels == c, w, 0.0).sum(axis=1)
        better = s > best_score
        best = np.where(better, c, best)
        best_score = np.maximum(s, best_score)
    return best


def predict_value(dist, ind, y, weights) -> np.ndarray:
    yv = np.asarray(y, dtype=float)[ind]
    w = get_weights(dist, weights)
    if w is None:
        return np.array([np.mean(r) for r in yv])
    return np.array([np.average(r, weights=wr) for r, wr in zip(yv, w)])


# ---------------------------------------------------------------------------
# Self-check against scikit-learn
# ---------------------------------------------------------------------------
def compare_with_sklearn(model, X_train: np.ndarray, Q: np.ndarray) -> dict:
    """Rebuild the search of a fitted KNeighbors* model and compare it on the queries Q."""
    k = model.n_neighbors
    ref = KNN(X_train, k, model.metric)
    out: dict = {"fit_method_sklearn": model._fit_method, "fit_method_reference": ref.method,
                 "n_queries": int(len(Q))}
    if ref.tree is not None:
        _, idx_array, _, _ = model._tree.get_arrays()
        out["idx_array_equal"] = bool(np.array_equal(np.asarray(idx_array), ref.tree.idx_array))
    d_sk, i_sk = model.kneighbors(Q)
    d_ref, i_ref = ref.kneighbors(Q)
    out["indices_equal_in_order"] = bool(np.array_equal(i_sk, i_ref))
    out["rows_with_different_indices"] = int(np.sum(np.any(i_sk != i_ref, axis=1)))
    out["rows_with_different_neighbour_sets"] = int(sum(
        set(a) != set(b) for a, b in zip(i_sk.tolist(), i_ref.tolist())))
    out["distances_bit_equal"] = bool(np.array_equal(d_sk, d_ref))
    out["distances_max_abs_diff"] = float(np.max(np.abs(d_sk - d_ref))) if d_sk.size else 0.0
    return out


def _self_check() -> None:  # pragma: no cover - manual tool
    import warnings

    from sklearn.datasets import load_diabetes, load_iris, load_wine
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    for name, loader, cls in (("iris", load_iris, KNeighborsClassifier),
                              ("wine", load_wine, KNeighborsClassifier),
                              ("diabetes", load_diabetes, KNeighborsRegressor)):
        X, y = loader(return_X_y=True)
        for k, metric in ((5, "minkowski"), (1, "manhattan"), (20, "euclidean")):
            m = cls(n_neighbors=k, metric=metric).fit(X, y)
            Q = X[rng.choice(len(X), 40)] + rng.normal(0, 0.05, (40, X.shape[1])) * X.std(0)
            Q = np.vstack([Q, X[:10]])
            print(name, k, metric, compare_with_sklearn(m, X, Q))


if __name__ == "__main__":
    _self_check()
