//! The pieces every neighbour search shares (`tools/knn_reference.py`): the reduced
//! distance, scikit-learn's fixed-size heap, its `simultaneous_sort` and the brute
//! force search.
//!
//! **Reduced distance** (`rdist`): what scikit-learn ranks with, before the final
//! square root.
//!
//! * Euclidean: `Σ (x_j − y_j)²` in feature order. The macOS arm64 wheel of
//!   scikit-learn 1.7.2 compiles this loop unrolled by 8 with a separate multiply and
//!   add, and the remaining `n_features % 8` iterations with a fused multiply-add
//!   (FMA). [`rdist`] does the same with [`f64::mul_add`], which is correctly rounded
//!   on every platform (D53), so the distances are bit for bit those of the fixtures.
//!   A plain sum differs by 1–4 ulp, enough to change the neighbours of one cell of
//!   the Iris boundary.
//! * Manhattan: `Σ |x_j − y_j|`.
//!
//! `dist = sqrt(rdist)` (Euclidean) or `rdist` (Manhattan), computed only for the
//! K neighbours that are returned.

use serde::{Deserialize, Serialize};

/// The distance actually used by the search (`effective_metric_`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectiveMetric {
    /// `minkowski` with p = 2 and `euclidean`.
    Euclidean,
    /// `manhattan` (L1).
    Manhattan,
}

impl EffectiveMetric {
    /// `effective_metric_` of scikit-learn (`"euclidean"` / `"manhattan"`).
    pub fn as_str(self) -> &'static str {
        match self {
            EffectiveMetric::Euclidean => "euclidean",
            EffectiveMetric::Manhattan => "manhattan",
        }
    }

    /// The exponent `p` of the Minkowski family (used by the kd-tree bounds).
    pub(crate) fn p_is_one(self) -> bool {
        self == EffectiveMetric::Manhattan
    }

    /// Reduced distance → distance.
    #[inline]
    pub fn to_distance(self, rdist: f64) -> f64 {
        match self {
            EffectiveMetric::Euclidean => rdist.sqrt(),
            EffectiveMetric::Manhattan => rdist,
        }
    }
}

/// Reduced distance between two rows of the same length (see the module docs).
#[inline]
pub fn rdist(x: &[f64], y: &[f64], metric: EffectiveMetric) -> f64 {
    match metric {
        EffectiveMetric::Manhattan => {
            let mut s = 0.0;
            for (a, b) in x.iter().zip(y) {
                s += (a - b).abs();
            }
            s
        }
        EffectiveMetric::Euclidean => {
            let n = x.len().min(y.len());
            let main = n - n % 8;
            let mut s = 0.0;
            for j in 0..main {
                let d = x[j] - y[j];
                s += d * d;
            }
            for j in main..n {
                let d = x[j] - y[j];
                // D53: the remainder of the wheel's loop is fused.
                s = d.mul_add(d, s);
            }
            s
        }
    }
}

/// Per-feature `(min, max)` of the row-major rows `data` (`p` columns): the
/// bounding box of the training data, for [`max_rdist_to_box`].
pub(crate) fn bounding_box(data: &[f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lo = vec![f64::INFINITY; p];
    let mut hi = vec![f64::NEG_INFINITY; p];
    if p == 0 {
        return (lo, hi);
    }
    for row in data.chunks_exact(p) {
        for j in 0..p {
            lo[j] = lo[j].min(row[j]);
            hi[j] = hi[j].max(row[j]);
        }
    }
    (lo, hi)
}

/// An upper bound of the reduced distance from `pt` to **any** point of the box
/// `[lo, hi]` (its farthest corner). If it is finite (with a factor 2 of margin for
/// the kd-tree's `d_lo + |d_lo|`), no distance of the search can overflow, so a
/// non-finite value means the search would rank with `inf` and return false
/// neighbours: the caller rejects the query with `ModelError::NonFinite`.
pub(crate) fn max_rdist_to_box(pt: &[f64], lo: &[f64], hi: &[f64], metric: EffectiveMetric) -> f64 {
    let mut s = 0.0;
    for ((&x, &a), &b) in pt.iter().zip(lo).zip(hi) {
        let d = (x - a).abs().max((x - b).abs());
        s += match metric {
            EffectiveMetric::Euclidean => d * d,
            EffectiveMetric::Manhattan => d,
        };
    }
    s
}

/// scikit-learn's `NeighborsHeap` for one query (`sklearn/utils/_heap.pyx`): a
/// max-heap of fixed size `k` initialised to +∞. A value is **rejected when it is
/// `>=` the largest** one, so among equal distances the first pushed stays.
#[derive(Debug, Clone)]
pub(crate) struct NeighborHeap {
    pub values: Vec<f64>,
    pub indices: Vec<usize>,
}

impl NeighborHeap {
    pub fn new(k: usize) -> Self {
        Self {
            values: vec![f64::INFINITY; k],
            indices: vec![0; k],
        }
    }

    /// Resets it for a new query (keeps the allocation).
    pub fn reset(&mut self) {
        self.values.fill(f64::INFINITY);
        self.indices.fill(0);
    }

    /// The largest value kept (the pruning bound).
    #[inline]
    pub fn largest(&self) -> f64 {
        self.values.first().copied().unwrap_or(f64::NEG_INFINITY)
    }

    /// `heap_push`: replaces the root and sifts the new value down with the exact
    /// child rule of scikit-learn (`>=` between siblings).
    #[inline]
    pub fn push(&mut self, val: f64, idx: usize) {
        let size = self.values.len();
        if size == 0 || val >= self.values[0] {
            return;
        }
        let v = &mut self.values;
        let ix = &mut self.indices;
        let mut cur = 0;
        loop {
            let lc = 2 * cur + 1;
            let rc = lc + 1;
            if lc >= size {
                break;
            }
            // Only a left child, or the left one is the larger (`>=`: left on ties).
            let sw = if rc >= size || v[lc] >= v[rc] {
                if val < v[lc] {
                    lc
                } else {
                    break;
                }
            } else if val < v[rc] {
                rc
            } else {
                break;
            };
            v[cur] = v[sw];
            ix[cur] = ix[sw];
            cur = sw;
        }
        v[cur] = val;
        ix[cur] = idx;
    }

    /// `_parallel_on_Y_synchronize` with one thread: the `k` entries, in heap-array
    /// order, pushed into a fresh heap. Same set, but the tied ones change places
    /// (`scratch` keeps the allocation between rows).
    pub fn repush(&mut self, scratch: &mut NeighborHeap) {
        std::mem::swap(self, scratch);
        self.reset();
        for (&v, &i) in scratch.values.iter().zip(&scratch.indices) {
            self.push(v, i);
        }
    }

    /// The `k` results in ascending order ([`simultaneous_sort`]).
    pub fn sort(&mut self) {
        simultaneous_sort(&mut self.values, &mut self.indices);
    }
}

/// `sklearn/utils/_sorting.pyx::simultaneous_sort`: a quicksort with the median of
/// `[0, n/2, n−1]` that sorts `values` ascending and moves `indices` with them. The
/// order it leaves among equal values is part of the parity (a stable sort gives
/// another order).
pub fn simultaneous_sort(values: &mut [f64], indices: &mut [usize]) {
    let size = values.len().min(indices.len());
    let swap = |v: &mut [f64], ix: &mut [usize], a: usize, b: usize| {
        v.swap(a, b);
        ix.swap(a, b);
    };
    match size {
        0 | 1 => {}
        2 => {
            if values[0] > values[1] {
                swap(values, indices, 0, 1);
            }
        }
        3 => {
            if values[0] > values[1] {
                swap(values, indices, 0, 1);
            }
            if values[1] > values[2] {
                swap(values, indices, 1, 2);
                if values[0] > values[1] {
                    swap(values, indices, 0, 1);
                }
            }
        }
        _ => {
            let piv = size / 2;
            let last = size - 1;
            if values[0] > values[last] {
                swap(values, indices, 0, last);
            }
            if values[last] > values[piv] {
                swap(values, indices, last, piv);
                if values[0] > values[last] {
                    swap(values, indices, 0, last);
                }
            }
            let pivot = values[last];
            let mut store = 0;
            for i in 0..last {
                if values[i] < pivot {
                    swap(values, indices, i, store);
                    store += 1;
                }
            }
            swap(values, indices, store, last);
            if store > 1 {
                simultaneous_sort(&mut values[..store], &mut indices[..store]);
            }
            if store + 2 < size {
                simultaneous_sort(&mut values[store + 1..size], &mut indices[store + 1..size]);
            }
        }
    }
}

/// `chunk_size` of scikit-learn's `ArgKmin` (pairwise distances reductions).
pub const ARGKMIN_CHUNK_SIZE: usize = 256;

/// The strategy scikit-learn's brute force (`ArgKmin`, one thread) picks for **one
/// call** with `n_query` rows: `parallel_on_X` if `n_train < n_query` or
/// `4 · 256 < n_query`, else `parallel_on_Y`. With `parallel_on_Y` the thread heap is
/// merged into the main one ([`NeighborHeap::repush`]) before sorting, which changes
/// the **order** of tied neighbours (not the set). So the order depends on how many
/// rows are searched together, and callers that split a query must keep the
/// strategy of the whole (`grid::decision_grid` predicts blocks of > 1024 cells).
pub fn brute_parallel_on_y(n_train: usize, n_query: usize) -> bool {
    n_train >= n_query && n_query <= 4 * ARGKMIN_CHUNK_SIZE
}

/// Brute force: every training row (row-major `data`, `p` columns) pushed in index
/// order into the heap; the heap is left unsorted.
///
/// This is **not** "the lowest index wins ties": [`NeighborHeap::push`] rejects a
/// value `>=` the root, but a smaller value evicts the root, and which of several
/// tied maxima sits at the root depends on the heap layout. So among rows tied at
/// the K-th distance the survivors are not always the lowest indices. It is what
/// scikit-learn's single-threaded brute force does (`tools/knn_reference.py`); with
/// several threads scikit-learn splits the training rows and merges heaps, so its
/// ties change with the number of cores (MEMORY.md, F4). With K = n_train every
/// row is kept and only the order *inside* runs of equal distances can differ from
/// a given scikit-learn build; that is accepted (the set and the distances agree).
/// With `parallel_on_Y` ([`brute_parallel_on_y`]) the caller then calls
/// [`NeighborHeap::repush`].
pub(crate) fn brute_query(
    data: &[f64],
    p: usize,
    pt: &[f64],
    metric: EffectiveMetric,
    heap: &mut NeighborHeap,
) {
    if p == 0 {
        return;
    }
    for (idx, row) in data.chunks_exact(p).enumerate() {
        heap.push(rdist(pt, row, metric), idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_keeps_the_first_of_equal_values() {
        let mut h = NeighborHeap::new(2);
        for (v, i) in [(1.0, 0), (2.0, 1), (2.0, 2), (0.5, 3), (1.0, 4)] {
            h.push(v, i);
        }
        h.sort();
        assert_eq!(h.values, vec![0.5, 1.0]);
        assert_eq!(h.indices, vec![3, 0]);
    }

    #[test]
    fn simultaneous_sort_sorts() {
        let mut v = vec![3.0, 1.0, 2.0, 5.0, 4.0, 0.0, 2.0, 7.0, 6.0];
        let mut ix: Vec<usize> = (0..v.len()).collect();
        simultaneous_sort(&mut v, &mut ix);
        assert_eq!(v, vec![0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(ix[0], 5);
        assert_eq!(ix[1], 1);
    }

    #[test]
    fn euclidean_tail_is_fused() {
        // 3 features: all of them in the fused remainder.
        let x = [0.1_f64, 0.2, 0.3];
        let y = [0.0, 0.0, 0.0];
        let mut s = 0.0_f64;
        for v in x {
            s = v.mul_add(v, s);
        }
        assert_eq!(
            rdist(&x, &y, EffectiveMetric::Euclidean).to_bits(),
            s.to_bits()
        );
        assert_eq!(rdist(&x, &y, EffectiveMetric::Manhattan), (0.1 + 0.2) + 0.3);
    }
}
