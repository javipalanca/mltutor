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

/// Brute force: every training row (row-major `data`, `p` columns) in index order
/// into the heap, so among equal distances the **lowest index** wins. The heap is
/// left unsorted.
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
