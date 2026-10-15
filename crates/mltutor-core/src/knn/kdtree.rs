//! scikit-learn 1.7.2's `KDTree` (`_binary_tree.pxi.tp`, `_kd_tree.pyx.tp`) as
//! `KNeighbors*` builds it with `algorithm='auto'` (`leaf_size = 30`), transcribed from
//! `tools/knn_reference.py`.
//!
//! **Construction.** `n_levels = int(log2(max(1, (n − 1)/leaf_size)) + 1)` and
//! `n_nodes = 2^n_levels − 1`, stored as an implicit binary tree (children of `i` are
//! `2i + 1` and `2i + 2`). Node `i` covers `idx_array[start..end]`; its bounds are the
//! per-feature min and max of its points. It is a leaf when `2i + 1 >= n_nodes` (or it
//! has fewer than 2 points); otherwise it is split on the **first** feature with the
//! largest spread, at `n_mid = (end − start)/2`, with `std::nth_element` and the
//! comparator `(x[a, d], a) < (x[b, d], b)`.
//!
//! The *set* of points of every node does not depend on how `nth_element` is
//! implemented (the comparator is a strict total order), but the **order inside the
//! leaves** does, and that order decides which of two equally distant points the query
//! keeps. The fixtures were generated with the macOS wheel, which links **libc++**:
//! [`libcxx_nth_element`] is its algorithm (LLVM 15–19). A Linux (libstdc++) or Windows
//! (MSVC) build of scikit-learn may break ties differently; MLTutor always follows the
//! macOS order, so it is reproducible on every platform.
//!
//! **Query** (`_query_single_depthfirst`): depth first from the root; a node is pruned
//! when its `min_rdist` is **strictly** greater than the largest distance in the heap;
//! of two children the one with the smaller `min_rdist` goes first (the left one on
//! ties); a leaf pushes its points in `idx_array` order into the heap, which keeps the
//! **first visited** among equal distances.

use serde::{Deserialize, Serialize};

use super::search::{rdist, EffectiveMetric, NeighborHeap};

/// `leaf_size` of `KNeighbors*` (scikit-learn's default).
pub const LEAF_SIZE: usize = 30;

/// One node of the tree (`node_data` of scikit-learn).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct KdNode {
    pub idx_start: usize,
    pub idx_end: usize,
    pub is_leaf: bool,
}

/// A fitted kd-tree over `n` rows of `p` features.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct KdTree {
    n_features: usize,
    metric: EffectiveMetric,
    /// Permutation of the rows: the points of node `i` are `idx_array[start..end]`.
    idx_array: Vec<usize>,
    nodes: Vec<KdNode>,
    /// `n_nodes × p` row-major lower and upper bounds.
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// The rows in `idx_array` order (row-major), so that a leaf reads contiguous memory.
    data: Vec<f64>,
}

impl std::fmt::Debug for KdTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KdTree")
            .field("n_samples", &self.idx_array.len())
            .field("n_features", &self.n_features)
            .field("n_nodes", &self.nodes.len())
            .field("metric", &self.metric)
            .finish()
    }
}

/// `int(np.log2(max(1, (n − 1)/leaf_size)) + 1)`.
pub fn n_levels(n: usize, leaf_size: usize) -> usize {
    let ratio = (n.saturating_sub(1) as f64 / leaf_size.max(1) as f64).max(1.0);
    (ratio.log2() + 1.0) as usize
}

impl KdTree {
    /// Builds the tree over `data` (row-major `n × p`).
    ///
    /// Invariant: every value is finite. [`super::fit`] (the only caller) has already
    /// rejected NaN and ±∞ with `check_input`; with a NaN the spreads and the
    /// `nth_element` comparator would stop being a total order, so it is crate-private.
    pub(crate) fn build(data: &[f64], p: usize, metric: EffectiveMetric) -> Self {
        debug_assert!(
            data.iter().all(|v| v.is_finite()),
            "KdTree::build needs finite data"
        );
        let n = data.len().checked_div(p).unwrap_or(0);
        let levels = n_levels(n, LEAF_SIZE);
        // 2^levels − 1; `levels` is at most ~60 for any n that fits in memory.
        let n_nodes = (1usize << levels.min(62)) - 1;
        let mut tree = KdTree {
            n_features: p,
            metric,
            idx_array: (0..n).collect(),
            nodes: vec![KdNode::default(); n_nodes],
            lower: vec![0.0; n_nodes * p],
            upper: vec![0.0; n_nodes * p],
            data: Vec::new(),
        };
        if n > 0 && p > 0 {
            tree.recursive_build(data, 0, 0, n);
        }
        let mut sorted = Vec::with_capacity(n * p);
        for &i in &tree.idx_array {
            sorted.extend_from_slice(&data[i * p..(i + 1) * p]);
        }
        tree.data = sorted;
        tree
    }

    fn recursive_build(&mut self, data: &[f64], i_node: usize, start: usize, end: usize) {
        let p = self.n_features;
        // init_node: the bounds of the node's points.
        {
            let lo = &mut self.lower[i_node * p..(i_node + 1) * p];
            let hi = &mut self.upper[i_node * p..(i_node + 1) * p];
            lo.fill(f64::INFINITY);
            hi.fill(f64::NEG_INFINITY);
            for &i in &self.idx_array[start..end] {
                let row = &data[i * p..(i + 1) * p];
                for j in 0..p {
                    lo[j] = lo[j].min(row[j]);
                    hi[j] = hi[j].max(row[j]);
                }
            }
        }
        self.nodes[i_node] = KdNode {
            idx_start: start,
            idx_end: end,
            is_leaf: false,
        };
        if 2 * i_node + 1 >= self.nodes.len() || end - start < 2 {
            self.nodes[i_node].is_leaf = true;
            return;
        }
        // find_node_split_dim: the first feature with the strictly largest spread.
        let (mut j_max, mut max_spread) = (0, 0.0);
        for j in 0..p {
            let spread = self.upper[i_node * p + j] - self.lower[i_node * p + j];
            if spread > max_spread {
                max_spread = spread;
                j_max = j;
            }
        }
        let n_mid = (end - start) / 2;
        let less = |a: usize, b: usize| {
            let (va, vb) = (data[a * p + j_max], data[b * p + j_max]);
            if va == vb {
                a < b
            } else {
                va < vb
            }
        };
        libcxx_nth_element(&mut self.idx_array, start, start + n_mid, end, less);
        self.recursive_build(data, 2 * i_node + 1, start, start + n_mid);
        self.recursive_build(data, 2 * i_node + 2, start + n_mid, end);
    }

    /// The permutation of the rows (`get_arrays()[1]`).
    pub fn idx_array(&self) -> &[usize] {
        &self.idx_array
    }

    /// The nodes (`node_data`: `idx_start`, `idx_end`, `is_leaf`).
    pub fn nodes(&self) -> &[KdNode] {
        &self.nodes
    }

    /// `min_rdist`: the reduced distance from `pt` to the node's bounding box,
    /// `Σ (0.5·d)^p` with `d = (lo − x + |lo − x|) + (x − hi + |x − hi|)`.
    #[inline]
    fn min_rdist(&self, i_node: usize, pt: &[f64]) -> f64 {
        let p = self.n_features;
        let lo = &self.lower[i_node * p..(i_node + 1) * p];
        let hi = &self.upper[i_node * p..(i_node + 1) * p];
        let mut r = 0.0;
        if self.metric.p_is_one() {
            for j in 0..p {
                let d_lo = lo[j] - pt[j];
                let d_hi = pt[j] - hi[j];
                let d = (d_lo + d_lo.abs()) + (d_hi + d_hi.abs());
                r += 0.5 * d;
            }
        } else {
            for j in 0..p {
                let d_lo = lo[j] - pt[j];
                let d_hi = pt[j] - hi[j];
                let d = (d_lo + d_lo.abs()) + (d_hi + d_hi.abs());
                let h = 0.5 * d;
                // pow(h, 2.0) of libm is exactly h·h (correctly rounded).
                r += h * h;
            }
        }
        r
    }

    /// Fills `heap` with the K nearest rows of `pt` (row indices of the original
    /// data); the heap is left unsorted.
    pub(crate) fn query(&self, pt: &[f64], heap: &mut NeighborHeap) {
        if self.idx_array.is_empty() {
            return;
        }
        let lb = self.min_rdist(0, pt);
        self.visit(0, lb, pt, heap);
    }

    fn visit(&self, i_node: usize, lb: f64, pt: &[f64], heap: &mut NeighborHeap) {
        if lb > heap.largest() {
            return;
        }
        let node = self.nodes[i_node];
        if node.is_leaf {
            let p = self.n_features;
            for i in node.idx_start..node.idx_end {
                let row = &self.data[i * p..(i + 1) * p];
                heap.push(rdist(pt, row, self.metric), self.idx_array[i]);
            }
            return;
        }
        let (i1, i2) = (2 * i_node + 1, 2 * i_node + 2);
        let (lb1, lb2) = (self.min_rdist(i1, pt), self.min_rdist(i2, pt));
        if lb1 <= lb2 {
            self.visit(i1, lb1, pt, heap);
            self.visit(i2, lb2, pt, heap);
        } else {
            self.visit(i2, lb2, pt, heap);
            self.visit(i1, lb1, pt, heap);
        }
    }
}

// ---------------------------------------------------------------------------
// libc++ std::nth_element (include/__algorithm/nth_element.h, LLVM 15–19)
// ---------------------------------------------------------------------------

/// libc++'s `__sort3`: sorts `a[x], a[y], a[z]` and returns the number of swaps.
fn sort3(
    a: &mut [usize],
    x: usize,
    y: usize,
    z: usize,
    less: &impl Fn(usize, usize) -> bool,
) -> u32 {
    if !less(a[y], a[x]) {
        if !less(a[z], a[y]) {
            return 0;
        }
        a.swap(y, z);
        if less(a[y], a[x]) {
            a.swap(x, y);
            return 2;
        }
        return 1;
    }
    if less(a[z], a[y]) {
        a.swap(x, z);
        return 1;
    }
    a.swap(x, y);
    if less(a[z], a[y]) {
        a.swap(y, z);
        return 2;
    }
    1
}

/// libc++'s `__selection_sort` on `a[first..last]` (`last − first >= 2`).
fn selection_sort(
    a: &mut [usize],
    mut first: usize,
    last: usize,
    less: &impl Fn(usize, usize) -> bool,
) {
    let lm1 = last - 1;
    while first != lm1 {
        let mut m = first;
        for k in first + 1..last {
            if less(a[k], a[m]) {
                m = k;
            }
        }
        if m != first {
            a.swap(first, m);
        }
        first += 1;
    }
}

/// `std::nth_element(a + first, a + nth, a + last, less)` exactly as libc++ does it:
/// median-of-3 selection, selection sort for at most 7 elements, Hoare partition with
/// a guard, and the "already sorted" check when a partition made no swaps.
///
/// The branches for elements equivalent to the pivot are kept although scikit-learn's
/// comparator (value, then index) never makes two elements equivalent.
pub fn libcxx_nth_element(
    a: &mut [usize],
    mut first: usize,
    nth: usize,
    mut last: usize,
    less: impl Fn(usize, usize) -> bool,
) {
    const LIMIT: usize = 7;
    let less = &less;
    loop {
        if nth == last {
            return;
        }
        let n = last - first;
        if n <= 1 {
            return;
        }
        if n == 2 {
            if less(a[last - 1], a[first]) {
                a.swap(first, last - 1);
            }
            return;
        }
        if n == 3 {
            sort3(a, first, first + 1, last - 1, less);
            return;
        }
        if n <= LIMIT {
            selection_sort(a, first, last, less);
            return;
        }
        let mut m = first + n / 2;
        let lm1 = last - 1;
        let mut n_swaps = sort3(a, first, m, lm1, less);
        let mut i = first;
        let mut j = lm1;
        if !less(a[i], a[m]) {
            // *first == *m: look for an element smaller than the pivot from the end.
            let mut found = false;
            loop {
                j -= 1;
                if i == j {
                    break;
                }
                if less(a[j], a[m]) {
                    found = true;
                    break;
                }
            }
            if found {
                a.swap(i, j);
                n_swaps += 1;
            } else {
                // Every element in [first, last) is >= *first: partition on > *first.
                i += 1;
                j = last - 1;
                if !less(a[first], a[j]) {
                    loop {
                        if i == j {
                            return;
                        }
                        if less(a[first], a[i]) {
                            a.swap(i, j);
                            n_swaps += 1;
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                }
                if i == j {
                    return;
                }
                loop {
                    while !less(a[first], a[i]) {
                        i += 1;
                    }
                    j -= 1;
                    while less(a[first], a[j]) {
                        j -= 1;
                    }
                    if i >= j {
                        break;
                    }
                    a.swap(i, j);
                    n_swaps += 1;
                    i += 1;
                }
                if nth < i {
                    return;
                }
                first = i;
                continue;
            }
        }
        i += 1;
        if i < j {
            loop {
                while less(a[i], a[m]) {
                    i += 1;
                }
                j -= 1;
                while !less(a[j], a[m]) {
                    j -= 1;
                }
                if i >= j {
                    break;
                }
                a.swap(i, j);
                n_swaps += 1;
                if m == i {
                    m = j;
                }
                i += 1;
            }
        }
        if i != m && less(a[m], a[i]) {
            a.swap(i, m);
            n_swaps += 1;
        }
        if nth == i {
            return;
        }
        if n_swaps == 0 {
            // No swaps: the part that holds nth may already be sorted.
            let (mut j, mut m, end) = if nth < i {
                (first, first, i)
            } else {
                (i, i, last)
            };
            let mut sorted = false;
            loop {
                j += 1;
                if j == end {
                    sorted = true;
                    break;
                }
                if less(a[j], a[m]) {
                    break;
                }
                m = j;
            }
            if sorted {
                return;
            }
        }
        if nth < i {
            last = i;
        } else {
            i += 1;
            first = i;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny LCG shared with the Python check (`tools/knn_reference.py`'s
    /// `libcxx_nth_element` run on the same sequences gives the same FNV hashes).
    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed >> 33
    }

    fn fnv(values: &[usize]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &v in values {
            h ^= v as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }

    /// `(n, modulo of the values, nth)` and the FNV-1a hash of the permutation that
    /// `knn_reference.libcxx_nth_element` leaves (values: `lcg() % modulo`, comparator
    /// `(value, index)`, the LCG state carried over from case to case starting at 42).
    const CASES: &[(usize, u64, usize, u64)] = include!("nth_element_cases.in");

    #[test]
    fn nth_element_matches_libcxx_reference() {
        let mut seed = 42_u64;
        for &(n, modulo, nth, hash) in CASES {
            let vals: Vec<u64> = (0..n).map(|_| lcg(&mut seed) % modulo).collect();
            let mut a: Vec<usize> = (0..n).collect();
            libcxx_nth_element(&mut a, 0, nth, n, |x, y| {
                if vals[x] == vals[y] {
                    x < y
                } else {
                    vals[x] < vals[y]
                }
            });
            // The nth element and the partition hold whatever the implementation.
            let key = |i: usize| (vals[i], i);
            for &left in &a[..nth] {
                assert!(key(left) <= key(a[nth]), "n={n} nth={nth}");
            }
            for &right in &a[nth + 1..] {
                assert!(key(right) >= key(a[nth]), "n={n} nth={nth}");
            }
            assert_eq!(fnv(&a), hash, "n={n} modulo={modulo} nth={nth}: {a:?}");
        }
    }

    #[test]
    fn levels_like_sklearn() {
        assert_eq!(n_levels(1, 30), 1);
        assert_eq!(n_levels(105, 30), 2);
        assert_eq!(n_levels(150, 30), 3);
        assert_eq!(n_levels(14_448, 30), 9);
    }
}
