//! The CART builder: a line-by-line port of `tools/cart_reference.py`, itself a
//! transcription of scikit-learn 1.7.2 (`_tree.pyx`, `_splitter.pyx`,
//! `_partitioner.pyx`, `_criterion.pyx`) for the case MLTutor uses: dense `X`
//! without NaN, unit sample weights, `splitter="best"`, `max_features=None`,
//! `min_samples_leaf=1`, `min_impurity_decrease=0`.
//!
//! Exact parity with scikit-learn (D36) depends on details that look irrelevant:
//!
//! * **Ties between features.** For every node the features are visited in a
//!   random order (Fisher-Yates over the `features` array, which persists between
//!   nodes) drawn with scikit-learn's own xorshift RNG ([`RandR`]). The first
//!   feature that reaches the best improvement wins (strict `>`), so another
//!   order gives another tree: the root of Iris ties *petal length* and *petal
//!   width*.
//! * **float32.** `X` is converted to `f32` (like `check_array(dtype=float32)`);
//!   thresholds are the midpoint `x[p-1]/2 + x[p]/2` computed in `f64` from two
//!   `f32` values.
//! * **No minimum gap.** Two consecutive sorted values are separated whenever
//!   they differ, even by one float32 ulp, and a feature is constant only if
//!   `max <= min` (see [`FEATURE_THRESHOLD`]).
//! * **Order of the sums.** The samples are sorted with scikit-learn's introsort
//!   (not a stable sort) and every feature leaves them in its own order; the
//!   criteria accumulate in that order, forwards or, when shorter, backwards from
//!   the end (`reverse_reset`). Near-ties of 1 ulp (California, depth 8) are
//!   decided by that order.
//!
//! No FMA: Rust does not contract `a * b + c` unless asked to.

use crate::model::{ModelError, TrainControl, TrainProgress};

/// Sample index type of the splitter: `u32` halves the memory traffic of the
/// sort (the app's datasets have far fewer than 2³² rows; [`build`] checks it).
pub(crate) type Idx = u32;

/// `RAND_R_MAX` of `sklearn/utils/_random.pxd`.
pub const RAND_R_MAX: u32 = 0x7FFF_FFFF;
/// `FEATURE_THRESHOLD` of `_partitioner.pxd` *as the compiled scikit-learn
/// 1.7.2 behaves*: 0. The `.pxd` says `1e-7`, but sklearn splits two contiguous
/// float32 values and a column in the 1e-6 scale (fixture
/// `trees/tiny_scale_features`), so the splitter compares `x[p+1] <= x[p]` and a
/// feature is constant iff `max <= min`. Kept as documentation; the comparisons
/// below do not add it.
pub const FEATURE_THRESHOLD: f32 = 0.0;

/// The cancellation flag is also checked every this many candidate positions of
/// a feature: with `absolute_error` each position costs O(n) (two medians), so
/// one feature of 60 000 rows takes seconds. A relaxed atomic load every 256
/// positions is free for the other criteria.
const CANCEL_CHECK_POSITIONS: usize = 256;
/// `EPSILON` of `_tree.pyx` (`np.finfo('double').eps`).
pub const EPSILON: f64 = f64::EPSILON;

/// scikit-learn's splitter RNG: `our_rand_r` (32-bit xorshift) and `rand_int`
/// (`sklearn/tree/_utils.pyx`).
#[derive(Debug, Clone)]
pub struct RandR {
    state: u32,
}

impl RandR {
    /// Seeded with [`splitter_seed`].
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// `our_rand_r`: a value in `0..=RAND_R_MAX`.
    pub fn next_u31(&mut self) -> u32 {
        let mut s = self.state;
        if s == 0 {
            s = 1; // DEFAULT_SEED
        }
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.state = s;
        s % (RAND_R_MAX + 1)
    }

    /// `rand_int(low, high)`: `low + our_rand_r() % (high - low)` (`high > low`).
    pub fn rand_int(&mut self, low: usize, high: usize) -> usize {
        let span = high.saturating_sub(low).max(1);
        low + (self.next_u31() as usize) % span
    }
}

/// `Splitter.init`: `RandomState(random_state).randint(0, RAND_R_MAX)`
/// (1608637542 for 42).
pub fn splitter_seed(random_state: u32) -> u32 {
    let v = crate::sampling::RandomState::new(random_state).randint(0, i64::from(RAND_R_MAX));
    u32::try_from(v).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Introsort of `_partitioner.pyx`. scikit-learn sorts two parallel arrays
// (values, samples); here each value travels with its sample in one pair, which
// is the same algorithm (only the values are compared) with better locality.
// ---------------------------------------------------------------------------

/// A feature value (float32) and its sample index.
pub(crate) type Pair = (f32, Idx);

/// Median of the first, middle and last values of `v` (non-empty).
fn median3(v: &[Pair]) -> f32 {
    let n = v.len();
    let (a, b, c) = (v[0].0, v[n / 2].0, v[n - 1].0);
    if a < b {
        if b < c {
            b
        } else if a < c {
            c
        } else {
            a
        }
    } else if b < c {
        if a < c {
            a
        } else {
            c
        }
    } else {
        b
    }
}

fn sift_down(v: &mut [Pair], start: usize, end: usize) {
    let mut root = start;
    loop {
        let child = root * 2 + 1;
        let mut maxind = root;
        if child < end && v[maxind].0 < v[child].0 {
            maxind = child;
        }
        if child + 1 < end && v[maxind].0 < v[child + 1].0 {
            maxind = child + 1;
        }
        if maxind == root {
            break;
        }
        v.swap(root, maxind);
        root = maxind;
    }
}

fn heapsort(v: &mut [Pair]) {
    let n = v.len();
    if n < 2 {
        return;
    }
    let mut start = (n - 2) / 2;
    loop {
        sift_down(v, start, n);
        if start == 0 {
            break;
        }
        start -= 1;
    }
    let mut end = n - 1;
    while end > 0 {
        v.swap(0, end);
        sift_down(v, 0, end);
        end -= 1;
    }
}

/// Introsort on `v` (`lo` and `n` of scikit-learn become the slice itself).
fn introsort(mut v: &mut [Pair], mut maxd: i64) {
    while v.len() > 1 {
        if maxd <= 0 {
            heapsort(v);
            return;
        }
        maxd -= 1;
        let pivot = median3(v);
        // Three-way partition: < pivot | == pivot | > pivot.
        let (mut i, mut l, mut r) = (0, 0, v.len());
        while i < r {
            let x = v[i].0;
            if x < pivot {
                v.swap(i, l);
                i += 1;
                l += 1;
            } else if x > pivot {
                r -= 1;
                v.swap(i, r);
            } else {
                i += 1;
            }
        }
        let (left, rest) = std::mem::take(&mut v).split_at_mut(r);
        introsort(&mut left[..l], maxd);
        v = rest;
    }
}

/// `sort(feature_values, samples, n)`: introsort with `maxd = 2·⌊log2 n⌋`.
pub(crate) fn sort_pairs(v: &mut [Pair]) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let maxd = 2 * i64::from(n.ilog2());
    introsort(v, maxd);
}

/// The introsort on two parallel arrays, as scikit-learn's signature.
pub fn sort(v: &mut [f32], s: &mut [u32]) {
    let mut pairs: Vec<Pair> = v.iter().copied().zip(s.iter().copied()).collect();
    sort_pairs(&mut pairs);
    for ((dv, ds), (pv, ps)) in v.iter_mut().zip(s.iter_mut()).zip(pairs) {
        *dv = pv;
        *ds = ps;
    }
}

// ---------------------------------------------------------------------------
// Criteria
// ---------------------------------------------------------------------------

/// The interface of `sklearn.tree._criterion.Criterion` that the builder uses.
/// `samples` is the splitter's sample array; the criterion reads
/// `samples[start..end]`.
pub(crate) trait Criterion {
    fn init(&mut self, samples: &[Idx], start: usize, end: usize);
    fn reset(&mut self);
    fn update(&mut self, samples: &[Idx], new_pos: usize);
    fn node_impurity(&mut self, samples: &[Idx]) -> f64;
    fn children_impurity(&mut self, samples: &[Idx]) -> (f64, f64);
    /// `proxy_impurity_improvement`: ranks the splits of a node.
    fn proxy(&mut self, samples: &[Idx]) -> f64;
    /// `impurity_improvement` of the chosen split.
    fn improvement(&self, w_samples: f64, parent: f64, il: f64, ir: f64) -> f64;
    /// `tree_.value` of the node.
    fn node_value(&self) -> Vec<f64>;
    /// Class counts of the node (classification), empty otherwise.
    fn node_counts(&self) -> Vec<u32>;
}

/// `w_node/w · (parent − w_r/w_node · ir − w_l/w_node · il)`.
fn weighted_improvement(
    w_samples: f64,
    w_node: f64,
    wl: f64,
    wr: f64,
    parent: f64,
    il: f64,
    ir: f64,
) -> f64 {
    (w_node / w_samples) * (parent - (wr / w_node * ir) - (wl / w_node * il))
}

/// Gini (`1 − Σc²/w²`) or entropy (`−Σ (c/w)·log2(c/w)`).
pub(crate) struct ClassificationCriterion {
    y: Vec<usize>,
    k: usize,
    entropy: bool,
    start: usize,
    end: usize,
    pos: usize,
    sum_total: Vec<f64>,
    sum_left: Vec<f64>,
    sum_right: Vec<f64>,
    w_node: f64,
    w_left: f64,
    w_right: f64,
}

impl ClassificationCriterion {
    /// `y` = class index (`0..k`) of every training row.
    pub(crate) fn new(y: Vec<usize>, k: usize, entropy: bool) -> Self {
        Self {
            y,
            k,
            entropy,
            start: 0,
            end: 0,
            pos: 0,
            sum_total: vec![0.0; k],
            sum_left: vec![0.0; k],
            sum_right: vec![0.0; k],
            w_node: 0.0,
            w_left: 0.0,
            w_right: 0.0,
        }
    }

    fn impurity_of(&self, counts: &[f64], w: f64) -> f64 {
        if self.entropy {
            let ln2 = 2.0_f64.ln();
            let mut e = 0.0;
            for &c in counts {
                if c > 0.0 {
                    let c = c / w;
                    e -= c * (c.ln() / ln2);
                }
            }
            e
        } else {
            let mut sq = 0.0;
            for &c in counts {
                sq += c * c;
            }
            1.0 - sq / (w * w)
        }
    }

    fn reverse_reset(&mut self) {
        self.pos = self.end;
        self.sum_right.iter_mut().for_each(|v| *v = 0.0);
        self.sum_left.copy_from_slice(&self.sum_total);
        self.w_right = 0.0;
        self.w_left = self.w_node;
    }
}

impl Criterion for ClassificationCriterion {
    fn init(&mut self, samples: &[Idx], start: usize, end: usize) {
        self.start = start;
        self.end = end;
        self.sum_total.iter_mut().for_each(|v| *v = 0.0);
        let mut w = 0.0;
        for &i in &samples[start..end] {
            self.sum_total[self.y[i as usize]] += 1.0;
            w += 1.0;
        }
        self.w_node = w;
        self.reset();
    }

    fn reset(&mut self) {
        self.pos = self.start;
        self.sum_left.iter_mut().for_each(|v| *v = 0.0);
        self.sum_right.copy_from_slice(&self.sum_total);
        self.w_left = 0.0;
        self.w_right = self.w_node;
    }

    fn update(&mut self, samples: &[Idx], new_pos: usize) {
        let (pos, end) = (self.pos, self.end);
        if new_pos - pos <= end - new_pos {
            for &i in &samples[pos..new_pos] {
                self.sum_left[self.y[i as usize]] += 1.0;
                self.w_left += 1.0;
            }
        } else {
            self.reverse_reset();
            for &i in samples[new_pos..end].iter().rev() {
                self.sum_left[self.y[i as usize]] -= 1.0;
                self.w_left -= 1.0;
            }
        }
        self.w_right = self.w_node - self.w_left;
        for c in 0..self.k {
            self.sum_right[c] = self.sum_total[c] - self.sum_left[c];
        }
        self.pos = new_pos;
    }

    fn node_impurity(&mut self, _samples: &[Idx]) -> f64 {
        self.impurity_of(&self.sum_total, self.w_node)
    }

    fn children_impurity(&mut self, _samples: &[Idx]) -> (f64, f64) {
        (
            self.impurity_of(&self.sum_left, self.w_left),
            self.impurity_of(&self.sum_right, self.w_right),
        )
    }

    fn proxy(&mut self, samples: &[Idx]) -> f64 {
        let (il, ir) = self.children_impurity(samples);
        -self.w_right * ir - self.w_left * il
    }

    fn improvement(&self, w_samples: f64, parent: f64, il: f64, ir: f64) -> f64 {
        weighted_improvement(
            w_samples,
            self.w_node,
            self.w_left,
            self.w_right,
            parent,
            il,
            ir,
        )
    }

    fn node_value(&self) -> Vec<f64> {
        self.sum_total.iter().map(|c| c / self.w_node).collect()
    }

    fn node_counts(&self) -> Vec<u32> {
        // The sums are exact small integers (unit weights).
        self.sum_total.iter().map(|&c| c as u32).collect()
    }
}

/// Which regression criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RegressionKind {
    SquaredError,
    FriedmanMse,
    AbsoluteError,
}

/// `squared_error` (MSE), `friedman_mse` and `absolute_error` (MAE, medians).
pub(crate) struct RegressionCriterion {
    y: Vec<f64>,
    kind: RegressionKind,
    start: usize,
    end: usize,
    pos: usize,
    sum_total: f64,
    sq_sum_total: f64,
    sum_left: f64,
    sum_right: f64,
    w_node: f64,
    w_left: f64,
    w_right: f64,
    node_median: f64,
    /// Work buffer for the medians of MAE.
    scratch: Vec<f64>,
}

/// `WeightedMedianCalculator.get_median` with unit weights: the middle value, or
/// the mean of the two middle values when `n` is even. Reorders `values`.
fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    let cmp = |a: &f64, b: &f64| a.total_cmp(b);
    let (left, &mut mid, _) = values.select_nth_unstable_by(n / 2, cmp);
    if n % 2 == 0 {
        let lower = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lower + mid) / 2.0
    } else {
        mid
    }
}

impl RegressionCriterion {
    pub(crate) fn new(y: Vec<f64>, kind: RegressionKind) -> Self {
        Self {
            y,
            kind,
            start: 0,
            end: 0,
            pos: 0,
            sum_total: 0.0,
            sq_sum_total: 0.0,
            sum_left: 0.0,
            sum_right: 0.0,
            w_node: 0.0,
            w_left: 0.0,
            w_right: 0.0,
            node_median: f64::NAN,
            scratch: Vec::new(),
        }
    }

    fn reverse_reset(&mut self) {
        self.pos = self.end;
        self.sum_right = 0.0;
        self.sum_left = self.sum_total;
        self.w_right = 0.0;
        self.w_left = self.w_node;
    }

    /// Median of `y` over `samples[a..b]` and `Σ|y − median|` in sample order.
    fn abs_deviation(&mut self, samples: &[Idx], a: usize, b: usize) -> (f64, f64) {
        self.scratch.clear();
        self.scratch
            .extend(samples[a..b].iter().map(|&i| self.y[i as usize]));
        let m = median(&mut self.scratch);
        let mut s = 0.0;
        for &i in &samples[a..b] {
            s += (self.y[i as usize] - m).abs() * 1.0;
        }
        (m, s)
    }
}

impl Criterion for RegressionCriterion {
    fn init(&mut self, samples: &[Idx], start: usize, end: usize) {
        self.start = start;
        self.end = end;
        let (mut tot, mut sq, mut w) = (0.0, 0.0, 0.0);
        for &i in &samples[start..end] {
            let yi = self.y[i as usize];
            let wy = 1.0 * yi;
            tot += wy;
            sq += wy * yi;
            w += 1.0;
        }
        self.sum_total = tot;
        self.sq_sum_total = sq;
        self.w_node = w;
        if self.kind == RegressionKind::AbsoluteError {
            self.node_median = self.abs_deviation(samples, start, end).0;
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.pos = self.start;
        self.sum_left = 0.0;
        self.sum_right = self.sum_total;
        self.w_left = 0.0;
        self.w_right = self.w_node;
    }

    fn update(&mut self, samples: &[Idx], new_pos: usize) {
        let (pos, end) = (self.pos, self.end);
        if self.kind == RegressionKind::AbsoluteError {
            // Only the weights matter: the medians are order statistics,
            // recomputed exactly in `children_impurity`.
            self.w_left = (new_pos - self.start) as f64;
            self.w_right = self.w_node - self.w_left;
            self.pos = new_pos;
            return;
        }
        if new_pos - pos <= end - new_pos {
            for &i in &samples[pos..new_pos] {
                self.sum_left += 1.0 * self.y[i as usize];
                self.w_left += 1.0;
            }
        } else {
            self.reverse_reset();
            for &i in samples[new_pos..end].iter().rev() {
                self.sum_left -= 1.0 * self.y[i as usize];
                self.w_left -= 1.0;
            }
        }
        self.w_right = self.w_node - self.w_left;
        self.sum_right = self.sum_total - self.sum_left;
        self.pos = new_pos;
    }

    fn node_impurity(&mut self, samples: &[Idx]) -> f64 {
        if self.kind == RegressionKind::AbsoluteError {
            let m = self.node_median;
            let mut imp = 0.0;
            for &i in &samples[self.start..self.end] {
                imp += (self.y[i as usize] - m).abs() * 1.0;
            }
            return imp / self.w_node;
        }
        let mean = self.sum_total / self.w_node;
        self.sq_sum_total / self.w_node - mean * mean
    }

    fn children_impurity(&mut self, samples: &[Idx]) -> (f64, f64) {
        let (start, pos, end) = (self.start, self.pos, self.end);
        if self.kind == RegressionKind::AbsoluteError {
            let (_, il) = self.abs_deviation(samples, start, pos);
            let (_, ir) = self.abs_deviation(samples, pos, end);
            return (il / self.w_left, ir / self.w_right);
        }
        let mut sq_left = 0.0;
        for &i in &samples[start..pos] {
            let yi = self.y[i as usize];
            sq_left += 1.0 * yi * yi;
        }
        let sq_right = self.sq_sum_total - sq_left;
        let ml = self.sum_left / self.w_left;
        let mr = self.sum_right / self.w_right;
        (
            sq_left / self.w_left - ml * ml,
            sq_right / self.w_right - mr * mr,
        )
    }

    fn proxy(&mut self, samples: &[Idx]) -> f64 {
        match self.kind {
            RegressionKind::SquaredError => {
                (self.sum_left * self.sum_left) / self.w_left
                    + (self.sum_right * self.sum_right) / self.w_right
            }
            RegressionKind::FriedmanMse => {
                let diff = self.w_right * self.sum_left - self.w_left * self.sum_right;
                diff * diff / (self.w_left * self.w_right)
            }
            RegressionKind::AbsoluteError => {
                let (il, ir) = self.children_impurity(samples);
                -self.w_right * ir - self.w_left * il
            }
        }
    }

    fn improvement(&self, w_samples: f64, parent: f64, il: f64, ir: f64) -> f64 {
        if self.kind == RegressionKind::FriedmanMse {
            let diff = self.w_right * self.sum_left - self.w_left * self.sum_right;
            return diff * diff / (self.w_left * self.w_right * self.w_node);
        }
        weighted_improvement(
            w_samples,
            self.w_node,
            self.w_left,
            self.w_right,
            parent,
            il,
            ir,
        )
    }

    fn node_value(&self) -> Vec<f64> {
        if self.kind == RegressionKind::AbsoluteError {
            vec![self.node_median]
        } else {
            vec![self.sum_total / self.w_node]
        }
    }

    fn node_counts(&self) -> Vec<u32> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Builder + splitter
// ---------------------------------------------------------------------------

/// One node as built (sklearn's `Node` struct plus `value`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawNode {
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub feature: Option<usize>,
    pub threshold: f64,
    pub impurity: f64,
    pub n_samples: usize,
    pub missing_go_to_left: bool,
    pub value: Vec<f64>,
    pub counts: Vec<u32>,
    pub depth: usize,
}

/// What the RNG and the criterion did at a split node (`reference_impl.trace`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SplitTrace {
    /// Node id.
    pub node: usize,
    /// Features in the order they were drawn (constant ones included).
    pub visit_order: Vec<usize>,
    /// Best proxy improvement of each visited feature (`None` if constant), in
    /// visit order.
    pub best_proxy: Vec<Option<f64>>,
    /// The chosen feature (`None` if no valid split was found).
    pub chosen_feature: Option<usize>,
    /// Features that reached exactly the same proxy and lost (drawn later).
    pub exact_ties_lost: Vec<usize>,
}

struct Record {
    start: usize,
    end: usize,
    depth: usize,
    parent: Option<(usize, bool)>,
    impurity: f64,
    n_constant: usize,
}

struct Best {
    pos: usize,
    feature: usize,
    threshold: f64,
    improvement: f64,
    impurity_left: f64,
    impurity_right: f64,
    missing_go_to_left: bool,
}

/// Output of [`build`].
pub(crate) struct Built {
    pub nodes: Vec<RawNode>,
    pub max_depth: usize,
    pub traces: Vec<SplitTrace>,
}

/// Parameters of [`build`].
pub(crate) struct BuildParams {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub random_state: u32,
}

struct Splitter<'a> {
    /// X in float32, column-major: `x[f * n + i]`.
    x: &'a [f32],
    n: usize,
    n_features: usize,
    rng: RandR,
    samples: Vec<Idx>,
    fvals: Vec<f32>,
    /// Work buffer of the sort.
    pairs: Vec<Pair>,
    features: Vec<usize>,
    constant_features: Vec<usize>,
    w_samples: f64,
}

impl Splitter<'_> {
    /// `node_split_best`. Returns the best split (`pos == end` if none), the
    /// number of constant features for the children, and the trace.
    fn node_split<C: Criterion>(
        &mut self,
        crit: &mut C,
        start: usize,
        end: usize,
        n_known_constants: usize,
        impurity: f64,
        control: &TrainControl<'_>,
    ) -> Result<(Best, usize, SplitTrace), ModelError> {
        let max_features = self.n_features;
        let mut f_i = self.n_features;
        let (mut n_visited, mut n_found_constants, mut n_drawn_constants) = (0, 0, 0);
        let mut n_total_constants = n_known_constants;
        let mut best = Best {
            pos: end,
            feature: 0,
            threshold: 0.0,
            improvement: f64::NEG_INFINITY,
            impurity_left: f64::INFINITY,
            impurity_right: f64::INFINITY,
            missing_go_to_left: false,
        };
        let mut best_proxy = f64::NEG_INFINITY;
        let mut trace = SplitTrace {
            node: 0,
            visit_order: Vec::new(),
            best_proxy: Vec::new(),
            chosen_feature: None,
            exact_ties_lost: Vec::new(),
        };

        while f_i > n_total_constants
            && (n_visited < max_features || n_visited <= n_found_constants + n_drawn_constants)
        {
            n_visited += 1;
            let mut f_j = self
                .rng
                .rand_int(n_drawn_constants, f_i - n_found_constants);
            if f_j < n_known_constants {
                // A feature already known to be constant in this node.
                self.features.swap(n_drawn_constants, f_j);
                n_drawn_constants += 1;
                continue;
            }
            f_j += n_found_constants;
            let current = self.features[f_j];
            control.check()?;

            // Copy the column (in float32) and sort it with the samples.
            let x: &[f32] = self.x;
            let col = &x[current * self.n..(current + 1) * self.n];
            let pairs = &mut self.pairs[start..end];
            for (pair, &i) in pairs.iter_mut().zip(&self.samples[start..end]) {
                *pair = (col[i as usize], i);
            }
            sort_pairs(pairs);
            for ((v, s), &(pv, ps)) in self.fvals[start..end]
                .iter_mut()
                .zip(&mut self.samples[start..end])
                .zip(pairs.iter())
            {
                *v = pv;
                *s = ps;
            }
            trace.visit_order.push(current);

            // Constant in this node: `max <= min` (FEATURE_THRESHOLD = 0).
            if self.fvals[end - 1] <= self.fvals[start] {
                self.features.swap(f_j, n_total_constants);
                n_found_constants += 1;
                n_total_constants += 1;
                trace.best_proxy.push(None);
                continue;
            }
            f_i -= 1;
            self.features.swap(f_i, f_j);

            crit.reset();
            let mut feat_best = f64::NEG_INFINITY;
            let mut p = start;
            let mut n_candidates = 0_usize;
            while p < end {
                // `next_p`: skip equal values (FEATURE_THRESHOLD = 0).
                while p + 1 < end && self.fvals[p + 1] <= self.fvals[p] {
                    p += 1;
                }
                let p_prev = p;
                p += 1;
                if p >= end {
                    continue;
                }
                n_candidates += 1;
                if n_candidates % CANCEL_CHECK_POSITIONS == 0 {
                    control.check()?;
                }
                crit.update(&self.samples, p);
                let proxy = crit.proxy(&self.samples);
                if proxy > feat_best {
                    feat_best = proxy;
                }
                if proxy > best_proxy {
                    best_proxy = proxy;
                    let (a, b) = (f64::from(self.fvals[p_prev]), f64::from(self.fvals[p]));
                    let mut threshold = a / 2.0 + b / 2.0;
                    if threshold == b || threshold.is_infinite() || threshold.is_nan() {
                        threshold = a;
                    }
                    best.pos = p;
                    best.feature = current;
                    best.threshold = threshold;
                    best.missing_go_to_left = p - start > end - p;
                }
            }
            trace.best_proxy.push(Some(feat_best));
        }

        if best.pos < end {
            // `partition_samples_final` on the order left by the LAST feature.
            let x: &[f32] = self.x;
            let col = &x[best.feature * self.n..(best.feature + 1) * self.n];
            let (mut p, mut partition_end) = (start, end - 1);
            while p < partition_end {
                if f64::from(col[self.samples[p] as usize]) <= best.threshold {
                    p += 1;
                } else {
                    self.samples.swap(p, partition_end);
                    partition_end -= 1;
                }
            }
            crit.reset();
            crit.update(&self.samples, best.pos);
            let (il, ir) = crit.children_impurity(&self.samples);
            best.impurity_left = il;
            best.impurity_right = ir;
            best.improvement = crit.improvement(self.w_samples, impurity, il, ir);
            trace.chosen_feature = Some(best.feature);
            trace.exact_ties_lost = trace
                .visit_order
                .iter()
                .zip(&trace.best_proxy)
                .filter(|(&f, bp)| f != best.feature && **bp == Some(best_proxy))
                .map(|(&f, _)| f)
                .collect();
        }

        // Constant features keep their order for siblings and children.
        self.features[..n_known_constants]
            .copy_from_slice(&self.constant_features[..n_known_constants]);
        let found = n_known_constants..n_known_constants + n_found_constants;
        self.constant_features[found.clone()].copy_from_slice(&self.features[found]);
        Ok((best, n_total_constants, trace))
    }
}

/// `DepthFirstTreeBuilder.build`: `x32` is column-major float32 (`n_features`
/// columns of `n` values), `crit` holds `y`.
pub(crate) fn build<C: Criterion>(
    x32: &[f32],
    n: usize,
    n_features: usize,
    mut crit: C,
    params: &BuildParams,
    control: &TrainControl<'_>,
) -> Result<Built, ModelError> {
    if Idx::try_from(n).is_err() {
        return Err(ModelError::InvalidInput {
            reason: "demasiadas filas".to_owned(),
        });
    }
    let mut sp = Splitter {
        x: x32,
        n,
        n_features,
        rng: RandR::new(splitter_seed(params.random_state)),
        samples: (0..n as Idx).collect(),
        fvals: vec![0.0; n],
        pairs: vec![(0.0, 0); n],
        features: (0..n_features).collect(),
        constant_features: vec![0; n_features],
        w_samples: n as f64,
    };
    let mut nodes: Vec<RawNode> = Vec::new();
    let mut traces = Vec::new();
    let mut max_depth_seen = 0;
    let mut stack = vec![Record {
        start: 0,
        end: n,
        depth: 0,
        parent: None,
        impurity: f64::INFINITY,
        n_constant: 0,
    }];
    let mut first = true;
    while let Some(rec) = stack.pop() {
        control.check()?;
        let (start, end, depth) = (rec.start, rec.end, rec.depth);
        let n_node = end - start;
        crit.init(&sp.samples, start, end);
        let mut is_leaf =
            depth >= params.max_depth || n_node < params.min_samples_split || n_node < 2;
        let mut impurity = rec.impurity;
        if first {
            impurity = crit.node_impurity(&sp.samples);
            first = false;
        }
        is_leaf = is_leaf || impurity <= EPSILON;
        let mut split = None;
        let mut n_constant = rec.n_constant;
        if !is_leaf {
            let (best, n_total_constants, trace) =
                sp.node_split(&mut crit, start, end, n_constant, impurity, control)?;
            n_constant = n_total_constants;
            is_leaf = best.pos >= end || best.improvement + EPSILON < 0.0;
            traces.push((nodes.len(), trace));
            split = Some(best);
        }
        let node_id = nodes.len();
        let split = if is_leaf { None } else { split };
        nodes.push(RawNode {
            left: None,
            right: None,
            feature: split.as_ref().map(|s| s.feature),
            threshold: split.as_ref().map_or(-2.0, |s| s.threshold),
            impurity,
            n_samples: n_node,
            missing_go_to_left: split.as_ref().is_some_and(|s| s.missing_go_to_left),
            value: crit.node_value(),
            counts: crit.node_counts(),
            depth,
        });
        if let Some((parent, is_left)) = rec.parent {
            if is_left {
                nodes[parent].left = Some(node_id);
            } else {
                nodes[parent].right = Some(node_id);
            }
        }
        max_depth_seen = max_depth_seen.max(depth);
        if let Some(best) = split {
            stack.push(Record {
                start: best.pos,
                end,
                depth: depth + 1,
                parent: Some((node_id, false)),
                impurity: best.impurity_right,
                n_constant,
            });
            stack.push(Record {
                start,
                end: best.pos,
                depth: depth + 1,
                parent: Some((node_id, true)),
                impurity: best.impurity_left,
                n_constant,
            });
        }
        if node_id % 64 == 0 {
            control.report(TrainProgress {
                iteration: nodes.len() as u64,
                max_iter: 0,
                loss: None,
                finished: false,
            });
        }
    }
    control.report(TrainProgress {
        iteration: nodes.len() as u64,
        max_iter: 0,
        loss: None,
        finished: true,
    });
    let traces = traces
        .into_iter()
        .map(|(node, mut t)| {
            t.node = node;
            t
        })
        .collect();
    Ok(Built {
        nodes,
        max_depth: max_depth_seen,
        traces,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_seed_is_numpy_randint() {
        assert_eq!(splitter_seed(42), 1_608_637_542);
    }

    #[test]
    fn xorshift_never_sticks_at_zero() {
        let mut r = RandR::new(0);
        let v = r.next_u31();
        assert!(v > 0);
        assert!(v <= RAND_R_MAX);
    }

    #[test]
    fn introsort_sorts_with_samples() {
        let mut v: Vec<f32> = (0..200).map(|i| ((i * 37) % 101) as f32 * 0.5).collect();
        let orig = v.clone();
        let mut s: Vec<u32> = (0..200).collect();
        sort(&mut v, &mut s);
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
        assert!(s.iter().zip(&v).all(|(&i, &x)| orig[i as usize] == x));
    }

    #[test]
    fn median_like_sklearn() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
