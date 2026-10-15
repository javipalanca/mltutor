//! 🔍 K-Nearest Neighbors (fase4-knn.md C17–C20): `KNeighborsClassifier` and
//! `KNeighborsRegressor` of scikit-learn 1.7.2 with `algorithm='auto'`,
//! `leaf_size=30`, `p=2`, with **exact** parity in the neighbours (D52).
//!
//! KNN has no training: `fit` stores the rows and, when scikit-learn would, builds a
//! kd-tree. Parity is about choosing **the same neighbours**, in the same order, which
//! matters when two training rows are at the same distance as the K-th neighbour
//! (duplicated points are common: Titanic, Iris in 2-D, Diabetes' `sex`). Each engine
//! breaks those ties its own way, and both are reproduced:
//!
//! * [`SearchEngine::KdTree`] (≤ 15 features and K < n_train // 2,
//!   [`search_engine`]): the kd-tree of [`kdtree`] (libc++ `nth_element`, depth-first
//!   query, heap that keeps the first visited point, `simultaneous_sort`). Same
//!   indices in the same order and bit-identical distances as the fixtures.
//! * [`SearchEngine::Brute`]: every row pushed in index order into scikit-learn's
//!   heap. This is **not** "the lowest index wins ties": a smaller value evicts the
//!   heap's root, and which of several tied maxima is the root depends on the heap
//!   layout ([`search::brute_query`]). It is single-threaded scikit-learn; with
//!   several threads scikit-learn merges per-thread heaps and its ties change with
//!   the number of cores (the scripts use `threadpool_limits(1)`). Even with one
//!   thread, scikit-learn's `ArgKmin` merges its heap into a fresh one when the call
//!   has ≤ 1024 rows and no more than n_train (`parallel_on_Y`), which reorders the
//!   tied neighbours; that is reproduced too ([`search::brute_parallel_on_y`]), so
//!   the order depends on how many rows are searched **in one call** (as in
//!   scikit-learn). With K = n_train every row is a neighbour; only the order
//!   inside runs of equal distances could differ from another scikit-learn build.
//!   For the Euclidean metric scikit-learn ranks with the BLAS surrogate
//!   `‖x‖² − 2x·y + ‖y‖²`, so its distances differ from ours by ≤ 1e-9 relative
//!   (4.3e-11 in Cancer) and a near tie could choose another neighbour; Manhattan is
//!   exact. In particular, a query identical to a training row is at ~1e-5–1e-4
//!   from it in scikit-learn (cancellation), **not** 0, so its distance-0 rule does
//!   not fire there, while here the distance is exactly 0 and it does.
//!
//! The ONNX export (`mltutor-export`, `TopK` is stable) breaks ties by the lowest
//! training index, which is neither engine's rule: with discrete data its neighbours
//! can differ from the app's.
//!
//! Distances that overflow (features ~1e160) are rejected with
//! [`ModelError::NonFinite`] instead of ranking with `inf`
//! ([`search::max_rdist_to_box`]).
//!
//! Distances (D53) are computed like the macOS arm64 wheel, see [`search`].
//!
//! Prediction rules (`_classification.py`, `_regression.py`, `_base.py`):
//!
//! * weights: `uniform` → 1; `distance` → `1/d`, except that when some neighbour of
//!   the row is at distance 0 **only** those weigh (1) and the rest weigh 0;
//! * `predict_proba`: votes added in neighbour order, divided by their sum;
//! * `predict` (classes): `uniform` → the mode (smallest class among the most voted);
//!   `distance` → `weighted_mode` (per-class pairwise sums, strict `>`: the smallest
//!   class wins exact ties);
//! * regression: mean of the K values, or `Σ w·y / Σ w` (NumPy pairwise sums).
//!
//! Differences with scikit-learn: `K > n_train` is rejected when fitting
//! ([`ModelError::TooFewSamples`], D59; scikit-learn fits and fails when predicting),
//! and data with NaN is rejected (as in every lesson).
//!
//! Other modules: [`class_distances`] (C19, 📏 Análisis de Distancias) and
//! [`playground`] (C18, 🎮 Visualización Interactiva).

pub mod class_distances;
pub mod kdtree;
pub mod playground;
pub mod search;

use std::any::Any;
use std::fmt;

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

pub use class_distances::{class_distances, ClassDistances, SeparationVerdict};
pub use kdtree::{KdNode, KdTree, LEAF_SIZE};

pub use playground::{
    playground, playground_model, playground_points, Playground, PlaygroundPoint,
    PlaygroundRequest, PALETTE,
};
pub use search::EffectiveMetric;

use crate::dataset::{Target, TaskType};
use crate::model::{check_input, check_n_classes, ModelError, Prediction, Predictor, TrainControl};
use crate::stats::pairwise_sum;
use search::{brute_query, NeighborHeap};

/// `weights` of `KNeighbors*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnnWeights {
    Uniform,
    Distance,
}

impl KnnWeights {
    /// scikit-learn's name (`"uniform"` / `"distance"`).
    pub fn as_str(self) -> &'static str {
        match self {
            KnnWeights::Uniform => "uniform",
            KnnWeights::Distance => "distance",
        }
    }
}

/// `metric` of `KNeighbors*` (the three options of the lesson). `Minkowski` (with
/// scikit-learn's default `p = 2`) **is** the Euclidean distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnnMetric {
    Minkowski,
    Euclidean,
    Manhattan,
}

impl KnnMetric {
    /// scikit-learn's name (`"minkowski"`, `"euclidean"`, `"manhattan"`).
    pub fn as_str(self) -> &'static str {
        match self {
            KnnMetric::Minkowski => "minkowski",
            KnnMetric::Euclidean => "euclidean",
            KnnMetric::Manhattan => "manhattan",
        }
    }

    /// The distance the search uses (`effective_metric_`).
    pub fn effective(self) -> EffectiveMetric {
        match self {
            KnnMetric::Manhattan => EffectiveMetric::Manhattan,
            KnnMetric::Minkowski | KnnMetric::Euclidean => EffectiveMetric::Euclidean,
        }
    }
}

/// Which search scikit-learn's `algorithm='auto'` picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchEngine {
    KdTree,
    Brute,
}

impl SearchEngine {
    /// `_fit_method` of scikit-learn (`"kd_tree"` / `"brute"`).
    pub fn as_str(self) -> &'static str {
        match self {
            SearchEngine::KdTree => "kd_tree",
            SearchEngine::Brute => "brute",
        }
    }
}

/// `KNeighborsMixin._fit` with `algorithm='auto'` and the lesson's metrics: brute
/// force with more than 15 features or when `K >= n_train // 2`; a kd-tree otherwise.
pub fn search_engine(n_train: usize, n_features: usize, n_neighbors: usize) -> SearchEngine {
    if n_features > 15 || n_neighbors >= n_train / 2 {
        SearchEngine::Brute
    } else {
        SearchEngine::KdTree
    }
}

/// Hyperparameters: `KNeighbors*(n_neighbors, weights, metric)`; the rest are
/// scikit-learn's defaults (`algorithm='auto'`, `leaf_size=30`, `p=2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KnnConfig {
    /// K (≥ 1; the lesson offers 1–20).
    pub n_neighbors: u32,
    pub weights: KnnWeights,
    pub metric: KnnMetric,
}

impl Default for KnnConfig {
    /// The lesson's defaults: K = 5, `uniform`, `minkowski`.
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            weights: KnnWeights::Uniform,
            metric: KnnMetric::Minkowski,
        }
    }
}

impl KnnConfig {
    /// `Err(InvalidConfig)` with K = 0.
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.n_neighbors == 0 {
            return Err(ModelError::InvalidConfig {
                reason: "el número de vecinos (K) debe ser al menos 1".to_owned(),
            });
        }
        Ok(())
    }

    fn k(&self) -> usize {
        self.n_neighbors as usize
    }
}

/// What the training tab shows of a fitted KNN (contract `KnnSummary`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnnSummary {
    pub n_neighbors: u32,
    pub weights: KnnWeights,
    pub metric: KnnMetric,
    pub engine: SearchEngine,
    pub n_train: usize,
    pub n_features: usize,
}

/// The K neighbours of every query row (`kneighbors`), row-major `n_rows × k`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Neighbors {
    pub k: usize,
    /// Distances (not reduced), ascending in each row.
    pub distances: Vec<f64>,
    /// Positions in the training data (`X_train` rows, in fit order).
    pub indices: Vec<usize>,
}

impl Neighbors {
    /// Number of query rows.
    pub fn n_rows(&self) -> usize {
        self.indices.len().checked_div(self.k).unwrap_or(0)
    }

    /// Distances of row `i`.
    pub fn row_distances(&self, i: usize) -> &[f64] {
        &self.distances[i * self.k..(i + 1) * self.k]
    }

    /// Indices of row `i`.
    pub fn row_indices(&self, i: usize) -> &[usize] {
        &self.indices[i * self.k..(i + 1) * self.k]
    }
}

/// One neighbour of a prediction (C20).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Neighbor {
    /// Order (0 = nearest).
    pub rank: usize,
    /// Position in the training data.
    pub index: usize,
    pub distance: f64,
    /// Its weight in the vote or the average (1, `1/d`, or the distance-0 rule).
    pub weight: f64,
    /// Class code (classification).
    pub class: Option<usize>,
    /// Target value (regression).
    pub value: Option<f64>,
}

/// One prediction with its neighbours (🔮 Predicciones, D58; and the playground).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnnExplanation {
    /// Predicted class code (classification).
    pub class: Option<usize>,
    /// Predicted value (regression).
    pub value: Option<f64>,
    /// `predict_proba` in `classes()` order (classification).
    pub proba: Option<Vec<f64>>,
    pub neighbors: Vec<Neighbor>,
    /// Some neighbour is at distance 0 and `weights = distance`: only those count.
    pub zero_distance_rule: bool,
    /// More training samples than free places at the distance of the K-th neighbour
    /// (`[predicciones] vecinos_nota_empate`); `None` without a tie. Filled by
    /// [`Knn::explain`] (and the playground); [`Knn::explain_row`] leaves it `None`.
    pub tie_at_kth: Option<KthTie>,
}

/// A tie at the K-th distance: `n_tied` training samples are exactly at `distance`
/// and only `slots` of them fit among the K neighbours (scikit-learn's engine chose
/// which, see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KthTie {
    pub n_tied: usize,
    pub slots: usize,
    pub distance: f64,
}

/// Below this many query rows the search runs on the calling thread.
const PARALLEL_MIN_ROWS: usize = 2048;
/// Upper bound of worker threads of one search.
const MAX_THREADS: usize = 8;
/// The cancellation flag is checked every this many query rows.
const CANCEL_EVERY: usize = 256;

/// A fitted `KNeighborsClassifier` / `KNeighborsRegressor`.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Knn {
    config: KnnConfig,
    engine: SearchEngine,
    n_features: usize,
    /// The training rows in fit order (standard layout, so [`Knn::rows`] is the
    /// row-major buffer).
    data: Array2<f64>,
    /// Class codes seen in training (`classes_`), sorted; `None` for regression.
    classes: Option<Vec<usize>>,
    /// Classification: index into `classes` of each training row.
    labels: Vec<usize>,
    /// Regression: the target of each training row.
    values: Vec<f64>,
    tree: Option<KdTree>,
}

impl fmt::Debug for Knn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Knn")
            .field("config", &self.config)
            .field("engine", &self.engine)
            .field("n_train", &self.n_train())
            .field("n_features", &self.n_features)
            .field("classes", &self.classes)
            .finish()
    }
}

/// Fits a KNN on the training rows (the entry point of `train_model`). The task is
/// the target's: classes → `KNeighborsClassifier`, values → `KNeighborsRegressor`.
pub fn fit(
    config: KnnConfig,
    x: ArrayView2<'_, f64>,
    y: &Target,
    control: &TrainControl<'_>,
) -> Result<Knn, ModelError> {
    config.validate()?;
    let (n, p) = x.dim();
    if n == 0 {
        return Err(ModelError::Empty);
    }
    if y.len() != n {
        return Err(ModelError::LengthMismatch {
            rows: n,
            targets: y.len(),
        });
    }
    if p == 0 {
        return Err(ModelError::FeatureMismatch {
            expected: 1,
            found: 0,
        });
    }
    check_input(x, p)?;
    let k = config.k();
    if k > n {
        return Err(ModelError::TooFewSamples { k, n_train: n });
    }
    control.check()?;
    let (classes, labels, values) = match y {
        Target::Classes { .. } => {
            let codes = y.class_codes().unwrap_or_default();
            let mut classes = codes.clone();
            classes.sort_unstable();
            classes.dedup();
            check_n_classes(classes.len())?;
            let labels = codes
                .iter()
                .map(|c| classes.binary_search(c).unwrap_or(0))
                .collect();
            (Some(classes), labels, Vec::new())
        }
        Target::Values { values } => {
            if values.iter().any(|v| !v.is_finite()) {
                return Err(ModelError::NonFinite);
            }
            (None, Vec::new(), values.clone())
        }
    };
    let data = Array2::from_shape_vec((n, p), x.iter().copied().collect()).map_err(|e| {
        ModelError::Numerical {
            message: e.to_string(),
        }
    })?;
    let engine = search_engine(n, p, k);
    let tree = match engine {
        SearchEngine::KdTree => Some(KdTree::build(
            data.as_slice().unwrap_or(&[]),
            p,
            config.metric.effective(),
        )),
        SearchEngine::Brute => None,
    };
    Ok(Knn {
        config,
        engine,
        n_features: p,
        data,
        classes,
        labels,
        values,
        tree,
    })
}

/// [`fit`] boxed as a [`Predictor`].
pub fn fit_boxed(
    config: KnnConfig,
    x: ArrayView2<'_, f64>,
    y: &Target,
    control: &TrainControl<'_>,
) -> Result<Box<dyn Predictor>, ModelError> {
    Ok(Box::new(fit(config, x, y, control)?))
}

/// `_get_weights` for one row of distances.
fn row_weights(dist: &[f64], weights: KnnWeights, out: &mut Vec<f64>) -> bool {
    out.clear();
    match weights {
        KnnWeights::Uniform => {
            out.resize(dist.len(), 1.0);
            false
        }
        KnnWeights::Distance => {
            out.extend(dist.iter().map(|&d| 1.0 / d));
            let has_inf = out.iter().any(|w| w.is_infinite());
            if has_inf {
                for w in out.iter_mut() {
                    *w = if w.is_infinite() { 1.0 } else { 0.0 };
                }
            }
            has_inf
        }
    }
}

impl Knn {
    /// The hyperparameters.
    pub fn config(&self) -> &KnnConfig {
        &self.config
    }

    /// The engine `algorithm='auto'` chose (`_fit_method`).
    pub fn engine(&self) -> SearchEngine {
        self.engine
    }

    /// `effective_metric_`.
    pub fn effective_metric(&self) -> EffectiveMetric {
        self.config.metric.effective()
    }

    /// Number of training rows (`n_samples_fit_`).
    pub fn n_train(&self) -> usize {
        self.data.nrows()
    }

    /// The training rows (`n_train × n_features`), in fit order.
    pub fn train_x(&self) -> ArrayView2<'_, f64> {
        self.data.view()
    }

    /// The training rows as one row-major buffer. Invariant: `data` is built with
    /// `from_shape_vec` (standard layout), so `as_slice` never fails.
    fn rows(&self) -> &[f64] {
        self.data.as_slice().unwrap_or(&[])
    }

    /// Class code of each training row (classification).
    pub fn train_codes(&self) -> Option<Vec<usize>> {
        let classes = self.classes.as_ref()?;
        Some(
            self.labels
                .iter()
                .map(|&l| classes.get(l).copied().unwrap_or(0))
                .collect(),
        )
    }

    /// Target of each training row (regression).
    pub fn train_values(&self) -> Option<&[f64]> {
        self.classes.is_none().then_some(self.values.as_slice())
    }

    /// The kd-tree, when the engine is [`SearchEngine::KdTree`].
    pub fn tree(&self) -> Option<&KdTree> {
        self.tree.as_ref()
    }

    /// K, weights, metric, engine and sizes.
    pub fn summary(&self) -> KnnSummary {
        KnnSummary {
            n_neighbors: self.config.n_neighbors,
            weights: self.config.weights,
            metric: self.config.metric,
            engine: self.engine,
            n_train: self.n_train(),
            n_features: self.n_features,
        }
    }

    /// Whether it is a classifier.
    pub fn is_classifier(&self) -> bool {
        self.classes.is_some()
    }

    /// `kneighbors(x)`: the K nearest training rows of every row of `x`, ascending,
    /// with scikit-learn's tie rule (see the module docs). Long searches run on up to
    /// 8 threads (each row is independent, so the result does not depend on it) and
    /// check `control` every 256 rows.
    pub fn kneighbors(
        &self,
        x: ArrayView2<'_, f64>,
        control: &TrainControl<'_>,
    ) -> Result<Neighbors, ModelError> {
        check_input(x, self.n_features)?;
        let q = x.as_standard_layout();
        let q = q.as_slice().ok_or_else(|| ModelError::Numerical {
            message: "matriz de consulta no contigua".to_owned(),
        })?;
        self.kneighbors_rows(q, control)
    }

    /// [`Knn::kneighbors`] of a row-major buffer (already validated).
    fn kneighbors_rows(
        &self,
        q: &[f64],
        control: &TrainControl<'_>,
    ) -> Result<Neighbors, ModelError> {
        let p = self.n_features;
        let k = self.config.k();
        let n = q.len().checked_div(p).unwrap_or(0);
        // No distance of the search may overflow (finding 6 of the F4 review).
        let (lo, hi) = search::bounding_box(self.rows(), p);
        let metric = self.effective_metric();
        if q.chunks_exact(p.max(1))
            .any(|pt| !(2.0 * search::max_rdist_to_box(pt, &lo, &hi, metric)).is_finite())
        {
            return Err(ModelError::NonFinite);
        }
        let mut distances = vec![0.0; n * k];
        let mut indices = vec![0usize; n * k];
        let threads = if n >= PARALLEL_MIN_ROWS {
            std::thread::available_parallelism()
                .map_or(1, |t| t.get())
                .min(MAX_THREADS)
                .min(n / (PARALLEL_MIN_ROWS / 4))
                .max(1)
        } else {
            1
        };
        // The strategy of scikit-learn's single call with all `n` rows.
        let on_y = self.tree.is_none() && search::brute_parallel_on_y(self.n_train(), n);
        if threads <= 1 {
            self.search_chunk(q, on_y, &mut distances, &mut indices, control)?;
        } else {
            let rows_per = n.div_ceil(threads);
            let results: Vec<Result<(), ModelError>> = std::thread::scope(|s| {
                let handles: Vec<_> = q
                    .chunks(rows_per * p)
                    .zip(distances.chunks_mut(rows_per * k))
                    .zip(indices.chunks_mut(rows_per * k))
                    .map(|((qc, dc), ic)| {
                        s.spawn(move || self.search_chunk(qc, on_y, dc, ic, control))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            Err(ModelError::Numerical {
                                message: "fallo en la búsqueda de vecinos".to_owned(),
                            })
                        })
                    })
                    .collect()
            });
            for r in results {
                r?;
            }
        }
        Ok(Neighbors {
            k,
            distances,
            indices,
        })
    }

    fn search_chunk(
        &self,
        q: &[f64],
        parallel_on_y: bool,
        distances: &mut [f64],
        indices: &mut [usize],
        control: &TrainControl<'_>,
    ) -> Result<(), ModelError> {
        let p = self.n_features;
        let k = self.config.k();
        let metric = self.effective_metric();
        let mut heap = NeighborHeap::new(k);
        let mut scratch = NeighborHeap::new(k);
        for (r, pt) in q.chunks_exact(p).enumerate() {
            if r % CANCEL_EVERY == 0 {
                control.check()?;
            }
            heap.reset();
            match &self.tree {
                Some(tree) => tree.query(pt, &mut heap),
                None => {
                    brute_query(self.rows(), p, pt, metric, &mut heap);
                    if parallel_on_y {
                        heap.repush(&mut scratch);
                    }
                }
            }
            heap.sort();
            for (dst, &v) in distances[r * k..(r + 1) * k].iter_mut().zip(&heap.values) {
                *dst = metric.to_distance(v);
            }
            indices[r * k..(r + 1) * k].copy_from_slice(&heap.indices);
        }
        Ok(())
    }

    /// Class index (into `classes()`) voted by one row: the mode, or the weighted
    /// mode with per-class pairwise sums.
    fn vote(&self, labels: &[usize], w: &[f64], scratch: &mut Vec<f64>) -> usize {
        let n_classes = self.classes.as_ref().map_or(0, Vec::len);
        match self.config.weights {
            KnnWeights::Uniform => {
                let mut counts = vec![0usize; n_classes];
                for &l in labels {
                    if let Some(c) = counts.get_mut(l) {
                        *c += 1;
                    }
                }
                let mut best = 0;
                for (c, &v) in counts.iter().enumerate() {
                    if v > counts[best] {
                        best = c;
                    }
                }
                best
            }
            KnnWeights::Distance => {
                // weighted_mode: classes in ascending order, strict `>`, starting at 0.
                let (mut best, mut best_score) = (0, 0.0);
                for c in 0..n_classes {
                    scratch.clear();
                    scratch.extend(
                        labels
                            .iter()
                            .zip(w)
                            .map(|(&l, &wi)| if l == c { wi } else { 0.0 }),
                    );
                    let s = pairwise_sum(scratch);
                    if s > best_score {
                        best = c;
                        best_score = s;
                    }
                }
                best
            }
        }
    }

    /// Class probabilities of one row (votes in neighbour order / their sum).
    fn row_proba(&self, labels: &[usize], w: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for (&l, &wi) in labels.iter().zip(w) {
            if let Some(o) = out.get_mut(l) {
                *o += wi;
            }
        }
        let mut norm = pairwise_sum(out);
        if norm == 0.0 {
            norm = 1.0;
        }
        for o in out.iter_mut() {
            *o /= norm;
        }
    }

    /// Regression value of one row.
    fn row_value(&self, idx: &[usize], w: &[f64], scratch: &mut Vec<f64>) -> f64 {
        scratch.clear();
        match self.config.weights {
            KnnWeights::Uniform => {
                scratch.extend(idx.iter().map(|&i| self.values[i]));
                pairwise_sum(scratch) / idx.len().max(1) as f64
            }
            KnnWeights::Distance => {
                scratch.extend(idx.iter().zip(w).map(|(&i, &wi)| self.values[i] * wi));
                let num = pairwise_sum(scratch);
                num / pairwise_sum(w)
            }
        }
    }

    /// `predict` from the neighbours.
    pub fn predict_from(&self, nb: &Neighbors) -> Prediction {
        let n = nb.n_rows();
        let mut w = Vec::with_capacity(nb.k);
        let mut scratch = Vec::with_capacity(nb.k);
        match &self.classes {
            Some(classes) => {
                let mut labels = Vec::with_capacity(nb.k);
                let out = (0..n)
                    .map(|r| {
                        row_weights(nb.row_distances(r), self.config.weights, &mut w);
                        labels.clear();
                        labels.extend(nb.row_indices(r).iter().map(|&i| self.labels[i]));
                        let c = self.vote(&labels, &w, &mut scratch);
                        classes.get(c).copied().unwrap_or(0)
                    })
                    .collect();
                Prediction::Classes(out)
            }
            None => Prediction::Values(
                (0..n)
                    .map(|r| {
                        row_weights(nb.row_distances(r), self.config.weights, &mut w);
                        self.row_value(nb.row_indices(r), &w, &mut scratch)
                    })
                    .collect(),
            ),
        }
    }

    /// `predict_proba` from the neighbours (`None` for regressors).
    pub fn proba_from(&self, nb: &Neighbors) -> Option<Array2<f64>> {
        let n_classes = self.classes.as_ref()?.len();
        let n = nb.n_rows();
        let mut out = Array2::<f64>::zeros((n, n_classes));
        let mut w = Vec::with_capacity(nb.k);
        let mut labels = Vec::with_capacity(nb.k);
        let mut row = vec![0.0; n_classes];
        for r in 0..n {
            row_weights(nb.row_distances(r), self.config.weights, &mut w);
            labels.clear();
            labels.extend(nb.row_indices(r).iter().map(|&i| self.labels[i]));
            self.row_proba(&labels, &w, &mut row);
            for (dst, &v) in out.row_mut(r).iter_mut().zip(&row) {
                *dst = v;
            }
        }
        Some(out)
    }

    /// C20: the prediction of one sample with its K neighbours (in the order of the
    /// search), their weights and, for classifiers, the probabilities.
    pub fn explain(&self, values: &[f64]) -> Result<KnnExplanation, ModelError> {
        let x = ArrayView2::from_shape((1, values.len()), values).map_err(|e| {
            ModelError::InvalidInput {
                reason: e.to_string(),
            }
        })?;
        let nb = self.kneighbors(x, &TrainControl::none())?;
        let mut e = self.explain_row(&nb, 0);
        e.tie_at_kth = self.kth_tie(values, nb.row_distances(0));
        Ok(e)
    }

    /// The tie at the K-th distance of the query `point` whose K distances are
    /// `dist` (ascending), if any: every training row is compared (O(n)).
    pub fn kth_tie(&self, point: &[f64], dist: &[f64]) -> Option<KthTie> {
        let kth = *dist.last()?;
        let metric = self.effective_metric();
        let p = self.n_features;
        if point.len() != p || p == 0 {
            return None;
        }
        let (mut less, mut tied) = (0, 0);
        for row in self.rows().chunks_exact(p) {
            let d = metric.to_distance(search::rdist(point, row, metric));
            if d < kth {
                less += 1;
            } else if d == kth {
                tied += 1;
            }
        }
        let slots = dist.len().saturating_sub(less);
        (tied > slots).then_some(KthTie {
            n_tied: tied,
            slots,
            distance: kth,
        })
    }

    /// [`Knn::explain`] of row `r` of a search.
    pub fn explain_row(&self, nb: &Neighbors, r: usize) -> KnnExplanation {
        let dist = nb.row_distances(r);
        let idx = nb.row_indices(r);
        let mut w = Vec::with_capacity(nb.k);
        let zero = row_weights(dist, self.config.weights, &mut w);
        let codes = self.classes.as_ref();
        let neighbors = idx
            .iter()
            .zip(dist)
            .zip(&w)
            .enumerate()
            .map(|(rank, ((&i, &d), &wi))| Neighbor {
                rank,
                index: i,
                distance: d,
                weight: wi,
                class: codes.and_then(|c| c.get(self.labels[i]).copied()),
                value: codes.is_none().then(|| self.values[i]),
            })
            .collect();
        let mut scratch = Vec::with_capacity(nb.k);
        let (class, value, proba) = match codes {
            Some(classes) => {
                let labels: Vec<usize> = idx.iter().map(|&i| self.labels[i]).collect();
                let c = self.vote(&labels, &w, &mut scratch);
                let mut p = vec![0.0; classes.len()];
                self.row_proba(&labels, &w, &mut p);
                (classes.get(c).copied(), None, Some(p))
            }
            None => (None, Some(self.row_value(idx, &w, &mut scratch)), None),
        };
        KnnExplanation {
            class,
            value,
            proba,
            neighbors,
            zero_distance_rule: zero,
            tie_at_kth: None,
        }
    }
}

impl Predictor for Knn {
    fn task(&self) -> TaskType {
        if self.classes.is_some() {
            TaskType::Classification
        } else {
            TaskType::Regression
        }
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn classes(&self) -> Option<&[usize]> {
        self.classes.as_deref()
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Prediction, ModelError> {
        self.predict_with_control(x, &TrainControl::none())
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Option<Result<Array2<f64>, ModelError>> {
        self.predict_proba_with_control(x, &TrainControl::none())
    }

    /// The search checks `ctl` every 256 rows ([`Knn::kneighbors`]).
    fn predict_with_control(
        &self,
        x: ArrayView2<'_, f64>,
        ctl: &TrainControl<'_>,
    ) -> Result<Prediction, ModelError> {
        let nb = self.kneighbors(x, ctl)?;
        Ok(self.predict_from(&nb))
    }

    fn predict_proba_with_control(
        &self,
        x: ArrayView2<'_, f64>,
        ctl: &TrainControl<'_>,
    ) -> Option<Result<Array2<f64>, ModelError>> {
        self.classes.as_ref()?;
        Some(self.kneighbors(x, ctl).map(|nb| {
            self.proba_from(&nb)
                .unwrap_or_else(|| Array2::zeros((0, 0)))
        }))
    }

    fn refit(
        &self,
        x: ArrayView2<'_, f64>,
        y: &Target,
        control: &TrainControl<'_>,
    ) -> Result<Box<dyn Predictor>, ModelError> {
        fit_boxed(self.config, x, y, control)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn classes(codes: &[u32]) -> Target {
        let k = codes.iter().max().map_or(0, |&m| m as usize + 1);
        Target::Classes {
            codes: codes.to_vec(),
            class_names: (0..k).map(|i| format!("c{i}")).collect(),
            class_values: (0..k).map(|i| i as f64).collect(),
        }
    }

    #[test]
    fn engine_rule() {
        assert_eq!(search_engine(105, 4, 5), SearchEngine::KdTree);
        assert_eq!(search_engine(398, 30, 5), SearchEngine::Brute);
        assert_eq!(search_engine(21, 2, 10), SearchEngine::Brute);
        assert_eq!(search_engine(21, 2, 9), SearchEngine::KdTree);
    }

    #[test]
    fn k_larger_than_train_is_rejected() {
        let x = array![[0.0], [1.0]];
        let err = fit(
            KnnConfig {
                n_neighbors: 3,
                ..KnnConfig::default()
            },
            x.view(),
            &classes(&[0, 1]),
            &TrainControl::none(),
        )
        .unwrap_err();
        assert_eq!(err, ModelError::TooFewSamples { k: 3, n_train: 2 });
        assert!(err.to_string().contains("K = 3"));
    }

    #[test]
    fn zero_distance_rule_like_sklearn() {
        // Query at a training point: only that neighbour counts with `distance`.
        let x = array![[0.0], [1.0], [1.1]];
        let cfg = KnnConfig {
            n_neighbors: 3,
            weights: KnnWeights::Distance,
            metric: KnnMetric::Euclidean,
        };
        let m = fit(cfg, x.view(), &classes(&[0, 1, 1]), &TrainControl::none())
            .unwrap_or_else(|e| panic!("{e}"));
        let e = m.explain(&[0.0]).unwrap_or_else(|e| panic!("{e}"));
        assert!(e.zero_distance_rule);
        assert_eq!(e.class, Some(0));
        assert_eq!(e.proba, Some(vec![1.0, 0.0]));
        let w: Vec<f64> = e.neighbors.iter().map(|n| n.weight).collect();
        assert_eq!(w, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn regression_mean_and_weighted() {
        let x = array![[0.0], [1.0], [3.0]];
        let y = Target::Values {
            values: vec![1.0, 2.0, 4.0],
        };
        let mut cfg = KnnConfig {
            n_neighbors: 2,
            ..KnnConfig::default()
        };
        let m = fit(cfg, x.view(), &y, &TrainControl::none()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            m.predict(array![[0.25]].view())
                .unwrap_or_else(|e| panic!("{e}")),
            Prediction::Values(vec![1.5])
        );
        cfg.weights = KnnWeights::Distance;
        let m = fit(cfg, x.view(), &y, &TrainControl::none()).unwrap_or_else(|e| panic!("{e}"));
        let Prediction::Values(v) = m
            .predict(array![[0.25]].view())
            .unwrap_or_else(|e| panic!("{e}"))
        else {
            panic!("values expected")
        };
        let (w0, w1) = (1.0 / 0.25, 1.0 / 0.75);
        assert!((v[0] - (w0 * 1.0 + w1 * 2.0) / (w0 + w1)).abs() < 1e-15);
    }

    #[test]
    fn overflowing_distances_are_rejected() {
        // Both engines: kd-tree (K = 1) and brute (K = 2 >= 4 // 2).
        let x = array![[1e160, 0.0], [-1e160, 0.0], [0.0, 0.0], [1.0, 0.0]];
        for k in [1, 2] {
            let cfg = KnnConfig {
                n_neighbors: k,
                ..KnnConfig::default()
            };
            let m = fit(
                cfg,
                x.view(),
                &classes(&[0, 1, 0, 1]),
                &TrainControl::none(),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(
                m.predict(array![[0.5, 0.0]].view()).unwrap_err(),
                ModelError::NonFinite,
                "{:?}",
                m.engine()
            );
        }
        // Normal data, an absurd query.
        let x = array![[0.0], [1.0], [2.0]];
        let m = fit(
            KnnConfig {
                n_neighbors: 1,
                ..KnnConfig::default()
            },
            x.view(),
            &classes(&[0, 1, 0]),
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            m.predict(array![[1e200]].view()).unwrap_err(),
            ModelError::NonFinite
        );
        assert!(m.predict(array![[1e150]].view()).is_ok());
    }

    #[test]
    fn predict_with_control_cancels() {
        let x = Array2::from_shape_fn((300, 2), |(i, j)| (i * (j + 1)) as f64);
        let y = classes(&(0..300).map(|i| (i % 2) as u32).collect::<Vec<_>>());
        let m = fit(KnnConfig::default(), x.view(), &y, &TrainControl::none())
            .unwrap_or_else(|e| panic!("{e}"));
        let flag = std::sync::atomic::AtomicBool::new(true);
        let ctl = TrainControl::none().with_cancel(&flag);
        assert_eq!(
            m.predict_with_control(x.view(), &ctl).unwrap_err(),
            ModelError::Cancelled
        );
        assert!(matches!(
            m.predict_proba_with_control(x.view(), &ctl),
            Some(Err(ModelError::Cancelled))
        ));
        assert_eq!(
            m.predict_with_control(x.view(), &TrainControl::none()),
            m.predict(x.view())
        );
    }

    #[test]
    fn kneighbors_cancels_and_parallel_equals_serial() {
        let n = 3000;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| ((i * 7 + j * 13) % 101) as f64 * 0.1);
        let y = classes(&(0..n).map(|i| (i % 3) as u32).collect::<Vec<_>>());
        let m = fit(KnnConfig::default(), x.view(), &y, &TrainControl::none())
            .unwrap_or_else(|e| panic!("{e}"));
        let all = m
            .kneighbors(x.view(), &TrainControl::none())
            .unwrap_or_else(|e| panic!("{e}"));
        let part = m
            .kneighbors(x.slice(ndarray::s![..100, ..]), &TrainControl::none())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(&all.indices[..500], &part.indices[..]);
        let flag = std::sync::atomic::AtomicBool::new(true);
        assert_eq!(
            m.kneighbors(x.view(), &TrainControl::none().with_cancel(&flag))
                .unwrap_err(),
            ModelError::Cancelled
        );
    }
}
