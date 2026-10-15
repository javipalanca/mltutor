//! C10: feature importance (`viz/features.py`), native and by permutation.
//!
//! * [`native`]: `feature_importances_` of a decision tree (mean decrease of
//!   impurity, normalized to sum 1).
//! * [`permutation`]: `sklearn.inspection.permutation_importance` (1.7.2) for **any**
//!   [`Predictor`] — the 🔍 KNN lesson (F4) reuses it — with scikit-learn's exact
//!   random numbers, so the matrix of importances matches to 1e-12.
//! * [`rows`]: the rows of the chart and table (ascending, ties by column).

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::dataset::{Target, TaskType};
use crate::metrics::r2_score;
use crate::model::{ModelError, Prediction, Predictor, TrainControl, TrainProgress};
use crate::sampling::RandomState;
use crate::stats::pairwise_sum;
use crate::tree::DecisionTree;

/// `compute_feature_importances(normalize=True)`: for every internal node, in node
/// order, `imp[f] += n·impurity − n_left·impurity_left − n_right·impurity_right`;
/// then divided by the samples of the root and normalized by the (pairwise) sum
/// when it is positive. A tree that is a single leaf gives all zeros.
pub fn native(tree: &DecisionTree) -> Vec<f64> {
    let nodes = tree.nodes();
    let mut imp = vec![0.0_f64; crate::model::Predictor::n_features(tree)];
    for node in nodes {
        if let Some(s) = node.split {
            let (l, r) = (&nodes[s.left], &nodes[s.right]);
            imp[s.feature] += node.n_samples as f64 * node.impurity
                - l.n_samples as f64 * l.impurity
                - r.n_samples as f64 * r.impurity;
        }
    }
    let root = nodes.first().map_or(1.0, |n| n.n_samples as f64);
    for v in &mut imp {
        *v /= root;
    }
    let total = pairwise_sum(&imp);
    if total > 0.0 {
        for v in &mut imp {
            *v /= total;
        }
    }
    imp
}

/// Options of [`permutation`] (the app: 10 repetitions, `random_state=42`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermutationOptions {
    pub n_repeats: usize,
    pub random_state: u32,
}

impl Default for PermutationOptions {
    fn default() -> Self {
        Self {
            n_repeats: 10,
            random_state: 42,
        }
    }
}

/// Result of [`permutation`] (`importances`, `importances_mean`, `importances_std`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PermutationImportance {
    /// The seed shared by every column (`RandomState(42).randint(2**31)`).
    pub seed: u32,
    /// Score on the unpermuted test set (accuracy or R²).
    pub baseline: f64,
    /// `n_features × n_repeats`: `baseline − score` after each shuffle.
    pub importances: Array2<f64>,
    pub mean: Vec<f64>,
    /// Population standard deviation (`np.std`, ddof 0).
    pub std: Vec<f64>,
}

/// `estimator.score` of a prediction: accuracy for classifiers, R² for regressors.
fn score_of(model: &dyn Predictor, pred: Prediction, y: &Target) -> Result<f64, ModelError> {
    match (model.task(), pred) {
        (TaskType::Classification, Prediction::Classes(pred)) => {
            let codes = y.class_codes().ok_or(ModelError::NotClassifier)?;
            let hits = pred.iter().zip(&codes).filter(|(a, b)| a == b).count();
            Ok(hits as f64 / codes.len().max(1) as f64)
        }
        (_, pred) => Ok(r2_score(&y.numeric(), &pred.to_f64())),
    }
}

/// `estimator.score(x, y)`; `control` reaches inside the prediction
/// ([`Predictor::predict_with_control`]: a KNN cancels between blocks of rows).
fn score(
    model: &dyn Predictor,
    x: ArrayView2<'_, f64>,
    y: &Target,
    control: &TrainControl<'_>,
) -> Result<f64, ModelError> {
    score_of(model, model.predict_with_control(x, control)?, y)
}

/// The data being permuted. Models with [`Predictor::predict_f32_rows`] (trees)
/// get one float32 buffer, converted once, where only the shuffled column is
/// rewritten; the others get an `f64` copy and the normal `predict`. Without it,
/// a CSV with many columns cost O(p·R·n·p): every repetition revalidated and
/// converted the whole matrix.
enum Work {
    /// Row-major `n × p` float32.
    F32(Vec<f32>),
    F64(Array2<f64>),
}

impl Work {
    fn new(model: &dyn Predictor, x: ArrayView2<'_, f64>) -> Self {
        // `x` already passed `predict` (the baseline), so `as f32` is what the
        // model itself does. The empty probe tells whether the fast path exists.
        if model.predict_f32_rows(&[]).is_some() {
            Work::F32(x.iter().map(|&v| v as f32).collect())
        } else {
            Work::F64(x.to_owned())
        }
    }

    /// `X[:, c] = values`.
    fn set_column(&mut self, c: usize, p: usize, values: &[f64]) {
        match self {
            Work::F32(buf) => {
                for (row, &v) in buf.chunks_exact_mut(p).zip(values) {
                    row[c] = v as f32;
                }
            }
            Work::F64(xp) => {
                for (dst, &v) in xp.column_mut(c).iter_mut().zip(values) {
                    *dst = v;
                }
            }
        }
    }

    fn score(
        &self,
        model: &dyn Predictor,
        y: &Target,
        control: &TrainControl<'_>,
    ) -> Result<f64, ModelError> {
        match self {
            Work::F32(buf) => {
                let pred = model.predict_f32_rows(buf).ok_or(ModelError::Numerical {
                    message: "el modelo ya no admite la predicción en float32".to_owned(),
                })??;
                score_of(model, pred, y)
            }
            Work::F64(xp) => score(model, xp.view(), y, control),
        }
    }
}

/// `permutation_importance(model, x_test, y_test, n_repeats, random_state)` of
/// scikit-learn 1.7.2, reproduced exactly (D39):
///
/// 1. `seed = RandomState(random_state).randint(2**31)` — **one** seed for every
///    column (scikit-learn ≥ 1.5; older versions drew one per column).
/// 2. For each column `c`: `rs = RandomState(seed)`, `idx = 0..n`, the original
///    data, and `n_repeats` times: `rs.shuffle(idx)` (cumulative: the same `idx`
///    keeps being shuffled) and `X[:, c] = X[idx, c]` (the already permuted column).
/// 3. `importances[c, r] = baseline − score`.
///
/// Works with any [`Predictor`] (trees here, KNN in F4). Cancellation is checked
/// before every repetition and, through [`Predictor::predict_with_control`], inside
/// the predictions of models that support it (KNN); progress reports one iteration
/// per column.
pub fn permutation(
    model: &dyn Predictor,
    x: ArrayView2<'_, f64>,
    y: &Target,
    options: PermutationOptions,
    control: &TrainControl<'_>,
) -> Result<PermutationImportance, ModelError> {
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
    if p != model.n_features() {
        return Err(ModelError::FeatureMismatch {
            expected: model.n_features(),
            found: p,
        });
    }
    control.check()?;
    let baseline = score(model, x, y, control)?;
    let seed_i = RandomState::new(options.random_state).randint(0, 1 << 31);
    let seed = u32::try_from(seed_i).unwrap_or(0);
    let mut importances = Array2::<f64>::zeros((p, options.n_repeats));
    let mut work = Work::new(model, x);
    let mut column = vec![0.0; n];
    for c in 0..p {
        control.report(TrainProgress {
            iteration: c as u64,
            max_iter: p as u64,
            loss: None,
            finished: false,
        });
        let mut rs = RandomState::new(seed);
        let mut idx: Vec<usize> = (0..n).collect();
        // The column as it is now (permuted `r` times).
        let mut current: Vec<f64> = x.column(c).to_vec();
        for r in 0..options.n_repeats {
            control.check()?;
            rs.shuffle(&mut idx);
            // `X[:, c] = X[idx, c]` on the already permuted column.
            for (dst, &i) in column.iter_mut().zip(&idx) {
                *dst = current[i];
            }
            std::mem::swap(&mut current, &mut column);
            work.set_column(c, p, &current);
            importances[[c, r]] = baseline - work.score(model, y, control)?;
        }
        // Restore the column: the next one starts from the original data.
        current.clear();
        current.extend(x.column(c).iter());
        work.set_column(c, p, &current);
    }
    control.report(TrainProgress {
        iteration: p as u64,
        max_iter: p as u64,
        loss: None,
        finished: true,
    });
    let k = options.n_repeats.max(1) as f64;
    let mut mean = Vec::with_capacity(p);
    let mut std = Vec::with_capacity(p);
    for row in importances.rows() {
        let v = row.to_vec();
        let m = pairwise_sum(&v) / k;
        let sq: Vec<f64> = v.iter().map(|x| (x - m) * (x - m)).collect();
        mean.push(m);
        std.push((pairwise_sum(&sq) / k).sqrt());
    }
    Ok(PermutationImportance {
        seed,
        baseline,
        importances,
        mean,
        std,
    })
}

/// Which importance a view shows (D38).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportanceMethod {
    /// "🌳 Importancia del modelo" (`feature_importances_`).
    Native,
    /// "🔀 Importancia por permutación".
    Permutation,
}

/// One bar of the chart / row of the table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportanceRow {
    pub feature_index: usize,
    pub name: String,
    pub importance: f64,
    /// Standard deviation (permutation only).
    pub std: Option<f64>,
}

/// The rows in **ascending** order of importance (the bar chart's order; the
/// table shows them reversed). Ties keep the column order, where pandas'
/// quicksort `sort_values` is arbitrary.
pub fn rows(names: &[String], importance: &[f64], std: Option<&[f64]>) -> Vec<ImportanceRow> {
    let mut out: Vec<ImportanceRow> = importance
        .iter()
        .enumerate()
        .map(|(i, &v)| ImportanceRow {
            feature_index: i,
            name: names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("feature_{i}")),
            importance: v,
            std: std.and_then(|s| s.get(i).copied()),
        })
        .collect();
    out.sort_by(|a, b| {
        a.importance
            .total_cmp(&b.importance)
            .then(a.feature_index.cmp(&b.feature_index))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_ascending_with_ties_by_index() {
        let names: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let r = rows(&names, &[0.5, 0.0, 0.0], None);
        let idx: Vec<usize> = r.iter().map(|r| r.feature_index).collect();
        assert_eq!(idx, vec![1, 2, 0]);
    }

    /// Says whether the control it receives carries the caller's cancel flag.
    #[derive(Debug)]
    struct Probe;

    impl Predictor for Probe {
        fn task(&self) -> TaskType {
            TaskType::Regression
        }
        fn n_features(&self) -> usize {
            1
        }
        fn classes(&self) -> Option<&[usize]> {
            None
        }
        fn predict(&self, _x: ArrayView2<'_, f64>) -> Result<Prediction, ModelError> {
            Err(ModelError::Numerical {
                message: "predict without control".to_owned(),
            })
        }
        fn predict_proba(
            &self,
            _x: ArrayView2<'_, f64>,
        ) -> Option<Result<Array2<f64>, ModelError>> {
            None
        }
        fn predict_with_control(
            &self,
            x: ArrayView2<'_, f64>,
            ctl: &TrainControl<'_>,
        ) -> Result<Prediction, ModelError> {
            if ctl.cancel_flag().is_none() {
                return self.predict(x);
            }
            Ok(Prediction::Values(x.column(0).to_vec()))
        }
        fn refit(
            &self,
            _x: ArrayView2<'_, f64>,
            _y: &Target,
            _control: &TrainControl<'_>,
        ) -> Result<Box<dyn Predictor>, ModelError> {
            Ok(Box::new(Probe))
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn the_control_reaches_the_predictions() {
        let x = ndarray::array![[1.0], [2.0], [3.0], [4.0]];
        let y = Target::Values {
            values: vec![1.0, 2.0, 3.0, 4.0],
        };
        let flag = std::sync::atomic::AtomicBool::new(false);
        let ctl = TrainControl::none().with_cancel(&flag);
        let pi = permutation(&Probe, x.view(), &y, PermutationOptions::default(), &ctl)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(pi.baseline, 1.0);
    }

    #[test]
    fn single_seed_like_sklearn() {
        assert_eq!(RandomState::new(42).randint(0, 1 << 31), 1_608_637_542);
    }
}
