//! 🌲 Decision trees (fase3-arboles.md): `DecisionTreeClassifier` and
//! `DecisionTreeRegressor` of scikit-learn 1.7.2 with **exact** parity (D36).
//!
//! * [`fit`] / [`DecisionTree::fit_classifier`] / [`DecisionTree::fit_regressor`]:
//!   the CART of [`cart`] (a port of `tools/cart_reference.py`), with every
//!   criterion of the lesson ([`TreeCriterion`]).
//! * [`DecisionTree`] implements [`Predictor`]: like scikit-learn, every input is
//!   converted to **float32** before it is compared with the (float64) thresholds,
//!   and `predict_proba` returns the class fractions of the leaf.
//! * [`DecisionTree::apply`] (leaf of each row) and [`DecisionTree::decision_path`]
//!   (C16, the path of one sample).
//! * [`export_text()`] (C14), [`plot`] (C13, the boxes of `plot_tree`) and
//!   [`layout`] (Buchheim's algorithm, as `sklearn/tree/_reingold_tilford.py`).
//!
//! Why our own implementation: `linfa-trees` only classifies, does not keep the
//! per-node statistics and draws its splits differently (MEMORY.md).

pub mod cart;
pub mod export_text;
pub mod layout;
pub mod plot;

use std::any::Any;

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

pub use cart::SplitTrace;
pub use export_text::{export_text, ExportTextOptions};

use crate::dataset::{Target, TaskType};
use crate::model::{
    argmax, check_input, check_n_classes, ModelError, Prediction, Predictor, TrainControl,
};
use cart::{BuildParams, ClassificationCriterion, RegressionCriterion, RegressionKind};

/// `random_state` of the Python app.
pub const DEFAULT_RANDOM_STATE: u32 = 42;

/// Classification or regression tree (`tree_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TreeTask {
    Classification,
    Regression,
}

impl TreeTask {
    /// The model task.
    pub fn task_type(self) -> TaskType {
        match self {
            TreeTask::Classification => TaskType::Classification,
            TreeTask::Regression => TaskType::Regression,
        }
    }
}

/// Split criterion, with scikit-learn's names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TreeCriterion {
    Gini,
    Entropy,
    SquaredError,
    FriedmanMse,
    AbsoluteError,
}

impl TreeCriterion {
    /// The `criterion` parameter of scikit-learn (`"gini"`, `"friedman_mse"`…).
    pub fn as_str(self) -> &'static str {
        match self {
            TreeCriterion::Gini => "gini",
            TreeCriterion::Entropy => "entropy",
            TreeCriterion::SquaredError => "squared_error",
            TreeCriterion::FriedmanMse => "friedman_mse",
            TreeCriterion::AbsoluteError => "absolute_error",
        }
    }

    /// Task the criterion belongs to.
    pub fn task(self) -> TreeTask {
        match self {
            TreeCriterion::Gini | TreeCriterion::Entropy => TreeTask::Classification,
            _ => TreeTask::Regression,
        }
    }

    /// Options of the "Criterio de División:" select, in the app's order.
    pub fn for_task(task: TreeTask) -> &'static [TreeCriterion] {
        match task {
            TreeTask::Classification => &[TreeCriterion::Gini, TreeCriterion::Entropy],
            TreeTask::Regression => &[
                TreeCriterion::SquaredError,
                TreeCriterion::FriedmanMse,
                TreeCriterion::AbsoluteError,
            ],
        }
    }
}

/// Hyperparameters: `DecisionTree{Classifier,Regressor}(max_depth, min_samples_split,
/// criterion, random_state)`; the rest are scikit-learn's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeConfig {
    pub task: TreeTask,
    pub criterion: TreeCriterion,
    /// `max_depth` (≥ 1; the app offers 1–10).
    pub max_depth: u32,
    /// `min_samples_split` (≥ 2; the app offers 2–10).
    pub min_samples_split: u32,
    /// `random_state` (42 in the app).
    pub random_state: u32,
}

impl Default for TreeConfig {
    /// The app's defaults: classification, gini, depth 3, 2 samples, seed 42.
    fn default() -> Self {
        Self {
            task: TreeTask::Classification,
            criterion: TreeCriterion::Gini,
            max_depth: 3,
            min_samples_split: 2,
            random_state: DEFAULT_RANDOM_STATE,
        }
    }
}

impl TreeConfig {
    /// A config with `random_state = 42`.
    pub fn new(
        task: TreeTask,
        criterion: TreeCriterion,
        max_depth: u32,
        min_samples_split: u32,
    ) -> Self {
        Self {
            task,
            criterion,
            max_depth,
            min_samples_split,
            random_state: DEFAULT_RANDOM_STATE,
        }
    }

    /// `Err(InvalidConfig)` for a criterion of the other task, `max_depth = 0`
    /// or `min_samples_split < 2` (scikit-learn's parameter validation).
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.criterion.task() != self.task {
            return Err(ModelError::InvalidConfig {
                reason: format!(
                    "el criterio '{}' no es válido para un árbol de {}",
                    self.criterion.as_str(),
                    match self.task {
                        TreeTask::Classification => "clasificación",
                        TreeTask::Regression => "regresión",
                    }
                ),
            });
        }
        if self.max_depth == 0 {
            return Err(ModelError::InvalidConfig {
                reason: "la profundidad máxima debe ser al menos 1".to_owned(),
            });
        }
        if self.min_samples_split < 2 {
            return Err(ModelError::InvalidConfig {
                reason: "las muestras mínimas para dividir deben ser al menos 2".to_owned(),
            });
        }
        Ok(())
    }
}

/// The test of an internal node: `x[feature] <= threshold` goes to `left`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TreeSplit {
    pub feature: usize,
    /// float64 threshold, compared with the **float32** value of the feature.
    pub threshold: f64,
    pub left: usize,
    pub right: usize,
    /// `missing_go_to_left` of scikit-learn (no effect without NaN; kept for the
    /// exported model).
    pub missing_go_to_left: bool,
}

/// One node of the tree, in scikit-learn's order (pre-order: node, left subtree,
/// right subtree).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeNode {
    /// `None` for leaves.
    pub split: Option<TreeSplit>,
    /// Gini, entropy, MSE or MAE of the node.
    pub impurity: f64,
    /// Training samples that reach the node (`n_node_samples`).
    pub n_samples: usize,
    /// `tree_.value`: class fractions (classification, in `classes()` order) or
    /// `[mean]` / `[median]` (regression).
    pub value: Vec<f64>,
    /// Class counts (classification); empty for regression.
    pub counts: Vec<u32>,
    /// Depth (root = 0).
    pub depth: usize,
}

impl TreeNode {
    /// Whether the node is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.split.is_none()
    }
}

/// Size of a fitted tree (contract `TreeSummary`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeSummary {
    pub node_count: usize,
    pub n_leaves: usize,
    /// Real depth (`tree_.max_depth`), not the hyperparameter.
    pub max_depth: usize,
    pub n_features: usize,
}

/// One step of [`DecisionPath`]: an internal node and the comparison.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PathStep {
    pub node: usize,
    pub feature: usize,
    /// The value as scikit-learn compares it: `float32(x)`, widened to f64.
    pub value: f64,
    pub threshold: f64,
    pub goes_left: bool,
}

/// C16: the nodes a sample goes through (`decision_path`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionPath {
    pub steps: Vec<PathStep>,
    pub leaf: usize,
    pub leaf_samples: usize,
    /// `value` of the leaf (class fractions or `[value]`).
    pub leaf_value: Vec<f64>,
    /// Predicted class code (classification).
    pub leaf_class: Option<usize>,
}

/// A fitted decision tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionTree {
    config: TreeConfig,
    n_features: usize,
    /// Class codes seen in training (`classes_`), sorted; `None` for regression.
    classes: Option<Vec<usize>>,
    nodes: Vec<TreeNode>,
    max_depth: usize,
    /// What the RNG drew at each split node (debugging and teaching).
    #[serde(skip)]
    traces: Vec<SplitTrace>,
}

/// Trains a tree on the training rows (the entry point of `train_model`).
///
/// * Classification needs a classification target (otherwise
///   [`ModelError::ContinuousTarget`], D37) with 2 to [`crate::model::MAX_CLASSES`]
///   classes in the training rows.
/// * Regression takes the numeric target, or the class values of a classification
///   target (📈 Regresión on Iris trains on the codes 0/1/2, as in Python).
pub fn fit(
    config: TreeConfig,
    x: ArrayView2<'_, f64>,
    y: &Target,
    control: &TrainControl<'_>,
) -> Result<DecisionTree, ModelError> {
    config.validate()?;
    match config.task {
        TreeTask::Classification => {
            let codes = y.class_codes().ok_or(ModelError::ContinuousTarget)?;
            let mut seen = codes.clone();
            seen.sort_unstable();
            seen.dedup();
            check_n_classes(seen.len())?;
            if !codes.is_empty() && seen.len() < 2 {
                return Err(ModelError::TooFewClasses { found: seen.len() });
            }
            DecisionTree::fit_classifier(config, x, &codes, control)
        }
        TreeTask::Regression => DecisionTree::fit_regressor(config, x, &y.numeric(), control),
    }
}

/// [`fit`] boxed as a [`Predictor`].
pub fn fit_boxed(
    config: TreeConfig,
    x: ArrayView2<'_, f64>,
    y: &Target,
    control: &TrainControl<'_>,
) -> Result<Box<dyn Predictor>, ModelError> {
    Ok(Box::new(fit(config, x, y, control)?))
}

/// `check_array(X, dtype=float32)`, column-major. Values outside the float32
/// range become infinite and are rejected, as scikit-learn does.
fn to_f32_columns(x: ArrayView2<'_, f64>) -> Result<Vec<f32>, ModelError> {
    let (n, p) = x.dim();
    let mut out = Vec::with_capacity(n * p);
    for col in x.columns() {
        for &v in col {
            let f = v as f32;
            if !f.is_finite() {
                return Err(ModelError::NonFinite);
            }
            out.push(f);
        }
    }
    Ok(out)
}

impl DecisionTree {
    /// `DecisionTreeClassifier(...).fit(x, codes)`; `codes` are class codes (any
    /// values: `classes_` = their sorted unique values). No limit on the number of
    /// classes here; [`fit`] applies the app's limits.
    pub fn fit_classifier(
        config: TreeConfig,
        x: ArrayView2<'_, f64>,
        codes: &[usize],
        control: &TrainControl<'_>,
    ) -> Result<Self, ModelError> {
        let mut classes = codes.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let encoded: Vec<usize> = codes
            .iter()
            .map(|c| classes.binary_search(c).unwrap_or(0))
            .collect();
        let entropy = config.criterion == TreeCriterion::Entropy;
        let crit = ClassificationCriterion::new(encoded, classes.len(), entropy);
        Self::fit_with(config, x, codes.len(), Some(classes), crit, control)
    }

    /// `DecisionTreeRegressor(...).fit(x, y)`.
    pub fn fit_regressor(
        config: TreeConfig,
        x: ArrayView2<'_, f64>,
        y: &[f64],
        control: &TrainControl<'_>,
    ) -> Result<Self, ModelError> {
        if y.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::NonFinite);
        }
        let kind = match config.criterion {
            TreeCriterion::FriedmanMse => RegressionKind::FriedmanMse,
            TreeCriterion::AbsoluteError => RegressionKind::AbsoluteError,
            _ => RegressionKind::SquaredError,
        };
        let crit = RegressionCriterion::new(y.to_vec(), kind);
        Self::fit_with(config, x, y.len(), None, crit, control)
    }

    fn fit_with<C: cart::Criterion>(
        config: TreeConfig,
        x: ArrayView2<'_, f64>,
        n_targets: usize,
        classes: Option<Vec<usize>>,
        crit: C,
        control: &TrainControl<'_>,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        let (n, p) = x.dim();
        if n == 0 {
            return Err(ModelError::Empty);
        }
        if n_targets != n {
            return Err(ModelError::LengthMismatch {
                rows: n,
                targets: n_targets,
            });
        }
        if p == 0 {
            return Err(ModelError::FeatureMismatch {
                expected: 1,
                found: 0,
            });
        }
        check_input(x, p)?;
        control.check()?;
        let x32 = to_f32_columns(x)?;
        let params = BuildParams {
            max_depth: config.max_depth as usize,
            min_samples_split: config.min_samples_split as usize,
            random_state: config.random_state,
        };
        let built = cart::build(&x32, n, p, crit, &params, control)?;
        let nodes = built
            .nodes
            .into_iter()
            .map(|raw| TreeNode {
                split: match (raw.feature, raw.left, raw.right) {
                    (Some(feature), Some(left), Some(right)) => Some(TreeSplit {
                        feature,
                        threshold: raw.threshold,
                        left,
                        right,
                        missing_go_to_left: raw.missing_go_to_left,
                    }),
                    _ => None,
                },
                impurity: raw.impurity,
                n_samples: raw.n_samples,
                value: raw.value,
                counts: raw.counts,
                depth: raw.depth,
            })
            .collect();
        Ok(Self {
            config,
            n_features: p,
            classes,
            nodes,
            max_depth: built.max_depth,
            traces: built.traces,
        })
    }

    /// The hyperparameters it was trained with.
    pub fn config(&self) -> &TreeConfig {
        &self.config
    }

    /// Every node (index = node id).
    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    /// `tree_.node_count`.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// `get_n_leaves()`.
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// `tree_.max_depth` (the real depth).
    pub fn depth(&self) -> usize {
        self.max_depth
    }

    /// Node count, leaves, depth and features.
    pub fn summary(&self) -> TreeSummary {
        TreeSummary {
            node_count: self.node_count(),
            n_leaves: self.n_leaves(),
            max_depth: self.max_depth,
            n_features: self.n_features,
        }
    }

    /// Whether it is a classifier.
    pub fn is_classifier(&self) -> bool {
        self.classes.is_some()
    }

    /// The RNG trace of every split node (`reference_impl.trace` of the fixtures).
    pub fn split_traces(&self) -> &[SplitTrace] {
        &self.traces
    }

    /// `tree_.children_left` (−1 for leaves).
    pub fn children_left(&self) -> Vec<i64> {
        self.nodes
            .iter()
            .map(|n| n.split.map_or(-1, |s| s.left as i64))
            .collect()
    }

    /// `tree_.children_right` (−1 for leaves).
    pub fn children_right(&self) -> Vec<i64> {
        self.nodes
            .iter()
            .map(|n| n.split.map_or(-1, |s| s.right as i64))
            .collect()
    }

    /// `tree_.feature` (−2 for leaves).
    pub fn feature(&self) -> Vec<i64> {
        self.nodes
            .iter()
            .map(|n| n.split.map_or(-2, |s| s.feature as i64))
            .collect()
    }

    /// `tree_.threshold` (−2.0 for leaves).
    pub fn threshold(&self) -> Vec<f64> {
        self.nodes
            .iter()
            .map(|n| n.split.map_or(-2.0, |s| s.threshold))
            .collect()
    }

    /// `tree_.missing_go_to_left` (0/1).
    pub fn missing_go_to_left(&self) -> Vec<u8> {
        self.nodes
            .iter()
            .map(|n| u8::from(n.split.is_some_and(|s| s.missing_go_to_left)))
            .collect()
    }

    /// Leaf reached by one row of float32 values.
    fn leaf_of(&self, row: impl Fn(usize) -> f32) -> usize {
        let mut node = 0;
        while let Some(s) = self.nodes.get(node).and_then(|n| n.split) {
            node = if f64::from(row(s.feature)) <= s.threshold {
                s.left
            } else {
                s.right
            };
        }
        node
    }

    /// Validates `x` and converts it to float32 rows (row-major).
    fn rows_f32(&self, x: ArrayView2<'_, f64>) -> Result<Vec<f32>, ModelError> {
        check_input(x, self.n_features)?;
        let mut out = Vec::with_capacity(x.len());
        for &v in x.iter() {
            let f = v as f32;
            if !f.is_finite() {
                return Err(ModelError::NonFinite);
            }
            out.push(f);
        }
        Ok(out)
    }

    /// `apply(X)`: the leaf of every row.
    pub fn apply(&self, x: ArrayView2<'_, f64>) -> Result<Vec<usize>, ModelError> {
        let p = self.n_features;
        let rows = self.rows_f32(x)?;
        Ok(rows
            .chunks_exact(p.max(1))
            .map(|r| self.leaf_of(|f| r[f]))
            .collect())
    }

    /// Class code of the majority class of a node (first maximum on ties).
    fn node_class(&self, node: usize) -> Option<usize> {
        let classes = self.classes.as_ref()?;
        let value = &self.nodes.get(node)?.value;
        classes.get(argmax(value.iter().copied())).copied()
    }

    /// C16: the path of one sample (`values` in column order).
    pub fn decision_path(&self, values: &[f64]) -> Result<DecisionPath, ModelError> {
        if values.len() != self.n_features {
            return Err(ModelError::FeatureMismatch {
                expected: self.n_features,
                found: values.len(),
            });
        }
        let mut row = Vec::with_capacity(values.len());
        for &v in values {
            let f = v as f32;
            if !v.is_finite() || !f.is_finite() {
                return Err(ModelError::NonFinite);
            }
            row.push(f);
        }
        let mut steps = Vec::new();
        let mut node = 0;
        while let Some(s) = self.nodes.get(node).and_then(|n| n.split) {
            let value = f64::from(row[s.feature]);
            let goes_left = value <= s.threshold;
            steps.push(PathStep {
                node,
                feature: s.feature,
                value,
                threshold: s.threshold,
                goes_left,
            });
            node = if goes_left { s.left } else { s.right };
        }
        let leaf = self.nodes.get(node).ok_or(ModelError::Numerical {
            message: "árbol vacío".to_owned(),
        })?;
        Ok(DecisionPath {
            steps,
            leaf: node,
            leaf_samples: leaf.n_samples,
            leaf_value: leaf.value.clone(),
            leaf_class: self.node_class(node),
        })
    }

    /// `feature_importances_` (normalized); see [`crate::importance::native`].
    pub fn feature_importances(&self) -> Vec<f64> {
        crate::importance::native(self)
    }
}

impl Predictor for DecisionTree {
    fn task(&self) -> TaskType {
        self.config.task.task_type()
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn classes(&self) -> Option<&[usize]> {
        self.classes.as_deref()
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Prediction, ModelError> {
        let leaves = self.apply(x)?;
        match &self.classes {
            Some(classes) => Ok(Prediction::Classes(
                leaves
                    .iter()
                    .map(|&l| {
                        let v = &self.nodes[l].value;
                        classes.get(argmax(v.iter().copied())).copied().unwrap_or(0)
                    })
                    .collect(),
            )),
            None => Ok(Prediction::Values(
                leaves
                    .iter()
                    .map(|&l| self.nodes[l].value.first().copied().unwrap_or(f64::NAN))
                    .collect(),
            )),
        }
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Option<Result<Array2<f64>, ModelError>> {
        let k = self.classes.as_ref()?.len();
        Some(self.apply(x).map(|leaves| {
            let mut out = Array2::<f64>::zeros((leaves.len(), k));
            for (mut row, &l) in out.rows_mut().into_iter().zip(&leaves) {
                for (dst, &v) in row.iter_mut().zip(&self.nodes[l].value) {
                    *dst = v;
                }
            }
            out
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

    #[test]
    fn config_validation() {
        let mut c = TreeConfig::default();
        assert!(c.validate().is_ok());
        c.criterion = TreeCriterion::SquaredError;
        assert!(matches!(
            c.validate(),
            Err(ModelError::InvalidConfig { .. })
        ));
        let c = TreeConfig {
            max_depth: 0,
            ..TreeConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn classification_on_values_is_rejected() {
        let x = array![[0.0], [1.0]];
        let y = Target::Values {
            values: vec![0.5, 1.5],
        };
        let err = fit(TreeConfig::default(), x.view(), &y, &TrainControl::none()).unwrap_err();
        assert_eq!(err, ModelError::ContinuousTarget);
    }

    #[test]
    fn simple_split_and_float32_comparison() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let t = DecisionTree::fit_classifier(
            TreeConfig::default(),
            x.view(),
            &[0, 0, 1, 1],
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t.node_count(), 3);
        assert_eq!(t.threshold()[0], 1.5);
        let p = t
            .predict(array![[1.5], [1.500_000_01], [1.500_001]].view())
            .unwrap_or_else(|e| panic!("{e}"));
        // 1.50000001 rounds to 1.5 in float32 → left, like scikit-learn.
        assert_eq!(p, Prediction::Classes(vec![0, 0, 1]));
        assert!(t.predict(array![[1e300]].view()).is_err());
    }

    #[test]
    fn cancelled() {
        let flag = std::sync::atomic::AtomicBool::new(true);
        let x = array![[0.0], [1.0]];
        let r = DecisionTree::fit_regressor(
            TreeConfig::new(TreeTask::Regression, TreeCriterion::SquaredError, 3, 2),
            x.view(),
            &[0.0, 1.0],
            &TrainControl::none().with_cancel(&flag),
        );
        assert_eq!(r.unwrap_err(), ModelError::Cancelled);
    }
}
