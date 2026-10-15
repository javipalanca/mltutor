//! Trained models of the [`super::FixtureFacade`] (fase2-regresion.md §5–§6).
//!
//! Training looks up `linear_models/<slug>_<linear|logistic>.json` and replays it: the
//! split, coefficients, predictions, probabilities, metrics and curves are the exact
//! values computed by scikit-learn. What the fixture does not store (histograms, the
//! grid of the boundary, the coefficient table…) is derived here from those values with
//! the rules of §3, simplified where exactness is not needed (the parity tests of
//! `mltutor-core` check the real implementation). It only knows the configurations of
//! the fixtures (`test_size` 0.3 and `max_iter` 1000).

use std::path::Path;
use std::sync::atomic::AtomicBool;

use serde_json::Value;

use super::frame::{Frame, Target};
use super::{floats, strings};
use crate::core_facade::{
    check_cancelled, ModelHandle, ModelSummary, ModelView, Progress, TrainSpec,
};
use crate::ipc::{
    to_u32, AbsQuantiles, ClassCurves, ClassHistogram, ClassMetrics, ClassProbability,
    ClassificationEvaluation, CodeTemplate, CoefficientRow, CoefficientsView, CommandError, Curve,
    DatasetId, DecisionGrid, Effect, Evaluation, ExportFormat, FeatureCoefficient, FormControl,
    FormControlKind, GridNotice, GridValues, Histogram, KdeCurve, Level, Matrix, ModelConfig,
    PredictionForm, PredictionResult, ProbabilityView, RegressionEvaluation, ResidualChecks,
    ResidualsView, RocData, SampleRow, SelectOption, TaskType, ThresholdRow, TrainNotice,
};

pub use crate::models::not_converged_message;

fn fixture_name(id: DatasetId, config: ModelConfig) -> String {
    match (id, config) {
        (DatasetId::Iris, ModelConfig::Linear) => "iris_linear_mismatch".to_owned(),
        (_, ModelConfig::Linear) => format!("{}_linear", id.slug()),
        (_, ModelConfig::Logistic { .. }) => format!("{}_logistic", id.slug()),
        // Trees and KNN are replayed by `super::tree` and `super::knn` (never reach here).
        (_, ModelConfig::Tree { .. }) => format!("{}_tree", id.slug()),
        (_, ModelConfig::Knn { .. }) => format!("{}_knn", id.slug()),
    }
}

fn unavailable(reason: impl std::fmt::Display) -> CommandError {
    CommandError::training(format!("FixtureFacade: {reason}"))
}

fn rows_of(value: &Value) -> Vec<Vec<f64>> {
    match value.as_array() {
        Some(items) if items.first().is_some_and(Value::is_array) => {
            items.iter().map(floats).collect()
        }
        Some(_) => vec![floats(value)],
        None => vec![floats(&Value::Array(vec![value.clone()]))],
    }
}

pub(super) fn indices(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()))
                .collect()
        })
        .unwrap_or_default()
}

/// x axis, y axis and the notices of the grid.
pub(super) type Axes = (Vec<f64>, Vec<f64>, Vec<GridNotice>);

/// A model replayed from a fixture. Trees (`super::tree`) reuse it for the views they
/// share with F2 (evaluation, curves, residuals, the form).
pub struct FixtureModel {
    config: ModelConfig,
    task: TaskType,
    dataset_task: TaskType,
    frame: Frame,
    train: Vec<usize>,
    test: Vec<usize>,
    coef: Vec<Vec<f64>>,
    intercept: Vec<f64>,
    y_test: Vec<f64>,
    y_pred: Vec<f64>,
    /// Rows of `predict_proba(X_test)` (logistic).
    proba: Option<Vec<Vec<f64>>>,
    class_names: Option<Vec<String>>,
    expected: Value,
    code: Value,
    notices: Vec<TrainNotice>,
}

/// `FixtureDataset::train`.
pub fn train(
    root: &Path,
    id: DatasetId,
    frame: &Frame,
    dataset_task: TaskType,
    spec: &TrainSpec,
    progress: &Progress<'_>,
    cancel: &AtomicBool,
) -> Result<ModelHandle, CommandError> {
    if spec.config.is_tree() {
        return super::tree::train(root, id, frame, dataset_task, spec, progress, cancel);
    }
    if spec.config.is_knn() {
        return super::knn::train(root, id, frame, dataset_task, spec, progress, cancel);
    }
    progress("Entrenando modelo...", Some(10.0));
    let name = fixture_name(id, spec.config);
    let read = |relative: &str| -> Result<Value, CommandError> {
        let path = root.join(relative);
        let text = std::fs::read_to_string(&path)
            .map_err(|_| unavailable(format!("no hay fixture {relative}")))?;
        serde_json::from_str(&text).map_err(|e| CommandError::internal(format!("{relative}: {e}")))
    };
    let fixture = read(&format!("linear_models/{name}.json"))?;
    let input = &fixture["input"];
    if (input["test_size"].as_f64().unwrap_or(0.3) - spec.test_size).abs() > 1e-9 {
        return Err(unavailable("solo hay fixtures con test_size = 0.3"));
    }
    if let ModelConfig::Logistic { max_iter } = spec.config {
        if input["max_iter"].as_u64() != Some(u64::from(max_iter)) {
            return Err(unavailable("solo hay fixtures con max_iter = 1000"));
        }
    }
    check_cancelled(cancel)?;
    let code = read("code/regression.json")?["expected"].take();
    let expected = fixture["expected"].clone();
    let model = &expected["model"];
    if model["coef"].is_null() {
        return Err(unavailable(format!("{name}.json no guarda el modelo")));
    }
    let task = spec.config.task_on(dataset_task);
    let class_names = expected["class_names"]
        .as_array()
        .map(|_| strings(&expected["class_names"]));
    let proba = expected["proba"]
        .as_array()
        .map(|rows| rows.iter().map(floats).collect());
    let mut notices = Vec::new();
    if let ModelConfig::Logistic { max_iter } = spec.config {
        if expected["converged_reference"]["app_model_converged"] == Value::Bool(false) {
            notices.push(TrainNotice::NotConverged {
                max_iter,
                message: not_converged_message(max_iter),
            });
        }
    }
    Ok(std::sync::Arc::new(FixtureModel {
        config: spec.config,
        task,
        dataset_task,
        frame: frame.clone(),
        train: indices(&expected["train_indices"]),
        test: indices(&expected["test_indices"]),
        coef: rows_of(&model["coef"]),
        intercept: intercepts(&model["intercept"]),
        y_test: floats(&expected["y_test"]),
        y_pred: floats(&expected["y_pred"]),
        proba,
        class_names,
        expected,
        code,
        notices,
    }))
}

impl FixtureModel {
    /// The shared part of a tree fixture (no coefficients).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn for_tree(
        config: ModelConfig,
        dataset_task: TaskType,
        frame: &Frame,
        expected: Value,
        code: Value,
        class_names: Option<Vec<String>>,
        notices: Vec<TrainNotice>,
    ) -> Self {
        let proba = expected["proba"]
            .as_array()
            .map(|rows| rows.iter().map(floats).collect());
        Self {
            config,
            task: config.task_on(dataset_task),
            dataset_task,
            frame: frame.clone(),
            train: indices(&expected["train_indices"]),
            test: indices(&expected["test_indices"]),
            coef: Vec::new(),
            intercept: Vec::new(),
            y_test: floats(&expected["y_test"]),
            y_pred: floats(&expected["y_pred"]),
            proba,
            class_names,
            expected,
            code,
            notices,
        }
    }

    pub(super) fn train_rows(&self) -> &[usize] {
        &self.train
    }

    pub(super) fn frame(&self) -> &Frame {
        &self.frame
    }

    pub(super) fn expected(&self) -> &Value {
        &self.expected
    }

    pub(super) fn class_names_ref(&self) -> Option<&[String]> {
        self.class_names.as_deref()
    }
}

/// `intercept_` is an array for the logistic models and a number for OLS.
fn intercepts(value: &Value) -> Vec<f64> {
    value.as_f64().map_or_else(|| floats(value), |v| vec![v])
}

/// Equal-width histogram over the range of `values` (`np.histogram(values, bins)`).
pub fn histogram(values: &[f64], bins: usize) -> Histogram {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || bins == 0 {
        return Histogram {
            edges: Vec::new(),
            counts: Vec::new(),
        };
    }
    let mut lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let mut hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (hi - lo).abs() < f64::EPSILON {
        lo -= 0.5;
        hi += 0.5;
    }
    let width = (hi - lo) / bins as f64;
    let edges: Vec<f64> = (0..=bins).map(|i| lo + width * i as f64).collect();
    let mut counts = vec![0_u32; bins];
    for v in finite {
        let bin = (((v - lo) / width) as usize).min(bins - 1);
        counts[bin] += 1;
    }
    Histogram { edges, counts }
}

/// numpy `bins="auto"`: the larger of Sturges and Freedman–Diaconis.
fn auto_bins(values: &[f64]) -> usize {
    let n = values.len();
    if n < 2 {
        return 1;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let q = |p: f64| {
        let pos = p * (n - 1) as f64;
        let (i, frac) = (pos.floor() as usize, pos - pos.floor());
        sorted[i] + (sorted[(i + 1).min(n - 1)] - sorted[i]) * frac
    };
    let range = sorted[n - 1] - sorted[0];
    let sturges = range / ((n as f64).log2() + 1.0);
    let iqr = q(0.75) - q(0.25);
    let fd = 2.0 * iqr / (n as f64).cbrt();
    let width = if fd > 0.0 { fd.min(sturges) } else { sturges };
    if width <= 0.0 {
        1
    } else {
        ((range / width).ceil() as usize).clamp(1, 1000)
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn std_ddof1(values: &[f64]) -> f64 {
    let m = mean(values);
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
}

/// `sns.histplot(kde=True)`: Scott bandwidth, cut 0, 200 points, scaled to counts.
fn errors_kde(errors: &[f64], hist: &Histogram) -> Option<KdeCurve> {
    let n = errors.len();
    let sd = std_ddof1(errors);
    if n < 2 || sd.is_nan() || sd <= 0.0 {
        return None;
    }
    let bw = sd * (n as f64).powf(-0.2);
    let lo = errors.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = errors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let bin = hist
        .edges
        .get(1)
        .zip(hist.edges.first())
        .map_or(1.0, |(b, a)| b - a);
    let norm = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * bw * n as f64);
    let density = (0..200)
        .map(|i| {
            let x = lo + (hi - lo) * i as f64 / 199.0;
            let d: f64 = errors
                .iter()
                .map(|e| (-0.5 * ((x - e) / bw).powi(2)).exp())
                .sum::<f64>()
                * norm;
            d * n as f64 * bin
        })
        .collect();
    Some(KdeCurve {
        level: "errores".into(),
        n: to_u32(n),
        bandwidth: Some(bw),
        grid_min: lo,
        grid_max: hi,
        density,
    })
}

/// pandas `Series.quantile` (linear).
fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = p * (sorted.len() - 1) as f64;
    let i = pos.floor() as usize;
    let j = (i + 1).min(sorted.len() - 1);
    sorted[i] + (sorted[j] - sorted[i]) * (pos - i as f64)
}

fn curve(x: &Value, y: &Value, thresholds: &Value) -> Curve {
    Curve {
        x: floats(x),
        y: floats(y),
        thresholds: thresholds.as_array().map(|items| {
            items
                .iter()
                .map(|t| t.as_f64().filter(|v| v.is_finite()))
                .collect()
        }),
    }
}

fn class_curves(index: u32, name: String, value: &Value) -> ClassCurves {
    let roc = &value["roc"];
    let pr = &value["pr"];
    ClassCurves {
        class_index: index,
        name,
        roc: curve(&roc["fpr"], &roc["tpr"], &roc["thresholds"]),
        auc: roc["auc"].as_f64().unwrap_or(f64::NAN),
        pr: curve(&pr["recall"], &pr["precision"], &pr["thresholds"]),
        ap: pr["average_precision"].as_f64().unwrap_or(f64::NAN),
    }
}

/// Text of the first notice of `views.<view>` that starts with `prefix`.
fn view_notice(views: &Value, view: &str, prefix: &str) -> Option<String> {
    fn walk(node: &Value, prefix: &str) -> Option<String> {
        if node["kind"] == "notice" {
            if let Some(text) = node["text"].as_str().filter(|t| t.starts_with(prefix)) {
                return Some(text.to_owned());
            }
        }
        node["children"]
            .as_array()?
            .iter()
            .find_map(|child| walk(child, prefix))
    }
    walk(&views[view], prefix)
}

impl FixtureModel {
    fn n_features(&self) -> usize {
        self.frame.feature_columns.len()
    }

    fn names(&self) -> Vec<String> {
        self.class_names.clone().unwrap_or_default()
    }

    fn class_name(&self, code: f64) -> String {
        let names = self.names();
        names
            .get(code.max(0.0) as usize)
            .cloned()
            .unwrap_or_else(|| format!("{code}"))
    }

    fn require_classifier(&self) -> Result<&[Vec<f64>], CommandError> {
        match (&self.proba, self.task) {
            (Some(proba), TaskType::Classification) => Ok(proba),
            _ => Err(CommandError::not_classifier()),
        }
    }

    /// Probabilities (logistic) or value (linear) for one row of features.
    fn scores(coef: &[Vec<f64>], intercept: &[f64], x: &[f64]) -> Vec<f64> {
        coef.iter()
            .zip(intercept)
            .map(|(row, b)| row.iter().zip(x).map(|(w, v)| w * v).sum::<f64>() + b)
            .collect()
    }

    fn probabilities(scores: &[f64]) -> Vec<f64> {
        if scores.len() == 1 {
            let p = 1.0 / (1.0 + (-scores[0]).exp());
            return vec![1.0 - p, p];
        }
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f64 = exp.iter().sum();
        exp.into_iter().map(|e| e / total).collect()
    }

    fn metrics(&self, name: &str, value: &Value) -> ClassMetrics {
        ClassMetrics {
            name: name.to_owned(),
            precision: value["precision"].as_f64().unwrap_or(0.0),
            recall: value["recall"].as_f64().unwrap_or(0.0),
            f1: value["f1-score"].as_f64().unwrap_or(0.0),
            support: value["support"].as_f64().map_or(0, |s| s as u32),
        }
    }

    fn classification(&self) -> ClassificationEvaluation {
        let evaluation = &self.expected["evaluation"];
        let report = &evaluation["classification_report"];
        let names = strings(&evaluation["target_names"]);
        let k = names.len();
        let confusion: Vec<f64> = evaluation["confusion_matrix"]
            .as_array()
            .map(|rows| rows.iter().flat_map(floats).collect())
            .unwrap_or_default();
        let mut correct = Vec::new();
        let mut incorrect = Vec::new();
        for (position, (real, predicted)) in self.y_test.iter().zip(&self.y_pred).enumerate() {
            let ok = (real - predicted).abs() < 0.5;
            let row = SampleRow {
                position: to_u32(position),
                real: self.class_name(*real),
                predicted: self.class_name(*predicted),
                correct: ok,
            };
            let group = if ok { &mut correct } else { &mut incorrect };
            if group.len() < 10 {
                group.push(row);
            }
        }
        ClassificationEvaluation {
            accuracy: evaluation["accuracy"].as_f64().unwrap_or(0.0),
            per_class: names
                .iter()
                .map(|n| self.metrics(n, &report[n.as_str()]))
                .collect(),
            macro_avg: self.metrics("macro avg", &report["macro avg"]),
            weighted_avg: self.metrics("weighted avg", &report["weighted avg"]),
            confusion: Matrix::from_f64(k, k, confusion),
            correct_head: correct,
            incorrect_head: incorrect,
            interpretation_md: view_notice(
                &self.expected["views"],
                "evaluation",
                "**Resumen del Modelo:**",
            )
            .unwrap_or_default(),
            warnings: strings(&evaluation["classification_report_warnings"]),
        }
    }

    fn regression(&self) -> RegressionEvaluation {
        let evaluation = &self.expected["evaluation"];
        let errors: Vec<f64> = self
            .y_test
            .iter()
            .zip(&self.y_pred)
            .map(|(y, p)| y - p)
            .collect();
        let hist = histogram(&errors, auto_bins(&errors));
        let get = |key: &str| evaluation[key].as_f64().unwrap_or(f64::NAN);
        RegressionEvaluation {
            mse: get("mse"),
            rmse: get("rmse"),
            mae: get("mae"),
            r2: get("r2"),
            y_true: self.y_test.clone(),
            y_pred: self.y_pred.clone(),
            errors_kde: errors_kde(&errors, &hist),
            errors_hist: hist,
            interpretation_md: view_notice(
                &self.expected["views"],
                "evaluation",
                "**Resumen del Modelo:**",
            )
            .unwrap_or_default(),
        }
    }

    /// Axes of §C8 (`h = 0.02`, ±1, resized above 50 000 points).
    pub(super) fn axes(xs: &[f64], ys: &[f64]) -> Result<Axes, CommandError> {
        let range = |v: &[f64]| {
            let lo = v.iter().copied().fold(f64::INFINITY, f64::min) - 1.0;
            let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 1.0;
            (lo, hi)
        };
        let (x0, x1) = range(xs);
        let (y0, y1) = range(ys);
        let (rx, ry) = (x1 - x0, y1 - y0);
        let mut h: f64 = 0.02;
        if rx.max(ry) > 100.0 {
            h = h.max(rx.max(ry) / 200.0);
        }
        h = h.min(rx / 10.0).min(ry / 10.0);
        let arange = |lo: f64, hi: f64| -> Vec<f64> {
            let n = ((hi - lo) / h).ceil().max(0.0) as usize;
            let delta = (lo + h) - lo;
            let axis: Vec<f64> = (0..n).map(|i| lo + i as f64 * delta).collect();
            if axis.len() < 2 {
                linspace(lo, hi, 10)
            } else {
                axis
            }
        };
        let mut ax = arange(x0, x1);
        let mut ay = arange(y0, y1);
        let mut notices = vec![GridNotice::Created {
            rows: to_u32(ay.len()),
            cols: to_u32(ax.len()),
        }];
        let points = (ax.len() * ay.len()) as u64;
        if points > 50_000 {
            notices.push(GridNotice::TooLarge { points });
            ax = linspace(x0, x1, ax.len().min(200));
            ay = linspace(y0, y1, ay.len().min(200));
            notices.push(GridNotice::Resized {
                rows: to_u32(ay.len()),
                cols: to_u32(ax.len()),
            });
        }
        if ax.len() < 2 || ay.len() < 2 {
            return Err(CommandError::grid_too_small(ay.len(), ax.len()));
        }
        Ok((ax, ay, notices))
    }
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        return vec![lo; n];
    }
    let step = (hi - lo) / (n - 1) as f64;
    (0..n)
        .map(|i| if i == n - 1 { hi } else { lo + i as f64 * step })
        .collect()
}

impl FixtureModel {
    /// The fixtures train on splits that contain every class.
    fn model_classes(&self) -> Option<Vec<usize>> {
        (self.task == TaskType::Classification)
            .then(|| (0..self.class_names.as_ref().map_or(0, Vec::len)).collect())
    }
}

impl ModelView for FixtureModel {
    fn summary(&self) -> ModelSummary {
        ModelSummary {
            model_task: self.task,
            dataset_task: self.dataset_task,
            n_train: self.train.len(),
            n_test: self.test.len(),
            feature_columns: self.frame.feature_columns.clone(),
            feature_names: self.frame.feature_display.clone(),
            class_names: self.class_names.clone(),
            model_classes: self.model_classes(),
            notices: self.notices.clone(),
            tree: None,
            knn: None,
        }
    }

    fn evaluate(&self) -> Result<Evaluation, CommandError> {
        Ok(match self.task {
            TaskType::Classification => Evaluation::Classification(self.classification()),
            TaskType::Regression => Evaluation::Regression(self.regression()),
        })
    }

    fn roc_curves(&self) -> Result<RocData, CommandError> {
        self.require_classifier()?;
        let curves = &self.expected["evaluation"]["curves"];
        let names = self.names();
        if let Some(binary) = curves.get("binary") {
            let roc = &binary["roc"];
            let pr = &binary["pr"];
            return Ok(RocData::Binary {
                positive_class: names.get(1).cloned().unwrap_or_else(|| "1".into()),
                roc: curve(&roc["fpr"], &roc["tpr"], &roc["thresholds"]),
                auc: roc["auc"].as_f64().unwrap_or(f64::NAN),
                pr: curve(&pr["recall"], &pr["precision"], &pr["thresholds"]),
                ap: pr["average_precision"].as_f64().unwrap_or(f64::NAN),
                baseline: binary["baseline"].as_f64().unwrap_or(f64::NAN),
            });
        }
        let multi = &curves["multiclass"];
        let per_class = multi["per_class"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .map(|c| {
                        let index = c["class_index"].as_u64().unwrap_or(0) as u32;
                        let name = c["class_name"].as_str().unwrap_or_default().to_owned();
                        class_curves(index, name, c)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let macro_ = &multi["macro"];
        Ok(RocData::Multiclass {
            per_class,
            micro: class_curves(0, "micro".into(), &multi["micro"]),
            macro_roc: curve(&macro_["roc"]["fpr"], &macro_["roc"]["tpr"], &Value::Null),
            macro_auc: macro_["roc"]["auc"].as_f64().unwrap_or(f64::NAN),
            macro_ap: macro_["average_precision"].as_f64().unwrap_or(f64::NAN),
            weighted_auc: curves["not_in_app"]["roc_auc_ovr_weighted"].as_f64(),
            absent_classes: Vec::new(),
        })
    }

    fn probability_view(&self) -> Result<ProbabilityView, CommandError> {
        let proba = self.require_classifier()?;
        let names = self.names();
        if names.len() == 2 {
            let p1: Vec<f64> = proba
                .iter()
                .map(|r| r.get(1).copied().unwrap_or(0.0))
                .collect();
            let per_real_class = (0..2)
                .map(|class| {
                    let group: Vec<f64> = p1
                        .iter()
                        .zip(&self.y_test)
                        .filter(|(_, y)| (**y - class as f64).abs() < 0.5)
                        .map(|(p, _)| *p)
                        .collect();
                    ClassHistogram {
                        class_index: class,
                        name: names.get(class as usize).cloned().unwrap_or_default(),
                        histogram: histogram(&group, 20),
                    }
                })
                .collect();
            let ta = &self.expected["evaluation"]["threshold_analysis"];
            let thresholds: Vec<ThresholdRow> = ta["rows"]
                .as_array()
                .map(|rows| {
                    rows.iter()
                        .filter(|r| r["included"] != Value::Bool(false))
                        .map(|r| ThresholdRow {
                            threshold: r["threshold"].as_f64().unwrap_or(0.0),
                            accuracy: r["accuracy"].as_f64().unwrap_or(0.0),
                            precision: r["precision"].as_f64().unwrap_or(0.0),
                            recall: r["recall"].as_f64().unwrap_or(0.0),
                            f1: r["f1"].as_f64().unwrap_or(0.0),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let shown = thresholds
                .iter()
                .map(|r| r.threshold)
                .filter(|t| [0.3, 0.5, 0.7].iter().any(|s| (t - s).abs() < 1e-9))
                .collect();
            return Ok(ProbabilityView::Binary {
                per_real_class,
                uncertain: p1.iter().filter(|p| **p > 0.3 && **p < 0.7).count() as u32,
                confident: p1.iter().filter(|p| **p < 0.2 || **p > 0.8).count() as u32,
                n: to_u32(p1.len()),
                best_f1_threshold: ta["best_f1_threshold"].as_f64(),
                shown_thresholds: shown,
                thresholds,
            });
        }
        Ok(ProbabilityView::Multiclass {
            per_class: names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let column: Vec<f64> = proba
                        .iter()
                        .map(|r| r.get(i).copied().unwrap_or(0.0))
                        .collect();
                    ClassHistogram {
                        class_index: to_u32(i),
                        name: name.clone(),
                        histogram: histogram(&column, 20),
                    }
                })
                .collect(),
        })
    }

    fn residuals_view(&self) -> Result<ResidualsView, CommandError> {
        if self.task != TaskType::Regression {
            return Err(CommandError::not_regressor());
        }
        let evaluation = &self.expected["evaluation"];
        let r = &evaluation["residuals"];
        let residuals: Vec<f64> = self
            .y_test
            .iter()
            .zip(&self.y_pred)
            .map(|(y, p)| y - p)
            .collect();
        let mean = r["mean"].as_f64().unwrap_or_else(|| self::mean(&residuals));
        let std = r["std_ddof1"]
            .as_f64()
            .unwrap_or_else(|| std_ddof1(&residuals));
        let skew = r["skew_biased"].as_f64().unwrap_or(0.0);
        let linregress_r = r["linregress_r"].as_f64().filter(|_| residuals.len() > 10);
        let hist = Histogram {
            edges: floats(&r["hist20"]["edges"]),
            counts: super::counts(&r["hist20"]["counts"]),
        };
        Ok(ResidualsView {
            y_true: self.y_test.clone(),
            y_pred: self.y_pred.clone(),
            r2: evaluation["r2"].as_f64().unwrap_or(f64::NAN),
            mean,
            std,
            skew,
            linregress_r,
            hist,
            checks: ResidualChecks {
                mean_ok: mean.abs() < 0.1 * std,
                corr_ok: linregress_r.map(|r| r.abs() < 0.1),
                normal_ok: skew.abs() < 1.0,
            },
        })
    }

    fn decision_grid(
        &self,
        x: usize,
        y: usize,
        progress: &Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<DecisionGrid, CommandError> {
        let models = &self.expected["two_feature_models"];
        let fitted = ["default", "pair"]
            .iter()
            .map(|key| &models[*key])
            .find(|m| indices(&m["features"]) == [x, y])
            .ok_or_else(|| {
                CommandError::invalid_selection(
                    "FixtureFacade: este par de características no está en el fixture.",
                )
            })?;
        let coef = rows_of(&fitted["coef"]);
        let intercept = intercepts(&fitted["intercept"]);
        let xs: Vec<f64> = self
            .train
            .iter()
            .map(|&i| self.frame.features[x][i])
            .collect();
        let ys: Vec<f64> = self
            .train
            .iter()
            .map(|&i| self.frame.features[y][i])
            .collect();
        let (ax, ay, notices) = Self::axes(&xs, &ys)?;
        progress(
            &format!(
                "Calculando predicciones en {} puntos...",
                ax.len() * ay.len()
            ),
            Some(50.0),
        );
        let mut classes = Vec::new();
        let mut values = Vec::new();
        for gy in &ay {
            check_cancelled(cancel)?;
            for gx in &ax {
                let scores = Self::scores(&coef, &intercept, &[*gx, *gy]);
                if self.task == TaskType::Classification {
                    let p = Self::probabilities(&scores);
                    let best = p
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map_or(0, |(i, _)| i);
                    classes.push(u16::try_from(best).unwrap_or(0));
                } else {
                    values.push(scores.first().copied().unwrap_or(f64::NAN));
                }
            }
        }
        let values = if self.task == TaskType::Classification {
            GridValues::Classes { z: classes }
        } else {
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            GridValues::Values {
                z: values,
                min,
                max,
            }
        };
        let target = self.frame.target.as_f64();
        Ok(DecisionGrid {
            x: ax,
            y: ay,
            values,
            notices,
            points: Matrix::from_f64(
                xs.len(),
                2,
                xs.iter()
                    .zip(&ys)
                    .flat_map(|(a, b)| [*a, *b])
                    .collect::<Vec<_>>(),
            ),
            point_target: self.train.iter().map(|&i| target[i]).collect(),
            point_codes: match &self.frame.target {
                Target::Classes { codes, .. } if self.task == TaskType::Classification => {
                    Some(self.train.iter().map(|&i| u32::from(codes[i])).collect())
                }
                _ => None,
            },
            feature_names: [
                self.frame.feature_display[x].clone(),
                self.frame.feature_display[y].clone(),
            ],
            class_names: self
                .class_names
                .clone()
                .filter(|_| self.task == TaskType::Classification),
            show_2d_warning: self.n_features() > 2,
        })
    }

    fn coefficients(&self, class_index: Option<usize>) -> Result<CoefficientsView, CommandError> {
        let logistic = matches!(self.config, ModelConfig::Logistic { .. });
        let row = class_index
            .unwrap_or(0)
            .min(self.coef.len().saturating_sub(1));
        let coef = self.coef.get(row).cloned().unwrap_or_default();
        let mut sorted_abs: Vec<f64> = coef.iter().map(|c| c.abs()).collect();
        sorted_abs.sort_by(f64::total_cmp);
        let quantiles = AbsQuantiles {
            q25: quantile(&sorted_abs, 0.25),
            q60: quantile(&sorted_abs, 0.6),
            q75: quantile(&sorted_abs, 0.75),
            q80: quantile(&sorted_abs, 0.8),
            q95: quantile(&sorted_abs, 0.95),
        };
        let mut rows: Vec<CoefficientRow> = coef
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let odds = c.exp();
                let intensity = if logistic {
                    match (c > 0.0, odds) {
                        (true, o) if o > 3.0 => Level::High,
                        (true, o) if o > 1.5 => Level::Medium,
                        (false, o) if o < 0.33 => Level::High,
                        (false, o) if o < 0.67 => Level::Medium,
                        _ => Level::Low,
                    }
                } else if c.abs() > quantiles.q80 {
                    Level::High
                } else if c.abs() > quantiles.q60 {
                    Level::Medium
                } else {
                    Level::Low
                };
                CoefficientRow {
                    feature: self.frame.feature_display[i].clone(),
                    feature_position: to_u32(i),
                    coef: c,
                    abs: c.abs(),
                    odds_ratio: logistic.then_some(odds),
                    change_pct: logistic.then_some((odds - 1.0) * 100.0),
                    effect: if c > 0.0 {
                        Effect::Positive
                    } else {
                        Effect::Negative
                    },
                    importance: (!logistic).then(|| {
                        if c.abs() > quantiles.q75 {
                            Level::High
                        } else if c.abs() > quantiles.q25 {
                            Level::Medium
                        } else {
                            Level::Low
                        }
                    }),
                    strength: logistic.then(|| match c.abs() {
                        a if a > 2.0 => Level::VeryHigh,
                        a if a > 1.0 => Level::High,
                        a if a > 0.5 => Level::Medium,
                        _ => Level::Low,
                    }),
                    intensity,
                }
            })
            .collect();
        rows.sort_by(|a, b| b.abs.total_cmp(&a.abs));
        let extreme = rows
            .iter()
            .filter(|r| r.abs > quantiles.q95)
            .map(|r| r.feature.clone())
            .collect();
        Ok(CoefficientsView {
            rows,
            class_index: (logistic && self.coef.len() > 1).then_some(to_u32(row)),
            model_classes: self
                .model_classes()
                .map(|c| c.into_iter().map(to_u32).collect()),
            n_rows: to_u32(self.coef.len()),
            intercept: self.intercept.get(row).copied().unwrap_or(0.0),
            intercept_odds: (logistic && self.coef.len() == 1)
                .then(|| self.intercept.first().copied().unwrap_or(0.0).exp()),
            quantiles,
            extreme,
        })
    }

    fn prediction_form(&self) -> Result<PredictionForm, CommandError> {
        let controls = self.expected["prediction_form"]["controls"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let form = controls
            .iter()
            .map(|c| {
                let index = c["index"].as_u64().unwrap_or(0) as usize;
                let label = c["label"].as_str().unwrap_or_default();
                let (name, description) = match label
                    .strip_prefix("**")
                    .and_then(|l| l.split_once("**\n\n*"))
                {
                    Some((name, rest)) => {
                        (name.to_owned(), Some(rest.trim_end_matches('*').to_owned()))
                    }
                    None => (label.to_owned(), None),
                };
                let control = match c["kind"].as_str() {
                    Some("checkbox") => FormControlKind::Checkbox {
                        default: c["value"].as_bool().unwrap_or(false),
                    },
                    Some("select") => {
                        let display = strings(&c["display"]);
                        let mut unique: Vec<f64> = self
                            .train
                            .iter()
                            .map(|&i| self.frame.features[index][i])
                            .collect();
                        unique.sort_by(f64::total_cmp);
                        unique.dedup();
                        let numeric = floats(&c["options"]);
                        let options: Vec<SelectOption> = display
                            .iter()
                            .enumerate()
                            .map(|(k, label)| SelectOption {
                                label: label.clone(),
                                code: numeric
                                    .get(k)
                                    .copied()
                                    .filter(|v| v.is_finite())
                                    .or_else(|| unique.get(k).copied())
                                    .unwrap_or(k as f64),
                            })
                            .collect();
                        let value = c["value"].to_string();
                        let default_index = strings(&c["options"])
                            .iter()
                            .position(|o| *o == value.trim_matches('"'))
                            .unwrap_or(display.len() / 2);
                        FormControlKind::Select {
                            options,
                            default_index: to_u32(default_index),
                        }
                    }
                    _ => FormControlKind::Slider {
                        min: c["min_value"].as_f64().unwrap_or(0.0),
                        max: c["max_value"].as_f64().unwrap_or(1.0),
                        step: c["step"].as_f64().unwrap_or(0.1),
                        default: c["value"].as_f64().unwrap_or(0.0),
                    },
                };
                FormControl {
                    index: to_u32(index),
                    column: self
                        .frame
                        .feature_columns
                        .get(index)
                        .cloned()
                        .unwrap_or_default(),
                    label: name,
                    description,
                    control,
                }
            })
            .collect();
        Ok(PredictionForm { controls: form })
    }

    fn predict_one(&self, values: &[f64]) -> Result<PredictionResult, CommandError> {
        let scores = Self::scores(&self.coef, &self.intercept, values);
        if matches!(self.config, ModelConfig::Linear) {
            let coef = self.coef.first().cloned().unwrap_or_default();
            return Ok(PredictionResult::Value {
                value: scores.first().copied().unwrap_or(f64::NAN),
                coefficients: Some(
                    self.frame
                        .feature_display
                        .iter()
                        .zip(coef)
                        .map(|(f, c)| FeatureCoefficient {
                            feature: f.clone(),
                            coef: c,
                        })
                        .collect(),
                ),
                path: None,
                neighbors: None,
            });
        }
        let p = Self::probabilities(&scores);
        let best = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i);
        let names = self.names();
        Ok(PredictionResult::Class {
            index: to_u32(best),
            name: names.get(best).cloned().unwrap_or_else(|| best.to_string()),
            proba: Some(
                p.iter()
                    .enumerate()
                    .map(|(i, prob)| ClassProbability {
                        name: names.get(i).cloned().unwrap_or_else(|| i.to_string()),
                        probability: *prob,
                    })
                    .collect(),
            ),
            path: None,
            neighbors: None,
        })
    }

    fn export(&self, format: ExportFormat) -> Result<Vec<u8>, CommandError> {
        match format {
            ExportFormat::Onnx => Err(CommandError::export(
                "FixtureFacade no genera ONNX (usa mltutor-export)",
            )),
            // Not a ZIP: the fixture facade only writes `modelo.json` for development.
            ExportFormat::PythonPackage => {
                serde_json::to_vec_pretty(&self.expected["export_package"]["model_json"])
                    .map_err(CommandError::export)
            }
            ExportFormat::TreeRules | ExportFormat::TreePython => Err(CommandError::not_tree()),
        }
    }

    fn code(
        &self,
        template: CodeTemplate,
        _pair: Option<(usize, usize)>,
    ) -> Result<String, CommandError> {
        let key = format!("{}.py.j2", template.file_stem());
        self.code[key.as_str()]["text"]
            .as_str()
            .map(|t| t.trim_start_matches('\n').to_owned())
            .ok_or_else(|| CommandError::internal(format!("falta la plantilla {key}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histograms_cover_the_range() {
        let h = histogram(&[0.0, 0.5, 1.0], 2);
        assert_eq!(h.edges, vec![0.0, 0.5, 1.0]);
        assert_eq!(h.counts, vec![1, 2]);
        let flat = histogram(&[2.0, 2.0], 4);
        assert_eq!(flat.counts.iter().sum::<u32>(), 2);
    }

    #[test]
    fn quantiles_are_linear() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile(&v, 0.25) - 1.75).abs() < 1e-12);
        assert!((quantile(&v, 0.95) - 3.85).abs() < 1e-12);
    }

    #[test]
    fn probabilities_sum_to_one() {
        let p = FixtureModel::probabilities(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let b = FixtureModel::probabilities(&[0.0]);
        assert_eq!(b, vec![0.5, 0.5]);
    }
}

#[cfg(test)]
mod view_tests {
    use super::*;
    use crate::core_facade::CoreFacade;
    use crate::fixture_facade::FixtureFacade;

    fn trained(id: DatasetId, config: ModelConfig) -> ModelHandle {
        let dataset = FixtureFacade::from_checkout().load_builtin(id).unwrap();
        let spec = TrainSpec {
            config,
            test_size: 0.3,
            dataset_label: String::new(),
        };
        dataset
            .train(&spec, &|_, _| {}, &AtomicBool::new(false))
            .unwrap()
    }

    const LOGISTIC: ModelConfig = ModelConfig::Logistic { max_iter: 1000 };

    #[test]
    fn every_view_of_the_fixtures_is_served() {
        let go = AtomicBool::new(false);
        for (id, config) in [
            (DatasetId::Iris, LOGISTIC),
            (DatasetId::BreastCancer, LOGISTIC),
            (DatasetId::Titanic, LOGISTIC),
            (DatasetId::Diabetes, ModelConfig::Linear),
        ] {
            let model = trained(id, config);
            let summary = model.summary();
            assert!(model.evaluate().is_ok(), "{id:?}");
            let classifier = summary.model_task == TaskType::Classification;
            assert_eq!(model.roc_curves().is_ok(), classifier, "{id:?}");
            assert_eq!(model.probability_view().is_ok(), classifier, "{id:?}");
            assert_eq!(model.residuals_view().is_ok(), !classifier, "{id:?}");
            {
                let grid = model.decision_grid(0, 1, &|_, _| {}, &go).unwrap();
                let cells = grid.x.len() * grid.y.len();
                match grid.values {
                    GridValues::Classes { z } => assert_eq!(z.len(), cells),
                    GridValues::Values { z, .. } => assert_eq!(z.len(), cells),
                }
            }
            let coefficients = model.coefficients(None).unwrap();
            assert_eq!(coefficients.rows.len(), summary.feature_columns.len());
            assert!(coefficients.rows.windows(2).all(|w| w[0].abs >= w[1].abs));
            let form = model.prediction_form().unwrap();
            assert_eq!(form.controls.len(), summary.feature_columns.len(), "{id:?}");
            let values: Vec<f64> = form
                .controls
                .iter()
                .map(|c| match &c.control {
                    FormControlKind::Checkbox { default } => f64::from(u8::from(*default)),
                    FormControlKind::Select {
                        options,
                        default_index,
                    } => options[*default_index as usize].code,
                    FormControlKind::Slider { default, .. } => *default,
                })
                .collect();
            let result = model.predict_one(&values).unwrap();
            assert_eq!(
                matches!(result, PredictionResult::Class { .. }),
                matches!(config, ModelConfig::Logistic { .. })
            );
            assert!(!model
                .code(CodeTemplate::ConfusionMatrix, None)
                .unwrap()
                .is_empty());
        }
    }

    #[test]
    fn iris_matches_the_app() {
        let model = trained(DatasetId::Iris, LOGISTIC);
        let Evaluation::Classification(eval) = model.evaluate().unwrap() else {
            panic!("clasificación")
        };
        assert!((eval.accuracy - 1.0).abs() < 1e-12);
        assert!(eval
            .interpretation_md
            .starts_with("**Resumen del Modelo:**"));
        let form = model.prediction_form().unwrap();
        let defaults: Vec<f64> = form
            .controls
            .iter()
            .map(|c| match c.control {
                FormControlKind::Slider { default, .. } => default,
                _ => f64::NAN,
            })
            .collect();
        let PredictionResult::Class { name, proba, .. } = model.predict_one(&defaults).unwrap()
        else {
            panic!("clase")
        };
        assert_eq!(name, "versicolor");
        let p = proba.unwrap()[1].probability;
        assert!((p - 0.928_776_5).abs() < 1e-5, "{p}");
        let coefficients = model.coefficients(Some(0)).unwrap();
        assert_eq!(coefficients.rows[0].feature, "petal length (cm)");
        assert!((coefficients.intercept - 8.8638).abs() < 1e-3);
    }

    #[test]
    fn cancer_reports_d19_and_d21() {
        let model = trained(DatasetId::BreastCancer, LOGISTIC);
        assert!(matches!(
            model.summary().notices.as_slice(),
            [TrainNotice::NotConverged { max_iter: 1000, .. }]
        ));
        let ProbabilityView::Binary {
            shown_thresholds,
            uncertain,
            n,
            ..
        } = model.probability_view().unwrap()
        else {
            panic!("binaria")
        };
        assert_eq!((uncertain, n), (5, 171));
        assert_eq!(shown_thresholds.len(), 3);
    }
}
