//! Trained models of [`crate::real_facade::RealFacade`], over `mltutor-core` and
//! `mltutor-export` (fase2-regresion.md §8). Only orchestration and conversions to the
//! IPC DTOs live here: every computation is a core function.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

use mltutor_core as core;
use mltutor_core::model::{ModelError, Prediction, Predictor, TrainControl, TrainProgress};
use ndarray::{Array2, ArrayView2, Axis};

use crate::core_facade::{ModelHandle, ModelSummary, ModelView, Progress, TrainSpec};
use crate::ipc::{
    self, to_u32, CodeTemplate, CommandError, ExportFormat, Matrix, ModelConfig, TrainNotice,
};
use crate::models::not_converged_message;
use crate::real_facade::task;

/// `random_state=42` of the Python version (D7).
const SEED: u32 = 42;

/// Core model errors → Spanish command errors.
pub fn model_error(error: ModelError) -> CommandError {
    match error {
        ModelError::InvalidTarget => CommandError::invalid_target(error.to_string()),
        ModelError::Cancelled => CommandError::cancelled(),
        ModelError::NotClassifier => CommandError::not_classifier(),
        ModelError::NotRegressor => CommandError::not_regressor(),
        ModelError::GridTooSmall { rows, cols } => CommandError::grid_too_small(rows, cols),
        ModelError::InvalidInput { .. } | ModelError::FeatureMismatch { .. } => {
            CommandError::invalid_input(error.to_string())
        }
        ModelError::InvalidSelection { .. } => CommandError::invalid_selection(error.to_string()),
        other => CommandError::training(other),
    }
}

fn content_error(error: impl std::fmt::Display) -> CommandError {
    CommandError::internal(format!("contenido: {error}"))
}

fn core_config(config: ModelConfig) -> core::linear::ModelConfig {
    match config {
        ModelConfig::Linear => core::linear::ModelConfig::Linear,
        ModelConfig::Logistic { max_iter } => core::linear::ModelConfig::Logistic { max_iter },
    }
}

/// True codes of the test set, probabilities and `classes_` of a classifier.
type ClassifierOutputs<'a> = (Vec<usize>, ArrayView2<'a, f64>, &'a [usize]);

/// Predictions of the test set, computed once per model.
struct TestOutputs {
    prediction: Prediction,
    proba: Option<Array2<f64>>,
}

pub struct CoreModel {
    dataset: Arc<core::Dataset>,
    id: Option<core::DatasetId>,
    spec: TrainSpec,
    model: Box<dyn Predictor>,
    x_train: Array2<f64>,
    y_train: core::Target,
    x_test: Array2<f64>,
    y_test: core::Target,
    notices: Vec<TrainNotice>,
    outputs: OnceLock<Result<TestOutputs, CommandError>>,
    form: OnceLock<core::form::PredictionForm>,
}

/// `train_model`: split (D7), fit and report progress.
pub fn train(
    dataset: &Arc<core::Dataset>,
    id: Option<core::DatasetId>,
    spec: &TrainSpec,
    progress: &Progress<'_>,
    cancel: &AtomicBool,
) -> Result<ModelHandle, CommandError> {
    let split = core::split::train_test_split(dataset.n_rows(), spec.test_size, SEED)
        .map_err(crate::real_facade::dataset_error)?;
    let rows = |idx: &[usize]| dataset.features.select(Axis(0), idx);
    let x_train = rows(&split.train);
    let x_test = rows(&split.test);
    let y_train = dataset.target.select(&split.train);
    let y_test = dataset.target.select(&split.test);
    let report = |p: TrainProgress| {
        let percent =
            (p.max_iter > 0).then(|| (p.iteration as f32 / p.max_iter as f32 * 100.0).min(99.0));
        if !p.finished {
            progress("Entrenando modelo...", percent);
        }
    };
    let control = TrainControl::none()
        .with_cancel(cancel)
        .with_progress(&report);
    let model = core::linear::fit(core_config(spec.config), x_train.view(), &y_train, &control)
        .map_err(model_error)?;
    let mut notices = Vec::new();
    if let (ModelConfig::Logistic { max_iter }, Some(fit)) = (spec.config, model.fit_report()) {
        if !fit.converged {
            notices.push(TrainNotice::NotConverged {
                max_iter,
                message: not_converged_message(max_iter),
            });
        }
    }
    Ok(Arc::new(CoreModel {
        dataset: dataset.clone(),
        id,
        spec: spec.clone(),
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        notices,
        outputs: OnceLock::new(),
        form: OnceLock::new(),
    }))
}

fn histogram(h: core::stats::Histogram) -> ipc::Histogram {
    ipc::Histogram {
        edges: h.edges,
        counts: h.counts,
    }
}

fn curve(c: core::curves::Curve) -> ipc::Curve {
    ipc::Curve {
        x: c.x,
        y: c.y,
        thresholds: Some(c.thresholds.into_iter().map(ipc::finite).collect()),
    }
}

fn level(l: core::coefficients::Level) -> ipc::Level {
    match l {
        core::coefficients::Level::VeryHigh => ipc::Level::VeryHigh,
        core::coefficients::Level::High => ipc::Level::High,
        core::coefficients::Level::Medium => ipc::Level::Medium,
        core::coefficients::Level::Low => ipc::Level::Low,
    }
}

impl CoreModel {
    fn class_names(&self) -> Option<Vec<String>> {
        self.dataset.class_names().map(<[String]>::to_vec)
    }

    fn name_of(&self, code: usize) -> String {
        self.dataset
            .class_names()
            .and_then(|names| names.get(code).cloned())
            .unwrap_or_else(|| code.to_string())
    }

    fn outputs(&self) -> Result<&TestOutputs, CommandError> {
        self.outputs
            .get_or_init(|| {
                let prediction = self
                    .model
                    .predict(self.x_test.view())
                    .map_err(model_error)?;
                let proba = self
                    .model
                    .predict_proba(self.x_test.view())
                    .transpose()
                    .map_err(model_error)?;
                Ok(TestOutputs { prediction, proba })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Codes of the test set and probabilities of a classifier.
    fn classifier(&self) -> Result<ClassifierOutputs<'_>, CommandError> {
        if self.model.task() != core::TaskType::Classification {
            return Err(CommandError::not_classifier());
        }
        let outputs = self.outputs()?;
        let (Some(proba), Some(classes), Some(y)) = (
            outputs.proba.as_ref(),
            self.model.classes(),
            self.y_test.class_codes(),
        ) else {
            return Err(CommandError::not_classifier());
        };
        Ok((y, proba.view(), classes))
    }

    fn form(&self) -> &core::form::PredictionForm {
        self.form.get_or_init(|| {
            let metadata = self.id.and_then(|id| match id.metadata() {
                Ok(meta) => Some(meta),
                Err(error) => {
                    log::warn!("metadatos de {id:?} no disponibles: {error}");
                    None
                }
            });
            core::form::build(self.x_train.view(), &self.dataset, metadata.as_ref())
        })
    }

    fn training_context(&self) -> core::codegen::RegressionTrainingContext {
        let csv = self.id.is_none();
        core::codegen::RegressionTrainingContext {
            dataset: self
                .id
                .map_or_else(|| "csv".to_owned(), |id| id.slug().to_owned()),
            dataset_label: self.spec.dataset_label.clone(),
            csv_file: csv.then(|| self.spec.dataset_label.trim_start_matches("📄 ").to_owned()),
            csv_target: csv.then(|| self.dataset.target_column.clone()),
            csv_sep: None,
            csv_decimal: None,
            csv_encoding: None,
            model: match self.spec.config {
                ModelConfig::Linear => "linear".to_owned(),
                ModelConfig::Logistic { .. } => "logistic".to_owned(),
            },
            max_iter: match self.spec.config {
                ModelConfig::Linear => None,
                ModelConfig::Logistic { max_iter } => Some(max_iter),
            },
            test_size: self.spec.test_size,
            feature_names: self.dataset.feature_columns.clone(),
            class_names: self.class_names(),
            sample: None,
        }
    }
}

impl ModelView for CoreModel {
    fn summary(&self) -> ModelSummary {
        ModelSummary {
            model_task: task(self.model.task()),
            dataset_task: task(self.dataset.task()),
            n_train: self.x_train.nrows(),
            n_test: self.x_test.nrows(),
            feature_columns: self.dataset.feature_columns.clone(),
            feature_names: self.dataset.feature_display.clone(),
            class_names: self.class_names(),
            notices: self.notices.clone(),
        }
    }

    fn evaluate(&self) -> Result<ipc::Evaluation, CommandError> {
        let outputs = self.outputs()?;
        match &outputs.prediction {
            Prediction::Classes(predicted) => {
                let truth = self
                    .y_test
                    .class_codes()
                    .ok_or_else(CommandError::not_classifier)?;
                let names = self.class_names().unwrap_or_default();
                let n = names
                    .len()
                    .max(self.model.classes().map_or(0, <[usize]>::len));
                let eval = core::metrics::classification(&truth, predicted, n);
                let interpretation =
                    core::interpretation::classification(&eval).map_err(content_error)?;
                let metrics = |name: String, m: &core::metrics::ClassMetrics| ipc::ClassMetrics {
                    name,
                    precision: m.precision,
                    recall: m.recall,
                    f1: m.f1,
                    support: m.support,
                };
                let sample = |s: &core::metrics::SampleRow| ipc::SampleRow {
                    position: s.position,
                    real: self.name_of(s.real),
                    predicted: self.name_of(s.predicted),
                    correct: s.correct,
                };
                let k = eval.confusion.len();
                Ok(ipc::Evaluation::Classification(
                    ipc::ClassificationEvaluation {
                        accuracy: eval.accuracy,
                        per_class: eval
                            .per_class
                            .iter()
                            .enumerate()
                            .map(|(i, m)| metrics(self.name_of(i), m))
                            .collect(),
                        macro_avg: metrics("macro avg".into(), &eval.macro_avg),
                        weighted_avg: metrics("weighted avg".into(), &eval.weighted_avg),
                        confusion: Matrix::from_f64(
                            k,
                            k,
                            eval.confusion
                                .iter()
                                .flatten()
                                .map(|&v| f64::from(v))
                                .collect::<Vec<_>>(),
                        ),
                        correct_head: eval.correct_head.iter().map(sample).collect(),
                        incorrect_head: eval.incorrect_head.iter().map(sample).collect(),
                        interpretation_md: interpretation,
                        warnings: eval.warnings.iter().map(|w| w.message_es(&names)).collect(),
                    },
                ))
            }
            Prediction::Values(predicted) => {
                let truth = self.y_test.numeric();
                let eval = core::metrics::regression(&truth, predicted);
                let interpretation =
                    core::interpretation::regression(&eval).map_err(content_error)?;
                let kde = eval.errors_kde.map(|k| ipc::KdeCurve {
                    level: "errores".into(),
                    n: to_u32(truth.len()),
                    bandwidth: None,
                    grid_min: k.x.first().copied().unwrap_or(0.0),
                    grid_max: k.x.last().copied().unwrap_or(0.0),
                    density: k.y,
                });
                Ok(ipc::Evaluation::Regression(ipc::RegressionEvaluation {
                    mse: eval.mse,
                    rmse: eval.rmse,
                    mae: eval.mae,
                    r2: eval.r2,
                    y_true: truth,
                    y_pred: predicted.clone(),
                    errors_hist: histogram(eval.errors_hist),
                    errors_kde: kde,
                    interpretation_md: interpretation,
                }))
            }
        }
    }

    fn roc_curves(&self) -> Result<ipc::RocData, CommandError> {
        let (y, proba, classes) = self.classifier()?;
        if classes.len() == 2 {
            let positive = classes[1];
            let score = proba.column(1).to_vec();
            let b = core::curves::binary_curves(&y, &score, positive);
            return Ok(ipc::RocData::Binary {
                positive_class: self.name_of(b.positive_class),
                roc: curve(b.roc),
                auc: b.auc,
                pr: curve(b.pr),
                ap: b.ap,
                baseline: b.baseline,
            });
        }
        let m = core::curves::multiclass_curves(&y, proba, classes);
        let class_curves = |c: core::curves::ClassCurves, name: String| ipc::ClassCurves {
            class_index: to_u32(c.class.unwrap_or(0)),
            name,
            roc: curve(c.roc),
            auc: c.auc,
            pr: curve(c.pr),
            ap: c.ap,
        };
        Ok(ipc::RocData::Multiclass {
            per_class: m
                .per_class
                .into_iter()
                .map(|c| {
                    let name = self.name_of(c.class.unwrap_or(0));
                    class_curves(c, name)
                })
                .collect(),
            micro: class_curves(m.micro, "micro".into()),
            macro_roc: curve(m.macro_roc),
            macro_auc: m.macro_auc,
            macro_ap: m.macro_ap,
            weighted_auc: ipc::finite(m.weighted_auc),
            absent_classes: m.absent_classes.iter().map(|&c| self.name_of(c)).collect(),
        })
    }

    fn probability_view(&self) -> Result<ipc::ProbabilityView, CommandError> {
        let (y, proba, classes) = self.classifier()?;
        let hist = |h: core::probabilities::ClassHistogram| ipc::ClassHistogram {
            class_index: to_u32(h.class),
            name: self.name_of(h.class),
            histogram: histogram(h.hist),
        };
        Ok(
            match core::probabilities::distribution(&y, proba, classes) {
                core::probabilities::ProbabilityView::Binary(b) => ipc::ProbabilityView::Binary {
                    per_real_class: b.per_real_class.into_iter().map(hist).collect(),
                    uncertain: b.uncertain,
                    confident: b.confident,
                    n: b.n,
                    thresholds: b
                        .thresholds
                        .rows
                        .into_iter()
                        .map(|r| ipc::ThresholdRow {
                            threshold: r.threshold,
                            accuracy: r.accuracy,
                            precision: r.precision,
                            recall: r.recall,
                            f1: r.f1,
                        })
                        .collect(),
                    best_f1_threshold: b.thresholds.best_f1_threshold,
                    shown_thresholds: b.thresholds.shown_thresholds,
                },
                core::probabilities::ProbabilityView::Multiclass { per_class } => {
                    ipc::ProbabilityView::Multiclass {
                        per_class: per_class.into_iter().map(hist).collect(),
                    }
                }
            },
        )
    }

    fn residuals_view(&self) -> Result<ipc::ResidualsView, CommandError> {
        let Prediction::Values(predicted) = &self.outputs()?.prediction else {
            return Err(CommandError::not_regressor());
        };
        let truth = self.y_test.numeric();
        let a = core::residuals::analysis(&truth, predicted);
        Ok(ipc::ResidualsView {
            r2: core::metrics::r2_score(&truth, predicted),
            y_true: truth,
            y_pred: predicted.clone(),
            mean: a.mean,
            std: a.std,
            skew: a.skew,
            linregress_r: a.linregress_r,
            hist: histogram(a.hist),
            checks: ipc::ResidualChecks {
                mean_ok: a.checks.mean_ok,
                corr_ok: a.checks.corr_ok,
                normal_ok: a.checks.normal_ok,
            },
        })
    }

    fn decision_grid(
        &self,
        x: usize,
        y: usize,
        progress: &Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<ipc::DecisionGrid, CommandError> {
        progress("Generando frontera de decisión...", None);
        let control = TrainControl::none().with_cancel(cancel);
        let grid = core::grid::decision_grid(
            self.model.as_ref(),
            self.x_train.view(),
            &self.y_train,
            x,
            y,
            core::grid::GridOptions::default(),
            &control,
        )
        .map_err(model_error)?;
        let notice = |n: core::grid::GridNotice| match n {
            core::grid::GridNotice::Created { rows, cols } => ipc::GridNotice::Created {
                rows: to_u32(rows),
                cols: to_u32(cols),
            },
            core::grid::GridNotice::TooLarge { points } => ipc::GridNotice::TooLarge {
                points: points as u64,
            },
            core::grid::GridNotice::Resized { rows, cols } => ipc::GridNotice::Resized {
                rows: to_u32(rows),
                cols: to_u32(cols),
            },
        };
        let classifier = self.model.task() == core::TaskType::Classification;
        let (n, _) = grid.points.dim();
        Ok(ipc::DecisionGrid {
            values: match grid.values {
                core::grid::GridValues::Classes { z } => ipc::GridValues::Classes { z },
                core::grid::GridValues::Values { z, min, max } => {
                    ipc::GridValues::Values { z, min, max }
                }
            },
            notices: grid.axes.notices.into_iter().map(notice).collect(),
            x: grid.axes.x,
            y: grid.axes.y,
            points: Matrix::from_f64(n, 2, grid.points.iter().copied().collect::<Vec<_>>()),
            point_target: grid.point_target,
            feature_names: [
                self.dataset.feature_display[x].clone(),
                self.dataset.feature_display[y].clone(),
            ],
            class_names: self.class_names().filter(|_| classifier),
            show_2d_warning: grid.show_2d_warning,
        })
    }

    fn coefficients(
        &self,
        class_index: Option<usize>,
    ) -> Result<ipc::CoefficientsView, CommandError> {
        let v = core::coefficients::view(
            self.model.as_ref(),
            &self.dataset.feature_display,
            class_index,
        )
        .map_err(model_error)?;
        Ok(ipc::CoefficientsView {
            rows: v
                .rows
                .into_iter()
                .map(|r| ipc::CoefficientRow {
                    feature: r.feature,
                    feature_position: to_u32(r.feature_position),
                    coef: r.coef,
                    abs: r.abs,
                    odds_ratio: r.odds_ratio,
                    change_pct: r.change_pct,
                    effect: match r.effect {
                        core::coefficients::Effect::Positive => ipc::Effect::Positive,
                        core::coefficients::Effect::Negative => ipc::Effect::Negative,
                    },
                    importance: r.importance.map(level),
                    strength: r.strength.map(level),
                    intensity: level(r.intensity),
                })
                .collect(),
            class_index: v.class_index.map(to_u32),
            n_rows: to_u32(v.n_rows),
            intercept: v.intercept,
            intercept_odds: v.intercept_odds,
            quantiles: ipc::AbsQuantiles {
                q25: v.quantiles.q25,
                q60: v.quantiles.q60,
                q75: v.quantiles.q75,
                q80: v.quantiles.q80,
                q95: v.quantiles.q95,
            },
            extreme: v.extreme,
        })
    }

    fn prediction_form(&self) -> Result<ipc::PredictionForm, CommandError> {
        Ok(ipc::PredictionForm {
            controls: self
                .form()
                .controls
                .iter()
                .map(|c| ipc::FormControl {
                    index: to_u32(c.index),
                    column: c.column.clone(),
                    label: c.name.clone(),
                    description: c.description.clone(),
                    control: match &c.control {
                        core::form::ControlKind::Checkbox { default } => {
                            ipc::FormControlKind::Checkbox { default: *default }
                        }
                        core::form::ControlKind::Select {
                            options,
                            default_index,
                        } => ipc::FormControlKind::Select {
                            options: options
                                .iter()
                                .map(|o| ipc::SelectOption {
                                    label: o.label.clone(),
                                    code: o.code,
                                })
                                .collect(),
                            default_index: to_u32(*default_index),
                        },
                        core::form::ControlKind::Slider {
                            min,
                            max,
                            step,
                            default,
                        } => ipc::FormControlKind::Slider {
                            min: *min,
                            max: *max,
                            step: *step,
                            default: *default,
                        },
                    },
                })
                .collect(),
        })
    }

    fn predict_one(&self, values: &[f64]) -> Result<ipc::PredictionResult, CommandError> {
        match core::form::predict_one(self.model.as_ref(), values).map_err(model_error)? {
            core::form::PredictionOutcome::Class { class, proba } => {
                Ok(ipc::PredictionResult::Class {
                    index: to_u32(class),
                    name: self.name_of(class),
                    proba: proba.map(|rows| {
                        rows.into_iter()
                            .map(|(code, probability)| ipc::ClassProbability {
                                name: self.name_of(code),
                                probability,
                            })
                            .collect()
                    }),
                })
            }
            core::form::PredictionOutcome::Value { value } => Ok(ipc::PredictionResult::Value {
                value,
                coefficients: self.model.linear_coefficients().map(|c| {
                    self.dataset
                        .feature_display
                        .iter()
                        .zip(c.coef.row(0))
                        .map(|(feature, coef)| ipc::FeatureCoefficient {
                            feature: feature.clone(),
                            coef: *coef,
                        })
                        .collect()
                }),
            }),
        }
    }

    fn export(&self, format: ExportFormat) -> Result<Vec<u8>, CommandError> {
        crate::real_exports::export(self, format)
    }

    fn code(
        &self,
        template: CodeTemplate,
        pair: Option<(usize, usize)>,
    ) -> Result<String, CommandError> {
        use core::codegen::{render, render_static, CodeTemplate as T, DecisionBoundaryContext};
        let rendered = match template {
            CodeTemplate::ConfusionMatrix => render_static(T::ConfusionMatrix),
            CodeTemplate::PrecisionByClass => render_static(T::PrecisionByClass),
            CodeTemplate::PredVsReal => render_static(T::PredVsReal),
            CodeTemplate::ErrorDistribution => render_static(T::ErrorDistribution),
            CodeTemplate::DecisionBoundary => {
                let (x, y) =
                    pair.unwrap_or((0, 1.min(self.dataset.n_features().saturating_sub(1))));
                render(
                    T::DecisionBoundary,
                    &DecisionBoundaryContext {
                        feature_indices: [x, y],
                        feature_names: vec![
                            self.dataset.feature_display[x].clone(),
                            self.dataset.feature_display[y].clone(),
                        ],
                        class_names: self.class_names(),
                    },
                )
            }
            // D29: the complete training script (Python's literal snippet did not run).
            CodeTemplate::RegressionCode => render(T::RegressionTraining, &self.training_context()),
        };
        rendered.map_err(|error| CommandError::internal(format!("plantilla: {error}")))
    }
}

/// What the exports need from a trained model.
pub struct ExportSource<'a> {
    pub model: &'a dyn Predictor,
    pub dataset: &'a core::Dataset,
    pub spec: &'a TrainSpec,
    pub x_test: ArrayView2<'a, f64>,
    pub n_train: usize,
}

impl CoreModel {
    pub fn export_source(&self) -> ExportSource<'_> {
        ExportSource {
            model: self.model.as_ref(),
            dataset: &self.dataset,
            spec: &self.spec,
            x_test: self.x_test.view(),
            n_train: self.x_train.nrows(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_facade::CoreFacade;
    use crate::fixture_facade::FixtureFacade;
    use crate::ipc::{
        DatasetId, Evaluation, FormControlKind, PredictionResult, ProbabilityView, RocData,
    };
    use crate::real_facade::RealFacade;

    const LOGISTIC: ModelConfig = ModelConfig::Logistic { max_iter: 1000 };
    const CASES: [(DatasetId, ModelConfig); 4] = [
        (DatasetId::Iris, LOGISTIC),
        (DatasetId::BreastCancer, LOGISTIC),
        (DatasetId::Titanic, LOGISTIC),
        (DatasetId::Diabetes, ModelConfig::Linear),
    ];

    fn trained(facade: &dyn CoreFacade, id: DatasetId, config: ModelConfig) -> ModelHandle {
        let spec = TrainSpec {
            config,
            test_size: 0.3,
            dataset_label: String::new(),
        };
        facade
            .load_builtin(id)
            .unwrap()
            .train(&spec, &|_, _| {}, &AtomicBool::new(false))
            .unwrap()
    }

    fn defaults(form: &ipc::PredictionForm) -> Vec<f64> {
        form.controls
            .iter()
            .map(|c| match &c.control {
                FormControlKind::Checkbox { default } => f64::from(u8::from(*default)),
                FormControlKind::Select {
                    options,
                    default_index,
                } => options[*default_index as usize].code,
                FormControlKind::Slider { default, .. } => *default,
            })
            .collect()
    }

    /// Both facades must show the same thing (the fixtures are scikit-learn's values;
    /// Cáncer does not converge in Python, D19, so only its structure is compared).
    #[test]
    fn real_models_match_the_fixture_facade() {
        let real = RealFacade;
        let fixture = FixtureFacade::from_checkout();
        let go = AtomicBool::new(false);
        for (id, config) in CASES {
            let a = trained(&real, id, config);
            let b = trained(&fixture, id, config);
            let (sa, sb) = (a.summary(), b.summary());
            assert_eq!(
                (sa.n_train, sa.n_test, &sa.class_names),
                (sb.n_train, sb.n_test, &sb.class_names),
                "{id:?}"
            );
            assert_eq!(sa.model_task, sb.model_task);
            let exact = id != DatasetId::BreastCancer;
            match (a.evaluate().unwrap(), b.evaluate().unwrap()) {
                (Evaluation::Classification(x), Evaluation::Classification(y)) => {
                    if exact {
                        assert!((x.accuracy - y.accuracy).abs() < 1e-12, "{id:?}");
                        assert_eq!(x.confusion, y.confusion, "{id:?}");
                        assert_eq!(x.interpretation_md, y.interpretation_md, "{id:?}");
                    }
                }
                (Evaluation::Regression(x), Evaluation::Regression(y)) => {
                    assert!(
                        (x.r2 - y.r2).abs() < 1e-9 && (x.mse - y.mse).abs() < 1e-6,
                        "{id:?}"
                    );
                    assert_eq!(x.interpretation_md, y.interpretation_md);
                }
                _ => panic!("tareas distintas en {id:?}"),
            }
            if sa.model_task == ipc::TaskType::Classification {
                let (ra, rb) = (a.roc_curves().unwrap(), b.roc_curves().unwrap());
                match (ra, rb) {
                    (
                        RocData::Binary {
                            auc: x,
                            positive_class: p,
                            ..
                        },
                        RocData::Binary {
                            auc: y,
                            positive_class: q,
                            ..
                        },
                    ) => {
                        assert_eq!(p, q);
                        // Titanic: sklearn stops 2.6e-3 from the optimum (fase2 §12).
                        assert!(!exact || (x - y).abs() < 1e-3, "{id:?} {x} {y}");
                    }
                    (
                        RocData::Multiclass { macro_auc: x, .. },
                        RocData::Multiclass { macro_auc: y, .. },
                    ) => {
                        assert!((x - y).abs() < 1e-9);
                    }
                    _ => panic!("ROC distinta en {id:?}"),
                }
                assert!(matches!(
                    (a.probability_view().unwrap(), b.probability_view().unwrap()),
                    (
                        ProbabilityView::Binary { .. },
                        ProbabilityView::Binary { .. }
                    ) | (
                        ProbabilityView::Multiclass { .. },
                        ProbabilityView::Multiclass { .. }
                    )
                ));
            } else {
                let (x, y) = (a.residuals_view().unwrap(), b.residuals_view().unwrap());
                assert_eq!(x.checks, y.checks);
                assert_eq!(x.hist.counts, y.hist.counts);
            }
            let grid = a.decision_grid(0, 1, &|_, _| {}, &go).unwrap();
            let fixture_grid = b.decision_grid(0, 1, &|_, _| {}, &go).unwrap();
            assert_eq!(grid.notices.len(), fixture_grid.notices.len(), "{id:?}");
            let (ca, cb) = (a.coefficients(None).unwrap(), b.coefficients(None).unwrap());
            assert_eq!(ca.rows.len(), cb.rows.len());
            // Titanic has coefficients within 5e-3 of each other: their order depends on
            // where sklearn's solver stopped.
            if exact && id != DatasetId::Titanic {
                let features = |v: &ipc::CoefficientsView| {
                    v.rows.iter().map(|r| r.feature.clone()).collect::<Vec<_>>()
                };
                assert_eq!(features(&ca), features(&cb), "{id:?}");
            }
            let (fa, fb) = (a.prediction_form().unwrap(), b.prediction_form().unwrap());
            assert_eq!(fa.controls.len(), fb.controls.len());
            for (x, y) in fa.controls.iter().zip(&fb.controls) {
                // Iris descriptions exist in Rust; Python never showed them (seaborn names).
                assert_eq!(x.label, y.label, "{id:?}");
                assert_eq!(
                    std::mem::discriminant(&x.control),
                    std::mem::discriminant(&y.control),
                    "{id:?} {}",
                    x.label
                );
            }
            let (pa, pb) = (
                a.predict_one(&defaults(&fa)).unwrap(),
                b.predict_one(&defaults(&fb)).unwrap(),
            );
            match (pa, pb) {
                (
                    PredictionResult::Class { name: x, .. },
                    PredictionResult::Class { name: y, .. },
                ) => {
                    assert!(!exact || x == y, "{id:?}");
                }
                (
                    PredictionResult::Value { value: x, .. },
                    PredictionResult::Value { value: y, .. },
                ) => {
                    assert!((x - y).abs() < 1e-6 * y.abs().max(1.0), "{id:?}");
                }
                _ => panic!("predicción distinta en {id:?}"),
            }
            for template in [
                CodeTemplate::ConfusionMatrix,
                CodeTemplate::DecisionBoundary,
                CodeTemplate::RegressionCode,
            ] {
                assert!(!a.code(template, Some((0, 1))).unwrap().is_empty());
            }
        }
    }

    #[test]
    fn d19_notice_follows_the_fit_report_and_logistic_rejects_regression() {
        // Rust's solver converges in Cáncer with 1000 iterations (Python does not, D19):
        // no notice. With a single iteration it cannot converge.
        let cancer = trained(&RealFacade, DatasetId::BreastCancer, LOGISTIC);
        assert!(cancer.summary().notices.is_empty());
        let short = trained(
            &RealFacade,
            DatasetId::BreastCancer,
            ModelConfig::Logistic { max_iter: 1 },
        );
        assert!(matches!(
            short.summary().notices.as_slice(),
            [TrainNotice::NotConverged { max_iter: 1, .. }]
        ));
        let spec = TrainSpec {
            config: LOGISTIC,
            test_size: 0.3,
            dataset_label: String::new(),
        };
        let error = RealFacade
            .load_builtin(DatasetId::Diabetes)
            .unwrap()
            .train(&spec, &|_, _| {}, &AtomicBool::new(false))
            .err()
            .unwrap();
        assert!(matches!(error, CommandError::InvalidTarget { .. }));
    }

    #[test]
    fn training_is_cancellable() {
        let spec = TrainSpec {
            config: LOGISTIC,
            test_size: 0.3,
            dataset_label: String::new(),
        };
        let error = RealFacade
            .load_builtin(DatasetId::BreastCancer)
            .unwrap()
            .train(&spec, &|_, _| {}, &AtomicBool::new(true))
            .err()
            .unwrap();
        assert_eq!(error, CommandError::cancelled());
    }
}
