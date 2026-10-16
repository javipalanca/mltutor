//! A trained network as seen by the commands (fase5-redes.md §5.2): its summary, its
//! history and the views of 📈 Evaluación and 🎯 Visualización (F5c). Both facades build
//! it; the views that need the weights go through [`NnEngine`] (the real engine in the
//! app; the FixtureFacade only replays the history). The exports arrive in F5d.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use mltutor_core as core;

use crate::core_facade::{BatchOutcome, ModelSummary, ModelView, Progress, TrainSpec};
use crate::ipc::{
    ActivationsRequest, ActivationsView, BatchTemplate, CodeTemplate, CoefficientsView,
    CommandError, DatasetSummary, DecisionGrid, DiagnosticsView, Divergence, EpochLog, Evaluation,
    ExploreView, ExportFormat, HistoryView, ModelConfig, NnActivation, NnArchitecture,
    NnEvaluationView, NnMetric, NnSeverity, NnSummary, NnTraining, NnViewNode, PredictionForm,
    PredictionResult, ProbabilityView, ResidualsView, RocData, SplitSizes, TaskType, WeightsView,
};
use crate::nn::{self, NnTask, Resolved};

/// What every engine needs before fitting: the validated controls, the task, the loss
/// and the sizes of the split.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub arch: NnArchitecture,
    pub training: NnTraining,
    pub task: NnTask,
    pub resolved: Resolved,
    pub layers: Vec<u32>,
    pub sizes: SplitSizes,
}

/// Checks the request (ranges, D71, a class with one sample) before any work.
pub fn prepare(
    summary: &DatasetSummary,
    spec: &TrainSpec,
    class_counts: Option<&[(String, u32)]>,
) -> Result<Prepared, CommandError> {
    let ModelConfig::NeuralNet { arch, training } = &spec.config else {
        return Err(CommandError::not_nn());
    };
    nn::validate(arch, training)?;
    let task = NnTask::of(summary);
    nn::check_trainable(task, arch.output_activation)?;
    if task.task() == TaskType::Classification {
        if let Some((name, _)) = class_counts
            .unwrap_or_default()
            .iter()
            .find(|(_, count)| *count < 2)
        {
            return Err(nn::too_few_per_class(name));
        }
    }
    Ok(Prepared {
        arch: arch.clone(),
        training: training.clone(),
        task,
        resolved: nn::resolve(task, arch.output_activation),
        layers: nn::layer_sizes(summary.n_features, arch, task),
        sizes: nn::split_sizes(summary.n_rows, training.validation_pct),
    })
}

/// Epochs (1-based) after which ReduceLROnPlateau lowered the learning rate: the LR is
/// logged before reducing, so a drop shows in the next epoch.
pub fn lr_drops(epochs: &[EpochLog]) -> Vec<u16> {
    epochs
        .windows(2)
        .filter_map(|w| match (w[0].learning_rate, w[1].learning_rate) {
            (Some(a), Some(b)) if b < a * (1.0 - 1e-6) => Some(w[0].epoch),
            _ => None,
        })
        .collect()
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// The outcome of a training, whatever the engine.
pub struct Outcome {
    pub epochs: Vec<EpochLog>,
    pub best_epoch: Option<u16>,
    /// K1 (1-based): the engine's own detection (a NaN batch loss can come before a NaN
    /// epoch log); `None` = look at the epoch logs.
    pub diverged_epoch: Option<u16>,
    pub started: Instant,
}

/// The views of 📈 and 🎯 that need the trained weights (F5c). The real engine
/// (`real_nn::RealNn`) implements them; the FixtureFacade replays only the history.
pub trait NnEngine: Send + Sync {
    /// The model as a classic [`ModelView`] (C1/C2, C9, static code snippets).
    fn delegate(&self) -> &dyn ModelView;
    /// `model.evaluate(X_test, y_test)`: loss and metric.
    fn test_scores(&self) -> (f64, f64);
    /// D87: `false` with a tanh output (its outputs are not probabilities).
    fn output_is_probability(&self) -> bool;
    fn weights(&self, bins_weights: usize, bins_biases: usize)
        -> Result<WeightsView, CommandError>;
    fn activations(
        &self,
        req: &ActivationsRequest,
        cancel: &AtomicBool,
    ) -> Result<ActivationsView, CommandError>;
    /// D78: the whole network on a grid of two features, the rest at their mean.
    fn mean_grid(
        &self,
        x: usize,
        y: usize,
        cancel: &AtomicBool,
    ) -> Result<DecisionGrid, CommandError>;
    /// F5d: the files of 💾 Exportar (D75). The model has already checked K1.
    fn export(&self, _format: ExportFormat) -> Result<Vec<u8>, CommandError> {
        Err(pending())
    }
    /// F5d: the Keras training script (D74, with the seed) and the NumPy predictor.
    fn code(&self, _template: CodeTemplate) -> Result<String, CommandError> {
        Err(pending())
    }
    /// F5d: 🎲 Exploración (C29).
    fn explore(&self, _row: usize, _feature: usize) -> Result<ExploreView, CommandError> {
        Err(pending())
    }
    /// F5d: "📋 Formato esperado del archivo CSV".
    fn batch_template(&self, _rows: usize) -> Result<BatchTemplate, CommandError> {
        Err(pending())
    }
    /// F5d: 📊 Predicción por lotes (C29).
    fn predict_batch(
        &self,
        _path: &std::path::Path,
        _cancel: &AtomicBool,
    ) -> Result<BatchOutcome, CommandError> {
        Err(pending())
    }
    /// F5d: 📋 Metadatos (D75); `trained_at` = local date of the training.
    fn metadata(&self, _trained_at: &str) -> Result<String, CommandError> {
        Err(pending())
    }
}

/// A trained network: summary, history (both facades) and, in the app, the engine
/// with the weights.
pub struct NnModel {
    summary: ModelSummary,
    history: HistoryView,
    series: core::nn_data::HistorySeries,
    task: TaskType,
    n_val: usize,
    output_activation: NnActivation,
    engine: Option<Arc<dyn NnEngine>>,
}

/// D77: "Oculta i" / "Salida" (the names of the architecture).
pub fn layer_name(index: usize, n_dense: usize) -> String {
    if index + 1 == n_dense {
        nn::nn_text("arquitectura", "detalles_salida", "Salida")
    } else {
        nn::nn_text("arquitectura", "detalles_oculta", "Oculta {i}")
            .replace("{i}", &(index + 1).to_string())
    }
}

/// The Keras name of dense layer `index` (`dense_1`…, `output_layer`).
pub fn keras_name(index: usize, n_dense: usize) -> String {
    if index + 1 == n_dense {
        "output_layer".to_owned()
    } else {
        format!("dense_{}", index + 1)
    }
}

/// K1: first epoch (1-based) whose loss or validation loss is not finite.
pub fn diverged_epoch(epochs: &[EpochLog]) -> Option<u16> {
    epochs
        .iter()
        .find(|l| l.loss.is_none() || l.val_loss.is_none())
        .map(|l| l.epoch)
}

/// The IPC epochs as the series the core diagnostics read (`None` → NaN).
fn series(epochs: &[EpochLog], metric: NnMetric) -> core::nn_data::HistorySeries {
    let col = |f: fn(&EpochLog) -> Option<f64>| -> Vec<f64> {
        epochs.iter().map(|l| f(l).unwrap_or(f64::NAN)).collect()
    };
    core::nn_data::HistorySeries {
        loss: col(|l| l.loss),
        val_loss: Some(col(|l| l.val_loss)),
        metric_key: match metric {
            NnMetric::Accuracy => "accuracy",
            NnMetric::Mae => "mae",
        }
        .to_owned(),
        metric: Some(col(|l| l.metric)),
        val_metric: Some(col(|l| l.val_metric)),
    }
}

fn severity(s: core::nn_data::Severity) -> NnSeverity {
    match s {
        core::nn_data::Severity::Success => NnSeverity::Success,
        core::nn_data::Severity::Info => NnSeverity::Info,
        core::nn_data::Severity::Warning => NnSeverity::Warning,
        core::nn_data::Severity::Error => NnSeverity::Error,
    }
}

/// A core view tree → the IPC one (same shape).
pub fn view_node(node: core::view::ViewNode) -> NnViewNode {
    use core::view::ViewNode as V;
    let all = |children: Vec<V>| children.into_iter().map(view_node).collect();
    match node {
        V::Page { children } => NnViewNode::Page {
            children: all(children),
        },
        V::Text { text, html } => NnViewNode::Text { text, html },
        V::Heading { text, level } => NnViewNode::Heading { text, level },
        V::Notice { text, severity: s } => NnViewNode::Notice {
            text,
            severity: severity(s),
        },
        V::Metric {
            label,
            value,
            delta,
            delta_color,
            help,
        } => NnViewNode::Metric {
            label,
            value,
            delta,
            delta_color,
            help,
        },
        V::Row { weights, children } => NnViewNode::Row {
            weights,
            children: all(children),
        },
        V::Column { children } => NnViewNode::Column {
            children: all(children),
        },
        V::Expander {
            label,
            expanded,
            children,
        } => NnViewNode::Expander {
            label,
            expanded,
            children: all(children),
        },
    }
}

fn core_task(task: TaskType) -> core::TaskType {
    match task {
        TaskType::Classification => core::TaskType::Classification,
        TaskType::Regression => core::TaskType::Regression,
    }
}

fn content_error(error: impl std::fmt::Display) -> CommandError {
    CommandError::internal(format!("contenido: {error}"))
}

impl NnModel {
    pub fn new(
        summary: &DatasetSummary,
        prepared: &Prepared,
        outcome: Outcome,
        model_classes: Option<Vec<usize>>,
    ) -> Self {
        let max_epochs = prepared.training.epochs;
        let trained = u16::try_from(outcome.epochs.len()).unwrap_or(u16::MAX);
        let task = prepared.task.task();
        let nn_summary = NnSummary {
            layers: prepared.layers.clone(),
            loss: prepared.resolved.loss.to_owned(),
            metric: prepared.resolved.metric,
            epochs_trained: trained,
            best_epoch: outcome.best_epoch,
            stopped_early: trained < max_epochs,
            n_fit: prepared.sizes.train,
            n_val: prepared.sizes.val,
            n_test: prepared.sizes.test,
            trained_at_ms: now_ms(),
            duration_ms: u32::try_from(outcome.started.elapsed().as_millis()).unwrap_or(u32::MAX),
        };
        let model_classes = match task {
            TaskType::Classification => model_classes.or_else(|| {
                summary
                    .class_names
                    .as_ref()
                    .map(|names| (0..names.len()).collect())
            }),
            TaskType::Regression => None,
        };
        let metric = prepared.resolved.metric;
        let series = series(&outcome.epochs, metric);
        let diverged = outcome
            .diverged_epoch
            .or_else(|| diverged_epoch(&outcome.epochs));
        let cards = if diverged.is_none() && !outcome.epochs.is_empty() {
            match core::nn_diagnostics::history_cards(&series) {
                Ok(cards) => Some(view_node(cards.row)),
                Err(error) => {
                    log::error!("contenido de las tarjetas del historial: {error}");
                    None
                }
            }
        } else {
            None
        };
        Self {
            summary: ModelSummary {
                model_task: task,
                dataset_task: summary.task,
                n_train: (prepared.sizes.train + prepared.sizes.val) as usize,
                n_test: prepared.sizes.test as usize,
                feature_columns: summary.feature_columns.clone(),
                feature_names: summary.feature_display.clone(),
                class_names: summary.class_names.clone(),
                model_classes,
                notices: Vec::new(),
                tree: None,
                knn: None,
                nn: Some(nn_summary),
            },
            history: HistoryView {
                metric,
                lr_drops: lr_drops(&outcome.epochs),
                epochs: outcome.epochs,
                best_epoch: outcome.best_epoch,
                diverged_epoch: diverged,
                cards,
            },
            series,
            task,
            n_val: prepared.sizes.val as usize,
            output_activation: prepared.arch.output_activation,
            engine: None,
        }
    }

    pub fn with_engine(mut self, engine: Arc<dyn NnEngine>) -> Self {
        self.engine = Some(engine);
        self
    }

    fn engine(&self) -> Result<&dyn NnEngine, CommandError> {
        self.engine.as_deref().ok_or_else(pending)
    }

    fn divergence(&self) -> Option<Divergence> {
        self.history
            .diverged_epoch
            .map(|epoch| Divergence { epoch })
    }

    /// K1: the views that need meaningful weights are blocked after a divergence.
    fn check_not_diverged(&self) -> Result<(), CommandError> {
        match self.history.diverged_epoch {
            Some(epoch) => Err(diverged_error(epoch)),
            None => Ok(()),
        }
    }
}

/// K1: the message of the views that a diverged network cannot show.
pub fn diverged_error(epoch: u16) -> CommandError {
    CommandError::invalid_selection(
        nn::nn_text(
            "evaluacion",
            "aviso_divergencia",
            "🚨 El entrenamiento divergió en la época {epoca}: la pérdida se hizo infinita o NaN (no es un número), así que los pesos de la red no sirven y esta vista no se puede calcular. Suele deberse a un learning rate demasiado alto: prueba 0.01 o 0.001 y vuelve a entrenar.",
        )
        .replace("{epoca}", &epoch.to_string()),
    )
}

/// K1 in 🔮 and 💾: `[comun] red_no_valida` with `motivo_no_finita` (its weights are NaN:
/// nothing can be predicted with it or exported).
pub fn invalid_network_text() -> String {
    let motivo = nn::nn_text(
        "comun",
        "motivo_no_finita",
        "sus pesos contienen valores no numéricos (NaN o infinitos) porque el entrenamiento divergió; prueba con un learning rate menor",
    );
    nn::nn_text(
        "comun",
        "red_no_valida",
        "❌ **Esta red no se puede usar:** {motivo}. No se puede predecir con ella ni exportarla. Vuelve a ⚙️ Entrenamiento y entrena de nuevo.",
    )
    .replace("{motivo}", &motivo)
}

/// The error of an export of a diverged network (K1).
pub fn not_exportable(_epoch: u16) -> CommandError {
    CommandError::Export {
        message: invalid_network_text(),
    }
}

/// The views of the network that arrive in F5d (or need the real engine).
fn pending() -> CommandError {
    CommandError::invalid_selection(
        "Esta vista de la red neuronal todavía no está disponible en esta versión.",
    )
}

impl ModelView for NnModel {
    fn summary(&self) -> ModelSummary {
        self.summary.clone()
    }
    fn evaluate(&self) -> Result<Evaluation, CommandError> {
        self.check_not_diverged()?;
        self.engine()?.delegate().evaluate()
    }
    fn roc_curves(&self) -> Result<RocData, CommandError> {
        Err(pending())
    }
    fn probability_view(&self) -> Result<ProbabilityView, CommandError> {
        Err(pending())
    }
    fn residuals_view(&self) -> Result<ResidualsView, CommandError> {
        Err(pending())
    }
    fn decision_grid(
        &self,
        x: usize,
        y: usize,
        _progress: &Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<DecisionGrid, CommandError> {
        self.check_not_diverged()?;
        self.engine()?.mean_grid(x, y, cancel)
    }
    fn coefficients(&self, _class_index: Option<usize>) -> Result<CoefficientsView, CommandError> {
        Err(pending())
    }
    fn prediction_form(&self) -> Result<PredictionForm, CommandError> {
        self.engine()?.delegate().prediction_form()
    }
    fn predict_one(&self, values: &[f64]) -> Result<PredictionResult, CommandError> {
        if self.history.diverged_epoch.is_some() {
            return Err(CommandError::invalid_selection(invalid_network_text()));
        }
        let result = self.engine()?.delegate().predict_one(values)?;
        let finite = match &result {
            PredictionResult::Class { proba, .. } => proba
                .as_ref()
                .is_none_or(|p| p.iter().all(|c| c.probability.is_finite())),
            PredictionResult::Value { value, .. } => value.is_finite(),
        };
        if finite {
            Ok(result)
        } else {
            Err(CommandError::invalid_input(nn::nn_text(
                "predicciones",
                "prediccion_no_finita",
                "❌ La red ha devuelto un valor no numérico (NaN o infinito) para esta entrada.",
            )))
        }
    }
    fn export(&self, format: ExportFormat) -> Result<Vec<u8>, CommandError> {
        if !matches!(format, ExportFormat::Onnx) && !format.needs_nn() {
            return Err(pending());
        }
        // K1: a diverged network has NaN weights; nothing of it can be exported.
        if let Some(epoch) = self.history.diverged_epoch {
            return Err(not_exportable(epoch));
        }
        self.engine()?.export(format)
    }
    fn code(
        &self,
        template: CodeTemplate,
        pair: Option<(usize, usize)>,
    ) -> Result<String, CommandError> {
        // The snippets of C1/C2 (confusion matrix, precision by class, predictions,
        // errors) are the same for every model; the rest belongs to F5d.
        match template {
            CodeTemplate::ConfusionMatrix
            | CodeTemplate::PrecisionByClass
            | CodeTemplate::PredVsReal
            | CodeTemplate::ErrorDistribution => self.engine()?.delegate().code(template, pair),
            // The training script does not need the weights: also after a divergence.
            CodeTemplate::NnTraining => self.engine()?.code(template),
            CodeTemplate::NnNumpyPredictor => {
                if let Some(epoch) = self.history.diverged_epoch {
                    return Err(not_exportable(epoch));
                }
                self.engine()?.code(template)
            }
            _ => Err(pending()),
        }
    }
    fn nn_history(&self) -> Result<HistoryView, CommandError> {
        Ok(self.history.clone())
    }
    fn nn_explore(&self, row: usize, feature: usize) -> Result<ExploreView, CommandError> {
        self.check_not_diverged()?;
        self.engine()?.explore(row, feature)
    }
    fn nn_batch_template(&self, rows: usize) -> Result<BatchTemplate, CommandError> {
        self.engine()?.batch_template(rows)
    }
    fn nn_predict_batch(
        &self,
        path: &std::path::Path,
        cancel: &AtomicBool,
    ) -> Result<BatchOutcome, CommandError> {
        self.check_not_diverged()?;
        self.engine()?.predict_batch(path, cancel)
    }
    fn nn_metadata(&self, trained_at: &str) -> Result<String, CommandError> {
        self.summary.nn.as_ref().ok_or_else(CommandError::not_nn)?;
        self.engine()?.metadata(trained_at)
    }
    fn nn_evaluation(&self) -> Result<NnEvaluationView, CommandError> {
        let engine = self.engine()?;
        let (loss, metric) = engine.test_scores();
        let diverged = self.divergence();
        let evaluation = if diverged.is_some() {
            None
        } else {
            Some(engine.delegate().evaluate()?)
        };
        let r2_negative = match &evaluation {
            Some(Evaluation::Regression(r)) => {
                core::nn_diagnostics::r2_negative_notice(r.r2).map_err(content_error)?
            }
            _ => None,
        };
        Ok(NnEvaluationView {
            metric: self.history.metric,
            test_loss: nn::finite_log(loss),
            test_metric: nn::finite_log(metric),
            evaluation,
            r2_negative,
            diverged,
            output_is_probability: engine.output_is_probability(),
            output_activation: self.output_activation,
        })
    }
    fn nn_diagnostics(&self) -> Result<DiagnosticsView, CommandError> {
        if let Some(d) = self.divergence() {
            return Ok(DiagnosticsView {
                blocks: Vec::new(),
                diverged: Some(d),
            });
        }
        if self.series.loss.is_empty() {
            return Ok(DiagnosticsView {
                blocks: Vec::new(),
                diverged: None,
            });
        }
        let view = core::nn_diagnostics::diagnostics_view(
            &self.series,
            core_task(self.task),
            Some(self.n_val),
        )
        .map_err(content_error)?;
        Ok(DiagnosticsView {
            blocks: view.blocks.into_iter().map(view_node).collect(),
            diverged: None,
        })
    }
    fn nn_weights(
        &self,
        bins_weights: usize,
        bins_biases: usize,
    ) -> Result<WeightsView, CommandError> {
        self.check_not_diverged()?;
        self.engine()?.weights(bins_weights, bins_biases)
    }
    fn nn_activations(
        &self,
        req: &ActivationsRequest,
        cancel: &AtomicBool,
    ) -> Result<ActivationsView, CommandError> {
        self.check_not_diverged()?;
        self.engine()?.activations(req, cancel)
    }
}

/// Class counts of a classification dataset (for [`prepare`]).
pub fn class_counts(dataset: &dyn crate::core_facade::DatasetView) -> Option<Vec<(String, u32)>> {
    match dataset.target_distribution() {
        crate::ipc::TargetDistribution::Bar { labels, counts, .. } => {
            Some(labels.into_iter().zip(counts).collect())
        }
        crate::ipc::TargetDistribution::Histogram { .. } => None,
    }
}
