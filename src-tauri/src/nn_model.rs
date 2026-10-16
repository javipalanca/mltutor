//! A trained network as seen by the commands (fase5-redes.md §5.2): its summary and its
//! history. Both facades build it; the views of 📈 Evaluación and 🎯 Visualización (F5c)
//! and the exports (F5d) will be added on top of the engine's own model.

use std::sync::atomic::AtomicBool;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::core_facade::{ModelSummary, ModelView, Progress, TrainSpec};
use crate::ipc::{
    CodeTemplate, CoefficientsView, CommandError, DatasetSummary, DecisionGrid, EpochLog,
    Evaluation, ExportFormat, HistoryView, ModelConfig, NnArchitecture, NnSummary, NnTraining,
    PredictionForm, PredictionResult, ProbabilityView, ResidualsView, RocData, SplitSizes,
    TaskType,
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
    pub started: Instant,
}

/// A trained network: what tabs 2 and 3 need in F5b.
pub struct NnModel {
    summary: ModelSummary,
    history: HistoryView,
    /// The engine's network (`mltutor_nn::TrainedNetwork` in the app), for the views of
    /// F5c/F5d.
    #[allow(dead_code)]
    engine: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
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
                metric: prepared.resolved.metric,
                lr_drops: lr_drops(&outcome.epochs),
                epochs: outcome.epochs,
                best_epoch: outcome.best_epoch,
            },
            engine: None,
        }
    }

    pub fn with_engine(mut self, engine: std::sync::Arc<dyn std::any::Any + Send + Sync>) -> Self {
        self.engine = Some(engine);
        self
    }
}

/// The views of the network that arrive in F5c/F5d.
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
        Err(pending())
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
        _x: usize,
        _y: usize,
        _progress: &Progress<'_>,
        _cancel: &AtomicBool,
    ) -> Result<DecisionGrid, CommandError> {
        Err(pending())
    }
    fn coefficients(&self, _class_index: Option<usize>) -> Result<CoefficientsView, CommandError> {
        Err(pending())
    }
    fn prediction_form(&self) -> Result<PredictionForm, CommandError> {
        Err(pending())
    }
    fn predict_one(&self, _values: &[f64]) -> Result<PredictionResult, CommandError> {
        Err(pending())
    }
    fn export(&self, _format: ExportFormat) -> Result<Vec<u8>, CommandError> {
        Err(pending())
    }
    fn code(
        &self,
        _template: CodeTemplate,
        _pair: Option<(usize, usize)>,
    ) -> Result<String, CommandError> {
        Err(pending())
    }
    fn nn_history(&self) -> Result<HistoryView, CommandError> {
        Ok(self.history.clone())
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
