//! Boundary between the Tauri commands and the computation (fase1-shell-datos.md §5.5).
//!
//! The commands only talk to [`CoreFacade`]. Two implementations exist:
//! - [`crate::real_facade::RealFacade`], over `mltutor-core` (used by the app);
//! - [`crate::fixture_facade::FixtureFacade`], over the golden fixtures of
//!   `tests/parity/` (tests, and development while the core was being written).
//!
//! Every method may be slow (parsing, KDE…): the commands call them from
//! `spawn_blocking`, never on the main thread. Implementations return IPC DTOs directly
//! so that the conversion from core types lives in one place per implementation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::ipc::{
    ActivationsRequest, ActivationsView, BatchPredictView, BatchTemplate, ClassDistancesView,
    CodeTemplate, CoefficientsView, CommandError, CorrelationMatrix, CsvPreview, CsvTargetStats,
    DatasetEntry, DatasetId, DatasetSummary, DecisionGrid, DescribeTable, DiagKind,
    DiagnosticsView, EpochLog, Evaluation, ExploreView, ExportFormat, FeatureImportanceView,
    HistoryView, ImportanceMethod, KnnSummary, ModelConfig, NnEvaluationView, NnSummary,
    PairplotData, PlaygroundRequest, PlaygroundView, PredictionForm, PredictionResult,
    ProbabilityView, ResidualsView, RocData, RulesKind, TablePreview, TargetDistribution,
    TargetInfo, TaskChoice, TaskType, TrainNotice, TrainPhase, TreeDiagram, TreeSummary,
    WeightsView,
};

/// Maximum number of pairplot features (slider limit of §2.6).
pub const MAX_PAIRPLOT_FEATURES: usize = 6;

/// A parsed dataset, ready for the 📊 Datos views.
pub trait DatasetView: Send + Sync {
    /// `dataset` and `label` are filled in by the caller (they depend on registration).
    fn summary(&self) -> DatasetSummary;
    fn preview(&self, rows: usize) -> TablePreview;
    fn describe(&self) -> DescribeTable;
    fn target_distribution(&self) -> TargetDistribution;
    /// Checks `cancel` cooperatively (`CommandError::Cancelled` when set).
    fn correlation(&self, cancel: &AtomicBool) -> Result<CorrelationMatrix, CommandError>;
    /// `features` are friendly names; already validated as non-empty, ≤ 6 and unique.
    fn pairplot(&self, features: &[String], diag: DiagKind) -> Result<PairplotData, CommandError>;
    /// Splits the dataset (D7, `random_state=42`) and trains `spec.config` on the train
    /// part. Reports progress through `progress` and checks `cancel` cooperatively.
    fn train(
        &self,
        spec: &TrainSpec,
        progress: &Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<ModelHandle, CommandError>;
    /// `train` for iterative models: neural networks report their phases and every
    /// finished epoch to `observer` (fase5-redes.md §5.4). The other models ignore it
    /// (no default: wrappers such as `CachedViews` must forward it).
    fn train_observed(
        &self,
        spec: &TrainSpec,
        progress: &Progress<'_>,
        observer: &TrainObserver<'_>,
        cancel: &AtomicBool,
    ) -> Result<ModelHandle, CommandError>;
    /// What every training script needs to load this dataset (`red_entrenamiento.py.j2`
    /// can be generated without a model, D74). `label` = the selector label.
    fn script_base(&self, label: &str) -> ScriptBase;
}

/// Phases and epochs of an iterative training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainEvent<'a> {
    Phase(TrainPhase),
    Epoch(&'a EpochLog),
}

/// Receives the [`TrainEvent`]s of a training (called from the worker thread).
pub type TrainObserver<'a> = dyn Fn(TrainEvent<'_>) + Sync + 'a;

/// Variables shared by the training scripts (`regresion_entrenamiento.py.j2` and the
/// following ones): how to load the dataset, its names and classes.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ScriptBase {
    pub dataset: String,
    pub dataset_label: String,
    pub csv_file: Option<String>,
    pub csv_target: Option<String>,
    pub csv_sep: Option<String>,
    pub csv_decimal: Option<String>,
    pub csv_encoding: Option<String>,
    pub feature_names: Vec<String>,
    pub class_names: Option<Vec<String>>,
}

pub type DatasetHandle = Arc<dyn DatasetView>;

/// Progress callback of long operations: message and optional percentage (0..=100).
pub type Progress<'a> = dyn Fn(&str, Option<f32>) + Sync + 'a;

/// What to train (the command has already validated ranges and the target, D26).
#[derive(Debug, Clone, PartialEq)]
pub struct TrainSpec {
    pub config: ModelConfig,
    pub test_size: f64,
    /// "🌸 Iris - Clasificación de flores" (training block of the Python package).
    pub dataset_label: String,
}

/// Fixed facts of a trained model (the command adds id, lesson and dataset).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSummary {
    pub model_task: TaskType,
    pub dataset_task: TaskType,
    pub n_train: usize,
    pub n_test: usize,
    pub feature_columns: Vec<String>,
    pub feature_names: Vec<String>,
    pub class_names: Option<Vec<String>>,
    /// Class codes the classifier saw in the train set (`classes_`, sorted): indices
    /// into `class_names`. `None` for regressors.
    pub model_classes: Option<Vec<usize>>,
    pub notices: Vec<TrainNotice>,
    /// Decision trees: size of the tree.
    pub tree: Option<TreeSummary>,
    /// KNN: K, weights, metric and engine.
    pub knn: Option<KnnSummary>,
    /// Neural networks (fase5-redes.md §5.3).
    pub nn: Option<NnSummary>,
}

/// A trained model with its split, ready for the views of tabs 2–6 (fase2 §3).
/// Implementations may cache predictions and probabilities internally.
pub trait ModelView: Send + Sync {
    fn summary(&self) -> ModelSummary;
    /// C1/C2/C3 on the test set.
    fn evaluate(&self) -> Result<Evaluation, CommandError>;
    /// C4. `NotClassifier` for regressors.
    fn roc_curves(&self) -> Result<RocData, CommandError>;
    /// C5 + C6. `NotClassifier` for regressors.
    fn probability_view(&self) -> Result<ProbabilityView, CommandError>;
    /// C7. `NotRegressor` for classifiers.
    fn residuals_view(&self) -> Result<ResidualsView, CommandError>;
    /// C8: refits the same model on features `x`, `y` of the train set (KNN: of the whole
    /// dataset, D63) and predicts the grid. Indices already validated (distinct and in range).
    fn decision_grid(
        &self,
        x: usize,
        y: usize,
        progress: &Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<DecisionGrid, CommandError>;
    /// §2.5. `class_index` (logistic multiclass) already validated.
    fn coefficients(&self, class_index: Option<usize>) -> Result<CoefficientsView, CommandError>;
    /// C9, with the metadata of the training dataset (D32). KNN: ranges of the whole
    /// dataset (D63).
    fn prediction_form(&self) -> Result<PredictionForm, CommandError>;
    /// C9. `values` already checked: one finite value per feature. Trees add the decision
    /// path (C16) and the probabilities of the leaf (D46).
    fn predict_one(&self, values: &[f64]) -> Result<PredictionResult, CommandError>;
    /// C11: the bytes of the `.onnx` file, the `.zip` package, or (trees) the rules `.txt`
    /// and the Python function `.py`.
    fn export(&self, format: ExportFormat) -> Result<Vec<u8>, CommandError>;
    /// C12. `pair` = features of the decision boundary (D30).
    fn code(
        &self,
        template: CodeTemplate,
        pair: Option<(usize, usize)>,
    ) -> Result<String, CommandError>;
    /// C13 (fase3-arboles.md): boxes, colours and layout of `plot_tree`.
    fn tree_diagram(&self) -> Result<TreeDiagram, CommandError> {
        Err(CommandError::not_tree())
    }
    /// C14: `export_text`, with weights (`View`) or without (`Export`); D40 names.
    fn tree_rules(&self, _kind: RulesKind, _class_names: bool) -> Result<String, CommandError> {
        Err(CommandError::not_tree())
    }
    /// C10: native importance (trees) or permutation importance on the test set
    /// (any model; checks `cancel` between columns).
    fn feature_importance(
        &self,
        _method: ImportanceMethod,
        _progress: &Progress<'_>,
        _cancel: &AtomicBool,
    ) -> Result<FeatureImportanceView, CommandError> {
        Err(CommandError::not_tree())
    }
    /// C19 (📏 Análisis de Distancias): mean distances between the classes of the train
    /// set of the model (D56), with the within-class means of D67 and the J2 verdict.
    /// Checks `cancel` while it sums the O(n²) pairs.
    fn class_distances(&self, _cancel: &AtomicBool) -> Result<ClassDistancesView, CommandError> {
        Err(CommandError::not_knn())
    }
    /// C18 (🎮 Visualización Interactiva): a 2D KNN on the pair of `req` over every row
    /// of the dataset (D63) with the controls of the canvas, and its test points. The
    /// pair is already validated; checks `cancel` between points.
    fn knn_playground(
        &self,
        _req: &PlaygroundRequest,
        _cancel: &AtomicBool,
    ) -> Result<PlaygroundView, CommandError> {
        Err(CommandError::not_knn())
    }
    /// C27: the logs of every epoch, the restored epoch and the reductions of the LR.
    fn nn_history(&self) -> Result<HistoryView, CommandError> {
        Err(CommandError::not_nn())
    }
    /// 📈 of a network: test loss and metric, C1/C2, D84 and the divergence (K1).
    fn nn_evaluation(&self) -> Result<NnEvaluationView, CommandError> {
        Err(CommandError::not_nn())
    }
    /// C23: the diagnostics tree of the history.
    fn nn_diagnostics(&self) -> Result<DiagnosticsView, CommandError> {
        Err(CommandError::not_nn())
    }
    /// C25: statistics and histograms (already binned) of every dense layer.
    fn nn_weights(
        &self,
        _bins_weights: usize,
        _bins_biases: usize,
    ) -> Result<WeightsView, CommandError> {
        Err(CommandError::not_nn())
    }
    /// C26: activations of the first test rows, dead and constant neurons and the map.
    /// Ranges already validated; checks `cancel` between layers.
    fn nn_activations(
        &self,
        _req: &ActivationsRequest,
        _cancel: &AtomicBool,
    ) -> Result<ActivationsView, CommandError> {
        Err(CommandError::not_nn())
    }
    /// C29 🎲 Exploración (D81): row and feature already validated against the dataset.
    fn nn_explore(&self, _row: usize, _feature: usize) -> Result<ExploreView, CommandError> {
        Err(CommandError::not_nn())
    }
    /// C29 📊 Predicción por lotes: the expected columns and the first `rows` rows of the
    /// features of the dataset.
    fn nn_batch_template(&self, _rows: usize) -> Result<BatchTemplate, CommandError> {
        Err(CommandError::not_nn())
    }
    /// C29: reads the CSV at `path` (chosen in Rust), checks its columns and predicts
    /// every row. Checks `cancel` while reading.
    fn nn_predict_batch(
        &self,
        _path: &std::path::Path,
        _cancel: &AtomicBool,
    ) -> Result<BatchOutcome, CommandError> {
        Err(CommandError::not_nn())
    }
    /// 📋 Metadatos (D75): the JSON shown in 💾 Exportar (pretty-printed). `trained_at` =
    /// the local date of the training, formatted by the UI (the core has no clock or
    /// time zone).
    fn nn_metadata(&self, _trained_at: &str) -> Result<String, CommandError> {
        Err(CommandError::not_nn())
    }
}

/// A batch prediction: the view and, if every column was found, the CSV of
/// "📥 Descargar Resultados" (all the rows).
pub struct BatchOutcome {
    pub view: BatchPredictView,
    pub csv: Option<Vec<u8>>,
}

pub type ModelHandle = Arc<dyn ModelView>;

/// Result of confirming the target of a CSV draft.
pub struct CsvFinal {
    pub dataset: DatasetHandle,
    pub resolved_task: TaskType,
    pub n_rows: usize,
    pub n_cols: usize,
    pub describe: DescribeTable,
    pub target_stats: CsvTargetStats,
}

/// A CSV that has been read but whose target is not confirmed yet.
pub trait CsvDraft: Send + Sync {
    /// Name of the file it was read from (without directories).
    fn file_name(&self) -> &str;
    /// `draft_id` is filled in by the caller.
    fn preview(&self) -> CsvPreview;
    fn target_info(&self, target: &str) -> Result<TargetInfo, CommandError>;
    /// Checks `cancel` cooperatively (`CommandError::Cancelled` when set).
    fn finalize(
        &self,
        target: &str,
        task: TaskChoice,
        cancel: &AtomicBool,
    ) -> Result<CsvFinal, CommandError>;
}

pub type CsvDraftHandle = Arc<dyn CsvDraft>;

pub trait CoreFacade: Send + Sync {
    /// Implementation name, for the log.
    fn name(&self) -> &'static str;
    /// The 12 built-in datasets, in selector order. Callers cache it (`DataState`).
    fn builtin_catalog(&self) -> Result<Vec<DatasetEntry>, CommandError>;
    /// Parses a built-in dataset. Moons is generated with seed 42 (D9); the session caches
    /// the result, so it stays fixed for the whole session.
    fn load_builtin(&self, id: DatasetId) -> Result<DatasetHandle, CommandError>;
    /// Reads a user CSV (`bytes` = whole file; size already checked). Checks `cancel`
    /// cooperatively (`CommandError::Cancelled` when set).
    fn read_csv(
        &self,
        file_name: &str,
        bytes: &[u8],
        cancel: &AtomicBool,
    ) -> Result<CsvDraftHandle, CommandError>;
}

/// `Err(Cancelled)` once `cancel` is set; for the checkpoints of the operations whose
/// core functions are not cancellable themselves.
pub fn check_cancelled(cancel: &AtomicBool) -> Result<(), CommandError> {
    if cancel.load(Ordering::Relaxed) {
        Err(CommandError::cancelled())
    } else {
        Ok(())
    }
}

/// Checks a pairplot selection against the dataset's friendly names.
pub fn validate_pairplot_features(
    features: &[String],
    available: &[String],
) -> Result<(), CommandError> {
    if features.is_empty() {
        return Err(CommandError::invalid_selection(
            "Selecciona al menos una característica para el pairplot.",
        ));
    }
    if features.len() > MAX_PAIRPLOT_FEATURES {
        return Err(CommandError::invalid_selection(format!(
            "El pairplot admite como máximo {MAX_PAIRPLOT_FEATURES} características."
        )));
    }
    for (index, feature) in features.iter().enumerate() {
        if !available.contains(feature) {
            return Err(CommandError::invalid_selection(format!(
                "La característica '{feature}' no existe en este dataset."
            )));
        }
        if features[..index].contains(feature) {
            return Err(CommandError::invalid_selection(format!(
                "La característica '{feature}' está repetida."
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_owned()).collect()
    }

    #[test]
    fn pairplot_selection_rules() {
        let available = names(&["a", "b", "c", "d", "e", "f", "g"]);
        assert!(validate_pairplot_features(&names(&["a", "b"]), &available).is_ok());
        for bad in [
            names(&[]),
            names(&["a", "b", "c", "d", "e", "f", "g"]),
            names(&["a", "zz"]),
            names(&["a", "a"]),
        ] {
            assert!(matches!(
                validate_pairplot_features(&bad, &available),
                Err(CommandError::InvalidSelection { .. })
            ));
        }
    }
}
