//! Algorithms, metrics, preprocessing and embedded datasets for MLTutor (no UI).
//!
//! Pure functions over `ndarray` and serde structs: no Tauri, no I/O besides
//! [`csv::read_csv_file`], no global state. Everything random takes an explicit
//! seed (42 by default, like `random_state=42` in the Python version).
//!
//! # Public API used by the IPC contract (fase1-shell-datos.md §5.4–5.5)
//!
//! This list is **stable**: `src-tauri` programs against it and converts these
//! domain types into its DTOs (`src-tauri/src/ipc/`). Session concepts
//! (`DatasetRef`, `CsvId`, `draft_id`) live in `src-tauri`.
//!
//! | Command | Core function |
//! |---|---|
//! | `list_datasets` | [`dataset::list_builtin`]`() -> Result<Vec<DatasetEntry>, DatasetError>` (12, selector order) |
//! | `select_dataset`, `dataset_summary` | [`dataset::load_builtin`]`(DatasetId) -> Result<Dataset, DatasetError>`, [`Dataset::summary`]`() -> DatasetSummary` |
//! | Moons (D9) | [`dataset::moons::moons_dataset`]`(seed: u32) -> Result<Dataset, DatasetError>` (`42` = embedded); [`dataset::make_moons`]`(n, noise, seed) -> (Array2<f64>, Vec<u32>)` |
//! | `dataset_preview` | [`explore::preview`]`(&Dataset, rows: usize) -> TablePreview` |
//! | `dataset_describe` | [`explore::describe`]`(&Dataset) -> DescribeTable` |
//! | `target_distribution` | [`explore::target_distribution`]`(&Dataset) -> TargetDistribution` |
//! | `correlation_matrix` | [`explore::correlation_matrix`]`(&Dataset) -> CorrelationMatrix` |
//! | pairplot options | [`explore::pairplot_defaults`]`(&Dataset) -> PairplotDefaults` |
//! | `pairplot_data` | [`explore::pairplot_data`]`(&Dataset, features: &[String], DiagKind) -> Result<PairplotData, DatasetError>` |
//! | `load_csv` | [`csv::read_csv_file`]`(&Path) -> Result<RawTable, CsvError>` (or [`csv::read_csv`]`(&[u8])`), then [`csv::RawTable::preview`]`() -> Result<CsvPreview, CsvError>` |
//! | `csv_target_info` | [`csv::RawTable::target_info`]`(&str) -> Result<TargetInfo, CsvError>` |
//! | `set_csv_target` | [`csv::RawTable::to_dataset`]`(target: &str, TaskChoice, file_name: &str) -> Result<Dataset, CsvError>`, [`csv::RawTable::describe`]`() -> DescribeTable`, [`csv::RawTable::target_stats`]`(&str, TaskChoice) -> Result<CsvTargetStats, CsvError>`; the resolved task is [`Dataset::task`] |
//! | cancellation | [`csv::read_csv_cancellable`]`(&[u8], &AtomicBool)`, [`csv::read_csv_file_cancellable`]`(&Path, &AtomicBool)`, [`csv::RawTable::to_dataset_cancellable`]`(…, &AtomicBool)` → [`CsvError::Cancelled`]; [`explore::correlation_matrix_cancellable`]`(&Dataset, &AtomicBool) -> Result<CorrelationMatrix, DatasetError>` → [`DatasetError::Cancelled`] |
//! | F2+ (D7) | [`split::train_test_split`]`(n_rows, test_size: f64, seed: u32) -> Result<TrainTestSplit, DatasetError>` |
//!
//! # Public API of Phase 2 (fase2-regresion.md §5.4 and §8) — stable
//!
//! Every model is a [`model::Predictor`] (`Box<dyn Predictor>`, `Send + Sync`;
//! wrap it in an `Arc`). Class labels are dataset codes; [`model::Predictor::classes`]
//! gives the code of each `predict_proba` column. Long work takes a
//! [`model::TrainControl`] (`TrainControl::none().with_cancel(&flag).with_progress(&cb)`)
//! and returns [`model::ModelError::Cancelled`].
//!
//! | Command | Core function |
//! |---|---|
//! | `train_model` | [`split::train_test_split`] then [`linear::fit`]`(ModelConfig, x_train: ArrayView2, y_train: &Target, &TrainControl) -> Result<Box<dyn Predictor>, ModelError>` ([`Target::select`]`(&rows)` for `y`); `ModelConfig::{Linear, Logistic { max_iter }}`; D26 → `ModelError::InvalidTarget`; D19 → [`model::Predictor::fit_report`]`() -> Option<FitReport { n_iter, max_iter, converged }>` |
//! | `evaluate_model` | classification: [`metrics::classification`]`(y_true: &[usize], y_pred: &[usize], n_classes) -> ClassificationEvaluation` + [`interpretation::classification`]`(&eval) -> Result<String, ContentError>`; regression: [`metrics::regression`]`(y_true: &[f64], y_pred: &[f64]) -> RegressionEvaluation` + [`interpretation::regression`]; warnings: [`metrics::MetricWarning::message_es`] |
//! | `roc_curves` | [`curves::binary_curves`]`(y_true: &[usize], p_positive: &[f64], positive_class) -> BinaryCurves`; [`curves::multiclass_curves`]`(y_true, proba: ArrayView2, classes: &[usize]) -> MulticlassCurves` (thresholds: first ROC one is `f64::INFINITY`) |
//! | `probability_view` | [`probabilities::distribution`]`(y_true, proba, classes) -> ProbabilityView` (binary includes [`probabilities::threshold_analysis`]) |
//! | `residuals_view` | [`residuals::analysis`]`(y_true: &[f64], y_pred: &[f64]) -> ResidualAnalysis` |
//! | `decision_grid` | [`grid::decision_grid`]`(&dyn Predictor, x_train, &y_train, fx, fy, GridOptions, &TrainControl) -> Result<DecisionGrid, ModelError>`; axes only: [`grid::decision_axes`]; KNN step [`grid::knn_step`] |
//! | `coefficients_view` | [`coefficients::view`]`(&dyn Predictor, feature_names: &[String], class_index: Option<usize>) -> Result<CoefficientsView, ModelError>` |
//! | `prediction_form` | [`form::build`]`(x_train: ArrayView2, &Dataset, Option<&DatasetMetadata>) -> PredictionForm` |
//! | `predict_one` | [`form::predict_one`]`(&dyn Predictor, values: &[f64]) -> Result<PredictionOutcome, ModelError>` |
//! | `generate_code` | [`codegen::render`]`(CodeTemplate, &impl Serialize) -> Result<String, CodegenError>` ([`codegen::DecisionBoundaryContext`], [`codegen::RegressionTrainingContext`]); [`codegen::render_static`]; any `content/code/*.j2` by name: [`codegen::render_named`] |
//! | `export_model` | `mltutor-export` (see its crate docs) |
//!
//! Python-style formatting of the TOML messages: [`pyfmt::py_format`]; embedded
//! TOML: [`content::Messages`].
//!
//! Other public modules: [`sampling`] (NumPy `RandomState`, `DataFrame.sample`),
//! [`stats`] (describe, value counts, histograms, Pearson, KDE) and [`pyfmt`].
//!
//! Matrices are `ndarray::Array2<f64>` with NaN for missing values; the DTOs turn
//! them into the contract's `Matrix { rows, cols, values: Vec<Option<f64>> }`.

pub mod codegen;
pub mod coefficients;
pub mod content;
pub mod csv;
pub mod curves;
pub mod dataset;
pub mod error;
pub mod explore;
pub mod form;
pub mod frame;
pub mod grid;
pub mod interpretation;
pub mod linear;
pub mod metrics;
pub mod model;
pub mod probabilities;
pub mod pyfmt;
pub mod residuals;
pub mod sampling;
pub mod split;
pub mod stats;

pub use dataset::{
    ColumnKind, Dataset, DatasetEntry, DatasetId, DatasetSummary, LabelColumn, Target, TaskChoice,
    TaskType,
};
pub use error::{ContentError, CsvError, DatasetError};
pub use frame::{Cell, DescribeTable, TablePreview};

/// Version of the MLTutor workspace this crate was built from.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_matches_workspace() {
        assert!(VERSION.starts_with("0.4.0"));
    }
}
