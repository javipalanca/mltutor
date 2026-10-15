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
//! # Public API of Phase 3 — 🌲 Árboles (fase3-arboles.md §5 and §8) — stable
//!
//! [`tree::DecisionTree`] is a [`model::Predictor`] (downcast with `as_any()` for the
//! tree-only views). Exact parity with scikit-learn 1.7.2 (D36): structure,
//! thresholds, predictions, probabilities, paths and texts bit for bit.
//!
//! | Command | Core function |
//! |---|---|
//! | `train_model` (`ModelConfig::Tree`) | [`tree::fit`]`(TreeConfig, x_train: ArrayView2, y_train: &Target, &TrainControl) -> Result<DecisionTree, ModelError>` (or [`tree::fit_boxed`] → `Box<dyn Predictor>`); [`tree::TreeConfig::new`]`(TreeTask, TreeCriterion, max_depth: u32, min_samples_split: u32)` (`random_state` = 42); D37: classification on a regression target → [`model::ModelError::ContinuousTarget`]; wrong criterion/depth → [`model::ModelError::InvalidConfig`]; options of the select: [`tree::TreeCriterion::for_task`], name [`tree::TreeCriterion::as_str`]. Cancellation per node (and per feature), progress every 64 nodes |
//! | `TrainedModelInfo.tree` | [`tree::DecisionTree::summary`]`() -> TreeSummary { node_count, n_leaves, max_depth, n_features }` |
//! | `tree_diagram` | [`tree::plot::diagram`]`(&DecisionTree, feature_names: &[String], class_names: Option<&[String]>) -> TreeDiagram { nodes: Vec<DiagramNode>, edge_labels, max_x, max_y, criterion, summary }` (C13; `class_names` by code, only for classifiers); layout alone: [`tree::layout::buchheim`] |
//! | `tree_rules` | [`tree::export_text()`]`(&DecisionTree, feature_names, &ExportTextOptions { class_names: Option<&[String]> /* D40 */, show_weights /* View: true, Export: false */, .. })` |
//! | `feature_importance(Native)` | [`importance::native`]`(&DecisionTree) -> Vec<f64>` (= [`tree::DecisionTree::feature_importances`]) |
//! | `feature_importance(Permutation)` | [`importance::permutation`]`(&dyn Predictor, x_test, y_test: &Target, PermutationOptions::default(), &TrainControl) -> Result<PermutationImportance { seed, baseline, importances: Array2 /* n_features × n_repeats */, mean, std }, ModelError>` (any model: KNN reuses it in F4; cancellable per repetition, progress per column) |
//! | rows of the chart/table | [`importance::rows`]`(names, importance, std: Option<&[f64]>) -> Vec<ImportanceRow>` (ascending, ties by column) and [`importance::ImportanceMethod`] |
//! | `predict_one` (+ `path`) | [`form::predict_one`] as in F2 (proba = leaf fractions, D46) and [`tree::DecisionTree::decision_path`]`(values: &[f64]) -> Result<DecisionPath { steps: Vec<PathStep { node, feature, value /* f32 */, threshold, goes_left }>, leaf, leaf_samples, leaf_value, leaf_class }, ModelError>` (C16) |
//! | other per-row data | [`tree::DecisionTree::apply`] (leaf of each row), [`tree::DecisionTree::nodes`], sklearn-style arrays (`children_left()`, `feature()`, `threshold()`…) |
//! | `generate_code` | new [`codegen::CodeTemplate`]s: `TreeViz`, `TreeText`, `DecisionPath` (no variables), `TreeUsage` ([`codegen::TreeUsageContext`]), `TreePython` ([`codegen::tree_python_corrected_context`]`(&tree, feature_names, class_names) -> TreePythonCorrectedContext`, D44), `TreePythonLiteral` ([`codegen::tree_python_literal_context`]), `TreeTraining` ([`codegen::TreeTrainingContext`]), `TreePackageScript`/`TreePackageReadme` (used by `mltutor-export`) |
//! | `export_model` | `mltutor-export`: ONNX `TreeEnsemble*`, Python package (`modelo_arbol.zip`); rules `.txt` = `export_text` without weights; `.py` = `TreePython` |
//!
//! # Public API of Phase 4 — 🔍 KNN (fase4-knn.md §5 and §8) — stable
//!
//! [`knn::Knn`] is a [`model::Predictor`] (downcast with `as_any()` for the KNN-only
//! views). Exact parity with scikit-learn 1.7.2 `algorithm='auto'` (D52): kd-tree
//! (libc++ `nth_element`, heap, `simultaneous_sort`) or brute force by
//! [`knn::search_engine`]; neighbours in the same order and distances bit for bit
//! (D53, FMA in the `n % 8` tail). The task is the target's.
//!
//! | Command | Core function |
//! |---|---|
//! | `train_model` (`ModelConfig::Knn`) | [`knn::fit`]`(KnnConfig { n_neighbors, weights: KnnWeights, metric: KnnMetric }, x_train: ArrayView2, y_train: &Target, &TrainControl) -> Result<Knn, ModelError>` (or [`knn::fit_boxed`]); D59: `K > n_train` → [`model::ModelError::TooFewSamples`]`{ k, n_train }`; K = 0 → `InvalidConfig`; > 50 classes → `TooManyClasses` |
//! | `TrainedModelInfo.knn` | [`knn::Knn::summary`]`() -> KnnSummary { n_neighbors, weights, metric, engine: SearchEngine, n_train, n_features }`; names with `as_str()` (`"uniform"`, `"minkowski"`, `"kd_tree"`…) |
//! | `evaluate_model`, `roc_curves`, `probability_distribution`, `threshold_analysis`, `residuals_view` | as in F2 (`predict`, `predict_proba`; cancellable: [`model::Predictor::predict_with_control`] / `predict_proba_with_control`, J1). Threshold table: thresholds `i/10` and `p > 0.5` in the 0.5 row (= `predict`) |
//! | `decision_grid` | [`grid::decision_grid`] with the **whole** dataset (`x`, `y`, D63) and [`grid::knn_step`]`(n_total)`; the 2-D model is `refit` (kd-tree) |
//! | `feature_importance(Permutation)` | [`importance::permutation`] as in F3 (≈ 0.17 s on California, progress per column, cancellable per repetition) |
//! | `class_distances` (C19) | [`knn::class_distances`]`(x_train, codes: &[usize], KnnMetric, &TrainControl) -> Result<ClassDistances { classes, counts, matrix, within /* D67 */, separation /* J2: min M[i,j] / max(W_i, W_j) */, min, max, ratio, ratio_verdict /* Python, informative */, verdict /* J2: on separation (≥ 2, ≥ 1, < 1), binary too */, metric, n_samples, sampled /* some class capped at knn::class_distances::MAX_ROWS_PER_CLASS */ }, ModelError>` (D56: the model's split; codes via `y_train.class_codes()`; cancellable, J3) |
//! | `knn_playground` (C18) | [`knn::playground`]`(x_all, y_all: &Target, &PlaygroundRequest { features: [fx, fy], config: KnnConfig, points: Vec<[f64; 2]> }, &TrainControl) -> Result<Playground { features, data /* n × 2 */, codes, classes, colors /* PALETTE[code % 8] */, bounds: [x_min, x_max, y_min, y_max], points: Vec<PlaygroundPoint { point, class, confidence, proba, neighbors, mean_distance, tie_at_kth }>, mean_confidence, engine, config }, ModelError>`; errors `InvalidSelection` (x = y), `NotClassifier`, `TooFewSamples`. Cached: [`knn::playground_model`]`(x_all, y_all, features, KnnConfig, &TrainControl) -> Result<Knn, ModelError>` once per `(features, K, weights, metric)` and [`knn::playground_points`]`(&Knn, features, &[[f64; 2]], &TrainControl) -> Result<Playground, ModelError>` per drag |
//! | `predict_one` (+ neighbours, C20/D58) | [`knn::Knn::explain`]`(values: &[f64]) -> Result<KnnExplanation { class, value, proba, neighbors: Vec<Neighbor { rank, index /* X_train row */, distance, weight, class, value }>, zero_distance_rule, tie_at_kth: Option<KthTie { n_tied, slots, distance }> }, ModelError>` (`[predicciones] vecinos_nota_empate`) |
//! | other | [`knn::Knn::kneighbors`]`(x, &TrainControl) -> Neighbors { k, distances, indices }` (row-major; parallel above 2048 rows, deterministic, cancellable every 256 rows); [`knn::Knn::train_x`], `train_codes`, `train_values`, `tree() -> Option<&KdTree>` (`idx_array`, `nodes`) |
//! | `generate_code` | [`codegen::CodeTemplate::KnnTraining`] ([`codegen::KnnTrainingContext`]); the boundary code with `datos = "todos"`: [`codegen::DecisionBoundaryContext::all_rows`] (D63; `None` renders F2/F3's text); `KnnPackageScript`/`KnnPackageReadme` (used by `mltutor-export`) |
//! | `export_model` | `mltutor-export`: ONNX (`knn.onnx`, D57) and the Python package (`modelo_knn.zip`) |
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
pub mod importance;
pub mod interpretation;
pub mod knn;
pub mod linear;
pub mod metrics;
pub mod model;
pub mod probabilities;
pub mod pyfmt;
pub mod residuals;
pub mod sampling;
pub mod split;
pub mod stats;
pub mod tree;

pub use dataset::{
    ColumnKind, CsvDialect, CsvEncoding, Dataset, DatasetEntry, DatasetId, DatasetSummary,
    LabelColumn, Target, TaskChoice, TaskType,
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
