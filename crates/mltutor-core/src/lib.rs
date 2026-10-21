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
//! | cancellation | [`csv::read_csv_cancellable`]`(&[u8], &AtomicBool)` (or [`csv::read_csv_cancellable_with`]`(&[u8], &AtomicBool, ReadOptions { min_columns })`: 1 for the batch of a one-feature network; repeated headers are reported in `CsvNotice::RenamedHeaders { renamed: Vec<(original, new)> }`), [`csv::read_csv_file_cancellable`]`(&Path, &AtomicBool)`, [`csv::RawTable::to_dataset_cancellable`]`(…, &AtomicBool)` → [`CsvError::Cancelled`]; [`explore::correlation_matrix_cancellable`]`(&Dataset, &AtomicBool) -> Result<CorrelationMatrix, DatasetError>` → [`DatasetError::Cancelled`] |
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
//! # Public API of Phase 5 — 🧠 Redes Neuronales (fase5-redes.md §5 and §8) — stable
//!
//! The engine is `mltutor-nn` (own float32 MLP, D68); the core has the lesson types,
//! the data preparation (C22), the diagnostics (C23) and the view statistics (C25/C26).
//!
//! | Use | Core function |
//! |---|---|
//! | types | [`nn_data::NnArchitecture`], [`nn_data::NnTraining`] (`seed`, D70), [`nn_data::Activation`], [`nn_data::OptimizerKind`], [`nn_data::LossKind`], [`nn_data::MetricKind`], [`nn_data::History`]`{ metric, epochs: Vec<EpochLog { epoch /* 0-based: the UI shows epoch + 1 */, loss, metric, val_loss, val_metric, learning_rate }>, best_epoch, stopped_epoch, diverged_epoch /* K1 */ }` (`lr_drops()`, `series()`; non-finite `f64`s serialise as `"nan"`/`"inf"` via [`serde_nonfinite`]); errors [`nn_data::NnError`] (`UntrainableOutput { key }` D71, `TooFewPerClass`, `SplitTooSmall`, `ValidationTooSmall`, `Cancelled`…) |
//! | C22 | [`nn_data::stratified_split`]`(codes, class_names, 0.2, 42)`, [`nn_data::lesson_split`], [`nn_data::Scaler`]`::fit/transform/transform_f32/transform_rows_f32` (all rows, D72; sklearn's constant-feature rule), [`nn_data::keras_validation_split`], [`nn_data::resolve_training`]`(task, output, output_size)`, [`nn_data::output_size`] |
//! | `nn_architecture` | [`nn_data::total_parameters`], [`nn_data::layer_rows`] (D76), [`nn_data::complexity`], [`nn_data::suggested_neurons`] + [`nn_data::default_neurons`], [`nn_data::default_epochs`], [`nn_data::split_sizes`] (D73), [`nn_data::output_check`] (`[arquitectura]` key + severity + `untrainable_key`), [`nn_data::Activation::output_options`] / `recommended_output` (D85); cost warning (K6) [`nn_data::training_cost`]`(total_params, n_fit, epochs) -> TrainingCost { work, estimated_seconds, level: CostLevel }` |
//! | `nn_diagnostics` (C23) | [`nn_diagnostics::diagnostics_view`]`(&history.series(), task, n_val) -> DiagnosticsView { diagnostics, trends, recommendations, blocks: Vec<view::ViewNode> }` (texts filled in from `content/redes/mensajes.toml`) |
//! | `nn_history` cards | [`nn_diagnostics::history_cards`]`(&series) -> HistoryCards { …, row: ViewNode }`; help text [`nn_diagnostics::history_help_markdown`] |
//! | `evaluate_model` D84 | [`nn_diagnostics::r2_negative_notice`]`(r2, learning_rate) -> Result<Option<String>, ContentError>` (`aviso_r2_negativo_lr_alto` when LR ≥ 0.1); [`nn_diagnostics::near_divergence_notice`]`(history.near_divergence)` (`aviso_casi_divergencia`) |
//! | `nn_weights` (C25, K2) | [`nn_stats::weights_view`]`(&[DenseView], weight_bins, bias_bins) -> WeightsView { stats: WeightStats { layers, global_*, verdict, insights }, histograms }`; bins as Plotly's `nbinsx` ([`stats::histogram_plotly`]) |
//! | `nn_activations` (C26, K2) | [`nn_stats::activations_view`]`(&[DenseView], hidden, rows_used, &ActivationParams) -> ActivationsView { rows_used, layers: [{ stats, histogram }], dead_pct, map }` (parts: [`nn_stats::activation_stats`], [`nn_stats::dead_pct`], [`nn_stats::network_map`]) |
//! | `decision_grid` (D78) | [`grid::mean_grid`]`(&dyn Predictor, x_test, y_test, &Scaler, fx, fy, &TrainControl) -> MeanGrid { grid: DecisionGrid, fixed }` (original units, ≤ 200 × 200) |
//! | `generate_code(NnTraining)` (D74) | [`codegen::NnTrainingContext::new`]`(ScriptDataset, task, n_classes, &arch, &training)` (the only `NnTrainingContext`, L1; `seed: u64` = `training.seed`, D70, rendered in `keras.utils.set_random_seed({{ seed }})`) + [`codegen::CodeTemplate::NnTraining`]; package templates `NnPackageScript`, `NnPackageReadme`, `NnNumpyPredictor` (used by `mltutor-export`) |
//! | `predict_one` (🔮 individual) | [`form::predict_one`]`(&network, values)` as in F2: the network scales inside; `proba` = output (`[1 − p, p]` with one neuron) |
//! | `nn_explore` (C29, D81, L2) | [`nn_predict::nn_explore`]`(&dyn NetworkOutput, x_dataset /* whole dataset, original units */, sample_index, feature, n_points /* 50; 2..=1000 */, &TrainControl) -> Result<ExploreView { sample_index, feature, base, base_value, base_prediction: RowPrediction, kind: OutputKind, output_is_probability, xs /* linspace(min, max) of the WHOLE dataset, as Python */, curve: ExploreCurve::{Binary { probability } \| Multiclass { classes, confidence, base_class, base_class_probability /* NEW */ } \| Regression { values }}, stats: ExploreStats { range /* f32 */, std, mean, relative_impact_pct }, influence: Influence::{High, Moderate, Low} }, ModelError>`; thresholds [`nn_predict::probability_influence`] (0.3/0.1, also on the base-class probability in multiclass, NEW) and [`nn_predict::regression_influence`] (20 %/5 %). `mltutor_nn::NeuralNetwork::explore` is the same call |
//! | `nn_predict_batch` (C29, L2) | [`nn_predict::batch_columns`]`(features /* X.columns */, file_columns) -> BatchColumns { positions, missing, extra }`, then [`nn_predict::nn_predict_batch`]`(&dyn NetworkOutput, rows: ArrayView2 /* model column order */, &TrainControl) -> Result<BatchView { rows, kind, classes, output_is_probability, finite: Vec<bool>, predictions: BatchPredictions::{Binary { predicted: Vec<Option<usize>>, probability, confidence } \| Multiclass { predicted: Vec<Option<usize>>, confidence, probabilities } \| Regression { values }} }, ModelError>` (≤ [`nn_predict::BATCH_MAX_ROWS`], chunks of 4096 with a cancellation check; NaN, ∞ or `\|v\| > f32::MAX` → `ModelError::NonFiniteCell { row, column }` ([`nn_predict::check_f32_inputs`]); a row with a non-finite output has `finite[i] = false` and class `None`); `NeuralNetwork::predict_batch` |
//! | out-of-range warnings | [`nn_predict::training_ranges`]`(x, train_rows) -> Vec<FeatureRange { min, max }>` (unscaled training rows; `TrainedNetwork::training_ranges(x)`) and [`nn_predict::out_of_range`]`(rows, &ranges) -> Vec<(row, columns)>` |
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
pub mod nn_data;
pub mod nn_diagnostics;
pub mod nn_predict;
pub mod nn_stats;
pub mod probabilities;
pub mod pyfmt;
pub mod residuals;
pub mod sampling;
pub mod serde_nonfinite;
pub mod split;
pub mod stats;
pub mod tree;
pub mod view;

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
        // The first `version = "…"` of the workspace manifest is `[workspace.package]`'s.
        let manifest = include_str!("../../../Cargo.toml");
        let workspace = manifest
            .lines()
            .find_map(|l| l.strip_prefix("version = \"")?.strip_suffix('"'));
        assert_eq!(Some(VERSION), workspace);
    }
}
