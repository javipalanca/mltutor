//! Payloads of the model commands (fase2-regresion.md §5.3), shared by the four lessons:
//! training, evaluation (C1–C3), curves (C4), probabilities and thresholds (C5–C6),
//! residuals (C7), decision grids (C8), coefficients, the prediction form (C9), exports
//! (C11) and generated code (C12). snake_case on the wire; tagged enums use `kind`.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use super::types::{DatasetRef, KdeCurve, Matrix, TaskType};

/// Lessons that train models. Each one keeps at most one trained model (§5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum LessonId {
    Trees,
    Regression,
    Knn,
    Nn,
}

/// Task of a decision tree ("🏷️ Clasificación" / "📈 Regresión" of fase3-arboles.md §2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum TreeTask {
    Classification,
    Regression,
}

impl TreeTask {
    pub fn task(self) -> TaskType {
        match self {
            Self::Classification => TaskType::Classification,
            Self::Regression => TaskType::Regression,
        }
    }
}

/// `criterion` of scikit-learn (same names on the wire as in Python).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum TreeCriterion {
    Gini,
    Entropy,
    SquaredError,
    FriedmanMse,
    AbsoluteError,
}

impl TreeCriterion {
    /// The name of scikit-learn (`plot_tree` prints it in every box).
    pub fn sklearn_name(self) -> &'static str {
        match self {
            Self::Gini => "gini",
            Self::Entropy => "entropy",
            Self::SquaredError => "squared_error",
            Self::FriedmanMse => "friedman_mse",
            Self::AbsoluteError => "absolute_error",
        }
    }

    /// Whether the criterion belongs to `task` (gini/entropy classify, the rest regress).
    pub fn fits(self, task: TreeTask) -> bool {
        matches!(
            (self, task),
            (Self::Gini | Self::Entropy, TreeTask::Classification)
                | (
                    Self::SquaredError | Self::FriedmanMse | Self::AbsoluteError,
                    TreeTask::Regression
                )
        )
    }
}

/// Hyperparameters of a model. F4–F5 add their own variants.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum ModelConfig {
    Linear,
    /// `LogisticRegression(max_iter, random_state=42)`; 100..=2000, step 100.
    Logistic {
        max_iter: u32,
    },
    /// `DecisionTree{Classifier,Regressor}(criterion, max_depth, min_samples_split,
    /// random_state=42)` with the other hyperparameters at their defaults.
    Tree {
        task: TreeTask,
        criterion: TreeCriterion,
        /// 1..=10
        max_depth: u8,
        /// 2..=10
        min_samples_split: u8,
    },
}

impl ModelConfig {
    /// The task the model solves (the evaluation follows it, not the dataset's).
    pub fn task(self) -> TaskType {
        match self {
            Self::Linear => TaskType::Regression,
            Self::Logistic { .. } => TaskType::Classification,
            Self::Tree { task, .. } => task.task(),
        }
    }

    pub fn is_tree(self) -> bool {
        matches!(self, Self::Tree { .. })
    }
}

/// Size of a trained tree (`tree_.node_count`, `get_n_leaves()`, `get_depth()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct TreeSummary {
    pub node_count: u32,
    pub n_leaves: u32,
    pub max_depth: u32,
    pub n_features: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, TS)]
#[ts(export)]
pub struct TrainRequest {
    pub lesson: LessonId,
    pub dataset: DatasetRef,
    pub config: ModelConfig,
    /// 0.1..=0.5
    pub test_size: f64,
    /// Newest request wins (like `select_dataset`).
    #[ts(type = "number")]
    pub seq: u64,
    pub task_id: u32,
}

/// Pedagogical notices produced by training (shown in 🏋️ Entrenamiento).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum TrainNotice {
    /// D19: L-BFGS stopped at `max_iter` before converging.
    NotConverged { max_iter: u32, message: String },
    /// The model type does not match the task of the dataset (warning, as in Python).
    ModelTaskMismatch { message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct TrainedModelInfo {
    pub model_id: u32,
    pub lesson: LessonId,
    /// The dataset it was trained on (D32: it may differ from the current selection).
    pub dataset: DatasetRef,
    pub dataset_label: String,
    pub config: ModelConfig,
    pub test_size: f64,
    /// The model's task (decides the evaluation and the visualisations).
    pub model_task: TaskType,
    /// The dataset's task (decides the prediction form branch in Python).
    pub dataset_task: TaskType,
    pub n_train: u32,
    pub n_test: u32,
    /// `X.columns`.
    pub feature_columns: Vec<String>,
    /// Friendly names (tables, selectors, charts).
    pub feature_names: Vec<String>,
    /// Classes of the dataset (indexed by class code).
    pub class_names: Option<Vec<String>>,
    /// `classes_` of a classifier: the codes (indices into `class_names`) seen in the
    /// train set, sorted. Row `i` of `coef_` belongs to `model_classes[i]`; with 2 of 3
    /// classes the model is binary and `model_classes[1]` is the positive class.
    pub model_classes: Option<Vec<u32>>,
    pub notices: Vec<TrainNotice>,
    /// Decision trees: size of the tree (💾 Exportar, "Información del código generado").
    pub tree: Option<TreeSummary>,
}

/// Progress of a long task (event `task-progress`).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct TaskProgress {
    pub task_id: u32,
    pub message: String,
    /// 0..=100, `None` when unknown.
    pub percent: Option<f32>,
}

// --- C1–C3: evaluation ---

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ClassMetrics {
    pub name: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: u32,
}

/// One row of the "correctly / incorrectly classified" tables.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct SampleRow {
    /// Position in the test set (the index shown by Python after `reset_index`).
    pub position: u32,
    pub real: String,
    pub predicted: String,
    pub correct: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ClassificationEvaluation {
    pub accuracy: f64,
    /// labels = 0..n_classes (a class absent from the test set has support 0).
    pub per_class: Vec<ClassMetrics>,
    pub macro_avg: ClassMetrics,
    pub weighted_avg: ClassMetrics,
    /// rows = real, columns = predicted.
    pub confusion: Matrix,
    pub correct_head: Vec<SampleRow>,
    pub incorrect_head: Vec<SampleRow>,
    /// "### 🔍 Interpretación de Resultados" body, from `componentes/mensajes.toml`.
    pub interpretation_md: String,
    /// E.g. undefined precision of a class that is never predicted (`zero_division`).
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct Histogram {
    pub edges: Vec<f64>,
    pub counts: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct RegressionEvaluation {
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub y_true: Vec<f64>,
    pub y_pred: Vec<f64>,
    /// `y − ŷ`, `bins="auto"`.
    pub errors_hist: Histogram,
    /// `histplot(kde=True)`: Scott, cut 0, scaled to counts.
    pub errors_kde: Option<KdeCurve>,
    pub interpretation_md: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum Evaluation {
    Classification(ClassificationEvaluation),
    Regression(RegressionEvaluation),
}

// --- C4: ROC / PR ---

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct Curve {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// `None` inside = +∞ (first ROC threshold).
    pub thresholds: Option<Vec<Option<f64>>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ClassCurves {
    pub class_index: u32,
    pub name: String,
    /// x = FPR, y = TPR.
    pub roc: Curve,
    pub auc: f64,
    /// x = recall, y = precision.
    pub pr: Curve,
    pub ap: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum RocData {
    Binary {
        /// Name of class 1 (the positive one; *benign* in Cáncer).
        positive_class: String,
        roc: Curve,
        auc: f64,
        pr: Curve,
        ap: f64,
        /// Share of positives in the test set.
        baseline: f64,
    },
    Multiclass {
        per_class: Vec<ClassCurves>,
        micro: ClassCurves,
        macro_roc: Curve,
        macro_auc: f64,
        macro_ap: f64,
        /// Computed for F3/F4, not shown (D20).
        weighted_auc: Option<f64>,
        /// Classes of the model that do not appear in the test set (D20).
        absent_classes: Vec<String>,
    },
}

// --- C5–C6: probabilities and thresholds ---

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ClassHistogram {
    pub class_index: u32,
    pub name: String,
    pub histogram: Histogram,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ThresholdRow {
    pub threshold: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum ProbabilityView {
    Binary {
        /// Histogram of p(class 1) for each real class (20 bins over the group's range).
        per_real_class: Vec<ClassHistogram>,
        /// 0.3 < p < 0.7
        uncertain: u32,
        /// p < 0.2 or p > 0.8
        confident: u32,
        n: u32,
        /// Every threshold of `np.arange(0.1, 1, 0.1)` with both classes predicted.
        thresholds: Vec<ThresholdRow>,
        best_f1_threshold: Option<f64>,
        /// D21: 0.3, 0.5 and 0.7 (those present in `thresholds`).
        shown_thresholds: Vec<f64>,
    },
    Multiclass {
        /// Histogram of `proba[:, i]` for each class i (20 bins).
        per_class: Vec<ClassHistogram>,
    },
}

// --- C7: residuals ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct ResidualChecks {
    /// |mean| < 0.1·std
    pub mean_ok: bool,
    /// |r| < 0.1 (`None` with ≤ 10 test samples: the bullet is omitted).
    pub corr_ok: Option<bool>,
    /// |skew| < 1
    pub normal_ok: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ResidualsView {
    pub y_true: Vec<f64>,
    pub y_pred: Vec<f64>,
    /// D22: the real R² of the test set.
    pub r2: f64,
    pub mean: f64,
    /// ddof = 1
    pub std: f64,
    pub skew: f64,
    pub linregress_r: Option<f64>,
    /// 20 bins of the residuals.
    pub hist: Histogram,
    pub checks: ResidualChecks,
}

// --- C8: decision boundary / prediction surface ---

#[derive(Debug, Clone, PartialEq, Deserialize, TS)]
#[ts(export)]
pub struct GridRequest {
    pub model_id: u32,
    pub feature_x: u32,
    pub feature_y: u32,
    pub task_id: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum GridNotice {
    /// "Malla creada: {rows} x {cols} = {n:,} puntos" (rows = points of y).
    Created { rows: u32, cols: u32 },
    /// "⚠️ La malla es muy grande ({points:,} puntos). Reduciendo resolución..."
    TooLarge {
        #[ts(type = "number")]
        points: u64,
    },
    /// "Nueva resolución: {rows} x {cols} = {n:,} puntos"
    Resized { rows: u32, cols: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum GridValues {
    /// Predicted class per cell, row-major (rows = y).
    Classes { z: Vec<u16> },
    /// Predicted value per cell, row-major.
    Values { z: Vec<f64>, min: f64, max: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct DecisionGrid {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub values: GridValues,
    pub notices: Vec<GridNotice>,
    /// Training points, 2 columns.
    pub points: Matrix,
    /// Target value of each point (regression colours; the class *value* in
    /// classification, e.g. 3 or 6).
    pub point_target: Vec<f64>,
    /// Classification: class code of each point (index into `class_names`); the UI
    /// colours the points with it. `None` for regressors.
    pub point_codes: Option<Vec<u32>>,
    pub feature_names: [String; 2],
    pub class_names: Option<Vec<String>>,
    /// More than 2 features: the 2D model is a simplification.
    pub show_2d_warning: bool,
}

// --- Coefficients (§2.5) ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Effect {
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Level {
    VeryHigh,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct CoefficientRow {
    pub feature: String,
    /// Original position of the feature (0-based).
    pub feature_position: u32,
    pub coef: f64,
    pub abs: f64,
    /// Logistic: e^coef.
    pub odds_ratio: Option<f64>,
    /// Logistic: (OR − 1)·100.
    pub change_pct: Option<f64>,
    pub effect: Effect,
    /// Linear: "Importancia" (High = 🔥 Alta, Medium = 🔶 Media, Low = 🔹 Baja).
    pub importance: Option<Level>,
    /// Logistic: "Fuerza del Efecto" (D27: symmetric by |β|).
    pub strength: Option<Level>,
    /// "Análisis detallado": FUERTE / MODERADO / DÉBIL.
    pub intensity: Level,
}

/// Linear quantiles of |coef| (pandas `quantile`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct AbsQuantiles {
    pub q25: f64,
    pub q60: f64,
    pub q75: f64,
    pub q80: f64,
    pub q95: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct CoefficientsView {
    /// Sorted by |coef| descending.
    pub rows: Vec<CoefficientRow>,
    /// Row of `coef_` shown (logistic multiclass).
    pub class_index: Option<u32>,
    /// `classes_` of the model (codes into the dataset's `class_names`); see
    /// `TrainedModelInfo::model_classes`.
    pub model_classes: Option<Vec<u32>>,
    /// Rows of `coef_`: 1 for linear and binary logistic, one per class otherwise.
    pub n_rows: u32,
    /// D27: of the chosen class.
    pub intercept: f64,
    /// `e^intercept` (binary logistic: the "Odds base" metric).
    pub intercept_odds: Option<f64>,
    pub quantiles: AbsQuantiles,
    /// Features with |coef| > q95.
    pub extreme: Vec<String>,
}

// --- C9: prediction form ---

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct SelectOption {
    pub label: String,
    /// Value the model receives (e.g. "Mujer" → 0).
    pub code: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum FormControlKind {
    Checkbox {
        default: bool,
    },
    Select {
        options: Vec<SelectOption>,
        default_index: u32,
    },
    Slider {
        min: f64,
        max: f64,
        step: f64,
        default: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct FormControl {
    pub index: u32,
    pub column: String,
    /// Friendly name.
    pub label: String,
    pub description: Option<String>,
    pub control: FormControlKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct PredictionForm {
    pub controls: Vec<FormControl>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, TS)]
#[ts(export)]
pub struct PredictRequest {
    pub model_id: u32,
    /// Codes, in column order.
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ClassProbability {
    pub name: String,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct FeatureCoefficient {
    pub feature: String,
    pub coef: f64,
}

/// One internal node of the decision path (C16): `value ≤ threshold` goes left.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct PathStep {
    pub node: u32,
    pub feature: u32,
    /// Friendly name of the feature.
    pub name: String,
    /// The input converted to float32 (what scikit-learn compares).
    pub value: f64,
    pub threshold: f64,
    pub goes_left: bool,
    /// D46: label of a categorical value ("0 (Mujer)"); `None` for numbers.
    pub display: Option<String>,
}

/// `decision_path` of one sample (C16) and the leaf it reaches (D46).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct DecisionPath {
    pub steps: Vec<PathStep>,
    pub leaf: u32,
    /// Training samples of the leaf.
    pub leaf_samples: u32,
    /// Class fractions (classification) or `[value]` (regression) of the leaf.
    pub leaf_value: Vec<f64>,
    /// Classification: the predicted class code (`argmax`, first on ties).
    pub leaf_class: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum PredictionResult {
    Class {
        index: u32,
        name: String,
        /// Logistic: `predict_proba`; trees: the fractions of the leaf (D46).
        proba: Option<Vec<ClassProbability>>,
        /// Decision trees (C16).
        path: Option<DecisionPath>,
    },
    /// D25: also for a linear model trained on a classification dataset.
    Value {
        value: f64,
        /// Linear regression: "### Coeficientes" table.
        coefficients: Option<Vec<FeatureCoefficient>>,
        /// Decision trees (C16).
        path: Option<DecisionPath>,
    },
}

// --- C13: tree diagram ---

/// One box of `plot_tree` (C13), in pre-order (box i = node i of `tree_`).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct TreeDiagramNode {
    pub id: u32,
    pub parent: Option<u32>,
    /// Left child of its parent (the "True" branch).
    pub is_left: bool,
    pub depth: u32,
    /// Buchheim coordinates, not scaled: `plot_tree` draws the box at
    /// `((x + 0.5) / max_x, (max_y − depth − 0.5) / max_y)`.
    pub x: f64,
    pub y: f64,
    /// Text of the box, already formatted as `plot_tree` (one entry per line).
    pub lines: Vec<String>,
    /// `"#rrggbb"` (`filled=True`).
    pub fill: String,
    pub leaf: bool,
    pub feature: Option<u32>,
    pub threshold: Option<f64>,
    pub impurity: f64,
    pub n_samples: u32,
    /// Class fractions (classification) or `[value]` (regression).
    pub value: Vec<f64>,
    /// Classification: samples of each class (`value × n_samples`).
    pub value_counts: Option<Vec<u32>>,
    /// Classification: `argmax(value)` (first on ties).
    pub class_index: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct TreeDiagram {
    pub nodes: Vec<TreeDiagramNode>,
    pub max_x: f64,
    pub max_y: f64,
    /// `gini`, `entropy`, `squared_error`…
    pub criterion: String,
    pub summary: TreeSummary,
    /// Classification: names of the classes (tooltip).
    pub class_names: Option<Vec<String>>,
}

// --- C14: rules text ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum RulesKind {
    /// 📝 Texto: `export_text(show_weights=True)`.
    View,
    /// 💾 Exportar: `export_text` without weights (`decision_tree_rules.txt`).
    Export,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, TS)]
#[ts(export)]
pub struct RulesRequest {
    pub model_id: u32,
    pub kind: RulesKind,
    /// D40: class names instead of the codes of `classes_`.
    pub class_names: bool,
}

// --- C10: feature importance ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ImportanceMethod {
    /// `feature_importances_` (decision trees).
    Native,
    /// `permutation_importance(n_repeats=10, random_state=42)` on the test set.
    Permutation,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ImportanceRow {
    /// Position of the feature (the index of the Python table).
    pub feature_index: u32,
    pub name: String,
    pub importance: f64,
    /// Permutation: standard deviation over the repeats (ddof 0).
    pub std: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct FeatureImportanceView {
    pub method: ImportanceMethod,
    /// Ascending by importance, ties by feature index (the order of the bars).
    pub rows: Vec<ImportanceRow>,
    pub n_repeats: Option<u32>,
    /// Permutation: score of the unpermuted test set (accuracy or R²).
    pub baseline: Option<f64>,
    /// Permutation: the single seed of every column (sklearn ≥ 1.5).
    #[ts(type = "number | null")]
    pub seed: Option<u64>,
    /// The task of the model (decides the explanation texts).
    pub task: TaskType,
}

// --- C11–C12: export and code ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ExportFormat {
    Onnx,
    PythonPackage,
    /// Decision trees: `export_text` without weights (`decision_tree_rules.txt`).
    TreeRules,
    /// Decision trees: the corrected Python function of the tree (D44, `arbol_decision.py`).
    TreePython,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum CodeTemplate {
    ConfusionMatrix,
    PrecisionByClass,
    PredVsReal,
    ErrorDistribution,
    DecisionBoundary,
    /// D29: "📄 Generar Código" of 💾 Exportar.
    RegressionCode,
    /// 🌲 Árbol: "Código para visualizar el árbol".
    TreeViz,
    /// 📝 Texto: "Código para obtener el texto del árbol".
    TreeText,
    /// 🔮 Predicciones: "Código para generar el camino de decisión".
    DecisionPath,
    /// 💾 Exportar: "Código para Usar el Modelo".
    TreeUsage,
    /// 💾 Exportar: the tree as a Python function, corrected (D44).
    TreePython,
    /// The literal text of Python's `export_tree_as_python_code` (it does not compile).
    TreePythonLiteral,
    /// 💾 Exportar: "📄 Código de Entrenamiento" of the tree (`arbol_entrenamiento.py.j2`).
    TreeTraining,
}

impl CodeTemplate {
    /// File of `content/code/` and default download name.
    pub fn file_stem(self) -> &'static str {
        match self {
            Self::ConfusionMatrix => "matriz_confusion",
            Self::PrecisionByClass => "precision_por_clase",
            Self::PredVsReal => "predicciones_vs_reales",
            Self::ErrorDistribution => "distribucion_errores",
            Self::DecisionBoundary => "frontera_decision",
            Self::RegressionCode => "regresion_codigo",
            Self::TreeViz => "visualizar_arbol",
            Self::TreeText => "texto_arbol",
            Self::DecisionPath => "camino_decision",
            Self::TreeUsage => "usar_modelo",
            Self::TreePython | Self::TreePythonLiteral => "arbol_decision",
            Self::TreeTraining => "arbol_decision_entrenamiento",
        }
    }

    /// Templates that need a decision tree.
    pub fn needs_tree(self) -> bool {
        matches!(
            self,
            Self::TreeViz
                | Self::TreeText
                | Self::DecisionPath
                | Self::TreeUsage
                | Self::TreePython
                | Self::TreePythonLiteral
                | Self::TreeTraining
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, TS)]
#[ts(export)]
pub struct CodeExtra {
    pub feature_x: u32,
    pub feature_y: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configs_and_lessons_are_tagged() {
        assert_eq!(
            serde_json::to_value(ModelConfig::Logistic { max_iter: 1000 }).unwrap(),
            serde_json::json!({ "kind": "logistic", "max_iter": 1000 })
        );
        let linear: ModelConfig = serde_json::from_str(r#"{"kind":"linear"}"#).unwrap();
        assert_eq!(linear.task(), TaskType::Regression);
        assert_eq!(
            serde_json::to_value(LessonId::Regression).unwrap(),
            "regression"
        );
        assert_eq!(
            serde_json::to_value(ExportFormat::PythonPackage).unwrap(),
            "python_package"
        );
    }

    #[test]
    fn curves_serialize_infinite_thresholds_as_null() {
        let curve = Curve {
            x: vec![0.0, 1.0],
            y: vec![0.0, 1.0],
            thresholds: Some(vec![None, Some(0.5)]),
        };
        assert_eq!(
            serde_json::to_value(curve).unwrap()["thresholds"],
            serde_json::json!([null, 0.5])
        );
    }
}
