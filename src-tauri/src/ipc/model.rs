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

/// Hyperparameters of a model. F3–F5 add their own variants.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum ModelConfig {
    Linear,
    /// `LogisticRegression(max_iter, random_state=42)`; 100..=2000, step 100.
    Logistic {
        max_iter: u32,
    },
}

impl ModelConfig {
    /// The task the model solves (the evaluation follows it, not the dataset's).
    pub fn task(self) -> TaskType {
        match self {
            Self::Linear => TaskType::Regression,
            Self::Logistic { .. } => TaskType::Classification,
        }
    }
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

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum PredictionResult {
    Class {
        index: u32,
        name: String,
        proba: Option<Vec<ClassProbability>>,
    },
    /// D25: also for a linear model trained on a classification dataset.
    Value {
        value: f64,
        /// Linear regression: "### Coeficientes" table.
        coefficients: Option<Vec<FeatureCoefficient>>,
    },
}

// --- C11–C12: export and code ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ExportFormat {
    Onnx,
    PythonPackage,
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
        }
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
