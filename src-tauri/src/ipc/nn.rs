//! Payloads of 🧠 Redes Neuronales (fase5-redes.md §5.3): the configuration of 🏗️
//! Arquitectura and ⚙️ Entrenamiento, the view of 🏗️ (`nn_architecture`), the live
//! training events (`nn-epoch`) and the history of a trained network.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use super::types::{DatasetRef, TaskType};

/// Activation of a `Dense` layer (Keras names on the wire).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum NnActivation {
    Relu,
    Tanh,
    Sigmoid,
    Softmax,
    Linear,
}

impl NnActivation {
    pub fn keras_name(self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Tanh => "tanh",
            Self::Sigmoid => "sigmoid",
            Self::Softmax => "softmax",
            Self::Linear => "linear",
        }
    }
}

/// Optimizer of `model.compile` (Keras names on the wire).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum NnOptimizer {
    Adam,
    Sgd,
    Rmsprop,
}

impl NnOptimizer {
    pub fn keras_name(self) -> &'static str {
        match self {
            Self::Adam => "adam",
            Self::Sgd => "sgd",
            Self::Rmsprop => "rmsprop",
        }
    }
}

/// The controls of 🏗️ Arquitectura (§2.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct NnArchitecture {
    /// Neurons of each hidden layer: 1..=5 layers of 1..=256 neurons.
    pub hidden: Vec<u16>,
    /// Hidden activation: relu, tanh or sigmoid.
    pub activation: NnActivation,
    pub output_activation: NnActivation,
    /// 0.0..=0.8, step 0.1.
    pub dropout: f32,
    /// 16, 32, 64, 128 or 256.
    pub batch_size: u16,
    pub optimizer: NnOptimizer,
}

/// The controls of ⚙️ Entrenamiento (§2.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct NnTraining {
    /// 0.001, 0.01, 0.1 or 0.3.
    pub learning_rate: f32,
    /// 10..=500.
    pub epochs: u16,
    /// "% Datos de Validación", 10..=40.
    pub validation_pct: u8,
    /// EarlyStopping patience (5..=50); `None` = "Activar Parada Temprana" off.
    pub patience: Option<u8>,
    /// ReduceLROnPlateau factor (0.1..=0.9); `None` = scheduler off.
    pub lr_factor: Option<f32>,
    /// D70: seed of the initial weights, the dropout masks and the batch order (1..=9999).
    pub seed: u32,
}

/// What `nn_architecture` describes: the selected dataset with the current controls.
#[derive(Debug, Clone, PartialEq, Deserialize, TS)]
#[ts(export)]
pub struct ArchitectureRequest {
    pub dataset: DatasetRef,
    pub arch: NnArchitecture,
    /// "% Datos de Validación" of ⚙️ (for the real distribution of D73).
    pub validation_pct: u8,
    /// "Épocas" of ⚙️: with it the view carries the cost estimate (K6).
    #[serde(default)]
    #[ts(optional)]
    pub epochs: Option<u16>,
}

/// K6: how long the training may take (`mltutor_core::nn_data::training_cost`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum CostLevel {
    Low,
    High,
    VeryHigh,
}

/// K6: the estimate under "🧠 Entrenar Red Neuronal".
#[derive(Debug, Clone, Copy, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct TrainingCost {
    pub level: CostLevel,
    /// Seconds on the reference machine (indicative).
    pub estimated_seconds: f64,
}

/// "⚖️ Complejidad": ratio parameters/rows < 0.1, < 1 or more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Complexity {
    Low,
    Medium,
    High,
}

/// Severity of the message under the output activation selector (§2.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum OutputLevel {
    Success,
    Warning,
    Error,
}

/// The message of the output activation and, for the combinations that cannot be
/// trained (D71), the reason in Spanish (the button of ⚙️ is disabled).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct OutputCheck {
    pub level: OutputLevel,
    /// Key of `[arquitectura]` in `content/redes/mensajes.toml` (`multiclase_softmax`…).
    pub key: String,
    /// D71: `no_entrenable_*` already formatted; `None` if the network can be trained.
    pub untrainable: Option<String>,
}

/// Caption of "Tamaño de Batch": batch ≥ rows // 4, ≥ 32, or smaller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum BatchHint {
    Large,
    Optimal,
    Small,
}

/// One row of "📊 Detalles por Capa" (D76: one row per layer; the sum is the total).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct LayerRow {
    /// "Entrada", "Oculta i" or "Salida".
    pub name: String,
    pub neurons: u32,
    /// Neurons of the previous layer (0 for the input).
    pub inputs: u32,
    /// inputs × neurons + neurons.
    #[ts(type = "number")]
    pub params: u64,
    /// `None` for the input layer.
    pub activation: Option<NnActivation>,
}

/// D73: the real sizes Keras uses (`n_fit`, the last `n_val` rows of the train set, test).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct SplitSizes {
    pub train: u32,
    pub val: u32,
    pub test: u32,
}

/// Numbers of `arquitectura-tamano.md` and `arquitectura-consejos.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
pub struct SizeTips {
    /// input // 2, input × 2, input × 4.
    pub half: u32,
    pub double: u32,
    pub quadruple: u32,
    /// rows // 10.
    pub max_params: u32,
    /// 32 (< 1000 rows), 64 (< 10 000) or 128.
    pub batch_recommended: u32,
}

/// 🏗️ Arquitectura of the dataset with the current controls (`nn_architecture`).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ArchitectureView {
    pub dataset: DatasetRef,
    /// Number of features.
    pub input: u32,
    /// 1 in binary classification and regression, the number of classes otherwise.
    pub output: u32,
    pub task: TaskType,
    /// Classes of the target (`len(y.unique())`), `None` in regression.
    pub n_classes: Option<u32>,
    pub n_rows: u32,
    /// "💡 Sugerencia para capa i" of the 5 possible hidden layers.
    pub suggested_neurons: Vec<u16>,
    /// Default value of the slider of each of the 5 layers: `min(suggestion, 64)`.
    pub default_neurons: Vec<u16>,
    pub layers: Vec<LayerRow>,
    #[ts(type = "number")]
    pub total_params: u64,
    pub complexity_ratio: f64,
    pub complexity: Complexity,
    /// Options of the output selector, in Python's order for the task.
    pub output_options: Vec<NnActivation>,
    pub recommended_output: NnActivation,
    pub output_check: OutputCheck,
    pub batch_hint: BatchHint,
    pub size_tips: SizeTips,
    /// Default of "Épocas": `min(100, suggested)`.
    pub default_epochs: u16,
    /// `min(200, max(50, rows // 10))`.
    pub suggested_epochs: u16,
    /// D73 with the `validation_pct` of the request.
    pub split: SplitSizes,
    /// K6, when the request carries the epochs.
    pub cost: Option<TrainingCost>,
}

/// Steps 1–4 of the training progress (D82).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum TrainPhase {
    Preparing,
    Building,
    Compiling,
    Training,
}

/// The logs of one epoch (`History.history` of Keras). `None` = NaN or infinite (a
/// diverging learning rate).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct EpochLog {
    /// 1-based.
    pub epoch: u16,
    pub loss: Option<f64>,
    /// accuracy (classification) or MAE (regression).
    pub metric: Option<f64>,
    pub val_loss: Option<f64>,
    pub val_metric: Option<f64>,
    /// Only with ReduceLROnPlateau (logged before reducing, as Keras).
    pub learning_rate: Option<f64>,
}

/// Event `nn-epoch`: the phase of the training and the epochs finished since the
/// previous event (at most 20 events per second; the last epoch is always sent).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct NnEpochEvent {
    pub task_id: u32,
    pub phase: TrainPhase,
    /// Epochs finished so far.
    pub epoch: u16,
    /// Maximum epochs of the training.
    pub epochs: u16,
    /// New epochs since the previous event, in order.
    pub logs: Vec<EpochLog>,
    /// Wall clock of the emission (ms since the Unix epoch): the smoke test measures the
    /// latency from the backend to the chart with it.
    pub sent_at_ms: f64,
}

/// Loss and metric that the app chose for the output activation (§2.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum NnMetric {
    Accuracy,
    Mae,
}

/// Facts of a trained network (part of `TrainedModelInfo`).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct NnSummary {
    /// `[input, hidden…, output]`.
    pub layers: Vec<u32>,
    /// `binary_crossentropy`, `categorical_crossentropy` or `mse`.
    pub loss: String,
    pub metric: NnMetric,
    pub epochs_trained: u16,
    /// 1-based epoch whose weights EarlyStopping restored.
    pub best_epoch: Option<u16>,
    pub stopped_early: bool,
    pub n_fit: u32,
    pub n_val: u32,
    pub n_test: u32,
    /// Milliseconds since the Unix epoch (the UI shows the local date, D75).
    #[ts(type = "number")]
    pub trained_at_ms: u64,
    /// Training time in milliseconds.
    pub duration_ms: u32,
}

/// The history of a trained network (`nn_history`; ⚙️, 📈 and 📊 Historial).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct HistoryView {
    pub metric: NnMetric,
    pub epochs: Vec<EpochLog>,
    pub best_epoch: Option<u16>,
    /// Epochs (1-based) after which ReduceLROnPlateau lowered the learning rate.
    pub lr_drops: Vec<u16>,
    /// K1: first epoch (1-based) whose loss is NaN or infinite.
    pub diverged_epoch: Option<u16>,
    /// K1: the weights are finite (`NeuralNetwork::is_valid`). After a divergence they
    /// can still be: EarlyStopping restored the weights of `best_epoch`. Only an invalid
    /// network blocks the views that need its weights.
    pub network_valid: bool,
    /// The 4 cards of 📊 Historial (`st.metric` row); `None` if the history is empty or
    /// the training diverged.
    pub cards: Option<NnViewNode>,
}

/// Severity of a notice of a view tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum NnSeverity {
    Success,
    Info,
    Warning,
    Error,
}

/// A view built in the core with its texts already resolved (C23, the history cards):
/// the shape `tools/golden_render.py` records from the Python lessons. The UI renders it
/// as it comes.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum NnViewNode {
    Page {
        children: Vec<NnViewNode>,
    },
    /// Markdown (`html` = `unsafe_allow_html`; the UI never renders raw HTML).
    Text {
        text: String,
        html: bool,
    },
    Heading {
        text: String,
        level: u8,
    },
    Notice {
        text: String,
        severity: NnSeverity,
    },
    Metric {
        label: String,
        value: String,
        delta: Option<String>,
        /// `"normal"` / `"inverse"` when Python passes `delta_color`.
        delta_color: Option<String>,
        help: Option<String>,
    },
    /// `st.columns(weights)`; the children are `column`s.
    Row {
        weights: Vec<u32>,
        children: Vec<NnViewNode>,
    },
    Column {
        children: Vec<NnViewNode>,
    },
    Expander {
        label: String,
        expanded: bool,
        children: Vec<NnViewNode>,
    },
}

/// K1: the training diverged (NaN or infinite loss). If the network is not valid, the
/// views that need the weights are blocked and show this instead; if it is (the weights
/// of `best_epoch` were restored), they only warn.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct Divergence {
    /// 1-based.
    pub epoch: u16,
    /// Finite weights (`NeuralNetwork::is_valid`): `[evaluacion] aviso_divergencia_recuperada`.
    pub network_valid: bool,
    /// 1-based epoch whose weights EarlyStopping restored.
    pub best_epoch: Option<u16>,
}

/// 📈 Evaluación of a network (`nn_evaluation`): "📊 Resultados del Entrenamiento" (loss
/// and metric of `model.evaluate` on the test set) plus C1/C2.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct NnEvaluationView {
    pub metric: NnMetric,
    /// `None` = NaN or infinite.
    pub test_loss: Option<f64>,
    pub test_metric: Option<f64>,
    /// C1/C2 on the test set; `None` when the training diverged.
    pub evaluation: Option<super::model::Evaluation>,
    /// D84: `[evaluacion] aviso_r2_negativo` already formatted (regression, R² < 0).
    pub r2_negative: Option<String>,
    /// `[evaluacion] aviso_casi_divergencia`: the loss grew by orders of magnitude
    /// without reaching NaN/∞ (`History::near_divergence`).
    pub near_divergence: Option<String>,
    pub diverged: Option<Divergence>,
    /// D87: `false` with a tanh output, whose "probabilities" are neuron outputs in
    /// [−1, 1] (the class is still `> 0.5` or the argmax).
    pub output_is_probability: bool,
    pub output_activation: NnActivation,
}

/// C23: "🔍 Diagnóstico del Entrenamiento" (`nn_diagnostics`).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct DiagnosticsView {
    /// The children of the page, in order (empty if the training diverged).
    pub blocks: Vec<NnViewNode>,
    pub diverged: Option<Divergence>,
}

/// `nn_weights`: the bins of the two histograms of each layer (⚙️ Opciones).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, TS)]
#[ts(export)]
pub struct WeightsRequest {
    pub model_id: u32,
    /// "Bins pesos", 10..=120.
    pub bins_weights: u16,
    /// "Bins sesgos", 5..=60.
    pub bins_biases: u16,
}

/// One dense layer of 🧠 Análisis de Pesos (C25).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct LayerWeightsView {
    /// D77: "Oculta 1" … "Salida".
    pub name: String,
    /// Keras name (`dense_1`, `output_layer`).
    pub keras_name: String,
    pub inputs: u32,
    pub units: u32,
    pub mean_w: f64,
    pub std_w: f64,
    /// `% |w|<1e-3` rounded to 1 decimal (the table).
    pub near_zero_pct: f64,
    /// Unrounded (the caption).
    pub near_zero_pct_raw: f64,
    pub mean_b: f64,
    pub std_b: f64,
    pub n_params: u32,
    pub min_w: f64,
    pub max_w: f64,
    pub min_b: f64,
    pub max_b: f64,
    /// K2: the histograms are binned in Rust (the raw weights never cross the IPC).
    pub weights_hist: super::model::Histogram,
    pub biases_hist: super::model::Histogram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum WeightsVerdict {
    TooSmall,
    TooLarge,
    Healthy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum WeightsInsight {
    ManyNearZero,
    LowVariance,
    HighVariance,
    Balanced,
}

/// 🧠 Análisis de Pesos (`nn_weights`).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct WeightsView {
    pub layers: Vec<LayerWeightsView>,
    pub global_mean: f64,
    pub global_std: f64,
    pub global_near_zero_pct: f64,
    pub verdict: WeightsVerdict,
    pub insights: Vec<WeightsInsight>,
}

/// `nn_activations`: ⚙️ Parámetros de Análisis.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, TS)]
#[ts(export)]
pub struct ActivationsRequest {
    pub model_id: u32,
    pub task_id: u32,
    /// "Muestras", 10..=200: the first rows of the test set.
    pub sample_size: u16,
    /// "Umbral muerta (<=)", 0..=1.
    pub dead_threshold: f64,
    /// "Std constante <", 1e-6..=1.
    pub constant_std: f64,
}

/// One hidden layer of 📉 Activaciones (C26).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct LayerActivationsView {
    /// D77: "Oculta i".
    pub name: String,
    /// `dense_i`.
    pub keras_name: String,
    pub units: u32,
    pub dead: Vec<u32>,
    pub constant: Vec<u32>,
    pub saturation_pct: f64,
    pub mean_act: f64,
    pub std_act: f64,
    /// Per neuron ("Mostrar medias por neurona").
    pub neuron_means: Vec<f64>,
    /// 50 bins of every activation of the layer ("Mostrar histograma").
    pub hist: super::model::Histogram,
}

/// Tooltips of the neurons of one dense layer of the map (structure of arrays: with
/// 5 × 256 neurons the answer stays small). f32 is the precision of the network.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct MapLayer {
    pub bias: Vec<f32>,
    pub mean_w: Vec<f32>,
    pub std_w: Vec<f32>,
    pub max_abs_w: Vec<f32>,
}

/// "🗺️ Mapa de Neuronas Muertas (💀) y Constantes (😴)"; the state of each hidden
/// neuron comes from `LayerActivationsView.dead/constant`.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct NetworkMapView {
    /// `[input, hidden…, output]`.
    pub layers: Vec<u32>,
    /// One per dense layer (hidden… and output).
    pub neurons: Vec<MapLayer>,
}

/// 📉 Análisis de Activaciones (`nn_activations`).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ActivationsView {
    /// K3: rows really used (`min(sample_size, n_test)`).
    pub rows_used: u32,
    pub layers: Vec<LayerActivationsView>,
    pub dead_pct: f64,
    pub map: NetworkMapView,
}

/// `nn_training_code`: the Keras script of D74 with the current controls (no model needed).
#[derive(Debug, Clone, PartialEq, Deserialize, TS)]
#[ts(export)]
pub struct NnCodeRequest {
    pub dataset: DatasetRef,
    pub arch: NnArchitecture,
    pub training: NnTraining,
}

// --- 🔮 Predicciones: exploración y lotes (C29, D81) ---

/// `nn_explore`: the base sample (a row of the dataset) and the feature to vary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, TS)]
#[ts(export)]
pub struct ExploreRequest {
    pub model_id: u32,
    /// "Índice de muestra": 0..n_rows of the dataset the network was trained on.
    pub row: u32,
    /// "Característica a explorar": column index.
    pub feature: u32,
}

/// Which chart and analysis of 🎲 Exploración (Python's three branches).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ExploreKind {
    Binary,
    Multiclass,
    Regression,
}

/// "Característica muy / moderadamente / poco influyente" (thresholds of Python:
/// range 0.3 / 0.1 of the probability; relative impact 20 % / 5 % in regression).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Influence {
    High,
    Moderate,
    Low,
}

/// 🎲 Exploración interactiva (`nn_explore`): the response curve of the network to one
/// feature of one sample, the rest fixed (an ICE curve).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct ExploreView {
    pub kind: ExploreKind,
    pub row: u32,
    pub feature: u32,
    /// Rows of the dataset (options of "Índice de muestra").
    pub n_rows: u32,
    /// "2. 📋 Valores base de la muestra", original units, column order.
    pub base: Vec<f64>,
    pub base_prediction: super::model::PredictionResult,
    /// Value of the explored feature in the base sample ("Valor Base").
    pub base_value: f64,
    /// 50 values from the minimum to the maximum of the feature in the dataset.
    pub xs: Vec<f64>,
    /// The analysed series: P(clase positiva) (binary), P(clase de la muestra base)
    /// (multiclass, NEW) or the prediction (regression).
    pub ys: Vec<f64>,
    /// Multiclass: the predicted class code at each x ("Clase Predicha").
    pub classes: Option<Vec<u32>>,
    /// Multiclass: the largest output at each x ("Confianza").
    pub confidence: Option<Vec<f64>>,
    /// Multiclass: the class of the base sample (the one `ys` follows).
    pub base_class: Option<u32>,
    /// max − min of `ys`.
    pub range: f64,
    /// Population standard deviation of `ys` (`np.std`).
    pub volatility: f64,
    /// Regression: range / |mean| × 100 (0 if the mean is 0).
    pub relative_impact: Option<f64>,
    pub influence: Influence,
    /// D87: `false` with a tanh output.
    pub output_is_probability: bool,
}

/// `nn_batch_template`: "📋 Formato esperado del archivo CSV" (the first rows of the
/// features) and the range of every feature in the training rows (the "fuera de rango"
/// warnings of the form and of the batch use it).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct BatchTemplate {
    /// The column names the CSV must have (`X.columns`, model order).
    pub columns: Vec<String>,
    pub sample: super::types::TablePreview,
    /// Minimum and maximum of every feature in the training rows (original units).
    pub train_min: Vec<f64>,
    pub train_max: Vec<f64>,
}

/// `nn_predict_batch`: the CSV is chosen with the native dialog in Rust (or comes from
/// the automation modes); the path never goes through JS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, TS)]
#[ts(export)]
pub struct BatchPredictRequest {
    pub model_id: u32,
    pub task_id: u32,
}

/// 📊 Predicción por lotes (`nn_predict_batch`). The checks run in the order of
/// `[lotes]` (sin_filas → duplicadas → faltan → no_numericas → vacias); with any error
/// nothing is predicted.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct BatchPredictView {
    /// Name of the file (without directories).
    pub file_name: String,
    /// Rows and columns of the file ("✅ Archivo cargado: …").
    pub rows: u32,
    pub cols: u32,
    /// Features of the model that the file lacks ("❌ Faltan características").
    pub missing: Vec<String>,
    /// Columns of the file that are ignored ("⚠️ Características adicionales").
    pub extra: Vec<String>,
    /// The ❌ messages of the checks, already formatted (Markdown).
    pub errors: Vec<String>,
    /// Every feature is there but in another order (`[lotes] reordenadas`).
    pub reordered: bool,
    pub kind: ExploreKind,
    pub output_is_probability: bool,
    /// `[lotes] fuera_rango` already formatted, if some row leaves the training range.
    pub out_of_range: Option<String>,
    /// The first rows of the results: the features in model order + "Predicción" +
    /// "Probabilidad" and "Confianza" (binary) or "Confianza" (multiclass) + "Fuera de
    /// rango". With a tanh output (not a probability) there is no "Confianza" and the
    /// binary column is "Salida de la neurona". `None` if a check failed.
    pub results: Option<super::types::TablePreview>,
    /// Rows with a valid prediction ("✅ Predicciones generadas para {n} muestras"); the
    /// CSV of "📥 Descargar Resultados" has every row of the file (`nn_save_batch`).
    pub predicted: u32,
    /// Rows whose network output is NaN or infinite: no class, `[lotes] fila_no_finita`
    /// in their "Predicción" cell.
    pub non_finite: u32,
}
