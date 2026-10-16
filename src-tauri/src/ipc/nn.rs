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

/// The history of a trained network (`nn_history`; 📈 and 📊 Historial reuse it in F5c).
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct HistoryView {
    pub metric: NnMetric,
    pub epochs: Vec<EpochLog>,
    pub best_epoch: Option<u16>,
    /// Epochs (1-based) after which ReduceLROnPlateau lowered the learning rate.
    pub lr_drops: Vec<u16>,
}

/// `nn_training_code`: the Keras script of D74 with the current controls (no model needed).
#[derive(Debug, Clone, PartialEq, Deserialize, TS)]
#[ts(export)]
pub struct NnCodeRequest {
    pub dataset: DatasetRef,
    pub arch: NnArchitecture,
    pub training: NnTraining,
}
