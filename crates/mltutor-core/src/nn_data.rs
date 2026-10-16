//! 🧠 Redes Neuronales: configuration types and data preparation (fase5-redes.md C22).
//!
//! Everything the lesson does **before** the network sees the data, exactly as the
//! Python app (`algorithms/model_training.py::train_neural_network`) and
//! `tools/nn_reference.py`, the executable specification:
//!
//! 1. [`Scaler`]: `StandardScaler().fit_transform(X)` with **all** the rows, before
//!    splitting (D72: a leak kept for parity; the generated Keras script explains it).
//! 2. [`stratified_split`]: `train_test_split(test_size=0.2, random_state=42,
//!    stratify=y)` in classification (unlike D7), [`crate::split::train_test_split`]
//!    in regression.
//! 3. [`resolve_training`]: the loss, metric and target encoding that Keras ends up
//!    using for each output activation, or [`NnError::UntrainableOutput`] for the four
//!    combinations Python lets the student pick but cannot train (D71).
//! 4. [`keras_validation_split`]: `fit(validation_split=v)` keeps the first
//!    `floor(n·(1 − v))` rows and validates on the **last** ones.
//!
//! It also has the numbers of 🏗️ Arquitectura (parameters per layer, complexity,
//! suggested neurons, default epochs, real split sizes) and the history type shared
//! with `mltutor-nn` and [`crate::nn_diagnostics`].

use serde::{Deserialize, Serialize};

use crate::dataset::TaskType;
use crate::model::ModelError;
use crate::sampling::RandomState;
use crate::split::train_test_split;

/// Activation of a dense layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Relu,
    Tanh,
    Sigmoid,
    Softmax,
    Linear,
}

impl Activation {
    /// Keras name (`"relu"`…).
    pub fn as_str(self) -> &'static str {
        match self {
            Activation::Relu => "relu",
            Activation::Tanh => "tanh",
            Activation::Sigmoid => "sigmoid",
            Activation::Softmax => "softmax",
            Activation::Linear => "linear",
        }
    }

    /// Parses a Keras name.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "relu" => Activation::Relu,
            "tanh" => Activation::Tanh,
            "sigmoid" => Activation::Sigmoid,
            "softmax" => Activation::Softmax,
            "linear" => Activation::Linear,
            _ => return None,
        })
    }

    /// Options of the hidden-layer select, in the app's order.
    pub const HIDDEN: [Activation; 3] = [Activation::Relu, Activation::Tanh, Activation::Sigmoid];

    /// Options of the output select, in the app's order for the task.
    pub fn output_options(task: TaskType) -> [Activation; 4] {
        match task {
            TaskType::Classification => [
                Activation::Sigmoid,
                Activation::Softmax,
                Activation::Linear,
                Activation::Tanh,
            ],
            TaskType::Regression => [
                Activation::Linear,
                Activation::Sigmoid,
                Activation::Tanh,
                Activation::Softmax,
            ],
        }
    }

    /// The output activation the app selects by default: sigmoid (binary), softmax
    /// (multiclass), linear (regression). D85: the UI goes back to it when the task
    /// or the number of classes changes.
    pub fn recommended_output(task: TaskType, n_classes: usize) -> Activation {
        match task {
            TaskType::Regression => Activation::Linear,
            TaskType::Classification if n_classes <= 2 => Activation::Sigmoid,
            TaskType::Classification => Activation::Softmax,
        }
    }
}

/// Optimizer, with Keras 3.12 defaults (fase5-redes.md C21.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    Adam,
    Sgd,
    Rmsprop,
}

impl OptimizerKind {
    /// Keras name (`"adam"`, `"sgd"`, `"rmsprop"`).
    pub fn as_str(self) -> &'static str {
        match self {
            OptimizerKind::Adam => "adam",
            OptimizerKind::Sgd => "sgd",
            OptimizerKind::Rmsprop => "rmsprop",
        }
    }

    /// Parses a Keras name.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "adam" => OptimizerKind::Adam,
            "sgd" => OptimizerKind::Sgd,
            "rmsprop" => OptimizerKind::Rmsprop,
            _ => return None,
        })
    }
}

/// Loss the model is compiled with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossKind {
    /// With a sigmoid output: from the logits (`sigmoid_cross_entropy_with_logits`).
    BinaryCrossentropy,
    /// With a softmax output: from the logits (`log-sum-exp`), one-hot target.
    CategoricalCrossentropy,
    /// Mean of `(p − y)²` per row, through the output activation.
    Mse,
}

impl LossKind {
    /// Keras name (`"binary_crossentropy"`…).
    pub fn as_str(self) -> &'static str {
        match self {
            LossKind::BinaryCrossentropy => "binary_crossentropy",
            LossKind::CategoricalCrossentropy => "categorical_crossentropy",
            LossKind::Mse => "mse",
        }
    }
}

/// The metric Keras resolves for `metrics=['accuracy']` / `['mae']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    /// `p > 0.5` against the 0/1 label (one output neuron, also with tanh).
    BinaryAccuracy,
    /// `argmax(p) == argmax(y_one_hot)`.
    CategoricalAccuracy,
    /// Mean absolute error.
    Mae,
}

impl MetricKind {
    /// Key of the history (`"accuracy"` or `"mae"`; the validation one adds `val_`).
    pub fn key(self) -> &'static str {
        match self {
            MetricKind::Mae => "mae",
            _ => "accuracy",
        }
    }

    /// Keras name of the resolved metric (`"binary_accuracy"`…).
    pub fn as_str(self) -> &'static str {
        match self {
            MetricKind::BinaryAccuracy => "binary_accuracy",
            MetricKind::CategoricalAccuracy => "categorical_accuracy",
            MetricKind::Mae => "mae",
        }
    }
}

/// Default seed of the network (D70): 42, as `random_state=42` elsewhere.
pub const DEFAULT_SEED: u32 = 42;

fn default_seed() -> u32 {
    DEFAULT_SEED
}

/// 🏗️ Arquitectura (contract §5.3). The input and output sizes come from the data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NnArchitecture {
    /// Neurons of each hidden layer (1..=5 layers of 1..=256).
    pub hidden: Vec<u16>,
    /// Hidden activation (relu, tanh or sigmoid).
    pub activation: Activation,
    /// Output activation.
    pub output_activation: Activation,
    /// Dropout after every hidden layer (0..=0.8).
    pub dropout: f32,
    /// 16, 32, 64, 128 or 256.
    pub batch_size: u16,
    pub optimizer: OptimizerKind,
}

/// ⚙️ Entrenamiento (contract §5.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NnTraining {
    /// 0.001, 0.01, 0.1 or 0.3 (stored in f32 like Keras).
    pub learning_rate: f32,
    /// Maximum epochs (10..=500).
    pub epochs: u16,
    /// `% Datos de Validación` (10..=40).
    pub validation_pct: u8,
    /// EarlyStopping patience; `None` = disabled.
    pub patience: Option<u8>,
    /// ReduceLROnPlateau factor; `None` = disabled.
    pub lr_factor: Option<f32>,
    /// D70: seed of the initial weights, the dropout masks and the batch order.
    #[serde(default = "default_seed")]
    pub seed: u32,
}

impl NnTraining {
    /// `validation_pct / 100`.
    pub fn validation_split(&self) -> f64 {
        f64::from(self.validation_pct) / 100.0
    }

    /// ReduceLROnPlateau patience of the app: `max(1, (patience or 10) // 2)`.
    pub fn reduce_lr_patience(&self) -> usize {
        (usize::from(self.patience.unwrap_or(10)) / 2).max(1)
    }
}

/// Errors of the neural-network lesson (data preparation, configuration and
/// training). Messages in Spanish; `src-tauri` maps them to `CommandError`.
#[derive(Debug, Clone, PartialEq, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NnError {
    /// D71: an output activation Keras cannot train for this target. `key` is the
    /// `[arquitectura]` message (`no_entrenable_*`) and `reason` what TensorFlow
    /// hit in Python (`rank`, `sparse_rank`, `label_range`).
    #[error("La activación de salida {output} no se puede entrenar con este objetivo ({reason})")]
    UntrainableOutput {
        output: String,
        n_classes: usize,
        key: String,
        reason: String,
    },
    /// The stratified split needs at least 2 rows of every class.
    #[error("La clase con menos muestras ({class_name}) solo tiene {count}: la división estratificada necesita al menos 2 por clase.")]
    TooFewPerClass { class_name: String, count: usize },
    /// The training or test set would have fewer rows than classes (sklearn refuses).
    #[error("Con {n_rows} filas no se pueden repartir las {n_classes} clases entre entrenamiento y prueba (prueba: {n_test}).")]
    SplitTooSmall {
        n_rows: usize,
        n_classes: usize,
        n_test: usize,
    },
    /// The validation split leaves no training or no validation rows.
    #[error("Con {n_train} filas de entrenamiento y {pct} % de validación no quedan datos para entrenar o validar.")]
    ValidationTooSmall { n_train: usize, pct: u8 },
    /// Invalid configuration (sizes, rates…).
    #[error("Configuración no válida: {reason}")]
    InvalidConfig { reason: String },
    /// The loss became NaN or infinite.
    #[error("La pérdida se ha vuelto NaN o infinita en la época {epoch}: prueba un learning rate menor.")]
    NonFinite { epoch: usize },
    /// The cancellation flag was set.
    #[error("Operación cancelada")]
    Cancelled,
    /// A shared model error (shape, NaN in the inputs…).
    #[error(transparent)]
    Model(#[from] ModelError),
}

// ---------------------------------------------------------------------------
// C22: split, scaler, validation
// ---------------------------------------------------------------------------

/// `sklearn.utils._approximate_mode` with the legacy `RandomState.choice`
/// (`permutation(n)[:size]`, which consumes a whole permutation even when it keeps
/// every element). Float expressions in NumPy's order: `counts / total * n_draws`.
fn approximate_mode(class_counts: &[usize], n_draws: usize, rng: &mut RandomState) -> Vec<usize> {
    let total: usize = class_counts.iter().sum();
    let continuous: Vec<f64> = class_counts
        .iter()
        .map(|&c| c as f64 / total as f64 * n_draws as f64)
        .collect();
    let mut floored: Vec<f64> = continuous.iter().map(|c| c.floor()).collect();
    let floored_sum: f64 = floored.iter().sum();
    let mut need_to_add = (n_draws as f64 - floored_sum) as i64;
    if need_to_add > 0 {
        let remainder: Vec<f64> = continuous
            .iter()
            .zip(&floored)
            .map(|(c, f)| c - f)
            .collect();
        // np.sort(np.unique(remainder))[::-1]
        let mut values = remainder.clone();
        values.sort_by(|a, b| b.total_cmp(a));
        values.dedup_by(|a, b| a == b);
        for value in values {
            // Exact comparison, like `np.where(remainder == value)`.
            #[allow(clippy::float_cmp)]
            let inds: Vec<usize> = (0..remainder.len())
                .filter(|&i| remainder[i] == value)
                .collect();
            let add_now = inds.len().min(usize::try_from(need_to_add).unwrap_or(0));
            let perm = rng.permutation(inds.len());
            for &p in &perm[..add_now] {
                floored[inds[p]] += 1.0;
            }
            need_to_add -= add_now as i64;
            if need_to_add == 0 {
                break;
            }
        }
    }
    floored.iter().map(|&f| f as usize).collect()
}

/// Positions of `train_test_split(range(n), test_size, random_state=seed,
/// stratify=codes)`, in scikit-learn's order (`StratifiedShuffleSplit`).
///
/// `codes` are the class of each row (any integers; classes are their sorted
/// distinct values, like `np.unique`). `class_names[c]` names code `c` in the
/// [`NnError::TooFewPerClass`] message (missing names fall back to the code).
pub fn stratified_split(
    codes: &[usize],
    class_names: &[String],
    test_size: f64,
    seed: u32,
) -> Result<crate::split::TrainTestSplit, NnError> {
    let n = codes.len();
    if !(test_size > 0.0 && test_size < 1.0) {
        return Err(NnError::InvalidConfig {
            reason: format!("test_size debe estar entre 0 y 1 (es {test_size})"),
        });
    }
    let n_test = (test_size * n as f64).ceil() as usize;
    let n_train = n.saturating_sub(n_test);
    // np.unique(y, return_inverse=True) + bincount.
    let mut classes: Vec<usize> = codes.to_vec();
    classes.sort_unstable();
    classes.dedup();
    let inverse: Vec<usize> = codes
        .iter()
        .map(|c| classes.partition_point(|x| x < c))
        .collect();
    let mut counts = vec![0usize; classes.len()];
    for &k in &inverse {
        counts[k] += 1;
    }
    if let Some((k, &count)) = counts.iter().enumerate().min_by_key(|(_, &c)| c) {
        if count < 2 {
            let code = classes[k];
            let class_name = class_names
                .get(code)
                .cloned()
                .unwrap_or_else(|| code.to_string());
            return Err(NnError::TooFewPerClass { class_name, count });
        }
    }
    if n_train < classes.len() || n_test < classes.len() {
        return Err(NnError::SplitTooSmall {
            n_rows: n,
            n_classes: classes.len(),
            n_test,
        });
    }
    // np.split(np.argsort(y_indices, kind="mergesort"), cumsum(counts)[:-1]): the rows
    // of each class in increasing order.
    let mut class_indices: Vec<Vec<usize>> =
        counts.iter().map(|&c| Vec::with_capacity(c)).collect();
    for (row, &k) in inverse.iter().enumerate() {
        class_indices[k].push(row);
    }
    let mut rng = RandomState::new(seed);
    let n_i = approximate_mode(&counts, n_train, &mut rng);
    let remaining: Vec<usize> = counts.iter().zip(&n_i).map(|(c, t)| c - t).collect();
    let t_i = approximate_mode(&remaining, n_test, &mut rng);
    let mut train = Vec::with_capacity(n_train);
    let mut test = Vec::with_capacity(n_test);
    for (k, rows) in class_indices.iter().enumerate() {
        let perm = rng.permutation(counts[k]);
        let idx: Vec<usize> = perm.iter().map(|&p| rows[p]).collect();
        train.extend_from_slice(&idx[..n_i[k]]);
        test.extend_from_slice(&idx[n_i[k]..n_i[k] + t_i[k]]);
    }
    let perm = rng.permutation(train.len());
    let train = perm.iter().map(|&p| train[p]).collect();
    let perm = rng.permutation(test.len());
    let test = perm.iter().map(|&p| test[p]).collect();
    Ok(crate::split::TrainTestSplit { train, test })
}

/// The split of the lesson: stratified on the class codes in classification,
/// plain (D7) in regression; `test_size = 0.2`, `random_state = 42`.
pub fn lesson_split(
    task: TaskType,
    codes: Option<&[usize]>,
    class_names: &[String],
    n_rows: usize,
) -> Result<crate::split::TrainTestSplit, NnError> {
    match (task, codes) {
        (TaskType::Classification, Some(codes)) => {
            stratified_split(codes, class_names, LESSON_TEST_SIZE, DEFAULT_SEED)
        }
        _ => train_test_split(n_rows, LESSON_TEST_SIZE, DEFAULT_SEED).map_err(|e| {
            NnError::InvalidConfig {
                reason: e.to_string(),
            }
        }),
    }
}

/// The lesson's fixed test size (no control in Python).
pub const LESSON_TEST_SIZE: f64 = 0.2;

/// `StandardScaler` fitted on all the rows (D72): population standard deviation and
/// scale 1 for constant columns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scaler {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    /// Rows it was fitted on (`n_samples_seen_`).
    pub n_samples: usize,
}

impl Scaler {
    /// Fits on the rows of `x` (`n × p`). NaN propagates (the caller rejects it).
    pub fn fit(x: ndarray::ArrayView2<'_, f64>) -> Scaler {
        let n = x.nrows();
        let mut mean = Vec::with_capacity(x.ncols());
        let mut scale = Vec::with_capacity(x.ncols());
        for col in x.columns() {
            let values: Vec<f64> = col.iter().copied().collect();
            let m = crate::stats::sum::pairwise_sum(&values) / n as f64;
            let var = crate::stats::sum::pairwise_sum_by(n, |i| (values[i] - m) * (values[i] - m))
                / n as f64;
            let s = var.sqrt();
            mean.push(m);
            scale.push(if s == 0.0 { 1.0 } else { s });
        }
        Scaler {
            mean,
            scale,
            n_samples: n,
        }
    }

    /// `(x − mean) / scale` in f64.
    pub fn transform(&self, x: ndarray::ArrayView2<'_, f64>) -> ndarray::Array2<f64> {
        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            let (m, s) = (self.mean[j], self.scale[j]);
            col.mapv_inplace(|v| (v - m) / s);
        }
        out
    }

    /// [`Scaler::transform`] and conversion to f32 (what Keras receives), row-major.
    pub fn transform_f32(&self, x: ndarray::ArrayView2<'_, f64>) -> Vec<f32> {
        let p = x.ncols();
        let mut out = Vec::with_capacity(x.len());
        for row in x.rows() {
            for (j, &v) in row.iter().enumerate().take(p) {
                out.push(((v - self.mean[j]) / self.scale[j]) as f32);
            }
        }
        out
    }

    /// Back to original units.
    pub fn inverse(&self, j: usize, scaled: f64) -> f64 {
        scaled * self.scale[j] + self.mean[j]
    }
}

/// `model.fit(validation_split=v)`: rows kept for training (`floor(n·(1 − v))`); the
/// last `n − that` rows validate.
pub fn keras_validation_split(n_train: usize, validation_split: f64) -> usize {
    (n_train as f64 * (1.0 - validation_split)).floor() as usize
}

/// Real sizes of the three sets (D73): `test = ceil(0.2·n)`, `fit = floor(train·(1 −
/// v))`, `val = train − fit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitSizes {
    pub train: u32,
    pub val: u32,
    pub test: u32,
}

/// [`SplitSizes`] of `n_rows` rows with `validation_pct` %.
pub fn split_sizes(n_rows: usize, validation_pct: u8) -> SplitSizes {
    let n_test = (LESSON_TEST_SIZE * n_rows as f64).ceil() as usize;
    let n_train = n_rows.saturating_sub(n_test);
    let fit = keras_validation_split(n_train, f64::from(validation_pct) / 100.0);
    let as_u32 = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
    SplitSizes {
        train: as_u32(fit),
        val: as_u32(n_train - fit),
        test: as_u32(n_test),
    }
}

/// What `train_neural_network` compiles for an output activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedTraining {
    /// One-hot target (`to_categorical`) instead of a 0/1 column.
    pub one_hot: bool,
    pub loss: LossKind,
    pub metric: MetricKind,
}

/// The loss, metric and target encoding of the app for `output` with `output_size`
/// neurons (`nn_reference.resolve_training`), or [`NnError::UntrainableOutput`] for
/// the four combinations that fail in Python (D71): multiclass sigmoid (`rank`),
/// multiclass linear (`sparse_rank`), binary softmax (`sparse_rank`) and binary
/// linear (`label_range`).
pub fn resolve_training(
    task: TaskType,
    output: Activation,
    output_size: usize,
) -> Result<ResolvedTraining, NnError> {
    if task == TaskType::Regression {
        return Ok(ResolvedTraining {
            one_hot: false,
            loss: LossKind::Mse,
            metric: MetricKind::Mae,
        });
    }
    let one_hot =
        output == Activation::Softmax || (output_size > 1 && output != Activation::Sigmoid);
    let n_classes = if output_size == 1 { 2 } else { output_size };
    let untrainable = |key: &str, reason: &str| NnError::UntrainableOutput {
        output: output.as_str().to_owned(),
        n_classes,
        key: key.to_owned(),
        reason: reason.to_owned(),
    };
    let metric = if output_size == 1 {
        MetricKind::BinaryAccuracy
    } else {
        MetricKind::CategoricalAccuracy
    };
    match output {
        Activation::Sigmoid if output_size > 1 => {
            Err(untrainable("no_entrenable_multiclase_sigmoid", "rank"))
        }
        Activation::Sigmoid => Ok(ResolvedTraining {
            one_hot,
            loss: LossKind::BinaryCrossentropy,
            metric,
        }),
        Activation::Softmax if output_size == 1 => {
            Err(untrainable("no_entrenable_binaria_softmax", "sparse_rank"))
        }
        Activation::Softmax => Ok(ResolvedTraining {
            one_hot,
            loss: LossKind::CategoricalCrossentropy,
            metric,
        }),
        Activation::Linear if output_size == 1 => {
            Err(untrainable("no_entrenable_linear", "label_range"))
        }
        Activation::Linear => Err(untrainable("no_entrenable_linear", "sparse_rank")),
        // tanh (and relu, which the select never offers): MSE on the encoded target.
        Activation::Tanh | Activation::Relu => Ok(ResolvedTraining {
            one_hot,
            loss: LossKind::Mse,
            metric,
        }),
    }
}

/// Output neurons of the app: 1 in binary classification and regression, one per
/// class otherwise.
pub fn output_size(task: TaskType, n_classes: usize) -> usize {
    match task {
        TaskType::Classification if n_classes > 2 => n_classes,
        _ => 1,
    }
}

// ---------------------------------------------------------------------------
// 🏗️ Arquitectura: numbers of the view
// ---------------------------------------------------------------------------

/// `calculate_network_parameters` (= Keras `count_params()`): Σ nᵢ·nᵢ₊₁ + nᵢ₊₁.
pub fn total_parameters(layers: &[usize]) -> u64 {
    layers
        .windows(2)
        .map(|w| (w[0] as u64) * (w[1] as u64) + w[1] as u64)
        .sum()
}

/// One row of "📊 Detalles por Capa", corrected (D76): one row per layer, the sum
/// of `params` is the total.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerRow {
    /// "Entrada", "Oculta {i}" or "Salida".
    pub kind: LayerKind,
    pub neurons: u32,
    pub params: u64,
    /// `None` for the input layer.
    pub activation: Option<Activation>,
}

/// Which layer a [`LayerRow`] is ("Entrada", "Oculta {i}", "Salida").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "index", rename_all = "snake_case")]
pub enum LayerKind {
    Input,
    /// 1-based hidden layer.
    Hidden(u32),
    Output,
}

/// The corrected per-layer table (D76) of `layers = [input, hidden…, output]`.
pub fn layer_rows(layers: &[usize], hidden: Activation, output: Activation) -> Vec<LayerRow> {
    let last = layers.len().saturating_sub(1);
    layers
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let neurons = u32::try_from(n).unwrap_or(u32::MAX);
            if i == 0 {
                LayerRow {
                    kind: LayerKind::Input,
                    neurons,
                    params: 0,
                    activation: None,
                }
            } else {
                let params = (layers[i - 1] as u64) * (n as u64) + n as u64;
                let (kind, activation) = if i == last {
                    (LayerKind::Output, output)
                } else {
                    (
                        LayerKind::Hidden(u32::try_from(i).unwrap_or(u32::MAX)),
                        hidden,
                    )
                };
                LayerRow {
                    kind,
                    neurons,
                    params,
                    activation: Some(activation),
                }
            }
        })
        .collect()
}

/// "⚖️ Complejidad" (ratio parameters / rows): Baja < 0.1 ≤ Media < 1 ≤ Alta.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Complexity {
    Low,
    Medium,
    High,
}

/// `(ratio, level)` with the app's thresholds (`ratio = 0` without rows).
pub fn complexity(total_params: u64, n_rows: usize) -> (f64, Complexity) {
    let ratio = if n_rows > 0 {
        total_params as f64 / n_rows as f64
    } else {
        0.0
    };
    let level = if ratio < 0.1 {
        Complexity::Low
    } else if ratio < 1.0 {
        Complexity::Medium
    } else {
        Complexity::High
    };
    (ratio, level)
}

/// "💡 Sugerencia para capa i": `max(10, input // i)` and, in multiclass, at least
/// `2·classes`. The slider starts at `min(suggestion, 64)` ([`default_neurons`]).
pub fn suggested_neurons(
    input: usize,
    n_hidden: usize,
    multiclass_classes: Option<usize>,
) -> Vec<u16> {
    (1..=n_hidden)
        .map(|i| {
            let mut s = (input / i).max(10);
            if let Some(k) = multiclass_classes {
                s = s.max(2 * k);
            }
            u16::try_from(s).unwrap_or(u16::MAX)
        })
        .collect()
}

/// Initial value of each neurons slider: `min(suggestion, 64)`.
pub fn default_neurons(suggested: &[u16]) -> Vec<u16> {
    suggested.iter().map(|&s| s.min(64)).collect()
}

/// `(suggested, default)` epochs: `min(200, max(50, n // 10))` and `min(100, it)`.
pub fn default_epochs(n_rows: usize) -> (u16, u16) {
    let suggested = (n_rows / 10).clamp(50, 200);
    let suggested = u16::try_from(suggested).unwrap_or(200);
    (suggested, suggested.min(100))
}

/// Severity of a message of the view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Success,
    Info,
    Warning,
    Error,
}

/// The message under the output select (`[arquitectura]` keys) and whether Rust
/// can train that combination (D71).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputCheck {
    /// Python's message: `binaria_sigmoid`, `multiclase_tanh`, `regresion_softmax`…
    pub key: String,
    pub severity: Severity,
    /// `Some(no_entrenable_*)` when the combination cannot be trained (the train
    /// button is disabled and this extra `st.error` is shown).
    pub untrainable_key: Option<String>,
}

/// [`OutputCheck`] of `output` for a task with `n_classes` classes (ignored in
/// regression).
pub fn output_check(task: TaskType, n_classes: usize, output: Activation) -> OutputCheck {
    let (prefix, severity) = match (task, n_classes <= 2, output) {
        (TaskType::Regression, _, Activation::Linear) => ("regresion", Severity::Success),
        (TaskType::Regression, _, Activation::Softmax) => ("regresion", Severity::Error),
        (TaskType::Regression, _, _) => ("regresion", Severity::Warning),
        (_, true, Activation::Sigmoid) => ("binaria", Severity::Success),
        (_, true, _) => ("binaria", Severity::Error),
        (_, false, Activation::Softmax) => ("multiclase", Severity::Success),
        (_, false, Activation::Sigmoid) => ("multiclase", Severity::Warning),
        (_, false, _) => ("multiclase", Severity::Error),
    };
    let untrainable_key = match resolve_training(task, output, output_size(task, n_classes)) {
        Err(NnError::UntrainableOutput { key, .. }) => Some(key),
        _ => None,
    };
    OutputCheck {
        key: format!("{prefix}_{}", output.as_str()),
        severity,
        untrainable_key,
    }
}

// ---------------------------------------------------------------------------
// Training history (shared by mltutor-nn, the diagnostics and the views)
// ---------------------------------------------------------------------------

/// Logs of one epoch (Keras `history.history[k][epoch]`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EpochLog {
    /// 0-based.
    pub epoch: usize,
    pub loss: f64,
    /// accuracy or MAE (see [`MetricKind::key`]).
    pub metric: f64,
    pub val_loss: f64,
    pub val_metric: f64,
    /// Learning rate at the start of the epoch's callbacks (before ReduceLROnPlateau
    /// reduces it); only recorded with ReduceLROnPlateau, as in Keras.
    pub learning_rate: Option<f64>,
}

/// `history.history` plus what the callbacks decided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct History {
    pub metric: MetricKind,
    pub epochs: Vec<EpochLog>,
    /// EarlyStopping's best epoch (0-based), whose weights the model has (Keras 3.12
    /// restores them at the end even without stopping early). `None` without
    /// EarlyStopping.
    pub best_epoch: Option<usize>,
    /// Epoch at which EarlyStopping stopped (Keras `stopped_epoch`, 0 = did not
    /// stop). `None` without EarlyStopping.
    pub stopped_epoch: Option<usize>,
}

impl History {
    /// `history.history['loss']`.
    pub fn loss(&self) -> Vec<f64> {
        self.epochs.iter().map(|e| e.loss).collect()
    }

    /// `history.history['val_loss']`.
    pub fn val_loss(&self) -> Vec<f64> {
        self.epochs.iter().map(|e| e.val_loss).collect()
    }

    /// The metric per epoch.
    pub fn metric_values(&self) -> Vec<f64> {
        self.epochs.iter().map(|e| e.metric).collect()
    }

    /// The validation metric per epoch.
    pub fn val_metric_values(&self) -> Vec<f64> {
        self.epochs.iter().map(|e| e.val_metric).collect()
    }

    /// Epochs (0-based) after which ReduceLROnPlateau lowered the learning rate: the
    /// next epoch records a smaller `learning_rate`.
    pub fn lr_drops(&self) -> Vec<usize> {
        self.epochs
            .windows(2)
            .filter_map(|w| match (w[0].learning_rate, w[1].learning_rate) {
                (Some(a), Some(b)) if b < a => Some(w[0].epoch),
                _ => None,
            })
            .collect()
    }

    /// The series the diagnostics read.
    pub fn series(&self) -> HistorySeries {
        HistorySeries {
            loss: self.loss(),
            val_loss: Some(self.val_loss()),
            metric_key: self.metric.key().to_owned(),
            metric: Some(self.metric_values()),
            val_metric: Some(self.val_metric_values()),
        }
    }
}

/// `history.history` as plain series, with the optional keys the Python
/// diagnostics test (`'val_loss' in history.history`…).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistorySeries {
    pub loss: Vec<f64>,
    pub val_loss: Option<Vec<f64>>,
    /// `"accuracy"` or `"mae"`.
    pub metric_key: String,
    pub metric: Option<Vec<f64>>,
    pub val_metric: Option<Vec<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approximate_mode_matches_sklearn_example() {
        // sklearn docstring: _approximate_mode([4, 2], 3, 0) -> [2, 1]
        let mut rng = RandomState::new(0);
        assert_eq!(approximate_mode(&[4, 2], 3, &mut rng), vec![2, 1]);
        assert_eq!(approximate_mode(&[5, 2], 4, &mut rng), vec![3, 1]);
    }

    #[test]
    fn resolve_matches_reference_table() {
        use Activation::*;
        let c = TaskType::Classification;
        assert!(resolve_training(c, Sigmoid, 3).is_err());
        assert!(resolve_training(c, Linear, 3).is_err());
        assert!(resolve_training(c, Softmax, 1).is_err());
        assert!(resolve_training(c, Linear, 1).is_err());
        let r = resolve_training(c, Tanh, 3).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.one_hot);
        assert_eq!(r.loss, LossKind::Mse);
        assert_eq!(r.metric, MetricKind::CategoricalAccuracy);
        let r = resolve_training(c, Tanh, 1).unwrap_or_else(|e| panic!("{e}"));
        assert!(!r.one_hot);
        assert_eq!(r.metric, MetricKind::BinaryAccuracy);
        let r =
            resolve_training(TaskType::Regression, Softmax, 1).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.loss, LossKind::Mse);
    }

    #[test]
    fn sizes_are_the_real_ones() {
        assert_eq!(
            split_sizes(150, 20),
            SplitSizes {
                train: 96,
                val: 24,
                test: 30
            }
        );
        assert_eq!(
            split_sizes(569, 20),
            SplitSizes {
                train: 364,
                val: 91,
                test: 114
            }
        );
        assert_eq!(total_parameters(&[4, 10, 10, 3]), 193);
        let rows = layer_rows(&[4, 10, 10, 3], Activation::Relu, Activation::Softmax);
        assert_eq!(rows.iter().map(|r| r.params).sum::<u64>(), 193);
        assert_eq!(default_epochs(569), (56, 56));
        assert_eq!(default_epochs(20640), (200, 100));
    }
}
