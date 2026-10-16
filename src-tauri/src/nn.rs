//! 🧠 Redes Neuronales, the part of the lesson that does not need a trained network
//! (fase5-redes.md §2.2–§2.3 and C22): the view of 🏗️ Arquitectura, the validation of
//! the controls, which outputs can be trained (D71), the loss and metric the app picks,
//! the real sizes of the split (D73), the Keras script with the current controls (D74)
//! and the rate limit of the `nn-epoch` event.
//!
//! Everything here is a direct transcription of `apps/neural_network_app.py` and
//! `algorithms/model_training.py::train_neural_network`, checked against
//! `tests/parity/nn/controls.json`.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::core_facade::ScriptBase;
use crate::ipc::{
    ArchitectureView, BatchHint, CommandError, Complexity, DatasetRef, DatasetSummary, EpochLog,
    LayerRow, NnActivation, NnArchitecture, NnMetric, NnTraining, OutputCheck, OutputLevel,
    SizeTips, SplitSizes, TaskType,
};

/// Event with the epochs of a training in progress.
pub const EPOCH_EVENT: &str = "nn-epoch";

/// `train_test_split(test_size=0.2, stratify=…)` of the lesson (fixed, no control).
pub const TEST_SIZE: f64 = 0.2;

/// Options of the selectors (§2.2–§2.3).
pub const BATCH_SIZES: [u16; 5] = [16, 32, 64, 128, 256];
pub const LEARNING_RATES: [f32; 4] = [0.001, 0.01, 0.1, 0.3];
pub const MAX_LAYERS: usize = 5;
pub const MAX_NEURONS: u16 = 256;

/// `content/redes/mensajes.toml`: the backend sends the messages of D71 and the error of a
/// class with a single sample; the UI reads the same file for everything else.
const MESSAGES: &str = include_str!("../../content/redes/mensajes.toml");

/// A text of `content/redes/mensajes.toml` (`fallback` if the file lacks it, logged).
pub fn nn_text(section: &str, key: &str, fallback: &str) -> String {
    static PARSED: OnceLock<Option<mltutor_core::content::Messages>> = OnceLock::new();
    let messages = PARSED.get_or_init(|| {
        mltutor_core::content::Messages::parse("content/redes/mensajes.toml", MESSAGES)
            .map_err(|error| log::error!("contenido: {error}"))
            .ok()
    });
    match messages.as_ref().map(|m| m.text(section, key)) {
        Some(Ok(text)) => text.to_owned(),
        Some(Err(error)) => {
            log::warn!("contenido: {error}");
            fallback.to_owned()
        }
        None => fallback.to_owned(),
    }
}

/// The task of the network and the size of its output layer (§2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NnTask {
    /// Two classes: one sigmoid neuron.
    Binary,
    /// `n` classes (`len(y.unique())`, Python also lands here with a single class).
    Multiclass(u32),
    Regression,
}

impl NnTask {
    pub fn of(summary: &DatasetSummary) -> Self {
        match (summary.task, &summary.class_names) {
            (TaskType::Classification, Some(names)) if names.len() == 2 => Self::Binary,
            (TaskType::Classification, Some(names)) => {
                Self::Multiclass(crate::ipc::to_u32(names.len()))
            }
            (TaskType::Classification, None) => Self::Multiclass(0),
            (TaskType::Regression, _) => Self::Regression,
        }
    }

    pub fn output_size(self) -> u32 {
        match self {
            Self::Multiclass(n) => n.max(1),
            Self::Binary | Self::Regression => 1,
        }
    }

    pub fn n_classes(self) -> Option<u32> {
        match self {
            Self::Binary => Some(2),
            Self::Multiclass(n) => Some(n),
            Self::Regression => None,
        }
    }

    pub fn task(self) -> TaskType {
        match self {
            Self::Regression => TaskType::Regression,
            _ => TaskType::Classification,
        }
    }

    /// Options of "Función de activación de salida", in Python's order.
    pub fn output_options(self) -> Vec<NnActivation> {
        use NnActivation::*;
        match self {
            Self::Regression => vec![Linear, Sigmoid, Tanh, Softmax],
            _ => vec![Sigmoid, Softmax, Linear, Tanh],
        }
    }

    /// The default (and recommended) output activation: D85 goes back to it when the
    /// task or the number of classes changes.
    pub fn recommended_output(self) -> NnActivation {
        match self {
            Self::Binary => NnActivation::Sigmoid,
            Self::Multiclass(_) => NnActivation::Softmax,
            Self::Regression => NnActivation::Linear,
        }
    }
}

/// Message under the output selector (§2.2.1) and, if it cannot be trained, why (D71).
pub fn output_check(task: NnTask, output: NnActivation) -> OutputCheck {
    use NnActivation::*;
    use OutputLevel::*;
    let (prefix, level) = match (task, output) {
        (NnTask::Binary, Sigmoid) => ("binaria", Success),
        (NnTask::Binary, _) => ("binaria", Error),
        (NnTask::Multiclass(_), Softmax) => ("multiclase", Success),
        (NnTask::Multiclass(_), Sigmoid) => ("multiclase", Warning),
        (NnTask::Multiclass(_), _) => ("multiclase", Error),
        (NnTask::Regression, Linear) => ("regresion", Success),
        (NnTask::Regression, Sigmoid | Tanh) => ("regresion", Warning),
        (NnTask::Regression, _) => ("regresion", Error),
    };
    let key = format!("{prefix}_{}", output.keras_name());
    OutputCheck {
        level,
        key,
        untrainable: untrainable(task, output).map(|(_, message)| message),
    }
}

/// D71: the key and message of a combination Keras cannot train, or `None`.
pub fn untrainable(task: NnTask, output: NnActivation) -> Option<(&'static str, String)> {
    let recommended = task.recommended_output().keras_name();
    match (task, output) {
        (NnTask::Multiclass(n), NnActivation::Sigmoid) => Some((
            "no_entrenable_multiclase_sigmoid",
            nn_text(
                "arquitectura",
                "no_entrenable_multiclase_sigmoid",
                "🚫 Con {clases} clases y salida sigmoid, MLTutor no puede entrenar la red. Usa softmax (o tanh).",
            )
            .replace("{clases}", &n.to_string()),
        )),
        (NnTask::Binary | NnTask::Multiclass(_), NnActivation::Linear) => Some((
            "no_entrenable_linear",
            nn_text(
                "arquitectura",
                "no_entrenable_linear",
                "🚫 Con salida linear la pérdida de clasificación no se puede calcular. Usa {recomendada}.",
            )
            .replace("{recomendada}", recommended),
        )),
        (NnTask::Binary, NnActivation::Softmax) => Some((
            "no_entrenable_binaria_softmax",
            nn_text(
                "arquitectura",
                "no_entrenable_binaria_softmax",
                "🚫 Softmax con una sola neurona siempre vale 1: la red no puede distinguir las dos clases. Usa sigmoid.",
            ),
        )),
        _ => None,
    }
}

/// `CommandError::UntrainableOutput` for the combination, if it is one of D71.
pub fn check_trainable(task: NnTask, output: NnActivation) -> Result<(), CommandError> {
    match untrainable(task, output) {
        Some((key, message)) => Err(CommandError::UntrainableOutput {
            key: key.to_owned(),
            n_classes: task.n_classes(),
            message,
        }),
        None => Ok(()),
    }
}

/// The class with fewer than 2 samples that makes the stratified split fail.
pub fn too_few_per_class(class_name: &str) -> CommandError {
    let message = nn_text(
        "entrenamiento",
        "error_pocas_muestras_clase",
        "🚫 La clase «{clase}» solo tiene una muestra. La red divide los datos de forma estratificada (cada clase en entrenamiento y en prueba), así que necesita al menos 2 muestras de cada clase.",
    )
    .replace("{clase}", class_name);
    CommandError::TooFewPerClass {
        class_name: class_name.to_owned(),
        message,
    }
}

/// Loss, metric and labels the app uses for the output activation (§2.2.1; only for
/// trainable combinations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved {
    pub loss: &'static str,
    pub metric: NnMetric,
    /// y is one-hot encoded (softmax, or more than one neuron without sigmoid).
    pub one_hot: bool,
}

pub fn resolve(task: NnTask, output: NnActivation) -> Resolved {
    match task {
        NnTask::Regression => Resolved {
            loss: "mse",
            metric: NnMetric::Mae,
            one_hot: false,
        },
        NnTask::Binary | NnTask::Multiclass(_) => {
            let neurons = task.output_size();
            let one_hot =
                output == NnActivation::Softmax || (neurons > 1 && output != NnActivation::Sigmoid);
            let loss = match output {
                NnActivation::Sigmoid => "binary_crossentropy",
                NnActivation::Softmax => "categorical_crossentropy",
                _ => "mse",
            };
            Resolved {
                loss,
                metric: NnMetric::Accuracy,
                one_hot,
            }
        }
    }
}

/// "💡 Sugerencia para capa i" (`max(10, entrada // i)`, at least 2·clases in multiclass).
pub fn suggested_neurons(input: u32, task: NnTask, layer: usize) -> u16 {
    let divisor = u32::try_from(layer + 1).unwrap_or(u32::MAX);
    let mut suggestion = (input / divisor).max(10);
    if let NnTask::Multiclass(n) = task {
        if n > 2 {
            suggestion = suggestion.max(2 * n);
        }
    }
    u16::try_from(suggestion).unwrap_or(u16::MAX)
}

/// `min(200, max(50, filas // 10))`.
pub fn suggested_epochs(n_rows: u32) -> u16 {
    u16::try_from((n_rows / 10).clamp(50, 200)).unwrap_or(200)
}

/// D73: `test = ceil(0.2·n)`, `fit = floor(train·(1 − v))` and the last rows as validation.
pub fn split_sizes(n_rows: u32, validation_pct: u8) -> SplitSizes {
    let n = f64::from(n_rows);
    let test = ((TEST_SIZE * n).ceil() as u32).min(n_rows);
    let train = n_rows - test;
    let fit = (f64::from(train) * (1.0 - f64::from(validation_pct) / 100.0)).floor() as u32;
    let fit = fit.min(train);
    SplitSizes {
        train: fit,
        val: train - fit,
        test,
    }
}

/// `[entrada, ocultas…, salida]`.
pub fn layer_sizes(input: u32, arch: &NnArchitecture, task: NnTask) -> Vec<u32> {
    std::iter::once(input)
        .chain(arch.hidden.iter().map(|&n| u32::from(n)))
        .chain(std::iter::once(task.output_size()))
        .collect()
}

/// `calculate_network_parameters` (= `count_params()` of Keras).
pub fn total_params(layers: &[u32]) -> u64 {
    layers
        .windows(2)
        .map(|w| u64::from(w[0]) * u64::from(w[1]) + u64::from(w[1]))
        .sum()
}

/// 🏗️ Arquitectura of `summary` with the controls `arch` (§2.2; D76 table).
pub fn architecture_view(
    dataset: DatasetRef,
    summary: &DatasetSummary,
    arch: &NnArchitecture,
    validation_pct: u8,
) -> ArchitectureView {
    let task = NnTask::of(summary);
    let input = summary.n_features;
    let n_rows = summary.n_rows;
    let sizes = layer_sizes(input, arch, task);
    let last = sizes.len() - 1;
    let layers: Vec<LayerRow> = sizes
        .iter()
        .enumerate()
        .map(|(i, &neurons)| {
            let inputs = if i == 0 { 0 } else { sizes[i - 1] };
            LayerRow {
                name: match i {
                    0 => nn_text("arquitectura", "detalles_entrada", "Entrada"),
                    i if i == last => nn_text("arquitectura", "detalles_salida", "Salida"),
                    i => nn_text("arquitectura", "detalles_oculta", "Oculta {i}")
                        .replace("{i}", &i.to_string()),
                },
                neurons,
                inputs,
                params: if i == 0 {
                    0
                } else {
                    u64::from(inputs) * u64::from(neurons) + u64::from(neurons)
                },
                activation: match i {
                    0 => None,
                    i if i == last => Some(arch.output_activation),
                    _ => Some(arch.activation),
                },
            }
        })
        .collect();
    let total = total_params(&sizes);
    let ratio = if n_rows > 0 {
        total as f64 / f64::from(n_rows)
    } else {
        0.0
    };
    let batch = u32::from(arch.batch_size);
    let suggested: Vec<u16> = (0..MAX_LAYERS)
        .map(|i| suggested_neurons(input, task, i))
        .collect();
    ArchitectureView {
        dataset,
        input,
        output: task.output_size(),
        task: task.task(),
        n_classes: task.n_classes(),
        n_rows,
        default_neurons: suggested.iter().map(|&s| s.min(64)).collect(),
        suggested_neurons: suggested,
        layers,
        total_params: total,
        complexity_ratio: ratio,
        complexity: if ratio < 0.1 {
            Complexity::Low
        } else if ratio < 1.0 {
            Complexity::Medium
        } else {
            Complexity::High
        },
        output_options: task.output_options(),
        recommended_output: task.recommended_output(),
        output_check: output_check(task, arch.output_activation),
        batch_hint: if batch >= n_rows / 4 {
            BatchHint::Large
        } else if batch >= 32 {
            BatchHint::Optimal
        } else {
            BatchHint::Small
        },
        size_tips: SizeTips {
            half: input / 2,
            double: input * 2,
            quadruple: input * 4,
            max_params: n_rows / 10,
            batch_recommended: if n_rows < 1000 {
                32
            } else if n_rows < 10_000 {
                64
            } else {
                128
            },
        },
        default_epochs: suggested_epochs(n_rows).min(100),
        suggested_epochs: suggested_epochs(n_rows),
        split: split_sizes(n_rows, validation_pct),
    }
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1e-3)
}

/// The ranges of the controls (§5.3). The UI never sends anything else; a bad request
/// gets a Spanish message instead of reaching the engine.
pub fn validate(arch: &NnArchitecture, training: &NnTraining) -> Result<(), CommandError> {
    let bad = |message: &str| Err(CommandError::invalid_config(message));
    if arch.hidden.is_empty() || arch.hidden.len() > MAX_LAYERS {
        return bad("El número de capas ocultas debe estar entre 1 y 5.");
    }
    if arch.hidden.iter().any(|&n| n == 0 || n > MAX_NEURONS) {
        return bad("Cada capa oculta debe tener entre 1 y 256 neuronas.");
    }
    if matches!(
        arch.activation,
        NnActivation::Softmax | NnActivation::Linear
    ) {
        return bad("La activación de las capas ocultas debe ser relu, tanh o sigmoid.");
    }
    if !(arch.dropout.is_finite() && (0.0..=0.8 + 1e-6).contains(&arch.dropout)) {
        return bad("La tasa de dropout debe estar entre 0 y 0,8.");
    }
    if !BATCH_SIZES.contains(&arch.batch_size) {
        return bad("El tamaño de batch debe ser 16, 32, 64, 128 o 256.");
    }
    if !LEARNING_RATES
        .iter()
        .any(|&lr| close(training.learning_rate, lr))
    {
        return bad("La tasa de aprendizaje debe ser 0,001, 0,01, 0,1 o 0,3.");
    }
    if !(10..=500).contains(&training.epochs) {
        return bad("Las épocas deben estar entre 10 y 500.");
    }
    if !(10..=40).contains(&training.validation_pct) {
        return bad("El porcentaje de validación debe estar entre 10 y 40.");
    }
    if training.patience.is_some_and(|p| !(5..=50).contains(&p)) {
        return bad("La paciencia debe estar entre 5 y 50 épocas.");
    }
    if training
        .lr_factor
        .is_some_and(|f| !(f.is_finite() && (0.1 - 1e-6..=0.9 + 1e-6).contains(&f)))
    {
        return bad("El factor de reducción debe estar entre 0,1 y 0,9.");
    }
    if !(1..=9999).contains(&training.seed) {
        return bad("La semilla debe estar entre 1 y 9999.");
    }
    Ok(())
}

/// The patience of ReduceLROnPlateau: `max(1, paciencia // 2)`, or 5 without early stopping.
pub fn reduce_lr_patience(training: &NnTraining) -> u8 {
    training.patience.map_or(5, |p| (p / 2).max(1))
}

/// Context of `red_entrenamiento.py.j2` (fixture `code/nn_training.json` → `context`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NnTrainingContext {
    #[serde(flatten)]
    pub base: ScriptBase,
    pub task: String,
    pub architecture: Vec<u32>,
    pub activation: String,
    pub output_activation: String,
    pub loss: String,
    pub one_hot: bool,
    pub dropout_rate: f64,
    pub optimizer: String,
    pub learning_rate: f64,
    pub batch_size: u16,
    pub epochs: u16,
    pub validation_split: f64,
    pub early_stopping: bool,
    pub patience: Option<u8>,
    pub reduce_lr: bool,
    pub lr_factor: Option<f64>,
    pub reduce_lr_patience: Option<u8>,
    pub sample: Option<Vec<f64>>,
}

/// The f32 of a control as the f64 Python wrote (0.1, not 0.10000000149011612).
pub fn control_f64(value: f32) -> f64 {
    // The shortest decimal of the f32 is the value of the selector or slider.
    value.to_string().parse().unwrap_or(f64::from(value))
}

impl NnTrainingContext {
    pub fn new(
        base: ScriptBase,
        summary: &DatasetSummary,
        arch: &NnArchitecture,
        training: &NnTraining,
    ) -> Result<Self, CommandError> {
        let task = NnTask::of(summary);
        check_trainable(task, arch.output_activation)?;
        let resolved = resolve(task, arch.output_activation);
        Ok(Self {
            base,
            task: match task.task() {
                TaskType::Classification => "classification",
                TaskType::Regression => "regression",
            }
            .to_owned(),
            architecture: layer_sizes(summary.n_features, arch, task),
            activation: arch.activation.keras_name().to_owned(),
            output_activation: arch.output_activation.keras_name().to_owned(),
            loss: resolved.loss.to_owned(),
            one_hot: resolved.one_hot,
            dropout_rate: (control_f64(arch.dropout) * 10.0).round() / 10.0,
            optimizer: arch.optimizer.keras_name().to_owned(),
            learning_rate: control_f64(training.learning_rate),
            batch_size: arch.batch_size,
            epochs: training.epochs,
            validation_split: f64::from(training.validation_pct) / 100.0,
            early_stopping: training.patience.is_some(),
            patience: training.patience,
            reduce_lr: training.lr_factor.is_some(),
            lr_factor: training
                .lr_factor
                .map(|f| (control_f64(f) * 100.0).round() / 100.0),
            reduce_lr_patience: training.lr_factor.map(|_| reduce_lr_patience(training)),
            sample: None,
        })
    }

    pub fn render(&self) -> Result<String, CommandError> {
        mltutor_core::codegen::render_named("red_entrenamiento.py.j2", self)
            .map_err(|error| CommandError::internal(format!("plantilla: {error}")))
    }
}

/// Minimum time between two `nn-epoch` events (at most 20 per second, §11).
pub const EPOCH_EVENT_INTERVAL: Duration = Duration::from_millis(50);

/// Groups the epochs of a fast training: Iris trains 500 epochs in well under a second,
/// and an event per epoch would flood the IPC and the chart. The epochs are never
/// dropped: each event carries every epoch since the previous one.
#[derive(Default)]
pub struct EpochBatcher {
    inner: Mutex<BatcherState>,
}

#[derive(Default)]
struct BatcherState {
    pending: Vec<EpochLog>,
    last: Option<Instant>,
}

impl EpochBatcher {
    /// Adds an epoch; returns the batch to send now, if it is time (or it is the last).
    pub fn push(&self, log: EpochLog, last: bool, now: Instant) -> Option<Vec<EpochLog>> {
        // A poisoned batcher only means a panicked emitter: its state is still valid.
        let mut state = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        state.pending.push(log);
        let due = state
            .last
            .is_none_or(|at| now.saturating_duration_since(at) >= EPOCH_EVENT_INTERVAL);
        if last || due {
            state.last = Some(now);
            Some(std::mem::take(&mut state.pending))
        } else {
            None
        }
    }

    /// The epochs not sent yet (the training ended early or was cancelled).
    pub fn flush(&self) -> Vec<EpochLog> {
        let mut state = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        std::mem::take(&mut state.pending)
    }
}

/// Status bar during the epochs: `[entrenamiento] progreso_epoca` ("Época 3/50 · pérdida: 0.8123").
pub fn epoch_status(epoch: u16, epochs: u16, loss: Option<f64>) -> String {
    let template = nn_text(
        "entrenamiento",
        "progreso_epoca",
        "Época {epoca}/{epocas} · pérdida: {perdida:.4f}",
    );
    template
        .replace("{epoca}", &epoch.to_string())
        .replace("{epocas}", &epochs.to_string())
        .replace(
            "{perdida:.4f}",
            &loss.map_or_else(|| "nan".to_owned(), |l| format!("{l:.4}")),
        )
}

/// Status bar during the steps 1–4: the text of `paso_i` without Markdown.
pub fn phase_status(phase: crate::ipc::TrainPhase, epochs: u16) -> String {
    use crate::ipc::TrainPhase::*;
    let key = match phase {
        Preparing => "paso_1",
        Building => "paso_2",
        Compiling => "paso_3",
        Training => "paso_4",
    };
    nn_text("entrenamiento", key, "Entrenando la red neuronal...")
        .replace("{epocas}", &epochs.to_string())
        .replace("**", "")
}

/// `None` for NaN and infinities (JSON has no NaN).
pub fn finite_log(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests;
