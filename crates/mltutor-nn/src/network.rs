//! The lesson's whole training (`algorithms/model_training.py::train_neural_network`)
//! and the trained network as a [`Predictor`].

use mltutor_core::dataset::{Target, TaskType};
use mltutor_core::grid::MeanGrid;
use mltutor_core::model::{ModelError, Prediction, Predictor, TrainControl};
use mltutor_core::nn_data::{
    self, Activation, History, NnArchitecture, NnError, NnTraining, ResolvedTraining, Scaler,
    SplitSizes,
};
use mltutor_core::nn_stats::{self, ActivationParams, ActivationsView, WeightsView};
use mltutor_core::sampling::RandomState;
use mltutor_core::split::TrainTestSplit;
use ndarray::{s, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::loss::argmax;
use crate::mlp::Mlp;
use crate::train::{evaluate, fit, glorot_init, FitControl, FitData, FitSpec};

/// A trained network with its preprocessing: it takes rows in **original units**
/// (the [`Scaler`] is applied inside) and returns dataset class codes or values.
///
/// Built with [`NeuralNetwork::new`] (or deserialised, which runs the same checks):
/// the shapes of the network, the scaler and the classes agree, so prediction cannot
/// panic. A network whose weights or scaler are not finite (a training that
/// diverged, K1) is kept for the views, but [`NeuralNetwork::is_valid`] is `false`
/// and every prediction returns [`ModelError::Numerical`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "NetworkRepr", into = "NetworkRepr")]
pub struct NeuralNetwork {
    mlp: Mlp,
    scaler: Scaler,
    task: TaskType,
    classes: Vec<usize>,
    resolved: ResolvedTraining,
    /// Every weight, bias, mean and scale is finite (computed in `new`).
    valid: bool,
}

/// The serialised form of [`NeuralNetwork`] (without the derived `valid`).
#[derive(Clone, Serialize, Deserialize)]
struct NetworkRepr {
    mlp: Mlp,
    scaler: Scaler,
    task: TaskType,
    classes: Vec<usize>,
    resolved: ResolvedTraining,
}

impl TryFrom<NetworkRepr> for NeuralNetwork {
    type Error = ModelError;

    fn try_from(r: NetworkRepr) -> Result<Self, ModelError> {
        NeuralNetwork::new(r.mlp, r.scaler, r.task, r.classes, r.resolved)
    }
}

impl From<NeuralNetwork> for NetworkRepr {
    fn from(n: NeuralNetwork) -> Self {
        NetworkRepr {
            mlp: n.mlp,
            scaler: n.scaler,
            task: n.task,
            classes: n.classes,
            resolved: n.resolved,
        }
    }
}

/// Message of [`ModelError::Numerical`] for a network that is not valid.
const DIVERGED: &str = "la red ha divergido y sus pesos contienen NaN o infinito; vuelve a entrenarla con un learning rate menor";

/// Hidden activations of some rows (C26, K3).
#[derive(Debug, Clone, PartialEq)]
pub struct Activations {
    /// `min(sample_size, rows)`.
    pub rows_used: usize,
    /// One row-major `rows_used × units` matrix per hidden layer.
    pub layers: Vec<Vec<f32>>,
}

impl NeuralNetwork {
    /// A network with its scaler and classes. `classes` are the dataset codes of the
    /// classes the label encoder saw (sorted): the i-th output neuron (or `[0, 1]` of
    /// a single sigmoid/tanh neuron) is `classes[i]`; empty in regression.
    ///
    /// Errors: malformed `mlp`, a scaler of another width, or classes that do not
    /// match the outputs.
    pub fn new(
        mlp: Mlp,
        scaler: Scaler,
        task: TaskType,
        classes: Vec<usize>,
        resolved: ResolvedTraining,
    ) -> Result<Self, ModelError> {
        mlp.validate()?;
        let p = mlp.n_inputs();
        if scaler.mean.len() != p || scaler.scale.len() != p {
            return Err(ModelError::FeatureMismatch {
                expected: p,
                found: scaler.mean.len(),
            });
        }
        let k = mlp.n_outputs();
        let classes_ok = match task {
            TaskType::Classification => {
                let expected = if k == 1 { 2 } else { k };
                classes.len() == expected && classes.windows(2).all(|w| w[0] < w[1])
            }
            TaskType::Regression => classes.is_empty(),
        };
        if !classes_ok {
            return Err(ModelError::InvalidConfig {
                reason: format!("{} clases para {k} neuronas de salida", classes.len()),
            });
        }
        let valid = mlp.is_finite() && scaler.is_finite();
        Ok(Self {
            mlp,
            scaler,
            task,
            classes,
            resolved,
            valid,
        })
    }

    /// The network (weights for the views and the exports).
    pub fn mlp(&self) -> &Mlp {
        &self.mlp
    }

    /// The StandardScaler applied inside.
    pub fn scaler(&self) -> &Scaler {
        &self.scaler
    }

    /// Dataset codes of the output classes (see [`NeuralNetwork::new`]); empty in
    /// regression. The same as [`Predictor::classes`], without the `Option`.
    pub fn class_codes(&self) -> &[usize] {
        &self.classes
    }

    /// Loss, metric and target encoding it was trained with.
    pub fn resolved(&self) -> ResolvedTraining {
        self.resolved
    }

    /// `false` when some weight, bias or scaler value is NaN or infinite (a diverged
    /// training): predictions fail with [`ModelError::Numerical`].
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Whether [`Predictor::predict_proba`] really is a probability: softmax rows
    /// sum to 1 and a sigmoid lies in (0, 1), but a **tanh** output (which the app
    /// lets the student choose, with MSE) lies in (−1, 1), so `[1 − p, p]` can be
    /// negative or above 1. Views that read probabilities (ROC/PR, confidence,
    /// probability tables of F2) must be hidden or carry a warning when this is
    /// `false` (fase5-redes.md D87).
    pub fn output_is_probability(&self) -> bool {
        matches!(self.mlp.output, Activation::Softmax | Activation::Sigmoid)
    }

    /// Scales `x` (original units) to float32 rows.
    pub fn scale(&self, x: ArrayView2<'_, f64>) -> Vec<f32> {
        self.scaler.transform_f32(x)
    }

    /// Raw network output for rows in original units (`n × n_outputs`, f32 forward).
    pub fn raw_output(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, ModelError> {
        self.check(x)?;
        let n = x.nrows();
        let out = self.mlp.forward(&self.scale(x), n)?;
        let k = self.mlp.n_outputs();
        Array2::from_shape_vec((n, k), out.into_iter().map(f64::from).collect()).map_err(|e| {
            ModelError::Numerical {
                message: e.to_string(),
            }
        })
    }

    /// Hidden activations (C26, K3) of the **first** `min(sample_size, rows)` rows of
    /// `x` (original units; the app passes X_test): scaled inside, after the
    /// activation and before dropout.
    pub fn hidden_activations(
        &self,
        x: ArrayView2<'_, f64>,
        sample_size: usize,
    ) -> Result<Activations, ModelError> {
        let rows = sample_size.min(x.nrows());
        let x = x.slice(s![..rows, ..]);
        self.check(x)?;
        let layers = self.mlp.hidden_activations(&self.scale(x), rows)?;
        Ok(Activations {
            rows_used: rows,
            layers,
        })
    }

    /// 📉 Análisis de Activaciones of the first `params.sample_size` rows of `x`
    /// (original units): statistics, binned histograms and the neuron map (K2, K3).
    pub fn activations_view(
        &self,
        x: ArrayView2<'_, f64>,
        params: &ActivationParams,
    ) -> Result<ActivationsView, ModelError> {
        let acts = self.hidden_activations(x, params.sample_size)?;
        Ok(nn_stats::activations_view(
            &self.mlp.views(),
            &acts.layers,
            acts.rows_used,
            params,
        ))
    }

    /// 🧠 Análisis de Pesos: statistics and binned histograms (K2; the sliders' bins,
    /// [`nn_stats::DEFAULT_WEIGHT_BINS`] and [`nn_stats::DEFAULT_BIAS_BINS`] by
    /// default). Also for an invalid network (NaN values are left out of the bins).
    pub fn weights_view(&self, weight_bins: usize, bias_bins: usize) -> WeightsView {
        nn_stats::weights_view(&self.mlp.views(), weight_bins, bias_bins)
    }

    /// 🎯 Superficie (D78): the whole network on a grid of features `fx`/`fy`, the
    /// rest at their mean over `x_ref` (X_test), in the scaled space of the network
    /// ([`mltutor_core::grid::mean_grid`] with this network's scaler). Classification
    /// and regression.
    pub fn mean_grid(
        &self,
        x_ref: ArrayView2<'_, f64>,
        y_ref: &Target,
        fx: usize,
        fy: usize,
        control: &TrainControl<'_>,
    ) -> Result<MeanGrid, ModelError> {
        mltutor_core::grid::mean_grid(self, x_ref, y_ref, &self.scaler, fx, fy, control)
    }

    fn check(&self, x: ArrayView2<'_, f64>) -> Result<(), ModelError> {
        if x.ncols() != self.mlp.n_inputs() {
            return Err(ModelError::FeatureMismatch {
                expected: self.mlp.n_inputs(),
                found: x.ncols(),
            });
        }
        if !self.valid {
            return Err(ModelError::Numerical {
                message: DIVERGED.to_owned(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::NonFinite);
        }
        Ok(())
    }

    /// Class index (into `classes`) of one output row: argmax, or `p > 0.5` with one
    /// neuron (Keras' `BinaryAccuracy` threshold, D87).
    fn class_of(&self, row: &[f32]) -> usize {
        if row.len() == 1 {
            usize::from(row[0] > 0.5)
        } else {
            argmax(row)
        }
    }
}

impl Predictor for NeuralNetwork {
    fn task(&self) -> TaskType {
        self.task
    }

    fn n_features(&self) -> usize {
        self.mlp.n_inputs()
    }

    fn classes(&self) -> Option<&[usize]> {
        (self.task == TaskType::Classification).then_some(self.classes.as_slice())
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Prediction, ModelError> {
        self.check(x)?;
        let n = x.nrows();
        let k = self.mlp.n_outputs();
        let out = self.mlp.forward(&self.scale(x), n)?;
        Ok(match self.task {
            TaskType::Classification => Prediction::Classes(
                out.chunks(k)
                    .map(|row| self.classes.get(self.class_of(row)).copied().unwrap_or(0))
                    .collect(),
            ),
            // Regression: the first output (Python `raw[:, 0]`).
            TaskType::Regression => {
                Prediction::Values(out.chunks(k).map(|r| f64::from(r[0])).collect())
            }
        })
    }

    /// The network output as "probabilities" (contract: `proba` = output): with one
    /// neuron `[1 − p, p]`; with several, the outputs as they are. Softmax rows sum
    /// to 1 and a sigmoid is in (0, 1), but a **tanh** output is in (−1, 1): it is
    /// not a probability ([`NeuralNetwork::output_is_probability`]).
    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Option<Result<Array2<f64>, ModelError>> {
        if self.task != TaskType::Classification {
            return None;
        }
        Some(self.raw_output(x).map(|raw| {
            if raw.ncols() == 1 {
                let mut out = Array2::zeros((raw.nrows(), 2));
                for (i, p) in raw.column(0).iter().enumerate() {
                    out[[i, 0]] = 1.0 - p;
                    out[[i, 1]] = *p;
                }
                out
            } else {
                raw
            }
        }))
    }

    fn refit(
        &self,
        _x: ArrayView2<'_, f64>,
        _y: &Target,
        _control: &TrainControl<'_>,
    ) -> Result<Box<dyn Predictor>, ModelError> {
        // D78: the surface keeps the full network with the other features at their
        // mean (`mltutor_core::grid::mean_grid`); it is never retrained on 2 columns.
        Err(ModelError::InvalidSelection {
            reason: "la red neuronal no se reentrena con dos características".to_owned(),
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Everything `train_model` keeps (fase5-redes.md §5.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainedNetwork {
    pub network: NeuralNetwork,
    pub history: History,
    /// Rows of the dataset in train/test (stratified in classification).
    pub split: TrainTestSplit,
    /// Rows really used to fit, to validate and to test (D73).
    pub sizes: SplitSizes,
    /// `model.evaluate(X_test, y_test)` (batches of 32): loss and metric.
    pub test_loss: f64,
    pub test_metric: f64,
    /// Learning rate at the end (after ReduceLROnPlateau).
    pub final_learning_rate: f32,
    /// `[input, hidden…, output]`.
    pub layers: Vec<usize>,
}

impl TrainedNetwork {
    /// The scaled test rows of the dataset `x` in float32 (without copying them).
    pub fn x_test_scaled(&self, x: ArrayView2<'_, f64>) -> Vec<f32> {
        self.network.scaler.transform_rows_f32(x, &self.split.test)
    }

    /// The test rows of the dataset `x` (original units, split order).
    pub fn x_test(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        x.select(ndarray::Axis(0), &self.split.test)
    }

    /// 📉 Activaciones of the first `params.sample_size` test rows of the dataset `x`
    /// (only those rows are copied).
    pub fn test_activations(
        &self,
        x: ArrayView2<'_, f64>,
        params: &ActivationParams,
    ) -> Result<ActivationsView, ModelError> {
        let rows = &self.split.test[..params.sample_size.min(self.split.test.len())];
        if rows.iter().any(|&r| r >= x.nrows()) {
            return Err(ModelError::InvalidSelection {
                reason: "el dataset no es el del entrenamiento".to_owned(),
            });
        }
        let first = x.select(ndarray::Axis(0), rows);
        self.network.activations_view(first.view(), params)
    }
}

/// Options of [`train_network`] that the app leaves at their defaults (K5).
#[derive(Debug, Clone, PartialEq)]
pub struct TrainOptions {
    /// Shuffle the training rows every epoch (Keras `shuffle=True`; the app always
    /// does). The level (a) parity tests turn it off.
    pub shuffle: bool,
    /// Initial weights instead of the Glorot ones (level (a) parity starts from
    /// Keras' weights); they must have the architecture's shape.
    pub init: Option<Mlp>,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            shuffle: true,
            init: None,
        }
    }
}

/// Encodes a target for the network: `(y row-major n × k, classes)`.
fn encode(
    target: &Target,
    rows: &[usize],
    task: TaskType,
    resolved: ResolvedTraining,
    k: usize,
    classes: &[usize],
) -> Result<Vec<f32>, NnError> {
    match (task, target) {
        (TaskType::Classification, Target::Classes { codes, .. }) => {
            let mut y = Vec::with_capacity(rows.len() * k);
            for &r in rows {
                let code = codes.get(r).map(|&c| c as usize).unwrap_or(usize::MAX);
                let idx = classes.partition_point(|&c| c < code);
                if resolved.one_hot {
                    y.extend((0..k).map(|j| if j == idx { 1.0 } else { 0.0 }));
                } else {
                    y.push(idx as f32);
                }
            }
            Ok(y)
        }
        (_, Target::Values { values }) => {
            let y: Vec<f32> = rows.iter().map(|&r| values[r] as f32).collect();
            if y.iter().any(|v| !v.is_finite()) {
                return Err(ModelError::NonFinite.into());
            }
            Ok(y)
        }
        _ => Err(ModelError::InvalidTarget.into()),
    }
}

/// The lesson's training (§1.2): StandardScaler on **all** rows (D72), split
/// (stratified, 0.2, 42), label encoding / one-hot, Glorot initialisation and `fit`
/// with the seed of `training` (D70), then `evaluate` on the test rows.
///
/// `options` ([`TrainOptions`], `Default` for the app) can replace the Glorot weights
/// and turn shuffling off (parity tests). Errors: [`NnError::UntrainableOutput`]
/// (D71), [`NnError::TooFewPerClass`], [`NnError::Cancelled`] (checked before every
/// training and validation batch and in the final `evaluate`), shape and NaN errors.
/// A diverged training is **not** an error (K1): see `history.diverged_epoch` and
/// [`NeuralNetwork::is_valid`].
pub fn train_network(
    x: ArrayView2<'_, f64>,
    y: &Target,
    arch: &NnArchitecture,
    training: &NnTraining,
    options: TrainOptions,
    control: &FitControl<'_>,
) -> Result<TrainedNetwork, NnError> {
    let TrainOptions { shuffle, init } = options;
    let n = x.nrows();
    if n == 0 {
        return Err(ModelError::Empty.into());
    }
    if y.len() != n {
        return Err(ModelError::LengthMismatch {
            rows: n,
            targets: y.len(),
        }
        .into());
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ModelError::NonFinite.into());
    }
    validate(arch, training)?;
    let task = match y {
        Target::Classes { .. } => TaskType::Classification,
        Target::Values { .. } => TaskType::Regression,
    };
    let codes = y.class_codes();
    let mut present: Vec<usize> = codes.clone().unwrap_or_default();
    present.sort_unstable();
    present.dedup();
    if task == TaskType::Classification {
        mltutor_core::model::check_n_classes(present.len())?;
        if present.len() < 2 {
            return Err(ModelError::TooFewClasses {
                found: present.len(),
            }
            .into());
        }
    }
    let k = nn_data::output_size(task, present.len());
    let resolved = nn_data::resolve_training(task, arch.output_activation, k)?;
    let scaler = Scaler::fit(x);
    let class_names: Vec<String> = match y {
        Target::Classes { class_names, .. } => class_names.clone(),
        Target::Values { .. } => Vec::new(),
    };
    let split = nn_data::lesson_split(task, codes.as_deref(), &class_names, n)?;
    // LabelEncoder on y_train.
    let mut classes: Vec<usize> = match &codes {
        Some(c) => split.train.iter().map(|&r| c[r]).collect(),
        None => Vec::new(),
    };
    classes.sort_unstable();
    classes.dedup();
    // Scaled by row index: no copy of the train/test rows in f64 (review F5a, 11).
    let x_fit = scaler.transform_rows_f32(x, &split.train);
    let y_fit = encode(y, &split.train, task, resolved, k, &classes)?;
    let x_te = scaler.transform_rows_f32(x, &split.test);
    let y_te = encode(y, &split.test, task, resolved, k, &classes)?;

    let mut layers = vec![x.ncols()];
    layers.extend(arch.hidden.iter().map(|&h| usize::from(h)));
    layers.push(k);
    let mut rng = RandomState::new(training.seed);
    let glorot = glorot_init(&layers, arch.activation, arch.output_activation, &mut rng);
    let init = match init {
        Some(m) if m.architecture() == layers => Mlp {
            hidden: arch.activation,
            output: arch.output_activation,
            ..m
        },
        Some(_) => {
            return Err(NnError::InvalidConfig {
                reason: "los pesos iniciales no tienen la forma de la arquitectura".to_owned(),
            })
        }
        None => glorot,
    };
    let spec = FitSpec {
        loss: resolved.loss,
        metric: resolved.metric,
        optimizer: arch.optimizer,
        learning_rate: training.learning_rate,
        batch_size: usize::from(arch.batch_size),
        epochs: usize::from(training.epochs),
        validation_split: training.validation_split(),
        patience: training.patience.map(usize::from),
        // The slider value as the student reads it (0.3, not f32's 0.30000001): Python
        // multiplies the learning rate by the float `factor`.
        lr_factor: training
            .lr_factor
            .map(|f| f.to_string().parse::<f64>().unwrap_or(f64::from(f))),
        reduce_lr_patience: training.reduce_lr_patience(),
        dropout: arch.dropout,
        shuffle,
    };
    let result = fit(
        init,
        FitData {
            x: &x_fit,
            y: &y_fit,
            n: split.train.len(),
        },
        &spec,
        &mut rng,
        control,
    )?;
    let (test_loss, test_metric) = evaluate(
        &result.mlp,
        &x_te,
        &y_te,
        split.test.len(),
        resolved.loss,
        resolved.metric,
        32,
        control,
    )?;
    let sizes = nn_data::split_sizes(n, training.validation_pct);
    // In classification `classes` has the codes of y_train: 2 for one output
    // neuron, k otherwise (the stratified split puts every class in train).
    let network = NeuralNetwork::new(result.mlp, scaler, task, classes, resolved)?;
    Ok(TrainedNetwork {
        network,
        history: result.history,
        split,
        sizes,
        test_loss,
        test_metric,
        final_learning_rate: result.final_learning_rate,
        layers,
    })
}

/// Ranges of the controls (the UI cannot send others; the IPC is checked anyway).
pub fn validate(arch: &NnArchitecture, training: &NnTraining) -> Result<(), NnError> {
    let bad = |reason: &str| {
        Err(NnError::InvalidConfig {
            reason: reason.to_owned(),
        })
    };
    if arch.hidden.is_empty() || arch.hidden.len() > 5 {
        return bad("entre 1 y 5 capas ocultas");
    }
    if arch.hidden.iter().any(|&h| h == 0 || h > 256) {
        return bad("entre 1 y 256 neuronas por capa");
    }
    if !(0.0..=0.8).contains(&arch.dropout) {
        return bad("dropout entre 0 y 0.8");
    }
    if arch.batch_size == 0 {
        return bad("batch mayor que 0");
    }
    if training.epochs == 0 || training.epochs > 500 {
        return bad("entre 1 y 500 épocas");
    }
    if !(1..=90).contains(&training.validation_pct) {
        return bad("validación entre 1 % y 90 %");
    }
    if !(training.learning_rate.is_finite() && training.learning_rate > 0.0) {
        return bad("learning rate positivo");
    }
    if training.lr_factor.is_some_and(|f| !(f > 0.0 && f < 1.0)) {
        return bad("factor de reducción entre 0 y 1");
    }
    if training.patience == Some(0) {
        return bad("paciencia mayor que 0");
    }
    Ok(())
}
