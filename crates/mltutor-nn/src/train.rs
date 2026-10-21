//! `model.fit` with the app's callbacks (`tools/nn_reference.py::train`, C21.6–7).
//!
//! Per batch the predictions are computed once with the weights **before** the step;
//! the loss and the metric of the epoch are means weighted by the batch size of
//! those predictions. Validation uses the last rows (`validation_split`), in batches
//! of the same size, with the weights at the end of the epoch. Then the callbacks
//! run in the app's order: EarlyStopping (restores the best weights at the end of
//! training **always**, Keras 3.12) and ReduceLROnPlateau (`min_delta` 1e-4,
//! `min_lr` 1e-7, cooldown 0).
//!
//! Randomness (D70) comes from one NumPy-compatible `RandomState(seed)` stream: the
//! Glorot initialisation (layer by layer, kernel row-major), then per epoch the
//! permutation of the training rows and, per batch and hidden layer, the dropout
//! masks (`random_sample() >= rate`). The same seed gives the same history on any
//! machine and with any number of threads.
//!
//! Divergence (K1): when the loss of a batch or the validation loss is NaN or
//! infinite, [`History::diverged_epoch`] records the first such epoch and training
//! **goes on**, as Keras without `TerminateOnNaN` (the level (a) parity needs the
//! same epochs). EarlyStopping may then restore finite weights; if not, the network
//! is marked invalid ([`crate::NeuralNetwork::is_valid`]).

use std::sync::atomic::{AtomicBool, Ordering};

use mltutor_core::model::TrainControl;
use mltutor_core::nn_data::{
    keras_validation_split, Activation, EpochLog, History, LossKind, MetricKind, NnError,
    OptimizerKind,
};
use mltutor_core::sampling::RandomState;

use crate::loss::{metric_values, per_sample_loss};
use crate::mlp::{Dense, DropoutMasks, Mlp};
use crate::optim::Optimizer;

/// Glorot uniform (Keras' default kernel initializer): `U(−l, l)`, `l =
/// sqrt(6 / (fan_in + fan_out))`, biases at 0. `layers = [input, hidden…, output]`.
pub fn glorot_init(
    layers: &[usize],
    hidden: Activation,
    output: Activation,
    rng: &mut RandomState,
) -> Mlp {
    let dense = layers
        .windows(2)
        .map(|w| {
            let (fan_in, fan_out) = (w[0], w[1]);
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let kernel = (0..fan_in * fan_out)
                .map(|_| (-limit + 2.0 * limit * rng.random_sample()) as f32)
                .collect();
            Dense {
                inputs: fan_in,
                units: fan_out,
                kernel,
                bias: vec![0.0; fan_out],
            }
        })
        .collect();
    Mlp {
        layers: dense,
        hidden,
        output,
    }
}

/// Everything `fit` needs besides the data (the lesson's configuration already
/// resolved, see [`crate::network::train_network`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FitSpec {
    pub loss: LossKind,
    pub metric: MetricKind,
    pub optimizer: OptimizerKind,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
    /// Fraction of the training rows (the last ones) used for validation.
    pub validation_split: f64,
    /// EarlyStopping patience (`None` = off).
    pub patience: Option<usize>,
    /// ReduceLROnPlateau factor (`None` = off).
    pub lr_factor: Option<f64>,
    /// ReduceLROnPlateau patience (`max(1, patience // 2)`, 5 without EarlyStopping).
    pub reduce_lr_patience: usize,
    /// Dropout after every hidden layer (0 = none).
    pub dropout: f32,
    /// Shuffle the training rows every epoch (Keras `shuffle=True`).
    pub shuffle: bool,
}

/// Progress and cancellation (C21.9). `on_epoch` runs on the training thread after
/// the callbacks of every epoch; `cancel` is checked before every batch, every
/// validation batch and every batch of the final `evaluate`.
///
/// K4: it converts to and from the core's [`TrainControl`] (the cancellation flag;
/// the epoch callback has no core equivalent), so the app has one cancellation type.
#[derive(Default, Clone, Copy)]
pub struct FitControl<'a> {
    pub on_epoch: Option<&'a (dyn Fn(&EpochLog) + Sync)>,
    pub cancel: Option<&'a AtomicBool>,
}

impl std::fmt::Debug for FitControl<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FitControl")
            .field("on_epoch", &self.on_epoch.is_some())
            .field("cancel", &self.cancel)
            .finish()
    }
}

impl FitControl<'_> {
    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_some_and(|c| c.load(Ordering::Relaxed))
    }

    fn check(&self) -> Result<(), NnError> {
        if self.is_cancelled() {
            Err(NnError::Cancelled)
        } else {
            Ok(())
        }
    }
}

impl<'a> From<&TrainControl<'a>> for FitControl<'a> {
    fn from(c: &TrainControl<'a>) -> Self {
        FitControl {
            on_epoch: None,
            cancel: c.cancel_flag(),
        }
    }
}

impl<'a> From<&FitControl<'a>> for TrainControl<'a> {
    fn from(c: &FitControl<'a>) -> Self {
        match c.cancel {
            Some(flag) => TrainControl::none().with_cancel(flag),
            None => TrainControl::none(),
        }
    }
}

/// Row-major training data: `x` is `n × n_features`, `y` is `n × n_outputs` (0/1 or
/// one-hot in classification, values in regression).
#[derive(Debug, Clone, Copy)]
pub struct FitData<'a> {
    pub x: &'a [f32],
    pub y: &'a [f32],
    pub n: usize,
}

/// Result of [`fit`]: the final network (the best epoch's weights with
/// EarlyStopping) and the history.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    pub mlp: Mlp,
    pub history: History,
    /// Learning rate at the end (after the last reduction).
    pub final_learning_rate: f32,
}

fn rows(data: &[f32], width: usize, idx: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(idx.len() * width);
    for &i in idx {
        out.extend_from_slice(&data[i * width..(i + 1) * width]);
    }
    out
}

/// `model.evaluate` / validation: batch-size-weighted means of the loss and the
/// metric (`nn_reference.evaluate`). Keras' `evaluate` uses batches of 32. NaN
/// without rows. Cancellation is checked before every batch.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    mlp: &Mlp,
    x: &[f32],
    y: &[f32],
    n: usize,
    loss: LossKind,
    metric: MetricKind,
    batch_size: usize,
    control: &FitControl<'_>,
) -> Result<(f64, f64), NnError> {
    let (p, k) = (mlp.n_inputs(), mlp.n_outputs());
    mlp.validate()?;
    if x.len() != n * p || y.len() != n * k {
        return Err(NnError::InvalidConfig {
            reason: "las dimensiones de los datos no coinciden con la red".to_owned(),
        });
    }
    if n == 0 {
        return Ok((f64::NAN, f64::NAN));
    }
    let (mut tot_l, mut tot_m) = (0.0_f64, 0.0_f64);
    let bs = batch_size.max(1);
    let lsz = mlp.layers.len();
    let mut s = 0;
    while s < n {
        control.check()?;
        let e = (s + bs).min(n);
        let rows = e - s;
        let xb = &x[s * p..e * p];
        let yb = &y[s * k..e * k];
        let f = mlp.forward_cache(xb, rows, None);
        let per = per_sample_loss(loss, mlp.output, &f.z[lsz - 1], &f.a[lsz], yb, k);
        tot_l += f64::from(crate::loss::mean_f32(&per)) * rows as f64;
        let mv = metric_values(metric, &f.a[lsz], yb, k);
        tot_m += f64::from(mv.iter().sum::<f32>());
        s = e;
    }
    Ok((tot_l / n as f64, tot_m / n as f64))
}

/// `model.fit(x, y, epochs, batch_size, validation_split, shuffle)` from the
/// weights `init`, with the RNG `rng` for shuffling and dropout.
pub fn fit(
    init: Mlp,
    data: FitData<'_>,
    spec: &FitSpec,
    rng: &mut RandomState,
    control: &FitControl<'_>,
) -> Result<FitResult, NnError> {
    let mut mlp = init;
    mlp.validate()?;
    let (p, k) = (mlp.n_inputs(), mlp.n_outputs());
    if data.x.len() != data.n * p || data.y.len() != data.n * k {
        return Err(NnError::InvalidConfig {
            reason: "las dimensiones de los datos no coinciden con la red".to_owned(),
        });
    }
    if spec.batch_size == 0 || spec.epochs == 0 {
        return Err(NnError::InvalidConfig {
            reason: "batch y épocas deben ser mayores que 0".to_owned(),
        });
    }
    let split = keras_validation_split(data.n, spec.validation_split);
    if split == 0 || split == data.n {
        return Err(NnError::ValidationTooSmall {
            n_train: data.n,
            pct: (spec.validation_split * 100.0).round() as u8,
        });
    }
    let (xv, yv) = (&data.x[split * p..], &data.y[split * k..]);
    let n_val = data.n - split;
    let mut opt = Optimizer::new(spec.optimizer, spec.learning_rate);
    let n_hidden = mlp.layers.len().saturating_sub(1);
    let units: Vec<usize> = mlp.layers.iter().map(|l| l.units).collect();

    let mut epochs = Vec::with_capacity(spec.epochs);
    // EarlyStopping / ReduceLROnPlateau state (`best = None` until the first epoch).
    let (mut es_best, mut es_wait, mut es_best_epoch, mut stopped) =
        (None::<f64>, 0usize, 0usize, 0usize);
    let mut es_weights: Option<Mlp> = None;
    // Keras 3.12: both callbacks are `MonitorCallback`s that start from `best =
    // baseline = None`, and `_is_improvement(_, None)` is True, so the first epoch
    // always becomes the best, even when its val_loss is NaN (Keras 2's
    // ReduceLROnPlateau started from `np.inf` instead; review F5a finding 7 assumed
    // that and does not apply: `keras/src/callbacks/monitor_callback.py`).
    let (mut rl_best, mut rl_wait) = (None::<f64>, 0usize);
    let mut diverged: Option<usize> = None;
    let mut order: Vec<usize> = (0..split).collect();

    for epoch in 0..spec.epochs {
        if spec.shuffle {
            order = rng.permutation(split);
        }
        let (mut tot_l, mut tot_m) = (0.0_f64, 0.0_f64);
        let mut s = 0;
        while s < split {
            control.check()?;
            let e = (s + spec.batch_size).min(split);
            let n_b = e - s;
            let (xb, yb) = if spec.shuffle {
                (rows(data.x, p, &order[s..e]), rows(data.y, k, &order[s..e]))
            } else {
                (data.x[s * p..e * p].to_vec(), data.y[s * k..e * k].to_vec())
            };
            let masks = (spec.dropout > 0.0).then(|| DropoutMasks {
                rate: spec.dropout,
                masks: (0..n_hidden)
                    .map(|l| {
                        (0..n_b * units[l])
                            .map(|_| {
                                if rng.random_sample() >= f64::from(spec.dropout) {
                                    1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect()
                    })
                    .collect(),
            });
            let (lval, grads, out) = mlp.gradients(&xb, &yb, n_b, spec.loss, masks.as_ref())?;
            if !lval.is_finite() && diverged.is_none() {
                diverged = Some(epoch);
            }
            tot_m += f64::from(metric_values(spec.metric, &out, &yb, k).iter().sum::<f32>());
            opt.step(&mut mlp, &grads)?;
            tot_l += f64::from(lval) * n_b as f64;
            s = e;
        }
        let (vl, vm) = evaluate(
            &mlp,
            xv,
            yv,
            n_val,
            spec.loss,
            spec.metric,
            spec.batch_size,
            control,
        )?;
        if !vl.is_finite() && diverged.is_none() {
            diverged = Some(epoch);
        }
        let log = EpochLog {
            epoch,
            loss: tot_l / split as f64,
            metric: tot_m / split as f64,
            val_loss: vl,
            val_metric: vm,
            learning_rate: spec.lr_factor.map(|_| f64::from(opt.learning_rate)),
        };
        epochs.push(log);

        let mut stop = false;
        if let Some(patience) = spec.patience {
            if es_weights.is_none() {
                es_weights = Some(mlp.clone());
                es_best_epoch = epoch;
            }
            es_wait += 1;
            if es_best.is_none_or(|b| vl < b) {
                es_best = Some(vl);
                es_best_epoch = epoch;
                es_wait = 0;
                es_weights = Some(mlp.clone());
            } else if es_wait >= patience && epoch > 0 {
                stopped = epoch;
                stop = true;
            }
        }
        if let Some(factor) = spec.lr_factor {
            if rl_best.is_none_or(|b| vl + 1e-4 < b) {
                rl_best = Some(vl);
                rl_wait = 0;
            } else {
                rl_wait += 1;
                if rl_wait >= spec.reduce_lr_patience {
                    let old = f64::from(opt.learning_rate);
                    if opt.learning_rate > 1e-7_f32 {
                        opt.learning_rate = (old * factor).max(1e-7) as f32;
                        rl_wait = 0;
                    }
                }
            }
        }
        if let Some(cb) = control.on_epoch {
            cb(&log);
        }
        if stop {
            break;
        }
    }
    let early = spec.patience.is_some();
    if let Some(best) = es_weights.filter(|_| early) {
        mlp = best; // Keras 3.12 restores them in on_train_end, always.
    }
    Ok(FitResult {
        mlp,
        history: History {
            metric: spec.metric,
            best_epoch: early.then_some(es_best_epoch),
            stopped_epoch: early.then_some(stopped),
            near_divergence: History::detect_near_divergence(&epochs, diverged),
            epochs,
            diverged_epoch: diverged,
        },
        final_learning_rate: opt.learning_rate,
    })
}
