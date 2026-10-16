//! 🧠 training of [`crate::real_facade::RealFacade`] over `mltutor-nn` (fase5-redes.md §8):
//! the controls are checked by [`crate::nn_model::prepare`] (same messages as the
//! FixtureFacade), the engine reports every epoch through the observer and checks the
//! cancellation before every batch.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use mltutor_core as core;
use mltutor_nn::{FitControl, NnError};

use mltutor_nn::TrainedNetwork;

use crate::core_facade::{
    DatasetView, ModelHandle, ModelView, TrainEvent, TrainObserver, TrainSpec,
};
use crate::ipc::{
    ActivationsRequest, ActivationsView, CommandError, DecisionGrid, EpochLog,
    LayerActivationsView, LayerWeightsView, MapLayer, ModelConfig, NetworkMapView, TaskType,
    TrainPhase, WeightsView,
};
use crate::nn::finite_log;
use crate::nn_model::{keras_name, layer_name, prepare, NnEngine, NnModel, Outcome};
use crate::real_models::CoreModel;

/// Core errors of the lesson → Spanish command errors (the texts of `content/redes`).
pub fn nn_error(error: NnError) -> CommandError {
    match error {
        NnError::Cancelled => CommandError::cancelled(),
        NnError::UntrainableOutput { key, n_classes, .. } => {
            let message = crate::nn::nn_text("arquitectura", &key, &error_text(&key))
                .replace("{clases}", &n_classes.to_string())
                .replace(
                    "{recomendada}",
                    if n_classes > 2 { "softmax" } else { "sigmoid" },
                );
            CommandError::UntrainableOutput {
                key,
                n_classes: Some(crate::ipc::to_u32(n_classes)),
                message,
            }
        }
        NnError::TooFewPerClass { class_name, .. } => crate::nn::too_few_per_class(&class_name),
        NnError::InvalidConfig { .. } => CommandError::invalid_config(error.to_string()),
        NnError::Model(model) => crate::real_models::model_error(model),
        other => CommandError::training(other),
    }
}

fn error_text(key: &str) -> String {
    format!("🚫 Esta activación de salida no se puede entrenar ({key}).")
}

/// A core epoch (0-based, f64 with NaN) → the IPC log (1-based, `None` for NaN).
pub fn epoch_log(log: &core::nn_data::EpochLog) -> EpochLog {
    EpochLog {
        epoch: u16::try_from(log.epoch + 1).unwrap_or(u16::MAX),
        loss: finite_log(log.loss),
        metric: finite_log(log.metric),
        val_loss: finite_log(log.val_loss),
        val_metric: finite_log(log.val_metric),
        learning_rate: log.learning_rate.and_then(finite_log),
    }
}

fn arch(arch: &crate::ipc::NnArchitecture) -> core::nn_data::NnArchitecture {
    use crate::ipc::{NnActivation as A, NnOptimizer as O};
    use core::nn_data::{Activation, OptimizerKind};
    let act = |a: A| match a {
        A::Relu => Activation::Relu,
        A::Tanh => Activation::Tanh,
        A::Sigmoid => Activation::Sigmoid,
        A::Softmax => Activation::Softmax,
        A::Linear => Activation::Linear,
    };
    core::nn_data::NnArchitecture {
        hidden: arch.hidden.clone(),
        activation: act(arch.activation),
        output_activation: act(arch.output_activation),
        dropout: arch.dropout,
        batch_size: arch.batch_size,
        optimizer: match arch.optimizer {
            O::Adam => OptimizerKind::Adam,
            O::Sgd => OptimizerKind::Sgd,
            O::Rmsprop => OptimizerKind::Rmsprop,
        },
    }
}

fn training(t: &crate::ipc::NnTraining) -> core::nn_data::NnTraining {
    core::nn_data::NnTraining {
        learning_rate: t.learning_rate,
        epochs: t.epochs,
        validation_pct: t.validation_pct,
        patience: t.patience,
        lr_factor: t.lr_factor,
        seed: t.seed,
    }
}

pub fn train(
    view: &dyn DatasetView,
    dataset: &Arc<core::Dataset>,
    id: Option<core::DatasetId>,
    spec: &TrainSpec,
    observer: &TrainObserver<'_>,
    cancel: &AtomicBool,
) -> Result<ModelHandle, CommandError> {
    let started = Instant::now();
    let summary = view.summary();
    let counts = (summary.task == TaskType::Classification)
        .then(|| crate::nn_model::class_counts(view))
        .flatten();
    let prepared = prepare(&summary, spec, counts.as_deref())?;
    observer(TrainEvent::Phase(TrainPhase::Preparing));
    let missing = crate::real_models::missing_columns(dataset);
    if !missing.is_empty() {
        return Err(crate::models::missing_values_error(&missing));
    }
    let ModelConfig::NeuralNet {
        arch: ipc_arch,
        training: ipc_training,
    } = &spec.config
    else {
        return Err(CommandError::not_nn());
    };
    let core_arch = arch(ipc_arch);
    let core_training = training(ipc_training);
    observer(TrainEvent::Phase(TrainPhase::Building));
    observer(TrainEvent::Phase(TrainPhase::Compiling));
    observer(TrainEvent::Phase(TrainPhase::Training));
    let on_epoch = |log: &core::nn_data::EpochLog| observer(TrainEvent::Epoch(&epoch_log(log)));
    let control = FitControl {
        on_epoch: Some(&on_epoch),
        cancel: Some(cancel),
    };
    let trained = mltutor_nn::train_network(
        dataset.features.view(),
        &dataset.target,
        &core_arch,
        &core_training,
        mltutor_nn::TrainOptions::default(),
        &control,
    )
    .map_err(nn_error)?;
    crate::core_facade::check_cancelled(cancel)?;
    let epochs: Vec<EpochLog> = trained.history.epochs.iter().map(epoch_log).collect();
    let best_epoch = trained
        .history
        .best_epoch
        .and_then(|b| u16::try_from(b + 1).ok());
    let classes =
        (summary.task == TaskType::Classification).then(|| trained.network.class_codes().to_vec());
    let model = NnModel::new(
        &summary,
        &prepared,
        Outcome {
            epochs,
            best_epoch,
            diverged_epoch: trained
                .history
                .diverged_epoch
                .and_then(|e| u16::try_from(e + 1).ok()),
            started,
        },
        classes,
    )
    .with_engine(Arc::new(RealNn::new(
        dataset.clone(),
        id,
        spec.clone(),
        trained,
    )));
    Ok(Arc::new(model))
}

/// The trained network and, around the same predictor and split, a [`CoreModel`] for
/// the views every lesson shares (C1/C2, C9).
pub struct RealNn {
    trained: TrainedNetwork,
    core: CoreModel,
}

impl RealNn {
    pub fn new(
        dataset: Arc<core::Dataset>,
        id: Option<core::DatasetId>,
        spec: TrainSpec,
        trained: TrainedNetwork,
    ) -> Self {
        let core = CoreModel::from_split(
            dataset,
            id,
            spec,
            Box::new(trained.network.clone()),
            trained.split.clone(),
        );
        Self { trained, core }
    }

    fn x_test(&self) -> ndarray::Array2<f64> {
        self.core.x_test()
    }
}

fn hist(h: core::stats::Histogram) -> crate::ipc::Histogram {
    crate::ipc::Histogram {
        edges: h.edges,
        counts: h.counts,
    }
}

fn indices(values: &[usize]) -> Vec<u32> {
    values.iter().map(|&v| crate::ipc::to_u32(v)).collect()
}

impl NnEngine for RealNn {
    fn delegate(&self) -> &dyn ModelView {
        &self.core
    }

    fn test_scores(&self) -> (f64, f64) {
        (self.trained.test_loss, self.trained.test_metric)
    }

    fn output_is_probability(&self) -> bool {
        self.trained.network.output_is_probability()
    }

    fn weights(
        &self,
        bins_weights: usize,
        bins_biases: usize,
    ) -> Result<WeightsView, CommandError> {
        use crate::ipc::{WeightsInsight as I, WeightsVerdict as V};
        use core::nn_stats::{WeightsInsight, WeightsVerdict};
        // K2: statistics and Plotly's `nbinsx` bins, both computed in the core.
        let view = self.trained.network.weights_view(bins_weights, bins_biases);
        let stats = view.stats;
        let n = stats.layers.len();
        let layers = stats
            .layers
            .iter()
            .zip(view.histograms)
            .enumerate()
            .map(|(i, (s, h))| LayerWeightsView {
                name: layer_name(i, n),
                keras_name: keras_name(i, n),
                inputs: crate::ipc::to_u32(s.shape[0]),
                units: crate::ipc::to_u32(s.shape[1]),
                mean_w: s.mean_w,
                std_w: s.std_w,
                near_zero_pct: s.near_zero_pct,
                near_zero_pct_raw: s.near_zero_pct_raw,
                mean_b: s.mean_b,
                std_b: s.std_b,
                n_params: crate::ipc::to_u32(s.n_params),
                min_w: s.min_w,
                max_w: s.max_w,
                min_b: s.min_b,
                max_b: s.max_b,
                weights_hist: hist(h.weights),
                biases_hist: hist(h.biases),
            })
            .collect();
        Ok(WeightsView {
            layers,
            global_mean: stats.global_mean,
            global_std: stats.global_std,
            global_near_zero_pct: stats.global_near_zero_pct,
            verdict: match stats.verdict {
                WeightsVerdict::TooSmall => V::TooSmall,
                WeightsVerdict::TooLarge => V::TooLarge,
                WeightsVerdict::Healthy => V::Healthy,
            },
            insights: stats
                .insights
                .iter()
                .map(|i| match i {
                    WeightsInsight::ManyNearZero => I::ManyNearZero,
                    WeightsInsight::LowVariance => I::LowVariance,
                    WeightsInsight::HighVariance => I::HighVariance,
                    WeightsInsight::Balanced => I::Balanced,
                })
                .collect(),
        })
    }

    fn activations(
        &self,
        req: &ActivationsRequest,
        cancel: &AtomicBool,
    ) -> Result<ActivationsView, CommandError> {
        crate::core_facade::check_cancelled(cancel)?;
        let params = core::nn_stats::ActivationParams {
            sample_size: usize::from(req.sample_size),
            dead_threshold: req.dead_threshold,
            constant_std: req.constant_std,
            bins: core::nn_stats::ACTIVATION_BINS,
        };
        // K3: the first `sample_size` test rows (only those are copied), in Rust.
        let view = self
            .trained
            .test_activations(self.core.dataset.features.view(), &params)
            .map_err(crate::real_models::model_error)?;
        crate::core_facade::check_cancelled(cancel)?;
        let n_dense = view.map.neurons.len();
        let to32 = |v: f64| v as f32;
        Ok(ActivationsView {
            rows_used: crate::ipc::to_u32(view.rows_used),
            dead_pct: view.dead_pct,
            layers: view
                .layers
                .into_iter()
                .enumerate()
                .map(|(i, l)| LayerActivationsView {
                    name: layer_name(i, n_dense),
                    keras_name: keras_name(i, n_dense),
                    units: crate::ipc::to_u32(l.stats.units),
                    dead: indices(&l.stats.dead),
                    constant: indices(&l.stats.constant),
                    saturation_pct: l.stats.saturation_pct,
                    mean_act: l.stats.mean_act,
                    std_act: l.stats.std_act,
                    neuron_means: l.stats.neuron_means,
                    hist: hist(l.histogram),
                })
                .collect(),
            map: NetworkMapView {
                layers: view
                    .map
                    .layers
                    .iter()
                    .map(|&n| crate::ipc::to_u32(n))
                    .collect(),
                neurons: view
                    .map
                    .neurons
                    .iter()
                    .map(|layer| MapLayer {
                        bias: layer.iter().map(|n| to32(n.bias)).collect(),
                        mean_w: layer.iter().map(|n| to32(n.mean_w)).collect(),
                        std_w: layer.iter().map(|n| to32(n.std_w)).collect(),
                        max_abs_w: layer.iter().map(|n| to32(n.max_abs_w)).collect(),
                    })
                    .collect(),
            },
        })
    }

    fn mean_grid(
        &self,
        x: usize,
        y: usize,
        cancel: &AtomicBool,
    ) -> Result<DecisionGrid, CommandError> {
        let control = core::model::TrainControl::none().with_cancel(cancel);
        let x_test = self.x_test();
        let grid = self
            .trained
            .network
            .mean_grid(x_test.view(), &self.core.y_test, x, y, &control)
            .map_err(crate::real_models::model_error)?;
        Ok(self.core.grid_dto(grid.grid, x, y, Some(grid.fixed)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;

    use super::*;
    use crate::core_facade::{CoreFacade, ModelHandle};
    use crate::ipc::{DatasetId, NnActivation, NnArchitecture, NnOptimizer, NnTraining};
    use crate::real_facade::RealFacade;

    fn spec(output: NnActivation, epochs: u16, lr: f32) -> TrainSpec {
        TrainSpec {
            config: ModelConfig::NeuralNet {
                arch: NnArchitecture {
                    hidden: vec![10, 10],
                    activation: NnActivation::Relu,
                    output_activation: output,
                    dropout: 0.2,
                    batch_size: 64,
                    optimizer: NnOptimizer::Adam,
                },
                training: NnTraining {
                    learning_rate: lr,
                    epochs,
                    validation_pct: 20,
                    patience: Some(10),
                    lr_factor: Some(0.5),
                    seed: 42,
                },
            },
            test_size: 0.2,
            dataset_label: String::new(),
        }
    }

    /// The real engine reports every epoch live, 1-based, with the D73 sizes, and the
    /// same seed gives the same history (D70).
    #[test]
    fn real_networks_train_and_report_every_epoch() {
        for (id, output, n) in [
            (DatasetId::Iris, NnActivation::Softmax, (96, 24, 30)),
            (
                DatasetId::BreastCancer,
                NnActivation::Sigmoid,
                (364, 91, 114),
            ),
            (DatasetId::Diabetes, NnActivation::Linear, (282, 71, 89)),
        ] {
            let dataset = RealFacade.load_builtin(id).unwrap();
            let seen = Mutex::new(Vec::new());
            let observer = |event: TrainEvent<'_>| {
                if let TrainEvent::Epoch(log) = event {
                    seen.lock().unwrap().push(log.clone());
                }
            };
            let model = dataset
                .train_observed(
                    &spec(output, 30, 0.01),
                    &|_, _| {},
                    &observer,
                    &AtomicBool::new(false),
                )
                .unwrap();
            let summary = model.summary();
            let nn = summary.nn.unwrap();
            assert_eq!((nn.n_fit, nn.n_val, nn.n_test), n, "{id:?}");
            let history = model.nn_history().unwrap();
            let seen = seen.into_inner().unwrap();
            assert_eq!(
                seen, history.epochs,
                "{id:?}: the live epochs are the history"
            );
            assert_eq!(seen.first().map(|l| l.epoch), Some(1));
            assert!(
                seen.iter()
                    .all(|l| l.loss.is_some() && l.val_loss.is_some()),
                "{id:?}"
            );
            let again = dataset
                .train_observed(
                    &spec(output, 30, 0.01),
                    &|_, _| {},
                    &|_| {},
                    &AtomicBool::new(false),
                )
                .unwrap();
            assert_eq!(again.nn_history().unwrap(), history, "{id:?}: D70");
        }
    }

    #[test]
    fn real_networks_cancel_between_batches_and_block_untrainable_outputs() {
        let dataset = RealFacade
            .load_builtin(DatasetId::CaliforniaHousing)
            .unwrap();
        let cancel = AtomicBool::new(false);
        let last = Mutex::new(0u16);
        let observer = |event: TrainEvent<'_>| {
            if let TrainEvent::Epoch(log) = event {
                *last.lock().unwrap() = log.epoch;
                if log.epoch == 2 {
                    cancel.store(true, Ordering::SeqCst);
                }
            }
        };
        let started = Instant::now();
        let result = dataset.train_observed(
            &spec(NnActivation::Linear, 500, 0.001),
            &|_, _| {},
            &observer,
            &cancel,
        );
        assert!(matches!(result, Err(CommandError::Cancelled { .. })));
        assert_eq!(*last.lock().unwrap(), 2, "stopped in the third epoch");
        assert!(started.elapsed().as_secs() < 30);
        let iris = RealFacade.load_builtin(DatasetId::Iris).unwrap();
        let error = iris
            .train_observed(
                &spec(NnActivation::Sigmoid, 10, 0.01),
                &|_, _| {},
                &|_| {},
                &AtomicBool::new(false),
            )
            .err()
            .unwrap();
        assert!(
            matches!(error, CommandError::UntrainableOutput { .. }),
            "{error:?}"
        );
    }

    fn trained(id: DatasetId, spec: &TrainSpec) -> ModelHandle {
        RealFacade
            .load_builtin(id)
            .unwrap()
            .train_observed(spec, &|_, _| {}, &|_| {}, &AtomicBool::new(false))
            .unwrap()
    }

    /// F5c: every view of 📈 and 🎯 on the real engine, classification and regression.
    #[test]
    fn real_network_views_of_evaluation_and_visualization() {
        let go = AtomicBool::new(false);
        let iris = trained(DatasetId::Iris, &spec(NnActivation::Softmax, 30, 0.01));
        let eval = iris.nn_evaluation().unwrap();
        let Some(crate::ipc::Evaluation::Classification(c)) = &eval.evaluation else {
            panic!("{eval:?}");
        };
        // Keras' CategoricalAccuracy on the test set = the app's accuracy (argmax).
        assert!(
            (c.accuracy - eval.test_metric.unwrap()).abs() < 1e-6,
            "{eval:?}"
        );
        assert!(eval.output_is_probability && eval.r2_negative.is_none());
        let diag = iris.nn_diagnostics().unwrap();
        assert!(diag.blocks.len() > 5 && diag.diverged.is_none());
        let history = iris.nn_history().unwrap();
        assert!(matches!(
            history.cards,
            Some(crate::ipc::NnViewNode::Row { .. })
        ));
        let weights = iris.nn_weights(50, 20).unwrap();
        let names: Vec<_> = weights.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["Oculta 1", "Oculta 2", "Salida"]);
        for l in &weights.layers {
            // K2: Plotly's `nbinsx` = at most 50 "nice" bins.
            assert!((1..=50).contains(&l.weights_hist.counts.len()));
            assert_eq!(l.weights_hist.edges.len(), l.weights_hist.counts.len() + 1);
            let total: u32 = l.weights_hist.counts.iter().sum();
            assert_eq!(total, l.inputs * l.units);
            assert_eq!(l.biases_hist.counts.iter().sum::<u32>(), l.units);
        }
        let req = ActivationsRequest {
            model_id: 0,
            task_id: 0,
            sample_size: 50,
            dead_threshold: 0.0,
            constant_std: 1e-3,
        };
        let acts = iris.nn_activations(&req, &go).unwrap();
        assert_eq!(acts.rows_used, 30, "min(50, n_test)");
        assert_eq!(acts.layers.len(), 2);
        assert_eq!(acts.map.layers, vec![4, 10, 10, 3]);
        assert_eq!(acts.map.neurons.len(), 3);
        assert_eq!(acts.layers[0].hist.counts.iter().sum::<u32>(), 30 * 10);
        let grid = iris.decision_grid(2, 3, &|_, _| {}, &go).unwrap();
        assert_eq!(grid.fixed.as_ref().map(Vec::len), Some(4));
        assert!(grid.x.len() <= 200 && grid.y.len() <= 200);
        assert_eq!(grid.points.rows, 30, "the test points");
        // The static C1 snippets work; F5d code does not yet.
        assert!(iris
            .code(crate::ipc::CodeTemplate::ConfusionMatrix, None)
            .is_ok());

        let diabetes = trained(DatasetId::Diabetes, &spec(NnActivation::Linear, 20, 0.001));
        let eval = diabetes.nn_evaluation().unwrap();
        assert!(matches!(
            eval.evaluation,
            Some(crate::ipc::Evaluation::Regression(_))
        ));
        let notice = eval.r2_negative.expect("D84: R² < 0 with the defaults");
        assert!(notice.contains("R²"), "{notice}");
        let grid = diabetes.decision_grid(0, 2, &|_, _| {}, &go).unwrap();
        assert!(matches!(grid.values, crate::ipc::GridValues::Values { .. }));
    }

    /// K1: a diverged network blocks the views that need its weights.
    #[test]
    fn diverged_networks_block_the_weight_views() {
        let mut s = spec(NnActivation::Linear, 30, 0.3);
        if let ModelConfig::NeuralNet { arch, training } = &mut s.config {
            arch.optimizer = crate::ipc::NnOptimizer::Sgd;
            arch.dropout = 0.0;
            training.patience = None;
            training.lr_factor = None;
        }
        // Diabetes (targets ≈ 150) with tanh and SGD at LR 0.3 explodes in epoch 7.
        if let ModelConfig::NeuralNet { arch, .. } = &mut s.config {
            arch.activation = NnActivation::Tanh;
        }
        let model = trained(DatasetId::Diabetes, &s);
        let history = model.nn_history().unwrap();
        let Some(epoch) = history.diverged_epoch else {
            panic!("SGD with LR 0.3 on Diabetes should diverge: {history:?}");
        };
        assert!(history.cards.is_none());
        let eval = model.nn_evaluation().unwrap();
        assert_eq!(eval.diverged.map(|d| d.epoch), Some(epoch));
        assert!(eval.evaluation.is_none());
        assert!(model.nn_diagnostics().unwrap().blocks.is_empty());
        let error = model.nn_weights(50, 20).err().unwrap();
        assert!(error.message().contains("divergió"), "{error:?}");
        assert!(model
            .decision_grid(0, 1, &|_, _| {}, &AtomicBool::new(false))
            .is_err());
    }

    /// §3 of the task: the views of the largest network (5 × 256) stay small and fast.
    #[test]
    fn big_network_views_are_small_and_fast() {
        let mut s = spec(NnActivation::Softmax, 10, 0.001);
        if let ModelConfig::NeuralNet { arch, .. } = &mut s.config {
            arch.hidden = vec![256; 5];
        }
        let model = trained(DatasetId::Digits, &s);
        let started = Instant::now();
        let weights = model.nn_weights(120, 60).unwrap();
        let t_weights = started.elapsed();
        let req = ActivationsRequest {
            model_id: 0,
            task_id: 0,
            sample_size: 200,
            dead_threshold: 0.0,
            constant_std: 1e-3,
        };
        let started = Instant::now();
        let acts = model.nn_activations(&req, &AtomicBool::new(false)).unwrap();
        let t_acts = started.elapsed();
        let w_bytes = serde_json::to_vec(&weights).unwrap().len();
        let a_bytes = serde_json::to_vec(&acts).unwrap().len();
        eprintln!(
            "5x256 Digits: nn_weights {t_weights:?} ({w_bytes} B), nn_activations {t_acts:?} ({a_bytes} B)"
        );
        assert!(w_bytes < 60_000, "{w_bytes}");
        assert!(a_bytes < 200_000, "{a_bytes}");
        assert_eq!(acts.layers.len(), 5);
    }

    #[test]
    fn engine_errors_have_spanish_messages() {
        let error = nn_error(NnError::UntrainableOutput {
            output: "linear".into(),
            n_classes: 3,
            key: "no_entrenable_linear".into(),
            reason: "sparse_rank".into(),
        });
        assert!(error.message().contains("softmax"), "{error:?}");
        assert!(!error.message().contains('{'));
        assert!(matches!(
            nn_error(NnError::TooFewPerClass {
                class_name: "x".into(),
                count: 1
            }),
            CommandError::TooFewPerClass { .. }
        ));
        assert_eq!(nn_error(NnError::Cancelled), CommandError::cancelled());
    }
}
