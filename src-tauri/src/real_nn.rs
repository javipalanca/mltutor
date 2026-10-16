//! 🧠 training of [`crate::real_facade::RealFacade`] over `mltutor-nn` (fase5-redes.md §8):
//! the controls are checked by [`crate::nn_model::prepare`] (same messages as the
//! FixtureFacade), the engine reports every epoch through the observer and checks the
//! cancellation before every batch.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use mltutor_core as core;
use mltutor_nn::{FitControl, NnError};

use crate::core_facade::{DatasetView, ModelHandle, TrainEvent, TrainObserver, TrainSpec};
use crate::ipc::{CommandError, EpochLog, ModelConfig, TaskType, TrainPhase};
use crate::nn::finite_log;
use crate::nn_model::{prepare, NnModel, Outcome};

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
        true,
        None,
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
        (summary.task == TaskType::Classification).then(|| trained.network.classes.clone());
    let model = NnModel::new(
        &summary,
        &prepared,
        Outcome {
            epochs,
            best_epoch,
            started,
        },
        classes,
    )
    .with_engine(Arc::new(trained));
    Ok(Arc::new(model))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;

    use super::*;
    use crate::core_facade::CoreFacade;
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
