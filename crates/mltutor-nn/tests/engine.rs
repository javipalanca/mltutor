//! Determinism, cancellation, progress and the Predictor of the trained network.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::model::{Prediction, Predictor};
use mltutor_nn::{
    train_network, Activation, EpochLog, FitControl, NnArchitecture, NnError, NnTraining,
    OptimizerKind, TrainOptions,
};

fn arch(output: Activation) -> NnArchitecture {
    NnArchitecture {
        hidden: vec![10, 10],
        activation: Activation::Relu,
        output_activation: output,
        dropout: 0.2,
        batch_size: 16,
        optimizer: OptimizerKind::Adam,
    }
}

fn training(seed: u32) -> NnTraining {
    NnTraining {
        learning_rate: 0.01,
        epochs: 30,
        validation_pct: 20,
        patience: Some(10),
        lr_factor: Some(0.5),
        seed,
    }
}

#[test]
fn same_seed_same_history_other_seed_other_history() {
    let ds = load_builtin(DatasetId::Iris).unwrap();
    let run = |seed| {
        let ds = ds.clone();
        std::thread::spawn(move || {
            train_network(
                ds.features.view(),
                &ds.target,
                &arch(Activation::Softmax),
                &training(seed),
                TrainOptions::default(),
                &FitControl::default(),
            )
            .unwrap()
        })
        .join()
        .unwrap()
    };
    let (a, b, c) = (run(42), run(42), run(7));
    assert_eq!(a.history, b.history);
    assert_eq!(a.network, b.network);
    assert_ne!(a.history, c.history);
}

#[test]
fn progress_every_epoch_and_cancellation() {
    let ds = load_builtin(DatasetId::Iris).unwrap();
    let seen = AtomicUsize::new(0);
    let cancel = AtomicBool::new(false);
    let on_epoch = |log: &EpochLog| {
        seen.fetch_add(1, Ordering::Relaxed);
        if log.epoch == 4 {
            cancel.store(true, Ordering::Relaxed);
        }
    };
    let ctl = FitControl {
        on_epoch: Some(&on_epoch),
        cancel: Some(&cancel),
    };
    let err = train_network(
        ds.features.view(),
        &ds.target,
        &arch(Activation::Softmax),
        &training(42),
        TrainOptions::default(),
        &ctl,
    )
    .unwrap_err();
    assert_eq!(err, NnError::Cancelled);
    assert_eq!(seen.load(Ordering::Relaxed), 5);
}

#[test]
fn untrainable_outputs_are_refused() {
    let ds = load_builtin(DatasetId::Iris).unwrap();
    let err = train_network(
        ds.features.view(),
        &ds.target,
        &arch(Activation::Sigmoid),
        &training(42),
        TrainOptions::default(),
        &FitControl::default(),
    )
    .unwrap_err();
    assert!(matches!(err, NnError::UntrainableOutput { .. }));
}

#[test]
fn predictor_scales_inside() {
    let ds = load_builtin(DatasetId::BreastCancer).unwrap();
    let t = train_network(
        ds.features.view(),
        &ds.target,
        &arch(Activation::Sigmoid),
        &training(42),
        TrainOptions::default(),
        &FitControl::default(),
    )
    .unwrap();
    let net = &t.network;
    assert_eq!(net.classes(), Some(&[0usize, 1][..]));
    let x = ds.features.view();
    let Prediction::Classes(p) = net.predict(x).unwrap() else {
        panic!()
    };
    let proba = net.predict_proba(x).unwrap().unwrap();
    for (i, c) in p.iter().enumerate() {
        assert_eq!(*c, usize::from(proba[[i, 1]] > 0.5));
        assert!((proba[[i, 0]] + proba[[i, 1]] - 1.0).abs() < 1e-12);
    }
    // ~95 % on Cancer with LR 0.01.
    let hits = p
        .iter()
        .zip(ds.target.class_codes().unwrap())
        .filter(|(a, b)| **a == *b)
        .count();
    assert!(hits as f64 / p.len() as f64 > 0.9);
    assert!(net.refit(x, &ds.target, &Default::default()).is_err());
}

/// Review F5a (high, finding 1) and K1: California with SGD, LR 0.3 and [64, 64]
/// diverges in the first epoch. Training goes on (as Keras), the history marks the
/// epoch, survives JSON with its NaN, and the network refuses to predict.
#[test]
fn divergence_is_marked_and_the_network_is_invalid() {
    use mltutor_core::model::ModelError;
    let ds = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    let arch = NnArchitecture {
        hidden: vec![64, 64],
        activation: Activation::Relu,
        output_activation: Activation::Linear,
        dropout: 0.2,
        batch_size: 32,
        optimizer: OptimizerKind::Sgd,
    };
    let training = NnTraining {
        learning_rate: 0.3,
        epochs: 3,
        validation_pct: 20,
        patience: None,
        lr_factor: None,
        seed: 42,
    };
    let t = train_network(
        ds.features.view(),
        &ds.target,
        &arch,
        &training,
        TrainOptions::default(),
        &FitControl::default(),
    )
    .unwrap();
    assert_eq!(t.history.epochs.len(), 3, "training goes on, as in Keras");
    assert_eq!(t.history.diverged_epoch, Some(0));
    assert!(t.history.epochs.iter().all(|e| e.loss.is_nan()));
    assert!(!t.network.is_valid());
    let x = ds.features.view();
    assert!(matches!(
        t.network.predict(x),
        Err(ModelError::Numerical { .. })
    ));
    assert!(matches!(
        t.network.hidden_activations(x, 5),
        Err(ModelError::Numerical { .. })
    ));
    // The weights view still works (NaN left out of the bins).
    let _ = t.network.weights_view(50, 20);
    let text = serde_json::to_string(&t.history).unwrap();
    assert!(text.contains("\"nan\""), "{text}");
    let back: mltutor_core::nn_data::History = serde_json::from_str(&text).unwrap();
    assert_eq!(back.diverged_epoch, Some(0));
    assert!(back.epochs[2].val_loss.is_nan());
    // The same network with a small learning rate does not diverge.
    let ok = train_network(
        ds.features.view(),
        &ds.target,
        &arch,
        &NnTraining {
            learning_rate: 0.001,
            ..training
        },
        TrainOptions::default(),
        &FitControl::default(),
    )
    .unwrap();
    assert_eq!(ok.history.diverged_epoch, None);
    assert!(ok.network.is_valid());
}

/// Keras 3.12 with NaN weights from the start (`tools`-free check, run by hand in
/// Keras 3.12.0: 40 rows, validation_split 0.25, batch 8, SGD 0.1,
/// ReduceLROnPlateau(factor 0.5, patience 1), EarlyStopping(patience 2,
/// restore_best_weights)) gives `learning_rate [0.1, 0.1, 0.05]` and stops after
/// 3 epochs: both callbacks start from `best = None`, so the NaN of epoch 0 is the
/// "best" (review F5a, finding 7 does not apply to Keras 3).
#[test]
fn callbacks_with_nan_from_the_start_follow_keras_3() {
    use mltutor_core::nn_data::{LossKind, MetricKind};
    use mltutor_core::sampling::RandomState;
    use mltutor_nn::{fit, Dense, FitData, FitSpec, Mlp};
    let x: Vec<f32> = (0..120)
        .map(|i| ((i * 37) % 17) as f32 / 10.0 - 0.8)
        .collect();
    let y: Vec<f32> = x.chunks(3).map(|r| r.iter().sum()).collect();
    let mut first = Dense {
        inputs: 3,
        units: 4,
        kernel: vec![f32::NAN; 12],
        bias: vec![0.0; 4],
    };
    first.kernel[0] = f32::NAN;
    let mlp = Mlp::new(
        vec![
            first,
            Dense {
                inputs: 4,
                units: 1,
                kernel: vec![0.5; 4],
                bias: vec![0.0],
            },
        ],
        Activation::Relu,
        Activation::Linear,
    )
    .unwrap();
    let spec = FitSpec {
        loss: LossKind::Mse,
        metric: MetricKind::Mae,
        optimizer: OptimizerKind::Sgd,
        learning_rate: 0.1,
        batch_size: 8,
        epochs: 5,
        validation_split: 0.25,
        patience: Some(2),
        lr_factor: Some(0.5),
        reduce_lr_patience: 1,
        dropout: 0.0,
        shuffle: false,
    };
    let r = fit(
        mlp,
        FitData {
            x: &x,
            y: &y,
            n: 40,
        },
        &spec,
        &mut RandomState::new(42),
        &FitControl::default(),
    )
    .unwrap();
    let lr: Vec<f64> = r
        .history
        .epochs
        .iter()
        .map(|e| e.learning_rate.unwrap())
        .collect();
    assert_eq!(lr.len(), 3);
    assert!((lr[0] - 0.1).abs() < 1e-7 && (lr[1] - 0.1).abs() < 1e-7);
    assert!((lr[2] - 0.05).abs() < 1e-7, "{lr:?}");
    assert_eq!(r.history.stopped_epoch, Some(2));
    assert_eq!(r.history.diverged_epoch, Some(0));
}

/// Review F5a (low, finding 10): cancellation also reaches the validation of the
/// last epoch and the final `evaluate`.
#[test]
fn cancellation_after_the_last_batch_is_honoured() {
    let ds = load_builtin(DatasetId::Iris).unwrap();
    let cancel = AtomicBool::new(false);
    let on_epoch = |log: &EpochLog| {
        if log.epoch == 4 {
            cancel.store(true, Ordering::Relaxed);
        }
    };
    let ctl = FitControl {
        on_epoch: Some(&on_epoch),
        cancel: Some(&cancel),
    };
    let err = train_network(
        ds.features.view(),
        &ds.target,
        &arch(Activation::Softmax),
        &NnTraining {
            epochs: 5,
            ..training(42)
        },
        TrainOptions::default(),
        &ctl,
    )
    .unwrap_err();
    assert_eq!(err, NnError::Cancelled);
    // `evaluate` alone.
    let t = train_network(
        ds.features.view(),
        &ds.target,
        &arch(Activation::Softmax),
        &training(42),
        TrainOptions::default(),
        &FitControl::default(),
    )
    .unwrap();
    let stop = AtomicBool::new(true);
    let mlp = t.network.mlp();
    let x = t.x_test_scaled(ds.features.view());
    let y = vec![0.0_f32; t.split.test.len() * 3];
    let r = mltutor_nn::evaluate(
        mlp,
        &x,
        &y,
        t.split.test.len(),
        mltutor_core::nn_data::LossKind::CategoricalCrossentropy,
        mltutor_core::nn_data::MetricKind::CategoricalAccuracy,
        32,
        &FitControl {
            on_epoch: None,
            cancel: Some(&stop),
        },
    );
    assert_eq!(r, Err(NnError::Cancelled));
    // K4: the core's TrainControl converts both ways.
    let core = mltutor_core::model::TrainControl::none().with_cancel(&stop);
    assert!(FitControl::from(&core).is_cancelled());
    let back = mltutor_core::model::TrainControl::from(&FitControl::from(&core));
    assert!(back.is_cancelled());
}

/// Review F5a (high, finding 2): the training does not depend on the number of
/// threads (the large products are parallel).
#[test]
fn training_is_the_same_with_any_number_of_threads() {
    let ds = load_builtin(DatasetId::Diabetes).unwrap();
    let arch = NnArchitecture {
        hidden: vec![128, 128],
        activation: Activation::Relu,
        output_activation: Activation::Linear,
        dropout: 0.2,
        batch_size: 64,
        optimizer: OptimizerKind::Adam,
    };
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            train_network(
                ds.features.view(),
                &ds.target,
                &arch,
                &NnTraining {
                    epochs: 5,
                    ..training(42)
                },
                TrainOptions::default(),
                &FitControl::default(),
            )
            .unwrap()
        })
    };
    let one = run(1);
    for t in [2, 5] {
        let other = run(t);
        assert_eq!(other.history, one.history, "{t} threads");
        assert_eq!(other.network, one.network, "{t} threads");
    }
}

/// K3, K2, finding 8 and finding 12: activations of the first rows in original
/// units, binned views, the tanh warning flag and a validated deserialisation.
#[test]
fn views_accessors_and_serialisation() {
    use mltutor_core::model::ModelError;
    use mltutor_core::nn_stats::ActivationParams;
    let ds = load_builtin(DatasetId::BreastCancer).unwrap();
    let t = train_network(
        ds.features.view(),
        &ds.target,
        &arch(Activation::Sigmoid),
        &training(42),
        TrainOptions::default(),
        &FitControl::default(),
    )
    .unwrap();
    let net = &t.network;
    assert!(net.output_is_probability());
    assert_eq!(net.class_codes(), &[0, 1]);
    let x_test = t.x_test(ds.features.view());
    // More samples than rows: all the rows.
    let a = net.hidden_activations(x_test.view(), 10_000).unwrap();
    assert_eq!(a.rows_used, x_test.nrows());
    assert_eq!(a.layers.len(), 2);
    assert_eq!(a.layers[0].len(), x_test.nrows() * 10);
    let a = net.hidden_activations(x_test.view(), 7).unwrap();
    assert_eq!(a.rows_used, 7);
    // Wrong width: an error, not a panic.
    let narrow = x_test.slice(ndarray::s![.., ..3]);
    assert!(matches!(
        net.hidden_activations(narrow, 5),
        Err(ModelError::FeatureMismatch { .. })
    ));
    let v = t
        .test_activations(ds.features.view(), &ActivationParams::default())
        .unwrap();
    assert_eq!(v.rows_used, 50);
    assert_eq!(v.layers[0].histogram.counts.iter().sum::<u32>(), 500);
    let direct = net
        .activations_view(x_test.view(), &ActivationParams::default())
        .unwrap();
    assert_eq!(direct, v);
    let w = net.weights_view(50, 20);
    assert_eq!(w.histograms.len(), 3);
    assert_eq!(
        w.histograms[0].weights.counts.iter().sum::<u32>() as usize,
        30 * 10
    );
    // The surface (D78) with the network's scaler.
    let y_test = ds.target.select(&t.split.test);
    let g = net
        .mean_grid(x_test.view(), &y_test, 0, 1, &Default::default())
        .unwrap();
    assert_eq!(g.fixed.len(), 30);
    // Round trip, and a corrupted network is refused.
    let text = serde_json::to_string(net).unwrap();
    let back: mltutor_nn::NeuralNetwork = serde_json::from_str(&text).unwrap();
    assert_eq!(&back, net);
    let mut doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    doc["classes"] = serde_json::json!([0, 1, 2]);
    assert!(serde_json::from_value::<mltutor_nn::NeuralNetwork>(doc).is_err());
    // tanh output: not a probability.
    let tanh = train_network(
        ds.features.view(),
        &ds.target,
        &arch(Activation::Tanh),
        &NnTraining {
            epochs: 3,
            ..training(42)
        },
        TrainOptions::default(),
        &FitControl::default(),
    )
    .unwrap();
    assert!(!tanh.network.output_is_probability());
}
