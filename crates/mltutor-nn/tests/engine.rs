//! Determinism, cancellation, progress and the Predictor of the trained network.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::model::{Prediction, Predictor};
use mltutor_nn::{
    train_network, Activation, EpochLog, FitControl, NnArchitecture, NnError, NnTraining,
    OptimizerKind,
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
                true,
                None,
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
        true,
        None,
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
        true,
        None,
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
        true,
        None,
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
