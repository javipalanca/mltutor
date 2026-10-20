//! Criterion benchmarks of the 🧠 network (PLAN F7, docs/migration/benchmarks.md).
//!
//! `cargo bench -p mltutor-nn --bench nn` (about 1 min). The trainings use the
//! lesson's pipeline (`train_network`: scaler, stratified split, Glorot with seed
//! 42, dropout masks), so they are deterministic and bit-for-bit repeatable.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::model::TrainControl;
use mltutor_nn::{
    train_network, Activation, FitControl, NnArchitecture, NnTraining, OptimizerKind, TrainOptions,
};

fn arch(hidden: Vec<u16>, output: Activation, batch_size: u16) -> NnArchitecture {
    NnArchitecture {
        hidden,
        activation: Activation::Relu,
        output_activation: output,
        dropout: 0.2,
        batch_size,
        optimizer: OptimizerKind::Adam,
    }
}

/// The lesson's defaults with `epochs` and no early stopping (every run trains
/// exactly `epochs` epochs).
fn training(epochs: u16) -> NnTraining {
    NnTraining {
        learning_rate: 0.001,
        epochs,
        validation_pct: 20,
        patience: None,
        lr_factor: Some(0.5),
        seed: 42,
    }
}

fn bench_training(c: &mut Criterion) {
    let mut g = c.benchmark_group("nn/train");
    let iris = load_builtin(DatasetId::Iris).unwrap();
    let a = arch(vec![10, 10], Activation::Softmax, 64);
    let t = training(100);
    g.bench_function("iris_10x10_100_epochs", |b| {
        b.iter(|| {
            train_network(
                iris.features.view(),
                &iris.target,
                &a,
                &t,
                TrainOptions::default(),
                &FitControl::default(),
            )
            .unwrap()
        });
    });
    let cal = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    let a = arch(vec![10, 10], Activation::Linear, 64);
    let t = training(10);
    g.bench_function("california_10x10_10_epochs", |b| {
        b.iter(|| {
            train_network(
                cal.features.view(),
                &cal.target,
                &a,
                &t,
                TrainOptions::default(),
                &FitControl::default(),
            )
            .unwrap()
        });
    });
    g.finish();

    // The largest network the lesson allows: 5 × 256 (≈ 266 000 weights), batch 32.
    let mut g = c.benchmark_group("nn/train_large");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(10));
    let a = arch(vec![256; 5], Activation::Linear, 32);
    let t = training(1);
    g.bench_function("california_5x256_1_epoch", |b| {
        b.iter(|| {
            train_network(
                cal.features.view(),
                &cal.target,
                &a,
                &t,
                TrainOptions::default(),
                &FitControl::default(),
            )
            .unwrap()
        });
    });
    g.finish();
}

fn bench_predict(c: &mut Criterion) {
    let cal = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    let ctl = TrainControl::none();
    let mut g = c.benchmark_group("nn/predict_batch");
    g.throughput(Throughput::Elements(cal.n_rows() as u64));
    for (name, hidden) in [("10x10", vec![10_u16, 10]), ("5x256", vec![256; 5])] {
        let a = arch(hidden, Activation::Linear, 64);
        let trained = train_network(
            cal.features.view(),
            &cal.target,
            &a,
            &training(1),
            TrainOptions::default(),
            &FitControl::default(),
        )
        .unwrap();
        g.bench_function(format!("california_all_rows_{name}"), |b| {
            b.iter(|| {
                trained
                    .network
                    .predict_batch(black_box(cal.features.view()), &ctl)
                    .unwrap()
            });
        });
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = bench_training, bench_predict
}
criterion_main!(benches);
