//! Criterion benchmarks of the ONNX export of every model family (PLAN F7,
//! docs/migration/benchmarks.md): building the graph with [`onnx_model`] and
//! serialising it ([`mltutor_export::onnx::Model::to_bytes`]), which is what
//! `export_model` does before writing the file.
//!
//! `cargo bench -p mltutor-export --bench onnx` (about 30 s).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::knn::{self, KnnConfig};
use mltutor_core::linear::{self, ModelConfig};
use mltutor_core::model::{Predictor, TrainControl};
use mltutor_core::split;
use mltutor_core::tree::{self, TreeConfig, TreeCriterion, TreeTask};
use mltutor_export::onnx_model;
use mltutor_nn::{
    train_network, Activation, FitControl, NnArchitecture, NnTraining, OptimizerKind, TrainOptions,
};
use ndarray::Axis;

/// A model trained on the app's split (`test_size = 0.3`, seed 42).
fn train(
    id: DatasetId,
    fit: impl Fn(ndarray::ArrayView2<'_, f64>, &mltutor_core::dataset::Target) -> Box<dyn Predictor>,
) -> Box<dyn Predictor> {
    let ds = load_builtin(id).unwrap();
    let s = split::train_test_split(ds.n_rows(), 0.3, 42).unwrap();
    let x = ds.features.select(Axis(0), &s.train);
    fit(x.view(), &ds.target.select(&s.train))
}

fn network(id: DatasetId, hidden: Vec<u16>, output: Activation) -> Box<dyn Predictor> {
    let ds = load_builtin(id).unwrap();
    let arch = NnArchitecture {
        hidden,
        activation: Activation::Relu,
        output_activation: output,
        dropout: 0.2,
        batch_size: 64,
        optimizer: OptimizerKind::Adam,
    };
    let training = NnTraining {
        learning_rate: 0.001,
        epochs: 1,
        validation_pct: 20,
        patience: None,
        lr_factor: Some(0.5),
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
    Box::new(t.network)
}

fn bench_onnx(c: &mut Criterion) {
    let ctl = TrainControl::none();
    let models: Vec<(&str, Box<dyn Predictor>)> = vec![
        (
            "linear_california",
            train(DatasetId::CaliforniaHousing, |x, y| {
                linear::fit(ModelConfig::Linear, x, y, &ctl).unwrap()
            }),
        ),
        (
            "logistic_digits",
            train(DatasetId::Digits, |x, y| {
                linear::fit(ModelConfig::Logistic { max_iter: 1000 }, x, y, &ctl).unwrap()
            }),
        ),
        (
            "tree_iris_d3",
            train(DatasetId::Iris, |x, y| {
                let cfg = TreeConfig::new(TreeTask::Classification, TreeCriterion::Gini, 3, 2);
                Box::new(tree::fit(cfg, x, y, &ctl).unwrap())
            }),
        ),
        (
            "tree_california_d10",
            train(DatasetId::CaliforniaHousing, |x, y| {
                let cfg = TreeConfig::new(TreeTask::Regression, TreeCriterion::SquaredError, 10, 2);
                Box::new(tree::fit(cfg, x, y, &ctl).unwrap())
            }),
        ),
        (
            "knn_iris",
            train(DatasetId::Iris, |x, y| {
                Box::new(knn::fit(KnnConfig::default(), x, y, &ctl).unwrap())
            }),
        ),
        (
            "knn_california",
            train(DatasetId::CaliforniaHousing, |x, y| {
                Box::new(knn::fit(KnnConfig::default(), x, y, &ctl).unwrap())
            }),
        ),
        (
            "mlp_iris_10x10",
            network(DatasetId::Iris, vec![10, 10], Activation::Softmax),
        ),
        (
            "mlp_california_5x256",
            network(
                DatasetId::CaliforniaHousing,
                vec![256; 5],
                Activation::Linear,
            ),
        ),
    ];
    let mut g = c.benchmark_group("onnx");
    for (name, model) in &models {
        g.bench_function(*name, |b| {
            b.iter(|| onnx_model(model.as_ref(), None).unwrap().to_bytes());
        });
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_onnx
}
criterion_main!(benches);
