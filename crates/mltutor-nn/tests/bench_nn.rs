//! Training time budgets (fase5-redes.md C21.11): Iris, 100 epochs, < 100 ms;
//! California with the default network (100 epochs max, no early stop) < 10 s.
//! `cargo test -p mltutor-nn --release --test bench_nn -- --ignored --nocapture`
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_nn::{
    train_network, Activation, FitControl, NnArchitecture, NnTraining, OptimizerKind, TrainOptions,
};

/// Multiplier for the time budgets (`MLTUTOR_BUDGET_FACTOR`, default 1): the budgets are
/// calibrated on an M-series Mac, and CI runners with 4 vCPUs are slower.
fn budget_factor() -> f64 {
    std::env::var("MLTUTOR_BUDGET_FACTOR")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|f| f.is_finite() && *f >= 1.0)
        .unwrap_or(1.0)
}

fn time(id: DatasetId, output: Activation, epochs: u16, patience: Option<u8>) -> (f64, usize) {
    let ds = load_builtin(id).unwrap();
    let arch = NnArchitecture {
        hidden: vec![10, 10],
        activation: Activation::Relu,
        output_activation: output,
        dropout: 0.2,
        batch_size: 64,
        optimizer: OptimizerKind::Adam,
    };
    let training = NnTraining {
        learning_rate: 0.001,
        epochs,
        validation_pct: 20,
        patience,
        lr_factor: Some(0.5),
        seed: 42,
    };
    let best = (0..3)
        .map(|_| {
            let t0 = Instant::now();
            let t = train_network(
                ds.features.view(),
                &ds.target,
                &arch,
                &training,
                TrainOptions::default(),
                &FitControl::default(),
            )
            .unwrap();
            (t0.elapsed().as_secs_f64(), t.history.epochs.len())
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    best
}

#[test]
#[ignore = "benchmark: run with --release --ignored --nocapture"]
fn training_budgets() {
    let (iris, e1) = time(DatasetId::Iris, Activation::Softmax, 100, None);
    eprintln!("Iris, 100 epochs: {:.1} ms ({e1} epochs)", iris * 1e3);
    let (cal, e2) = time(DatasetId::CaliforniaHousing, Activation::Linear, 100, None);
    eprintln!(
        "California, 100 epochs (no early stopping): {:.2} s ({e2} epochs)",
        cal
    );
    assert!(iris < 0.1 * budget_factor(), "Iris {iris}");
    assert!(cal < 10.0 * budget_factor(), "California {cal}");
}

/// Review F5a (finding 2) and K6: California with 5 × 256 neurons (before the
/// parallel products: 3.18 s per epoch with batch 32 on an M-series Mac, 12 cores;
/// the reviewer measured 4.54 s), and the throughput (`work = params × n_fit` per
/// epoch, per second) of several sizes, which calibrates
/// `mltutor_core::nn_data::WORK_PER_SECOND` and the `CostLevel` thresholds.
#[test]
#[ignore = "benchmark: run with --release --ignored --nocapture"]
fn california_large_networks_and_cost_model() {
    use mltutor_core::nn_data::{total_parameters, training_cost};
    let ds = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    for (hidden, batch, epochs) in [
        (vec![256_u16; 5], 32_u16, 2_u16),
        (vec![256; 5], 64, 2),
        (vec![128; 3], 32, 3),
        (vec![64, 64], 32, 10),
        (vec![10, 10], 32, 20),
    ] {
        let arch = NnArchitecture {
            hidden: hidden.clone(),
            activation: Activation::Relu,
            output_activation: Activation::Linear,
            dropout: 0.2,
            batch_size: batch,
            optimizer: OptimizerKind::Adam,
        };
        let training = NnTraining {
            learning_rate: 0.001,
            epochs,
            validation_pct: 20,
            patience: None,
            lr_factor: Some(0.5),
            seed: 42,
        };
        let t0 = Instant::now();
        let t = train_network(
            ds.features.view(),
            &ds.target,
            &arch,
            &training,
            TrainOptions::default(),
            &FitControl::default(),
        )
        .unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let mut layers = vec![8_usize];
        layers.extend(hidden.iter().map(|&h| usize::from(h)));
        layers.push(1);
        let cost = training_cost(total_parameters(&layers), t.sizes.train as usize, epochs);
        eprintln!(
            "{hidden:?} batch {batch}: {:.3} s/epoch, work/s {:.2e}, estimated {:.2} s for {epochs} epochs vs {secs:.2} s ({:?})",
            secs / f64::from(epochs),
            cost.work / secs,
            cost.estimated_seconds,
            cost.level
        );
        if hidden == vec![256; 5] && batch == 32 {
            assert!(
                secs / f64::from(epochs) < 2.0 * budget_factor(),
                "5x256: {secs} s"
            );
        }
    }
}
