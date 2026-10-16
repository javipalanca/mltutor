//! Training time budgets (fase5-redes.md C21.11): Iris, 100 epochs, < 100 ms;
//! California with the default network (100 epochs max, no early stop) < 10 s.
//! `cargo test -p mltutor-nn --release --test bench_nn -- --ignored --nocapture`
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_nn::{
    train_network, Activation, FitControl, NnArchitecture, NnTraining, OptimizerKind,
};

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
                true,
                None,
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
    assert!(iris < 0.1, "Iris {iris}");
    assert!(cal < 10.0, "California {cal}");
}
