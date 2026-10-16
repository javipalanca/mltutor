//! Level (b) of D69: the real training (Glorot, dropout and shuffling with our
//! RandomState, seeds 0–4) against the distribution of 5 Keras runs
//! (`tests/parity/nn/stat/*.json`). The mean of the 5 Rust seeds must fall inside
//! `acceptance` and each seed inside `[min − 2m, max + 2m]`. The two CSV cases are
//! informative (a few dozen rows: the bands say little).
//!
//! 55 trainings in about 2 s (California is most of it): it runs with the normal
//! tests. Report: `cargo test -p mltutor-nn --test parity_level_b -- --nocapture`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use std::time::Instant;

use common::*;
use mltutor_core::dataset::TaskType;
use mltutor_core::model::{Prediction, Predictor};
use mltutor_core::Target;
use mltutor_nn::{train_network, FitControl, NnArchitecture, NnTraining, OptimizerKind};
use ndarray::Axis;
use serde_json::Value;

const CASES: &[&str] = &[
    "breast_cancer_default",
    "california_housing_default",
    "csv_numeric_regression_default",
    "csv_string_target_default",
    "diabetes_adam_lr01",
    "diabetes_default",
    "iris_default",
    "iris_rmsprop",
    "iris_sgd_lr001",
    "moons_default",
    "wine_default",
];

fn metrics(case: &Value, seed: u32) -> (Vec<(String, f64)>, usize) {
    let input = &case["input"];
    let ds = dataset(input);
    let cfg = &input["nn_config"];
    let sizes: Vec<u16> = cfg["architecture"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u16)
        .collect();
    let arch = NnArchitecture {
        hidden: sizes[1..sizes.len() - 1].to_vec(),
        activation: activation(&cfg["activation"]),
        output_activation: activation(&cfg["output_activation"]),
        dropout: cfg["dropout_rate"].as_f64().unwrap() as f32,
        batch_size: cfg["batch_size"].as_u64().unwrap() as u16,
        optimizer: OptimizerKind::parse(cfg["optimizer"].as_str().unwrap()).unwrap(),
    };
    let training = NnTraining {
        learning_rate: input["lr"].as_f64().unwrap() as f32,
        epochs: input["epochs"].as_u64().unwrap() as u16,
        validation_pct: (input["validation_split"].as_f64().unwrap() * 100.0).round() as u8,
        patience: Some(input["patience"].as_u64().unwrap() as u8),
        lr_factor: Some(input["lr_factor"].as_f64().unwrap() as f32),
        seed,
    };
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
    let x_test = ds.features.select(Axis(0), &t.split.test);
    let y_test = ds.target.select(&t.split.test);
    let mut out = vec![("test_loss".to_owned(), t.test_loss)];
    match (t.network.predict(x_test.view()).unwrap(), &y_test) {
        (Prediction::Classes(p), Target::Classes { codes, .. }) => {
            let hits = p
                .iter()
                .zip(codes)
                .filter(|(a, b)| **a == **b as usize)
                .count();
            out.push(("accuracy".into(), hits as f64 / p.len() as f64));
        }
        (Prediction::Values(p), Target::Values { values }) => {
            let n = p.len() as f64;
            let mse = p
                .iter()
                .zip(values)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                / n;
            let mae = p
                .iter()
                .zip(values)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / n;
            let mean = values.iter().sum::<f64>() / n;
            let tss = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
            out.push(("mse".into(), mse));
            out.push(("rmse".into(), mse.sqrt()));
            out.push(("mae".into(), mae));
            out.push(("r2".into(), 1.0 - mse * n / tss));
        }
        _ => unreachable!(),
    }
    let _ = TaskType::Classification;
    (out, t.history.epochs.len())
}

#[test]
fn rust_seeds_fall_in_the_keras_bands() {
    let mut failures = Vec::new();
    let total = Instant::now();
    for name in CASES {
        let case = fixture(&format!("nn/stat/{name}.json"));
        let e = &case["expected"];
        let start = Instant::now();
        let runs: Vec<_> = (0..5).map(|s| metrics(&case, s)).collect();
        let secs = start.elapsed().as_secs_f64();
        let informative = name.starts_with("csv_");
        let mut line = format!(
            "{name} ({secs:.2} s, epochs {:?}):",
            runs.iter().map(|r| r.1).collect::<Vec<_>>()
        );
        for (k, _) in &runs[0].0 {
            let vals: Vec<f64> = runs
                .iter()
                .map(|r| r.0.iter().find(|(n, _)| n == k).unwrap().1)
                .collect();
            let mean = vals.iter().sum::<f64>() / 5.0;
            let acc = &e["acceptance"][k];
            let band = &e["bands"][k];
            let (lo, hi) = (acc["lo"].as_f64().unwrap(), acc["hi"].as_f64().unwrap());
            let m = band["min"].as_f64().unwrap() - lo;
            let (lo2, hi2) = (
                band["min"].as_f64().unwrap() - 2.0 * m,
                band["max"].as_f64().unwrap() + 2.0 * m,
            );
            let ok_mean = (lo..=hi).contains(&mean);
            let ok_each = vals.iter().all(|v| (lo2..=hi2).contains(v));
            line += &format!(
                " {k} mean {mean:.4} in [{lo:.4}, {hi:.4}] (Keras {:.4}–{:.4}){}",
                band["min"].as_f64().unwrap(),
                band["max"].as_f64().unwrap(),
                if ok_mean && ok_each { "" } else { " ✗" }
            );
            if !(ok_mean && ok_each) && !informative {
                failures.push(format!("{name} {k}: {vals:?}"));
            }
        }
        let ep = &e["acceptance"]["epochs_trained"];
        let mean_ep = runs.iter().map(|r| r.1 as f64).sum::<f64>() / 5.0;
        if !(ep["lo"].as_f64().unwrap()..=ep["hi"].as_f64().unwrap()).contains(&mean_ep)
            && !informative
        {
            failures.push(format!("{name} epochs {mean_ep}"));
        }
        eprintln!("{line}");
    }
    eprintln!("total {:.1} s", total.elapsed().as_secs_f64());
    assert!(failures.is_empty(), "{failures:#?}");
}
