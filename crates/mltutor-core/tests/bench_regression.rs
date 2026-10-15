//! fase2-regresion.md §5.4 / §9.1-13: on California (14 448 train / 6 192 test rows)
//! the training and every view of the 📊 Regresión lesson must take < 100 ms each
//! in release. Logistic regression is timed on Breast Cancer and Digits. Run with:
//! `cargo test -p mltutor-core --release --test bench_regression -- --ignored --nocapture`
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::grid::{decision_grid, GridOptions};
use mltutor_core::linear::{self, ModelConfig};
use mltutor_core::model::{Prediction, TrainControl};
use mltutor_core::{
    coefficients, curves, form, interpretation, metrics, probabilities, residuals, split,
};
use ndarray::Axis;

fn median<F: FnMut()>(mut f: F) -> Duration {
    f();
    let mut runs: Vec<Duration> = (0..7)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    runs.sort();
    runs[runs.len() / 2]
}

fn report(name: &str, d: Duration) {
    println!("{name:<40} {d:>10.2?}");
    if !cfg!(debug_assertions) {
        assert!(d < Duration::from_millis(100), "{name}: {d:?}");
    }
}

#[test]
#[ignore = "benchmark: run in release with --ignored"]
fn regression_lesson_under_100_ms_per_command() {
    let ds = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    let s = split::train_test_split(ds.n_rows(), 0.3, 42).unwrap();
    let x_train = ds.features.select(Axis(0), &s.train);
    let x_test = ds.features.select(Axis(0), &s.test);
    let y_train = ds.target.select(&s.train);
    let y_test = ds.target.select(&s.test).numeric();
    let ctl = TrainControl::none();
    report(
        "California: split + train (OLS)",
        median(|| {
            let s = split::train_test_split(ds.n_rows(), 0.3, 42).unwrap();
            let xt = ds.features.select(Axis(0), &s.train);
            linear::fit(
                ModelConfig::Linear,
                xt.view(),
                &ds.target.select(&s.train),
                &ctl,
            )
            .unwrap();
        }),
    );
    let model = linear::fit(ModelConfig::Linear, x_train.view(), &y_train, &ctl).unwrap();
    report(
        "California: evaluate_model",
        median(|| {
            let Prediction::Values(p) = model.predict(x_test.view()).unwrap() else {
                panic!()
            };
            let ev = metrics::regression(&y_test, &p);
            interpretation::regression(&ev).unwrap();
        }),
    );
    let Prediction::Values(pred) = model.predict(x_test.view()).unwrap() else {
        panic!()
    };
    report(
        "California: residuals_view",
        median(|| {
            residuals::analysis(&y_test, &pred);
        }),
    );
    report(
        "California: decision_grid (surface 200x200)",
        median(|| {
            decision_grid(
                model.as_ref(),
                x_train.view(),
                &y_train,
                0,
                1,
                GridOptions::default(),
                &ctl,
            )
            .unwrap();
        }),
    );
    report(
        "California: coefficients_view",
        median(|| {
            coefficients::view(model.as_ref(), &ds.feature_display, None).unwrap();
        }),
    );
    report(
        "California: prediction_form + predict_one",
        median(|| {
            let f = form::build(x_train.view(), &ds, None);
            form::predict_one(model.as_ref(), &f.default_values()).unwrap();
        }),
    );

    for (id, label) in [
        (DatasetId::BreastCancer, "Cancer"),
        (DatasetId::Digits, "Digits"),
        (DatasetId::Iris, "Iris"),
    ] {
        let ds = load_builtin(id).unwrap();
        let s = split::train_test_split(ds.n_rows(), 0.3, 42).unwrap();
        let xt = ds.features.select(Axis(0), &s.train);
        let xs = ds.features.select(Axis(0), &s.test);
        let yt = ds.target.select(&s.train);
        let ys = ds.target.select(&s.test).class_codes().unwrap();
        let cfg = ModelConfig::Logistic { max_iter: 1000 };
        let m = linear::fit(cfg, xt.view(), &yt, &ctl).unwrap();
        println!("    fit report: {:?}", m.fit_report());
        let t = median(|| {
            linear::fit(cfg, xt.view(), &yt, &ctl).unwrap();
        });
        if label == "Digits" {
            // 650 parameters: Newton (~45 ms per Hessian, 12 iterations) since the
            // F2 review (~0.59 s); before, L-BFGS ran to max_iter (482-540 ms, 1000
            // iterations, 2.8e-3 from the optimum). Outside the California
            // budget, reported only (progress events).
            println!(
                "{:<40} {t:>10.2?} (not asserted)",
                format!("{label}: train (logistic, Newton)")
            );
        } else {
            report(&format!("{label}: train (logistic, Newton)"), t);
        }
        let m = linear::fit(cfg, xt.view(), &yt, &ctl).unwrap();
        println!("    fit report: {:?}", m.fit_report());
        let proba = m.predict_proba(xs.view()).unwrap().unwrap();
        let classes = m.classes().unwrap().to_vec();
        report(
            &format!("{label}: roc_curves + probability_view"),
            median(|| {
                if classes.len() == 2 {
                    curves::binary_curves(&ys, &proba.column(1).to_vec(), classes[1]);
                } else {
                    curves::multiclass_curves(&ys, proba.view(), &classes);
                }
                probabilities::distribution(&ys, proba.view(), &classes);
            }),
        );
        report(
            &format!("{label}: decision_grid (boundary)"),
            median(|| {
                decision_grid(
                    m.as_ref(),
                    xt.view(),
                    &yt,
                    0,
                    1,
                    GridOptions::default(),
                    &ctl,
                )
                .unwrap();
            }),
        );
    }
}

/// splitmix64 in [0, 1): reproducible data without a RNG crate.
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1_u64 << 53) as f64
}

/// Review F2 (OLS cost): large CSVs. Before the QR rewrite (2026-10-14, full SVD
/// of the n × p matrix + linfa QR): 100 000 × 200 took 36.3 s and 1 000 000 × 20
/// 3.23 s; after (Householder QR + SVD of R): 1.17 s and 0.32 s. Only reported.
/// `cargo test -p mltutor-core --release --test bench_regression ols_large -- --ignored --nocapture`
#[test]
#[ignore = "benchmark: run in release with --ignored"]
fn ols_large_design_matrices() {
    for (n, p) in [(100_000_usize, 200_usize), (1_000_000, 20)] {
        let mut s = 42_u64;
        let x = ndarray::Array2::from_shape_fn((n, p), |_| uniform(&mut s) * 2.0 - 1.0);
        let y = ndarray::Array1::from_shape_fn(n, |i| {
            x.row(i)
                .iter()
                .enumerate()
                .map(|(j, v)| v * (j as f64 + 1.0))
                .sum::<f64>()
                + uniform(&mut s)
        });
        let t = Instant::now();
        let m =
            mltutor_core::linear::LinearRegression::fit(x.view(), y.view(), &TrainControl::none())
                .unwrap();
        let d = t.elapsed();
        println!("OLS {n} x {p}: {d:>10.2?} (rank {})", m.rank());
        assert_eq!(m.rank(), p);
        assert!((m.coef()[p - 1] - p as f64).abs() < 1e-2);
    }
}

/// Review F2 (pending): 1 500 columns. With the SVD of R always: about 11 s.
/// Full rank (3 000 × 1 500) takes the back-substitution path; 1 500 × 1 500 is
/// rank deficient after centring (rank ≤ n − 1) and needs the SVD.
/// `cargo test -p mltutor-core --release --test bench_regression ols_wide -- --ignored --nocapture`
#[test]
#[ignore = "benchmark: run in release with --ignored"]
fn ols_wide_design_matrices() {
    for (n, p) in [(3_000_usize, 1_500_usize), (1_500, 1_500)] {
        let mut s = 7_u64;
        let x = ndarray::Array2::from_shape_fn((n, p), |_| uniform(&mut s) * 2.0 - 1.0);
        let y = ndarray::Array1::from_shape_fn(n, |i| x[[i, 0]] * 3.0 + uniform(&mut s));
        let t = Instant::now();
        let m =
            mltutor_core::linear::LinearRegression::fit(x.view(), y.view(), &TrainControl::none())
                .unwrap();
        println!("OLS {n} x {p}: {:>10.2?} (rank {})", t.elapsed(), m.rank());
        assert_eq!(m.rank(), p.min(n - 1));
        let pred = m.predict_values(x.view()).unwrap();
        if n > p {
            let r2 = mltutor_core::metrics::r2_score(&y.to_vec(), &pred);
            assert!(r2 > 0.9, "{r2}");
        }
    }
}
