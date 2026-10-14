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
            // 650 parameters: L-BFGS (above NEWTON_MAX_PARAMS), runs to max_iter;
            // outside the California budget, reported only (progress events).
            println!(
                "{:<40} {t:>10.2?} (not asserted)",
                format!("{label}: train (logistic, L-BFGS)")
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
