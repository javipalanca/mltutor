//! fase4-knn.md C17.7 / §9.1: KNN budgets in release. Run with:
//! `cargo test -p mltutor-core --release --test bench_knn -- --ignored --nocapture`
//!
//! * California (14 448 train / 6 192 test, 8 features): fit + predicting the test
//!   set < 200 ms (scikit-learn: 11 ms to predict); the fit alone < 10 ms.
//! * The default surface of Diabetes (age/sex, all 442 rows, the most tie-sensitive
//!   grid) and California's grids (33 200 cells over 20 640 points) < 60 ms.
//! * California's permutation importance (8 × 10 predictions of the test) < 1 s
//!   (scikit-learn: 0.9 s).
//! * The class distances of Digits (1 257 train rows, 64 features) and the playground.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::grid::{decision_grid, knn_step, GridOptions};
use mltutor_core::importance::{self, PermutationOptions};
use mltutor_core::knn::{self, KnnConfig, KnnMetric, KnnWeights, PlaygroundRequest};
use mltutor_core::model::{Predictor, TrainControl};
use mltutor_core::split;
use ndarray::Axis;

fn median<F: FnMut()>(mut f: F) -> Duration {
    f();
    let mut runs: Vec<Duration> = (0..5)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    runs.sort();
    runs[runs.len() / 2]
}

fn report(name: &str, d: Duration, budget_ms: u64) {
    println!("{name:<56} {d:>10.2?}  (budget {budget_ms} ms)");
    if !cfg!(debug_assertions) {
        assert!(d < Duration::from_millis(budget_ms), "{name}: {d:?}");
    }
}

#[test]
#[ignore = "benchmark: run in release with --ignored"]
fn knn_budgets() {
    let ctl = TrainControl::none();
    let ds = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    let s = split::train_test_split(ds.n_rows(), 0.3, 42).unwrap();
    let x_train = ds.features.select(Axis(0), &s.train);
    let x_test = ds.features.select(Axis(0), &s.test);
    let (y_train, y_test) = (ds.target.select(&s.train), ds.target.select(&s.test));
    for (k, weights, metric) in [
        (5, KnnWeights::Uniform, KnnMetric::Minkowski),
        (8, KnnWeights::Distance, KnnMetric::Euclidean),
        (5, KnnWeights::Uniform, KnnMetric::Manhattan),
    ] {
        let cfg = KnnConfig {
            n_neighbors: k,
            weights,
            metric,
        };
        let label = format!("California K={k} {} {}", weights.as_str(), metric.as_str());
        report(
            &format!("{label}: fit"),
            median(|| {
                knn::fit(cfg, x_train.view(), &y_train, &ctl).unwrap();
            }),
            10,
        );
        report(
            &format!("{label}: fit + predict test"),
            median(|| {
                let m = knn::fit(cfg, x_train.view(), &y_train, &ctl).unwrap();
                m.predict(x_test.view()).unwrap();
            }),
            200,
        );
        let m = knn::fit(cfg, x_train.view(), &y_train, &ctl).unwrap();
        let t = Instant::now();
        importance::permutation(
            &m,
            x_test.view(),
            &y_test,
            PermutationOptions::default(),
            &ctl,
        )
        .unwrap();
        report(&format!("{label}: permutation"), t.elapsed(), 1000);
        let opts = GridOptions {
            step: knn_step(ds.n_rows()),
        };
        report(
            &format!("{label}: grid lon/lat (all rows)"),
            median(|| {
                decision_grid(&m, ds.features.view(), &ds.target, 6, 7, opts, &ctl).unwrap();
            }),
            60,
        );
        report(
            &format!("{label}: grid MedInc/HouseAge (all rows)"),
            median(|| {
                decision_grid(&m, ds.features.view(), &ds.target, 0, 1, opts, &ctl).unwrap();
            }),
            60,
        );
    }

    let ds = load_builtin(DatasetId::Diabetes).unwrap();
    let s = split::train_test_split(ds.n_rows(), 0.3, 42).unwrap();
    let m = knn::fit(
        KnnConfig::default(),
        ds.features.select(Axis(0), &s.train).view(),
        &ds.target.select(&s.train),
        &ctl,
    )
    .unwrap();
    let opts = GridOptions {
        step: knn_step(ds.n_rows()),
    };
    report(
        "Diabetes K=5: default surface age/sex (all rows)",
        median(|| {
            decision_grid(&m, ds.features.view(), &ds.target, 0, 1, opts, &ctl).unwrap();
        }),
        60,
    );

    let ds = load_builtin(DatasetId::Digits).unwrap();
    let s = split::train_test_split(ds.n_rows(), 0.3, 42).unwrap();
    let x = ds.features.select(Axis(0), &s.train);
    let codes = ds.target.select(&s.train).class_codes().unwrap();
    report(
        "Digits: class distances (1257 rows, 64 features)",
        median(|| {
            knn::class_distances(x.view(), &codes, KnnMetric::Minkowski, &ctl).unwrap();
        }),
        100,
    );

    let ds = load_builtin(DatasetId::Titanic).unwrap();
    let req = PlaygroundRequest {
        features: [1, 4],
        config: KnnConfig::default(),
        points: (0..20)
            .map(|i| [f64::from(i) * 4.0, f64::from(i % 5) * 20.0])
            .collect(),
    };
    report(
        "Titanic: playground with 20 points (891 rows)",
        median(|| {
            knn::playground(ds.features.view(), &ds.target, &req, &ctl).unwrap();
        }),
        10,
    );
}
