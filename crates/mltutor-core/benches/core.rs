//! Criterion benchmarks of `mltutor-core` (PLAN F7, docs/migration/benchmarks.md).
//!
//! `cargo bench -p mltutor-core --bench core` (about 2 min; filter with a name,
//! e.g. `cargo bench -p mltutor-core --bench core -- knn/`). Everything is
//! deterministic: embedded datasets, the app's split (`test_size = 0.3`,
//! `random_state = 42`) and synthetic data from a fixed splitmix64 stream.
//!
//! The `#[ignore]` timing tests in `tests/bench_*.rs` keep the per-command budgets
//! (assertions); these benches are for measuring and comparing (`--save-baseline`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mltutor_core::csv::read_csv;
use mltutor_core::dataset::{load_builtin, Dataset, DatasetId, Target};
use mltutor_core::explore;
use mltutor_core::grid::{decision_grid, knn_step, GridOptions};
use mltutor_core::knn::{self, KnnConfig};
use mltutor_core::linear::{self, LinearRegression, ModelConfig};
use mltutor_core::model::{Predictor, TrainControl};
use mltutor_core::split;
use mltutor_core::tree::{self, TreeConfig, TreeCriterion, TreeTask};
use ndarray::{Array1, Array2, Axis};

/// splitmix64 in [0, 1).
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1_u64 << 53) as f64
}

/// The training rows of the app's split (`test_size = 0.3`, seed 42).
struct Split {
    x_train: Array2<f64>,
    y_train: Target,
    x_test: Array2<f64>,
}

fn split_of(ds: &Dataset) -> Split {
    let s = split::train_test_split(ds.n_rows(), 0.3, 42).unwrap();
    Split {
        x_train: ds.features.select(Axis(0), &s.train),
        y_train: ds.target.select(&s.train),
        x_test: ds.features.select(Axis(0), &s.test),
    }
}

/// A CSV of `rows` rows: 8 float columns, an integer and a 5-level text column
/// (the target), about 100 bytes per row.
fn synthetic_csv(rows: usize) -> Vec<u8> {
    use std::fmt::Write;
    let mut s = String::with_capacity(rows * 110);
    s.push_str("a,b,c,d,e,f,g,h,count,label\n");
    let mut st = 42_u64;
    let labels = ["rojo", "verde", "azul", "amarillo", "negro"];
    for _ in 0..rows {
        for _ in 0..8 {
            let _ = write!(s, "{:.6},", uniform(&mut st) * 200.0 - 100.0);
        }
        let k = (uniform(&mut st) * 5.0) as usize;
        let _ = writeln!(s, "{},{}", (uniform(&mut st) * 1000.0) as u32, labels[k]);
    }
    s.into_bytes()
}

fn bench_csv(c: &mut Criterion) {
    let mut g = c.benchmark_group("csv");
    g.sample_size(10);
    for rows in [100_000_usize, 500_000] {
        let bytes = synthetic_csv(rows);
        g.throughput(Throughput::Bytes(bytes.len() as u64));
        g.bench_with_input(BenchmarkId::new("read", rows), &bytes, |b, bytes| {
            b.iter(|| read_csv(black_box(bytes)).unwrap());
        });
    }
    g.finish();
}

fn bench_explore(c: &mut Criterion) {
    let ds = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    let mut g = c.benchmark_group("explore/california");
    g.bench_function("load", |b| {
        b.iter(|| load_builtin(black_box(DatasetId::CaliforniaHousing)).unwrap());
    });
    // The embedded CSV is deflated in the binary: inflating it is part of `load`.
    g.bench_function("inflate_csv", |b| {
        b.iter(|| black_box(DatasetId::CaliforniaHousing).csv_bytes().unwrap());
    });
    g.bench_function("describe", |b| b.iter(|| explore::describe(black_box(&ds))));
    g.bench_function("correlation", |b| {
        b.iter(|| explore::correlation_matrix(black_box(&ds)));
    });
    g.finish();
}

fn bench_linear(c: &mut Criterion) {
    let ctl = TrainControl::none();
    let mut g = c.benchmark_group("linear");
    for id in [DatasetId::Iris, DatasetId::BreastCancer, DatasetId::Digits] {
        let ds = load_builtin(id).unwrap();
        let s = split_of(&ds);
        let name = format!("{id:?}");
        g.bench_function(BenchmarkId::new("ols", &name), |b| {
            b.iter(|| {
                linear::fit(ModelConfig::Linear, s.x_train.view(), &s.y_train, &ctl).unwrap()
            });
        });
        let logistic = ModelConfig::Logistic { max_iter: 1000 };
        g.bench_function(BenchmarkId::new("logistic", &name), |b| {
            b.iter(|| linear::fit(logistic, s.x_train.view(), &s.y_train, &ctl).unwrap());
        });
    }
    let ds = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    let s = split_of(&ds);
    g.bench_function(BenchmarkId::new("ols", "CaliforniaHousing"), |b| {
        b.iter(|| linear::fit(ModelConfig::Linear, s.x_train.view(), &s.y_train, &ctl).unwrap());
    });
    g.finish();

    // The rank-deficient SVD path (`linear/lstsq.rs`): 1 500 × 1 500, rank 1 499.
    let mut g = c.benchmark_group("linear/ols_wide");
    g.sample_size(10);
    for (n, p) in [(1_500_usize, 300_usize), (300, 300), (1_500, 1_500)] {
        let mut st = 7_u64;
        let x = Array2::from_shape_fn((n, p), |_| uniform(&mut st) * 2.0 - 1.0);
        let y = Array1::from_shape_fn(n, |i| x[[i, 0]] * 3.0 + uniform(&mut st));
        g.bench_function(BenchmarkId::from_parameter(format!("{n}x{p}")), |b| {
            b.iter(|| LinearRegression::fit(x.view(), y.view(), &ctl).unwrap());
        });
    }
    g.finish();
}

fn bench_trees(c: &mut Criterion) {
    let ctl = TrainControl::none();
    let mut g = c.benchmark_group("tree");
    let cal = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    let s = split_of(&cal);
    let config = TreeConfig::new(TreeTask::Regression, TreeCriterion::SquaredError, 10, 2);
    g.bench_function("fit/california_d10_squared_error", |b| {
        b.iter(|| tree::fit(config, s.x_train.view(), &s.y_train, &ctl).unwrap());
    });
    let t = tree::fit(config, s.x_train.view(), &s.y_train, &ctl).unwrap();
    g.bench_function("predict/california_d10_test", |b| {
        b.iter(|| t.predict(s.x_test.view()).unwrap());
    });
    let iris = load_builtin(DatasetId::Iris).unwrap();
    let s = split_of(&iris);
    for (criterion, depth) in [(TreeCriterion::Gini, 3), (TreeCriterion::Entropy, 10)] {
        let config = TreeConfig::new(TreeTask::Classification, criterion, depth, 2);
        g.bench_function(format!("fit/iris_d{depth}_{}", criterion.as_str()), |b| {
            b.iter(|| tree::fit(config, s.x_train.view(), &s.y_train, &ctl).unwrap());
        });
    }
    g.finish();
}

fn bench_knn(c: &mut Criterion) {
    let ctl = TrainControl::none();
    let mut g = c.benchmark_group("knn");
    g.sample_size(20);
    for (id, fx, fy) in [
        (DatasetId::CaliforniaHousing, 6, 7), // longitude/latitude, kd_tree
        (DatasetId::Digits, 20, 28),          // two pixels, brute force (64 features)
    ] {
        let ds = load_builtin(id).unwrap();
        let s = split_of(&ds);
        let name = format!("{id:?}");
        let cfg = KnnConfig::default();
        g.bench_function(BenchmarkId::new("fit", &name), |b| {
            b.iter(|| knn::fit(cfg, s.x_train.view(), &s.y_train, &ctl).unwrap());
        });
        let m = knn::fit(cfg, s.x_train.view(), &s.y_train, &ctl).unwrap();
        g.bench_function(BenchmarkId::new("predict_test", &name), |b| {
            b.iter(|| m.predict(s.x_test.view()).unwrap());
        });
        let opts = GridOptions {
            step: knn_step(ds.n_rows()),
        };
        g.bench_function(BenchmarkId::new("decision_grid", &name), |b| {
            b.iter(|| {
                decision_grid(&m, ds.features.view(), &ds.target, fx, fy, opts, &ctl).unwrap()
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
    targets = bench_csv, bench_explore, bench_linear, bench_trees, bench_knn
}
criterion_main!(benches);
