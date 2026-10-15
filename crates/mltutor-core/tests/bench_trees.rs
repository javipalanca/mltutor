//! fase3-arboles.md §9.1-8: on California (14 448 train / 6 192 test rows) the
//! tree with max_depth 10 trains in < 50 ms and every view of the 🌲 lesson takes
//! < 100 ms, in release. Run with:
//! `cargo test -p mltutor-core --release --test bench_trees -- --ignored --nocapture`
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use mltutor_core::codegen::{self, CodeTemplate};
use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::grid::{decision_grid, GridOptions};
use mltutor_core::importance::{self, PermutationOptions};
use mltutor_core::model::{Prediction, Predictor, TrainControl};
use mltutor_core::tree::{self, plot, ExportTextOptions, TreeConfig, TreeCriterion, TreeTask};
use mltutor_core::{form, metrics, residuals, split};
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

fn report(name: &str, d: Duration, budget_ms: u64) {
    println!("{name:<48} {d:>10.2?}");
    if !cfg!(debug_assertions) {
        assert!(d < Duration::from_millis(budget_ms), "{name}: {d:?}");
    }
}

#[test]
#[ignore = "benchmark: run in release with --ignored"]
fn trees_lesson_on_california() {
    let ds = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    let s = split::train_test_split(ds.n_rows(), 0.3, 42).unwrap();
    let x_train = ds.features.select(Axis(0), &s.train);
    let x_test = ds.features.select(Axis(0), &s.test);
    let y_train = ds.target.select(&s.train);
    let y_test_t = ds.target.select(&s.test);
    let y_test = y_test_t.numeric();
    let ctl = TrainControl::none();
    let names = &ds.feature_display;
    for criterion in [TreeCriterion::SquaredError, TreeCriterion::FriedmanMse] {
        let config = TreeConfig::new(TreeTask::Regression, criterion, 10, 2);
        report(
            &format!("California d10 {}: train", criterion.as_str()),
            median(|| {
                tree::fit(config, x_train.view(), &y_train, &ctl).unwrap();
            }),
            50,
        );
    }
    let config = TreeConfig::new(TreeTask::Regression, TreeCriterion::SquaredError, 10, 2);
    let t = tree::fit(config, x_train.view(), &y_train, &ctl).unwrap();
    println!("nodes: {}", t.node_count());
    report(
        "evaluate_model",
        median(|| {
            let Prediction::Values(p) = t.predict(x_test.view()).unwrap() else {
                panic!()
            };
            metrics::regression(&y_test, &p);
        }),
        100,
    );
    let Prediction::Values(pred) = t.predict(x_test.view()).unwrap() else {
        panic!()
    };
    report(
        "residuals_view",
        median(|| {
            residuals::analysis(&y_test, &pred);
        }),
        100,
    );
    report(
        "tree_diagram",
        median(|| {
            plot::diagram(&t, names, None);
        }),
        100,
    );
    report(
        "tree_rules (view + export)",
        median(|| {
            tree::export_text(
                &t,
                names,
                &ExportTextOptions {
                    show_weights: true,
                    ..Default::default()
                },
            );
            tree::export_text(&t, names, &ExportTextOptions::default());
        }),
        100,
    );
    report(
        "feature_importance (native)",
        median(|| {
            t.feature_importances();
        }),
        100,
    );
    report(
        "feature_importance (permutation, 8x10)",
        median(|| {
            importance::permutation(
                &t,
                x_test.view(),
                &y_test_t,
                PermutationOptions::default(),
                &ctl,
            )
            .unwrap();
        }),
        100,
    );
    report(
        "decision_grid (surface, refit 2-D)",
        median(|| {
            decision_grid(
                &t,
                x_train.view(),
                &y_train,
                0,
                1,
                GridOptions::default(),
                &ctl,
            )
            .unwrap();
        }),
        100,
    );
    report(
        "prediction_form + predict_one + path",
        median(|| {
            let f = form::build(x_train.view(), &ds, None);
            let v = f.default_values();
            form::predict_one(&t, &v).unwrap();
            t.decision_path(&v).unwrap();
        }),
        100,
    );
    report(
        "generate_code (TreePython, corrected)",
        median(|| {
            let ctx = codegen::tree_python_corrected_context(&t, names, None);
            codegen::render(CodeTemplate::TreePython, &ctx).unwrap();
        }),
        100,
    );
    // MAE is O(n²) per node in scikit-learn 1.7.2 too: only reported.
    let mae = TreeConfig::new(TreeTask::Regression, TreeCriterion::AbsoluteError, 3, 2);
    let t0 = Instant::now();
    tree::fit(mae, x_train.view(), &y_train, &ctl).unwrap();
    println!(
        "{:<48} {:>10.2?}",
        "California d3 absolute_error: train (no budget)",
        t0.elapsed()
    );
    let _ = t.n_features();
}

/// Permutation importance on a wide table (a CSV with many columns): it used to
/// be O(p·R·n·p) because every repetition revalidated and converted the whole
/// matrix to float32; the tree now predicts on one float32 buffer.
#[test]
#[ignore = "benchmark: run in release with --ignored"]
fn permutation_importance_on_a_wide_table() {
    let (n, p) = (3_000, 300);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let x = ndarray::Array2::from_shape_fn((n, p), |_| next());
    let codes: Vec<usize> = (0..n)
        .map(|i| usize::from(x[[i, 0]] + x[[i, 1]] > 1.0))
        .collect();
    let config = TreeConfig::new(TreeTask::Classification, TreeCriterion::Gini, 6, 2);
    let t = tree::DecisionTree::fit_classifier(config, x.view(), &codes, &TrainControl::none())
        .unwrap();
    let y = mltutor_core::Target::Classes {
        codes: codes.iter().map(|&c| c as u32).collect(),
        class_names: vec!["0".into(), "1".into()],
        class_values: vec![0.0, 1.0],
    };
    let d = median(|| {
        importance::permutation(
            &t,
            x.view(),
            &y,
            PermutationOptions::default(),
            &TrainControl::none(),
        )
        .unwrap();
    });
    report(
        &format!("permutation importance {n}x{p}, 10 repeats"),
        d,
        1_000,
    );
}
