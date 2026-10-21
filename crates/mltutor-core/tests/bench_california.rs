//! Benchmark of §5.4: the whole "📊 Datos" pipeline on California (20 640 × 8) must
//! take < 100 ms in release. Run with:
//! `cargo test -p mltutor-core --release --test bench_california -- --ignored --nocapture`
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::explore::{self, DiagKind};

/// Multiplier for the time budgets (`MLTUTOR_BUDGET_FACTOR`, default 1): the budgets are
/// calibrated on an M-series Mac, and CI runners with 4 vCPUs are slower.
fn budget_factor() -> f64 {
    std::env::var("MLTUTOR_BUDGET_FACTOR")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|f| f.is_finite() && *f >= 1.0)
        .unwrap_or(1.0)
}

fn pipeline() -> Duration {
    let start = Instant::now();
    let ds = load_builtin(DatasetId::CaliforniaHousing).unwrap();
    let _ = ds.summary();
    let _ = explore::preview(&ds, explore::PREVIEW_ROWS);
    let _ = explore::describe(&ds);
    let _ = explore::target_distribution(&ds);
    let _ = explore::correlation_matrix(&ds);
    let sel = explore::pairplot_defaults(&ds).default_selected;
    let _ = explore::pairplot_data(&ds, &sel, DiagKind::Kde).unwrap();
    let _ = explore::pairplot_data(&ds, &sel, DiagKind::Hist).unwrap();
    start.elapsed()
}

#[test]
#[ignore = "benchmark: run in release with --ignored"]
fn california_full_pipeline_under_100_ms() {
    let _ = pipeline(); // warm-up
    let mut runs: Vec<Duration> = (0..10).map(|_| pipeline()).collect();
    runs.sort();
    let median = runs[runs.len() / 2];
    println!(
        "California full pipeline (load + summary + preview + describe + target + corr + pairplot KDE + hist): \
         min {:?}, median {:?}, max {:?}",
        runs[0],
        median,
        runs[runs.len() - 1]
    );
    if !cfg!(debug_assertions) {
        assert!(
            median.as_secs_f64() * 1e3 < 100.0 * budget_factor(),
            "{median:?}"
        );
    }
}
