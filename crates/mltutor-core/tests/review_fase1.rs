//! Regression tests of the Rust review of Phase 1 (findings on `mltutor-core`):
//! signed zeros and fast class codes, non-finite values, wide CSV files, pairplot
//! with too many classes and cooperative cancellation. Memory: `csv_memory.rs`;
//! histogram bins, header mangling and streaming inference: unit tests.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use mltutor_core::csv::{read_csv, read_csv_file_cancellable, CsvNotice, CsvTargetStats, RawTable};
use mltutor_core::dataset::{list_builtin, load_builtin};
use mltutor_core::explore::{self, DiagKind, Hue, TargetDistribution, CORRELATION_MAX_COLUMNS};
use mltutor_core::{CsvError, DatasetError, Target, TaskChoice};

fn table(text: &str) -> RawTable {
    read_csv(text.as_bytes()).unwrap()
}

#[test]
fn signed_zeros_are_one_value() {
    let t = table("a,y\n1,0.0\n2,-0.0\n3,1.5\n4,0.0\n");
    let y = &t.column("y").unwrap().data;
    assert_eq!(y.nunique(), 2);
    assert_eq!(
        y.value_counts_by_frequency(),
        vec![("0.0".to_owned(), 3), ("1.5".to_owned(), 1)]
    );
    let ds = t
        .to_dataset("y", TaskChoice::Classification, "z.csv")
        .unwrap();
    let Target::Classes {
        codes,
        class_names,
        class_values,
    } = &ds.target
    else {
        panic!("classification")
    };
    assert_eq!(class_names, &["0.0", "1.5"]);
    assert_eq!(class_values, &[0.0, 1.5]);
    assert_eq!(codes, &[0, 0, 1, 0]);
}

#[test]
fn nan_class_goes_last_and_codes_are_found_by_binary_search() {
    let t = table("a,y\n1,3.5\n2,\n3,-1\n4,3.5\n5,NA\n");
    let ds = t
        .to_dataset("y", TaskChoice::Classification, "n.csv")
        .unwrap();
    let Target::Classes {
        codes, class_names, ..
    } = &ds.target
    else {
        panic!("classification")
    };
    assert_eq!(class_names, &["-1.0", "3.5", "nan"]);
    assert_eq!(codes, &[1, 2, 0, 1, 2]);
}

#[test]
fn many_classes_are_encoded_in_linearithmic_time() {
    // Review finding: `class_index` was a linear scan (80 k rows → 0.93 s). Here
    // 100 k rows and 50 k classes: ~2.5e9 comparisons before, ~1.6e6 now.
    let mut text = String::from("a,y\n");
    for i in 0..100_000_u32 {
        text.push_str(&format!("{i},{}\n", (i * 7919) % 50_000));
    }
    let t = table(&text);
    let start = Instant::now();
    let ds = t
        .to_dataset("y", TaskChoice::Classification, "big.csv")
        .unwrap();
    let elapsed = start.elapsed();
    let Target::Classes {
        codes,
        class_values,
        ..
    } = &ds.target
    else {
        panic!("classification")
    };
    assert_eq!(class_values.len(), 50_000);
    for (i, &c) in codes.iter().enumerate().step_by(997) {
        let v = f64::from((i as u32 * 7919) % 50_000);
        assert_eq!(class_values[c as usize], v);
    }
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
}

#[test]
fn infinities_are_left_out_of_statistics_and_charts() {
    let t = table("a,y\n1,1\n2,inf\n3,3\n4,-inf\n5,5\n");
    assert_eq!(
        t.notices,
        vec![CsvNotice::NonFinite {
            columns: vec!["y".into()]
        }]
    );
    assert!(t.notices[0].message().contains("infinitos"));
    // Like pandas: float64, not missing, and inf/-inf count in `nunique` (5 values).
    let y = &t.column("y").unwrap().data;
    assert_eq!(y.kind().dtype_name(), "float64");
    assert_eq!(y.missing(), 0);
    assert_eq!(y.nunique(), 5);

    let CsvTargetStats::Numeric {
        min,
        max,
        mean,
        median,
        std,
        hist_edges,
        hist_counts,
    } = t.target_stats("y", TaskChoice::Regression).unwrap()
    else {
        panic!("numeric")
    };
    assert_eq!((min, max, mean, median), (1.0, 5.0, 3.0, 3.0));
    assert_eq!(std, 2.0);
    assert!(hist_edges.iter().all(|e| e.is_finite()));
    assert_eq!(hist_counts.iter().sum::<u32>(), 3);

    let d = t.describe();
    let y_col = d.columns.iter().position(|c| c == "y").unwrap();
    assert_eq!(d.values[[0, y_col]], 3.0, "count of finite values");
    assert_eq!(d.values[[1, y_col]], 3.0);

    let ds = t
        .to_dataset("y", TaskChoice::Regression, "inf.csv")
        .unwrap();
    let TargetDistribution::Histogram {
        counts,
        mean,
        min,
        max,
        ..
    } = explore::target_distribution(&ds)
    else {
        panic!("histogram")
    };
    assert_eq!(counts.iter().sum::<u32>(), 3);
    assert_eq!((mean, min, max), (3.0, 1.0, 5.0));

    let plot = explore::pairplot_data(&ds, &["a".to_owned()], DiagKind::Kde).unwrap();
    let Hue::Continuous { values, min, max } = plot.hue else {
        panic!("continuous hue")
    };
    assert_eq!((min, max), (1.0, 5.0));
    assert!(values[1].is_nan() && values[3].is_nan());

    let corr = explore::correlation_matrix(&ds);
    assert!(
        (corr.values[[0, 1]] - 1.0).abs() < 1e-12,
        "{:?}",
        corr.values
    );
}

#[test]
fn hue_range_without_finite_values_is_zero() {
    let t = table("a,y\n1,inf\n2,\n3,-inf\n");
    let ds = t.to_dataset("y", TaskChoice::Regression, "x.csv").unwrap();
    let plot = explore::pairplot_data(&ds, &["a".to_owned()], DiagKind::Hist).unwrap();
    let Hue::Continuous { min, max, .. } = plot.hue else {
        panic!("continuous hue")
    };
    assert_eq!((min, max), (0.0, 0.0));
}

#[test]
fn balance_ratio_is_always_finite() {
    let t = table("a,y\n1,x\n2,x\n3,y\n4,inf\n");
    let ds = t
        .to_dataset("y", TaskChoice::Classification, "b.csv")
        .unwrap();
    let TargetDistribution::Bar { balance_ratio, .. } = explore::target_distribution(&ds) else {
        panic!("bar")
    };
    assert_eq!(balance_ratio, 2.0);
}

fn wide_csv(n_features: usize, rows: usize) -> RawTable {
    let mut text: Vec<String> = (0..n_features).map(|j| format!("f{j}")).collect();
    text.push("y".into());
    let mut out = text.join(",");
    out.push('\n');
    for r in 0..rows {
        let row: Vec<String> = (0..=n_features)
            .map(|j| ((r * (j + 1)) % 7).to_string())
            .collect();
        out.push_str(&row.join(","));
        out.push('\n');
    }
    table(&out)
}

#[test]
fn wide_correlation_is_truncated_with_a_message() {
    let t = wide_csv(150, 8);
    let ds = t.to_dataset("y", TaskChoice::Regression, "w.csv").unwrap();
    let corr = explore::correlation_matrix(&ds);
    assert!(corr.truncated);
    assert_eq!(corr.total_columns, 151);
    assert_eq!(corr.labels.len(), CORRELATION_MAX_COLUMNS);
    assert_eq!(corr.values.dim(), (100, 100));
    assert_eq!(corr.labels.last().map(String::as_str), Some("target"));
    assert_eq!(corr.labels[98], "f98");
    let msg = corr.truncation_message().unwrap();
    assert!(msg.contains("151 columnas") && msg.contains("99"), "{msg}");

    let t = wide_csv(99, 8);
    let ds = t.to_dataset("y", TaskChoice::Regression, "w.csv").unwrap();
    let corr = explore::correlation_matrix(&ds);
    assert!(!corr.truncated);
    assert_eq!(corr.labels.len(), 100);
    assert!(corr.truncation_message().is_none());
}

#[test]
fn builtin_correlations_are_not_truncated() {
    for entry in list_builtin().unwrap() {
        let ds = load_builtin(entry.id).unwrap();
        let corr = explore::correlation_matrix(&ds);
        assert!(!corr.truncated, "{:?}", entry.id);
        assert_eq!(corr.labels.len(), ds.n_features() + 1);
        assert_eq!(corr.total_columns, ds.n_features() + 1);
    }
}

#[test]
fn pairplot_with_more_classes_than_points_is_an_error() {
    let mut text = String::from("a,b,y\n");
    for i in 0..2500 {
        text.push_str(&format!("{i},{},{i}\n", i % 13));
    }
    let t = table(&text);
    let ds = t
        .to_dataset("y", TaskChoice::Classification, "c.csv")
        .unwrap();
    assert!(explore::pairplot_rows(&ds).is_empty());
    let err = explore::pairplot_data(&ds, &["a".to_owned(), "b".to_owned()], DiagKind::Kde)
        .err()
        .unwrap();
    let DatasetError::InvalidSelection { reason } = &err else {
        panic!("{err:?}")
    };
    assert!(reason.contains("2500 clases"), "{reason}");
    assert!(err.to_string().starts_with("Selección no válida"));
}

#[test]
fn cancellation_flags_are_honoured() {
    let t = wide_csv(5, 20);
    let set = AtomicBool::new(true);
    let unset = AtomicBool::new(false);

    assert_eq!(
        t.to_dataset_cancellable("y", TaskChoice::Auto, "c.csv", &set)
            .err(),
        Some(CsvError::Cancelled)
    );
    let ds = t
        .to_dataset_cancellable("y", TaskChoice::Auto, "c.csv", &unset)
        .unwrap();
    assert_eq!(ds, t.to_dataset("y", TaskChoice::Auto, "c.csv").unwrap());

    assert_eq!(
        explore::correlation_matrix_cancellable(&ds, &set).err(),
        Some(DatasetError::Cancelled)
    );
    let corr = explore::correlation_matrix_cancellable(&ds, &unset).unwrap();
    let plain = explore::correlation_matrix(&ds);
    assert_eq!(corr.labels, plain.labels);
    assert!(corr
        .values
        .iter()
        .zip(plain.values.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits()));

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.csv");
    std::fs::write(&path, "a,b\n1,2\n").unwrap();
    assert_eq!(
        read_csv_file_cancellable(&path, &set).err(),
        Some(CsvError::Cancelled)
    );
    assert_eq!(read_csv_file_cancellable(&path, &unset).unwrap().n_rows, 1);
    assert_eq!(CsvError::Cancelled.to_string(), "Operación cancelada");
}
