//! Parity of the "📊 Datos" exploration with `tests/parity/stats/*.json`.
//!
//! Tolerances (fixture `meta.tolerance`): describe 1e-12 relative, correlation
//! 1e-12 absolute, KDE density 1e-9 relative, KDE bandwidth 1e-12 relative,
//! histograms and pairplot sample rows exact.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{abs_err, fixture, num, nums, rel_err, strings, u32s, MaxErr, SLUGS};
use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::explore::{self, DiagKind, DiagPanel, Hue, TargetDistribution};
use mltutor_core::frame::Cell;
use serde_json::Value;

#[derive(Default)]
struct Report {
    describe: MaxErr,
    corr: MaxErr,
    kde_density: MaxErr,
    kde_bandwidth: MaxErr,
    kde_factor: MaxErr,
    kde_grid: MaxErr,
    target_stats: MaxErr,
}

fn check_cell(cell: &Cell, expected: &Value, ctx: &str) {
    match (cell, expected) {
        (Cell::Number { v }, Value::Number(n)) => assert_eq!(*v, n.as_f64().unwrap(), "{ctx}"),
        (Cell::Bool { v }, Value::Bool(b)) => assert_eq!(v, b, "{ctx}"),
        (Cell::Text { v }, Value::String(s)) => assert_eq!(v, s, "{ctx}"),
        (Cell::Missing, Value::Null) => {}
        other => panic!("{ctx}: {other:?}"),
    }
}

fn check_dataset(slug: &str, report: &mut Report) {
    let fx = fixture(&format!("stats/{slug}.json"));
    let tol = &fx["meta"]["tolerance"];
    assert_eq!(num(&tol["describe_rel"]), 1e-12);
    assert_eq!(num(&tol["correlation_abs"]), 1e-12);
    assert_eq!(num(&tol["kde_density_rel"]), 1e-9);
    assert_eq!(num(&tol["kde_bandwidth_rel"]), 1e-12);
    let exp = &fx["expected"];
    let ds = load_builtin(DatasetId::from_slug(slug).unwrap()).unwrap();

    // Preview (positional alignment: no Penguins NaN rows).
    let preview = explore::preview(&ds, explore::PREVIEW_ROWS);
    assert_eq!(
        preview.columns,
        strings(&exp["display_columns"]),
        "{slug} display columns"
    );
    let rows = exp["preview_rows"].as_array().unwrap();
    assert_eq!(preview.rows.len(), rows.len());
    for (r, (row, erow)) in preview.rows.iter().zip(rows).enumerate() {
        for (c, (cell, e)) in row.iter().zip(erow.as_array().unwrap()).enumerate() {
            check_cell(cell, e, &format!("{slug} preview [{r},{c}]"));
        }
    }

    // describe
    let d = explore::describe(&ds);
    let ed = &exp["describe"];
    assert_eq!(
        d.columns,
        strings(&ed["columns"]),
        "{slug} describe columns"
    );
    assert_eq!(d.stats, strings(&ed["index"]));
    assert_eq!(
        d.excluded_columns,
        strings(&exp["describe_excluded_columns"]),
        "{slug} excluded"
    );
    let ev = nums(&ed["values"]);
    for (i, v) in d.values.iter().enumerate() {
        let e = rel_err(*v, ev[i]);
        assert!(e <= 1e-12, "{slug} describe[{i}] {v} vs {}", ev[i]);
        report.describe.add(e);
    }

    // target distribution
    let et = &exp["target_distribution"];
    match explore::target_distribution(&ds) {
        TargetDistribution::Bar {
            labels,
            counts,
            balance_ratio,
            balanced,
        } => {
            assert_eq!(et["kind"], "bar");
            assert_eq!(labels, strings(&et["labels"]), "{slug}");
            assert_eq!(counts, u32s(&et["counts"]), "{slug}");
            assert_eq!(balance_ratio, num(&et["balance_ratio"]), "{slug}");
            assert_eq!(balanced, et["balanced"].as_bool().unwrap());
        }
        TargetDistribution::Histogram {
            edges,
            counts,
            mean,
            std,
            min,
            max,
        } => {
            assert_eq!(et["kind"], "histogram");
            assert_eq!(edges, nums(&et["edges"]), "{slug} target edges");
            assert_eq!(counts, u32s(&et["counts"]), "{slug} target counts");
            assert_eq!((min, max), (num(&et["min"]), num(&et["max"])));
            for (a, b) in [(mean, num(&et["mean"])), (std, num(&et["std_ddof1"]))] {
                let e = rel_err(a, b);
                assert!(e <= 1e-12, "{slug} target stats {a} vs {b}");
                report.target_stats.add(e);
            }
        }
    }

    // correlation
    let c = explore::correlation_matrix(&ds);
    let ec = &exp["correlation"];
    assert_eq!(c.labels, strings(&ec["columns"]), "{slug} corr labels");
    let ev = nums(&ec["values"]);
    for (i, v) in c.values.iter().enumerate() {
        let e = abs_err(*v, ev[i]);
        assert!(e <= 1e-12, "{slug} corr[{i}] {v} vs {}", ev[i]);
        report.corr.add(e);
    }
    for (got, key) in [
        (c.visible_min, "visible_min"),
        (c.visible_max, "visible_max"),
    ] {
        match ec[key].as_f64() {
            Some(e) => assert!(abs_err(got.unwrap(), e) <= 1e-12, "{slug} {key}"),
            None => assert!(got.is_none()),
        }
    }
    assert_eq!(
        c.nan_columns,
        strings(&ec["nan_columns"]),
        "{slug} nan columns"
    );

    // pairplot
    let ep = &exp["pairplot"];
    let defaults = explore::pairplot_defaults(&ds);
    assert_eq!(
        defaults.n_features as u64,
        ep["n_features"].as_u64().unwrap()
    );
    assert_eq!(
        defaults.default_selected,
        strings(&ep["default_selected"]),
        "{slug}"
    );
    match ep["slider"].as_object() {
        Some(s) => {
            let got = defaults.slider.unwrap();
            assert_eq!(
                (got.min as u64, got.max as u64, got.default as u64),
                (
                    s["min"].as_u64().unwrap(),
                    s["max"].as_u64().unwrap(),
                    s["default"].as_u64().unwrap()
                )
            );
        }
        None => assert!(defaults.slider.is_none()),
    }

    let kde = explore::pairplot_data(&ds, &defaults.default_selected, DiagKind::Kde).unwrap();
    let hist = explore::pairplot_data(&ds, &defaults.default_selected, DiagKind::Hist).unwrap();
    assert_eq!(
        kde.sampled,
        ep["sampled"].as_bool().unwrap(),
        "{slug} sampled"
    );
    assert_eq!(kde.n_points as u64, ep["n_points"].as_u64().unwrap());
    assert_eq!(kde.high_quality, ep["style"]["s"] == 40);
    if let Some(rows) = ep["sample_rows"].as_array() {
        let rows: Vec<usize> = rows.iter().map(|r| r.as_u64().unwrap() as usize).collect();
        assert_eq!(kde.rows, rows, "{slug} sample rows");
        assert!(kde
            .title
            .ends_with(&format!(" (Muestra: 2000/{} puntos)", ds.n_rows())));
    }
    match (&kde.hue, ep["hue_order"].as_array()) {
        (Hue::Categorical { levels, .. }, Some(_)) => {
            assert_eq!(levels, &strings(&ep["hue_order"]), "{slug} hue order");
        }
        (Hue::Continuous { values, .. }, None) => assert_eq!(values.len(), kde.n_points),
        other => panic!("{slug} hue {other:?}"),
    }
    for (f, feat) in defaults.default_selected.iter().enumerate() {
        let DiagPanel::Kde { curves } = &kde.diagonal[f] else {
            panic!("kde panel")
        };
        let ek = ep["kde"][feat].as_object().unwrap();
        assert_eq!(curves.len(), ek.len(), "{slug} {feat} levels");
        for curve in curves {
            let e = &ek[&curve.level];
            assert_eq!(
                curve.n as u64,
                e["n"].as_u64().unwrap(),
                "{slug} {feat} {}",
                curve.level
            );
            if e.get("skipped").is_some() {
                assert!(
                    curve.density.is_empty() && curve.bandwidth.is_none(),
                    "{slug} {feat} skip"
                );
                continue;
            }
            let ctx = format!("{slug} {feat} {}", curve.level);
            let bw = rel_err(curve.bandwidth.unwrap(), num(&e["bandwidth"]));
            assert!(bw <= 1e-12, "{ctx} bandwidth");
            report.kde_bandwidth.add(bw);
            report.kde_factor.add(rel_err(
                curve.scott_factor.unwrap(),
                num(&e["scott_factor"]),
            ));
            for key in ["grid_min", "grid_max"] {
                let got = if key == "grid_min" {
                    curve.grid_min
                } else {
                    curve.grid_max
                };
                let g = rel_err(got, num(&e[key]));
                assert!(g <= 1e-12, "{ctx} {key}");
                report.kde_grid.add(g);
            }
            let ed = nums(&e["density"]);
            assert_eq!(curve.density.len(), ed.len());
            for (a, b) in curve.density.iter().zip(&ed) {
                let r = rel_err(*a, *b);
                assert!(r <= 1e-9, "{ctx} density {a} vs {b}");
                report.kde_density.add(r);
            }
        }
        let DiagPanel::Hist { edges, series } = &hist.diagonal[f] else {
            panic!("hist panel")
        };
        let eh = &ep["hist"][feat];
        assert_eq!(edges, &nums(&eh["edges"]), "{slug} {feat} edges");
        let ecounts = eh["counts"].as_object().unwrap();
        assert_eq!(series.len(), ecounts.len());
        for s in series {
            assert_eq!(
                s.counts,
                u32s(&ecounts[&s.level]),
                "{slug} {feat} {} counts",
                s.level
            );
        }
    }
}

#[test]
fn exploration_matches_python_for_all_datasets() {
    let mut report = Report::default();
    for slug in SLUGS {
        check_dataset(slug, &mut report);
    }
    println!(
        "max errors: describe rel {:e} (n={}, tol 1e-12) · target mean/std rel {:e} · \
         correlation abs {:e} (n={}, tol 1e-12) · KDE density rel {:e} (n={}, tol 1e-9) · \
         KDE bandwidth rel {:e} (tol 1e-12) · Scott factor rel {:e} · KDE grid rel {:e}",
        report.describe.max,
        report.describe.n,
        report.target_stats.max,
        report.corr.max,
        report.corr.n,
        report.kde_density.max,
        report.kde_density.n,
        report.kde_bandwidth.max,
        report.kde_factor.max,
        report.kde_grid.max,
    );
}

#[test]
fn pairplot_rejects_invalid_selections() {
    let ds = load_builtin(DatasetId::Wine).unwrap();
    let names = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    assert!(explore::pairplot_data(&ds, &[], DiagKind::Kde).is_err());
    assert!(explore::pairplot_data(&ds, &names(&["nope"]), DiagKind::Kde).is_err());
    assert!(explore::pairplot_data(&ds, &names(&["alcohol", "alcohol"]), DiagKind::Kde).is_err());
    let seven = ds.feature_display[..7].to_vec();
    assert!(explore::pairplot_data(&ds, &seven, DiagKind::Hist).is_err());
    assert!(explore::pairplot_data(&ds, &ds.feature_display[..6], DiagKind::Hist).is_ok());
}

#[test]
fn mpg_pairplot_has_car_name_tooltips() {
    let ds = load_builtin(DatasetId::Mpg).unwrap();
    let sel = explore::pairplot_defaults(&ds).default_selected;
    let p = explore::pairplot_data(&ds, &sel, DiagKind::Kde).unwrap();
    let tips = p.tooltips.unwrap();
    assert_eq!(tips.len(), p.n_points);
    assert_eq!(tips[0], "chevrolet chevelle malibu");
}
