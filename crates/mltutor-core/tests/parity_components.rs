//! F2 parity of the shared components (fase2-regresion.md C1–C9) against the
//! rendered Python lesson (`tests/parity/linear_models/*.json`).
//!
//! Where a component depends on the optimiser (logistic regression), the test
//! feeds it the **fixture's** predictions/probabilities/coefficients, so that the
//! comparison measures the component and not L-BFGS (see `parity_linear_models.rs`
//! for the models themselves).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]

mod common;

use std::any::Any;

use common::lm::{collections, matrix, nodes_of, notice_texts, Case, LINEAR, LOGISTIC};
use common::{nums, rel_err, strings, MaxErr};
use mltutor_core::coefficients::{self, Effect, Level};
use mltutor_core::curves::{binary_curves, multiclass_curves, Curve};
use mltutor_core::form::{self, ControlKind, PredictionOutcome};
use mltutor_core::grid::{decision_grid, GridNotice, GridOptions, GridValues};
use mltutor_core::linear::{LinearRegression, LogisticParams, LogisticRegression};
use mltutor_core::model::{LinearCoefficients, ModelError, Prediction, Predictor, TrainControl};
use mltutor_core::probabilities::{distribution, ProbabilityView};
use mltutor_core::pyfmt::{py_format, FmtArg};
use mltutor_core::{interpretation, metrics, residuals, Target, TaskType};
use ndarray::{Array2, ArrayView2};
use serde_json::Value;

fn codes(v: &Value) -> Vec<usize> {
    nums(v).iter().map(|&x| x as usize).collect()
}

fn class_names(case: &Case) -> Vec<String> {
    strings(&case.expected()["class_names"])
}

// ---------------------------------------------------------------- C1 + C3

#[test]
fn classification_evaluation_and_interpretation() {
    for name in LOGISTIC {
        let case = Case::load(name);
        let e = case.expected();
        let names = class_names(&case);
        let y_test = codes(&e["y_test"]);
        let y_pred = codes(&e["y_pred"]);
        let ev = metrics::classification(&y_test, &y_pred, names.len());
        let rep = &e["evaluation"]["classification_report"];
        let mut err = MaxErr::default();
        for (i, n) in names.iter().enumerate() {
            let r = &rep[n];
            let m = &ev.per_class[i];
            err.add((m.precision - r["precision"].as_f64().unwrap()).abs());
            err.add((m.recall - r["recall"].as_f64().unwrap()).abs());
            err.add((m.f1 - r["f1-score"].as_f64().unwrap()).abs());
            assert_eq!(f64::from(m.support), r["support"].as_f64().unwrap());
        }
        for (key, m) in [
            ("macro avg", &ev.macro_avg),
            ("weighted avg", &ev.weighted_avg),
        ] {
            err.add((m.precision - rep[key]["precision"].as_f64().unwrap()).abs());
            err.add((m.recall - rep[key]["recall"].as_f64().unwrap()).abs());
            err.add((m.f1 - rep[key]["f1-score"].as_f64().unwrap()).abs());
        }
        err.add((ev.accuracy - e["evaluation"]["accuracy"].as_f64().unwrap()).abs());
        let cm: Vec<Vec<u32>> = e["evaluation"]["confusion_matrix"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect()
            })
            .collect();
        assert_eq!(ev.confusion, cm, "{name}");
        assert_eq!(
            ev.warnings.len(),
            e["evaluation"]["classification_report_warnings"]
                .as_array()
                .unwrap()
                .len(),
            "{name}: {:?}",
            ev.warnings
        );
        println!(
            "{name}: report max abs err {:.1e} (exact required)",
            err.max
        );
        assert!(err.max <= 1e-12, "{name}");

        // Sample tables.
        let view = &e["views"]["evaluation"];
        let tables: Vec<&Value> = nodes_of(view, "table")
            .into_iter()
            .filter(|t| t["table"]["columns"][0] == "Real")
            .collect();
        let as_rows = |rows: &[metrics::SampleRow]| -> Vec<Value> {
            rows.iter()
                .map(|r| serde_json::json!([names[r.real], names[r.predicted], r.correct]))
                .collect()
        };
        assert_eq!(
            tables[0]["table"]["rows"].as_array().unwrap(),
            &as_rows(&ev.correct_head)
        );
        if ev.incorrect_head.is_empty() {
            assert_eq!(tables.len(), 1);
        } else {
            assert_eq!(
                tables[1]["table"]["rows"].as_array().unwrap(),
                &as_rows(&ev.incorrect_head)
            );
        }

        // C3.
        let md = interpretation::classification(&ev).unwrap();
        assert!(notice_texts(view).contains(&md), "{name}: {md}");
    }
}

// ---------------------------------------------------------------- C2 + C3

#[test]
fn regression_evaluation_histogram_kde_and_interpretation() {
    for name in LINEAR.iter().copied().chain(["iris_linear_mismatch"]) {
        let case = Case::load(name);
        let e = case.expected();
        let y_test = nums(&e["y_test"]);
        let y_pred = nums(&e["y_pred"]);
        let ev = metrics::regression(&y_test, &y_pred);
        let view = &e["views"]["evaluation"];
        let mut err = MaxErr::default();
        if let Some(ref_ev) = e["evaluation"]
            .as_object()
            .filter(|o| o.contains_key("mse"))
        {
            for (k, v) in [
                ("mse", ev.mse),
                ("rmse", ev.rmse),
                ("mae", ev.mae),
                ("r2", ev.r2),
            ] {
                err.add(rel_err(v, ref_ev[k].as_f64().unwrap()));
            }
        }
        // Metrics as printed ({:.4f}).
        let metrics_shown: Vec<String> = nodes_of(view, "metric")
            .iter()
            .map(|m| m["value"].as_str().unwrap().to_owned())
            .collect();
        let ours: Vec<String> = [ev.mse, ev.rmse, ev.mae, ev.r2]
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect();
        assert_eq!(metrics_shown, ours, "{name}");

        // Histogram of errors (seaborn bins="auto") and its KDE.
        let ax = common::lm::axes(view)
            .into_iter()
            .find(|a| a["title"] == "Distribución de errores")
            .unwrap();
        let heights: Vec<u32> = nums(&ax["bars"]["height"])
            .iter()
            .map(|&h| h as u32)
            .collect();
        assert_eq!(ev.errors_hist.counts, heights, "{name}");
        for (a, b) in ev.errors_hist.edges.iter().zip(nums(&ax["bars"]["x"])) {
            err.add(
                (a - b).abs()
                    / ev.errors_hist
                        .edges
                        .iter()
                        .fold(1.0_f64, |m, v| m.max(v.abs())),
            );
        }
        let line = ax["lines"]
            .as_array()
            .unwrap()
            .iter()
            .find(|l| l["xy"]["n"] == 200)
            .unwrap();
        let kde = ev.errors_kde.as_ref().unwrap();
        let mut kerr = MaxErr::default();
        for (i, p) in line["xy"]["values"].as_array().unwrap().iter().enumerate() {
            kerr.add(rel_err(kde.x[i], p[0].as_f64().unwrap()));
            kerr.add(rel_err(kde.y[i], p[1].as_f64().unwrap()));
        }
        println!(
            "{name}: metrics/edges rel {:.1e} (tol 1e-8), KDE rel {:.1e} (tol 1e-9)",
            err.max, kerr.max
        );
        assert!(err.max <= 1e-8, "{name}");
        assert!(kerr.max <= 1e-9, "{name}");

        let md = interpretation::regression(&ev).unwrap();
        assert!(notice_texts(view).contains(&md), "{name}: {md}");
    }
}

// ---------------------------------------------------------------- C4

fn json_thresholds(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|t| {
            if t == "inf" {
                f64::INFINITY
            } else {
                t.as_f64().unwrap()
            }
        })
        .collect()
}

fn same_curve(ours: &Curve, x: &Value, y: &Value, thr: Option<&Value>, what: &str) -> f64 {
    assert_eq!(ours.x.len(), x.as_array().unwrap().len(), "{what}: length");
    let mut err = MaxErr::default();
    for (a, b) in ours.x.iter().zip(nums(x)) {
        err.add((a - b).abs());
    }
    for (a, b) in ours.y.iter().zip(nums(y)) {
        err.add((a - b).abs());
    }
    if let Some(t) = thr {
        assert_eq!(ours.thresholds, json_thresholds(t), "{what}: thresholds");
    }
    err.max
}

#[test]
fn roc_and_pr_curves_with_fixture_probabilities() {
    for name in LOGISTIC {
        let case = Case::load(name);
        let e = case.expected();
        let y_test = codes(&e["y_test"]);
        let proba = matrix(&e["proba"]);
        let classes: Vec<usize> = codes(&e["model"]["classes"]);
        let c = &e["evaluation"]["curves"];
        let mut err = MaxErr::default();
        if let Some(b) = c.get("binary") {
            let ours = binary_curves(&y_test, &proba.column(1).to_vec(), classes[1]);
            err.add(same_curve(
                &ours.roc,
                &b["roc"]["fpr"],
                &b["roc"]["tpr"],
                Some(&b["roc"]["thresholds"]),
                name,
            ));
            err.add(same_curve(
                &ours.pr,
                &b["pr"]["recall"],
                &b["pr"]["precision"],
                Some(&b["pr"]["thresholds"]),
                name,
            ));
            err.add((ours.auc - b["roc"]["auc"].as_f64().unwrap()).abs());
            err.add((ours.ap - b["pr"]["average_precision"].as_f64().unwrap()).abs());
            err.add((ours.baseline - b["baseline"].as_f64().unwrap()).abs());
        } else {
            let m = &c["multiclass"];
            let ours = multiclass_curves(&y_test, proba.view(), &classes);
            assert!(ours.absent_classes.is_empty());
            for (pc, oc) in m["per_class"]
                .as_array()
                .unwrap()
                .iter()
                .zip(&ours.per_class)
            {
                err.add(same_curve(
                    &oc.roc,
                    &pc["roc"]["fpr"],
                    &pc["roc"]["tpr"],
                    Some(&pc["roc"]["thresholds"]),
                    name,
                ));
                err.add(same_curve(
                    &oc.pr,
                    &pc["pr"]["recall"],
                    &pc["pr"]["precision"],
                    Some(&pc["pr"]["thresholds"]),
                    name,
                ));
                err.add((oc.auc - pc["roc"]["auc"].as_f64().unwrap()).abs());
                err.add((oc.ap - pc["pr"]["average_precision"].as_f64().unwrap()).abs());
            }
            let mi = &m["micro"];
            err.add(same_curve(
                &ours.micro.roc,
                &mi["roc"]["fpr"],
                &mi["roc"]["tpr"],
                Some(&mi["roc"]["thresholds"]),
                name,
            ));
            err.add(same_curve(
                &ours.micro.pr,
                &mi["pr"]["recall"],
                &mi["pr"]["precision"],
                Some(&mi["pr"]["thresholds"]),
                name,
            ));
            err.add((ours.micro.auc - mi["roc"]["auc"].as_f64().unwrap()).abs());
            err.add((ours.micro.ap - mi["pr"]["average_precision"].as_f64().unwrap()).abs());
            let ma = &m["macro"];
            err.add(same_curve(
                &ours.macro_roc,
                &ma["roc"]["fpr"],
                &ma["roc"]["tpr"],
                None,
                name,
            ));
            err.add((ours.macro_auc - ma["roc"]["auc"].as_f64().unwrap()).abs());
            err.add((ours.macro_ap - ma["average_precision"].as_f64().unwrap()).abs());
            // The macro AUC interpolates the curves; it is not the mean of the AUCs.
            let mean_auc =
                ours.per_class.iter().map(|c| c.auc).sum::<f64>() / ours.per_class.len() as f64;
            println!(
                "{name}: macro AUC {:.4} vs mean of per-class AUCs {mean_auc:.4}",
                ours.macro_auc
            );
            let w = &c["not_in_app"];
            err.add((ours.weighted_auc - w["roc_auc_ovr_weighted"].as_f64().unwrap()).abs());
            err.add((ours.weighted_ap - w["average_precision_weighted"].as_f64().unwrap()).abs());
        }
        println!(
            "{name}: curves max abs err {:.1e} (exact; AUC/AP tol 1e-3)",
            err.max
        );
        assert!(err.max <= 1e-12, "{name}: {}", err.max);
    }
}

// ---------------------------------------------------------------- C5 + C6

#[test]
fn probability_histograms_and_thresholds() {
    for name in LOGISTIC {
        let case = Case::load(name);
        let e = case.expected();
        let Some(view) = e["views"].get("viz_probs") else {
            continue;
        };
        let y_test = codes(&e["y_test"]);
        let proba = matrix(&e["proba"]);
        let classes: Vec<usize> = codes(&e["model"]["classes"]);
        let pv = distribution(&y_test, proba.view(), &classes);
        let bars: Vec<&Value> = common::lm::axes(view)
            .into_iter()
            .map(|a| &a["bars"])
            .filter(|b| b.is_object())
            .collect();
        let hists = match &pv {
            ProbabilityView::Binary(b) => {
                b.per_real_class.iter().map(|h| &h.hist).collect::<Vec<_>>()
            }
            ProbabilityView::Multiclass { per_class } => {
                per_class.iter().map(|h| &h.hist).collect()
            }
        };
        // Binary: both groups in one axes (40 bars); multiclass: one axes per class.
        let heights: Vec<u32> = bars
            .iter()
            .flat_map(|b| nums(&b["height"]))
            .map(|h| h as u32)
            .collect();
        let lefts: Vec<f64> = bars.iter().flat_map(|b| nums(&b["x"])).collect();
        let ours_h: Vec<u32> = hists.iter().flat_map(|h| h.counts.clone()).collect();
        let ours_x: Vec<f64> = hists
            .iter()
            .flat_map(|h| h.edges[..h.edges.len() - 1].to_vec())
            .collect();
        assert_eq!(ours_h, heights, "{name}");
        let mut err = MaxErr::default();
        for (a, b) in ours_x.iter().zip(&lefts) {
            err.add((a - b).abs());
        }
        println!("{name}: probability bins max abs err {:.1e}", err.max);
        assert!(err.max <= 1e-15, "{name}");
        if let ProbabilityView::Binary(b) = &pv {
            let ta = &e["evaluation"]["threshold_analysis"];
            assert_eq!(
                b.uncertain as u64,
                ta["uncertain_0.3_0.7"].as_u64().unwrap()
            );
            assert_eq!(
                b.confident as u64,
                ta["confident_lt_0.2_or_gt_0.8"].as_u64().unwrap()
            );
            let shown = notice_metric_values(view);
            assert!(shown.contains(&format!("{}/{}", b.uncertain, b.n)));
            assert!(shown.contains(&py_format(
                "{p:.1%}",
                &[("p", FmtArg::Float(b.uncertain_fraction()))]
            )));
            let rows: Vec<&Value> = ta["rows"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["included"] == true)
                .collect();
            assert_eq!(rows.len(), b.thresholds.rows.len());
            for (r, o) in rows.iter().zip(&b.thresholds.rows) {
                assert_eq!(o.threshold, r["threshold"].as_f64().unwrap());
                for (k, v) in [
                    ("accuracy", o.accuracy),
                    ("precision", o.precision),
                    ("recall", o.recall),
                    ("f1", o.f1),
                ] {
                    assert!((v - r[k].as_f64().unwrap()).abs() <= 1e-15, "{name} {k}");
                }
            }
            assert_eq!(
                b.thresholds.best_f1_threshold,
                ta["best_f1_threshold"].as_f64()
            );
            // D21: Python only showed 0.5; we show 0.3, 0.5 and 0.7.
            assert_eq!(nums(&ta["comparison_rows_shown_by_app"]), vec![0.5]);
            assert_eq!(b.thresholds.shown_thresholds.len(), 3);
        }
    }
}

fn notice_metric_values(view: &Value) -> Vec<String> {
    nodes_of(view, "metric")
        .iter()
        .map(|m| m["value"].as_str().unwrap().to_owned())
        .collect()
}

// ---------------------------------------------------------------- C7

#[test]
fn residual_analysis() {
    for name in LINEAR {
        let case = Case::load(name);
        let e = case.expected();
        let r = &e["evaluation"]["residuals"];
        let a = residuals::analysis(&nums(&e["y_test"]), &nums(&e["y_pred"]));
        let mut err = MaxErr::default();
        err.add(rel_err(a.mean, r["mean"].as_f64().unwrap()));
        err.add(rel_err(a.std, r["std_ddof1"].as_f64().unwrap()));
        err.add(rel_err(a.skew, r["skew_biased"].as_f64().unwrap()));
        err.add(rel_err(
            a.linregress_r.unwrap(),
            r["linregress_r"].as_f64().unwrap(),
        ));
        let h = &r["hist20"];
        assert_eq!(
            a.hist.counts,
            nums(&h["counts"])
                .iter()
                .map(|&c| c as u32)
                .collect::<Vec<_>>(),
            "{name}"
        );
        for (x, y) in a.hist.edges.iter().zip(nums(&h["edges"])) {
            err.add(rel_err(*x, y));
        }
        println!("{name}: residuals rel {:.1e} (tol 1e-8)", err.max);
        assert!(err.max <= 1e-8);
        assert_eq!(a.checks.mean_ok, r["mean_close_to_zero"].as_bool().unwrap());

        // Texts of the view (D22: Python's "R² = 0.0000" is not reproduced).
        let view = &e["views"]["viz_residuals"];
        let texts: Vec<String> = common::lm::axes(view)
            .iter()
            .flat_map(|ax| ax["texts"].as_array().cloned().unwrap_or_default())
            .map(|t| t["text"].as_str().unwrap().to_owned())
            .collect();
        assert!(
            texts.contains(&format!("Desv. Estándar: {:.3}", a.std)),
            "{name}"
        );
        assert!(texts.contains(&"R² = 0.0000".to_owned()));
        let legend: Vec<String> = common::lm::axes(view)
            .iter()
            .flat_map(|ax| {
                strings(&ax["legend"]["labels"])
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(legend.contains(&format!("Media: {:.3}", a.mean)), "{name}");
        // Bullets: the media one is literal; the other two were CORREGIDO, so only
        // their ✅/⚠️ outcome is compared.
        let bullets: Vec<String> = nodes_of(view, "text")
            .iter()
            .map(|t| t["text"].as_str().unwrap().to_owned())
            .filter(|t| t.starts_with("- "))
            .collect();
        let mark = |ok: bool| if ok { "- ✅" } else { "- ⚠️" };
        assert!(bullets[0].starts_with(mark(a.checks.mean_ok)));
        assert!(bullets[1].starts_with(mark(a.checks.corr_ok.unwrap())));
        assert!(bullets[2].starts_with(mark(a.checks.normal_ok)));
        let m = mltutor_core::content::Messages::components().unwrap();
        let media = m
            .text(
                "residuos",
                if a.checks.mean_ok {
                    "media_ok"
                } else {
                    "media_mal"
                },
            )
            .unwrap();
        assert_eq!(bullets[0], format!("- {media}"));
    }
}

// ---------------------------------------------------------------- C8

fn grid_of(view: &Value) -> &Value {
    &collections(view, "contourf")[0]["grid"]
}

fn notice_of(n: &GridNotice) -> String {
    match n {
        GridNotice::Created { rows, cols } => py_format(
            "Malla creada: {r} x {c} = {n:,} puntos",
            &[
                ("r", FmtArg::Int(*rows as i64)),
                ("c", FmtArg::Int(*cols as i64)),
                ("n", FmtArg::Int((rows * cols) as i64)),
            ],
        ),
        GridNotice::TooLarge { points } => py_format(
            "⚠️ La malla es muy grande ({n:,} puntos). Reduciendo resolución...",
            &[("n", FmtArg::Int(*points as i64))],
        ),
        GridNotice::Resized { rows, cols } => py_format(
            "Nueva resolución: {r} x {c} = {n:,} puntos",
            &[
                ("r", FmtArg::Int(*rows as i64)),
                ("c", FmtArg::Int(*cols as i64)),
                ("n", FmtArg::Int((rows * cols) as i64)),
            ],
        ),
    }
}

#[test]
fn decision_boundary_and_surface_grids() {
    for name in LOGISTIC.iter().chain(LINEAR.iter()) {
        let case = Case::load(name);
        let e = case.expected();
        let x_train = case.x_train();
        let y_train = case.y_train();
        let logistic = LOGISTIC.contains(name);
        let model: Box<dyn Predictor> = if logistic {
            Box::new(
                LogisticRegression::fit_target(
                    x_train.view(),
                    &y_train,
                    LogisticParams::default(),
                    &TrainControl::none(),
                )
                .unwrap(),
            )
        } else {
            Box::new(
                LinearRegression::fit_target(x_train.view(), &y_train, &TrainControl::none())
                    .unwrap(),
            )
        };
        let prefix = if logistic {
            "viz_boundary"
        } else {
            "viz_surface"
        };
        for which in ["default", "pair"] {
            let Some(pair) = case.fx["input"]["boundary_pairs"].get(which) else {
                continue;
            };
            let Some(view) = e["views"].get(format!("{prefix}_{which}")) else {
                continue;
            };
            let (fx, fy) = (
                pair[0].as_u64().unwrap() as usize,
                pair[1].as_u64().unwrap() as usize,
            );
            let g = decision_grid(
                model.as_ref(),
                x_train.view(),
                &y_train,
                fx,
                fy,
                GridOptions::default(),
                &TrainControl::none(),
            )
            .unwrap();
            let rg = grid_of(view);
            assert_eq!(g.axes.x, nums(&rg["x"]), "{name} {which} x");
            assert_eq!(g.axes.y, nums(&rg["y"]), "{name} {which} y");
            // Notices: the boundary shows three; the surface joins them in one.
            let ours: Vec<String> = g.axes.notices.iter().map(notice_of).collect();
            let shown = notice_texts(view);
            if logistic {
                for t in &ours {
                    assert!(shown.contains(t), "{name} {which}: {t}");
                }
            } else {
                let joined = format!("- {}", ours.join("\n - "));
                assert!(shown.contains(&joined), "{name} {which}: {joined}");
            }
            match &g.values {
                GridValues::Classes { z } => {
                    let digits: String = rg["z_rows_digits"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|r| r.as_str().unwrap())
                        .collect();
                    let ours: String = z.iter().map(|c| char::from(b'0' + *c as u8)).collect();
                    let diff = digits
                        .chars()
                        .zip(ours.chars())
                        .filter(|(a, b)| a != b)
                        .count();
                    let frac = diff as f64 / z.len() as f64;
                    println!(
                        "{name} {which}: boundary cells differing {diff}/{} = {:.3}% (tol 0.5%)",
                        z.len(),
                        frac * 100.0
                    );
                    assert!(frac <= 0.005, "{name} {which}");
                }
                GridValues::Values { z, .. } => {
                    let d = rg["z_decimation"].as_u64().unwrap() as usize;
                    let cols = g.axes.x.len();
                    let mut err = MaxErr::default();
                    for (ri, row) in rg["z_decimated"].as_array().unwrap().iter().enumerate() {
                        for (ci, v) in nums(row).iter().enumerate() {
                            let ours = z[ri * d * cols + ci * d];
                            err.add((ours - v).abs() / v.abs().max(1.0));
                        }
                    }
                    println!("{name} {which}: surface Z rel {:.1e} (tol 1e-9)", err.max);
                    assert!(err.max <= 1e-9, "{name} {which}");
                }
            }
            assert!(g.show_2d_warning == (x_train.ncols() > 2));
        }
    }
}

#[test]
fn same_feature_twice_is_rejected() {
    let case = Case::load("iris_logistic");
    let x = case.x_train();
    let m = LogisticRegression::fit_target(
        x.view(),
        &case.y_train(),
        LogisticParams::default(),
        &TrainControl::none(),
    )
    .unwrap();
    let r = decision_grid(
        &m,
        x.view(),
        &case.y_train(),
        1,
        1,
        GridOptions::default(),
        &TrainControl::none(),
    );
    assert!(matches!(r, Err(ModelError::InvalidSelection { .. })));
}

// ---------------------------------------------------------------- C9

#[test]
fn prediction_form_and_predictions() {
    let all = LOGISTIC
        .iter()
        .chain(LINEAR.iter())
        .chain(["iris_linear_mismatch"].iter());
    for name in all {
        let case = Case::load(name);
        let e = case.expected();
        let pf = &e["prediction_form"];
        let x_train = case.x_train();
        let f = form::build(x_train.view(), &case.ds, case.metadata.as_ref());
        let controls = pf["controls"].as_array().unwrap();
        assert_eq!(f.controls.len(), controls.len());
        let mut err = MaxErr::default();
        let mut custom = Vec::new();
        for (c, r) in f.controls.iter().zip(controls) {
            // Labels: Rust shows the metadata descriptions also for Iris (MEMORY:
            // Python's Iris metadata never matched); otherwise identical.
            let label = r["label"].as_str().unwrap();
            assert!(
                c.label == label || (label == c.name && c.description.is_some()),
                "{name}: {} vs {label}",
                c.label
            );
            match (&c.control, r["kind"].as_str().unwrap()) {
                (
                    ControlKind::Slider {
                        min,
                        max,
                        step,
                        default,
                    },
                    "slider",
                ) => {
                    err.add(rel_err(*min, r["min_value"].as_f64().unwrap()));
                    err.add(rel_err(*max, r["max_value"].as_f64().unwrap()));
                    err.add(rel_err(*step, r["step"].as_f64().unwrap()));
                    err.add(rel_err(*default, r["value"].as_f64().unwrap()));
                    custom.push(*max);
                }
                (
                    ControlKind::Select {
                        options,
                        default_index,
                    },
                    "select",
                ) => {
                    let labels: Vec<String> = options.iter().map(|o| o.label.clone()).collect();
                    assert_eq!(labels, strings(&r["display"]), "{name} {}", c.column);
                    let value = &r["value"];
                    let idx = r["options"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .position(|o| o == value)
                        .unwrap();
                    assert_eq!(*default_index, idx, "{name} {}", c.column);
                    custom.push(options.last().unwrap().code);
                }
                (ControlKind::Checkbox { default }, "checkbox") => {
                    assert!(!default);
                    custom.push(1.0);
                }
                (k, r) => panic!("{name} {}: {k:?} vs {r}", c.column),
            }
        }
        assert!(err.max <= 1e-12, "{name}: {}", err.max);

        // Predictions with the defaults and with the "custom" values.
        let x_test_rows = case.y_train();
        let model: Box<dyn Predictor> = if e["model"]["params"].get("solver").is_some() {
            if !matches!(x_test_rows, Target::Classes { .. }) {
                continue; // diabetes_logistic_mismatch: blocked (D26)
            }
            Box::new(
                LogisticRegression::fit_target(
                    x_train.view(),
                    &x_test_rows,
                    LogisticParams::default(),
                    &TrainControl::none(),
                )
                .unwrap(),
            )
        } else {
            Box::new(
                LinearRegression::fit_target(x_train.view(), &x_test_rows, &TrainControl::none())
                    .unwrap(),
            )
        };
        let tol_app = case.fx["meta"]["tolerance"]["proba_abs_vs_app"].as_f64();
        for (which, values) in [("default", f.default_values()), ("custom", custom)] {
            let want = &pf[which];
            let input32 = nums(&want["input_float32"]);
            for (a, b) in values.iter().zip(&input32) {
                assert!(
                    rel_err(f64::from(*a as f32), *b) == 0.0,
                    "{name} {which}: {a} vs {b}"
                );
            }
            match form::predict_one(model.as_ref(), &values).unwrap() {
                PredictionOutcome::Value { value } => {
                    let r = rel_err(value, want["predict"].as_f64().unwrap());
                    println!(
                        "{name} {which}: prediction rel {r:.1e} (tol 1e-5, D31 f64 vs float32)"
                    );
                    assert!(r <= 1e-5, "{name} {which}");
                }
                PredictionOutcome::Class { class, proba } => {
                    let Some(tol) = tol_app else { continue }; // Cáncer: app not converged
                    let p = proba.unwrap();
                    let wp = nums(&want["predict_proba"]);
                    let d = p
                        .iter()
                        .zip(&wp)
                        .fold(0.0_f64, |m, ((_, a), b)| m.max((a - b).abs()));
                    println!("{name} {which}: proba abs {d:.1e} (tol {tol})");
                    assert!(d <= tol, "{name} {which}");
                    assert_eq!(
                        class as u64,
                        want["predict"].as_u64().unwrap(),
                        "{name} {which}"
                    );
                }
            }
        }
    }
}

#[test]
fn linear_model_on_classes_predicts_a_value() {
    // D25: no int() truncation to a class.
    let case = Case::load("iris_linear_mismatch");
    let x = case.x_train();
    let m = LinearRegression::fit_target(x.view(), &case.y_train(), &TrainControl::none()).unwrap();
    let f = form::build(x.view(), &case.ds, case.metadata.as_ref());
    assert!(matches!(
        form::predict_one(&m, &f.default_values()).unwrap(),
        PredictionOutcome::Value { .. }
    ));
    assert!(matches!(
        form::predict_one(&m, &[1.0]),
        Err(ModelError::InvalidInput { .. })
    ));
    assert!(matches!(
        form::predict_one(&m, &[f64::NAN; 4]),
        Err(ModelError::InvalidInput { .. })
    ));
}

// ---------------------------------------------------------------- Coefficients

/// The app's fitted coefficients behind the `Predictor` interface.
#[derive(Debug)]
struct Fixed {
    task: TaskType,
    coefs: LinearCoefficients,
}

impl Predictor for Fixed {
    fn task(&self) -> TaskType {
        self.task
    }
    fn n_features(&self) -> usize {
        self.coefs.coef.ncols()
    }
    fn classes(&self) -> Option<&[usize]> {
        None
    }
    fn predict(&self, _x: ArrayView2<'_, f64>) -> Result<Prediction, ModelError> {
        Err(ModelError::NotClassifier)
    }
    fn predict_proba(&self, _x: ArrayView2<'_, f64>) -> Option<Result<Array2<f64>, ModelError>> {
        None
    }
    fn refit(
        &self,
        _x: ArrayView2<'_, f64>,
        _y: &Target,
        _c: &TrainControl<'_>,
    ) -> Result<Box<dyn Predictor>, ModelError> {
        Err(ModelError::NotClassifier)
    }
    fn linear_coefficients(&self) -> Option<&LinearCoefficients> {
        Some(&self.coefs)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn coefficients_view() {
    for name in LOGISTIC.iter().chain(LINEAR.iter()) {
        let case = Case::load(name);
        let e = case.expected();
        let logistic = LOGISTIC.contains(name);
        let coef = matrix(&e["model"]["coef"]);
        let intercept = match &e["model"]["intercept"] {
            Value::Array(_) => nums(&e["model"]["intercept"]),
            v => vec![v.as_f64().unwrap()],
        };
        let fixed = Fixed {
            task: if logistic {
                TaskType::Classification
            } else {
                TaskType::Regression
            },
            coefs: LinearCoefficients { coef, intercept },
        };
        let names = strings(&e["feature_names"]);
        let v = coefficients::view(&fixed, &names, None).unwrap();
        let q = &e["coefficients"]["abs_quantiles_row0"];
        for (k, val) in [
            ("0.25", v.quantiles.q25),
            ("0.6", v.quantiles.q60),
            ("0.75", v.quantiles.q75),
            ("0.8", v.quantiles.q80),
            ("0.95", v.quantiles.q95),
        ] {
            assert!(rel_err(val, q[k].as_f64().unwrap()) <= 1e-12, "{name} q{k}");
        }
        let view = &e["views"]["coefficients"];
        let table = nodes_of(view, "table")
            .into_iter()
            .find(|t| t["table"]["columns"][0] == "Característica")
            .unwrap();
        let rows = table["table"]["rows"].as_array().unwrap();
        for (r, o) in rows.iter().zip(&v.rows) {
            assert_eq!(r[0].as_str().unwrap(), o.feature, "{name}");
            assert_eq!(r[1].as_str().unwrap(), format!("{:.4}", o.coef), "{name}");
            let effect = if o.effect == Effect::Positive {
                "📈 Positivo"
            } else {
                "📉 Negativo"
            };
            if logistic {
                assert_eq!(
                    r[2].as_str().unwrap(),
                    format!("{:.3}", o.odds_ratio.unwrap())
                );
                assert_eq!(
                    r[3].as_str().unwrap(),
                    format!("{:+.1}%", o.change_pct.unwrap())
                );
                assert_eq!(r[4].as_str().unwrap(), effect);
                // r[5] "Fuerza del Efecto" is CORREGIDO (D27, symmetric): not compared.
            } else {
                assert_eq!(r[2].as_str().unwrap(), effect);
                let imp = match o.importance.unwrap() {
                    Level::High => "🔥 Alta",
                    Level::Medium => "🔶 Media",
                    _ => "🔹 Baja",
                };
                assert_eq!(r[3].as_str().unwrap(), imp, "{name} {}", o.feature);
            }
        }
        let metric = nodes_of(view, "metric")
            .into_iter()
            .find(|m| m["label"] == "Intercepto")
            .unwrap();
        assert_eq!(
            metric["value"].as_str().unwrap(),
            format!("{:.4}", v.intercept),
            "{name}"
        );
        // Detailed analysis: same features and intensities (the numbering is D27).
        let items: Vec<String> = nodes_of(view, "text")
            .iter()
            .map(|t| t["text"].as_str().unwrap().to_owned())
            .filter(|t| t.starts_with("**") && t.contains("- Coeficiente:"))
            .collect();
        let ours: Vec<&coefficients::CoefficientRow> = v
            .top(Effect::Positive)
            .into_iter()
            .chain(v.top(Effect::Negative))
            .collect();
        assert_eq!(items.len(), ours.len(), "{name}");
        for (t, o) in items.iter().zip(ours) {
            let mark = match o.intensity {
                Level::High | Level::VeryHigh => "🔥 **FUERTE**",
                Level::Medium => "🔶 **MODERADO**",
                Level::Low => "🔹 **DÉBIL**",
            };
            let head = t.lines().next().unwrap();
            assert!(
                head.ends_with(&format!(". {}** {mark}", o.feature)),
                "{name}: {head}"
            );
        }
        let extreme = notice_texts(view)
            .into_iter()
            .find(|t| t.contains("Coeficientes Extremos"));
        assert_eq!(extreme.is_some(), !v.extreme.is_empty(), "{name}");
        if let Some(t) = extreme {
            assert!(
                t.contains(&format!("muy altos: {}\n", v.extreme.join(", "))),
                "{name}"
            );
        }
    }
}
