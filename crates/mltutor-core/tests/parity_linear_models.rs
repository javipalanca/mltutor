//! F2 parity: linear and logistic regression against scikit-learn
//! (`tests/parity/linear_models/*.json`), using the fixtures' train/test indices.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::lm::{matrix, Case, LINEAR, LOGISTIC};
use common::{abs_err, nums, rel_err, MaxErr};
use mltutor_core::linear::{
    self, LinearRegression, LogisticParams, LogisticRegression, ModelConfig,
};
use mltutor_core::model::{ModelError, Prediction, Predictor, TrainControl};
use mltutor_core::TaskType;
use ndarray::{Array2, Axis};

/// scikit-learn's objective `mean(log-loss) + ‖W‖²/(2·C·n)` (C = 1) and
/// `max|∇|` at `coef`/`intercept` (binary: 1 row; multinomial: one row per class).
fn sklearn_objective(
    x: &Array2<f64>,
    y: &[usize],
    coef: &Array2<f64>,
    intercept: &[f64],
) -> (f64, f64) {
    let n = x.nrows() as f64;
    let mut z = x.dot(&coef.t());
    for mut row in z.rows_mut() {
        for (v, b) in row.iter_mut().zip(intercept) {
            *v += b;
        }
    }
    let k = coef.nrows();
    let mut loss = 0.0;
    let mut resid = Array2::<f64>::zeros(z.dim());
    for (i, row) in z.rows().into_iter().enumerate() {
        if k == 1 {
            let zi = row[0];
            let sp = if zi > 0.0 {
                zi + (-zi).exp().ln_1p()
            } else {
                zi.exp().ln_1p()
            };
            loss += sp - if y[i] == 1 { zi } else { 0.0 };
            resid[[i, 0]] = 1.0 / (1.0 + (-zi).exp()) - if y[i] == 1 { 1.0 } else { 0.0 };
        } else {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let s: f64 = row.iter().map(|v| (v - max).exp()).sum();
            loss += max + s.ln() - row[y[i]];
            for c in 0..k {
                resid[[i, c]] = (row[c] - max).exp() / s - if y[i] == c { 1.0 } else { 0.0 };
            }
        }
    }
    let gw = resid.t().dot(x) / n + coef / n;
    let gb = resid.sum_axis(Axis(0)) / n;
    let gmax = gw
        .iter()
        .chain(gb.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    (
        loss / n + coef.iter().map(|w| w * w).sum::<f64>() / (2.0 * n),
        gmax,
    )
}

fn fit_logistic(case: &Case, max_iter: u32) -> LogisticRegression {
    LogisticRegression::fit_target(
        case.x_train().view(),
        &case.y_train(),
        LogisticParams {
            max_iter,
            ..LogisticParams::default()
        },
        &TrainControl::none(),
    )
    .unwrap()
}

#[test]
fn ols_matches_sklearn() {
    for name in LINEAR {
        let case = Case::load(name);
        let tol = &case.fx["meta"]["tolerance"];
        let (coef_tol, pred_tol) = (
            tol["coef_rel"].as_f64().unwrap(),
            tol["pred_rel"].as_f64().unwrap(),
        );
        let m = LinearRegression::fit_target(
            case.x_train().view(),
            &case.y_train(),
            &TrainControl::none(),
        )
        .unwrap();
        let e = case.expected();
        let mut coef = MaxErr::default();
        let scale = nums(&e["model"]["coef"])
            .iter()
            .fold(0.0_f64, |a, b| a.max(b.abs()));
        for (a, b) in m.coef().iter().zip(nums(&e["model"]["coef"])) {
            // Relative to the largest coefficient: near-zero ones carry the
            // absolute rounding of the solve.
            coef.add((a - b).abs() / scale);
        }
        let icpt = rel_err(m.intercept(), e["model"]["intercept"].as_f64().unwrap());
        let pred = m.predict_values(case.x_test().view()).unwrap();
        let mut perr = MaxErr::default();
        for (a, b) in pred.iter().zip(nums(&e["y_pred"])) {
            perr.add(rel_err(*a, b));
        }
        println!("{name}: coef {:.2e} (tol {coef_tol:e}), intercept {icpt:.2e}, pred {:.2e} (tol {pred_tol:e})", coef.max, perr.max);
        assert!(coef.max <= coef_tol, "{name} coef {}", coef.max);
        assert!(icpt <= coef_tol, "{name} intercept {icpt}");
        assert!(perr.max <= pred_tol, "{name} pred {}", perr.max);
    }
}

#[test]
fn logistic_matches_converged_reference_and_app() {
    for name in LOGISTIC {
        let case = Case::load(name);
        let e = case.expected();
        let tol = &case.fx["meta"]["tolerance"];
        let max_iter = case.fx["input"]["max_iter"].as_u64().unwrap() as u32;
        let m = fit_logistic(&case, max_iter);
        let report = m.fit_report().unwrap();
        let proba = m.proba(case.x_test().view()).unwrap();

        let conv = &e["converged_reference"];
        let p_conv = matrix(&conv["proba"]);
        let mut pc = MaxErr::default();
        for (a, b) in proba.iter().zip(p_conv.iter()) {
            pc.add(abs_err(*a, *b));
        }
        let c_conv = matrix(&conv["coef"]);
        let mut cc = MaxErr::default();
        let cscale = c_conv.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        for (a, b) in m.coef().iter().zip(c_conv.iter()) {
            cc.add((a - b).abs() / cscale);
        }
        let p_app = matrix(&e["proba"]);
        let mut pa = MaxErr::default();
        for (a, b) in proba.iter().zip(p_app.iter()) {
            pa.add(abs_err(*a, *b));
        }
        println!(
            "{name}: n_iter {} (sklearn app {}, converged ref {}), converged {}, |p - p_conv| {:.2e}, coef rel {:.2e}, |p - p_app| {:.2e} (tol {})",
            report.n_iter, e["model"]["n_iter"], conv["n_iter"], report.converged, pc.max, cc.max, pa.max, tol["proba_abs_vs_app"]
        );
        // Is the reference really at the optimum? (Breast Cancer: scipy stopped on
        // its `ftol` with max|∇| = 1.2e-5, not on tol=1e-12.)
        let x_train = case.x_train();
        let y_codes = case.y_train().class_codes().unwrap();
        let (f_ref, g_ref) =
            sklearn_objective(&x_train, &y_codes, &c_conv, &nums(&conv["intercept"]));
        let (f_rust, g_rust) = sklearn_objective(&x_train, &y_codes, m.coef(), m.intercept());
        println!("    objective rust {f_rust:.15} (max|∇| {g_rust:.1e}), reference {f_ref:.15} (max|∇| {g_ref:.1e})");
        assert!(g_rust < 1e-6, "{name}: rust not at the optimum");
        assert!(
            f_rust <= f_ref + 1e-12,
            "{name}: worse objective than sklearn"
        );
        if g_ref < 1e-6 {
            assert!(
                pc.max <= tol["proba_abs_vs_converged"].as_f64().unwrap(),
                "{name}: {}",
                pc.max
            );
            assert!(
                cc.max <= tol["coef_rel_vs_converged"].as_f64().unwrap(),
                "{name}: {}",
                cc.max
            );
        } else {
            // Not converged reference: only the optimality checks above apply;
            // the distance is reported (8.3e-5 in probability for Cancer).
            assert!(pc.max <= 1e-3, "{name}: {}", pc.max);
        }
        if let Some(t) = tol["proba_abs_vs_app"].as_f64() {
            assert!(pa.max <= t, "{name}: {}", pa.max);
        }
        // Predictions: exact unless the app's probabilities are within 2·tol of a
        // tie for that sample; always equal to the converged reference's argmax.
        let tol_app = tol["proba_abs_vs_app"].as_f64().unwrap_or(f64::INFINITY);
        let Prediction::Classes(pred) = m.predict(case.x_test().view()).unwrap() else {
            panic!()
        };
        let want: Vec<usize> = nums(&e["y_pred"]).iter().map(|&v| v as usize).collect();
        let classes = m.classes().unwrap().to_vec();
        let mut flipped = 0;
        for (i, (a, b)) in pred.iter().zip(&want).enumerate() {
            let row = p_app.row(i);
            let mut sorted: Vec<f64> = row.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let margin = sorted[0] - sorted.get(1).copied().unwrap_or(0.0);
            if a != b {
                flipped += 1;
                assert!(
                    margin < 2.0 * tol_app,
                    "{name}: sample {i} differs with margin {margin}"
                );
            }
            let conv_row = p_conv.row(i);
            let conv_best = classes[(0..conv_row.len())
                .max_by(|&x, &y| conv_row[x].total_cmp(&conv_row[y]))
                .unwrap()];
            assert_eq!(*a, conv_best, "{name}: sample {i} vs converged reference");
        }
        println!("    predictions: {flipped} differ from the app (near-ties), 0 from the converged reference");
    }
}

#[test]
fn shapes_classes_and_tasks() {
    let iris = Case::load("iris_logistic");
    let m = fit_logistic(&iris, 1000);
    assert_eq!(m.coef().dim(), (3, 4));
    assert_eq!(m.classes(), Some(&[0, 1, 2][..]));
    assert_eq!(m.task(), TaskType::Classification);
    let cancer = Case::load("breast_cancer_logistic");
    let m = fit_logistic(&cancer, 1000);
    assert_eq!(m.coef().dim(), (1, 30));
    assert_eq!(m.intercept().len(), 1);
    // Model task vs dataset task: a line fitted to Iris is a regressor.
    let lin = linear::fit(
        ModelConfig::Linear,
        iris.x_train().view(),
        &iris.y_train(),
        &TrainControl::none(),
    )
    .unwrap();
    assert_eq!(lin.task(), TaskType::Regression);
}

#[test]
fn multinomial_coefficients_sum_to_zero_per_feature() {
    for name in ["iris_logistic", "csv_string_target_logistic"] {
        let case = Case::load(name);
        let m = fit_logistic(&case, 1000);
        let worst = m
            .coef()
            .columns()
            .into_iter()
            .map(|c| c.sum().abs())
            .fold(0.0_f64, f64::max);
        let ref_sum = matrix(&case.expected()["model"]["coef"])
            .columns()
            .into_iter()
            .map(|c| c.sum().abs())
            .fold(0.0_f64, f64::max);
        let b: f64 = m.intercept().iter().sum();
        println!("{name}: max |Σₖ coef[k, j]| = {worst:.1e} (sklearn {ref_sum:.1e}), Σ intercept = {b:.1e}");
        assert!(worst < 1e-12 && b.abs() < 1e-12);
    }
}

#[test]
fn iris_linear_mismatch_predicts_like_sklearn() {
    let case = Case::load("iris_linear_mismatch");
    let m = LinearRegression::fit_target(
        case.x_train().view(),
        &case.y_train(),
        &TrainControl::none(),
    )
    .unwrap();
    let pred = m.predict_values(case.x_test().view()).unwrap();
    let mut err = MaxErr::default();
    for (a, b) in pred.iter().zip(nums(&case.expected()["y_pred"])) {
        err.add(rel_err(*a, b));
    }
    println!("iris_linear_mismatch: pred rel {:.2e}", err.max);
    assert!(err.max < 1e-9);
}

#[test]
fn logistic_on_regression_target_is_blocked() {
    // D26: Python trains a 192-class model on Diabetes; Rust refuses.
    let case = Case::load("diabetes_logistic_mismatch");
    let r = linear::fit(
        ModelConfig::Logistic { max_iter: 1000 },
        case.x_train().view(),
        &case.y_train(),
        &TrainControl::none(),
    );
    assert_eq!(r.err(), Some(ModelError::InvalidTarget));
}
