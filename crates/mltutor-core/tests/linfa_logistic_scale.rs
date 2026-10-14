//! fase2-regresion.md §8/§11: the scale of the objective. scikit-learn minimises
//! `C·Σ loss + ½‖W‖²`; linfa-logistic `Σ loss + (α/2)‖W‖²`. Same minimiser with
//! `α = 1/C`: our solver (scikit-learn's objective) and linfa-logistic agree.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::lm::Case;
use linfa::prelude::*;
use linfa_logistic::{LogisticRegression as LinfaBinary, MultiLogisticRegression};
use mltutor_core::linear::{LogisticParams, LogisticRegression};
use mltutor_core::model::TrainControl;
use ndarray::{Array1, Axis};

#[test]
fn multinomial_alpha_is_one_over_c() {
    let case = Case::load("iris_logistic");
    let (x, y) = (case.x_train(), case.y_train().class_codes().unwrap());
    for c in [1.0, 0.1] {
        let ours = LogisticRegression::fit(
            x.view(),
            &y,
            LogisticParams {
                c,
                ..Default::default()
            },
            &TrainControl::none(),
        )
        .unwrap();
        let ds = Dataset::new(x.clone(), Array1::from(y.clone()));
        let theirs = MultiLogisticRegression::default()
            .alpha(1.0 / c)
            .max_iterations(10_000)
            .gradient_tolerance(1e-10)
            .fit(&ds)
            .unwrap();
        let p_ours = ours.proba(case.x_test().view()).unwrap();
        let p_linfa = theirs.predict_probabilities(&case.x_test());
        let err = p_ours
            .iter()
            .zip(p_linfa.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        println!("multinomial C={c}: max |p_ours - p_linfa(alpha=1/C)| = {err:.2e}");
        assert!(err < 1e-6, "{err}");
    }
}

#[test]
fn binary_alpha_is_one_over_c_and_positive_class_is_the_larger_one() {
    // Iris versicolor (1) vs virginica (2): 2 classes.
    let case = Case::load("iris_logistic");
    let codes = case.ds.target.class_codes().unwrap();
    let rows: Vec<usize> = (0..codes.len()).filter(|&i| codes[i] != 0).collect();
    let x = case.ds.features.select(Axis(0), &rows);
    let y: Vec<usize> = rows.iter().map(|&i| codes[i]).collect();
    let ours = LogisticRegression::fit(
        x.view(),
        &y,
        LogisticParams::default(),
        &TrainControl::none(),
    )
    .unwrap();
    let ds = Dataset::new(x.clone(), Array1::from(y.clone()));
    let theirs = LinfaBinary::default()
        .alpha(1.0)
        .max_iterations(10_000)
        .gradient_tolerance(1e-10)
        .fit(&ds)
        .unwrap();
    // linfa's "positive" class is the more frequent one (ties: the first seen),
    // not classes_[1]: remap before comparing (MEMORY trap).
    let p_pos = theirs.predict_probabilities(&x);
    let p_ours = ours.proba(x.view()).unwrap();
    let col = if theirs.labels().pos.class == 2 { 1 } else { 0 };
    let err = p_ours
        .column(col)
        .iter()
        .zip(p_pos.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    println!(
        "binary: positive class of linfa = {}, max |p| diff = {err:.2e}",
        theirs.labels().pos.class
    );
    assert!(err < 1e-6, "{err}");
}
