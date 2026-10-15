//! F3 parity: the CART against scikit-learn 1.7.2 (`tests/parity/trees/*.json`),
//! bit for bit in structure and thresholds (D36), impurities to 1e-12.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::trees::{with_views, Case, TRAINED};
use common::{nums, rel_err, MaxErr};
use mltutor_core::tree::DecisionTree;
use serde_json::Value;

fn i64s(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// Relative error, absolute below 1 (the fixtures' rule).
fn tol_err(a: f64, b: f64) -> f64 {
    if b.abs() < 1.0 {
        (a - b).abs()
    } else {
        rel_err(a, b)
    }
}

/// Compares a fitted tree with `expected.tree`; returns the largest impurity
/// and value errors.
/// Returns whether every impurity is also bit-exact.
fn check_tree(
    label: &str,
    t: &DecisionTree,
    e: &Value,
    imp: &mut MaxErr,
    val: &mut MaxErr,
) -> bool {
    let mut all_bits = true;
    assert_eq!(
        t.node_count() as u64,
        e["node_count"].as_u64().unwrap(),
        "{label}: node_count"
    );
    assert_eq!(
        t.children_left(),
        i64s(&e["children_left"]),
        "{label}: children_left"
    );
    assert_eq!(
        t.children_right(),
        i64s(&e["children_right"]),
        "{label}: children_right"
    );
    assert_eq!(t.feature(), i64s(&e["feature"]), "{label}: feature");
    let thr = nums(&e["threshold"]);
    for (i, (a, b)) in t.threshold().iter().zip(&thr).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: threshold of node {i}: {a} vs {b}"
        );
    }
    let n: Vec<i64> = t.nodes().iter().map(|n| n.n_samples as i64).collect();
    assert_eq!(n, i64s(&e["n_node_samples"]), "{label}: n_node_samples");
    let mg: Vec<i64> = t
        .missing_go_to_left()
        .iter()
        .map(|&v| i64::from(v))
        .collect();
    assert_eq!(
        mg,
        i64s(&e["missing_go_to_left"]),
        "{label}: missing_go_to_left"
    );
    assert_eq!(
        t.depth() as u64,
        e["max_depth"].as_u64().unwrap(),
        "{label}: max_depth"
    );
    assert_eq!(
        t.n_leaves() as u64,
        e["n_leaves"].as_u64().unwrap(),
        "{label}: n_leaves"
    );
    for (i, (node, b)) in t.nodes().iter().zip(nums(&e["impurity"])).enumerate() {
        let err = tol_err(node.impurity, b);
        assert!(
            err <= 1e-12,
            "{label}: impurity of node {i}: {} vs {b}",
            node.impurity
        );
        imp.add(err);
        all_bits &= node.impurity.to_bits() == b.to_bits();
    }
    let classifier = t.is_classifier();
    for (i, (node, row)) in t
        .nodes()
        .iter()
        .zip(e["value"].as_array().unwrap())
        .enumerate()
    {
        let b = nums(row);
        assert_eq!(node.value.len(), b.len(), "{label}: value width");
        for (x, y) in node.value.iter().zip(&b) {
            if classifier {
                assert_eq!(x.to_bits(), y.to_bits(), "{label}: value of node {i}");
            } else {
                let err = tol_err(*x, *y);
                assert!(err <= 1e-12, "{label}: value of node {i}: {x} vs {y}");
                val.add(err);
            }
        }
    }
    if classifier {
        if let Some(vc) = e["value_counts"].as_array() {
            for (i, (node, row)) in t.nodes().iter().zip(vc).enumerate() {
                let b: Vec<u32> = nums(row).iter().map(|&v| v as u32).collect();
                assert_eq!(node.counts, b, "{label}: value_counts of node {i}");
            }
        }
    }
    all_bits
}

#[test]
fn cart_structure_is_bit_exact() {
    let (mut imp, mut val) = (MaxErr::default(), MaxErr::default());
    let mut n_trees = 0;
    let mut n_bits = 0;
    for name in TRAINED {
        let case = Case::load(name);
        let e = case.expected();
        let t = case.fit();
        assert_eq!(
            e["reference_impl"]["splitter_seed"].as_u64().unwrap(),
            u64::from(mltutor_core::tree::cart::splitter_seed(42))
        );
        // Trace: the first split node draws its features in the same order.
        if let (Some(tr), Some(first)) = (
            e["reference_impl"]["trace"]
                .as_array()
                .and_then(|a| a.first()),
            t.split_traces().first(),
        ) {
            let order: Vec<usize> = tr["visit_order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(first.visit_order, order, "{name}: root visit order");
        }
        n_bits += usize::from(check_tree(name, &t, &e["tree"], &mut imp, &mut val));
        n_trees += 1;
        if let Some(two) = e["two_feature_models"].as_object() {
            for (k, m) in two {
                let f: Vec<usize> = m["features"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();
                let x = case.x_train().select(ndarray::Axis(1), &f);
                let t2 = case.fit_on(&x);
                n_bits += usize::from(check_tree(
                    &format!("{name}/{k}"),
                    &t2,
                    &m["tree"],
                    &mut imp,
                    &mut val,
                ));
                n_trees += 1;
            }
        }
    }
    println!("trees bit-exact in structure: {n_trees}; also bit-exact in impurity: {n_bits}; impurity max err {:e} ({}), regression value max err {:e} ({})", imp.max, imp.n, val.max, val.n);
    assert_eq!(n_trees, 40);
    let _ = with_views;
}

fn strings(v: &Value) -> Vec<String> {
    common::strings(v)
}

/// `y_pred` of the fixture as model outputs: classes → codes (Diabetes in
/// classification: index of the float value in `classes`).
fn expected_pred(case: &Case) -> Vec<f64> {
    let e = case.expected();
    let classes = e["classes"]
        .as_array()
        .map(|c| nums(&Value::Array(c.clone())));
    let y = nums(&e["y_pred"]);
    match (classes, case.config.task) {
        (Some(c), mltutor_core::tree::TreeTask::Classification) => y
            .iter()
            .map(|v| c.iter().position(|w| w == v).unwrap() as f64)
            .collect(),
        _ => y,
    }
}

#[test]
fn predictions_leaves_and_paths_are_exact() {
    use mltutor_core::model::Predictor;
    let mut n_rows = 0;
    for name in TRAINED {
        let case = Case::load(name);
        let e = case.expected();
        let t = case.fit();
        let xt = case.x_test();
        // Diabetes as a classifier: codes = position in classes_ (= the code we fit).
        let pred = t.predict(xt.view()).unwrap().to_f64();
        let want = expected_pred(&case);
        let raw_classifier = case.ds.target.class_codes().is_none() && t.is_classifier();
        for (i, (a, b)) in pred.iter().zip(&want).enumerate() {
            if t.is_classifier() && !raw_classifier {
                // Dataset codes: the fixture's classes are the codes too.
                assert_eq!(a, b, "{name}: y_pred[{i}]");
            } else if raw_classifier {
                assert_eq!(a, b, "{name}: y_pred[{i}] (192 classes)");
            } else {
                assert!(rel_err(*a, *b) <= 1e-12, "{name}: y_pred[{i}] {a} vs {b}");
            }
        }
        n_rows += pred.len();
        let leaves = t.apply(xt.view()).unwrap();
        let want_leaves: Vec<usize> = i64s(&e["test_leaves"])
            .iter()
            .map(|&v| v as usize)
            .collect();
        assert_eq!(leaves, want_leaves, "{name}: test_leaves");
        if let Some(proba) = e["proba"].as_array() {
            let p = t.predict_proba(xt.view()).unwrap().unwrap();
            for (i, row) in proba.iter().enumerate() {
                let b = nums(row);
                let a = p.row(i).to_vec();
                assert_eq!(a, b, "{name}: proba[{i}]");
            }
        }
        for (r, path) in e["decision_paths_test_head"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let row = xt.row(r).to_vec();
            let got = t.decision_path(&row).unwrap();
            assert_eq!(
                got.leaf as u64,
                path["leaf"].as_u64().unwrap(),
                "{name}: path leaf {r}"
            );
            let steps = path["steps"].as_array().unwrap();
            assert_eq!(got.steps.len(), steps.len(), "{name}: path len {r}");
            for (s, w) in got.steps.iter().zip(steps) {
                assert_eq!(s.node as u64, w["node"].as_u64().unwrap());
                assert_eq!(s.feature as u64, w["feature"].as_u64().unwrap());
                assert_eq!(
                    s.value,
                    w["value_float32"].as_f64().unwrap(),
                    "{name}: path value"
                );
                assert_eq!(s.threshold, w["threshold"].as_f64().unwrap());
                assert_eq!(s.goes_left, w["goes_left"].as_bool().unwrap());
            }
        }
        // Prediction form: default and custom inputs.
        if let Some(pf) = e["prediction_form"].as_object() {
            for key in ["default", "custom"] {
                let c = &pf[key];
                let input = nums(&c["input_float32"]);
                let path = t.decision_path(&input).unwrap();
                assert_eq!(
                    path.leaf as u64,
                    c["leaf"].as_u64().unwrap(),
                    "{name}: form {key}"
                );
                if let Some(p) = c["predict_proba"].as_array() {
                    assert_eq!(
                        path.leaf_value,
                        nums(&Value::Array(p.clone())),
                        "{name}: form proba"
                    );
                    assert_eq!(
                        path.leaf_class.map(|c| c as f64),
                        Some(c["predict"].as_f64().unwrap())
                    );
                } else {
                    assert!(rel_err(path.leaf_value[0], c["predict"].as_f64().unwrap()) <= 1e-12);
                }
            }
        }
    }
    println!("test rows predicted exactly: {n_rows}");
}

#[test]
fn native_importances_match() {
    let mut err = MaxErr::default();
    for name in TRAINED {
        let case = Case::load(name);
        let t = case.fit();
        let got = t.feature_importances();
        let want = nums(&case.expected()["feature_importances"]);
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            let e = tol_err(*a, *b);
            assert!(e <= 1e-12, "{name}: importance {a} vs {b}");
            err.add(e);
        }
    }
    println!("native importance max err {:e} ({})", err.max, err.n);
}

#[test]
fn permutation_importance_matches() {
    use mltutor_core::importance::{permutation, PermutationOptions};
    use mltutor_core::model::TrainControl;
    use mltutor_core::sampling::RandomState;
    let mut err = MaxErr::default();
    for name in with_views() {
        let case = Case::load(name);
        let e = &case.expected()["permutation_importance"];
        let t = case.fit();
        let pi = permutation(
            &t,
            case.x_test().view(),
            &case.y_test(),
            PermutationOptions::default(),
            &TrainControl::none(),
        )
        .unwrap();
        assert_eq!(u64::from(pi.seed), e["seed"].as_u64().unwrap());
        let mut add = |a: f64, b: f64, what: &str| {
            let d = (a - b).abs();
            assert!(d <= 1e-12, "{name}: {what} {a} vs {b}");
            err.add(d);
        };
        add(
            pi.baseline,
            e["baseline_score"].as_f64().unwrap(),
            "baseline",
        );
        for (c, row) in e["importances"].as_array().unwrap().iter().enumerate() {
            for (r, b) in nums(row).iter().enumerate() {
                add(pi.importances[[c, r]], *b, "importance");
            }
        }
        for (a, b) in pi.mean.iter().zip(nums(&e["importances_mean"])) {
            add(*a, b, "mean");
        }
        for (a, b) in pi.std.iter().zip(nums(&e["importances_std"])) {
            add(*a, b, "std");
        }
        if let Some(idx) = e["column0_shuffling_idx_after_each_repeat"].as_array() {
            let mut rs = RandomState::new(pi.seed);
            let mut cur: Vec<usize> = (0..case.test.len()).collect();
            for want in idx {
                rs.shuffle(&mut cur);
                let w: Vec<usize> = i64s(want).iter().map(|&v| v as usize).collect();
                assert_eq!(cur, w, "{name}: shuffled indices");
            }
        }
    }
    println!(
        "permutation importance max abs err {:e} ({})",
        err.max, err.n
    );
}

#[test]
fn export_text_is_exact() {
    use mltutor_core::tree::{export_text, ExportTextOptions};
    for name in with_views() {
        let case = Case::load(name);
        let e = &case.expected()["export_text"];
        let t = case.fit();
        let names = strings(&case.expected()["feature_names"]);
        let view = export_text(
            &t,
            &names,
            &ExportTextOptions {
                show_weights: true,
                ..Default::default()
            },
        );
        assert_eq!(view, e["view_text"].as_str().unwrap(), "{name}: view");
        let rules = export_text(&t, &names, &ExportTextOptions::default());
        assert_eq!(rules, e["rules_txt"].as_str().unwrap(), "{name}: rules");
        if let Some(d40) = e["view_text_with_class_names_D40"].as_str() {
            let cn = strings(&case.expected()["class_names"]);
            let v = export_text(
                &t,
                &names,
                &ExportTextOptions {
                    show_weights: true,
                    class_names: Some(&cn),
                    ..Default::default()
                },
            );
            assert_eq!(v, d40, "{name}: D40");
        }
    }
}

#[test]
fn plot_tree_boxes_are_exact() {
    use mltutor_core::tree::plot::diagram;
    let mut xy_err = MaxErr::default();
    let mut n_boxes = 0;
    let mut n_negative_zero = 0;
    for name in with_views() {
        let case = Case::load(name);
        let e = &case.expected()["plot_tree"];
        let t = case.fit();
        let names = strings(&case.expected()["feature_names"]);
        let cn = case.expected()["class_names"]
            .as_array()
            .map(|_| strings(&case.expected()["class_names"]));
        let cn = if t.is_classifier() { cn } else { None };
        let d = diagram(&t, &names, cn.as_deref());
        let boxes = e["boxes_in_drawing_order"].as_array().unwrap();
        assert_eq!(d.nodes.len(), boxes.len(), "{name}: n_boxes");
        for (n, b) in d.nodes.iter().zip(boxes) {
            // sklearn's macOS wheel prints "-0.0" for MSE noise (−1e-13) on some
            // constant leaves (FMA); we show negative noise as 0 (fase3 §8).
            let want = b["text"].as_str().unwrap().replace(" = -0.0\n", " = 0.0\n");
            if want != b["text"].as_str().unwrap() {
                n_negative_zero += 1;
            }
            assert_eq!(n.lines.join("\n"), want, "{name}: text of box {}", n.id);
            assert_eq!(
                n.fill,
                b["facecolor"].as_str().unwrap(),
                "{name}: colour of box {}",
                n.id
            );
            let xy = nums(&b["xy_axes"]);
            for k in 0..2 {
                let err = (n.xy_axes[k] - xy[k]).abs();
                assert!(
                    err <= 1e-12,
                    "{name}: xy of box {}: {:?} vs {xy:?}",
                    n.id,
                    n.xy_axes
                );
                xy_err.add(err);
            }
            n_boxes += 1;
        }
        let labels = e["edge_labels"].as_array().unwrap();
        assert_eq!(d.edge_labels.len(), labels.len(), "{name}: edge labels");
        for (l, w) in d.edge_labels.iter().zip(labels) {
            assert_eq!(l.text, w["text"].as_str().unwrap());
            assert_eq!(l.ha, w["ha"].as_str().unwrap());
            let xy = nums(&w["xy_axes"]);
            assert!((l.xy_axes[0] - xy[0]).abs() <= 1e-12 && (l.xy_axes[1] - xy[1]).abs() <= 1e-12);
        }
        let pal: Vec<Vec<f64>> = e["palette_color_brew"]
            .as_array()
            .unwrap()
            .iter()
            .map(nums)
            .collect();
        let brew = mltutor_core::tree::plot::color_brew(pal.len());
        for (a, b) in brew.iter().zip(&pal) {
            assert_eq!(
                a.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                *b,
                "{name}: palette"
            );
        }
    }
    println!(
        "plot_tree boxes: {n_boxes} ({n_negative_zero} with \"-0.0\" in Python), xy max err {:e}",
        xy_err.max
    );
}

#[test]
fn python_code_literal_and_corrected_are_exact() {
    use mltutor_core::codegen::{self, CodeTemplate};
    for name in with_views() {
        let case = Case::load(name);
        let e = &case.expected()["python_code"];
        let t = case.fit();
        let names = strings(&case.expected()["feature_names"]);
        let lit = codegen::tree_python_literal_context(&t, &names, None);
        let text = codegen::render(CodeTemplate::TreePythonLiteral, &lit).unwrap();
        assert_eq!(text, e["app_text"].as_str().unwrap(), "{name}: literal");
        let cn = case.expected()["class_names"]
            .as_array()
            .map(|_| strings(&case.expected()["class_names"]));
        let cor = codegen::tree_python_corrected_context(&t, &names, cn.as_deref());
        let want_ctx: codegen::TreePythonCorrectedContext =
            serde_json::from_value(e["corrected_context"].clone()).unwrap();
        assert_eq!(cor, want_ctx, "{name}: corrected context");
        let text = codegen::render(CodeTemplate::TreePython, &cor).unwrap();
        assert_eq!(
            text,
            e["corrected_text"].as_str().unwrap(),
            "{name}: corrected"
        );
    }
}

#[test]
fn tree_templates_match_jinja2() {
    use mltutor_core::codegen;
    let fx = common::fixture("code/trees.json");
    let mut n = 0;
    for (name, case) in fx["expected"].as_object().unwrap() {
        let text = case["text"].as_str().unwrap();
        let rendered = codegen::render_named(name, &case["context"]).unwrap();
        assert_eq!(
            rendered.trim_matches('\n'),
            text.trim_matches('\n'),
            "{name}"
        );
        n += 1;
    }
    assert_eq!(n, 4);
}

/// friedman_mse ranks the splits of a node exactly like squared_error, so a single
/// tree is the same (only the stored improvement differs).
#[test]
fn friedman_mse_grows_the_same_tree_as_squared_error() {
    use mltutor_core::tree::TreeCriterion;
    for name in [
        "diabetes_squared_error_d10",
        "california_housing_squared_error_d8_mss10",
        "csv_numeric_regression_squared_error_d3",
    ] {
        let mut case = Case::load(name);
        let mse = case.fit();
        case.config.criterion = TreeCriterion::FriedmanMse;
        let fr = case.fit();
        assert_eq!(mse.children_left(), fr.children_left(), "{name}");
        assert_eq!(mse.feature(), fr.feature(), "{name}");
        assert_eq!(mse.threshold(), fr.threshold(), "{name}");
        assert_eq!(
            mse.nodes()
                .iter()
                .map(|n| n.value.clone())
                .collect::<Vec<_>>(),
            fr.nodes()
                .iter()
                .map(|n| n.value.clone())
                .collect::<Vec<_>>(),
            "{name}"
        );
    }
}
