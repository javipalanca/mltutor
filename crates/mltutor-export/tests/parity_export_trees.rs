//! F3 parity: ONNX `TreeEnsemble*` and the tree's Python package against
//! `tests/parity/trees/*.json` (`onnx_reference`, `export_package`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use base64::Engine;
use mltutor_core::dataset::{load_builtin, Dataset, DatasetId};
use mltutor_core::model::{Prediction, Predictor, TrainControl};
use mltutor_core::tree::{self, DecisionTree, TreeConfig, TreeCriterion, TreeTask};
use mltutor_core::{Target, TaskChoice};
use mltutor_export::onnx::{self, Model};
use mltutor_export::package::{self, PackageInput};
use ndarray::{Array2, Axis};
use serde_json::Value;

const CASES: &[&str] = &[
    "breast_cancer_entropy_d10",
    "breast_cancer_gini_d3",
    "california_housing_squared_error_d3",
    "california_housing_squared_error_d8_mss10",
    "csv_numeric_regression_squared_error_d3",
    "csv_string_target_gini_d3",
    "diabetes_absolute_error_d4",
    "diabetes_friedman_mse_d5_mss10",
    "diabetes_squared_error_d10",
    "diabetes_squared_error_d3",
    "iris_entropy_d5_mss5",
    "iris_gini_d1",
    "iris_gini_d3",
    "titanic_gini_d4",
    "wine_entropy_d3_mss10",
    "wine_gini_d4",
];

fn parity(rel: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/parity")
        .join(rel);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

struct Case {
    fx: Value,
    ds: Dataset,
    tree: DecisionTree,
    x_train: Array2<f64>,
    x_test: Array2<f64>,
}

fn load(name: &str) -> Case {
    let fx = parity(&format!("trees/{name}.json"));
    let input = &fx["input"];
    let ds = if let Some(rel) = input["dataset"].as_str() {
        let slug = rel
            .trim_start_matches("datasets/")
            .trim_end_matches(".json");
        load_builtin(DatasetId::from_slug(slug).unwrap()).unwrap()
    } else {
        let csv = parity(input["csv_fixture"].as_str().unwrap());
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(csv["input"]["csv_base64"].as_str().unwrap())
            .unwrap();
        let file = input["app_label"]
            .as_str()
            .unwrap()
            .trim_start_matches("📄 ");
        mltutor_core::csv::read_csv(&bytes)
            .unwrap()
            .to_dataset(input["target"].as_str().unwrap(), TaskChoice::Auto, file)
            .unwrap()
    };
    let idx = |k: &str| -> Vec<usize> {
        fx["expected"][k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };
    let (train, test) = (idx("train_indices"), idx("test_indices"));
    let task = if input["tree_type"] == "Clasificación" {
        TreeTask::Classification
    } else {
        TreeTask::Regression
    };
    let criterion = match input["criterion"].as_str().unwrap() {
        "gini" => TreeCriterion::Gini,
        "entropy" => TreeCriterion::Entropy,
        "squared_error" => TreeCriterion::SquaredError,
        "friedman_mse" => TreeCriterion::FriedmanMse,
        _ => TreeCriterion::AbsoluteError,
    };
    let config = TreeConfig::new(
        task,
        criterion,
        input["max_depth"].as_u64().unwrap() as u32,
        input["min_samples_split"].as_u64().unwrap() as u32,
    );
    let x_train = ds.features.select(Axis(0), &train);
    let tree = tree::fit(
        config,
        x_train.view(),
        &ds.target.select(&train),
        &TrainControl::none(),
    )
    .unwrap();
    let x_test = ds.features.select(Axis(0), &test);
    Case {
        fx,
        ds,
        tree,
        x_train,
        x_test,
    }
}

fn class_values(ds: &Dataset) -> Option<Vec<f64>> {
    match &ds.target {
        Target::Classes { class_values, .. } => Some(class_values.clone()),
        Target::Values { .. } => None,
    }
}

fn f32_rows(x: &Array2<f64>) -> Vec<Vec<f32>> {
    x.rows()
        .into_iter()
        .map(|r| r.iter().map(|&v| v as f32).collect())
        .collect()
}

#[test]
fn tree_onnx_structure_matches_skl2onnx() {
    for name in CASES {
        let c = load(name);
        let cv = class_values(&c.ds);
        let m = mltutor_export::onnx_model(&c.tree, cv.as_deref()).unwrap();
        let m = Model::from_bytes(&m.to_bytes()).unwrap();
        let r = &c.fx["expected"]["onnx_reference"];
        let g = m.graph.as_ref().unwrap();
        let outputs: Vec<&str> = g.output.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(serde_json::json!(outputs), r["outputs"], "{name}");
        let theirs = &r["nodes"][0];
        let ours = &g.node[0];
        assert_eq!(ours.op_type, theirs["op_type"].as_str().unwrap(), "{name}");
        let attrs = theirs["attributes"].as_object().unwrap();
        let binary = c.tree.classes().is_some_and(|k| k.len() == 2);
        for a in &ours.attribute {
            let want = &attrs[&a.name];
            match want {
                Value::String(s) if s.starts_with('<') => {
                    let n = a.floats.len().max(a.ints.len());
                    assert!(s.starts_with(&format!("<{n} ")), "{name} {}: {s}", a.name);
                }
                Value::String(s) => assert_eq!(s.as_bytes(), a.s.as_slice(), "{name} {}", a.name),
                Value::Array(v) if v.first().is_some_and(Value::is_string) => {
                    let mut ours: Vec<String> = a
                        .strings
                        .iter()
                        .map(|s| String::from_utf8_lossy(s).into_owned())
                        .collect();
                    if ours.len() > 8 {
                        // The fixture keeps the sorted set of long string lists.
                        ours.sort();
                        ours.dedup();
                    }
                    assert_eq!(serde_json::json!(ours), *want, "{name} {}", a.name);
                }
                Value::Array(_) => {
                    assert_eq!(serde_json::json!(a.ints), *want, "{name} {}", a.name)
                }
                Value::Number(n) => assert_eq!(a.i, n.as_i64().unwrap(), "{name} {}", a.name),
                _ => panic!("{want}"),
            }
        }
        assert_eq!(ours.attribute.len(), attrs.len(), "{name}: attributes");
        let _ = binary;
    }
}

/// D47: with the largest float32 ≤ t the ONNX tree routes every train and test row
/// like scikit-learn (and our tree): labels and leaves exact.
#[test]
fn tree_onnx_outputs_match() {
    let (mut worst_p, mut worst_v) = (0.0_f64, 0.0_f64);
    let mut rows = 0;
    for name in CASES {
        let c = load(name);
        let cv = class_values(&c.ds);
        let m = mltutor_export::onnx_model(&c.tree, cv.as_deref()).unwrap();
        let r = &c.fx["expected"]["onnx_reference"];
        for x in [&c.x_train, &c.x_test] {
            let (labels, scores) = onnx::evaluate(&m, &f32_rows(x)).unwrap();
            rows += labels.len();
            match c.tree.predict(x.view()).unwrap() {
                Prediction::Classes(cl) => {
                    let lab =
                        mltutor_export::onnx_class_labels(c.tree.classes().unwrap(), cv.as_deref());
                    let pos: Vec<i64> = cl
                        .iter()
                        .map(|k| {
                            lab[c
                                .tree
                                .classes()
                                .unwrap()
                                .iter()
                                .position(|c| c == k)
                                .unwrap()]
                        })
                        .collect();
                    assert_eq!(labels, pos, "{name}: labels");
                    let p = c.tree.predict_proba(x.view()).unwrap().unwrap();
                    for (i, s) in scores.iter().enumerate() {
                        for (j, v) in s.iter().enumerate() {
                            worst_p = worst_p.max((f64::from(*v) - p[[i, j]]).abs());
                        }
                    }
                }
                Prediction::Values(v) => {
                    for (s, y) in scores.iter().zip(&v) {
                        worst_v = worst_v.max((f64::from(s[0]) - y).abs() / y.abs().max(1e-300));
                    }
                }
            }
        }
        // The fixture's outputs (skl2onnx, first rows of the test set).
        let n = r["label"]
            .as_array()
            .or(r["variable"].as_array())
            .unwrap()
            .len();
        let head = c.x_test.slice(ndarray::s![..n, ..]).to_owned();
        let (labels, scores) = onnx::evaluate(&m, &f32_rows(&head)).unwrap();
        if let Some(l) = r["label"].as_array() {
            let want: Vec<i64> = l.iter().map(|v| v.as_i64().unwrap()).collect();
            assert_eq!(labels, want, "{name}: fixture labels");
            for (s, w) in scores.iter().zip(r["probabilities"].as_array().unwrap()) {
                for (a, b) in s.iter().zip(w.as_array().unwrap()) {
                    assert!(
                        (f64::from(*a) - b.as_f64().unwrap()).abs() <= 1e-7,
                        "{name}: proba"
                    );
                }
            }
        } else {
            for (s, w) in scores.iter().zip(r["variable"].as_array().unwrap()) {
                let b = w.as_f64().unwrap();
                assert!(
                    (f64::from(s[0]) - b).abs() <= 1e-6 * b.abs().max(1.0),
                    "{name}: variable"
                );
            }
        }
    }
    println!("ONNX rows {rows}: proba max err {worst_p:e} (tol 1e-7), regression rel {worst_v:e} (tol 1e-6)");
    assert!(worst_p <= 1e-7 && worst_v <= 1e-6);
}

fn close(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) if x.is_f64() || y.is_f64() => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            let err = (x - y).abs() / y.abs().max(1.0);
            assert!(err <= 1e-12, "{path}: {x} vs {y}");
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: len");
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                close(p, q, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            let kx: Vec<&String> = x.keys().collect();
            let ky: Vec<&String> = y.keys().collect();
            assert_eq!(kx, ky, "{path}: keys");
            for (k, v) in x {
                close(v, &y[k], &format!("{path}.{k}"));
            }
        }
        _ => assert_eq!(a, b, "{path}"),
    }
}

#[test]
fn tree_python_package_matches_the_fixture() {
    for name in CASES {
        let c = load(name);
        let e = &c.fx["expected"];
        let ep = &e["export_package"];
        let cv = class_values(&c.ds);
        // Python names text classes "Clase i" (F1); Rust keeps the real names, so
        // the fixture's names are passed to compare the rest exactly.
        let names: Option<Vec<String>> = e["class_names"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_str().unwrap().to_owned()).collect());
        let input = PackageInput {
            model: &c.tree,
            feature_columns: &c.ds.feature_columns,
            display_names: &c.ds.feature_display,
            class_values: cv.as_deref(),
            class_names: names.as_deref(),
            dataset_label: c.fx["input"]["app_label"].as_str().unwrap(),
            test_size: 0.3,
            random_state: 42,
            n_train: c.x_train.nrows(),
            n_test: c.x_test.nrows(),
            x_test: c.x_test.view(),
        };
        let pkg = package::python_package(&input).unwrap();
        mltutor_testkit::assert_template_text(
            &format!("trees/{name}.json"),
            "/expected/export_package/script",
            &pkg.script,
        );
        let ours: Value = serde_json::from_str(&pkg.model_json).unwrap();
        close(&ours, &ep["model_json"], name);
        assert!(pkg.readme.contains("Árbol:"), "{name}");
        assert_eq!(package::zip_name(&c.tree), ep["zip_name"].as_str().unwrap());
        let _: &dyn Predictor = &c.tree;
    }
}

/// Writes the tree ONNX files for an external onnxruntime check:
/// `MLTUTOR_ONNX_OUT=dir cargo test -p mltutor-export --test parity_export_trees -- --ignored`
#[test]
#[ignore = "writes files for an external onnxruntime check"]
fn write_tree_onnx_for_onnxruntime() {
    let Ok(dir) = std::env::var("MLTUTOR_ONNX_OUT") else {
        return;
    };
    for name in CASES {
        let c = load(name);
        let cv = class_values(&c.ds);
        let m = mltutor_export::onnx_model(&c.tree, cv.as_deref()).unwrap();
        std::fs::write(format!("{dir}/tree_{name}.onnx"), m.to_bytes()).unwrap();
        let mut x = c.x_train.clone();
        x.append(Axis(0), c.x_test.view()).unwrap();
        let (labels, scores) = onnx::evaluate(&m, &f32_rows(&x)).unwrap();
        let expected: Vec<f64> = c.tree.predict(x.view()).unwrap().to_f64();
        let doc = serde_json::json!({"x": f32_rows(&x), "labels": labels, "scores": scores, "predict": expected});
        std::fs::write(format!("{dir}/tree_{name}.json"), doc.to_string()).unwrap();
    }
}
