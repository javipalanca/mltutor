//! F4 parity: the KNN's ONNX graph (D57) and its Python package against
//! `tests/parity/knn/*.json` (`onnx_proposal_D57`, `export_package`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use base64::Engine;
use mltutor_core::dataset::{load_builtin, Dataset, DatasetId};
use mltutor_core::knn::{self, Knn, KnnConfig, KnnMetric, KnnWeights};
use mltutor_core::model::{Prediction, Predictor, TrainControl};
use mltutor_core::{Target, TaskChoice};
use mltutor_export::onnx;
use mltutor_export::package::{self, PackageInput};
use ndarray::{Array2, Axis};
use serde_json::Value;

const CASES: &[&str] = &[
    "breast_cancer_k10_distance_euclidean",
    "breast_cancer_k5_uniform_minkowski",
    "california_housing_k5_uniform_minkowski",
    "california_housing_k8_distance_euclidean",
    "csv_numeric_regression_k20_distance_manhattan",
    "csv_numeric_regression_k5_uniform_minkowski",
    "csv_string_target_k15_distance_euclidean",
    "csv_string_target_k5_uniform_minkowski",
    "diabetes_k10_distance_manhattan",
    "diabetes_k5_uniform_minkowski",
    "iris_k1_uniform_manhattan",
    "iris_k20_distance_manhattan",
    "iris_k5_distance_euclidean",
    "iris_k5_uniform_minkowski",
    "iris_k5_uniform_minkowski_test02",
    "titanic_k5_uniform_minkowski",
    "wine_k5_uniform_minkowski",
    "wine_k7_distance_manhattan",
];

fn parity(rel: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/parity")
        .join(rel);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

struct Case {
    fx: Value,
    ds: Dataset,
    knn: Knn,
    x_train: Array2<f64>,
    x_test: Array2<f64>,
    test_size: f64,
}

fn load(name: &str) -> Case {
    let fx = parity(&format!("knn/{name}.json"));
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
    let config = KnnConfig {
        n_neighbors: input["n_neighbors"].as_u64().unwrap() as u32,
        weights: match input["weights"].as_str().unwrap() {
            "distance" => KnnWeights::Distance,
            _ => KnnWeights::Uniform,
        },
        metric: match input["metric"].as_str().unwrap() {
            "manhattan" => KnnMetric::Manhattan,
            "euclidean" => KnnMetric::Euclidean,
            _ => KnnMetric::Minkowski,
        },
    };
    let x_train = ds.features.select(Axis(0), &train);
    let x_test = ds.features.select(Axis(0), &test);
    let knn = knn::fit(
        config,
        x_train.view(),
        &ds.target.select(&train),
        &TrainControl::none(),
    )
    .unwrap();
    Case {
        test_size: input["test_size"].as_f64().unwrap(),
        fx,
        ds,
        knn,
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

/// The rows the fixture evaluated: train + test, or the first and last 1500.
fn onnx_rows(c: &Case) -> Array2<f64> {
    let mut all = c.x_train.clone();
    all.append(Axis(0), c.x_test.view()).unwrap();
    let n = all.nrows();
    if n <= 3000 {
        all
    } else {
        let idx: Vec<usize> = (0..1500).chain(n - 1500..n).collect();
        all.select(Axis(0), &idx)
    }
}

#[test]
fn knn_onnx_graph_matches_the_proposal() {
    let mut max_rel = 0.0_f64;
    for name in CASES {
        let c = load(name);
        let e = &c.fx["expected"]["onnx_proposal_D57"];
        let cv = class_values(&c.ds);
        let m = mltutor_export::onnx_model(&c.knn, cv.as_deref()).unwrap();
        assert_eq!(mltutor_export::onnx_file_name(&c.knn), "knn.onnx");
        assert_eq!(m.ir_version, e["ir_version"].as_i64().unwrap());
        assert_eq!(m.opset_import.len(), 1);
        assert_eq!(m.opset_import[0].version, e["opset"].as_i64().unwrap());
        let g = m.graph.as_ref().unwrap();
        let want = e["nodes"].as_array().unwrap();
        assert_eq!(g.node.len(), want.len(), "{name}: nodes");
        for (n, w) in g.node.iter().zip(want) {
            assert_eq!(n.op_type, w["op_type"].as_str().unwrap(), "{name}");
            assert_eq!(n.name, w["name"].as_str().unwrap(), "{name}");
            let strs = |v: &Value| -> Vec<String> {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|s| s.as_str().unwrap().to_owned())
                    .collect()
            };
            assert_eq!(n.input, strs(&w["inputs"]), "{name}: {}", n.name);
            assert_eq!(n.output, strs(&w["outputs"]), "{name}: {}", n.name);
        }
        // Unique node names, also inside the Scan body (onnxruntime rejects repeats).
        let mut names: Vec<&str> = g.node.iter().map(|n| n.name.as_str()).collect();
        for n in &g.node {
            for a in &n.attribute {
                if let Some(body) = &a.g {
                    names.extend(body.node.iter().map(|n| n.name.as_str()));
                }
            }
        }
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "{name}: repeated node names");
        // Round trip and the evaluation of the fixture's head rows.
        let m = onnx::Model::from_bytes(&m.to_bytes()).unwrap();
        let head: Vec<Vec<f32>> = e["head_input_float32"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| nums(r).iter().map(|&v| v as f32).collect())
            .collect();
        let (labels, scores) = onnx::evaluate(&m, &head).unwrap();
        let rel = |a: f32, b: f64| (f64::from(a) - b).abs() / b.abs().max(1.0);
        if let Some(lab) = e["head"].get("label") {
            let want: Vec<i64> = lab
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            assert_eq!(labels, want, "{name}: head labels");
            for (row, w) in scores
                .iter()
                .zip(e["head"]["probabilities"].as_array().unwrap())
            {
                for (a, b) in row.iter().zip(nums(w)) {
                    assert!(rel(*a, b) <= 1e-6, "{name}: proba {a} vs {b}");
                    max_rel = max_rel.max(rel(*a, b));
                }
            }
        } else {
            for (row, b) in scores.iter().zip(nums(&e["head"]["variable"])) {
                assert!(rel(row[0], b) <= 1e-6, "{name}: value {} vs {b}", row[0]);
                max_rel = max_rel.max(rel(row[0], b));
            }
        }
        // Every compared row: the graph agrees with MLTutor on the float32 inputs
        // except at ties of the K-th neighbour (as many as the fixture allows).
        let rows = onnx_rows(&c);
        assert_eq!(rows.nrows() as u64, e["rows_compared"].as_u64().unwrap());
        let x32 = f32_rows(&rows);
        let x64 = Array2::from_shape_fn(rows.dim(), |(i, j)| f64::from(x32[i][j]));
        let (labels, scores) = onnx::evaluate(&m, &x32).unwrap();
        match c.knn.predict(x64.view()).unwrap() {
            Prediction::Classes(pred) => {
                let lab = onnx_class_labels(&c);
                let mismatches = pred
                    .iter()
                    .zip(&labels)
                    .filter(|(p, l)| lab[**p] != **l)
                    .count();
                let allowed = e["label_mismatches_vs_sklearn_on_float32_inputs"]
                    .as_u64()
                    .unwrap() as usize;
                assert!(mismatches <= allowed, "{name}: {mismatches} > {allowed}");
            }
            Prediction::Values(v) => {
                let bad = v
                    .iter()
                    .zip(&scores)
                    .filter(|(a, s)| rel(s[0], **a) > 1e-6)
                    .count();
                assert!(
                    bad as u64 <= e["n_rel_diff_gt_1e-6"].as_u64().unwrap(),
                    "{name}: {bad} values"
                );
            }
        }
    }
    println!("ONNX head max rel err {max_rel:e}");
}

/// `classes` label of each class code.
fn onnx_class_labels(c: &Case) -> Vec<i64> {
    let codes: Vec<usize> = (0..c.ds.target.n_classes().unwrap()).collect();
    mltutor_export::onnx_class_labels(&codes, class_values(&c.ds).as_deref())
}

fn close(a: &Value, b: &Value, path: &str, tol: f64) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            let err = (x - y).abs() / y.abs().max(1.0);
            assert!(err <= tol, "{path}: {x} vs {y}");
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: len");
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                close(p, q, &format!("{path}[{i}]"), tol);
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            let kx: Vec<&String> = x.keys().collect();
            let ky: Vec<&String> = y.keys().collect();
            assert_eq!(kx, ky, "{path}: keys");
            for (k, v) in x {
                close(v, &y[k], &format!("{path}.{k}"), tol);
            }
        }
        _ => assert_eq!(a, b, "{path}"),
    }
}

fn sha256_f64(values: impl Iterator<Item = f64>) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

#[test]
fn knn_python_package_matches_the_fixture() {
    for name in CASES {
        let c = load(name);
        let e = &c.fx["expected"];
        let ep = &e["export_package"];
        let cv = class_values(&c.ds);
        // Python names text classes "Clase i" (F1); Rust keeps the real names, so the
        // fixture's names are passed to compare the rest exactly.
        let names: Option<Vec<String>> = e["class_names"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_str().unwrap().to_owned()).collect());
        let input = PackageInput {
            model: &c.knn,
            feature_columns: &c.ds.feature_columns,
            display_names: &c.ds.feature_display,
            class_values: cv.as_deref(),
            class_names: names.as_deref(),
            dataset_label: c.fx["input"]["app_label"].as_str().unwrap(),
            test_size: c.test_size,
            random_state: 42,
            n_train: c.x_train.nrows(),
            n_test: c.x_test.nrows(),
            x_test: c.x_test.view(),
        };
        let pkg = package::python_package(&input).unwrap();
        let fixture = format!("knn/{name}.json");
        mltutor_testkit::assert_template_text(
            &fixture,
            "/expected/export_package/script",
            &pkg.script,
        );
        mltutor_testkit::assert_template_text(
            &fixture,
            "/expected/export_package/leeme",
            &pkg.readme,
        );
        assert_eq!(package::zip_name(&c.knn), ep["zip_name"].as_str().unwrap());
        let mut ours: Value = serde_json::from_str(&pkg.model_json).unwrap();
        let want = &ep["model_json"];
        // Large training sets go to the fixture as a hash.
        for key in ["X_train", "y_train"] {
            if let Some(hash) = want[key].get("sha256_f64") {
                let flat: Vec<f64> = match &ours[key] {
                    Value::Array(rows) => rows
                        .iter()
                        .flat_map(|r| match r {
                            Value::Array(v) => v.iter().map(|x| x.as_f64().unwrap()).collect(),
                            x => vec![x.as_f64().unwrap()],
                        })
                        .collect(),
                    _ => panic!("{name}: {key}"),
                };
                assert_eq!(
                    sha256_f64(flat.into_iter()),
                    hash.as_str().unwrap(),
                    "{name}: {key}"
                );
                ours[key] = want[key].clone();
            }
        }
        // Brute-force Euclidean distances: scikit-learn's BLAS surrogate (1e-9).
        let brute_euclid = c.knn.engine() == knn::SearchEngine::Brute
            && c.knn.effective_metric() == knn::EffectiveMetric::Euclidean;
        let tol = if brute_euclid { 1e-9 } else { 1e-12 };
        close(&ours, want, name, tol);
        assert_eq!(
            pkg.to_zip().unwrap(),
            package::python_package(&input).unwrap().to_zip().unwrap()
        );
    }
}

/// Writes the KNN ONNX files for an external onnxruntime check:
/// `MLTUTOR_ONNX_OUT=dir cargo test -p mltutor-export --test parity_export_knn -- --ignored`
/// (each `knn_<case>.json` has the float32 rows, MLTutor's predictions on them and
/// the labels of the reference evaluation).
#[test]
#[ignore = "writes files for an external onnxruntime check"]
fn write_knn_onnx_for_onnxruntime() {
    let Ok(dir) = std::env::var("MLTUTOR_ONNX_OUT") else {
        return;
    };
    for name in CASES {
        let c = load(name);
        let cv = class_values(&c.ds);
        let m = mltutor_export::onnx_model(&c.knn, cv.as_deref()).unwrap();
        std::fs::write(format!("{dir}/knn_{name}.onnx"), m.to_bytes()).unwrap();
        let rows = onnx_rows(&c);
        let x32 = f32_rows(&rows);
        let x64 = Array2::from_shape_fn(rows.dim(), |(i, j)| f64::from(x32[i][j]));
        let (labels, scores) = onnx::evaluate(&m, &x32).unwrap();
        let (predict, proba): (Vec<f64>, Option<Vec<Vec<f64>>>) =
            match c.knn.predict(x64.view()).unwrap() {
                Prediction::Classes(p) => {
                    let lab = onnx_class_labels(&c);
                    let pr = c.knn.predict_proba(x64.view()).unwrap().unwrap();
                    (
                        p.iter().map(|&k| lab[k] as f64).collect(),
                        Some(pr.rows().into_iter().map(|r| r.to_vec()).collect()),
                    )
                }
                Prediction::Values(v) => (v, None),
            };
        let doc = serde_json::json!({"x": x32, "labels": labels, "scores": scores,
                                     "predict": predict, "proba": proba});
        std::fs::write(format!("{dir}/knn_{name}.json"), doc.to_string()).unwrap();
    }
}
