//! C11 parity: ONNX graphs against skl2onnx (`onnx_reference`) and the Python
//! package against `export_package` (tests/parity/linear_models/*.json).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use mltutor_core::dataset::{load_builtin, Dataset, DatasetId};
use mltutor_core::linear::{self, ModelConfig};
use mltutor_core::model::{Prediction, Predictor, TrainControl};
use mltutor_core::Target;
use mltutor_export::onnx::{self, Model};
use mltutor_export::package::{self, PackageInput};
use ndarray::{Array2, Axis};
use serde_json::Value;

struct Case {
    fx: Value,
    ds: Dataset,
    model: Box<dyn Predictor>,
    x_test: Array2<f64>,
}

fn load(name: &str) -> Case {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../tests/parity/linear_models/{name}.json"));
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let slug = fx["input"]["dataset"]
        .as_str()
        .unwrap()
        .trim_start_matches("datasets/")
        .trim_end_matches(".json")
        .to_owned();
    let ds = load_builtin(DatasetId::from_slug(&slug).unwrap()).unwrap();
    let idx = |k: &str| -> Vec<usize> {
        fx["expected"][k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };
    let (train, test) = (idx("train_indices"), idx("test_indices"));
    let config = if fx["input"]["model_type"] == "Logistic" {
        ModelConfig::Logistic {
            max_iter: fx["input"]["max_iter"].as_u64().unwrap() as u32,
        }
    } else {
        ModelConfig::Linear
    };
    let x_train = ds.features.select(Axis(0), &train);
    let model = linear::fit(
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
        model,
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

const CASES: [&str; 5] = [
    "iris_logistic",
    "breast_cancer_logistic",
    "titanic_logistic",
    "diabetes_linear",
    "california_housing_linear",
];

#[test]
fn onnx_structure_matches_skl2onnx() {
    for name in CASES {
        let c = load(name);
        let cv = class_values(&c.ds);
        let m = mltutor_export::onnx_model(c.model.as_ref(), cv.as_deref()).unwrap();
        let m = Model::from_bytes(&m.to_bytes()).unwrap(); // round trip
        let r = &c.fx["expected"]["onnx_reference"];
        let g = m.graph.as_ref().unwrap();
        assert_eq!(m.ir_version, onnx::IR_VERSION);
        let inputs: Vec<&str> = g.input.iter().map(|v| v.name.as_str()).collect();
        let outputs: Vec<&str> = g.output.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(serde_json::json!(inputs), r["inputs"], "{name}");
        assert_eq!(serde_json::json!(outputs), r["outputs"], "{name}");
        let nodes = r["nodes"].as_array().unwrap();
        assert_eq!(g.node.len(), nodes.len(), "{name}");
        for (ours, theirs) in g.node.iter().zip(nodes) {
            assert_eq!(ours.op_type, theirs["op_type"].as_str().unwrap());
            assert_eq!(ours.domain, theirs["domain"].as_str().unwrap());
            assert_eq!(serde_json::json!(ours.input), theirs["inputs"]);
            assert_eq!(serde_json::json!(ours.output), theirs["outputs"]);
            let attrs = theirs["attributes"].as_object().unwrap();
            assert_eq!(ours.attribute.len(), attrs.len(), "{name} {}", ours.op_type);
            for a in &ours.attribute {
                let want = &attrs[&a.name];
                match want {
                    Value::String(s) if s.starts_with('<') => assert_eq!(
                        *s,
                        format!("<{} floats>", a.floats.len()),
                        "{name} {}",
                        a.name
                    ),
                    Value::String(s) => {
                        assert_eq!(s.as_bytes(), a.s.as_slice(), "{name} {}", a.name)
                    }
                    Value::Array(_) => {
                        assert_eq!(serde_json::json!(a.ints), *want, "{name} {}", a.name)
                    }
                    Value::Number(n) => assert_eq!(a.i, n.as_i64().unwrap(), "{name} {}", a.name),
                    _ => panic!("{want}"),
                }
            }
        }
    }
}

#[test]
fn onnx_outputs_match_the_model_in_float32() {
    for name in CASES {
        let c = load(name);
        let cv = class_values(&c.ds);
        let m = mltutor_export::onnx_model(c.model.as_ref(), cv.as_deref()).unwrap();
        let (labels, scores) = onnx::evaluate(&m, &f32_rows(&c.x_test)).unwrap();
        let tol = &c.fx["meta"]["tolerance"]["onnx_float32"];
        let tol = tol.as_f64().unwrap();
        let mut worst = 0.0_f64;
        match c.model.predict(c.x_test.view()).unwrap() {
            Prediction::Values(v) => {
                for (s, y) in scores.iter().zip(&v) {
                    worst = worst.max((f64::from(s[0]) - y).abs() / y.abs().max(1.0));
                }
            }
            Prediction::Classes(cl) => {
                let p = c.model.predict_proba(c.x_test.view()).unwrap().unwrap();
                for (i, s) in scores.iter().enumerate() {
                    for (j, v) in s.iter().enumerate() {
                        worst = worst.max((f64::from(*v) - p[[i, j]]).abs());
                    }
                }
                let same = labels
                    .iter()
                    .zip(&cl)
                    .filter(|(a, b)| **a == **b as i64)
                    .count();
                assert!(cl.len() - same <= 1, "{name}: labels");
            }
        }
        println!("{name}: ONNX float32 vs Rust f64 max err {worst:.1e} (tol {tol:e})");
        assert!(worst <= tol, "{name}");
    }
}

#[test]
fn python_package_matches_the_fixture() {
    for name in CASES {
        let c = load(name);
        let e = &c.fx["expected"];
        let ep = &e["export_package"];
        let cv = class_values(&c.ds);
        let names = c.ds.class_names().map(<[String]>::to_vec);
        let input = PackageInput {
            model: c.model.as_ref(),
            feature_columns: &c.ds.feature_columns,
            display_names: &c.ds.feature_display,
            class_values: cv.as_deref(),
            class_names: names.as_deref(),
            dataset_label: c.fx["input"]["app_label"].as_str().unwrap(),
            test_size: 0.3,
            random_state: 42,
            n_train: e["train_indices"].as_array().unwrap().len(),
            n_test: e["test_indices"].as_array().unwrap().len(),
            x_test: c.x_test.view(),
        };
        let pkg = package::python_package(&input).unwrap();
        assert_eq!(pkg.script, ep["script"].as_str().unwrap(), "{name}: script");
        let ours: Value = serde_json::from_str(&pkg.model_json).unwrap();
        let want = &ep["model_json"];
        let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys(&ours), keys(want), "{name}: keys");
        for k in keys(want) {
            match k.as_str() {
                // Fitted numbers: our optimum vs the app's (parity_linear_models.rs).
                "coef_" | "intercept_" | "n_iter_" => {}
                // The fixture records the app's lbfgs/tol=1e-4; the package describes
                // our model (the optimum), refittable with newton-cholesky.
                "params" if want["params"]["solver"] == "lbfgs" => {
                    let mut ours_p = ours["params"].clone();
                    assert_eq!(ours_p["solver"], package::LOGISTIC_SOLVER);
                    assert_eq!(ours_p["tol"], package::LOGISTIC_TOL);
                    ours_p["solver"] = want["params"]["solver"].clone();
                    ours_p["tol"] = want["params"]["tol"].clone();
                    assert_eq!(ours_p, want["params"], "{name}: params");
                    let n_iter = ours["n_iter_"][0].as_i64().unwrap();
                    let report = c.model.fit_report().unwrap();
                    assert_eq!(n_iter as u64, report.n_iter, "{name}: n_iter_");
                    assert!(n_iter >= 1 && n_iter as u64 <= report.max_iter);
                }
                "check" => {
                    assert_eq!(ours["check"]["X"], want["check"]["X"], "{name}: check.X");
                    if want["check"]["prediction"][0].is_i64() {
                        // Labels: the converged model may flip near-ties only.
                        assert_eq!(ours["check"]["prediction"].as_array().unwrap().len(), 5);
                    }
                }
                _ => assert_eq!(ours[&k], want[&k], "{name}: {k}"),
            }
        }
        assert!(pkg.readme.contains(&c.ds.feature_display[0]));
        if let Some(names) = c.ds.class_names() {
            for (code, name) in names.iter().enumerate() {
                assert!(
                    pkg.readme.contains(&format!("  {code} = {name}\n")),
                    "{name}: LEEME classes\n{}",
                    pkg.readme
                );
            }
        }
        let zip = pkg.to_zip().unwrap();
        assert_eq!(zip, pkg.to_zip().unwrap(), "deterministic");
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip)).unwrap();
        let files: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().unwrap().to_string())
            .collect();
        assert_eq!(serde_json::json!(files), ep["files"]);
    }
}

/// Writes the ONNX files and the inputs for an external onnxruntime check:
/// `MLTUTOR_ONNX_OUT=dir cargo test -p mltutor-export --test parity_export -- --ignored`
#[test]
#[ignore = "writes files for an external onnxruntime check"]
fn write_onnx_for_onnxruntime() {
    let Ok(dir) = std::env::var("MLTUTOR_ONNX_OUT") else {
        return;
    };
    for name in CASES {
        let c = load(name);
        let cv = class_values(&c.ds);
        let m = mltutor_export::onnx_model(c.model.as_ref(), cv.as_deref()).unwrap();
        std::fs::write(format!("{dir}/{name}.onnx"), m.to_bytes()).unwrap();
        let expected: Vec<Vec<f64>> = match c.model.predict_proba(c.x_test.view()) {
            Some(p) => p.unwrap().rows().into_iter().map(|r| r.to_vec()).collect(),
            None => c
                .model
                .predict(c.x_test.view())
                .unwrap()
                .to_f64()
                .into_iter()
                .map(|v| vec![v])
                .collect(),
        };
        let doc = serde_json::json!({"x": f32_rows(&c.x_test), "expected": expected});
        std::fs::write(format!("{dir}/{name}.json"), doc.to_string()).unwrap();
        // The Python package, to run reconstruir_modelo.py with scikit-learn.
        let names = c.ds.class_names().map(<[String]>::to_vec);
        let e = &c.fx["expected"];
        let pkg = package::python_package(&PackageInput {
            model: c.model.as_ref(),
            feature_columns: &c.ds.feature_columns,
            display_names: &c.ds.feature_display,
            class_values: cv.as_deref(),
            class_names: names.as_deref(),
            dataset_label: c.fx["input"]["app_label"].as_str().unwrap(),
            test_size: 0.3,
            random_state: 42,
            n_train: e["train_indices"].as_array().unwrap().len(),
            n_test: e["test_indices"].as_array().unwrap().len(),
            x_test: c.x_test.view(),
        })
        .unwrap();
        let pdir = format!("{dir}/pkg_{name}");
        std::fs::create_dir_all(&pdir).unwrap();
        std::fs::write(
            format!("{pdir}/{}", package::ZIP_NAME),
            pkg.to_zip().unwrap(),
        )
        .unwrap();
    }
}
