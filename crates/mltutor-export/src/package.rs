//! "📦 Paquete Python (.zip)" (D1/D28, fase2-regresion.md C11): `modelo.json` with
//! the fitted parameters, `reconstruir_modelo.py` (rebuilds the scikit-learn object
//! without retraining, checks it predicts like MLTutor and saves `modelo.pkl`) and
//! `LEEME.txt`.

use std::io::{Cursor, Write};

use mltutor_core::codegen::{self, CodeTemplate};
use mltutor_core::linear::{LinearRegression, LogisticRegression};
use mltutor_core::model::{Prediction, Predictor};
use ndarray::ArrayView2;
use serde::Serialize;
use serde_json::{json, Value};

use crate::ExportError;

/// File names inside the ZIP.
pub const MODEL_JSON: &str = "modelo.json";
pub const SCRIPT: &str = "reconstruir_modelo.py";
pub const README: &str = "LEEME.txt";
/// Pickle written by the script.
pub const PKL_NAME: &str = "modelo.pkl";
/// Suggested ZIP name (Python app: `modelo_regresion.zip`).
pub const ZIP_NAME: &str = "modelo_regresion.zip";
/// Version of the `modelo.json` schema.
pub const FORMAT_VERSION: u32 = 1;
/// Rows of the test set embedded in `check` (`X_test.head(5)`).
pub const CHECK_ROWS: usize = 5;
/// `solver` written in the logistic `params`: MLTutor's model is the optimum of
/// scikit-learn's objective (Newton's method, see `mltutor_core::linear::logistic`),
/// which `newton-cholesky` also reaches; `lbfgs` with `tol=1e-4` stops elsewhere.
pub const LOGISTIC_SOLVER: &str = "newton-cholesky";
/// `tol` written with [`LOGISTIC_SOLVER`] (the one `reconstruir_modelo.py` and the
/// training script use to refit).
pub const LOGISTIC_TOL: f64 = 1e-8;

/// What the package needs besides the model.
#[derive(Debug, Clone, Copy)]
pub struct PackageInput<'a> {
    pub model: &'a dyn Predictor,
    /// `X.columns` (`feature_names_in`).
    pub feature_columns: &'a [String],
    /// Friendly names.
    pub display_names: &'a [String],
    /// Numeric value of each class code (`classes_` are these values).
    pub class_values: Option<&'a [f64]>,
    pub class_names: Option<&'a [String]>,
    /// Selector label of the training dataset.
    pub dataset_label: &'a str,
    pub test_size: f64,
    pub random_state: u32,
    pub n_train: usize,
    pub n_test: usize,
    /// First test rows (at most [`CHECK_ROWS`] are used).
    pub x_test: ArrayView2<'a, f64>,
}

/// A JSON number that prints like Python's `int` when it is integral.
fn number(v: f64) -> Value {
    if v.fract() == 0.0 && v.abs() < 9.0e15 {
        json!(v as i64)
    } else {
        json!(v)
    }
}

fn rows(x: ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    x.rows().into_iter().map(|r| r.to_vec()).collect()
}

/// The `estimator` of the model (`sklearn.linear_model.*`).
pub fn estimator_name(model: &dyn Predictor) -> Result<&'static str, ExportError> {
    let any = model.as_any();
    if any.is::<LinearRegression>() {
        Ok("sklearn.linear_model.LinearRegression")
    } else if any.is::<LogisticRegression>() {
        Ok("sklearn.linear_model.LogisticRegression")
    } else {
        Err(ExportError::Unsupported)
    }
}

/// `modelo.json` (schema v1).
pub fn model_json(input: &PackageInput<'_>) -> Result<Value, ExportError> {
    let model = input.model;
    let estimator = estimator_name(model)?;
    let coefs = model
        .linear_coefficients()
        .ok_or(ExportError::Unsupported)?;
    let n_check = input.x_test.nrows().min(CHECK_ROWS);
    let x_check = input.x_test.slice(ndarray::s![..n_check, ..]);
    let prediction = model.predict(x_check)?;
    let mut doc = serde_json::Map::new();
    doc.insert("format".into(), json!("mltutor-model"));
    doc.insert("format_version".into(), json!(FORMAT_VERSION));
    doc.insert("estimator".into(), json!(estimator));
    if let Some(log) = model.as_any().downcast_ref::<LogisticRegression>() {
        let p = log.params();
        doc.insert(
            "params".into(),
            json!({
                "C": p.c, "class_weight": null, "dual": false, "fit_intercept": true,
                "intercept_scaling": 1, "l1_ratio": null, "max_iter": p.max_iter,
                "multi_class": "deprecated", "n_jobs": null, "penalty": "l2",
                "random_state": input.random_state, "solver": LOGISTIC_SOLVER,
                "tol": LOGISTIC_TOL,
                "verbose": 0, "warm_start": false
            }),
        );
    } else {
        doc.insert(
            "params".into(),
            json!({"copy_X": true, "fit_intercept": true, "n_jobs": null, "positive": false, "tol": 1e-06}),
        );
    }
    doc.insert("feature_names_in".into(), json!(input.feature_columns));
    doc.insert("display_names".into(), json!(input.display_names));
    doc.insert("n_features_in".into(), json!(model.n_features()));
    let classifier = model.classes().is_some();
    if classifier {
        doc.insert("coef_".into(), json!(rows(coefs.coef.view())));
        doc.insert("intercept_".into(), json!(coefs.intercept));
    } else {
        doc.insert("coef_".into(), json!(coefs.coef.row(0).to_vec()));
        doc.insert(
            "intercept_".into(),
            json!(coefs.intercept.first().copied().unwrap_or(0.0)),
        );
    }
    doc.insert(
        "training".into(),
        json!({
            "dataset": input.dataset_label, "test_size": input.test_size,
            "random_state": input.random_state, "n_train": input.n_train, "n_test": input.n_test
        }),
    );
    let class_value = |code: usize| -> Value {
        input
            .class_values
            .and_then(|v| v.get(code))
            .map_or_else(|| json!(code), |&v| number(v))
    };
    let mut check = serde_json::Map::new();
    check.insert("X".into(), json!(rows(x_check)));
    match &prediction {
        Prediction::Classes(c) => {
            check.insert(
                "prediction".into(),
                Value::Array(c.iter().map(|&c| class_value(c)).collect()),
            );
        }
        Prediction::Values(v) => {
            check.insert("prediction".into(), json!(v));
        }
    }
    if let Some(p) = model.predict_proba(x_check) {
        check.insert("probabilities".into(), json!(rows(p?.view())));
    }
    doc.insert("check".into(), Value::Object(check));
    if let Some(classes) = model.classes() {
        doc.insert(
            "classes_".into(),
            Value::Array(classes.iter().map(|&c| class_value(c)).collect()),
        );
        let names: Vec<String> = classes
            .iter()
            .map(|&c| {
                input
                    .class_names
                    .and_then(|n| n.get(c).cloned())
                    .unwrap_or_else(|| c.to_string())
            })
            .collect();
        doc.insert("class_names".into(), json!(names));
        // `n_iter_` of newton-cholesky: the Newton iterations of our solver (int32,
        // shape (1,), at most max_iter), coherent with `solver` above.
        let n_iter = model.fit_report().map_or(0, |r| r.n_iter);
        let n_iter = i32::try_from(n_iter).unwrap_or(i32::MAX);
        doc.insert("n_iter_".into(), json!([n_iter]));
    }
    Ok(Value::Object(doc))
}

#[derive(Serialize)]
struct ScriptContext<'a> {
    estimator: &'a str,
    dataset: &'a str,
    classifier: bool,
    json_name: &'a str,
    pkl_name: &'a str,
}

/// One entry of `classes` in `paquete_leeme.txt.j2` (I4).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReadmeClass {
    /// The value of `classes_`, what `predict` returns (`3`, `6`, `2.5`…).
    pub value: Value,
    /// Readable name.
    pub name: String,
}

#[derive(Serialize)]
struct ReadmeContext<'a> {
    estimator: &'a str,
    dataset: &'a str,
    feature_names: &'a [String],
    /// I4: `[{value, name}]` in the order of `classes_`; `None` in regression.
    classes: Option<Vec<ReadmeClass>>,
    /// Kept for compatibility with older templates (the names alone).
    class_names: Option<Vec<String>>,
    n_train: usize,
    n_test: usize,
    test_size: f64,
    pkl_name: &'a str,
}

/// The three files of the package, as text.
#[derive(Debug, Clone, PartialEq)]
pub struct PythonPackage {
    pub model_json: String,
    pub script: String,
    pub readme: String,
}

impl PythonPackage {
    /// The ZIP (deflate, fixed timestamps: the same input gives the same bytes).
    pub fn to_zip(&self) -> Result<Vec<u8>, ExportError> {
        python_zip(&[
            (MODEL_JSON, self.model_json.as_bytes()),
            (SCRIPT, self.script.as_bytes()),
            (README, self.readme.as_bytes()),
        ])
    }
}

/// Builds `modelo.json`, `reconstruir_modelo.py` and `LEEME.txt`.
pub fn python_package(input: &PackageInput<'_>) -> Result<PythonPackage, ExportError> {
    let doc = model_json(input)?;
    let estimator = estimator_name(input.model)?;
    let classifier = input.model.classes().is_some();
    let script = codegen::render(
        CodeTemplate::PackageScript,
        &ScriptContext {
            estimator,
            dataset: input.dataset_label,
            classifier,
            json_name: MODEL_JSON,
            pkl_name: PKL_NAME,
        },
    )?;
    let readme = codegen::render(
        CodeTemplate::PackageReadme,
        &ReadmeContext {
            estimator,
            dataset: input.dataset_label,
            feature_names: input.display_names,
            classes: classifier.then(|| readme_classes(&doc)),
            class_names: classifier
                .then(|| readme_classes(&doc).into_iter().map(|c| c.name).collect()),
            n_train: input.n_train,
            n_test: input.n_test,
            test_size: input.test_size,
            pkl_name: PKL_NAME,
        },
    )?;
    let mut model_json = serde_json::to_string_pretty(&doc).map_err(|e| ExportError::Json {
        message: e.to_string(),
    })?;
    model_json.push('\n');
    Ok(PythonPackage {
        model_json,
        script,
        readme,
    })
}

/// `classes_` with their names, from `modelo.json` (so that the LEEME shows
/// exactly what `predict` returns: 3 and 6 in a CSV with those classes, not 0/1).
pub fn readme_classes(doc: &Value) -> Vec<ReadmeClass> {
    let values = doc["classes_"].as_array().cloned().unwrap_or_default();
    let names = doc["class_names"].as_array().cloned().unwrap_or_default();
    values
        .into_iter()
        .enumerate()
        .map(|(i, value)| ReadmeClass {
            name: names
                .get(i)
                .and_then(Value::as_str)
                .map_or_else(|| value.to_string(), str::to_owned),
            value,
        })
        .collect()
}

/// A ZIP with `files` (name, bytes), deflated, with the fixed 1980-01-01 date so
/// that the output is deterministic.
pub fn python_zip(files: &[(&str, &[u8])]) -> Result<Vec<u8>, ExportError> {
    let zip_err = |e: zip::result::ZipError| ExportError::Zip {
        message: e.to_string(),
    };
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .last_modified_time(zip::DateTime::default());
    for (name, bytes) in files {
        writer.start_file(*name, options).map_err(zip_err)?;
        writer.write_all(bytes).map_err(|e| ExportError::Zip {
            message: e.to_string(),
        })?;
    }
    Ok(writer.finish().map_err(zip_err)?.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mltutor_core::model::TrainControl;
    use mltutor_core::{Target, TaskChoice};

    /// Review F2 (high): the LEEME must list what `predict` returns (3 and 6),
    /// and modelo.json must name the solver that reproduces our model.
    #[test]
    fn package_of_csv_with_classes_3_and_6() {
        let mut text = String::from("a,b,y\n");
        for i in 0..30 {
            let y = if i % 3 == 0 { 6 } else { 3 };
            text.push_str(&format!("{},{},{y}\n", i % 7, f64::from(i) * 0.5));
        }
        let ds = mltutor_core::csv::read_csv(text.as_bytes())
            .and_then(|t| t.to_dataset("y", TaskChoice::Classification, "c.csv"))
            .unwrap_or_else(|e| panic!("{e}"));
        let Target::Classes { class_values, .. } = &ds.target else {
            panic!("classification expected")
        };
        let model = mltutor_core::linear::fit(
            mltutor_core::linear::ModelConfig::Logistic { max_iter: 1000 },
            ds.features.view(),
            &ds.target,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let input = PackageInput {
            model: model.as_ref(),
            feature_columns: &ds.feature_columns,
            display_names: &ds.feature_display,
            class_values: Some(class_values),
            class_names: ds.class_names(),
            dataset_label: &ds.label,
            test_size: 0.3,
            random_state: 42,
            n_train: 30,
            n_test: 0,
            x_test: ds.features.view(),
        };
        let pkg = python_package(&input).unwrap_or_else(|e| panic!("{e}"));
        assert!(pkg.readme.contains("  3 = Clase 3\n"), "{}", pkg.readme);
        assert!(pkg.readme.contains("  6 = Clase 6\n"), "{}", pkg.readme);
        let doc: Value = serde_json::from_str(&pkg.model_json).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(doc["classes_"], json!([3, 6]));
        assert_eq!(doc["params"]["solver"], "newton-cholesky");
        assert_eq!(doc["params"]["tol"], json!(1e-8));
        let n_iter = model.fit_report().map(|r| r.n_iter).unwrap_or(0);
        assert_eq!(doc["n_iter_"], json!([n_iter]));
        assert_eq!(
            readme_classes(&doc),
            vec![
                ReadmeClass {
                    value: json!(3),
                    name: "Clase 3".to_owned()
                },
                ReadmeClass {
                    value: json!(6),
                    name: "Clase 6".to_owned()
                }
            ]
        );
    }
}
