//! Loading of the `tests/parity/linear_models/*.json` cases (F2).
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use base64::Engine;
use mltutor_core::csv::read_csv;
use mltutor_core::dataset::{load_builtin, DatasetId, DatasetMetadata};
use mltutor_core::{Dataset, TaskChoice};
use ndarray::{Array2, Axis};
use serde_json::Value;

use super::{fixture, nums};

/// One linear-model case: fixture, dataset and its split (fixture indices).
pub struct Case {
    pub name: String,
    pub fx: Value,
    pub ds: Dataset,
    pub metadata: Option<DatasetMetadata>,
    pub train: Vec<usize>,
    pub test: Vec<usize>,
}

impl Case {
    pub fn load(name: &str) -> Case {
        let fx = fixture(&format!("linear_models/{name}.json"));
        let input = &fx["input"];
        let (ds, metadata) = if let Some(rel) = input["dataset"].as_str() {
            let slug = rel
                .trim_start_matches("datasets/")
                .trim_end_matches(".json");
            let id = DatasetId::from_slug(slug).unwrap();
            (load_builtin(id).unwrap(), Some(id.metadata().unwrap()))
        } else {
            let csv_fx = fixture(input["csv_fixture"].as_str().unwrap());
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(csv_fx["input"]["csv_base64"].as_str().unwrap())
                .unwrap();
            let table = read_csv(&bytes).unwrap();
            let label = input["app_label"].as_str().unwrap();
            let file = label.trim_start_matches("📄 ");
            let ds = table
                .to_dataset(input["target"].as_str().unwrap(), TaskChoice::Auto, file)
                .unwrap();
            (ds, None)
        };
        let idx = |k: &str| -> Vec<usize> {
            fx["expected"][k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        let train = idx("train_indices");
        let test = idx("test_indices");
        Case {
            name: name.to_owned(),
            ds,
            metadata,
            train,
            test,
            fx,
        }
    }

    pub fn expected(&self) -> &Value {
        &self.fx["expected"]
    }

    pub fn x_train(&self) -> Array2<f64> {
        self.ds.features.select(Axis(0), &self.train)
    }

    pub fn x_test(&self) -> Array2<f64> {
        self.ds.features.select(Axis(0), &self.test)
    }

    pub fn y_train(&self) -> mltutor_core::Target {
        self.ds.target.select(&self.train)
    }

    pub fn y_test(&self) -> mltutor_core::Target {
        self.ds.target.select(&self.test)
    }

    /// `expected.<key>` as a matrix (list of rows).
    pub fn matrix(&self, v: &Value) -> Array2<f64> {
        let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(nums).collect();
        let n = rows.len();
        let m = rows.first().map_or(0, Vec::len);
        Array2::from_shape_vec((n, m), rows.into_iter().flatten().collect()).unwrap()
    }
}

/// A JSON matrix (list of rows) or vector as rows.
pub fn matrix(v: &Value) -> Array2<f64> {
    let arr = v.as_array().unwrap();
    if arr.first().is_some_and(Value::is_array) {
        let rows: Vec<Vec<f64>> = arr.iter().map(nums).collect();
        let n = rows.len();
        let m = rows.first().map_or(0, Vec::len);
        Array2::from_shape_vec((n, m), rows.into_iter().flatten().collect()).unwrap()
    } else {
        let v = nums(v);
        Array2::from_shape_vec((1, v.len()), v).unwrap()
    }
}

/// Walks a rendered view (`expected.views.<name>`) and returns every node, depth first.
pub fn nodes(view: &Value) -> Vec<&Value> {
    let mut out = Vec::new();
    fn walk<'a>(v: &'a Value, out: &mut Vec<&'a Value>) {
        out.push(v);
        if let Some(children) = v["children"].as_array() {
            for c in children {
                walk(c, out);
            }
        }
    }
    walk(view, &mut out);
    out
}

/// Every node of a kind.
pub fn nodes_of<'a>(view: &'a Value, kind: &str) -> Vec<&'a Value> {
    nodes(view)
        .into_iter()
        .filter(|n| n["kind"].as_str() == Some(kind))
        .collect()
}

/// Every collection of a type in the figures of a view.
pub fn collections<'a>(view: &'a Value, ty: &str) -> Vec<&'a Value> {
    let mut out = Vec::new();
    for img in nodes_of(view, "image") {
        for ax in img["figure"]["axes"].as_array().unwrap() {
            if let Some(cs) = ax["collections"].as_array() {
                out.extend(cs.iter().filter(|c| c["type"].as_str() == Some(ty)));
            }
        }
    }
    out
}

/// Every axes of the figures of a view.
pub fn axes(view: &Value) -> Vec<&Value> {
    let mut out = Vec::new();
    for img in nodes_of(view, "image") {
        out.extend(img["figure"]["axes"].as_array().unwrap().iter());
    }
    out
}

/// Notices' texts.
pub fn notice_texts(view: &Value) -> Vec<String> {
    nodes_of(view, "notice")
        .iter()
        .map(|n| n["text"].as_str().unwrap().to_owned())
        .collect()
}

/// Names of the classification cases (logistic) and regression cases (linear).
pub const LOGISTIC: [&str; 4] = [
    "iris_logistic",
    "breast_cancer_logistic",
    "titanic_logistic",
    "csv_string_target_logistic",
];
pub const LINEAR: [&str; 3] = [
    "diabetes_linear",
    "california_housing_linear",
    "csv_numeric_regression_linear",
];
