//! Loading of the `tests/parity/knn/*.json` cases (F4).
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use base64::Engine;
use mltutor_core::csv::read_csv;
use mltutor_core::dataset::{load_builtin, DatasetId, DatasetMetadata};
use mltutor_core::knn::{self, Knn, KnnConfig, KnnMetric, KnnWeights};
use mltutor_core::model::TrainControl;
use mltutor_core::{Dataset, Target, TaskChoice};
use ndarray::{Array2, Axis};
use serde_json::Value;

use super::fixture;

/// Every trained case (18; `csv_string_target_k20_test05_error` is the error one).
pub const TRAINED: &[&str] = &[
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

pub const ERROR_CASE: &str = "csv_string_target_k20_test05_error";

pub fn weights(s: &str) -> KnnWeights {
    match s {
        "uniform" => KnnWeights::Uniform,
        "distance" => KnnWeights::Distance,
        other => panic!("weights {other}"),
    }
}

pub fn metric(s: &str) -> KnnMetric {
    match s {
        "minkowski" => KnnMetric::Minkowski,
        "euclidean" => KnnMetric::Euclidean,
        "manhattan" => KnnMetric::Manhattan,
        other => panic!("metric {other}"),
    }
}

pub struct Case {
    pub name: String,
    pub fx: Value,
    pub ds: Dataset,
    pub metadata: Option<DatasetMetadata>,
    pub train: Vec<usize>,
    pub test: Vec<usize>,
    pub config: KnnConfig,
    pub test_size: f64,
}

/// The dataset of a fixture `input` (built-in or user CSV).
pub fn dataset(input: &Value) -> (Dataset, Option<DatasetMetadata>) {
    if let Some(rel) = input["dataset"].as_str() {
        let slug = rel
            .trim_start_matches("datasets/")
            .trim_end_matches(".json");
        let id = DatasetId::from_slug(slug).unwrap();
        (load_builtin(id).unwrap(), id.metadata().ok())
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
    }
}

pub fn config_of(input: &Value) -> KnnConfig {
    KnnConfig {
        n_neighbors: input["n_neighbors"].as_u64().unwrap() as u32,
        weights: weights(input["weights"].as_str().unwrap()),
        metric: metric(input["metric"].as_str().unwrap()),
    }
}

impl Case {
    pub fn load(name: &str) -> Case {
        let fx = fixture(&format!("knn/{name}.json"));
        let input = &fx["input"];
        let (ds, metadata) = dataset(input);
        let idx = |k: &str| -> Vec<usize> {
            fx["expected"][k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        Case {
            name: name.to_owned(),
            train: idx("train_indices"),
            test: idx("test_indices"),
            config: config_of(input),
            test_size: input["test_size"].as_f64().unwrap(),
            ds,
            metadata,
            fx,
        }
    }

    pub fn expected(&self) -> &Value {
        &self.fx["expected"]
    }

    pub fn classifier(&self) -> bool {
        self.expected()["estimator"] == "KNeighborsClassifier"
    }

    pub fn x_train(&self) -> Array2<f64> {
        self.ds.features.select(Axis(0), &self.train)
    }

    pub fn x_test(&self) -> Array2<f64> {
        self.ds.features.select(Axis(0), &self.test)
    }

    pub fn y_train(&self) -> Target {
        self.ds.target.select(&self.train)
    }

    pub fn y_test(&self) -> Target {
        self.ds.target.select(&self.test)
    }

    /// Fits like the app (`knn::fit` on the training split).
    pub fn fit(&self) -> Knn {
        knn::fit(
            self.config,
            self.x_train().view(),
            &self.y_train(),
            &TrainControl::none(),
        )
        .unwrap()
    }
}
