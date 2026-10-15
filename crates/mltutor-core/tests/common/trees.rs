//! Loading of the `tests/parity/trees/*.json` cases (F3).
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use base64::Engine;
use mltutor_core::csv::read_csv;
use mltutor_core::dataset::{load_builtin, DatasetId, DatasetMetadata};
use mltutor_core::model::TrainControl;
use mltutor_core::tree::{DecisionTree, TreeConfig, TreeCriterion, TreeTask};
use mltutor_core::{Dataset, Target, TaskChoice};
use ndarray::{Array2, Axis};
use serde_json::Value;

use super::fixture;

/// Every trained case (18).
pub const TRAINED: &[&str] = &[
    "breast_cancer_entropy_d10",
    "breast_cancer_gini_d3",
    "california_housing_squared_error_d3",
    "california_housing_squared_error_d8_mss10",
    "csv_numeric_regression_squared_error_d3",
    "csv_string_target_gini_d3",
    "diabetes_absolute_error_d4",
    "diabetes_classification_default",
    "diabetes_friedman_mse_d5_mss10",
    "diabetes_squared_error_d10",
    "diabetes_squared_error_d3",
    "iris_entropy_d5_mss5",
    "iris_gini_d1",
    "iris_gini_d3",
    "iris_regression_mismatch",
    "titanic_gini_d4",
    "wine_entropy_d3_mss10",
    "wine_gini_d4",
];

/// Cases with every view (`export_text`, `plot_tree`, `python_code`…).
pub fn with_views() -> Vec<&'static str> {
    TRAINED
        .iter()
        .copied()
        .filter(|n| {
            !matches!(
                *n,
                "diabetes_classification_default" | "iris_regression_mismatch"
            )
        })
        .collect()
}

pub struct Case {
    pub name: String,
    pub fx: Value,
    pub ds: Dataset,
    pub metadata: Option<DatasetMetadata>,
    pub train: Vec<usize>,
    pub test: Vec<usize>,
    pub config: TreeConfig,
}

pub fn criterion(s: &str) -> TreeCriterion {
    match s {
        "gini" => TreeCriterion::Gini,
        "entropy" => TreeCriterion::Entropy,
        "squared_error" => TreeCriterion::SquaredError,
        "friedman_mse" => TreeCriterion::FriedmanMse,
        "absolute_error" => TreeCriterion::AbsoluteError,
        other => panic!("criterion {other}"),
    }
}

impl Case {
    pub fn load(name: &str) -> Case {
        let fx = fixture(&format!("trees/{name}.json"));
        let input = &fx["input"];
        let (ds, metadata) = if let Some(rel) = input["dataset"].as_str() {
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
        };
        let idx = |k: &str| -> Vec<usize> {
            fx["expected"][k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        let task = if input["tree_type"] == "Clasificación" {
            TreeTask::Classification
        } else {
            TreeTask::Regression
        };
        let config = TreeConfig::new(
            task,
            criterion(input["criterion"].as_str().unwrap()),
            input["max_depth"].as_u64().unwrap() as u32,
            input["min_samples_split"].as_u64().unwrap() as u32,
        );
        Case {
            name: name.to_owned(),
            train: idx("train_indices"),
            test: idx("test_indices"),
            ds,
            metadata,
            fx,
            config,
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

    pub fn y_train(&self) -> Target {
        self.ds.target.select(&self.train)
    }

    pub fn y_test(&self) -> Target {
        self.ds.target.select(&self.test)
    }

    /// Diabetes with "Clasificación": Python encodes the float target with
    /// `np.unique` (192 classes). Returns the codes and the class values.
    fn unique_codes(values: &[f64]) -> (Vec<usize>, Vec<f64>) {
        let mut u = values.to_vec();
        u.sort_by(f64::total_cmp);
        u.dedup();
        let codes = values
            .iter()
            .map(|v| u.iter().position(|w| w == v).unwrap())
            .collect();
        (codes, u)
    }

    /// Trains like the app (through [`mltutor_core::tree::fit`]), or with the raw
    /// classifier when Python trains a classifier on a continuous target.
    pub fn fit_on(&self, x: &Array2<f64>) -> DecisionTree {
        let y = self.y_train();
        let ctl = TrainControl::none();
        match (&y, self.config.task) {
            (Target::Values { values }, TreeTask::Classification) => {
                let (codes, _) = Self::unique_codes(values);
                DecisionTree::fit_classifier(self.config, x.view(), &codes, &ctl).unwrap()
            }
            _ => mltutor_core::tree::fit(self.config, x.view(), &y, &ctl).unwrap(),
        }
    }

    pub fn fit(&self) -> DecisionTree {
        self.fit_on(&self.x_train())
    }
}
