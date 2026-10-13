//! Metadata of the built-in datasets, embedded from `content/datasets/<id>.toml`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::TaskType;
use crate::error::DatasetError;

/// How the prediction form (F2+) presents a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKind {
    Binary,
    Categorical,
    Continuous,
}

/// One `[[features]]` table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureMetadata {
    /// Column name in the CSV (`X.columns`).
    pub column: String,
    /// Friendly name (`feature_names`).
    pub display: String,
    #[serde(default)]
    pub description: Option<String>,
    pub kind: FeatureKind,
    /// Whether `dataset_metadata.py` declared it categorical.
    #[serde(default)]
    pub categorical_in_metadata: bool,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// Code → text of the categories (`"0" = "Mujer"`).
    #[serde(default)]
    pub values: Option<BTreeMap<String, String>>,
    /// Distinct levels of a discrete numeric feature.
    #[serde(default)]
    pub levels: Option<Vec<f64>>,
}

/// The synthetic generator (`[generator]`, Moons).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratorMetadata {
    pub function: String,
    pub n_samples: usize,
    pub noise: f64,
    pub seed: u32,
}

/// The whole `content/datasets/<id>.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub id: String,
    /// Selector label, e.g. "🌸 Iris - Clasificación de flores".
    pub label: String,
    pub emoji: String,
    pub name: String,
    pub task: TaskType,
    pub task_label: String,
    /// Target column name in the CSV.
    pub target: String,
    pub samples: usize,
    /// `dataset_info["description"]`.
    pub description: String,
    /// Markdown line under the selector; empty when Python shows none.
    pub selector_blurb: String,
    pub selector_info: String,
    pub source: String,
    pub fixture: String,
    /// Class names, in code order (classification only).
    #[serde(default)]
    pub classes: Option<Vec<String>>,
    /// Columns shown but not used as features (D8).
    #[serde(default)]
    pub label_columns: Vec<String>,
    #[serde(default)]
    pub generator: Option<GeneratorMetadata>,
    pub features: Vec<FeatureMetadata>,
}

impl DatasetMetadata {
    /// Parses one TOML document.
    pub fn from_toml(dataset: &str, text: &str) -> Result<Self, DatasetError> {
        toml::from_str(text).map_err(|e| DatasetError::Metadata {
            dataset: dataset.to_owned(),
            message: e.to_string(),
        })
    }

    /// `selector_blurb`, or `None` when it is empty (Digits, Diabetes, Diamonds, MPG).
    pub fn blurb(&self) -> Option<&str> {
        let b = self.selector_blurb.trim();
        (!b.is_empty()).then_some(b)
    }

    /// Friendly names in column order.
    pub fn display_names(&self) -> Vec<String> {
        self.features.iter().map(|f| f.display.clone()).collect()
    }

    /// Metadata of a feature by its column name.
    pub fn feature(&self, column: &str) -> Option<&FeatureMetadata> {
        self.features.iter().find(|f| f.column == column)
    }
}
