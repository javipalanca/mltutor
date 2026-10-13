//! Datasets: the 12 built-in ones (embedded) and the in-memory [`Dataset`] type
//! that every lesson consumes.

pub mod catalog;
pub mod metadata;
pub mod moons;
pub mod table;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

pub use catalog::{list_builtin, load_builtin, DatasetEntry, DatasetId};
pub use metadata::{DatasetMetadata, FeatureKind, FeatureMetadata};
pub use moons::{make_moons, moons_dataset};

/// Kind of learning task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Classification,
    Regression,
}

impl TaskType {
    /// Spanish label used by the Python app ("Clasificación" / "Regresión").
    pub fn label_es(self) -> &'static str {
        match self {
            TaskType::Classification => "Clasificación",
            TaskType::Regression => "Regresión",
        }
    }
}

/// The "Tipo de tarea:" option of the CSV page (`auto` is kept as the user chose it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskChoice {
    Auto,
    Classification,
    Regression,
}

/// Storage type of a column, named after the pandas dtype it mirrors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnKind {
    /// `int64`.
    Int,
    /// `float64` (also integers with missing values, as in pandas).
    Float,
    /// `bool`: excluded from `describe()` like in pandas.
    Bool,
    /// `object` (text).
    Text,
}

impl ColumnKind {
    /// pandas dtype name: `int64`, `float64`, `bool` or `object`.
    pub fn dtype_name(self) -> &'static str {
        match self {
            ColumnKind::Int => "int64",
            ColumnKind::Float => "float64",
            ColumnKind::Bool => "bool",
            ColumnKind::Text => "object",
        }
    }

    /// Whether pandas treats it as numeric in `describe()`.
    pub fn is_numeric(self) -> bool {
        matches!(self, ColumnKind::Int | ColumnKind::Float)
    }
}

/// A text column shown in the exploration but never used as a feature (D8: `name`
/// in MPG).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelColumn {
    pub name: String,
    pub values: Vec<String>,
}

/// The target variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Target {
    /// Classification: `codes[i]` indexes `class_names` (0..k, sorted like
    /// `np.unique`). `class_values[k]` is the numeric value Python keeps in `y` for
    /// class `k` (the code itself for built-in datasets and text targets; the
    /// original number for numeric CSV targets, e.g. 3 and 6).
    Classes {
        codes: Vec<u32>,
        class_names: Vec<String>,
        class_values: Vec<f64>,
    },
    /// Regression: one value per row (NaN allowed in CSV data).
    Values { values: Vec<f64> },
}

/// A dataset ready for exploration and training. Rows are aligned by position
/// (the Python Penguins misalignment is not reproduced).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    /// Selector label: "🌸 Iris - Clasificación de flores" or "📄 {file}".
    pub label: String,
    /// `n_rows × n_features` feature matrix (`X.to_numpy(float64)`; bool = 0/1,
    /// encoded text = codes, NaN = missing).
    pub features: Array2<f64>,
    /// Original column names (`X.columns`): correlation labels and code samples.
    pub feature_columns: Vec<String>,
    /// Friendly names (`feature_names`): preview, describe and pairplot.
    pub feature_display: Vec<String>,
    /// pandas dtype of each feature column.
    pub feature_kinds: Vec<ColumnKind>,
    /// Text columns shown in the exploration only (D8).
    pub label_columns: Vec<LabelColumn>,
    /// Name of the target column in the source (`species`, `price`…). The UI
    /// always shows it as `target`.
    pub target_column: String,
    /// Target values.
    pub target: Target,
    /// `dataset_info["description"]` (or the CSV's "Dataset personalizado: …").
    pub description: String,
}

impl Dataset {
    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.features.nrows()
    }

    /// Number of feature columns (label columns excluded).
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    /// Classification or regression.
    pub fn task(&self) -> TaskType {
        match self.target {
            Target::Classes { .. } => TaskType::Classification,
            Target::Values { .. } => TaskType::Regression,
        }
    }

    /// Class names, for classification.
    pub fn class_names(&self) -> Option<&[String]> {
        match &self.target {
            Target::Classes { class_names, .. } => Some(class_names),
            Target::Values { .. } => None,
        }
    }

    /// The target as Python's numeric `y` (class values for classification).
    pub fn target_numeric(&self) -> Vec<f64> {
        match &self.target {
            Target::Classes {
                codes,
                class_values,
                ..
            } => codes
                .iter()
                .map(|&c| class_values.get(c as usize).copied().unwrap_or(f64::NAN))
                .collect(),
            Target::Values { values } => values.clone(),
        }
    }

    /// The target as shown in the exploration: the class name per row for
    /// classification, `None` for regression.
    pub fn target_labels(&self) -> Option<Vec<&str>> {
        match &self.target {
            Target::Classes {
                codes, class_names, ..
            } => Some(
                codes
                    .iter()
                    .map(|&c| class_names.get(c as usize).map_or("", String::as_str))
                    .collect(),
            ),
            Target::Values { .. } => None,
        }
    }

    /// Index of a feature by its friendly name.
    pub fn feature_index_by_display(&self, name: &str) -> Option<usize> {
        self.feature_display.iter().position(|n| n == name)
    }

    /// The figures of the "📊 Información del Dataset" block and the column lists
    /// the UI needs.
    pub fn summary(&self) -> DatasetSummary {
        DatasetSummary {
            label: self.label.clone(),
            task: self.task(),
            n_rows: self.n_rows(),
            n_features: self.n_features(),
            target_column: "target".to_owned(),
            feature_columns: self.feature_columns.clone(),
            feature_display: self.feature_display.clone(),
            label_columns: self.label_columns.iter().map(|c| c.name.clone()).collect(),
            class_names: self.class_names().map(<[String]>::to_vec),
            description: self.description.clone(),
        }
    }
}

/// Summary of a dataset for the metrics of §2.3 (contract `DatasetSummary` without
/// the session reference).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub label: String,
    pub task: TaskType,
    pub n_rows: usize,
    /// The "📊 Columnas" metric is `n_features + 1`.
    pub n_features: usize,
    /// Always `"target"` in the UI.
    pub target_column: String,
    pub feature_columns: Vec<String>,
    pub feature_display: Vec<String>,
    pub label_columns: Vec<String>,
    pub class_names: Option<Vec<String>>,
    pub description: String,
}
