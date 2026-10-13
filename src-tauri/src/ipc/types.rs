//! Payloads of the data commands (fase1-shell-datos.md §5.3). snake_case on the wire;
//! tagged enums use `kind`, like `LaunchMode`.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// Built-in datasets, in the order of the Python selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum DatasetId {
    Iris,
    Wine,
    BreastCancer,
    Digits,
    Diabetes,
    Diamonds,
    Mpg,
    Moons,
    Titanic,
    Tips,
    CaliforniaHousing,
    Penguins,
}

impl DatasetId {
    /// Selector order (`dataset_manager.py`).
    #[cfg(test)]
    pub const ALL: [DatasetId; 12] = [
        Self::Iris,
        Self::Wine,
        Self::BreastCancer,
        Self::Digits,
        Self::Diabetes,
        Self::Diamonds,
        Self::Mpg,
        Self::Moons,
        Self::Titanic,
        Self::Tips,
        Self::CaliforniaHousing,
        Self::Penguins,
    ];

    /// File stem used by the fixtures and by `content/datasets/<slug>.toml`.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Iris => "iris",
            Self::Wine => "wine",
            Self::BreastCancer => "breast_cancer",
            Self::Digits => "digits",
            Self::Diabetes => "diabetes",
            Self::Diamonds => "diamonds",
            Self::Mpg => "mpg",
            Self::Moons => "moons",
            Self::Titanic => "titanic",
            Self::Tips => "tips",
            Self::CaliforniaHousing => "california_housing",
            Self::Penguins => "penguins",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum DatasetRef {
    Builtin { id: DatasetId },
    Csv { id: u32 },
}

impl Default for DatasetRef {
    fn default() -> Self {
        Self::Builtin {
            id: DatasetId::Iris,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum TaskType {
    Classification,
    Regression,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum TaskChoice {
    Auto,
    Classification,
    Regression,
}

/// Row-major matrix. `None` stands for NaN (serde_json cannot read NaN back).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct Matrix {
    pub rows: u32,
    pub cols: u32,
    pub values: Vec<Option<f64>>,
}

impl Matrix {
    pub fn from_f64(rows: usize, cols: usize, values: impl IntoIterator<Item = f64>) -> Self {
        Self {
            rows: to_u32(rows),
            cols: to_u32(cols),
            values: values
                .into_iter()
                .map(|v| v.is_finite().then_some(v))
                .collect(),
        }
    }
}

/// Saturating `usize` → `u32` for counts sent over IPC (datasets are far below 4 G rows).
pub fn to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// One option of the dataset selector.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct DatasetEntry {
    pub dataset: DatasetRef,
    /// "🌸 Iris - Clasificación de flores" | "📄 ventas.csv"
    pub label: String,
    /// Markdown line under the selector (`selector_blurb`).
    pub blurb: Option<String>,
    pub task: TaskType,
    /// Moons: shows "Dataset generado aleatoriamente.".
    pub synthetic: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct DatasetSummary {
    pub dataset: DatasetRef,
    pub label: String,
    pub task: TaskType,
    pub n_rows: u32,
    /// The "📊 Columnas" metric is `n_features + 1`.
    pub n_features: u32,
    /// Always "target" in the UI.
    pub target_column: String,
    /// Original `X.columns` (correlation, code).
    pub feature_columns: Vec<String>,
    /// Friendly names (preview, pairplot).
    pub feature_display: Vec<String>,
    /// D8: `["name"]` in MPG.
    pub label_columns: Vec<String>,
    pub class_names: Option<Vec<String>>,
    /// `dataset_info.description` (shown in the training tab).
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum Cell {
    Number { v: f64 },
    Text { v: String },
    Bool { v: bool },
    Missing,
}

impl Cell {
    pub fn number(v: f64) -> Self {
        if v.is_finite() {
            Self::Number { v }
        } else {
            Self::Missing
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct TablePreview {
    pub columns: Vec<String>,
    /// "0".."9"
    pub index: Vec<String>,
    pub rows: Vec<Vec<Cell>>,
    pub total_rows: u32,
}

/// pandas `describe()`.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct DescribeTable {
    pub columns: Vec<String>,
    /// `["count","mean","std","min","25%","50%","75%","max"]`
    pub stats: Vec<String>,
    /// stats × columns
    pub values: Matrix,
    /// Non-numeric columns (bool, text, class target).
    pub excluded_columns: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum TargetDistribution {
    Bar {
        labels: Vec<String>,
        counts: Vec<u32>,
        balance_ratio: f64,
        balanced: bool,
    },
    Histogram {
        edges: Vec<f64>,
        counts: Vec<u32>,
        mean: f64,
        std: f64,
        min: f64,
        max: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct CorrelationMatrix {
    /// `feature_columns` + "target".
    pub labels: Vec<String>,
    /// n × n, symmetric, `None` where NaN.
    pub values: Matrix,
    /// Range of the strict lower triangle.
    pub visible_min: Option<f64>,
    pub visible_max: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum DiagKind {
    Kde,
    Hist,
}

#[derive(Debug, Clone, PartialEq, Deserialize, TS)]
#[ts(export)]
pub struct PairplotRequest {
    pub dataset: DatasetRef,
    /// Friendly names.
    pub features: Vec<String>,
    pub diag: DiagKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum Hue {
    /// Levels in order of appearance (tab10 palette).
    Categorical {
        levels: Vec<String>,
        codes: Vec<u16>,
    },
    /// Regression (D11): continuous colour.
    Continuous {
        values: Vec<f64>,
        min: f64,
        max: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct KdeCurve {
    pub level: String,
    pub n: u32,
    /// `None` (and an empty `density`) when the level is skipped (n < 2 or variance 0).
    pub bandwidth: Option<f64>,
    pub grid_min: f64,
    pub grid_max: f64,
    /// 200 values, already multiplied by the level weight (`common_norm`).
    pub density: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct HistSeries {
    pub level: String,
    pub counts: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum DiagPanel {
    Kde {
        curves: Vec<KdeCurve>,
    },
    Hist {
        edges: Vec<f64>,
        series: Vec<HistSeries>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct PairplotData {
    pub features: Vec<String>,
    pub n_points: u32,
    pub n_total: u32,
    pub sampled: bool,
    /// n_points × k, row-major.
    pub points: Matrix,
    pub hue: Hue,
    /// D8: car name in MPG.
    pub tooltips: Option<Vec<String>>,
    /// One per feature.
    pub diagonal: Vec<DiagPanel>,
    /// Style of §2.6 (more than 2000 rows).
    pub high_quality: bool,
    /// With " (Muestra: n/N puntos)" when sampled.
    pub title: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct CsvColumnInfo {
    pub name: String,
    /// int64 | float64 | bool | object
    pub dtype: String,
    pub missing: u32,
    pub nunique: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct TargetInfo {
    pub column: String,
    pub nunique: u32,
    pub dtype: String,
    /// When nunique ≤ 10, by descending frequency.
    pub value_counts: Option<Vec<(String, u32)>>,
    pub suggested_task: TaskType,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct CsvPreview {
    pub draft_id: u32,
    pub file_name: String,
    pub n_rows: u32,
    pub n_cols: u32,
    pub missing_values: u32,
    pub columns: Vec<CsvColumnInfo>,
    /// 5 rows.
    pub head: TablePreview,
    /// The last column.
    pub default_target: String,
    /// Of `default_target`.
    pub target: TargetInfo,
    /// E.g. detected encoding or separator (D10).
    pub notices: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum CsvTargetStats {
    /// By frequency (table + bars).
    Classes { value_counts: Vec<(String, u32)> },
    /// 30 bins.
    Numeric {
        min: f64,
        max: f64,
        mean: f64,
        median: f64,
        std: f64,
        hist_edges: Vec<f64>,
        hist_counts: Vec<u32>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct CsvLoaded {
    /// Already registered and selected.
    pub dataset: DatasetEntry,
    /// What the user chose ("auto" is shown as is).
    pub task_choice: TaskChoice,
    /// `auto_task` rule of `load_dataset_from_file`.
    pub resolved_task: TaskType,
    pub n_rows: u32,
    pub n_cols: u32,
    pub target: String,
    /// Of the raw CSV.
    pub describe: DescribeTable,
    pub target_stats: CsvTargetStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_ref_uses_kind_tags() {
        let iris = DatasetRef::default();
        assert_eq!(
            serde_json::to_value(iris).unwrap(),
            serde_json::json!({ "kind": "builtin", "id": "iris" })
        );
        let csv: DatasetRef = serde_json::from_str(r#"{"kind":"csv","id":3}"#).unwrap();
        assert_eq!(csv, DatasetRef::Csv { id: 3 });
        let housing: DatasetRef =
            serde_json::from_str(r#"{"kind":"builtin","id":"california_housing"}"#).unwrap();
        assert_eq!(
            housing,
            DatasetRef::Builtin {
                id: DatasetId::CaliforniaHousing
            }
        );
    }

    #[test]
    fn slugs_match_serde_names() {
        for id in DatasetId::ALL {
            assert_eq!(
                serde_json::to_value(id).unwrap(),
                serde_json::Value::String(id.slug().to_owned())
            );
        }
    }

    #[test]
    fn matrix_maps_nan_to_null() {
        let m = Matrix::from_f64(1, 3, [1.0, f64::NAN, f64::INFINITY]);
        assert_eq!(m.values, vec![Some(1.0), None, None]);
        assert_eq!(
            serde_json::to_string(&m).unwrap(),
            r#"{"rows":1,"cols":3,"values":[1.0,null,null]}"#
        );
    }

    #[test]
    fn cells_are_tagged() {
        assert_eq!(
            serde_json::to_value(Cell::number(f64::NAN)).unwrap(),
            serde_json::json!({ "kind": "missing" })
        );
        assert_eq!(
            serde_json::to_value(Cell::Text { v: "a".into() }).unwrap(),
            serde_json::json!({ "kind": "text", "v": "a" })
        );
    }
}
