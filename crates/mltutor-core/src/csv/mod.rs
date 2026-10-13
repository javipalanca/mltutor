//! CSV loader of the "📁 Cargar CSV Personalizado" page (fase1-shell-datos.md §3–4).
//!
//! [`read_csv`] reproduces `pd.read_csv` (types `int64`/`float64`/`bool`/`object`,
//! pandas NA strings, duplicate column names) and adds the D10 tolerances:
//! Windows-1252 when the file is not UTF-8, `;` or tab separators, and decimal
//! comma when the separator is `;`. [`RawTable::to_dataset`] reproduces
//! `load_dataset_from_file` (LabelEncoder per text column, task rules, class
//! names, `dataset_info`), except that text classes keep their original names (D10).

mod reader;

use serde::{Deserialize, Serialize};

pub use reader::{read_csv, read_csv_file, MAX_CSV_BYTES};

use crate::dataset::{ColumnKind, Dataset, LabelColumn, Target, TaskChoice, TaskType};
use crate::error::CsvError;
use crate::frame::{Cell, DescribeTable, TablePreview};
use crate::pyfmt::py_float_repr;
use crate::stats::{self, value_counts_by_frequency};

/// Rows of the "👀 Vista previa del dataset" table (`df.head()`).
pub const CSV_PREVIEW_ROWS: usize = 5;
/// Bins of the target histogram (`Series.hist(bins=30)`).
pub const CSV_TARGET_BINS: usize = 30;
/// `value_counts` is listed under "Información de la Variable Objetivo" up to this
/// number of distinct values.
pub const CSV_VALUE_COUNTS_MAX: usize = 10;

/// Something the reader did differently from pandas, to tell the student (D10).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CsvNotice {
    /// The file was not UTF-8 and was read as Windows-1252.
    Windows1252,
    /// The separator is not a comma.
    Separator { separator: char },
    /// These columns used a decimal comma (`1,5`), accepted because the separator is `;`.
    DecimalComma { columns: Vec<String> },
}

impl CsvNotice {
    /// Text for a `Notice` in the UI.
    pub fn message(&self) -> String {
        match self {
            CsvNotice::Windows1252 => "Se ha leído el archivo como Windows-1252/Latin-1".to_owned(),
            CsvNotice::Separator { separator } => {
                let name = match separator {
                    ';' => "punto y coma (;)".to_owned(),
                    '\t' => "tabulador".to_owned(),
                    other => format!("'{other}'"),
                };
                format!("Se ha detectado el separador {name}")
            }
            CsvNotice::DecimalComma { columns } => format!(
                "Se ha interpretado la coma como separador decimal en: {}",
                columns.join(", ")
            ),
        }
    }
}

/// Values of one column, typed like pandas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "values", rename_all = "snake_case")]
pub enum ColumnData {
    Int(Vec<i64>),
    /// NaN = missing.
    Float(Vec<f64>),
    Bool(Vec<bool>),
    /// `None` = missing.
    Text(Vec<Option<String>>),
}

impl ColumnData {
    /// The pandas dtype.
    pub fn kind(&self) -> ColumnKind {
        match self {
            ColumnData::Int(_) => ColumnKind::Int,
            ColumnData::Float(_) => ColumnKind::Float,
            ColumnData::Bool(_) => ColumnKind::Bool,
            ColumnData::Text(_) => ColumnKind::Text,
        }
    }

    /// Number of values.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int(v) => v.len(),
            ColumnData::Float(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::Text(v) => v.len(),
        }
    }

    /// Whether the column has no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Values as `f64` for numeric and bool columns (`to_numpy(float64)`); `None`
    /// for text.
    pub fn to_f64(&self) -> Option<Vec<f64>> {
        match self {
            ColumnData::Int(v) => Some(v.iter().map(|&x| x as f64).collect()),
            ColumnData::Float(v) => Some(v.clone()),
            ColumnData::Bool(v) => Some(v.iter().map(|&b| f64::from(u8::from(b))).collect()),
            ColumnData::Text(_) => None,
        }
    }

    /// Python `str()` of each value; `None` for missing values.
    pub fn labels(&self) -> Vec<Option<String>> {
        match self {
            ColumnData::Int(v) => v.iter().map(|x| Some(x.to_string())).collect(),
            ColumnData::Float(v) => v
                .iter()
                .map(|&x| (!x.is_nan()).then(|| py_float_repr(x)))
                .collect(),
            ColumnData::Bool(v) => v.iter().map(|&b| Some(py_bool(b).to_owned())).collect(),
            ColumnData::Text(v) => v.clone(),
        }
    }

    /// Number of missing values (`isnull().sum()`).
    pub fn missing(&self) -> usize {
        match self {
            ColumnData::Float(v) => v.iter().filter(|x| x.is_nan()).count(),
            ColumnData::Text(v) => v.iter().filter(|x| x.is_none()).count(),
            ColumnData::Int(_) | ColumnData::Bool(_) => 0,
        }
    }

    /// Distinct non-missing values (`nunique()`).
    pub fn nunique(&self) -> usize {
        let labels = self.labels();
        let set: std::collections::HashSet<&str> =
            labels.iter().flatten().map(String::as_str).collect();
        set.len()
    }

    fn cell(&self, row: usize) -> Cell {
        match self {
            ColumnData::Int(v) => v
                .get(row)
                .map_or(Cell::Missing, |&x| Cell::Number { v: x as f64 }),
            ColumnData::Float(v) => v.get(row).map_or(Cell::Missing, |&x| Cell::number(x)),
            ColumnData::Bool(v) => v.get(row).map_or(Cell::Missing, |&b| Cell::Bool { v: b }),
            ColumnData::Text(v) => match v.get(row) {
                Some(Some(s)) => Cell::Text { v: s.clone() },
                _ => Cell::Missing,
            },
        }
    }
}

fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

/// One parsed column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawColumn {
    pub name: String,
    pub data: ColumnData,
}

/// Description of a column for the UI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsvColumnInfo {
    pub name: String,
    /// `int64`, `float64`, `bool` or `object`.
    pub dtype: String,
    pub missing: usize,
    pub nunique: usize,
}

/// "🎯 Información de la Variable Objetivo".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetInfo {
    pub column: String,
    pub nunique: usize,
    pub dtype: String,
    /// By frequency, only when `nunique <= 10`.
    pub value_counts: Option<Vec<(String, u32)>>,
    /// Suggestion of the "Tipo de tarea:" selector ([`suggest_task`]).
    pub suggested_task: TaskType,
}

/// Everything the page shows right after reading a file (contract `CsvPreview`
/// without the session fields `draft_id` and `file_name`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CsvPreview {
    pub n_rows: usize,
    pub n_cols: usize,
    /// "Valores faltantes" (`isnull().sum().sum()`).
    pub missing_values: usize,
    pub columns: Vec<CsvColumnInfo>,
    /// `df.head()` (5 rows).
    pub head: TablePreview,
    /// The last column.
    pub default_target: String,
    /// [`RawTable::target_info`] of `default_target`.
    pub target: TargetInfo,
    pub notices: Vec<CsvNotice>,
}

/// "🎯 Distribución de la Variable Objetivo" of the loaded CSV.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CsvTargetStats {
    /// Table and bars, most frequent first.
    Classes { value_counts: Vec<(String, u32)> },
    /// Statistics and a 30-bin histogram.
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

/// A parsed CSV file, before choosing the target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawTable {
    pub columns: Vec<RawColumn>,
    pub n_rows: usize,
    /// Detected separator.
    pub separator: char,
    pub notices: Vec<CsvNotice>,
}

/// Task suggested by the CSV selector (`create_dataset_selector`):
/// Classification if `nunique <= 20` and `nunique < 0.1 · n_rows`.
pub fn suggest_task(nunique: usize, n_rows: usize) -> TaskType {
    if nunique <= 20 && (nunique as f64) < n_rows as f64 * 0.1 {
        TaskType::Classification
    } else {
        TaskType::Regression
    }
}

/// The `"auto"` rule of `load_dataset_from_file`: Classification if
/// `len(y.unique()) < max(5, 0.05 · n_rows)` (NaN counts as one value). It is not
/// the same rule as [`suggest_task`]; both are kept, as in Python.
pub fn auto_task(n_unique_with_nan: usize, n_rows: usize) -> TaskType {
    if (n_unique_with_nan as f64) < 5.0_f64.max(n_rows as f64 * 0.05) {
        TaskType::Classification
    } else {
        TaskType::Regression
    }
}

/// `LabelEncoder().fit_transform(col.astype(str))`: missing values become `"nan"`,
/// codes follow the sorted distinct strings. Returns `(codes, classes)`.
fn label_encode(values: &[Option<String>]) -> (Vec<u32>, Vec<String>) {
    let as_str: Vec<&str> = values
        .iter()
        .map(|v| v.as_deref().unwrap_or("nan"))
        .collect();
    let mut classes: Vec<&str> = as_str.clone();
    classes.sort_unstable();
    classes.dedup();
    let codes = as_str
        .iter()
        .map(|s| classes.binary_search(s).map_or(0, |i| i as u32))
        .collect();
    (codes, classes.into_iter().map(str::to_owned).collect())
}

/// Sorted distinct values of a numeric slice, NaN last (as one value).
fn sorted_unique(values: &[f64]) -> Vec<f64> {
    let mut u: Vec<f64> = values.to_vec();
    u.sort_by(f64::total_cmp);
    u.dedup_by(|a, b| a.total_cmp(b).is_eq() || (a.is_nan() && b.is_nan()));
    // total_cmp puts negative NaN first; move every NaN to the end.
    let (mut finite, nan): (Vec<f64>, Vec<f64>) = u.into_iter().partition(|v| !v.is_nan());
    if !nan.is_empty() {
        finite.push(f64::NAN);
    }
    finite
}

fn class_index(classes: &[f64], v: f64) -> u32 {
    let pos = if v.is_nan() {
        classes.iter().position(|c| c.is_nan())
    } else {
        classes.iter().position(|&c| c == v)
    };
    pos.map_or(0, |i| i as u32)
}

impl RawTable {
    /// Number of columns.
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Column by name.
    pub fn column(&self, name: &str) -> Option<&RawColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    fn column_or_err(&self, name: &str) -> Result<&RawColumn, CsvError> {
        self.column(name).ok_or_else(|| CsvError::UnknownColumn {
            column: name.to_owned(),
        })
    }

    /// Total missing values.
    pub fn missing_values(&self) -> usize {
        self.columns.iter().map(|c| c.data.missing()).sum()
    }

    /// The last column (default target).
    pub fn default_target(&self) -> &str {
        self.columns.last().map_or("", |c| c.name.as_str())
    }

    /// Type, missing values and distinct values per column.
    pub fn column_info(&self) -> Vec<CsvColumnInfo> {
        self.columns
            .iter()
            .map(|c| CsvColumnInfo {
                name: c.name.clone(),
                dtype: c.data.kind().dtype_name().to_owned(),
                missing: c.data.missing(),
                nunique: c.data.nunique(),
            })
            .collect()
    }

    /// `df.head(rows)`.
    pub fn head(&self, rows: usize) -> TablePreview {
        let n = rows.min(self.n_rows);
        TablePreview {
            columns: self.columns.iter().map(|c| c.name.clone()).collect(),
            index: (0..n).map(|i| i.to_string()).collect(),
            rows: (0..n)
                .map(|r| self.columns.iter().map(|c| c.data.cell(r)).collect())
                .collect(),
            total_rows: self.n_rows,
        }
    }

    /// `df.describe()`: the int and float columns (bool and text excluded).
    pub fn describe(&self) -> DescribeTable {
        let numeric: Vec<(String, Vec<f64>)> = self
            .columns
            .iter()
            .filter(|c| c.data.kind().is_numeric())
            .filter_map(|c| c.data.to_f64().map(|v| (c.name.clone(), v)))
            .collect();
        let excluded = self
            .columns
            .iter()
            .filter(|c| !c.data.kind().is_numeric())
            .map(|c| c.name.clone())
            .collect();
        DescribeTable::from_columns(
            numeric.iter().map(|(n, v)| (n.clone(), v.as_slice())),
            excluded,
        )
    }

    /// "🎯 Información de la Variable Objetivo" for `target`.
    pub fn target_info(&self, target: &str) -> Result<TargetInfo, CsvError> {
        let col = self.column_or_err(target)?;
        let nunique = col.data.nunique();
        let value_counts = (nunique <= CSV_VALUE_COUNTS_MAX).then(|| {
            let labels = col.data.labels();
            value_counts_by_frequency(labels.iter().flatten().map(String::as_str))
        });
        Ok(TargetInfo {
            column: col.name.clone(),
            nunique,
            dtype: col.data.kind().dtype_name().to_owned(),
            value_counts,
            suggested_task: suggest_task(nunique, self.n_rows),
        })
    }

    /// The preview shown after reading the file, with the last column as target.
    pub fn preview(&self) -> Result<CsvPreview, CsvError> {
        let default_target = self.default_target().to_owned();
        Ok(CsvPreview {
            n_rows: self.n_rows,
            n_cols: self.n_cols(),
            missing_values: self.missing_values(),
            columns: self.column_info(),
            head: self.head(CSV_PREVIEW_ROWS),
            target: self.target_info(&default_target)?,
            default_target,
            notices: self.notices.clone(),
        })
    }

    /// The target distribution block. Classification → value counts; otherwise
    /// (Regression or "auto") numeric statistics, as Python does. A text or bool
    /// target always gets value counts (Python fails there and shows a warning).
    pub fn target_stats(&self, target: &str, task: TaskChoice) -> Result<CsvTargetStats, CsvError> {
        let col = self.column_or_err(target)?;
        let numeric = col.data.kind().is_numeric();
        if task == TaskChoice::Classification || !numeric {
            let labels = col.data.labels();
            return Ok(CsvTargetStats::Classes {
                value_counts: value_counts_by_frequency(
                    labels.iter().flatten().map(String::as_str),
                ),
            });
        }
        let values: Vec<f64> = col
            .data
            .to_f64()
            .unwrap_or_default()
            .into_iter()
            .filter(|v| !v.is_nan())
            .collect();
        let d = stats::describe_column(&values);
        let hist = stats::histogram(&values, CSV_TARGET_BINS);
        Ok(CsvTargetStats::Numeric {
            min: d[3],
            max: d[7],
            mean: d[1],
            median: d[5],
            std: d[2],
            hist_edges: hist.edges,
            hist_counts: hist.counts,
        })
    }

    /// `load_dataset_from_file(path, target, task)`: features = the other columns
    /// (text → LabelEncoder codes, bool → 0/1), target encoded if it is text.
    ///
    /// Class names: `"Clase {v}"` for integer targets, `str(v)` for float and bool
    /// targets, and — **unlike Python** (D10) — the original strings for text
    /// targets (Python shows "Clase 0", "Clase 1"…). Codes are the same as Python's.
    pub fn to_dataset(
        &self,
        target: &str,
        task: TaskChoice,
        file_name: &str,
    ) -> Result<Dataset, CsvError> {
        let target_col = self.column_or_err(target)?;
        let n = self.n_rows;

        let mut feature_columns = Vec::new();
        let mut feature_kinds = Vec::new();
        let mut feature_values: Vec<Vec<f64>> = Vec::new();
        for col in self.columns.iter().filter(|c| c.name != target_col.name) {
            feature_columns.push(col.name.clone());
            match &col.data {
                ColumnData::Text(values) => {
                    let (codes, _) = label_encode(values);
                    feature_values.push(codes.into_iter().map(f64::from).collect());
                    feature_kinds.push(ColumnKind::Int);
                }
                other => {
                    feature_values.push(other.to_f64().unwrap_or_default());
                    feature_kinds.push(other.kind());
                }
            }
        }
        let n_features = feature_columns.len();
        let features = ndarray::Array2::from_shape_fn((n, n_features), |(r, c)| {
            feature_values[c].get(r).copied().unwrap_or(f64::NAN)
        });

        // y as Python keeps it: codes for text, the values otherwise.
        let (y, text_classes): (Vec<f64>, Option<Vec<String>>) = match &target_col.data {
            ColumnData::Text(values) => {
                let (codes, classes) = label_encode(values);
                (codes.into_iter().map(f64::from).collect(), Some(classes))
            }
            other => (other.to_f64().unwrap_or_default(), None),
        };
        let unique = sorted_unique(&y);
        let resolved = match task {
            TaskChoice::Auto => auto_task(unique.len(), n),
            TaskChoice::Classification => TaskType::Classification,
            TaskChoice::Regression => TaskType::Regression,
        };

        let target = match resolved {
            TaskType::Regression => Target::Values { values: y },
            TaskType::Classification => {
                let class_names: Vec<String> = match (&text_classes, &target_col.data) {
                    (Some(classes), _) => classes.clone(),
                    (None, ColumnData::Int(_)) => {
                        unique.iter().map(|v| format!("Clase {v}")).collect()
                    }
                    (None, ColumnData::Bool(_)) => unique
                        .iter()
                        .map(|&v| py_bool(v != 0.0).to_owned())
                        .collect(),
                    (None, _) => unique.iter().map(|&v| py_float_repr(v)).collect(),
                };
                let codes = y.iter().map(|&v| class_index(&unique, v)).collect();
                Target::Classes {
                    codes,
                    class_names,
                    class_values: unique,
                }
            }
        };
        let description = match &target {
            Target::Classes { class_names, .. } => format!(
                "Dataset personalizado: {n} muestras, {n_features} características, {} clases",
                class_names.len()
            ),
            Target::Values { .. } => {
                format!(
                    "Dataset personalizado: {n} muestras, {n_features} características, regresión"
                )
            }
        };
        Ok(Dataset {
            label: format!("📄 {file_name}"),
            features,
            feature_display: feature_columns.clone(),
            feature_columns,
            feature_kinds,
            label_columns: Vec::<LabelColumn>::new(),
            target_column: target_col.name.clone(),
            target,
            description,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_encoder_sorts_strings_and_maps_missing_to_nan() {
        let v = vec![
            Some("si".to_owned()),
            None,
            Some("no".to_owned()),
            Some("si".to_owned()),
        ];
        let (codes, classes) = label_encode(&v);
        assert_eq!(classes, vec!["nan", "no", "si"]);
        assert_eq!(codes, vec![2, 0, 1, 2]);
    }

    #[test]
    fn task_rules_differ_as_in_python() {
        // seven_levels_100_rows: 7 levels in 100 rows.
        assert_eq!(suggest_task(7, 100), TaskType::Classification);
        assert_eq!(auto_task(7, 100), TaskType::Regression);
    }

    #[test]
    fn sorted_unique_puts_nan_last() {
        let u = sorted_unique(&[3.0, f64::NAN, 1.0, 3.0, f64::NAN]);
        assert_eq!(u.len(), 3);
        assert_eq!(&u[..2], &[1.0, 3.0]);
        assert!(u[2].is_nan());
    }
}
