//! CSV loader of the "📁 Cargar CSV Personalizado" page (fase1-shell-datos.md §3–4).
//!
//! [`read_csv`] reproduces `pd.read_csv` (types `int64`/`float64`/`bool`/`object`,
//! pandas NA strings, duplicate column names) and adds the D10 tolerances:
//! Windows-1252 when the file is not UTF-8, `;` or tab separators, and decimal
//! comma when the separator is `;`. [`RawTable::to_dataset`] reproduces
//! `load_dataset_from_file` (LabelEncoder per text column, task rules, class
//! names, `dataset_info`), except that text classes keep their original names (D10).

//!
//! **Non-finite values.** `inf`/`-Infinity` are read as `float64`, as pandas does,
//! so dtype, `isnull()` (they are *not* missing) and `nunique()` match pandas. Every
//! statistic and chart computed afterwards — `describe`, target statistics and
//! histograms, correlation, KDE, the continuous hue of the pairplot — ignores them
//! as if they were missing (pandas would give `inf`/NaN, and `np.histogram` raises),
//! because the UI cannot draw them either (`Matrix`/`Cell` show them as missing).
//! [`CsvNotice::NonFinite`] tells the student which columns have them.

mod reader;

use std::collections::HashSet;
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, Ordering};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

pub use reader::{
    read_csv, read_csv_cancellable, read_csv_file, read_csv_file_cancellable, CANCEL_CHECK_ROWS,
    MAX_CSV_BYTES,
};

use crate::dataset::{ColumnKind, Dataset, LabelColumn, Target, TaskChoice, TaskType};
use crate::error::CsvError;
use crate::frame::{Cell, DescribeTable, TablePreview};
use crate::pyfmt::py_float_repr;
use crate::stats::{self, describe_column};

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
    /// These float columns contain ±infinity: kept in the data (as pandas does) but
    /// left out of statistics and charts (module docs).
    NonFinite { columns: Vec<String> },
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
            CsvNotice::NonFinite { columns } => format!(
                "Hay valores infinitos (inf) en: {}. Se conservan en los datos, pero no se \
                 tienen en cuenta en las estadísticas ni en los gráficos",
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

    /// Distinct non-missing values (`nunique()`), hashing the values themselves
    /// (floats by [`float_key`], so `-0.0` and `0.0` are one value, as in pandas).
    pub fn nunique(&self) -> usize {
        match self {
            ColumnData::Int(v) => count_distinct(v.iter().copied()),
            ColumnData::Float(v) => {
                count_distinct(v.iter().filter(|x| !x.is_nan()).map(|&x| float_key(x)))
            }
            ColumnData::Bool(v) => {
                usize::from(v.iter().any(|&b| b)) + usize::from(v.iter().any(|&b| !b))
            }
            ColumnData::Text(v) => count_distinct(v.iter().flatten().map(String::as_str)),
        }
    }

    /// `value_counts()` (missing values dropped): most frequent first, ties in order
    /// of first appearance, keys formatted with Python's `str()`. Counts are taken
    /// over the values, and only the distinct ones are formatted.
    pub fn value_counts_by_frequency(&self) -> Vec<(String, u32)> {
        match self {
            ColumnData::Int(v) => counts_by_frequency(v.iter().copied(), |i| i.to_string()),
            ColumnData::Float(v) => counts_by_frequency(
                v.iter().filter(|x| !x.is_nan()).map(|&x| float_key(x)),
                |&k| py_float_repr(f64::from_bits(k)),
            ),
            ColumnData::Bool(v) => {
                counts_by_frequency(v.iter().copied(), |&b| py_bool(b).to_owned())
            }
            ColumnData::Text(v) => {
                counts_by_frequency(v.iter().flatten().map(String::as_str), |s| (*s).to_owned())
            }
        }
    }

    /// Writes the values as `f64` (text columns: nothing).
    fn write_f64<'a>(&self, out: impl Iterator<Item = &'a mut f64>) {
        match self {
            ColumnData::Int(v) => out.zip(v).for_each(|(o, &x)| *o = x as f64),
            ColumnData::Float(v) => out.zip(v).for_each(|(o, &x)| *o = x),
            ColumnData::Bool(v) => out.zip(v).for_each(|(o, &b)| *o = f64::from(u8::from(b))),
            ColumnData::Text(_) => {}
        }
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

/// Hash key of a float value: its bits, with `-0.0` folded into `0.0` and every
/// NaN into one canonical NaN (pandas' float hash table treats them as equal).
pub fn float_key(v: f64) -> u64 {
    normalize_float(v).to_bits()
}

/// `-0.0` → `0.0`, any NaN → the canonical `f64::NAN`; other values unchanged.
fn normalize_float(v: f64) -> f64 {
    if v.is_nan() {
        f64::NAN
    } else if v == 0.0 {
        0.0
    } else {
        v
    }
}

/// Number of distinct keys. The set grows with the distinct values only:
/// `collect()` would reserve room for every row (`size_hint`), e.g. 144 MiB for a
/// column of 10 M integers with 900 distinct values.
fn count_distinct<K: Hash + Eq>(keys: impl Iterator<Item = K>) -> usize {
    let mut set = HashSet::new();
    for k in keys {
        set.insert(k);
    }
    set.len()
}

/// Counts in order of first appearance, then sorted by count (stable).
fn counts_by_frequency<K: Hash + Eq>(
    keys: impl Iterator<Item = K>,
    label: impl Fn(&K) -> String,
) -> Vec<(String, u32)> {
    let mut counts: IndexMap<K, u32> = IndexMap::new();
    for k in keys {
        *counts.entry(k).or_insert(0) += 1;
    }
    let mut out: Vec<(String, u32)> = counts.iter().map(|(k, &c)| (label(k), c)).collect();
    // `sort_by_key` is stable, so ties stay in order of appearance.
    out.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    out
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

/// `sorted(y.unique())`: sorted distinct values, `-0.0` and `0.0` as one value
/// (`0.0`) and NaN last as one value. With these normalizations `total_cmp` is a
/// total order in which the canonical NaN sorts after `+inf`.
fn sorted_unique(values: &[f64]) -> Vec<f64> {
    let mut u: Vec<f64> = values.iter().map(|&v| normalize_float(v)).collect();
    u.sort_unstable_by(f64::total_cmp);
    u.dedup_by(|a, b| a.total_cmp(b).is_eq());
    // It becomes `class_values`: do not keep the capacity of the whole column.
    u.shrink_to_fit();
    u
}

/// Position of `v` in `classes` (the output of [`sorted_unique`]) by binary
/// search: O(log k) per row instead of a linear scan.
fn class_index(classes: &[f64], v: f64) -> u32 {
    let v = normalize_float(v);
    classes
        .binary_search_by(|c| c.total_cmp(&v))
        .map_or(0, |i| u32::try_from(i).unwrap_or(u32::MAX))
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

    /// `df.describe()`: the int and float columns (bool and text excluded), one
    /// column at a time so that only one temporary copy is alive.
    pub fn describe(&self) -> DescribeTable {
        let mut names = Vec::new();
        let mut stats = Vec::new();
        let mut excluded = Vec::new();
        for c in &self.columns {
            match &c.data {
                ColumnData::Float(v) => stats.push(describe_column(v)),
                ColumnData::Int(v) => {
                    let values: Vec<f64> = v.iter().map(|&x| x as f64).collect();
                    stats.push(describe_column(&values));
                }
                ColumnData::Bool(_) | ColumnData::Text(_) => {
                    excluded.push(c.name.clone());
                    continue;
                }
            }
            names.push(c.name.clone());
        }
        DescribeTable::from_stats(names, &stats, excluded)
    }

    /// "🎯 Información de la Variable Objetivo" for `target`.
    pub fn target_info(&self, target: &str) -> Result<TargetInfo, CsvError> {
        let col = self.column_or_err(target)?;
        let nunique = col.data.nunique();
        let value_counts =
            (nunique <= CSV_VALUE_COUNTS_MAX).then(|| col.data.value_counts_by_frequency());
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
            return Ok(CsvTargetStats::Classes {
                value_counts: col.data.value_counts_by_frequency(),
            });
        }
        // Missing and ±inf values are left out (module docs).
        let mut values: Vec<f64> = col.data.to_f64().unwrap_or_default();
        values.retain(|v| v.is_finite());
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
        self.to_dataset_cancellable(target, task, file_name, &AtomicBool::new(false))
    }

    /// [`RawTable::to_dataset`] with cooperative cancellation: `cancel` is checked
    /// before each column, and [`CsvError::Cancelled`] is returned once it is set.
    pub fn to_dataset_cancellable(
        &self,
        target: &str,
        task: TaskChoice,
        file_name: &str,
        cancel: &AtomicBool,
    ) -> Result<Dataset, CsvError> {
        let check = || {
            if cancel.load(Ordering::Relaxed) {
                Err(CsvError::Cancelled)
            } else {
                Ok(())
            }
        };
        let target_col = self.column_or_err(target)?;
        let n = self.n_rows;

        // Each column is written straight into the matrix (no intermediate copies).
        let feature_cols: Vec<&RawColumn> = self
            .columns
            .iter()
            .filter(|c| c.name != target_col.name)
            .collect();
        let n_features = feature_cols.len();
        let mut features = ndarray::Array2::from_elem((n, n_features), f64::NAN);
        let mut feature_columns = Vec::with_capacity(n_features);
        let mut feature_kinds = Vec::with_capacity(n_features);
        for (j, col) in feature_cols.iter().enumerate() {
            check()?;
            feature_columns.push(col.name.clone());
            let out = features.column_mut(j).into_iter();
            match &col.data {
                ColumnData::Text(values) => {
                    let (codes, _) = label_encode(values);
                    out.zip(codes).for_each(|(o, c)| *o = f64::from(c));
                    feature_kinds.push(ColumnKind::Int);
                }
                other => {
                    other.write_f64(out);
                    feature_kinds.push(other.kind());
                }
            }
        }
        check()?;

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
