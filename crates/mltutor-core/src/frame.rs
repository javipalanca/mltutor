//! Small table types shared by the dataset explorer and the CSV loader.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::stats::{describe_column, DESCRIBE_STATS};

/// One cell of a preview table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Cell {
    /// Integer or float value (the UI formats it with `.6g`).
    Number { v: f64 },
    /// Text (class names, label columns, text columns of a CSV).
    Text { v: String },
    /// Boolean column value (shown as `True`/`False`, like pandas).
    Bool { v: bool },
    /// NaN or an empty field.
    Missing,
}

impl Cell {
    /// `Number`, or `Missing` for NaN.
    pub fn number(v: f64) -> Self {
        if v.is_nan() {
            Cell::Missing
        } else {
            Cell::Number { v }
        }
    }
}

/// The first rows of a table (`df.head(n)`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TablePreview {
    /// Column headers.
    pub columns: Vec<String>,
    /// Row labels: positional, `"0"`, `"1"`… (rows are aligned by position).
    pub index: Vec<String>,
    /// Row-major cells, `rows.len() == index.len()`.
    pub rows: Vec<Vec<Cell>>,
    /// Number of rows of the whole table.
    pub total_rows: usize,
}

/// `DataFrame.describe()` of the numeric columns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DescribeTable {
    /// Numeric columns, in table order.
    pub columns: Vec<String>,
    /// `["count", "mean", "std", "min", "25%", "50%", "75%", "max"]`.
    pub stats: Vec<String>,
    /// `stats × columns` (NaN where pandas shows NaN, e.g. std of one value).
    pub values: Array2<f64>,
    /// Columns left out because they are not numeric (bool, text, class target).
    pub excluded_columns: Vec<String>,
}

impl DescribeTable {
    /// Builds the table from `(name, values)` pairs of numeric columns.
    pub fn from_columns<'a, I>(columns: I, excluded_columns: Vec<String>) -> Self
    where
        I: IntoIterator<Item = (String, &'a [f64])>,
    {
        let mut names = Vec::new();
        let mut stats_per_col = Vec::new();
        for (name, values) in columns {
            names.push(name);
            stats_per_col.push(describe_column(values));
        }
        Self::from_stats(names, &stats_per_col, excluded_columns)
    }

    /// Builds the table from the already computed [`describe_column`] of each
    /// column (`stats_per_col[c]`, same order as `columns`).
    pub fn from_stats(
        columns: Vec<String>,
        stats_per_col: &[[f64; 8]],
        excluded_columns: Vec<String>,
    ) -> Self {
        let n = columns.len().min(stats_per_col.len());
        let values = Array2::from_shape_fn((DESCRIBE_STATS.len(), n), |(s, c)| stats_per_col[c][s]);
        Self {
            columns,
            stats: DESCRIBE_STATS.iter().map(|s| (*s).to_owned()).collect(),
            values,
            excluded_columns,
        }
    }
}
