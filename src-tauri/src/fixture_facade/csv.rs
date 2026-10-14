//! Development CSV reader for [`super::FixtureFacade`]: UTF-8 (with or without BOM) or
//! Windows-1252, separator `,` `;` or tab, pandas-like dtypes. The reference reader is
//! `mltutor-core::csv`; this one is enough to drive the UI and is checked against
//! `tests/parity/csv`.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use super::frame::{Frame, Target};
use super::stats;
use crate::core_facade::{check_cancelled, CsvDraft, CsvFinal, DatasetView};
use crate::ipc::{
    finite, to_u32, Cell, CommandError, CorrelationMatrix, CsvColumnInfo, CsvPreview,
    CsvTargetStats, DatasetRef, DatasetSummary, DescribeTable, DiagKind, PairplotData,
    TablePreview, TargetDistribution, TargetInfo, TaskChoice, TaskType,
};

/// pandas `read_csv` default NA tokens.
const NA_VALUES: [&str; 20] = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null", "-NAN",
];
const PREVIEW_ROWS: usize = 5;
const VALUE_COUNTS_LIMIT: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Int,
    Float,
    Bool,
    Object,
}

impl Dtype {
    fn name(self) -> &'static str {
        match self {
            Self::Int => "int64",
            Self::Float => "float64",
            Self::Bool => "bool",
            Self::Object => "object",
        }
    }

    fn numeric(self) -> bool {
        matches!(self, Self::Int | Self::Float)
    }
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub dtype: Dtype,
    /// `None` = missing.
    pub raw: Vec<Option<String>>,
    /// Numeric value (int, float, bool as 0/1); NaN for missing or text.
    pub numbers: Vec<f64>,
}

impl Column {
    fn missing(&self) -> usize {
        self.raw.iter().filter(|v| v.is_none()).count()
    }

    /// pandas-like text of each value (missing → "nan").
    fn texts(&self) -> Vec<String> {
        self.raw
            .iter()
            .zip(&self.numbers)
            .map(|(raw, number)| match (self.dtype, raw) {
                (_, None) => "nan".to_owned(),
                (Dtype::Int | Dtype::Float, Some(_)) => stats::format_value(*number),
                (Dtype::Bool, Some(_)) => if *number != 0.0 { "True" } else { "False" }.to_owned(),
                (Dtype::Object, Some(text)) => text.clone(),
            })
            .collect()
    }

    fn nunique(&self) -> usize {
        let mut values: Vec<String> = self
            .texts()
            .into_iter()
            .zip(&self.raw)
            .filter(|(_, raw)| raw.is_some())
            .map(|(t, _)| t)
            .collect();
        values.sort();
        values.dedup();
        values.len()
    }

    fn cell(&self, row: usize) -> Cell {
        match (self.dtype, &self.raw[row]) {
            (_, None) => Cell::Missing,
            (Dtype::Int | Dtype::Float, Some(_)) => Cell::number(self.numbers[row]),
            (Dtype::Bool, Some(_)) => Cell::Bool {
                v: self.numbers[row] != 0.0,
            },
            (Dtype::Object, Some(text)) => Cell::Text { v: text.clone() },
        }
    }

    /// `LabelEncoder`: text in alphabetical order (`astype(str)`, NaN → "nan"); numbers
    /// in numeric order. `class_names` follows `load_dataset_from_file`: "Clase {v}" for
    /// integers, `str(v)` for floats and bools, and the original text (D10) for strings.
    fn label_encode(&self) -> (Vec<u16>, Vec<String>) {
        let texts = self.texts();
        let mut order: Vec<usize> = (0..texts.len()).collect();
        if self.dtype == Dtype::Object {
            order.sort_by(|a, b| texts[*a].cmp(&texts[*b]));
        } else {
            order.sort_by(|a, b| self.numbers[*a].total_cmp(&self.numbers[*b]));
        }
        order.dedup_by(|a, b| texts[*a] == texts[*b]);
        let classes: Vec<&String> = order.iter().map(|i| &texts[*i]).collect();
        let codes = texts
            .iter()
            .map(|t| {
                classes
                    .iter()
                    .position(|c| *c == t)
                    .and_then(|i| u16::try_from(i).ok())
                    .unwrap_or(u16::MAX)
            })
            .collect();
        let names = classes
            .iter()
            .map(|c| match self.dtype {
                Dtype::Int => format!("Clase {c}"),
                _ => (*c).clone(),
            })
            .collect();
        (codes, names)
    }
}

/// Windows-1252 bytes 0x80..=0x9F; the rest of the range maps to Latin-1.
const CP1252_HIGH: [char; 32] = [
    '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8d}', 'Ž', '\u{8f}',
    '\u{90}', '‘', '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9d}', 'ž', 'Ÿ',
];

fn decode(bytes: &[u8], notices: &mut Vec<String>) -> String {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_owned(),
        Err(_) => {
            notices.push("Se ha leído el archivo como Windows-1252/Latin-1".to_owned());
            bytes
                .iter()
                .map(|b| match *b {
                    0x80..=0x9F => CP1252_HIGH[usize::from(*b - 0x80)],
                    other => char::from(other),
                })
                .collect()
        }
    }
}

fn detect_separator(text: &str, notices: &mut Vec<String>) -> u8 {
    let first = text.lines().next().unwrap_or("");
    let count = |c: char| first.chars().filter(|x| *x == c).count();
    let (comma, semicolon, tab) = (count(','), count(';'), count('\t'));
    if semicolon > comma && semicolon >= tab {
        notices.push("Se ha detectado «;» como separador de columnas".to_owned());
        b';'
    } else if tab > comma && tab > semicolon {
        notices.push("Se ha detectado el tabulador como separador de columnas".to_owned());
        b'\t'
    } else {
        b','
    }
}

fn dedup_names(names: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(names.len());
    for name in names {
        let mut candidate = name.clone();
        let mut suffix = 1;
        while out.contains(&candidate) {
            candidate = format!("{name}.{suffix}");
            suffix += 1;
        }
        out.push(candidate);
    }
    out
}

fn infer(name: String, raw: Vec<Option<String>>) -> Column {
    let present: Vec<&str> = raw.iter().flatten().map(String::as_str).collect();
    let has_missing = present.len() < raw.len();
    let all = |f: &dyn Fn(&str) -> bool| present.iter().all(|v| f(v));
    let is_bool = |v: &str| matches!(v, "True" | "False" | "true" | "false" | "TRUE" | "FALSE");
    let dtype = if !present.is_empty() && !has_missing && all(&|v| v.trim().parse::<i64>().is_ok())
    {
        Dtype::Int
    } else if all(&|v| v.trim().parse::<f64>().is_ok()) {
        Dtype::Float
    } else if !has_missing && all(&is_bool) {
        Dtype::Bool
    } else {
        Dtype::Object
    };
    let numbers = raw
        .iter()
        .map(|v| match (dtype, v) {
            (_, None) | (Dtype::Object, _) => f64::NAN,
            (Dtype::Bool, Some(v)) => {
                if v.eq_ignore_ascii_case("true") {
                    1.0
                } else {
                    0.0
                }
            }
            (_, Some(v)) => v.trim().parse::<f64>().unwrap_or(f64::NAN),
        })
        .collect();
    Column {
        name,
        dtype,
        raw,
        numbers,
    }
}

pub fn read(file_name: &str, bytes: &[u8]) -> Result<CsvTable, CommandError> {
    let mut notices = Vec::new();
    let text = decode(bytes, &mut notices);
    if text.trim().is_empty() {
        return Err(CommandError::csv_empty());
    }
    let separator = detect_separator(&text, &mut notices);
    let mut reader = ::csv::ReaderBuilder::new()
        .delimiter(separator)
        .flexible(true)
        .has_headers(true)
        .from_reader(text.as_bytes());
    let header: Vec<String> = reader
        .headers()
        .map_err(CommandError::csv_io)?
        .iter()
        .map(str::to_owned)
        .collect();
    let mut records: Vec<Vec<String>> = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let record = record.map_err(CommandError::csv_io)?;
        if record.len() == 1 && record.get(0).is_some_and(str::is_empty) {
            continue; // blank line
        }
        if record.len() != header.len() {
            let line = record
                .position()
                .map_or(index as u64 + 2, ::csv::Position::line);
            return Err(CommandError::csv_malformed(
                line,
                record.len(),
                header.len(),
            ));
        }
        records.push(record.iter().map(str::to_owned).collect());
    }
    if records.is_empty() {
        return Err(CommandError::csv_empty());
    }
    if header.len() < 2 {
        return Err(CommandError::csv_too_few_columns());
    }
    let names = dedup_names(header);
    let columns = names
        .into_iter()
        .enumerate()
        .map(|(c, name)| {
            let raw = records
                .iter()
                .map(|r| {
                    let v = &r[c];
                    (!NA_VALUES.contains(&v.as_str())).then(|| v.clone())
                })
                .collect();
            infer(name, raw)
        })
        .collect();
    Ok(CsvTable {
        file_name: file_name.to_owned(),
        columns,
        n_rows: records.len(),
        notices,
    })
}

#[derive(Debug, Clone)]
pub struct CsvTable {
    pub file_name: String,
    pub columns: Vec<Column>,
    pub n_rows: usize,
    pub notices: Vec<String>,
}

impl CsvTable {
    fn column(&self, name: &str) -> Result<&Column, CommandError> {
        self.columns
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| CommandError::unknown_column(name))
    }

    /// Selector suggestion: `nunique ≤ 20 and nunique < 0.1·n`.
    pub fn suggest_task(&self, column: &Column) -> TaskType {
        let unique = column.nunique();
        if unique <= 20 && (unique as f64) < 0.1 * self.n_rows as f64 {
            TaskType::Classification
        } else {
            TaskType::Regression
        }
    }

    /// `load_dataset_from_file` with "auto": `nunique < max(5, 0.05·n)`.
    pub fn auto_task(&self, column: &Column) -> TaskType {
        if (column.nunique() as f64) < (5f64).max(0.05 * self.n_rows as f64) {
            TaskType::Classification
        } else {
            TaskType::Regression
        }
    }

    pub fn describe(&self) -> DescribeTable {
        let numeric: Vec<(String, &[f64])> = self
            .columns
            .iter()
            .filter(|c| c.dtype.numeric())
            .map(|c| (c.name.clone(), c.numbers.as_slice()))
            .collect();
        let excluded = self
            .columns
            .iter()
            .filter(|c| !c.dtype.numeric())
            .map(|c| c.name.clone())
            .collect();
        stats::describe(&numeric, excluded)
    }
}

impl CsvDraft for CsvTable {
    fn file_name(&self) -> &str {
        &self.file_name
    }

    fn preview(&self) -> CsvPreview {
        let rows = PREVIEW_ROWS.min(self.n_rows);
        let default_target = self
            .columns
            .last()
            .map(|c| c.name.clone())
            .unwrap_or_default();
        let target = self
            .target_info(&default_target)
            .unwrap_or_else(|_| TargetInfo {
                column: default_target.clone(),
                nunique: 0,
                dtype: "object".into(),
                value_counts: None,
                suggested_task: TaskType::Regression,
            });
        CsvPreview {
            draft_id: 0,
            file_name: self.file_name.clone(),
            n_rows: to_u32(self.n_rows),
            n_cols: to_u32(self.columns.len()),
            missing_values: to_u32(self.columns.iter().map(Column::missing).sum()),
            columns: self
                .columns
                .iter()
                .map(|c| CsvColumnInfo {
                    name: c.name.clone(),
                    dtype: c.dtype.name().into(),
                    missing: to_u32(c.missing()),
                    nunique: to_u32(c.nunique()),
                })
                .collect(),
            head: TablePreview {
                columns: self.columns.iter().map(|c| c.name.clone()).collect(),
                index: (0..rows).map(|i| i.to_string()).collect(),
                rows: (0..rows)
                    .map(|r| self.columns.iter().map(|c| c.cell(r)).collect())
                    .collect(),
                total_rows: to_u32(self.n_rows),
            },
            default_target,
            target,
            notices: self.notices.clone(),
        }
    }

    fn target_info(&self, target: &str) -> Result<TargetInfo, CommandError> {
        let column = self.column(target)?;
        let nunique = column.nunique();
        let value_counts = (nunique <= VALUE_COUNTS_LIMIT).then(|| {
            let texts = column.texts();
            stats::value_counts_by_frequency(
                texts
                    .iter()
                    .zip(&column.raw)
                    .filter(|(_, raw)| raw.is_some())
                    .map(|(t, _)| t.as_str()),
            )
        });
        Ok(TargetInfo {
            column: column.name.clone(),
            nunique: to_u32(nunique),
            dtype: column.dtype.name().into(),
            value_counts,
            suggested_task: self.suggest_task(column),
        })
    }

    fn finalize(
        &self,
        target: &str,
        task: TaskChoice,
        cancel: &AtomicBool,
    ) -> Result<CsvFinal, CommandError> {
        check_cancelled(cancel)?;
        let target_column = self.column(target)?;
        let resolved = match task {
            TaskChoice::Auto => self.auto_task(target_column),
            TaskChoice::Classification => TaskType::Classification,
            TaskChoice::Regression => TaskType::Regression,
        };
        let target_values = match resolved {
            TaskType::Classification => {
                let (codes, names) = target_column.label_encode();
                Target::Classes { codes, names }
            }
            // Text targets are label-encoded first, as in Python.
            TaskType::Regression => Target::Numeric(match target_column.dtype {
                Dtype::Object => target_column
                    .label_encode()
                    .0
                    .iter()
                    .map(|c| f64::from(*c))
                    .collect(),
                _ => target_column.numbers.clone(),
            }),
        };
        let features: Vec<&Column> = self.columns.iter().filter(|c| c.name != target).collect();
        let names: Vec<String> = features.iter().map(|c| c.name.clone()).collect();
        let frame = Frame {
            feature_columns: names.clone(),
            feature_display: names.clone(),
            features: features
                .iter()
                .map(|c| match c.dtype {
                    Dtype::Object => c.label_encode().0.iter().map(|v| f64::from(*v)).collect(),
                    _ => c.numbers.clone(),
                })
                .collect(),
            bool_features: features.iter().map(|c| c.dtype == Dtype::Bool).collect(),
            labels: Vec::new(),
            target: target_values,
        };
        let description = match &frame.target {
            Target::Classes { names: classes, .. } => format!(
                "Dataset personalizado: {} muestras, {} características, {} clases",
                self.n_rows,
                names.len(),
                classes.len()
            ),
            Target::Numeric(_) => format!(
                "Dataset personalizado: {} muestras, {} características, regresión",
                self.n_rows,
                names.len()
            ),
        };
        let class_names = match &frame.target {
            Target::Classes { names, .. } => Some(names.clone()),
            Target::Numeric(_) => None,
        };
        let target_stats = if resolved == TaskType::Classification || !target_column.dtype.numeric()
        {
            let texts = target_column.texts();
            CsvTargetStats::Classes {
                value_counts: stats::value_counts_by_frequency(texts.iter().map(String::as_str)),
            }
        } else {
            let values = &target_column.numbers;
            let (hist_edges, hist_counts) = stats::histogram(values, 30);
            let d = stats::describe_column(values);
            CsvTargetStats::Numeric {
                min: finite(d[3]),
                max: finite(d[7]),
                mean: finite(d[1]),
                median: finite(d[5]),
                std: finite(d[2]),
                hist_edges,
                hist_counts,
            }
        };
        let summary = DatasetSummary {
            dataset: DatasetRef::Csv { id: 0 },
            label: format!("📄 {}", self.file_name),
            task: resolved,
            n_rows: to_u32(self.n_rows),
            n_features: to_u32(names.len()),
            target_column: "target".into(),
            feature_columns: names.clone(),
            feature_display: names,
            label_columns: Vec::new(),
            class_names,
            description,
        };
        Ok(CsvFinal {
            dataset: Arc::new(ComputedDataset { frame, summary }),
            resolved_task: resolved,
            n_rows: self.n_rows,
            n_cols: self.columns.len(),
            describe: self.describe(),
            target_stats,
        })
    }
}

/// A dataset whose views are all computed from its frame (user CSVs).
pub struct ComputedDataset {
    pub frame: Frame,
    pub summary: DatasetSummary,
}

impl DatasetView for ComputedDataset {
    fn summary(&self) -> DatasetSummary {
        self.summary.clone()
    }
    fn preview(&self, rows: usize) -> TablePreview {
        self.frame.preview(rows)
    }
    fn describe(&self) -> DescribeTable {
        self.frame.describe()
    }
    fn target_distribution(&self) -> TargetDistribution {
        self.frame.target_distribution()
    }
    fn correlation(&self, cancel: &AtomicBool) -> Result<CorrelationMatrix, CommandError> {
        check_cancelled(cancel)?;
        Ok(self.frame.correlation())
    }
    fn pairplot(&self, features: &[String], diag: DiagKind) -> Result<PairplotData, CommandError> {
        Ok(self
            .frame
            .pairplot(features, diag, self.frame.default_sample()))
    }
    fn train(
        &self,
        _spec: &crate::core_facade::TrainSpec,
        _progress: &crate::core_facade::Progress<'_>,
        _cancel: &AtomicBool,
    ) -> Result<crate::core_facade::ModelHandle, CommandError> {
        Err(CommandError::training(
            "FixtureFacade: los CSV propios solo se entrenan con mltutor-core",
        ))
    }
}
