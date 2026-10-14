//! Bytes → [`RawTable`]: decoding, separator detection, parsing and pandas-like
//! type inference.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use super::text::TextColumnBuilder;
use super::{ColumnData, CsvNotice, RawColumn, RawTable, TextColumn};
use crate::error::CsvError;

/// Largest CSV accepted (D10): 100 MB.
pub const MAX_CSV_BYTES: u64 = 100 * 1024 * 1024;

/// pandas' default `na_values` (plus the empty field).
const NA_VALUES: [&str; 19] = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
];
const TRUE_VALUES: [&str; 3] = ["True", "TRUE", "true"];
const FALSE_VALUES: [&str; 3] = ["False", "FALSE", "false"];

fn is_na(field: &str) -> bool {
    NA_VALUES.contains(&field)
}

/// Reads a CSV file from disk (size checked before reading).
pub fn read_csv_file(path: &Path) -> Result<RawTable, CsvError> {
    read_csv_file_cancellable(path, &AtomicBool::new(false))
}

/// [`read_csv_file`] with cooperative cancellation (see [`read_csv_cancellable`]).
pub fn read_csv_file_cancellable(path: &Path, cancel: &AtomicBool) -> Result<RawTable, CsvError> {
    let io = |e: std::io::Error| CsvError::Io {
        message: e.to_string(),
    };
    let len = std::fs::metadata(path).map_err(io)?.len();
    if len > MAX_CSV_BYTES {
        return Err(CsvError::TooLarge {
            bytes: len,
            limit: MAX_CSV_BYTES,
        });
    }
    let bytes = std::fs::read(path).map_err(io)?;
    read_csv_cancellable(&bytes, cancel)
}

/// Decodes UTF-8 (BOM removed) or, failing that, Windows-1252 (D10).
fn decode(bytes: &[u8]) -> Result<(Cow<'_, str>, Option<CsvNotice>), CsvError> {
    // NUL bytes mean UTF-16 or binary data, which no single-byte reading fixes.
    if bytes.contains(&0) {
        return Err(CsvError::Encoding);
    }
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok((Cow::Borrowed(text), None)),
        Err(_) => {
            let (text, _had_errors) = encoding_rs::WINDOWS_1252.decode_without_bom_handling(bytes);
            Ok((text, Some(CsvNotice::Windows1252)))
        }
    }
}

/// Separator of the first line: the most frequent of `,` `;` and tab outside
/// quotes; `,` on ties or when there is none (D10).
fn detect_separator(text: &str) -> u8 {
    let first_line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or_default();
    let mut counts = [0_usize; 3];
    let mut in_quotes = false;
    for c in first_line.chars() {
        match c {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => counts[0] += 1,
            ';' if !in_quotes => counts[1] += 1,
            '\t' if !in_quotes => counts[2] += 1,
            _ => {}
        }
    }
    let candidates = *b",;\t";
    let mut best = 0;
    for i in 1..3 {
        if counts[i] > counts[best] {
            best = i;
        }
    }
    candidates[best]
}

/// pandas names: empty headers become `Unnamed: {i}` and duplicates get `.1`, `.2`…
///
/// Same algorithm as pandas 2 `read_csv` (`PythonParser._infer_columns`; the C
/// parser gives the same names): named columns are mangled before the unnamed
/// ones, and a suffix is skipped if that name already exists in the header
/// (`a, a, a.1` → `a, a.2, a.1`). A map with the next suffix of each name and a
/// multiset of the current names keep it linear in the number of columns (the
/// previous search from `.1` upwards was quadratic with many repeated names).
fn header_names(raw: &[String]) -> Vec<String> {
    let mut columns: Vec<String> = Vec::with_capacity(raw.len());
    let mut unnamed: Vec<usize> = Vec::new();
    for (i, name) in raw.iter().enumerate() {
        if name.is_empty() {
            unnamed.push(i);
            columns.push(format!("Unnamed: {i}"));
        } else {
            columns.push(name.clone());
        }
    }
    // How many times each name appears in `columns` right now (`col in columns`).
    let mut present: HashMap<String, usize> = HashMap::with_capacity(columns.len());
    for c in &columns {
        *present.entry(c.clone()).or_insert(0) += 1;
    }
    let is_unnamed = {
        let mut v = vec![false; columns.len()];
        for &i in &unnamed {
            v[i] = true;
        }
        v
    };
    let order = (0..columns.len())
        .filter(|&i| !is_unnamed[i])
        .chain(unnamed.iter().copied());

    let mut counts: HashMap<String, usize> = HashMap::with_capacity(columns.len());
    for i in order {
        let old_col = columns[i].clone();
        let mut col = old_col.clone();
        let mut cur_count = counts.get(&col).copied().unwrap_or(0);
        while cur_count > 0 {
            counts.insert(old_col.clone(), cur_count + 1);
            col = format!("{old_col}.{cur_count}");
            if present.get(&col).is_some_and(|&n| n > 0) {
                cur_count += 1;
            } else {
                cur_count = counts.get(&col).copied().unwrap_or(0);
            }
        }
        if col != old_col {
            if let Some(n) = present.get_mut(&old_col) {
                *n = n.saturating_sub(1);
            }
            *present.entry(col.clone()).or_insert(0) += 1;
        }
        counts.insert(col.clone(), cur_count + 1);
        columns[i] = col;
    }
    columns
}

/// Parses a number accepting the decimal comma (`"1,5"`) when `decimal_comma`.
/// The flag is set when a comma had to be replaced.
///
/// Rust's `f64::from_str` also accepts `"NAN"`, `"Nan"`… but pandas only reads the
/// exact strings of [`NA_VALUES`] as missing (handled before) and leaves any other
/// spelling of NaN as text, so a NaN result is rejected here. `inf`, `-Infinity`…
/// (any case) are accepted, as pandas does.
fn parse_float(field: &str, decimal_comma: bool, used_comma: &mut bool) -> Option<f64> {
    let t = field.trim();
    if let Ok(v) = t.parse::<f64>() {
        return (!v.is_nan()).then_some(v);
    }
    if decimal_comma && t.matches(',').count() == 1 && !t.contains('.') {
        if let Ok(v) = t.replace(',', ".").parse::<f64>() {
            if v.is_nan() {
                return None;
            }
            *used_comma = true;
            return Some(v);
        }
    }
    None
}

/// A float field: NaN for a pandas NA string, the number otherwise, `None` if
/// it is not a number.
fn float_or_na(field: &str, decimal_comma: bool, used_comma: &mut bool) -> Option<f64> {
    if is_na(field) {
        Some(f64::NAN)
    } else {
        parse_float(field, decimal_comma, used_comma)
    }
}

fn parse_bool(field: &str) -> Option<bool> {
    let t = field.trim();
    if TRUE_VALUES.contains(&t) {
        Some(true)
    } else if FALSE_VALUES.contains(&t) {
        Some(false)
    } else {
        None
    }
}

/// Type inference of one column while the file is being read (streaming).
///
/// It reaches the same result as pandas' inference over the whole column —
/// `int64` (all integers, no missing) → `float64` (numbers, missing allowed) →
/// `bool` (True/False only, no missing) → `object` — but stores numbers as `i64`,
/// `f64` or `bool` (8 bytes or less per value) instead of one `String` per field.
/// Each state can only move to a more general one:
///
/// ```text
/// Empty ─┬─ int ──→ Int ── float or NA ──→ Float
///        ├─ float or NA ─────────────────→ Float
///        ├─ True/False ──→ Bool
///        └─ other ──→ Text  (any state also goes to Text on a value it cannot hold)
/// ```
///
/// A `Text` column keeps nothing in the first pass: its original strings are
/// collected in a second pass over the file (formatting the numbers back would
/// lose the original text, e.g. `"007"` or `"1.50"`).
#[derive(Debug)]
enum ColumnBuilder {
    /// No value seen yet.
    Empty,
    /// `neg_zero`: rows written as `-0` (`-00`…). pandas keeps them as `0` in an
    /// `int64` column but as `-0.0` if the column becomes `float64`.
    Int {
        values: Vec<i64>,
        neg_zero: Vec<usize>,
    },
    /// `used_comma`: some value needed the decimal comma.
    Float {
        values: Vec<f64>,
        used_comma: bool,
    },
    Bool(Vec<bool>),
    Text,
}

impl ColumnBuilder {
    /// Adds the next field; `capacity` is the estimated number of rows, used to
    /// reserve the column once and avoid the 2× peak of a growing `Vec`.
    fn push(&mut self, field: &str, decimal_comma: bool, capacity: usize) {
        match self {
            ColumnBuilder::Empty => *self = Self::start(field, decimal_comma, capacity),
            ColumnBuilder::Int { values, neg_zero } => {
                if let Ok(i) = field.trim().parse::<i64>() {
                    if i == 0 && field.trim().starts_with('-') {
                        neg_zero.push(values.len());
                    }
                    values.push(i);
                    return;
                }
                let mut used_comma = false;
                match float_or_na(field, decimal_comma, &mut used_comma) {
                    Some(x) => {
                        // Same allocation reused: `i64` and `f64` have the same layout
                        // and `into_iter().map().collect()` collects in place.
                        let mut floats: Vec<f64> = std::mem::take(values)
                            .into_iter()
                            .map(|i| i as f64)
                            .collect();
                        for &row in neg_zero.iter() {
                            floats[row] = -0.0;
                        }
                        floats.push(x);
                        *self = ColumnBuilder::Float {
                            values: floats,
                            used_comma,
                        };
                    }
                    None => *self = ColumnBuilder::Text,
                }
            }
            ColumnBuilder::Float { values, used_comma } => {
                match float_or_na(field, decimal_comma, used_comma) {
                    Some(x) => values.push(x),
                    None => *self = ColumnBuilder::Text,
                }
            }
            ColumnBuilder::Bool(values) => match parse_bool(field) {
                Some(b) => values.push(b),
                None => *self = ColumnBuilder::Text,
            },
            ColumnBuilder::Text => {}
        }
    }

    fn start(field: &str, decimal_comma: bool, capacity: usize) -> Self {
        if is_na(field) {
            let mut values = Vec::with_capacity(capacity);
            values.push(f64::NAN);
            return ColumnBuilder::Float {
                values,
                used_comma: false,
            };
        }
        if let Ok(i) = field.trim().parse::<i64>() {
            let mut values = Vec::with_capacity(capacity);
            values.push(i);
            let neg_zero = if i == 0 && field.trim().starts_with('-') {
                vec![0]
            } else {
                Vec::new()
            };
            return ColumnBuilder::Int { values, neg_zero };
        }
        let mut used_comma = false;
        if let Some(x) = parse_float(field, decimal_comma, &mut used_comma) {
            let mut values = Vec::with_capacity(capacity);
            values.push(x);
            return ColumnBuilder::Float { values, used_comma };
        }
        if let Some(b) = parse_bool(field) {
            let mut values = Vec::with_capacity(capacity);
            values.push(b);
            return ColumnBuilder::Bool(values);
        }
        ColumnBuilder::Text
    }

    fn is_text(&self) -> bool {
        matches!(self, ColumnBuilder::Text)
    }

    /// The typed column, and whether it used the decimal comma. `Text` columns are
    /// filled in by the second pass (`texts`).
    fn finish(self, texts: Option<TextColumn>) -> (ColumnData, bool) {
        fn fit<T>(mut v: Vec<T>) -> Vec<T> {
            // The row estimate counts every line; give back a large excess only.
            if v.capacity() - v.len() > v.len() / 8 {
                v.shrink_to_fit();
            }
            v
        }
        match self {
            ColumnBuilder::Int { values, .. } => (ColumnData::Int(fit(values)), false),
            ColumnBuilder::Float { values, used_comma } => {
                (ColumnData::Float(fit(values)), used_comma)
            }
            ColumnBuilder::Bool(v) => (ColumnData::Bool(fit(v)), false),
            // `Empty` only happens without data rows, which `read_csv` rejects.
            ColumnBuilder::Empty => (ColumnData::Float(Vec::new()), false),
            ColumnBuilder::Text => (ColumnData::Text(texts.unwrap_or_default()), false),
        }
    }
}

/// pandas-like inference over a whole column (the streaming [`ColumnBuilder`]
/// over its fields). Kept for the unit tests of the inference rules.
#[cfg(test)]
fn infer_column(fields: Vec<String>, decimal_comma: bool) -> (ColumnData, bool) {
    let mut builder = ColumnBuilder::Empty;
    for f in &fields {
        builder.push(f, decimal_comma, fields.len());
    }
    let texts = builder.is_text().then(|| {
        fields
            .iter()
            .map(|f| (!is_na(f)).then_some(f.as_str()))
            .collect()
    });
    builder.finish(texts)
}

/// Rows between two checks of the cancellation flag.
pub const CANCEL_CHECK_ROWS: usize = 16_384;

/// Calls `f(index, record)` for each non-blank record of `text`, the header being
/// index 0. Rows with a different number of fields than the header are
/// [`CsvError::Malformed`]; `cancel` is checked every [`CANCEL_CHECK_ROWS`] records.
fn for_each_record(
    text: &str,
    separator: u8,
    cancel: &AtomicBool,
    mut f: impl FnMut(usize, &::csv::StringRecord),
) -> Result<(), CsvError> {
    let mut reader = ::csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .delimiter(separator)
        .from_reader(text.as_bytes());
    // One record reused for every line: no allocation per row.
    let mut record = ::csv::StringRecord::new();
    let mut expected: Option<usize> = None;
    let mut index = 0_usize;
    loop {
        let more = reader
            .read_record(&mut record)
            .map_err(|e| CsvError::Malformed {
                line: e.position().map_or(0, ::csv::Position::line),
                found: 0,
                expected: expected.unwrap_or(0),
            })?;
        if !more {
            return Ok(());
        }
        // pandas `skip_blank_lines=True`.
        if record.len() == 1 && record.get(0).is_some_and(|f| f.trim().is_empty()) {
            continue;
        }
        match expected {
            None => expected = Some(record.len()),
            Some(n) if n != record.len() => {
                return Err(CsvError::Malformed {
                    line: record.position().map_or(0, ::csv::Position::line),
                    found: record.len(),
                    expected: n,
                });
            }
            Some(_) => {}
        }
        if index % CANCEL_CHECK_ROWS == 0 && cancel.load(Ordering::Relaxed) {
            return Err(CsvError::Cancelled);
        }
        f(index, &record);
        index += 1;
    }
}

/// Parses a CSV the way the Python app does with `pd.read_csv`, plus the D10
/// tolerances (Windows-1252, `;`/tab separators, decimal comma with `;`).
///
/// Validations, in the order the app applies them: no columns or no data rows →
/// [`CsvError::Empty`]; fewer than 2 columns → [`CsvError::TooFewColumns`]. Rows
/// with a different number of fields than the header → [`CsvError::Malformed`].
/// Blank lines are skipped. Never panics (fuzzed in `tests/csv_fuzz.rs`).
///
/// ±infinity (`inf`, `-Infinity`…) is read as a float, like pandas, and reported
/// with [`CsvNotice::NonFinite`]; spellings of NaN other than pandas' NA strings
/// (`"NAN"`, `"Nan"`) stay text, like pandas.
pub fn read_csv(bytes: &[u8]) -> Result<RawTable, CsvError> {
    read_csv_cancellable(bytes, &AtomicBool::new(false))
}

/// [`read_csv`] with cooperative cancellation: returns [`CsvError::Cancelled`] soon
/// after `cancel` becomes `true` (checked every [`CANCEL_CHECK_ROWS`] rows).
///
/// Memory: numeric columns are typed while reading (see `ColumnBuilder`), so a
/// numeric CSV peaks at about the file size plus 8 bytes per value; text columns
/// are read in a second pass into a dictionary ([`TextColumn`]: 4 bytes per row
/// plus each distinct value once). Measured in `tests/csv_memory.rs`.
pub fn read_csv_cancellable(bytes: &[u8], cancel: &AtomicBool) -> Result<RawTable, CsvError> {
    let len = bytes.len() as u64;
    if len > MAX_CSV_BYTES {
        return Err(CsvError::TooLarge {
            bytes: len,
            limit: MAX_CSV_BYTES,
        });
    }
    let (text, encoding_notice) = decode(bytes)?;
    let mut notices: Vec<CsvNotice> = encoding_notice.into_iter().collect();
    let separator = detect_separator(&text);
    if separator != b',' {
        notices.push(CsvNotice::Separator {
            separator: char::from(separator),
        });
    }
    let decimal_comma = separator == b';';
    // Upper bound of the data rows (blank lines and quoted newlines included).
    let row_estimate = text.bytes().filter(|&b| b == b'\n').count() + 1;

    // Pass 1: header and typed numeric columns.
    let mut header: Vec<String> = Vec::new();
    let mut builders: Vec<ColumnBuilder> = Vec::new();
    let mut n_rows = 0_usize;
    for_each_record(&text, separator, cancel, |index, record| {
        if index == 0 {
            header = record.iter().map(str::to_owned).collect();
            builders = (0..header.len()).map(|_| ColumnBuilder::Empty).collect();
        } else {
            for (builder, field) in builders.iter_mut().zip(record.iter()) {
                builder.push(field, decimal_comma, row_estimate);
            }
            n_rows += 1;
        }
    })?;

    if n_rows == 0 || header.is_empty() {
        return Err(CsvError::Empty);
    }
    if header.len() < 2 {
        return Err(CsvError::TooFewColumns {
            found: header.len(),
        });
    }

    // Pass 2, only if needed: the original strings of the text columns.
    let text_columns: Vec<usize> = (0..builders.len())
        .filter(|&c| builders[c].is_text())
        .collect();
    let mut texts: Vec<Option<TextColumn>> = (0..builders.len()).map(|_| None).collect();
    if !text_columns.is_empty() {
        let mut collected: Vec<TextColumnBuilder> = text_columns
            .iter()
            .map(|_| TextColumnBuilder::with_capacity(n_rows))
            .collect();
        for_each_record(&text, separator, cancel, |index, record| {
            if index == 0 {
                return;
            }
            for (out, &c) in collected.iter_mut().zip(&text_columns) {
                let field = record.get(c).unwrap_or_default();
                out.push((!is_na(field)).then_some(field));
            }
        })?;
        for (values, &c) in collected.into_iter().zip(&text_columns) {
            texts[c] = Some(values.finish());
        }
    }

    let mut comma_columns = Vec::new();
    let mut non_finite_columns = Vec::new();
    let names = header_names(&header);
    let columns: Vec<RawColumn> = names
        .into_iter()
        .zip(builders.into_iter().zip(texts))
        .map(|(name, (builder, texts))| {
            let (data, used_comma) = builder.finish(texts);
            if used_comma {
                comma_columns.push(name.clone());
            }
            if let ColumnData::Float(values) = &data {
                if values.iter().any(|v| v.is_infinite()) {
                    non_finite_columns.push(name.clone());
                }
            }
            RawColumn { name, data }
        })
        .collect();
    if !comma_columns.is_empty() {
        notices.push(CsvNotice::DecimalComma {
            columns: comma_columns,
        });
    }
    if !non_finite_columns.is_empty() {
        notices.push(CsvNotice::NonFinite {
            columns: non_finite_columns,
        });
    }
    Ok(RawTable {
        columns,
        n_rows,
        separator: char::from(separator),
        notices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mangles_duplicate_and_empty_headers() {
        // Expected: `list(pd.read_csv(io.StringIO(header + "\n" + row)).columns)`
        // with pandas 2.3.3.
        let cases: [(&str, &[&str]); 9] = [
            ("a,a,,a", &["a", "a.1", "Unnamed: 2", "a.2"]),
            ("a,a.1,a", &["a", "a.1", "a.2"]),
            ("a,a,a.1", &["a", "a.2", "a.1"]),
            (
                "x,,Unnamed: 1,x",
                &["x", "Unnamed: 1.1", "Unnamed: 1", "x.1"],
            ),
            (
                ",a,,a.1,a",
                &["Unnamed: 0", "a", "Unnamed: 2", "a.1", "a.2"],
            ),
            ("b,b,b,b.1,b.2", &["b", "b.3", "b.4", "b.1", "b.2"]),
            (
                "Unnamed: 2,,",
                &["Unnamed: 2", "Unnamed: 1", "Unnamed: 2.1"],
            ),
            ("a,a.1,a.1,a", &["a", "a.1", "a.1.1", "a.2"]),
            ("p,q", &["p", "q"]),
        ];
        for (header, expected) in cases {
            let raw: Vec<String> = header.split(',').map(str::to_owned).collect();
            assert_eq!(header_names(&raw), expected, "{header}");
        }
    }

    #[test]
    fn many_repeated_headers_are_linear() {
        // Review finding: 20 000 equal names took O(k²) string comparisons.
        let raw = vec!["x".to_owned(); 20_000];
        let start = std::time::Instant::now();
        let names = header_names(&raw);
        assert_eq!(names[19_999], "x.19999");
        let unique: std::collections::HashSet<&String> = names.iter().collect();
        assert_eq!(unique.len(), 20_000);
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
    }

    #[test]
    fn detects_separators() {
        assert_eq!(detect_separator("a;b;c\n1;2;3"), b';');
        assert_eq!(detect_separator("a\tb\n"), b'\t');
        assert_eq!(detect_separator("\"x;y\",b\n"), b',');
        assert_eq!(detect_separator("solo\n1"), b',');
    }

    #[test]
    fn infers_like_pandas() {
        let s = |v: &[&str]| v.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
        assert!(matches!(
            infer_column(s(&["1", "2"]), false).0,
            ColumnData::Int(_)
        ));
        assert!(matches!(
            infer_column(s(&["1", ""]), false).0,
            ColumnData::Float(_)
        ));
        assert!(matches!(
            infer_column(s(&["1.5", "2"]), false).0,
            ColumnData::Float(_)
        ));
        assert!(matches!(
            infer_column(s(&["True", "false"]), false).0,
            ColumnData::Bool(_)
        ));
        assert!(matches!(
            infer_column(s(&["True", ""]), false).0,
            ColumnData::Text(_)
        ));
        assert!(matches!(
            infer_column(s(&["1,5", "2"]), false).0,
            ColumnData::Text(_)
        ));
        let (data, comma) = infer_column(s(&["1,5", "2"]), true);
        assert!(comma);
        assert_eq!(data, ColumnData::Float(vec![1.5, 2.0]));
        assert!(matches!(
            infer_column(s(&["", "NA"]), false).0,
            ColumnData::Float(_)
        ));
    }

    /// The previous whole-column inference (one `String` per field), kept as the
    /// oracle of the streaming `ColumnBuilder`.
    fn reference_infer(fields: Vec<String>, decimal_comma: bool) -> (ColumnData, bool) {
        let any_na = fields.iter().any(|f| is_na(f));
        if !any_na {
            let ints: Option<Vec<i64>> = fields
                .iter()
                .map(|f| f.trim().parse::<i64>().ok())
                .collect();
            if let Some(ints) = ints {
                return (ColumnData::Int(ints), false);
            }
        }
        let mut used_comma = false;
        let floats: Option<Vec<f64>> = fields
            .iter()
            .map(|f| float_or_na(f, decimal_comma, &mut used_comma))
            .collect();
        if let Some(floats) = floats {
            return (ColumnData::Float(floats), used_comma);
        }
        if !any_na {
            let bools: Option<Vec<bool>> = fields.iter().map(|f| parse_bool(f)).collect();
            if let Some(bools) = bools {
                return (ColumnData::Bool(bools), false);
            }
        }
        let texts: Vec<Option<String>> = fields
            .into_iter()
            .map(|f| if is_na(&f) { None } else { Some(f) })
            .collect();
        (ColumnData::Text(texts.into()), false)
    }

    fn same(a: &ColumnData, b: &ColumnData) -> bool {
        match (a, b) {
            (ColumnData::Float(x), ColumnData::Float(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
            }
            _ => a == b,
        }
    }

    proptest::proptest! {
        #[test]
        fn streaming_inference_matches_whole_column_inference(
            fields in proptest::collection::vec(
                proptest::prop_oneof![
                    proptest::strategy::Just(String::new()),
                    proptest::strategy::Just("NA".to_owned()),
                    proptest::strategy::Just("NAN".to_owned()),
                    proptest::strategy::Just("inf".to_owned()),
                    proptest::strategy::Just("-Infinity".to_owned()),
                    proptest::strategy::Just("True".to_owned()),
                    proptest::strategy::Just("false".to_owned()),
                    proptest::strategy::Just(" 7 ".to_owned()),
                    proptest::strategy::Just("007".to_owned()),
                    proptest::strategy::Just("99999999999999999999".to_owned()),
                    "-?[0-9]{1,4}",
                    "-?[0-9]{1,3}\\.[0-9]{1,3}",
                    "-?[0-9]{1,3},[0-9]{1,3}",
                    "[a-z]{1,3}",
                ],
                1..12,
            ),
            decimal_comma in proptest::bool::ANY,
        ) {
            let (got, got_comma) = infer_column(fields.clone(), decimal_comma);
            let (want, want_comma) = reference_infer(fields, decimal_comma);
            proptest::prop_assert!(same(&got, &want), "{got:?} vs {want:?}");
            proptest::prop_assert_eq!(got_comma, want_comma);
        }
    }

    #[test]
    fn text_columns_keep_the_original_strings() {
        // Numbers seen before the first text value are not reformatted.
        let t = read_csv(b"a,b\n007,1\n1.50,2\nx,3\n,4\n").unwrap();
        assert_eq!(
            t.columns[0].data,
            ColumnData::Text(vec![Some("007"), Some("1.50"), Some("x"), None].into())
        );
        assert_eq!(t.columns[1].data, ColumnData::Int(vec![1, 2, 3, 4]));
    }

    #[test]
    fn nan_spellings_stay_text_and_infinities_are_floats() {
        // pandas 2.3: x = "NAN", "Nan", "inf" → object; y = inf, -inf, inf, 5 → float64
        // with 0 missing values.
        let t = read_csv(b"x,y\nNAN,inf\nNan,-Infinity\ninf,INF\nnan,5\n").unwrap();
        assert_eq!(t.columns[0].data.kind().dtype_name(), "object");
        assert_eq!(
            t.columns[1].data,
            ColumnData::Float(vec![f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, 5.0])
        );
        assert_eq!(t.columns[1].data.missing(), 0);
        assert_eq!(t.columns[1].data.nunique(), 3);
        assert_eq!(
            t.notices,
            vec![CsvNotice::NonFinite {
                columns: vec!["y".into()]
            }]
        );
    }

    #[test]
    fn cancellation_stops_the_reader() {
        let cancel = AtomicBool::new(true);
        assert_eq!(
            read_csv_cancellable(b"a,b\n1,2\n", &cancel).err(),
            Some(CsvError::Cancelled)
        );
        let cancel = AtomicBool::new(false);
        assert_eq!(
            read_csv_cancellable(b"a,b\n1,2\n", &cancel).ok(),
            read_csv(b"a,b\n1,2\n").ok()
        );
    }

    #[test]
    fn reports_malformed_rows_with_their_line() {
        let err = read_csv(b"a,b\n1,2\n3,4,5\n");
        assert_eq!(
            err.err(),
            Some(CsvError::Malformed {
                line: 3,
                found: 3,
                expected: 2
            })
        );
    }

    #[test]
    fn rejects_utf16() {
        assert_eq!(
            read_csv(b"\xFF\xFEa\0,\0b\0").err(),
            Some(CsvError::Encoding)
        );
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let missing = read_csv_file(Path::new("/nonexistent/mltutor/does-not-exist.csv"));
        assert!(matches!(missing, Err(CsvError::Io { .. })));
    }
}
