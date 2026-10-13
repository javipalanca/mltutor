//! Bytes → [`RawTable`]: decoding, separator detection, parsing and pandas-like
//! type inference.

use std::borrow::Cow;
use std::path::Path;

use super::{ColumnData, CsvNotice, RawColumn, RawTable};
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
    read_csv(&bytes)
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
fn header_names(raw: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(raw.len());
    for (i, name) in raw.iter().enumerate() {
        let base = if name.is_empty() {
            format!("Unnamed: {i}")
        } else {
            name.clone()
        };
        let mut candidate = base.clone();
        let mut k = 1;
        while out.contains(&candidate) {
            candidate = format!("{base}.{k}");
            k += 1;
        }
        out.push(candidate);
    }
    out
}

/// Parses a number accepting the decimal comma (`"1,5"`) when `decimal_comma`.
/// The flag is set when a comma had to be replaced.
fn parse_float(field: &str, decimal_comma: bool, used_comma: &mut bool) -> Option<f64> {
    let t = field.trim();
    if let Ok(v) = t.parse::<f64>() {
        return Some(v);
    }
    if decimal_comma && t.matches(',').count() == 1 && !t.contains('.') {
        if let Ok(v) = t.replace(',', ".").parse::<f64>() {
            *used_comma = true;
            return Some(v);
        }
    }
    None
}

/// pandas-like inference: `int64` (all integers, no missing) → `float64` (numbers,
/// missing allowed) → `bool` (True/False only, no missing) → `object`.
fn infer_column(fields: Vec<String>, decimal_comma: bool) -> (ColumnData, bool) {
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
        .map(|f| {
            if is_na(f) {
                Some(f64::NAN)
            } else {
                parse_float(f, decimal_comma, &mut used_comma)
            }
        })
        .collect();
    if let Some(floats) = floats {
        return (ColumnData::Float(floats), used_comma);
    }
    if !any_na {
        let bools: Option<Vec<bool>> = fields
            .iter()
            .map(|f| {
                let t = f.trim();
                if TRUE_VALUES.contains(&t) {
                    Some(true)
                } else if FALSE_VALUES.contains(&t) {
                    Some(false)
                } else {
                    None
                }
            })
            .collect();
        if let Some(bools) = bools {
            return (ColumnData::Bool(bools), false);
        }
    }
    let texts = fields
        .into_iter()
        .map(|f| if is_na(&f) { None } else { Some(f) })
        .collect();
    (ColumnData::Text(texts), false)
}

/// Parses a CSV the way the Python app does with `pd.read_csv`, plus the D10
/// tolerances (Windows-1252, `;`/tab separators, decimal comma with `;`).
///
/// Validations, in the order the app applies them: no columns or no data rows →
/// [`CsvError::Empty`]; fewer than 2 columns → [`CsvError::TooFewColumns`]. Rows
/// with a different number of fields than the header → [`CsvError::Malformed`].
/// Blank lines are skipped. Never panics (fuzzed in `tests/csv_fuzz.rs`).
pub fn read_csv(bytes: &[u8]) -> Result<RawTable, CsvError> {
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

    let mut reader = ::csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .delimiter(separator)
        .from_reader(text.as_bytes());

    let mut header: Option<Vec<String>> = None;
    let mut columns: Vec<Vec<String>> = Vec::new();
    let mut n_rows = 0_usize;
    for result in reader.records() {
        let record = result.map_err(|e| CsvError::Malformed {
            line: e.position().map_or(0, ::csv::Position::line),
            found: 0,
            expected: header.as_ref().map_or(0, Vec::len),
        })?;
        // pandas `skip_blank_lines=True`.
        if record.len() == 1 && record.get(0).is_some_and(|f| f.trim().is_empty()) {
            continue;
        }
        match &header {
            None => {
                let raw: Vec<String> = record.iter().map(str::to_owned).collect();
                columns = vec![Vec::new(); raw.len()];
                header = Some(raw);
            }
            Some(h) => {
                if record.len() != h.len() {
                    return Err(CsvError::Malformed {
                        line: record.position().map_or(0, ::csv::Position::line),
                        found: record.len(),
                        expected: h.len(),
                    });
                }
                for (col, field) in columns.iter_mut().zip(record.iter()) {
                    col.push(field.to_owned());
                }
                n_rows += 1;
            }
        }
    }

    let Some(header) = header else {
        return Err(CsvError::Empty);
    };
    if n_rows == 0 || header.is_empty() {
        return Err(CsvError::Empty);
    }
    if header.len() < 2 {
        return Err(CsvError::TooFewColumns {
            found: header.len(),
        });
    }

    let decimal_comma = separator == b';';
    let mut comma_columns = Vec::new();
    let names = header_names(&header);
    let columns: Vec<RawColumn> = names
        .into_iter()
        .zip(columns)
        .map(|(name, fields)| {
            let (data, used_comma) = infer_column(fields, decimal_comma);
            if used_comma {
                comma_columns.push(name.clone());
            }
            RawColumn { name, data }
        })
        .collect();
    if !comma_columns.is_empty() {
        notices.push(CsvNotice::DecimalComma {
            columns: comma_columns,
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
        let names = header_names(&["a".into(), "a".into(), String::new(), "a".into()]);
        assert_eq!(names, vec!["a", "a.1", "Unnamed: 2", "a.2"]);
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
