//! Dictionary-encoded text column (`object` dtype in pandas).
//!
//! A CSV text column usually repeats a few values (`"si"`, `"no"`, `"Madrid"`…).
//! Storing one `Option<String>` per cell costs 24 bytes plus a heap block of at
//! least 16 bytes, even for `"ab"`. Here every distinct value is stored once and
//! each row keeps a 4-byte code:
//!
//! ```text
//! rows:   "si"  "no"  NA   "si"         codes:  [0, 1, MISSING, 0]
//!                                       values: "sino"  ends: [2, 4]
//! ```
//!
//! Codes follow the order of first appearance, which is also the tie order of
//! pandas' `value_counts()`, so `nunique`, `value_counts` and `LabelEncoder` work
//! on the few distinct values instead of on every row. The distinct values are
//! concatenated in one `String` (`values`, delimited by `ends`), so even a column
//! of all-distinct ids costs about 12 bytes per row plus its text.

use std::hash::BuildHasher;

use serde::{Deserialize, Serialize, Serializer};

/// Code of a missing value (pandas `NaN` in an `object` column).
const MISSING: u32 = u32::MAX;

/// A text column: one code per row and the distinct values (module docs).
///
/// Invariant: fewer than `u32::MAX` distinct values. The reader accepts files of
/// at most [`super::MAX_CSV_BYTES`] (100 MB), so a column has at most ~50 M values.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(from = "Vec<Option<String>>")]
pub struct TextColumn {
    /// Code of each row: index of its distinct value, or `MISSING`.
    codes: Vec<u32>,
    /// The distinct values, concatenated in order of first appearance.
    values: String,
    /// End (in bytes) of each distinct value in `values`.
    ends: Vec<usize>,
}

impl TextColumn {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Value of row `row`; `None` if it is missing or out of range.
    pub fn get(&self, row: usize) -> Option<&str> {
        self.code(row).and_then(|c| self.unique(c))
    }

    /// Code of row `row` (index into [`TextColumn::uniques`]); `None` if missing.
    pub fn code(&self, row: usize) -> Option<u32> {
        self.codes.get(row).copied().filter(|&c| c != MISSING)
    }

    /// The values of every row, `None` for missing ones.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Option<&str>> + '_ {
        self.codes.iter().map(|&c| self.unique(c))
    }

    /// Number of distinct non-missing values (`nunique()`).
    pub fn n_unique(&self) -> usize {
        self.ends.len()
    }

    /// Distinct value with code `code`; `None` for the missing code.
    pub fn unique(&self, code: u32) -> Option<&str> {
        distinct_value(&self.values, &self.ends, code)
    }

    /// Distinct values in order of first appearance (code 0, 1, …).
    pub fn uniques(&self) -> impl ExactSizeIterator<Item = &str> + '_ {
        (0..self.ends.len()).map(|i| {
            let start = i.checked_sub(1).map_or(0, |prev| self.ends[prev]);
            self.values.get(start..self.ends[i]).unwrap_or_default()
        })
    }

    /// Number of missing values (`isnull().sum()`).
    pub fn missing(&self) -> usize {
        self.codes.iter().filter(|&&c| c == MISSING).count()
    }

    /// Rows per distinct value, indexed by code (missing values not counted).
    pub fn counts(&self) -> Vec<u32> {
        let mut counts = vec![0_u32; self.n_unique()];
        for &c in &self.codes {
            if let Some(n) = counts.get_mut(c as usize) {
                *n += 1;
            }
        }
        counts
    }
}

/// Distinct value `code` of the concatenated `values` delimited by `ends`.
fn distinct_value<'a>(values: &'a str, ends: &[usize], code: u32) -> Option<&'a str> {
    let i = usize::try_from(code).ok()?;
    let end = *ends.get(i)?;
    let start = match i.checked_sub(1) {
        Some(prev) => *ends.get(prev)?,
        None => 0,
    };
    values.get(start..end)
}

impl<S: AsRef<str>> FromIterator<Option<S>> for TextColumn {
    fn from_iter<I: IntoIterator<Item = Option<S>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = TextColumnBuilder::with_capacity(iter.size_hint().0);
        for v in iter {
            builder.push(v.as_ref().map(AsRef::as_ref));
        }
        builder.finish()
    }
}

impl From<Vec<Option<String>>> for TextColumn {
    fn from(values: Vec<Option<String>>) -> Self {
        values.into_iter().collect()
    }
}

impl From<Vec<Option<&str>>> for TextColumn {
    fn from(values: Vec<Option<&str>>) -> Self {
        values.into_iter().collect()
    }
}

/// Same JSON as before the dictionary encoding: a list of strings or `null`.
impl Serialize for TextColumn {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

/// Builds a [`TextColumn`] row by row, looking each value up in a hash table of
/// the codes seen so far (the table is dropped by [`TextColumnBuilder::finish`]).
pub(crate) struct TextColumnBuilder {
    column: TextColumn,
    /// Codes of the distinct values, hashed by their text.
    index: hashbrown::HashTable<u32>,
    hasher: std::collections::hash_map::RandomState,
}

impl TextColumnBuilder {
    /// A builder for `rows` rows (reserved once).
    pub(crate) fn with_capacity(rows: usize) -> Self {
        Self {
            column: TextColumn {
                codes: Vec::with_capacity(rows),
                values: String::new(),
                ends: Vec::new(),
            },
            index: hashbrown::HashTable::new(),
            hasher: std::collections::hash_map::RandomState::new(),
        }
    }

    /// Appends one row (`None` = missing).
    pub(crate) fn push(&mut self, value: Option<&str>) {
        let code = match value {
            None => MISSING,
            Some(s) => self.intern(s),
        };
        self.column.codes.push(code);
    }

    /// Code of `s`, adding it as a new distinct value the first time.
    fn intern(&mut self, s: &str) -> u32 {
        let TextColumn { values, ends, .. } = &mut self.column;
        let hash = self.hasher.hash_one(s);
        let found = self
            .index
            .find(hash, |&code| distinct_value(values, ends, code) == Some(s));
        if let Some(&code) = found {
            return code;
        }
        // Invariant (type docs): fewer than u32::MAX distinct values.
        let code = u32::try_from(ends.len()).unwrap_or(MISSING - 1);
        values.push_str(s);
        ends.push(values.len());
        let hasher = &self.hasher;
        // The table may grow and rehash: the hash of a code is that of its text.
        self.index.insert_unique(hash, code, |&code| {
            hasher.hash_one(distinct_value(values, ends, code).unwrap_or_default())
        });
        code
    }

    /// The column, without the hash table and the excess capacity.
    pub(crate) fn finish(self) -> TextColumn {
        let mut column = self.column;
        column.values.shrink_to_fit();
        column.ends.shrink_to_fit();
        column
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_follow_first_appearance() {
        let col = TextColumn::from(vec![Some("si"), Some("no"), None, Some("si"), Some("")]);
        assert_eq!(col.len(), 5);
        assert_eq!(col.n_unique(), 3);
        assert_eq!(col.uniques().collect::<Vec<_>>(), vec!["si", "no", ""]);
        assert_eq!(
            col.iter().collect::<Vec<_>>(),
            vec![Some("si"), Some("no"), None, Some("si"), Some("")]
        );
        assert_eq!(col.code(2), None);
        assert_eq!(col.code(3), Some(0));
        assert_eq!(col.get(9), None);
        assert_eq!(col.missing(), 1);
        assert_eq!(col.counts(), vec![2, 1, 1]);
    }

    #[test]
    fn equal_content_gives_equal_columns() {
        let a = TextColumn::from(vec![Some("x".to_owned()), None]);
        let b: TextColumn = vec![Some("x"), None].into_iter().collect();
        assert_eq!(a, b);
        assert_ne!(a, TextColumn::from(vec![None, Some("x")]));
    }

    #[test]
    fn many_distinct_values_survive_rehashing() {
        let values: Vec<Option<String>> = (0..10_000).map(|i| Some(format!("v{i}"))).collect();
        let col = TextColumn::from(values.clone());
        assert_eq!(col.n_unique(), 10_000);
        let back: Vec<Option<String>> = col.iter().map(|v| v.map(str::to_owned)).collect();
        assert_eq!(back, values);
    }
}
