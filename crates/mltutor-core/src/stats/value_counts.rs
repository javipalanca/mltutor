//! `pandas.Series.value_counts()`.

use indexmap::IndexMap;

/// Counts in order of first appearance.
fn counts_by_appearance<'a, I>(labels: I) -> IndexMap<&'a str, u32>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut counts: IndexMap<&str, u32> = IndexMap::new();
    for label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }
    counts
}

/// `value_counts()`: most frequent first; ties keep the order of first appearance
/// (what pandas 2 returns in every fixture). Missing values must be filtered out by
/// the caller (`dropna=True`).
pub fn value_counts_by_frequency<'a, I>(labels: I) -> Vec<(String, u32)>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut out: Vec<(String, u32)> = counts_by_appearance(labels)
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
    // `sort_by` is stable, so ties stay in order of appearance.
    out.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    out
}

/// `value_counts().sort_index()` over string labels: ascending by label
/// (code-point order, like Python string comparison).
pub fn value_counts_sorted<'a, I>(labels: I) -> Vec<(String, u32)>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut out: Vec<(String, u32)> = counts_by_appearance(labels)
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_order_is_stable() {
        let v = ["b", "a", "c", "a", "b", "d"];
        assert_eq!(
            value_counts_by_frequency(v),
            vec![
                ("b".to_owned(), 2),
                ("a".to_owned(), 2),
                ("c".to_owned(), 1),
                ("d".to_owned(), 1)
            ]
        );
        assert_eq!(value_counts_sorted(v)[0], ("a".to_owned(), 2));
    }
}
