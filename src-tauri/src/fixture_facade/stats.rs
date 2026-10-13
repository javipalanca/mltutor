//! Small statistics helpers for [`super::FixtureFacade`] (pairplot diagonals and the
//! views of user CSVs). They follow numpy/pandas/seaborn closely enough for development
//! and are checked against `tests/parity/stats`; the reference implementation lives in
//! `mltutor-core::stats`.

use crate::ipc::{to_u32, DescribeTable, KdeCurve, Matrix};

pub const DESCRIBE_STATS: [&str; 8] = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
const KDE_GRID: usize = 200;
const KDE_CUT: f64 = 3.0;

fn finite(values: &[f64]) -> Vec<f64> {
    values.iter().copied().filter(|v| v.is_finite()).collect()
}

pub fn mean(values: &[f64]) -> f64 {
    let v = finite(values);
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Sample standard deviation (ddof = 1), ignoring NaN.
pub fn std(values: &[f64]) -> f64 {
    let v = finite(values);
    if v.len() < 2 {
        return f64::NAN;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// numpy `quantile(method="linear")` of already sorted finite values.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let pos = q * (n - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = pos - lo as f64;
            sorted[lo] + (sorted[hi] - sorted[lo]) * frac
        }
    }
}

fn sorted_finite(values: &[f64]) -> Vec<f64> {
    let mut v = finite(values);
    v.sort_by(f64::total_cmp);
    v
}

pub fn describe_column(values: &[f64]) -> [f64; 8] {
    let sorted = sorted_finite(values);
    let count = sorted.len() as f64;
    let (min, max) = match (sorted.first(), sorted.last()) {
        (Some(min), Some(max)) => (*min, *max),
        _ => (f64::NAN, f64::NAN),
    };
    [
        count,
        mean(&sorted),
        std(&sorted),
        min,
        quantile_sorted(&sorted, 0.25),
        quantile_sorted(&sorted, 0.5),
        quantile_sorted(&sorted, 0.75),
        max,
    ]
}

pub fn describe(columns: &[(String, &[f64])], excluded: Vec<String>) -> DescribeTable {
    let stats: Vec<[f64; 8]> = columns.iter().map(|(_, v)| describe_column(v)).collect();
    let values = (0..8).flat_map(|row| stats.iter().map(move |col| col[row]));
    DescribeTable {
        columns: columns.iter().map(|(name, _)| name.clone()).collect(),
        stats: DESCRIBE_STATS.iter().map(|s| (*s).to_owned()).collect(),
        values: Matrix::from_f64(8, columns.len(), values.collect::<Vec<_>>()),
        excluded_columns: excluded,
    }
}

/// Pearson correlation over pairwise complete observations; NaN if a variance is 0.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (*x, *y))
        .collect();
    if pairs.len() < 2 {
        return f64::NAN;
    }
    let n = pairs.len() as f64;
    let (mx, my) = pairs
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x, sy + y));
    let (mx, my) = (mx / n, my / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in &pairs {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0)
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n)
        .map(|i| {
            if i == n - 1 {
                end
            } else {
                start + step * i as f64
            }
        })
        .collect()
}

fn range_of(values: &[f64]) -> Option<(f64, f64)> {
    let v = finite(values);
    let min = v.iter().copied().reduce(f64::min)?;
    let max = v.iter().copied().reduce(f64::max)?;
    Some(if min == max {
        (min - 0.5, max + 0.5)
    } else {
        (min, max)
    })
}

/// `np.histogram(values, bins)` with equal bins (the last one closed).
pub fn histogram(values: &[f64], bins: usize) -> (Vec<f64>, Vec<u32>) {
    let Some((lo, hi)) = range_of(values) else {
        return (linspace(0.0, 1.0, bins + 1), vec![0; bins]);
    };
    let edges = linspace(lo, hi, bins + 1);
    let counts = count_in_bins(values, &edges);
    (edges, counts)
}

/// Counts with numpy semantics: `[e_i, e_{i+1})`, last bin closed.
pub fn count_in_bins(values: &[f64], edges: &[f64]) -> Vec<u32> {
    let bins = edges.len().saturating_sub(1);
    let mut counts = vec![0u32; bins];
    if bins == 0 {
        return counts;
    }
    let (lo, hi) = (edges[0], edges[bins]);
    for &v in values.iter().filter(|v| v.is_finite()) {
        if v < lo || v > hi {
            continue;
        }
        // Binary search over the edges, like np.searchsorted(edges, v, "right") - 1.
        let index = edges
            .partition_point(|e| *e <= v)
            .saturating_sub(1)
            .min(bins - 1);
        counts[index] += 1;
    }
    counts
}

/// `np.histogram_bin_edges(values, "auto")`.
pub fn bin_edges_auto(values: &[f64]) -> Vec<f64> {
    let sorted = sorted_finite(values);
    let Some((lo, hi)) = range_of(&sorted) else {
        return linspace(0.0, 1.0, 2);
    };
    let n = sorted.len() as f64;
    // numpy measures the bin width on the data range (0 for a constant column → 1 bin).
    let range = sorted[sorted.len() - 1] - sorted[0];
    let sturges = range / (n.log2() + 1.0);
    let iqr = quantile_sorted(&sorted, 0.75) - quantile_sorted(&sorted, 0.25);
    let fd = 2.0 * iqr * n.powf(-1.0 / 3.0);
    let width = if fd > 0.0 { fd.min(sturges) } else { sturges };
    let bins = if width > 0.0 {
        (((hi - lo) / width).ceil() as usize).max(1)
    } else {
        1
    };
    linspace(lo, hi, bins + 1)
}

/// seaborn `kdeplot` defaults: `gaussian_kde` with Scott's factor, grid
/// `linspace(min - 3bw, max + 3bw, 200)` and density × weight (`common_norm`).
pub fn kde(level: &str, values: &[f64], weight: f64) -> KdeCurve {
    let v = finite(values);
    let n = v.len();
    let sd = std(&v);
    let skipped = KdeCurve {
        level: level.to_owned(),
        n: to_u32(n),
        bandwidth: None,
        grid_min: 0.0,
        grid_max: 0.0,
        density: Vec::new(),
    };
    if n < 2 || sd.is_nan() || sd <= 0.0 {
        return skipped;
    }
    let bandwidth = (n as f64).powf(-0.2) * sd;
    let (min, max) = range_of(&v).unwrap_or((0.0, 0.0));
    let grid = linspace(
        min - KDE_CUT * bandwidth,
        max + KDE_CUT * bandwidth,
        KDE_GRID,
    );
    let norm = weight / (n as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt());
    let density = grid
        .iter()
        .map(|g| {
            v.iter()
                .map(|x| (-0.5 * ((g - x) / bandwidth).powi(2)).exp())
                .sum::<f64>()
                * norm
        })
        .collect();
    KdeCurve {
        level: level.to_owned(),
        n: to_u32(n),
        bandwidth: Some(bandwidth),
        grid_min: grid[0],
        grid_max: grid[KDE_GRID - 1],
        density,
    }
}

/// Labels by descending frequency; ties keep the order of first appearance.
pub fn value_counts_by_frequency<'a>(
    values: impl IntoIterator<Item = &'a str>,
) -> Vec<(String, u32)> {
    let mut counts: Vec<(String, u32)> = Vec::new();
    for value in values {
        match counts.iter_mut().find(|(label, _)| label == value) {
            Some((_, count)) => *count += 1,
            None => counts.push((value.to_owned(), 1)),
        }
    }
    counts.sort_by_key(|c| std::cmp::Reverse(c.1));
    counts
}

/// Formats a number like pandas' `str()` of a value (integers without ".0").
pub fn format_value(v: f64) -> String {
    if v.is_nan() {
        "nan".to_owned()
    } else if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_are_linear() {
        let d = describe_column(&[1.0, 2.0, 3.0, 4.0, f64::NAN]);
        assert_eq!(d[0], 4.0);
        assert_eq!(d[1], 2.5);
        assert!((d[2] - 1.290_994_448_735_805_6).abs() < 1e-12);
        assert_eq!(&d[3..], &[1.0, 1.75, 2.5, 3.25, 4.0]);
    }

    #[test]
    fn histogram_closes_the_last_bin() {
        let (edges, counts) = histogram(&[0.0, 1.0, 2.0, 3.0], 3);
        assert_eq!(edges, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(counts, vec![1, 1, 2]);
    }

    #[test]
    fn constant_columns_have_nan_correlation_and_no_kde() {
        assert!(pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_nan());
        assert!((pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.5]) - 0.9986).abs() < 1e-3);
        assert!(kde("x", &[2.0, 2.0], 1.0).density.is_empty());
    }

    #[test]
    fn value_counts_are_stable() {
        let counts = value_counts_by_frequency(["b", "a", "b", "c", "a", "d"]);
        assert_eq!(
            counts,
            vec![
                ("b".into(), 2),
                ("a".into(), 2),
                ("c".into(), 1),
                ("d".into(), 1)
            ]
        );
    }
}
