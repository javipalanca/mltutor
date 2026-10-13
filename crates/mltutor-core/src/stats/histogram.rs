//! `np.linspace`, `np.histogram` and `np.histogram_bin_edges(..., "auto")`.

use serde::{Deserialize, Serialize};

use super::describe::quantile_linear;

/// Bin edges and counts of a histogram (`counts.len() == edges.len() - 1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    pub edges: Vec<f64>,
    pub counts: Vec<u32>,
}

/// `np.linspace(start, stop, num)` (endpoint included), with NumPy's formula
/// `start + i * ((stop - start) / (num - 1))` and the last value set to `stop`.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    match num {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let div = (num - 1) as f64;
            let delta = stop - start;
            let step = delta / div;
            let mut out: Vec<f64> = if step == 0.0 {
                (0..num).map(|i| (i as f64 / div) * delta + start).collect()
            } else {
                (0..num).map(|i| i as f64 * step + start).collect()
            };
            out[num - 1] = stop;
            out
        }
    }
}

/// Range of the finite values; NumPy widens an empty range by ±0.5.
fn outer_edges(values: &[f64]) -> Option<(f64, f64)> {
    let mut iter = values.iter().copied().filter(|v| v.is_finite());
    let first = iter.next()?;
    let (lo, hi) = iter.fold((first, first), |(lo, hi), v| (lo.min(v), hi.max(v)));
    if lo == hi {
        Some((lo - 0.5, hi + 0.5))
    } else {
        Some((lo, hi))
    }
}

/// Counts of `values` in the bins defined by ascending `edges`, as `np.histogram(a,
/// bins=edges)`: bin `i` holds `edges[i] <= v < edges[i + 1]`, the last bin also
/// includes its right edge, and values outside the range or NaN are ignored.
pub fn histogram_counts(values: &[f64], edges: &[f64]) -> Vec<u32> {
    if edges.len() < 2 {
        return Vec::new();
    }
    let n_bins = edges.len() - 1;
    let (lo, hi) = (edges[0], edges[n_bins]);
    let mut counts = vec![0_u32; n_bins];
    for &v in values {
        if !(v >= lo && v <= hi) {
            continue;
        }
        // Number of edges <= v, minus one; v == hi goes to the last bin.
        let idx = edges
            .partition_point(|&e| e <= v)
            .saturating_sub(1)
            .min(n_bins - 1);
        counts[idx] += 1;
    }
    counts
}

/// `np.histogram(values, bins)` with `bins` equal-width bins between the minimum
/// and the maximum (NaN ignored). Empty edges and counts if there are no values.
pub fn histogram(values: &[f64], bins: usize) -> Histogram {
    let Some((lo, hi)) = outer_edges(values) else {
        return Histogram {
            edges: Vec::new(),
            counts: Vec::new(),
        };
    };
    let edges = linspace(lo, hi, bins.max(1) + 1);
    let counts = histogram_counts(values, &edges);
    Histogram { edges, counts }
}

/// `np.histogram_bin_edges(values, bins="auto")` for float data (NaN ignored).
///
/// Bin width = min(Freedman–Diaconis, Sturges), or Sturges alone when the FD width
/// is 0 (IQR = 0); number of bins = `ceil(range / width)`, and a single bin when
/// both widths are 0.
pub fn histogram_bin_edges_auto(values: &[f64]) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(f64::total_cmp);
    let Some((first, last)) = outer_edges(&sorted) else {
        return Vec::new();
    };
    let n = sorted.len() as f64;
    let ptp = sorted[sorted.len() - 1] - sorted[0];
    let sturges = ptp / (n.log2() + 1.0);
    let iqr = quantile_linear(&sorted, 0.75) - quantile_linear(&sorted, 0.25);
    let fd = 2.0 * iqr * n.powf(-1.0 / 3.0);
    let width = if fd != 0.0 { fd.min(sturges) } else { sturges };
    let n_bins = if width != 0.0 {
        let bins = ((last - first) / width).ceil();
        if bins.is_finite() && bins >= 1.0 {
            // Bounded by the data: even 1e6 bins is far beyond the fixtures (max 542).
            bins.min(1.0e6) as usize
        } else {
            1
        }
    } else {
        1
    };
    linspace(first, last, n_bins + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_numpy() {
        // np.linspace(4.3, 7.9, 10)[5] == 6.300000000000001
        let l = linspace(4.3, 7.9, 10);
        assert_eq!(l[5], 6.300_000_000_000_001);
        assert_eq!(l[9], 7.9);
    }

    #[test]
    fn last_bin_is_closed() {
        let h = histogram(&[0.0, 1.0, 2.0, 3.0, f64::NAN], 3);
        assert_eq!(h.counts, vec![1, 1, 2]);
    }

    #[test]
    fn constant_data_gets_one_bin_around_the_value() {
        let e = histogram_bin_edges_auto(&[2.0, 2.0, 2.0]);
        assert_eq!(e, vec![1.5, 2.5]);
        assert!(histogram_bin_edges_auto(&[]).is_empty());
    }
}
