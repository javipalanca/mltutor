//! `np.linspace`, `np.histogram` and `np.histogram_bin_edges(..., "auto")`.

use serde::{Deserialize, Serialize};

use super::describe::quantile_linear;

/// Largest number of bins [`histogram_bin_edges_auto`] returns.
///
/// numpy has no limit: with 1999 values around 1.0 and one outlier at 1e9 the
/// Freedman–Diaconis width is tiny and `"auto"` asks for millions of bins, which
/// nobody can draw. Above this limit we use Sturges' rule alone (numpy's other
/// estimator, `ceil(log2(n)) + 1` bins). The largest histogram of the fixtures has
/// 542 bins (California, `AveBedrms`), so the limit does not change them.
pub const HIST_AUTO_MAX_BINS: usize = 1000;

/// Bin edges and counts of a histogram (`counts.len() == edges.len() - 1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    pub edges: Vec<f64>,
    pub counts: Vec<u32>,
}

/// `np.linspace(start, stop, num)` (endpoint included), with NumPy's formula
/// `start + i * ((stop - start) / (num - 1))` and the last value set to `stop`.
///
/// **Deviation from numpy:** when `stop - start` overflows (finite ends of
/// opposite sign near `f64::MAX`, e.g. `-1.7e308` and `1.7e308`) numpy returns
/// `inf`/NaN; here the values are the convex combination
/// `start · (1 − t) + stop · t` with `t = i / (num − 1)`, which cannot overflow.
/// Any range that numpy can represent keeps numpy's formula (bit-for-bit parity).
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    match num {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let div = (num - 1) as f64;
            let delta = stop - start;
            let step = delta / div;
            let mut out: Vec<f64> = if !delta.is_finite() && start.is_finite() && stop.is_finite() {
                (0..num)
                    .map(|i| {
                        let t = i as f64 / div;
                        start * (1.0 - t) + stop * t
                    })
                    .collect()
            } else if step == 0.0 {
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
/// and the maximum (NaN and ±inf ignored; numpy raises on infinite values). Empty
/// edges and counts if there are no finite values.
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

/// `np.histogram_bin_edges(values, bins="auto")` for float data (NaN and ±inf
/// ignored; numpy raises on infinite values).
///
/// Bin width = min(Freedman–Diaconis, Sturges), or Sturges alone when the FD width
/// is 0 (IQR = 0); number of bins = `ceil(range / width)`, and a single bin when
/// both widths are 0. **Deviation from numpy:** if that gives more than
/// [`HIST_AUTO_MAX_BINS`] bins, Sturges' width alone is used.
pub fn histogram_bin_edges_auto(values: &[f64]) -> Vec<f64> {
    let mut sorted: Vec<f64> = Vec::with_capacity(values.len());
    sorted.extend(values.iter().copied().filter(|v| v.is_finite()));
    sorted.sort_by(f64::total_cmp);
    let Some((first, last)) = outer_edges(&sorted) else {
        return Vec::new();
    };
    let n = sorted.len() as f64;
    let ptp = sorted[sorted.len() - 1] - sorted[0];
    let sturges = ptp / (n.log2() + 1.0);
    let iqr = quantile_linear(&sorted, 0.75) - quantile_linear(&sorted, 0.25);
    let fd = 2.0 * iqr * n.powf(-1.0 / 3.0);
    let bins_for = |width: f64| -> Option<usize> {
        if width == 0.0 {
            return Some(1);
        }
        let bins = ((last - first) / width).ceil();
        if !bins.is_finite() || bins < 1.0 {
            Some(1)
        } else if bins <= HIST_AUTO_MAX_BINS as f64 {
            Some(bins as usize)
        } else {
            None
        }
    };
    let width = if fd != 0.0 { fd.min(sturges) } else { sturges };
    let n_bins = bins_for(width)
        .or_else(|| bins_for(sturges))
        .unwrap_or(HIST_AUTO_MAX_BINS);
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
    fn huge_ranges_do_not_overflow() {
        // Review finding: `hi - lo` overflowed and the edges were NaN/inf.
        let h = histogram(&[-1.7e308, 1.7e308, 0.0], 30);
        assert_eq!(h.edges.len(), 31);
        assert!(h.edges.iter().all(|e| e.is_finite()), "{:?}", h.edges);
        assert!(h.edges.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(h.edges[0], -1.7e308);
        assert_eq!(h.edges[15], 0.0);
        assert_eq!(h.edges[30], 1.7e308);
        let mut want = vec![0_u32; 30];
        (want[0], want[15], want[29]) = (1, 1, 1);
        assert_eq!(h.counts, want);
        let auto = histogram_bin_edges_auto(&[-f64::MAX, f64::MAX]);
        assert!(auto.iter().all(|e| e.is_finite()), "{auto:?}");
        assert_eq!(
            histogram_counts(&[-f64::MAX, f64::MAX], &auto)
                .iter()
                .sum::<u32>(),
            2
        );
    }

    #[test]
    fn last_bin_is_closed() {
        let h = histogram(&[0.0, 1.0, 2.0, 3.0, f64::NAN], 3);
        assert_eq!(h.counts, vec![1, 1, 2]);
    }

    #[test]
    fn auto_bins_are_bounded_with_an_outlier() {
        // Review finding: 1999 values around 1.0 and an outlier at 1e9 used to give
        // 1 000 001 edges. Now: Sturges, ceil(log2(2000) + 1) = 12 bins.
        let mut v: Vec<f64> = (0..1999).map(|i| 1.0 + f64::from(i) * 1e-6).collect();
        v.push(1.0e9);
        let edges = histogram_bin_edges_auto(&v);
        assert_eq!(edges.len(), 13);
        assert_eq!(edges[0], 1.0);
        assert_eq!(edges[12], 1.0e9);
        let counts = histogram_counts(&v, &edges);
        assert_eq!(counts.iter().sum::<u32>(), 2000);
    }

    #[test]
    fn infinities_are_ignored() {
        let h = histogram(&[0.0, 1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY], 2);
        assert_eq!(h.edges, vec![0.0, 1.0, 2.0]);
        assert_eq!(h.counts, vec![1, 2]);
        let e = histogram_bin_edges_auto(&[1.0, f64::INFINITY, 3.0]);
        assert_eq!(e.first(), Some(&1.0));
        assert_eq!(e.last(), Some(&3.0));
        assert!(histogram(&[f64::INFINITY], 3).edges.is_empty());
    }

    #[test]
    fn constant_data_gets_one_bin_around_the_value() {
        let e = histogram_bin_edges_auto(&[2.0, 2.0, 2.0]);
        assert_eq!(e, vec![1.5, 2.5]);
        assert!(histogram_bin_edges_auto(&[]).is_empty());
    }
}
