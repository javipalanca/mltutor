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

/// Plotly's default for `go.Histogram(x=values, nbinsx=nbins)`: the bins the
/// student saw in the Python app (🧠 Pesos, 📉 Activaciones), computed here so that
/// the UI receives `{ edges, counts }` instead of every raw value (fase5-redes.md K2).
///
/// A port of plotly.js `Axes.autoBin` (with `nbins`, linear axis) and the binning of
/// `histogram/calc.js`, checked against plotly.js 4.1 (the algorithm has not changed
/// since 1.x):
///
/// 1. `size0 = (max − min) / nbins`, rounded **up** to a "nice" step `dtick` ∈ {2, 5,
///    10}·10ᵏ (strictly greater, so there are usually *fewer* than `nbins` bins: 50
///    asked → 20–50 drawn).
/// 2. The first bin starts one step below the first multiple of `dtick` at or above
///    `min − 10⁻⁴·span`, shifted half a step (or half a unit with integer data) when
///    many values fall on bin edges (`autoShiftNumericBins`).
/// 3. Edges by repeated `Lib.increment` (which cleans float noise to 12 significant
///    digits); a value goes to bin `floor((v − start)/size + 1e-9)`.
/// 4. Empty bins at both ends are dropped, as Plotly does when it draws the bars.
///
/// All values equal: one bin of width 1 (`[floor(v), floor(v) + 1)`, or centred on
/// an integer `v`). NaN and ±∞ are ignored; no finite values → empty histogram.
pub fn histogram_plotly(values: &[f64], nbins: usize) -> Histogram {
    let data: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let Some((min, max)) = data
        .iter()
        .fold(None, |acc: Option<(f64, f64)>, &v| match acc {
            None => Some((v, v)),
            Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
        })
    else {
        return Histogram {
            edges: Vec::new(),
            counts: Vec::new(),
        };
    };
    let (start, end, size) = plotly_auto_bin(&data, min, max, nbins.max(1));
    // Edges: f = start; while f < end − size/1e6 { T = increment(f, size) }.
    let stop = end + (start - js_increment(start, size)) / 1e6;
    let mut edges = Vec::new();
    let mut f = start;
    while f < stop && edges.len() < 1_000_000 {
        let t = js_increment(f, size);
        edges.push(f);
        if t <= f {
            break;
        }
        f = t;
    }
    let n_bins = edges.len();
    edges.push(f);
    let mut counts = vec![0_u32; n_bins];
    for &v in &data {
        let bin = ((v - start) / size + 1e-9).floor();
        if bin >= 0.0 && bin < n_bins as f64 {
            counts[bin as usize] += 1;
        }
    }
    // Plotly draws from the first to the last non-empty bin.
    let first = counts.iter().position(|&c| c > 0).unwrap_or(0);
    let last = counts.iter().rposition(|&c| c > 0).unwrap_or(0);
    Histogram {
        edges: edges[first..=last + 1].to_vec(),
        counts: counts[first..=last].to_vec(),
    }
}

/// `Axes.autoBin(data, linearAxis, nbins)` → `(start, end, size)`; `min < max`.
fn plotly_auto_bin(data: &[f64], min: f64, max: f64, nbins: usize) -> (f64, f64, f64) {
    let size0 = (max - min) / nbins as f64;
    // Axes.autoTicks: base = 10^floor(log10(rough)), dtick = base · roundUp(rough/base,
    // [2, 5, 10]) (the first strictly greater). JS computes log10 as ln/LN10.
    let base = 10_f64.powf((size0.ln() / std::f64::consts::LN_10).floor());
    let ratio = size0 / base;
    let nice = [2.0, 5.0, 10.0]
        .into_iter()
        .find(|&r| r > ratio)
        .unwrap_or(10.0);
    let mut dtick = base * nice;
    if dtick == 0.0 || !dtick.is_finite() {
        dtick = 1.0;
    }
    // Axes.tickFirst on the range [min, max] expanded by 1e-4 of the span (tick0 = 0),
    // then one step back (tickIncrement 'reverse').
    let r0 = min - (max - min) * 1e-4;
    let first = (r0 / dtick).ceil() * dtick;
    let start = auto_shift_numeric_bins(js_increment(first, -dtick), data, dtick, min, max);
    let count = 1.0 + ((max - start) / dtick).floor();
    (start, start + count * dtick, dtick)
}

/// plotly.js `autoShiftNumericBins`.
fn auto_shift_numeric_bins(start: f64, data: &[f64], dtick: f64, min: f64, max: f64) -> f64 {
    // JS `%` keeps the sign of the dividend, like Rust's.
    let near_edge = |v: f64| (1.0 + (v - start) * 100.0 / dtick) % 100.0 < 2.0;
    let (mut edge, mut mid, mut ints) = (0usize, 0usize, 0usize);
    for &v in data {
        if v % 1.0 == 0.0 {
            ints += 1;
        }
        if near_edge(v) {
            edge += 1;
        }
        if near_edge(v + dtick / 2.0) {
            mid += 1;
        }
    }
    let n = data.len();
    let mut start = start;
    if ints == n {
        if dtick < 1.0 {
            start = min - 0.5 * dtick;
        } else {
            start -= 0.5;
            if start + dtick < min {
                start += dtick;
            }
        }
    } else if (mid as f64) < n as f64 * 0.1
        && (edge as f64 > n as f64 * 0.3 || near_edge(min) || near_edge(max))
    {
        let shift = dtick / 2.0;
        start += if start + shift < min { shift } else { -shift };
    }
    start
}

/// plotly.js `Lib.increment(x, delta)`: `x + delta` computed as `(x/δ + 1)·δ` for
/// small steps and, when the result prints with more digits than both operands
/// together, rounded to 12 significant digits (`toPrecision(12)`, unless that is in
/// exponential notation with a positive exponent).
fn js_increment(x: f64, delta: f64) -> f64 {
    if delta == 0.0 {
        return x;
    }
    let i = 1.0 / delta.abs();
    let n = if i > 1.0 {
        (i * x + i * delta) / i
    } else {
        x + delta
    };
    let len = js_number_string(n).len();
    if len > 16 && len >= js_number_string(delta).len() + js_number_string(x).len() {
        // toPrecision(12) is exponential when the exponent is < −6 or ≥ 12; only
        // "e+" is rejected.
        let exp = if n == 0.0 {
            0
        } else {
            n.abs().log10().floor() as i32
        };
        if exp < 12 {
            if let Ok(v) = format!("{n:.11e}").parse::<f64>() {
                return v;
            }
        }
    }
    n
}

/// JavaScript's `String(number)` (shortest round-trip digits, exponential below
/// 1e-6 and from 1e21).
fn js_number_string(v: f64) -> String {
    if v == 0.0 {
        return "0".to_owned();
    }
    if !v.is_finite() {
        return if v.is_nan() {
            "NaN".to_owned()
        } else if v > 0.0 {
            "Infinity".to_owned()
        } else {
            "-Infinity".to_owned()
        };
    }
    // `{:e}` gives the shortest round-trip digits: "1.2345e-7".
    let sci = format!("{:e}", v.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exp + 1;
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let sign = if n - 1 < 0 { '-' } else { '+' };
        let rest = if k > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        format!("{}{rest}e{sign}{}", &digits[..1], (n - 1).abs())
    };
    if v < 0.0 {
        format!("-{body}")
    } else {
        body
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn js_strings_like_javascript() {
        assert_eq!(js_number_string(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(js_number_string(1e-7), "1e-7");
        assert_eq!(js_number_string(-1.5e-7), "-1.5e-7");
        assert_eq!(js_number_string(123.0), "123");
        assert_eq!(js_number_string(0.000_001), "0.000001");
        assert_eq!(js_number_string(1e21), "1e+21");
        assert_eq!(js_increment(0.1, 0.2), 0.3);
    }

    #[test]
    fn plotly_bins_like_plotly() {
        // 0..=99 with nbinsx=10: size0 9.9 → dtick 10; tickFirst(−0.0099) = 0, one
        // step back = −10; integer data shift by −0.5, then +10 → start −0.5.
        let v: Vec<f64> = (0..100).map(f64::from).collect();
        let h = histogram_plotly(&v, 10);
        assert_eq!(h.edges.first().copied(), Some(-0.5));
        assert_eq!(h.counts, vec![10; 10]);
        assert_eq!(h.edges.len(), 11);
        // Floats in [−1, 1]: size0 0.04 (nbins 50) → dtick 0.05 (ratio 4 → 5): 40 bins.
        let v: Vec<f64> = (0..=2000).map(|i| -1.0 + f64::from(i) * 0.001).collect();
        let h = histogram_plotly(&v, 50);
        assert_eq!(h.counts.iter().sum::<u32>(), 2001);
        let w = h.edges[1] - h.edges[0];
        assert!((w - 0.05).abs() < 1e-12, "{w}");
        assert!(
            h.counts.len() <= 50 && h.counts.len() >= 40,
            "{}",
            h.counts.len()
        );
        assert!(h.edges[0] <= -1.0 && *h.edges.last().unwrap_or(&0.0) > 1.0);
        // Constant values: one bin of width 1 (plotly.js: [0, 1) and [2.5, 3.5)).
        let h = histogram_plotly(&[0.25; 7], 50);
        assert_eq!(h.edges, vec![0.0, 1.0]);
        assert_eq!(histogram_plotly(&[3.0], 20).edges, vec![2.5, 3.5]);
        assert_eq!(histogram_plotly(&[-7.5, -7.5], 20).edges, vec![-8.0, -7.0]);
        assert_eq!(h.counts, vec![7]);
        assert!(histogram_plotly(&[f64::NAN], 5).counts.is_empty());
    }
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
