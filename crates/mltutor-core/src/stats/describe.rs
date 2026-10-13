//! `pandas.Series.describe()` for numeric columns.

use super::sum::pairwise_sum;

/// Row labels of `describe()`, in pandas order.
pub const DESCRIBE_STATS: [&str; 8] = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];

/// Mean ignoring NaN, as `pandas.Series.mean()`: NaN are replaced by 0, the whole
/// array is summed pairwise and divided by the number of valid values.
/// Returns NaN when there are no valid values.
pub fn mean(values: &[f64]) -> f64 {
    let count = values.iter().filter(|v| !v.is_nan()).count();
    if count == 0 {
        return f64::NAN;
    }
    let filled: Vec<f64> = values
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .collect();
    pairwise_sum(&filled) / count as f64
}

/// Sample standard deviation (ddof = 1) ignoring NaN, as `pandas.Series.std()`
/// (`nanops.nanvar`: mean, squared deviations with NaN set to 0, pairwise sum).
/// NaN with fewer than 2 valid values.
pub fn std_ddof1(values: &[f64]) -> f64 {
    variance_ddof1(values).sqrt()
}

/// Sample variance (ddof = 1) ignoring NaN, as `pandas.Series.var()`.
pub fn variance_ddof1(values: &[f64]) -> f64 {
    let count = values.iter().filter(|v| !v.is_nan()).count();
    if count < 2 {
        return f64::NAN;
    }
    let avg = mean(values);
    let sqr: Vec<f64> = values
        .iter()
        .map(|&v| {
            if v.is_nan() {
                0.0
            } else {
                let d = avg - v;
                d * d
            }
        })
        .collect();
    pairwise_sum(&sqr) / (count - 1) as f64
}

/// Quantile `q` of an ascending, NaN-free slice with NumPy's default `linear`
/// method (`np.percentile(..., method="linear")`), including NumPy's two-sided
/// interpolation formula (`_lerp`). NaN for an empty slice.
pub fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let virtual_index = (n - 1) as f64 * q;
    let (prev, next) = if virtual_index >= (n - 1) as f64 {
        (n - 1, n - 1)
    } else if virtual_index < 0.0 {
        (0, 0)
    } else {
        let p = virtual_index.floor() as usize;
        (p, p + 1)
    };
    let gamma = virtual_index - prev as f64;
    let (a, b) = (sorted[prev], sorted[next]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// The eight values of `describe()` for one column (see [`DESCRIBE_STATS`]),
/// ignoring NaN. Count is 0 and the rest NaN when there are no valid values.
pub fn describe_column(values: &[f64]) -> [f64; 8] {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    if count == 0 {
        return [
            0.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ];
    }
    [
        count as f64,
        mean(values),
        std_ddof1(values),
        sorted[0],
        quantile_linear(&sorted, 0.25),
        quantile_linear(&sorted, 0.5),
        quantile_linear(&sorted, 0.75),
        sorted[count - 1],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_ignores_nan() {
        let d = describe_column(&[1.0, f64::NAN, 2.0, 3.0, 4.0]);
        assert_eq!(d[0], 4.0);
        assert_eq!(d[1], 2.5);
        assert!((d[2] - 1.290_994_448_735_805_6).abs() < 1e-15);
        assert_eq!(&d[3..], &[1.0, 1.75, 2.5, 3.25, 4.0]);
    }

    #[test]
    fn empty_and_single() {
        let d = describe_column(&[f64::NAN]);
        assert_eq!(d[0], 0.0);
        assert!(d[1].is_nan());
        let d = describe_column(&[7.0]);
        assert_eq!(d[1], 7.0);
        assert!(d[2].is_nan());
        assert_eq!(d[5], 7.0);
    }
}
