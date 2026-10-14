//! `pandas.Series.describe()` for numeric columns.

use super::sum::pairwise_sum_by;

/// Row labels of `describe()`, in pandas order.
pub const DESCRIBE_STATS: [&str; 8] = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];

/// Mean ignoring missing values, as `pandas.Series.mean()`: missing values are
/// replaced by 0, the whole array is summed pairwise and divided by the number of
/// valid values. Returns NaN when there are no valid values.
///
/// "Missing" is NaN **and ±inf** (non-finite values are left out of every
/// statistic; `csv` module docs). pandas only skips NaN and would return ±inf or
/// NaN; no fixture has infinite values.
pub fn mean(values: &[f64]) -> f64 {
    let count = values.iter().filter(|v| v.is_finite()).count();
    if count == 0 {
        return f64::NAN;
    }
    pairwise_sum_by(values.len(), |i| finite_or_zero(values[i])) / count as f64
}

fn finite_or_zero(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Sample standard deviation (ddof = 1) ignoring missing values (NaN and ±inf, see
/// [`mean`]), as `pandas.Series.std()` (`nanops.nanvar`: mean, squared deviations
/// with missing values set to 0, pairwise sum). NaN with fewer than 2 valid values.
pub fn std_ddof1(values: &[f64]) -> f64 {
    variance_ddof1(values).sqrt()
}

/// Sample variance (ddof = 1) ignoring missing values (NaN and ±inf), as
/// `pandas.Series.var()`.
pub fn variance_ddof1(values: &[f64]) -> f64 {
    let count = values.iter().filter(|v| v.is_finite()).count();
    if count < 2 {
        return f64::NAN;
    }
    let avg = mean(values);
    let sqr = |i: usize| {
        let v = values[i];
        if v.is_finite() {
            let d = avg - v;
            d * d
        } else {
            0.0
        }
    };
    pairwise_sum_by(values.len(), sqr) / (count - 1) as f64
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
/// ignoring missing values (NaN and ±inf, see [`mean`]). Count is 0 and the rest
/// NaN when there are no valid values.
pub fn describe_column(values: &[f64]) -> [f64; 8] {
    // Reserved up front: a growing `Vec` would briefly need 3× the column.
    let mut sorted: Vec<f64> = Vec::with_capacity(values.len());
    sorted.extend(values.iter().copied().filter(|v| v.is_finite()));
    sorted.sort_by(f64::total_cmp);
    summary(mean(values), std_ddof1(values), &sorted)
}

/// [`describe_column`] of an owned column without a sorted copy: the mean and
/// the standard deviation are taken in the original order (pandas' summation
/// order), then `values` keeps only its finite values, sorted ascending.
pub(crate) fn describe_in_place(values: &mut Vec<f64>) -> [f64; 8] {
    let (avg, std) = (mean(values), std_ddof1(values));
    values.retain(|v| v.is_finite());
    values.sort_by(f64::total_cmp);
    summary(avg, std, values)
}

/// `describe()` from the mean, the standard deviation and the sorted finite values.
fn summary(mean: f64, std: f64, sorted: &[f64]) -> [f64; 8] {
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
        mean,
        std,
        sorted[0],
        quantile_linear(sorted, 0.25),
        quantile_linear(sorted, 0.5),
        quantile_linear(sorted, 0.75),
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
    fn describe_ignores_infinities() {
        let d = describe_column(&[1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY, 3.0, 4.0]);
        assert_eq!(d[0], 4.0);
        assert_eq!(d[1], 2.5);
        assert!((d[2] - 1.290_994_448_735_805_6).abs() < 1e-15);
        assert_eq!(&d[3..], &[1.0, 1.75, 2.5, 3.25, 4.0]);
        assert!(describe_column(&[f64::INFINITY])[1].is_nan());
    }

    #[test]
    fn in_place_matches_the_copying_version() {
        let values = vec![3.0, f64::NAN, -1.0, f64::INFINITY, 0.1, 0.2, 7.5, 0.3];
        let mut owned = values.clone();
        let a = describe_column(&values);
        let b = describe_in_place(&mut owned);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_eq!(owned, vec![-1.0, 0.1, 0.2, 0.3, 3.0, 7.5]);
        let mut empty = vec![f64::NAN];
        assert_eq!(describe_in_place(&mut empty)[0], 0.0);
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
