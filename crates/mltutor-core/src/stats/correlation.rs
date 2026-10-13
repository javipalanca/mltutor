//! Pearson correlation as `pandas.DataFrame.corr()`.

use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array2, ArrayView2};

/// Pearson correlation of two columns over the rows where both are finite, with
/// the same one-pass Welford update as pandas' `nancorr`. NaN if fewer than one
/// complete pair or if either column is constant (zero variance).
///
/// pandas only skips NaN (an infinite value makes the result NaN); here ±inf is
/// skipped like a missing value (`csv` module docs). No fixture has infinities.
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs = 0_u64;
    let (mut meanx, mut meany) = (0.0_f64, 0.0_f64);
    let (mut ssqdmx, mut ssqdmy, mut covxy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&vx, &vy) in x.iter().zip(y) {
        if !vx.is_finite() || !vy.is_finite() {
            continue;
        }
        nobs += 1;
        let dx = vx - meanx;
        let dy = vy - meany;
        // pandas multiplies by the reciprocal (`1. / nobs * dx`); keep it bit-identical.
        let inv = 1.0 / nobs as f64;
        meanx += inv * dx;
        meany += inv * dy;
        ssqdmx += (vx - meanx) * dx;
        ssqdmy += (vy - meany) * dy;
        covxy += (vx - meanx) * dy;
    }
    if nobs == 0 {
        return f64::NAN;
    }
    let divisor = (ssqdmx * ssqdmy).sqrt();
    if divisor != 0.0 {
        covxy / divisor
    } else {
        f64::NAN
    }
}

/// Symmetric correlation matrix of the columns of `data` (rows = observations).
pub fn pearson_matrix(data: ArrayView2<'_, f64>) -> Array2<f64> {
    pearson_matrix_cancellable(data, &AtomicBool::new(false)).unwrap_or_else(|| {
        // Unreachable: the flag above is never set.
        Array2::from_elem((data.ncols(), data.ncols()), f64::NAN)
    })
}

/// [`pearson_matrix`] with cooperative cancellation: `cancel` is checked before
/// each row of the matrix (k·(k+1)/2 column pairs in total); `None` if it was set.
pub fn pearson_matrix_cancellable(
    data: ArrayView2<'_, f64>,
    cancel: &AtomicBool,
) -> Option<Array2<f64>> {
    let k = data.ncols();
    let columns: Vec<Vec<f64>> = (0..k).map(|j| data.column(j).to_vec()).collect();
    let mut out = Array2::from_elem((k, k), f64::NAN);
    for i in 0..k {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        for j in 0..=i {
            let r = pearson(&columns[i], &columns[j]);
            out[[i, j]] = r;
            out[[j, i]] = r;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::pearson;

    #[test]
    fn perfect_and_constant() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert!((pearson(&x, &[2.0, 4.0, 6.0, 8.0]) - 1.0).abs() < 1e-15);
        assert!((pearson(&x, &[8.0, 6.0, 4.0, 2.0]) + 1.0).abs() < 1e-15);
        assert!(pearson(&x, &[1.0; 4]).is_nan());
        assert!(pearson(&[f64::NAN], &[1.0]).is_nan());
    }

    #[test]
    fn infinities_are_skipped_like_missing_values() {
        let x = [1.0, 2.0, f64::INFINITY, 3.0, 4.0];
        let y = [2.0, 4.0, 5.0, 6.0, f64::NEG_INFINITY];
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-15);
    }
}
