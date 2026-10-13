//! Univariate Gaussian KDE as drawn by `seaborn.kdeplot` (and so by `pairplot`).
//!
//! seaborn 0.13 fits `scipy.stats.gaussian_kde(x, bw_method="scott")`, multiplies
//! the factor by `bw_adjust = 1`, and evaluates it on
//! `linspace(min(x) - cut·bw, max(x) + cut·bw, gridsize)` with `cut = 3`,
//! `gridsize = 200` and `bw = sqrt(kde.covariance)`. SciPy's estimate is
//! `Σ wᵢ · φ((xᵢ − g) / h) / h` with `wᵢ = 1/n` and `h` the Cholesky factor of the
//! covariance (`sqrt(data_cov) · factor`).

use serde::{Deserialize, Serialize};

use super::describe::variance_ddof1;
use super::histogram::linspace;
use super::sum::pairwise_sum;

/// Grid points of each curve (seaborn's `gridsize`).
pub const KDE_GRIDSIZE: usize = 200;
/// How many bandwidths the grid extends past the data (seaborn's `cut`).
pub const KDE_CUT: f64 = 3.0;

/// A fitted and evaluated KDE.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Kde {
    /// Number of observations used.
    pub n: usize,
    /// Scott's factor `n_eff^(-1/5)`.
    pub scott_factor: f64,
    /// Kernel standard deviation `sqrt(var · factor²)` (seaborn's `bw`).
    pub bandwidth: f64,
    /// First grid point.
    pub grid_min: f64,
    /// Last grid point.
    pub grid_max: f64,
    /// Density on the grid, already multiplied by `weight`.
    pub density: Vec<f64>,
}

impl Kde {
    /// The evaluation grid (`linspace(grid_min, grid_max, density.len())`).
    pub fn grid(&self) -> Vec<f64> {
        linspace(self.grid_min, self.grid_max, self.density.len())
    }
}

/// seaborn's univariate KDE of `values` (NaN dropped) with Scott's rule, scaled by
/// `weight` (`common_norm`: `n_level / n_total`).
///
/// Returns `None` when seaborn skips the curve: fewer than 2 values or zero
/// variance (Digits `p0`).
pub fn gaussian_kde_scott(values: &[f64], weight: f64) -> Option<Kde> {
    let x: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = x.len();
    if n < 2 {
        return None;
    }
    let pandas_var = variance_ddof1(&x);
    if pandas_var.is_nan() || pandas_var == 0.0 {
        return None;
    }

    // gaussian_kde: weights = ones(n) / n, neff = 1 / sum(weights**2).
    let w = 1.0 / n as f64;
    let neff = 1.0 / pairwise_sum(&vec![w * w; n]);
    let factor = neff.powf(-1.0 / 5.0);

    // np.cov(x, aweights=weights, ddof=1).
    let w_sum = pairwise_sum(&vec![w; n]);
    let avg = pairwise_sum(&x.iter().map(|&v| v * w).collect::<Vec<_>>()) / w_sum;
    let fact = w_sum - pairwise_sum(&vec![w * w; n]) / w_sum;
    let centered: Vec<f64> = x.iter().map(|&v| v - avg).collect();
    let dot = pairwise_sum(&centered.iter().map(|&c| c * (c * w)).collect::<Vec<_>>());
    let data_cov = dot * (1.0 / fact);

    let covariance = data_cov * (factor * factor);
    let bandwidth = covariance.sqrt();
    let cho = data_cov.sqrt() * factor;

    let (lo, hi) = x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let grid = linspace(
        lo - bandwidth * KDE_CUT,
        hi + bandwidth * KDE_CUT,
        KDE_GRIDSIZE,
    );

    // scipy `gaussian_kernel_estimate`: whiten points and grid by the Cholesky
    // factor, then accumulate exp(-r²/2) · norm over the data in order.
    let norm = (2.0 * std::f64::consts::PI).powf(-0.5) / cho;
    let points: Vec<f64> = x.iter().map(|&v| v / cho).collect();
    let density = grid
        .iter()
        .map(|&g| {
            let gw = g / cho;
            let mut estimate = 0.0;
            for &p in &points {
                let r = p - gw;
                estimate += w * ((-(r * r) / 2.0).exp() * norm);
            }
            estimate * weight
        })
        .collect();

    Some(Kde {
        n,
        scott_factor: factor,
        bandwidth,
        grid_min: grid[0],
        grid_max: grid[KDE_GRIDSIZE - 1],
        density,
    })
}

#[cfg(test)]
mod tests {
    use super::gaussian_kde_scott;

    #[test]
    fn integrates_to_weight() {
        let x: Vec<f64> = (0..50).map(|i| f64::from(i).sin() * 3.0).collect();
        let kde = gaussian_kde_scott(&x, 0.5).unwrap_or_else(|| panic!("kde"));
        let dx = (kde.grid_max - kde.grid_min) / 199.0;
        let area: f64 = kde.density.iter().sum::<f64>() * dx;
        assert!((area - 0.5).abs() < 1e-2, "{area}");
    }

    #[test]
    fn skips_degenerate_levels() {
        assert!(gaussian_kde_scott(&[1.0], 1.0).is_none());
        assert!(gaussian_kde_scott(&[2.0, 2.0, 2.0], 1.0).is_none());
    }
}
