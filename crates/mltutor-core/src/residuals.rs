//! "📈 Análisis de Residuos" (fase2-regresion.md C7, `viz/residual.py`).

use serde::{Deserialize, Serialize};

use crate::metrics::r2_score;
use crate::stats::{histogram, mean, pairwise_sum, std_ddof1, Histogram};

/// Bins of the residuals histogram.
pub const RESIDUAL_BINS: usize = 20;

/// The three checks of "🔍 Interpretación de los Residuos".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualChecks {
    /// `|mean| < 0.1·std`.
    pub mean_ok: bool,
    /// `|r| < 0.1` of `linregress(ŷ, residuals)`; `None` with ≤ 10 samples (Python
    /// crashes with a `NameError` there; we omit the bullet).
    pub corr_ok: Option<bool>,
    /// `|skew| < 1`.
    pub normal_ok: bool,
}

impl ResidualChecks {
    /// Whether the "💡 Sugerencias de mejora" box is shown.
    pub fn any_failed(&self) -> bool {
        !self.mean_ok || self.corr_ok == Some(false) || !self.normal_ok
    }
}

/// Everything C7 shows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualAnalysis {
    /// `y − ŷ`.
    pub residuals: Vec<f64>,
    /// Mean of the residuals.
    pub mean: f64,
    /// Standard deviation, ddof = 1 (pandas `Series.std()`).
    pub std: f64,
    /// `scipy.stats.skew` (biased).
    pub skew: f64,
    /// Pearson r of `linregress(ŷ, residuals)` when there are more than 10 samples.
    pub linregress_r: Option<f64>,
    /// `R²` of the test predictions (D22: Python always printed 0.0000).
    pub r2: f64,
    /// `ax.hist(residuals, bins=20)`.
    pub hist: Histogram,
    pub checks: ResidualChecks,
}

/// `scipy.stats.skew(x)` (`bias=True`): `m3 / m2^1.5` with central moments; 0 for
/// constant data (scipy returns NaN with a warning).
pub fn skew(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.is_empty() {
        return f64::NAN;
    }
    let m = pairwise_sum(x) / n;
    let d: Vec<f64> = x.iter().map(|v| v - m).collect();
    let m2 = pairwise_sum(&d.iter().map(|v| v * v).collect::<Vec<_>>()) / n;
    let m3 = pairwise_sum(&d.iter().map(|v| v * v * v).collect::<Vec<_>>()) / n;
    if m2 == 0.0 {
        return 0.0;
    }
    m3 / m2.powf(1.5)
}

/// Pearson correlation of `scipy.stats.linregress(x, y)` (`ssxym / sqrt(ssxm·ssym)`,
/// clipped to [-1, 1]); 0 if either variable is constant.
pub fn linregress_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return f64::NAN;
    }
    let nf = n as f64;
    let xm = pairwise_sum(&x[..n]) / nf;
    let ym = pairwise_sum(&y[..n]) / nf;
    let sxx = pairwise_sum(
        &x[..n]
            .iter()
            .map(|v| (v - xm) * (v - xm))
            .collect::<Vec<_>>(),
    ) / nf;
    let syy = pairwise_sum(
        &y[..n]
            .iter()
            .map(|v| (v - ym) * (v - ym))
            .collect::<Vec<_>>(),
    ) / nf;
    let sxy = pairwise_sum(
        &(0..n)
            .map(|i| (x[i] - xm) * (y[i] - ym))
            .collect::<Vec<_>>(),
    ) / nf;
    if sxx == 0.0 || syy == 0.0 {
        return 0.0;
    }
    (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0)
}

/// C7 for test values `y_true` and predictions `y_pred`.
pub fn analysis(y_true: &[f64], y_pred: &[f64]) -> ResidualAnalysis {
    let n = y_true.len().min(y_pred.len());
    let residuals: Vec<f64> = (0..n).map(|i| y_true[i] - y_pred[i]).collect();
    let m = mean(&residuals);
    let std = std_ddof1(&residuals);
    let sk = skew(&residuals);
    let r = (n > 10).then(|| linregress_r(&y_pred[..n], &residuals));
    ResidualAnalysis {
        mean: m,
        std,
        skew: sk,
        linregress_r: r,
        r2: r2_score(&y_true[..n], &y_pred[..n]),
        hist: histogram(&residuals, RESIDUAL_BINS),
        checks: ResidualChecks {
            mean_ok: m.abs() < 0.1 * std,
            corr_ok: r.map(|r| r.abs() < 0.1),
            normal_ok: sk.abs() < 1.0,
        },
        residuals,
    }
}
