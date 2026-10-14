//! Ordinary least squares, as `sklearn.linear_model.LinearRegression()`.
//!
//! scikit-learn centres `X` and `y` (`X_offset`, `y_offset`), solves
//! `min ‖Xc·β − yc‖²` with `scipy.linalg.lstsq` and sets
//! `intercept_ = y_offset − X_offset·β`. We do the same: the centred problem is
//! solved by `linfa-linear` (Householder QR) when `Xc` has full column rank, and by
//! the SVD minimum-norm solution (what LAPACK `gelsd` returns) when it does not
//! (collinear or constant columns). Singular values below
//! `f64::EPSILON · σ_max` count as zero, LAPACK's default `rcond`.

use std::any::Any;

use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_linalg::svd::SVD;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::dataset::{Target, TaskType};
use crate::model::{
    check_input, LinearCoefficients, ModelError, Prediction, Predictor, TrainControl, TrainProgress,
};
use crate::stats::pairwise_sum;

/// A fitted `LinearRegression()` (`fit_intercept=True`, no positivity constraint).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearRegression {
    coefficients: LinearCoefficients,
    /// Rank of the centred design matrix (`rank_`).
    rank: usize,
}

impl LinearRegression {
    /// Fits the model. `y` must have one finite value per row of `x`.
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        control: &TrainControl<'_>,
    ) -> Result<Self, ModelError> {
        let (n, p) = x.dim();
        if n == 0 {
            return Err(ModelError::Empty);
        }
        if y.len() != n {
            return Err(ModelError::LengthMismatch {
                rows: n,
                targets: y.len(),
            });
        }
        check_input(x, p)?;
        if y.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::NonFinite);
        }
        control.check()?;
        control.report(TrainProgress {
            iteration: 0,
            max_iter: 0,
            loss: None,
            finished: false,
        });

        // np.average(X, axis=0) and np.average(y): pairwise sums like NumPy.
        let x_offset: Array1<f64> = x.axis_iter(Axis(1)).map(column_mean).collect();
        let y_offset = column_mean(y);
        let xc = &x - &x_offset;
        let yc = y.mapv(|v| v - y_offset);

        let (beta, rank) = solve_centered(xc, yc)?;
        let intercept = y_offset - x_offset.dot(&beta);
        control.report(TrainProgress {
            iteration: 0,
            max_iter: 0,
            loss: None,
            finished: true,
        });
        Ok(Self {
            coefficients: LinearCoefficients {
                coef: beta.insert_axis(Axis(0)),
                intercept: vec![intercept],
            },
            rank,
        })
    }

    /// Fits on a [`Target`]: regression values, or the numeric class values of a
    /// classification target (the app lets the student fit a line to Iris).
    pub fn fit_target(
        x: ArrayView2<'_, f64>,
        y: &Target,
        control: &TrainControl<'_>,
    ) -> Result<Self, ModelError> {
        let values = Array1::from(y.numeric());
        Self::fit(x, values.view(), control)
    }

    /// `coef_` (one value per feature).
    pub fn coef(&self) -> ArrayView1<'_, f64> {
        self.coefficients.coef.row(0)
    }

    /// `intercept_`.
    pub fn intercept(&self) -> f64 {
        self.coefficients.intercept.first().copied().unwrap_or(0.0)
    }

    /// `rank_` of the centred design matrix.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// `X @ coef_ + intercept_`.
    pub fn predict_values(&self, x: ArrayView2<'_, f64>) -> Result<Vec<f64>, ModelError> {
        check_input(x, self.coefficients.coef.ncols())?;
        let b = self.intercept();
        Ok(x.dot(&self.coef()).iter().map(|v| v + b).collect())
    }
}

fn column_mean(col: ArrayView1<'_, f64>) -> f64 {
    let n = col.len();
    if n == 0 {
        return f64::NAN;
    }
    match col.as_slice() {
        Some(s) => pairwise_sum(s) / n as f64,
        None => pairwise_sum(&col.to_vec()) / n as f64,
    }
}

/// Least squares on centred data; returns `(β, rank)`.
fn solve_centered(xc: Array2<f64>, yc: Array1<f64>) -> Result<(Array1<f64>, usize), ModelError> {
    let p = xc.ncols();
    if p == 0 {
        return Ok((Array1::zeros(0), 0));
    }
    let (_, sigma, _) = xc.svd(false, false).map_err(numerical)?;
    let s_max = sigma.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = f64::EPSILON * s_max;
    let rank = sigma.iter().filter(|&&s| s > cutoff).count();
    if rank == p && xc.nrows() >= p {
        let dataset = DatasetBase::new(xc, yc);
        let fitted = linfa_linear::LinearRegression::new()
            .with_intercept(false)
            .fit(&dataset)
            .map_err(numerical)?;
        let beta = fitted.params().to_owned();
        if beta.iter().all(|v| v.is_finite()) {
            return Ok((beta, rank));
        }
        let (xc, yc) = (dataset.records, dataset.targets);
        return min_norm_lstsq(xc, yc, rank);
    }
    min_norm_lstsq(xc, yc, rank)
}

/// `β = V · diag(1/σ) · Uᵀ · y` over the singular values above the cutoff.
fn min_norm_lstsq(
    xc: Array2<f64>,
    yc: Array1<f64>,
    rank: usize,
) -> Result<(Array1<f64>, usize), ModelError> {
    let (u, sigma, vt) = xc.svd(true, true).map_err(numerical)?;
    let (Some(u), Some(vt)) = (u, vt) else {
        return Err(ModelError::Numerical {
            message: "SVD sin vectores singulares".to_owned(),
        });
    };
    let s_max = sigma.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = f64::EPSILON * s_max;
    let uty = u.t().dot(&yc);
    let mut beta = Array1::<f64>::zeros(vt.ncols());
    for (k, &s) in sigma.iter().enumerate() {
        if s > cutoff {
            beta.scaled_add(uty[k] / s, &vt.row(k));
        }
    }
    Ok((beta, rank))
}

fn numerical(e: impl std::fmt::Display) -> ModelError {
    ModelError::Numerical {
        message: e.to_string(),
    }
}

impl Predictor for LinearRegression {
    fn task(&self) -> TaskType {
        TaskType::Regression
    }

    fn n_features(&self) -> usize {
        self.coefficients.coef.ncols()
    }

    fn classes(&self) -> Option<&[usize]> {
        None
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Prediction, ModelError> {
        self.predict_values(x).map(Prediction::Values)
    }

    fn predict_proba(&self, _x: ArrayView2<'_, f64>) -> Option<Result<Array2<f64>, ModelError>> {
        None
    }

    fn refit(
        &self,
        x: ArrayView2<'_, f64>,
        y: &Target,
        control: &TrainControl<'_>,
    ) -> Result<Box<dyn Predictor>, ModelError> {
        Ok(Box::new(Self::fit_target(x, y, control)?))
    }

    fn linear_coefficients(&self) -> Option<&LinearCoefficients> {
        Some(&self.coefficients)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn exact_line() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![1.0, 3.0, 5.0, 7.0];
        let m = LinearRegression::fit(x.view(), y.view(), &TrainControl::none())
            .unwrap_or_else(|e| panic!("{e}"));
        assert!((m.coef()[0] - 2.0).abs() < 1e-12);
        assert!((m.intercept() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn collinear_columns_give_min_norm_solution() {
        // Second column = 2 × first: scipy's lstsq gives β = (0.2, 0.4)·slope.
        let x = array![[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let m = LinearRegression::fit(x.view(), y.view(), &TrainControl::none())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(m.rank(), 1);
        assert!((m.coef()[0] - 0.2).abs() < 1e-12, "{:?}", m.coef());
        assert!((m.coef()[1] - 0.4).abs() < 1e-12, "{:?}", m.coef());
        assert!((m.intercept() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_column_gets_zero() {
        let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let y = array![2.0, 4.0, 6.0];
        let m = LinearRegression::fit(x.view(), y.view(), &TrainControl::none())
            .unwrap_or_else(|e| panic!("{e}"));
        assert!((m.coef()[0] - 2.0).abs() < 1e-12);
        assert!(m.coef()[1].abs() < 1e-12);
    }

    #[test]
    fn rejects_nan() {
        let x = array![[f64::NAN], [1.0]];
        let y = array![1.0, 2.0];
        assert_eq!(
            LinearRegression::fit(x.view(), y.view(), &TrainControl::none()),
            Err(ModelError::NonFinite)
        );
    }
}
