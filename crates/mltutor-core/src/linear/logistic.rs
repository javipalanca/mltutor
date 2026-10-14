//! `sklearn.linear_model.LogisticRegression(C=1.0, solver="lbfgs", max_iter, tol=1e-4)`:
//! L2-regularised logistic regression, binary or multinomial (softmax).
//!
//! # Objective (the same as scikit-learn 1.7 and linfa-logistic)
//!
//! scikit-learn minimises `C · Σᵢ loss(yᵢ, zᵢ) + ½‖W‖²`, written as
//! `(1/n) Σᵢ loss(yᵢ, zᵢ) + ‖W‖² / (2·C·n)` (the intercept is not penalised).
//! `linfa-logistic` minimises `Σᵢ loss + (α/2)‖W‖²`: the same minimiser with
//! `α = 1/C` (checked against scikit-learn in `tests/parity_linear_models.rs`).
//! Binary problems use one row of coefficients and the sigmoid, with
//! `classes_[1]` as the positive class; three or more classes use one row per
//! class and the softmax (`multi_class="multinomial"`, scikit-learn's default).
//!
//! # Solver
//!
//! The *minimiser* is scikit-learn's; the path to it is not (fase2-regresion.md D19):
//!
//! * **Damped Newton** (exact Hessian, Cholesky, Armijo backtracking) when the
//!   problem has at most [`NEWTON_MAX_PARAMS`] parameters (`k·(p+1)`): every
//!   built-in classification dataset except Digits. It converges in ~10–30
//!   iterations even on unscaled, collinear data such as Breast Cancer, where
//!   scipy's L-BFGS (scikit-learn) stops at `max_iter=1000` 0.20 away from the
//!   optimum in probability, and where L-BFGS needs >20 000 iterations.
//! * **L-BFGS** (10 correction pairs, More–Thuente line search, from `argmin`:
//!   exactly what `linfa-logistic` builds) for larger problems. We drive argmin's
//!   loop ourselves instead of calling `linfa-logistic`, because `linfa` neither
//!   lets us cancel the training nor reports iterations or convergence.
//!
//! Both work on **standardised features** (a change of variables `vⱼ = wⱼ·sⱼ`,
//! with the penalty still on `wⱼ = vⱼ/sⱼ`: same minimiser, better conditioning),
//! start from zero like scikit-learn and run until the gradient is at the noise
//! floor ([`INNER_GRADIENT_TOL`]) or `max_iter` is reached. scikit-learn stops much
//! earlier (`max|∇f| ≤ tol = 1e-4`) at a point that depends on its path (MEMORY:
//! 1.6e-3 of difference in Iris probabilities), so our models match its
//! *converged* solution (`tol=1e-12`) to ~1e-7 and the app's model within the
//! fixture tolerance (2e-3).
//!
//! In the multinomial case the objective is invariant to adding a constant to all
//! intercepts; like scikit-learn's L-BFGS started at zero, we return the solution
//! with `Σₖ intercept_ₖ = 0`.
//!
//! [`FitReport::converged`] is scikit-learn's criterion (`max|∇f| ≤ tol` in the
//! original variables) evaluated at our solution, so the "not converged" warning
//! (D19) means the same as Python's `ConvergenceWarning` would at the optimum.
//! `n_iter` counts Newton or L-BFGS iterations, so it is much smaller than
//! scikit-learn's `n_iter_` with Newton.

use std::any::Any;

use argmin::core::{CostFunction, Gradient, IterState, Problem, Solver, State, TerminationStatus};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::dataset::{Target, TaskType};
use crate::model::{
    argmax, check_input, FitReport, LinearCoefficients, ModelError, Prediction, Predictor,
    TrainControl, TrainProgress,
};
use crate::stats::pairwise_sum;

/// Gradient norm (of the mean objective, in standardised variables) at which the
/// solver stops before `max_iter`.
pub const INNER_GRADIENT_TOL: f64 = 1e-10;

/// Largest number of parameters (`k·(p+1)`) solved with Newton's method; above
/// it the Hessian (`n·d²` per iteration) is too expensive and L-BFGS is used.
pub const NEWTON_MAX_PARAMS: usize = 300;

/// L-BFGS memory (scikit-learn/scipy and linfa use 10).
const LBFGS_MEMORY: usize = 10;

/// Hyperparameters of `LogisticRegression` that the app exposes or fixes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LogisticParams {
    /// Inverse of the regularisation strength (`C=1.0`).
    pub c: f64,
    /// `max_iter` (the app's slider: 100..=2000, 1000 by default).
    pub max_iter: u32,
    /// `tol=1e-4`: scikit-learn's stopping criterion, used for [`FitReport::converged`].
    pub tol: f64,
}

impl Default for LogisticParams {
    fn default() -> Self {
        Self {
            c: 1.0,
            max_iter: 1000,
            tol: 1e-4,
        }
    }
}

/// A fitted logistic regression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogisticRegression {
    params: LogisticParams,
    /// Class codes seen in training, sorted (`classes_`).
    classes: Vec<usize>,
    coefficients: LinearCoefficients,
    report: FitReport,
}

impl LogisticRegression {
    /// Fits the model on class codes `y` (one per row of `x`).
    pub fn fit(
        x: ArrayView2<'_, f64>,
        y: &[usize],
        params: LogisticParams,
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
        if !(params.c.is_finite() && params.c > 0.0) {
            return Err(ModelError::InvalidSelection {
                reason: format!("C debe ser positivo (es {})", params.c),
            });
        }
        check_input(x, p)?;

        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        if classes.len() < 2 {
            return Err(ModelError::TooFewClasses {
                found: classes.len(),
            });
        }
        // Column of each row in the one-hot target (position in `classes`).
        let y_idx: Vec<usize> = y
            .iter()
            .map(|c| classes.binary_search(c).unwrap_or(0))
            .collect();

        let scaling = Scaling::new(x);
        let xs = scaling.apply(x);
        let k = if classes.len() == 2 { 1 } else { classes.len() };
        let problem = Objective {
            x: xs.view(),
            y: &y_idx,
            k,
            inv_scale_sq: scaling.std.mapv(|s| 1.0 / (s * s)),
            lambda: 1.0 / (params.c * n as f64),
        };

        let max_iter = u64::from(params.max_iter);
        let n_params = k * (p + 1);
        let (theta, n_iter) = if n_params <= NEWTON_MAX_PARAMS {
            run_newton(&problem, n_params, max_iter, control)?
        } else {
            run_lbfgs(problem.clone(), n_params, max_iter, control)?
        };

        let (coef, intercept) = scaling.unscale(theta.view(), k, p);
        let grad = problem.gradient_original(theta.view(), &scaling);
        let max_grad = grad.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        let report = FitReport {
            n_iter,
            max_iter,
            converged: max_grad <= params.tol,
        };
        control.report(TrainProgress {
            iteration: n_iter,
            max_iter,
            loss: Some(problem.loss(theta.view())),
            finished: true,
        });
        Ok(Self {
            params,
            classes,
            coefficients: LinearCoefficients { coef, intercept },
            report,
        })
    }

    /// Fits on a [`Target`]. A regression target is rejected (D26).
    pub fn fit_target(
        x: ArrayView2<'_, f64>,
        y: &Target,
        params: LogisticParams,
        control: &TrainControl<'_>,
    ) -> Result<Self, ModelError> {
        let codes = y.class_codes().ok_or(ModelError::InvalidTarget)?;
        Self::fit(x, &codes, params, control)
    }

    /// Hyperparameters.
    pub fn params(&self) -> LogisticParams {
        self.params
    }

    /// `coef_` (`1 × n_features` for binary problems, `n_classes × n_features`).
    pub fn coef(&self) -> &Array2<f64> {
        &self.coefficients.coef
    }

    /// `intercept_`.
    pub fn intercept(&self) -> &[f64] {
        &self.coefficients.intercept
    }

    /// `decision_function(X)`: `n_rows × coef.nrows()` (one column if binary).
    pub fn decision_function(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, ModelError> {
        check_input(x, self.coefficients.coef.ncols())?;
        let mut z = x.dot(&self.coefficients.coef.t());
        for mut row in z.rows_mut() {
            for (v, b) in row.iter_mut().zip(&self.coefficients.intercept) {
                *v += b;
            }
        }
        Ok(z)
    }

    /// `predict_proba(X)`: one column per entry of `classes_`.
    pub fn proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, ModelError> {
        let z = self.decision_function(x)?;
        if self.classes.len() == 2 {
            let mut out = Array2::zeros((z.nrows(), 2));
            for (i, &zi) in z.column(0).iter().enumerate() {
                let p1 = expit(zi);
                out[[i, 0]] = 1.0 - p1;
                out[[i, 1]] = p1;
            }
            Ok(out)
        } else {
            let mut out = z;
            for row in out.rows_mut() {
                softmax_in_place(row);
            }
            Ok(out)
        }
    }

    /// Predicted class codes (`argmax` of the decision function; in binary
    /// problems, class 1 when `z > 0`).
    pub fn predict_codes(&self, x: ArrayView2<'_, f64>) -> Result<Vec<usize>, ModelError> {
        let z = self.decision_function(x)?;
        Ok(z.rows()
            .into_iter()
            .map(|row| {
                let idx = if self.classes.len() == 2 {
                    usize::from(row[0] > 0.0)
                } else {
                    argmax(row.iter().copied())
                };
                self.classes[idx]
            })
            .collect())
    }
}

/// `scipy.special.expit`.
fn expit(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// `log(1 + exp(z))` without overflow.
fn softplus(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

/// scikit-learn's `softmax`: subtract the row maximum, exponentiate, normalise.
fn softmax_in_place(mut row: ndarray::ArrayViewMut1<'_, f64>) {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in row.iter_mut() {
        *v /= sum;
    }
}

/// Standardisation used as a preconditioner.
#[derive(Debug, Clone)]
struct Scaling {
    mean: Array1<f64>,
    /// Standard deviation (ddof 0); 1 for constant columns.
    std: Array1<f64>,
}

impl Scaling {
    fn new(x: ArrayView2<'_, f64>) -> Self {
        let n = x.nrows() as f64;
        let mut mean = Array1::zeros(x.ncols());
        let mut std = Array1::ones(x.ncols());
        for (j, col) in x.axis_iter(Axis(1)).enumerate() {
            let values = col.to_vec();
            let m = pairwise_sum(&values) / n;
            let var =
                pairwise_sum(&values.iter().map(|v| (v - m) * (v - m)).collect::<Vec<_>>()) / n;
            mean[j] = m;
            if var.is_finite() && var > 0.0 {
                std[j] = var.sqrt();
            }
        }
        Self { mean, std }
    }

    fn apply(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        (&x - &self.mean) / &self.std
    }

    /// From the solver's `θ` (k blocks of `[v₀…v_{p−1}, c]`) to `coef_`, `intercept_`.
    fn unscale(&self, theta: ArrayView1<'_, f64>, k: usize, p: usize) -> (Array2<f64>, Vec<f64>) {
        let mut coef = Array2::zeros((k, p));
        let mut intercept = vec![0.0; k];
        for r in 0..k {
            let block = theta.slice(s![r * (p + 1)..(r + 1) * (p + 1)]);
            let mut b = block[p];
            for j in 0..p {
                let w = block[j] / self.std[j];
                coef[[r, j]] = w;
                b -= w * self.mean[j];
            }
            intercept[r] = b;
        }
        if k > 1 {
            // Softmax is invariant to adding the same value to every class: at the
            // optimum the L2 penalty already makes Σₖ coef_[k, j] = 0 (scikit-learn
            // ~1e-15), and Σₖ intercept_ₖ = 0 is scikit-learn's L-BFGS choice.
            // Centring removes the rounding left by the solver without changing
            // any probability (`nota_odds_multiclase` relies on it).
            for mut col in coef.columns_mut() {
                let m = pairwise_sum(&col.to_vec()) / k as f64;
                col.mapv_inplace(|w| w - m);
            }
            let m = pairwise_sum(&intercept) / k as f64;
            for b in &mut intercept {
                *b -= m;
            }
        }
        (coef, intercept)
    }
}

/// The mean penalised log-loss in standardised variables.
#[derive(Debug, Clone)]
struct Objective<'a> {
    /// Standardised features.
    x: ArrayView2<'a, f64>,
    /// Position of each row's class in `classes_`.
    y: &'a [usize],
    /// Rows of coefficients: 1 (binary) or the number of classes.
    k: usize,
    /// `1/sⱼ²`: the penalty on `wⱼ = vⱼ/sⱼ`.
    inv_scale_sq: Array1<f64>,
    /// `1/(C·n)`.
    lambda: f64,
}

impl Objective<'_> {
    fn p(&self) -> usize {
        self.x.ncols()
    }

    /// `θ` as `k × (p+1)`: coefficients and intercept of each row (`θ` always has
    /// `k·(p+1)` values: the solver only sees vectors built by [`run_lbfgs`]).
    fn blocks(&self, theta: ArrayView1<'_, f64>) -> Array2<f64> {
        let p = self.p();
        Array2::from_shape_fn((self.k, p + 1), |(r, j)| theta[r * (p + 1) + j])
    }

    /// Linear predictors `Z = X̃·Vᵀ + c` (`n × k`).
    fn scores(&self, theta: ArrayView1<'_, f64>) -> Array2<f64> {
        let p = self.p();
        let b = self.blocks(theta);
        let v = b.slice(s![.., ..p]);
        let mut z = self.x.dot(&v.t());
        let c = b.column(p);
        for mut row in z.rows_mut() {
            row += &c;
        }
        z
    }

    fn penalty(&self, theta: ArrayView1<'_, f64>) -> f64 {
        let p = self.p();
        let b = self.blocks(theta);
        let mut s = 0.0;
        for row in b.rows() {
            for j in 0..p {
                s += row[j] * row[j] * self.inv_scale_sq[j];
            }
        }
        0.5 * self.lambda * s
    }

    fn loss(&self, theta: ArrayView1<'_, f64>) -> f64 {
        let z = self.scores(theta);
        let n = z.nrows() as f64;
        let per_row: Vec<f64> = if self.k == 1 {
            z.column(0)
                .iter()
                .zip(self.y)
                .map(|(&zi, &yi)| softplus(zi) - if yi == 1 { zi } else { 0.0 })
                .collect()
        } else {
            z.rows()
                .into_iter()
                .zip(self.y)
                .map(|(row, &yi)| {
                    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
                    lse - row[yi]
                })
                .collect()
        };
        pairwise_sum(&per_row) / n + self.penalty(theta)
    }

    /// Residuals `P − Y` (`n × k`).
    fn residuals(&self, theta: ArrayView1<'_, f64>) -> Array2<f64> {
        let mut z = self.scores(theta);
        if self.k == 1 {
            for (zi, &yi) in z.column_mut(0).iter_mut().zip(self.y) {
                *zi = expit(*zi) - if yi == 1 { 1.0 } else { 0.0 };
            }
        } else {
            for (mut row, &yi) in z.rows_mut().into_iter().zip(self.y) {
                softmax_in_place(row.view_mut());
                row[yi] -= 1.0;
            }
        }
        z
    }

    /// Gradient in the solver's variables.
    fn grad(&self, theta: ArrayView1<'_, f64>) -> Array1<f64> {
        let p = self.p();
        let n = self.x.nrows() as f64;
        let r = self.residuals(theta);
        let gv = r.t().dot(&self.x) / n; // k × p
        let gc = r.sum_axis(Axis(0)) / n;
        let b = self.blocks(theta);
        let mut g = Array1::zeros(self.k * (p + 1));
        for row in 0..self.k {
            for j in 0..p {
                g[row * (p + 1) + j] =
                    gv[[row, j]] + self.lambda * b[[row, j]] * self.inv_scale_sq[j];
            }
            g[row * (p + 1) + p] = gc[row];
        }
        g
    }

    /// Hessian in the solver's variables (`d × d`, `d = k·(p+1)`), plus a unit
    /// curvature on the mean of the intercepts in the multinomial case (the
    /// direction in which the objective is flat), which keeps the Newton step in
    /// the subspace `Σₖ cₖ = 0`.
    fn hessian(&self, theta: ArrayView1<'_, f64>) -> Array2<f64> {
        let p = self.p();
        let n = self.x.nrows();
        let d = self.k * (p + 1);
        // X̂ = [X̃, 1].
        let mut xh = Array2::<f64>::ones((n, p + 1));
        xh.slice_mut(s![.., ..p]).assign(&self.x);
        let z = self.scores(theta);
        let mut h = Array2::<f64>::zeros((d, d));
        let inv_n = 1.0 / n as f64;
        if self.k == 1 {
            let w: Array1<f64> = z.column(0).mapv(|zi| {
                let q = expit(zi);
                q * (1.0 - q) * inv_n
            });
            let xw = &xh * &w.view().insert_axis(Axis(1));
            h.assign(&xw.t().dot(&xh));
        } else {
            let mut prob = z;
            for row in prob.rows_mut() {
                softmax_in_place(row);
            }
            for a in 0..self.k {
                for b in a..self.k {
                    let w: Array1<f64> = prob
                        .rows()
                        .into_iter()
                        .map(|row| {
                            let delta = if a == b { row[a] } else { 0.0 };
                            (delta - row[a] * row[b]) * inv_n
                        })
                        .collect();
                    let xw = &xh * &w.view().insert_axis(Axis(1));
                    let block = xw.t().dot(&xh);
                    let (ra, rb) = (a * (p + 1), b * (p + 1));
                    h.slice_mut(s![ra..ra + p + 1, rb..rb + p + 1])
                        .assign(&block);
                    if a != b {
                        h.slice_mut(s![rb..rb + p + 1, ra..ra + p + 1])
                            .assign(&block.t());
                    }
                }
            }
            let flat = 1.0 / self.k as f64;
            for a in 0..self.k {
                for b in 0..self.k {
                    h[[a * (p + 1) + p, b * (p + 1) + p]] += flat;
                }
            }
        }
        for r in 0..self.k {
            for j in 0..p {
                h[[r * (p + 1) + j, r * (p + 1) + j]] += self.lambda * self.inv_scale_sq[j];
            }
        }
        h
    }

    /// Gradient of the same objective with respect to scikit-learn's variables
    /// (`coef_`, `intercept_`): `∂f/∂wⱼ = sⱼ·∂f/∂vⱼ + μⱼ·∂f/∂c`.
    fn gradient_original(&self, theta: ArrayView1<'_, f64>, scaling: &Scaling) -> Array1<f64> {
        let p = self.p();
        let g = self.grad(theta);
        let mut out = g.clone();
        for row in 0..self.k {
            let gc = g[row * (p + 1) + p];
            for j in 0..p {
                out[row * (p + 1) + j] =
                    scaling.std[j] * g[row * (p + 1) + j] + scaling.mean[j] * gc;
            }
        }
        out
    }
}

impl CostFunction for Objective<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<f64, argmin::core::Error> {
        Ok(self.loss(theta.view()))
    }
}

impl Gradient for Objective<'_> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, theta: &Self::Param) -> Result<Array1<f64>, argmin::core::Error> {
        Ok(self.grad(theta.view()))
    }
}

/// Solves `A·x = b` for a symmetric positive definite `A` (Cholesky, `A = L·Lᵀ`).
/// `None` if `A` is not numerically positive definite.
fn cholesky_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let d = a.nrows();
    let mut l = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for m in 0..j {
                sum -= l[[i, m]] * l[[j, m]];
            }
            if i == j {
                if sum.is_nan() || sum <= 0.0 {
                    return None;
                }
                l[[i, i]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    // L·y = b, then Lᵀ·x = y.
    let mut y = b.clone();
    for i in 0..d {
        for m in 0..i {
            y[i] -= l[[i, m]] * y[m];
        }
        y[i] /= l[[i, i]];
    }
    for i in (0..d).rev() {
        for m in i + 1..d {
            y[i] -= l[[m, i]] * y[m];
        }
        y[i] /= l[[i, i]];
    }
    Some(y)
}

/// Damped Newton from `θ = 0`: `H·Δ = −∇f`, then backtracking until the Armijo
/// condition holds. Stops when the gradient reaches [`INNER_GRADIENT_TOL`], when no
/// step decreases `f` any more (noise floor) or after `max_iter` iterations.
fn run_newton(
    objective: &Objective<'_>,
    n_params: usize,
    max_iter: u64,
    control: &TrainControl<'_>,
) -> Result<(Array1<f64>, u64), ModelError> {
    let mut theta = Array1::<f64>::zeros(n_params);
    let mut f = objective.loss(theta.view());
    control.check()?;
    control.report(TrainProgress {
        iteration: 0,
        max_iter,
        loss: Some(f),
        finished: false,
    });
    let mut iter = 0;
    while iter < max_iter {
        let g = objective.grad(theta.view());
        let g_norm = g.dot(&g).sqrt();
        if !g_norm.is_finite() {
            return Err(ModelError::Numerical {
                message: "gradiente no finito".to_owned(),
            });
        }
        if g_norm < INNER_GRADIENT_TOL {
            break;
        }
        control.check()?;
        let h = objective.hessian(theta.view());
        let neg_g = -&g;
        // Steepest descent if the Hessian is not numerically positive definite.
        let step = cholesky_solve(&h, &neg_g).unwrap_or_else(|| neg_g.clone());
        let slope = g.dot(&step);
        let mut t = 1.0;
        let mut accepted = None;
        for _ in 0..60 {
            let candidate = &theta + &(&step * t);
            let fc = objective.loss(candidate.view());
            if fc.is_finite() && fc <= f + 1e-4 * t * slope {
                accepted = Some((candidate, fc));
                break;
            }
            t *= 0.5;
        }
        iter += 1;
        let Some((next, fc)) = accepted else {
            break; // No representable decrease: we are at the optimum.
        };
        let decrease = f - fc;
        theta = next;
        f = fc;
        control.report(TrainProgress {
            iteration: iter,
            max_iter,
            loss: Some(f),
            finished: false,
        });
        if decrease <= 0.0 {
            break;
        }
    }
    Ok((theta, iter))
}

type Lbfgs =
    LBFGS<MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64>, Array1<f64>, Array1<f64>, f64>;
type LbfgsState = IterState<Array1<f64>, Array1<f64>, (), (), (), f64>;

/// Runs L-BFGS from `θ = 0` for at most `max_iter` iterations, checking the
/// cancellation flag before every iteration. Returns the best parameters and the
/// number of iterations.
///
/// This is `argmin`'s `Executor::run` loop written out (init, then
/// `terminate_internal`/`next_iter`/`update` until it stops). If the line search
/// fails close to the optimum (no further decrease is representable), the best
/// point found so far is kept.
fn run_lbfgs(
    objective: Objective<'_>,
    n_params: usize,
    max_iter: u64,
    control: &TrainControl<'_>,
) -> Result<(Array1<f64>, u64), ModelError> {
    let numerical = |e: argmin::core::Error| ModelError::Numerical {
        message: e.to_string(),
    };
    let init = Array1::<f64>::zeros(n_params);
    let mut solver: Lbfgs = LBFGS::new(MoreThuenteLineSearch::new(), LBFGS_MEMORY)
        .with_tolerance_grad(INNER_GRADIENT_TOL)
        .and_then(|s| s.with_tolerance_cost(0.0))
        .map_err(numerical)?;
    let mut problem = Problem::new(objective);
    let state: LbfgsState = IterState::new().param(init.clone()).max_iters(max_iter);

    control.check()?;
    control.report(TrainProgress {
        iteration: 0,
        max_iter,
        loss: None,
        finished: false,
    });
    let (mut state, _) = solver.init(&mut problem, state).map_err(numerical)?;
    state.update();

    loop {
        if !state.terminated() {
            if let TerminationStatus::Terminated(reason) =
                <Lbfgs as Solver<Objective<'_>, LbfgsState>>::terminate_internal(
                    &mut solver,
                    &state,
                )
            {
                state = state.terminate_with(reason);
            }
        }
        if state.terminated() {
            break;
        }
        control.check()?;
        let fallback = (state.get_best_param().cloned(), state.get_iter());
        match solver.next_iter(&mut problem, state) {
            Ok((next, _)) => state = next,
            Err(_) => {
                // Line search failure at the noise floor: keep the best point.
                let (best, iter) = fallback;
                return Ok((best.unwrap_or(init), iter));
            }
        }
        state.update();
        state.increment_iter();
        control.report(TrainProgress {
            iteration: state.get_iter(),
            max_iter,
            loss: Some(state.get_best_cost()),
            finished: false,
        });
    }
    let iter = state.get_iter();
    let best = state.take_best_param().unwrap_or(init);
    Ok((best, iter))
}

impl Predictor for LogisticRegression {
    fn task(&self) -> TaskType {
        TaskType::Classification
    }

    fn n_features(&self) -> usize {
        self.coefficients.coef.ncols()
    }

    fn classes(&self) -> Option<&[usize]> {
        Some(&self.classes)
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Prediction, ModelError> {
        self.predict_codes(x).map(Prediction::Classes)
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Option<Result<Array2<f64>, ModelError>> {
        Some(self.proba(x))
    }

    fn refit(
        &self,
        x: ArrayView2<'_, f64>,
        y: &Target,
        control: &TrainControl<'_>,
    ) -> Result<Box<dyn Predictor>, ModelError> {
        Ok(Box::new(Self::fit_target(x, y, self.params, control)?))
    }

    fn linear_coefficients(&self) -> Option<&LinearCoefficients> {
        Some(&self.coefficients)
    }

    fn fit_report(&self) -> Option<FitReport> {
        Some(self.report)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::sync::atomic::AtomicBool;

    fn toy() -> (Array2<f64>, Vec<usize>) {
        let x = array![
            [0.0, 1.0],
            [1.0, 0.5],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 3.0],
            [5.0, 2.5],
            [1.5, 1.0],
            [3.5, 2.0]
        ];
        (x, vec![0, 0, 0, 1, 1, 1, 1, 0])
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let (x, y) = toy();
        for k in [1, 3] {
            let yk: Vec<usize> = y
                .iter()
                .enumerate()
                .map(|(i, &c)| if k == 1 { c } else { i % 3 })
                .collect();
            let obj = Objective {
                x: x.view(),
                y: &yk,
                k,
                inv_scale_sq: array![0.5, 2.0],
                lambda: 0.1,
            };
            let theta = Array1::from_iter((0..k * 3).map(|i| 0.1 * i as f64 - 0.2));
            let g = obj.grad(theta.view());
            for i in 0..theta.len() {
                let mut tp = theta.clone();
                let mut tm = theta.clone();
                tp[i] += 1e-6;
                tm[i] -= 1e-6;
                let fd = (obj.loss(tp.view()) - obj.loss(tm.view())) / 2e-6;
                assert!((fd - g[i]).abs() < 1e-7, "k={k} i={i}: {fd} vs {}", g[i]);
            }
        }
    }

    #[test]
    fn binary_fit_converges_and_predicts() {
        let (x, y) = toy();
        let m = LogisticRegression::fit(
            x.view(),
            &y,
            LogisticParams::default(),
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(m.coef().dim(), (1, 2));
        let report = m.fit_report().unwrap_or_else(|| panic!("report"));
        assert!(report.converged);
        let proba = m.proba(x.view()).unwrap_or_else(|e| panic!("{e}"));
        for row in proba.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_regression_target_and_single_class() {
        let (x, _) = toy();
        let t = Target::Values {
            values: vec![1.0; 8],
        };
        assert_eq!(
            LogisticRegression::fit_target(
                x.view(),
                &t,
                LogisticParams::default(),
                &TrainControl::none()
            ),
            Err(ModelError::InvalidTarget)
        );
        assert_eq!(
            LogisticRegression::fit(
                x.view(),
                &[0; 8],
                LogisticParams::default(),
                &TrainControl::none()
            ),
            Err(ModelError::TooFewClasses { found: 1 })
        );
    }

    #[test]
    fn cancellation_stops_training() {
        let (x, y) = toy();
        let flag = AtomicBool::new(true);
        let ctl = TrainControl::none().with_cancel(&flag);
        assert_eq!(
            LogisticRegression::fit(x.view(), &y, LogisticParams::default(), &ctl),
            Err(ModelError::Cancelled)
        );
    }
}
