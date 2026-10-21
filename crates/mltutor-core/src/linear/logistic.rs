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
//!   problem has at most [`NEWTON_MAX_PARAMS`] parameters (`k·(p+1)`) and one
//!   Hessian costs at most [`NEWTON_MAX_WORK`]: every built-in classification
//!   dataset, Digits included. It converges in ~5–15 iterations even on unscaled,
//!   collinear data such as Breast Cancer, where scipy's L-BFGS (scikit-learn)
//!   stops at `max_iter=1000` 0.20 away from the optimum in probability, and where
//!   L-BFGS needs >20 000 iterations. On Digits (650 parameters) Newton reaches the
//!   optimum in 12 iterations (~0.5 s); our L-BFGS was still 2.8e-3 away after
//!   1000 (and scikit-learn's own model, which stops after 139, 0.25 away).
//! * **L-BFGS** (10 correction pairs, More–Thuente line search, from `argmin`:
//!   exactly what `linfa-logistic` builds) for larger problems. We drive argmin's
//!   loop ourselves instead of calling `linfa-logistic`, because `linfa` neither
//!   lets us cancel the training nor reports iterations or convergence. It stops
//!   at `max_iter`, at the noise floor, or once scikit-learn's criterion holds and
//!   the objective no longer decreases (relative [`LBFGS_FTOL`]).
//!
//! Both work on **standardised features** (a change of variables `vⱼ = wⱼ·sⱼ`,
//! with the penalty still on `wⱼ = vⱼ/sⱼ`: same minimiser, better conditioning).
//! The mean and standard deviation are computed on each column divided by its
//! largest `|x|`, so that values as large as `1e300` do not overflow; if the
//! objective still becomes non-finite, the error asks the student to scale the
//! features. The solvers then
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

use argmin::core::{
    CostFunction, Gradient, IterState, Problem, Solver, State, TerminationReason, TerminationStatus,
};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::dataset::{Target, TaskType};
use crate::model::{
    argmax, check_input, check_n_classes, FitReport, LinearCoefficients, ModelError, Prediction,
    Predictor, TrainControl, TrainProgress,
};
use crate::stats::pairwise_sum;

/// Gradient norm (of the mean objective, in standardised variables) at which the
/// solver stops before `max_iter`.
pub const INNER_GRADIENT_TOL: f64 = 1e-10;

/// Largest number of parameters (`d = k·(p+1)`) solved with Newton's method
/// (Digits has 650). Above it, or above [`NEWTON_MAX_WORK`], L-BFGS is used.
pub const NEWTON_MAX_PARAMS: usize = 700;

/// Largest `n·d²` (multiply-adds to build one Hessian) solved with Newton's
/// method: Digits needs 5.3e8 (~45 ms per iteration in release). A CSV with a
/// million rows and 300 parameters (9e10) goes to L-BFGS instead.
pub const NEWTON_MAX_WORK: f64 = 2e9;

/// Relative decrease of the objective in one L-BFGS iteration below which the
/// solver stops, once scikit-learn's criterion (`max|∇f| ≤ tol`) also holds.
pub const LBFGS_FTOL: f64 = 1e-10;

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
        check_n_classes(classes.len())?;
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
            inv_scale_sq: scaling.inv_scale_sq(),
            lambda: 1.0 / (params.c * n as f64),
            work: HessianWork::default(),
        };

        let max_iter = u64::from(params.max_iter);
        let n_params = k * (p + 1);
        let (theta, n_iter) = if use_newton(n, n_params) {
            run_newton(&problem, n_params, max_iter, control)?
        } else {
            let stop = SklearnStop {
                scaling: &scaling,
                tol: params.tol,
            };
            run_lbfgs(problem.clone(), n_params, max_iter, &stop, control)?
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
    /// Mean of each column.
    mean: Array1<f64>,
    /// Standard deviation (ddof 0); 1 for constant columns.
    std: Array1<f64>,
    /// Largest `|x|` of each column (1 for all-zero columns): the statistics are
    /// computed on `x / max_abs`, in `[-1, 1]`, so that no square overflows.
    max_abs: Array1<f64>,
}

impl Scaling {
    fn new(x: ArrayView2<'_, f64>) -> Self {
        let n = x.nrows() as f64;
        let p = x.ncols();
        let mut mean = Array1::zeros(p);
        let mut std = Array1::ones(p);
        let mut max_abs = Array1::ones(p);
        for (j, col) in x.axis_iter(Axis(1)).enumerate() {
            let c = col.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let c = if c > 0.0 && c.is_finite() { c } else { 1.0 };
            let values: Vec<f64> = col.iter().map(|v| v / c).collect();
            let m = pairwise_sum(&values) / n;
            let var =
                pairwise_sum(&values.iter().map(|v| (v - m) * (v - m)).collect::<Vec<_>>()) / n;
            max_abs[j] = c;
            mean[j] = m * c;
            if var.is_finite() && var > 0.0 {
                std[j] = var.sqrt() * c;
            }
        }
        Self { mean, std, max_abs }
    }

    /// `(x − mean) / std`, computed as `(x/c − mean/c) / (std/c)` (`c = max_abs`)
    /// so that it cannot overflow.
    fn apply(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut out = x.to_owned();
        for (j, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
            let c = self.max_abs[j];
            let (m, s) = (self.mean[j] / c, self.std[j] / c);
            col.mapv_inplace(|v| (v / c - m) / s);
        }
        out
    }

    /// `1/sⱼ²`, the penalty weight of `vⱼ`. Clamped to `f64::MAX` so that a
    /// tiny `sⱼ` gives a huge (not infinite) penalty and `0·penalty` stays 0.
    fn inv_scale_sq(&self) -> Array1<f64> {
        self.std.mapv(|s| {
            let inv = 1.0 / s;
            (inv * inv).min(f64::MAX)
        })
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
    /// Buffers of [`Objective::hessian`], reused between Newton iterations.
    work: HessianWork,
}

/// `X̂ = [X̃, 1]` and `W·X̂` (both `n × (p+1)`): built once, on the first Hessian,
/// instead of allocating them in every Newton iteration (review F2).
#[derive(Default)]
struct HessianWork {
    buffers: std::cell::RefCell<Option<(Array2<f64>, Array2<f64>)>>,
}

impl Clone for HessianWork {
    /// A clone starts with empty buffers (they are only a cache).
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for HessianWork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HessianWork")
    }
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
        let mut work = self.work.buffers.borrow_mut();
        let (xh, xw) = work.get_or_insert_with(|| {
            // X̂ = [X̃, 1].
            let mut xh = Array2::<f64>::ones((n, p + 1));
            xh.slice_mut(s![.., ..p]).assign(&self.x);
            let xw = Array2::<f64>::zeros((n, p + 1));
            (xh, xw)
        });
        let z = self.scores(theta);
        let mut h = Array2::<f64>::zeros((d, d));
        let inv_n = 1.0 / n as f64;
        // xw = diag(w)·X̂, then the block X̂ᵀ·diag(w)·X̂ is written into `h`.
        let mut weighted_block = |w: &mut dyn FnMut(usize) -> f64, ra: usize, rb: usize| {
            for (i, (mut dst, src)) in xw.rows_mut().into_iter().zip(xh.rows()).enumerate() {
                let wi = w(i);
                dst.zip_mut_with(&src, |d, &v| *d = v * wi);
            }
            general_mat_mul(
                1.0,
                &xw.t(),
                &*xh,
                0.0,
                &mut h.slice_mut(s![ra..ra + p + 1, rb..rb + p + 1]),
            );
        };
        if self.k == 1 {
            let col = z.column(0);
            weighted_block(
                &mut |i| {
                    let q = expit(col[i]);
                    q * (1.0 - q) * inv_n
                },
                0,
                0,
            );
        } else {
            let mut prob = z;
            for row in prob.rows_mut() {
                softmax_in_place(row);
            }
            for a in 0..self.k {
                for b in a..self.k {
                    let pr = &prob;
                    weighted_block(
                        &mut |i| {
                            let delta = if a == b { pr[[i, a]] } else { 0.0 };
                            (delta - pr[[i, a]] * pr[[i, b]]) * inv_n
                        },
                        a * (p + 1),
                        b * (p + 1),
                    );
                }
            }
            for a in 0..self.k {
                for b in (a + 1)..self.k {
                    let (ra, rb) = (a * (p + 1), b * (p + 1));
                    let block = h.slice(s![ra..ra + p + 1, rb..rb + p + 1]).t().to_owned();
                    h.slice_mut(s![rb..rb + p + 1, ra..ra + p + 1])
                        .assign(&block);
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
///
/// Column by column (left-looking): `l[i][j] = (a[i][j] − Σ_{m<j} l[i][m]·l[j][m]) / l[j][j]`,
/// with the sum over `m` in increasing order. The rows `i > j` of a column are
/// independent, so four of them are computed together (four dependency chains
/// instead of one) over contiguous row slices; every entry keeps its own
/// summation order, so the result is the same, bit for bit, as the plain triple
/// loop (checked on the coefficients and probabilities of five datasets). Digits
/// (`d = 650`, 12 Newton iterations): 100 ms → about 12 ms per factorisation, and the
/// whole fit from 1.5 s to 0.4 s (F7, 2026-10-20).
fn cholesky_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let d = a.nrows();
    let mut l = vec![0.0_f64; d * d]; // Row-major; only the lower triangle is used.
    for j in 0..d {
        let (head, tail) = l.split_at_mut((j + 1) * d);
        let row_j = &mut head[j * d..];
        let mut sum = a[[j, j]];
        for &v in &row_j[..j] {
            sum -= v * v;
        }
        if sum.is_nan() || sum <= 0.0 {
            return None;
        }
        let ljj = sum.sqrt();
        row_j[j] = ljj;
        let row_j = &row_j[..j];
        let quads = (d - j - 1) / 4 * 4;
        let (quad, rest) = tail.split_at_mut(quads * d);
        let mut i = j + 1;
        for block in quad.chunks_exact_mut(4 * d) {
            let (r0, r) = block.split_at_mut(d);
            let (r1, r) = r.split_at_mut(d);
            let (r2, r3) = r.split_at_mut(d);
            let mut s = [a[[i, j]], a[[i + 1, j]], a[[i + 2, j]], a[[i + 3, j]]];
            for (m, &v) in row_j.iter().enumerate() {
                s[0] -= r0[m] * v;
                s[1] -= r1[m] * v;
                s[2] -= r2[m] * v;
                s[3] -= r3[m] * v;
            }
            r0[j] = s[0] / ljj;
            r1[j] = s[1] / ljj;
            r2[j] = s[2] / ljj;
            r3[j] = s[3] / ljj;
            i += 4;
        }
        for r in rest.chunks_exact_mut(d) {
            let mut s = a[[i, j]];
            for (&x, &v) in r[..j].iter().zip(row_j) {
                s -= x * v;
            }
            r[j] = s / ljj;
            i += 1;
        }
    }
    // L·y = b, then Lᵀ·x = y.
    let mut y = b.clone();
    for i in 0..d {
        let row = &l[i * d..i * d + i];
        for (m, &v) in row.iter().enumerate() {
            y[i] -= v * y[m];
        }
        y[i] /= l[i * d + i];
    }
    for i in (0..d).rev() {
        for m in i + 1..d {
            y[i] -= l[m * d + i] * y[m];
        }
        y[i] /= l[i * d + i];
    }
    Some(y)
}

/// The objective or its gradient overflowed: in practice, features with huge
/// values. A message the student can act on.
fn values_too_large() -> ModelError {
    ModelError::Numerical {
        message: "valores demasiado grandes; escala las características (por ejemplo, \
                  con StandardScaler o cambiando las unidades)"
            .to_owned(),
    }
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
        if !g_norm.is_finite() || !f.is_finite() {
            return Err(values_too_large());
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

/// Whether to use Newton's method (small problems, see [`NEWTON_MAX_PARAMS`] and
/// [`NEWTON_MAX_WORK`]) or L-BFGS.
fn use_newton(n_rows: usize, n_params: usize) -> bool {
    let work = n_rows as f64 * (n_params as f64) * (n_params as f64);
    n_params <= NEWTON_MAX_PARAMS && work <= NEWTON_MAX_WORK
}

/// Early stop of L-BFGS: scikit-learn's criterion (`max|∇f| ≤ tol` in the original
/// variables, as in [`FitReport::converged`]) **and** a relative decrease of the
/// objective below [`LBFGS_FTOL`] in the last iteration. The second condition keeps
/// the solver going while it still makes progress (scikit-learn stops on the first
/// one, far from the optimum on ill-conditioned data), and stops it once only noise
/// is left instead of running to `max_iter`.
struct SklearnStop<'a> {
    scaling: &'a Scaling,
    tol: f64,
}

impl SklearnStop<'_> {
    fn holds(&self, grad: &Array1<f64>, f_prev: f64, f_new: f64, n_params: usize) -> bool {
        let scale = f_prev.abs().max(f_new.abs()).max(1.0);
        let decrease = f_prev - f_new;
        // NaN (no cost yet) never stops the solver.
        let stalled = decrease.is_finite() && decrease <= LBFGS_FTOL * scale;
        if !stalled {
            return false;
        }
        let p = self.scaling.std.len();
        if grad.len() != n_params || p == 0 || n_params % (p + 1) != 0 {
            return false;
        }
        // ∂f/∂wⱼ = sⱼ·∂f/∂vⱼ + μⱼ·∂f/∂c and ∂f/∂b = ∂f/∂c (see `gradient_original`).
        grad.as_slice()
            .map(|g| {
                g.chunks_exact(p + 1).all(|block| {
                    let gc = block[p];
                    gc.abs() <= self.tol
                        && (0..p).all(|j| {
                            let gw = self.scaling.std[j] * block[j] + self.scaling.mean[j] * gc;
                            gw.abs() <= self.tol
                        })
                })
            })
            .unwrap_or(false)
    }
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
    stop: &SklearnStop<'_>,
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
    if !state.get_cost().is_finite() {
        return Err(values_too_large());
    }

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
        let f_prev = state.get_cost();
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
        if let Some(g) = state.get_gradient() {
            if stop.holds(g, f_prev, state.get_cost(), n_params) {
                state = state.terminate_with(TerminationReason::SolverConverged);
            }
        }
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
                work: HessianWork::default(),
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
    fn rejects_too_many_classes() {
        let n = 2 * (crate::model::MAX_CLASSES + 1);
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y: Vec<usize> = (0..n).map(|i| i / 2).collect();
        assert_eq!(
            LogisticRegression::fit(
                x.view(),
                &y,
                LogisticParams::default(),
                &TrainControl::none()
            ),
            Err(ModelError::TooManyClasses {
                n_classes: crate::model::MAX_CLASSES + 1,
                max: crate::model::MAX_CLASSES
            })
        );
    }

    #[test]
    fn huge_feature_values_train_like_scaled_ones() {
        // Item 7 of the F2 review: 1e300·x used to give "gradiente no finito".
        let (x, y) = toy();
        let fit = |x: &Array2<f64>| {
            LogisticRegression::fit(
                x.view(),
                &y,
                LogisticParams::default(),
                &TrainControl::none(),
            )
            .unwrap_or_else(|e| panic!("{e}"))
        };
        let small = fit(&x);
        let big = fit(&x.mapv(|v| v * 1e300));
        let p_small = small.proba(x.view()).unwrap_or_else(|e| panic!("{e}"));
        let p_big = big
            .proba(x.mapv(|v| v * 1e300).view())
            .unwrap_or_else(|e| panic!("{e}"));
        // Not the same model (the penalty is on w, which scales by 1e-300, so the
        // big one is unregularised), but finite and a valid probability.
        assert!(p_big
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        assert!(p_small.iter().all(|v| v.is_finite()));
        // Tiny values: huge penalty, the coefficients stay ~0 instead of NaN.
        let tiny = fit(&x.mapv(|v| v * 1e-300));
        assert!(tiny.coef().iter().all(|v| v.is_finite()));
        assert!(tiny.intercept().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn digits_uses_newton_and_bounds_its_cost() {
        assert!(use_newton(1257, 650)); // Digits: 10 classes × 65.
        assert!(!use_newton(1257, NEWTON_MAX_PARAMS + 1));
        assert!(!use_newton(1_000_000, 300));
        assert!(use_newton(1_000_000, 21));
    }

    #[test]
    fn lbfgs_early_stop_needs_both_criteria() {
        let x = array![[1.0, 10.0], [3.0, 20.0]];
        let scaling = Scaling::new(x.view());
        let stop = SklearnStop {
            scaling: &scaling,
            tol: 1e-4,
        };
        let small = Array1::from(vec![1e-6, 1e-6, 1e-6]);
        let large = Array1::from(vec![1e-2, 0.0, 0.0]);
        assert!(stop.holds(&small, 0.5, 0.5, 3));
        assert!(!stop.holds(&small, 0.5, 0.4, 3)); // still decreasing
        assert!(!stop.holds(&large, 0.5, 0.5, 3)); // sklearn's criterion fails
    }

    #[test]
    fn lbfgs_reaches_newton_optimum() {
        let (x, y) = toy();
        let scaling = Scaling::new(x.view());
        let xs = scaling.apply(x.view());
        let problem = Objective {
            x: xs.view(),
            y: &y,
            k: 1,
            inv_scale_sq: scaling.inv_scale_sq(),
            lambda: 1.0 / 8.0,
            work: HessianWork::default(),
        };
        let ctl = TrainControl::none();
        let (newton, _) = run_newton(&problem, 3, 100, &ctl).unwrap_or_else(|e| panic!("{e}"));
        let stop = SklearnStop {
            scaling: &scaling,
            tol: 1e-4,
        };
        let (lbfgs, it) =
            run_lbfgs(problem.clone(), 3, 1000, &stop, &ctl).unwrap_or_else(|e| panic!("{e}"));
        assert!(it < 1000);
        let diff = (&newton - &lbfgs)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(diff < 1e-5, "{diff}");
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
