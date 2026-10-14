//! Ordinary least squares, as `sklearn.linear_model.LinearRegression()`.
//!
//! scikit-learn centres `X` and `y` (`X_offset`, `y_offset`), solves
//! `min ‖Xc·β − yc‖²` with `scipy.linalg.lstsq` and sets
//! `intercept_ = y_offset − X_offset·β`. We do the same, in two steps:
//!
//! 1. A Householder QR of the centred matrix, `Xc = Q·R`, applied to `yc` at the
//!    same time (`c = Qᵀ·yc`). This is the only pass over the `n × p` data and costs
//!    `O(n·p²)`; `R` is `min(n, p) × p`.
//! 2. The SVD of the small `R` (`O(p³)`): `Xc` and `R` have the same singular
//!    values, so the rank and the minimum-norm solution `β = V·Σ⁺·Uᵀ·c` are those of
//!    `lstsq(Xc, yc)` (LAPACK `gelsd`), also with collinear or constant columns and
//!    with more columns than rows.
//!
//! Singular values `≤ f64::EPSILON · max(n, p) · σ_max` count as zero: scikit-learn
//! passes `cond = max(X.shape) * eps` to `scipy.linalg.lstsq` (numpy's `lstsq` uses the
//! same default `rcond`). A smaller cutoff (`eps · σ_max`) keeps the rounding noise of
//! derived columns such as `x1 = 0.1·x0 + 0.7·x2` and returns huge coefficients.

use std::any::Any;

use linfa_linalg::svd::SVDInto;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
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

        let (beta, rank) = solve_centered(x, x_offset.view(), y, y_offset, control)?;
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

/// Least squares on the centred data `x − x_offset`, `y − y_offset`; returns
/// `(β, rank)`. See the module docs for the method.
fn solve_centered(
    x: ArrayView2<'_, f64>,
    x_offset: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    y_offset: f64,
    control: &TrainControl<'_>,
) -> Result<(Array1<f64>, usize), ModelError> {
    let (n, p) = x.dim();
    if p == 0 {
        return Ok((Array1::zeros(0), 0));
    }
    // Column-major work copy: row j holds column j of Xc and the last row holds yc,
    // so every Householder update below runs over contiguous memory.
    let mut a = Array2::<f64>::zeros((p + 1, n));
    for (j, mut row) in a.outer_iter_mut().take(p).enumerate() {
        let off = x_offset[j];
        row.zip_mut_with(&x.column(j), |dst, &v| *dst = v - off);
    }
    a.row_mut(p).zip_mut_with(&y, |dst, &v| *dst = v - y_offset);
    control.check()?;

    let m = householder_qr(&mut a, n.min(p), control)?;

    // R (m × p, upper trapezoidal) and c = (Qᵀ·yc)[..m].
    let r = Array2::from_shape_fn((m, p), |(i, j)| if i <= j { a[[j, i]] } else { 0.0 });
    let c = a.slice(s![p, ..m]).to_owned();
    drop(a);
    control.check()?;
    min_norm_lstsq(r, c, n.max(p))
}

/// In-place Householder QR of the transposed matrix `a` (`(p + 1) × n`: the `p`
/// columns of `X` as rows, plus `y`), for the first `m` columns. Afterwards
/// `a[[j, i]] = R[i, j]` for `i ≤ j`, and `a[[p, ..m]] = (Qᵀ·y)[..m]`. Returns `m`.
fn householder_qr(
    a: &mut Array2<f64>,
    m: usize,
    control: &TrainControl<'_>,
) -> Result<usize, ModelError> {
    let mut v: Vec<f64> = Vec::new();
    for k in 0..m {
        control.check()?;
        // Reflector H = I − 2·v·vᵀ/(vᵀv) that maps x = column k (rows k..) to α·e₁.
        v.clear();
        v.extend(a.slice(s![k, k..]).iter().copied());
        let norm = scaled_norm(&v);
        if norm == 0.0 {
            continue; // Nothing to annihilate (a zero column).
        }
        let alpha = if v[0] > 0.0 { -norm } else { norm };
        v[0] -= alpha;
        let vtv = dot(&v, &v);
        a[[k, k]] = alpha;
        a.slice_mut(s![k, k + 1..]).fill(0.0);
        if vtv == 0.0 {
            continue;
        }
        for mut row in a.slice_mut(s![k + 1.., k..]).outer_iter_mut() {
            let Some(row) = row.as_slice_mut() else {
                return Err(ModelError::Numerical {
                    message: "matriz de trabajo no contigua".to_owned(),
                });
            };
            let f = 2.0 * dot(&v, row) / vtv;
            if f != 0.0 {
                row.iter_mut().zip(&v).for_each(|(r, &vi)| *r -= f * vi);
            }
        }
    }
    Ok(m)
}

/// Euclidean norm without overflow or underflow for any finite input.
fn scaled_norm(v: &[f64]) -> f64 {
    let scale = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    let ss: f64 = v.iter().map(|x| (x / scale) * (x / scale)).sum();
    scale * ss.sqrt()
}

/// Dot product with four independent accumulators, so that the compiler can use
/// SIMD; the order of the additions is fixed, so the result is deterministic.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0_f64; 4];
    let (ca, ra) = a.split_at(a.len() - a.len() % 4);
    let (cb, rb) = b.split_at(ca.len().min(b.len()));
    for (x, y) in ca.chunks_exact(4).zip(cb.chunks_exact(4)) {
        for i in 0..4 {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f64 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// Minimum-norm solution of `min ‖R·β − c‖` through the SVD of `R`,
/// `β = Σ_k (u_kᵀ·c / σ_k)·v_k` over `σ_k > eps · max_dim · σ_max`; returns
/// `(β, rank)`. `max_dim` is `max(n, p)` of the original matrix (sklearn's `cond`).
fn min_norm_lstsq(
    r: Array2<f64>,
    c: Array1<f64>,
    max_dim: usize,
) -> Result<(Array1<f64>, usize), ModelError> {
    let p = r.ncols();
    if r.iter().all(|&v| v == 0.0) {
        return Ok((Array1::zeros(p), 0)); // Every column is constant.
    }
    let (u, sigma, vt) = r.svd_into(true, true).map_err(numerical)?;
    let (Some(u), Some(vt)) = (u, vt) else {
        return Err(ModelError::Numerical {
            message: "SVD sin vectores singulares".to_owned(),
        });
    };
    let s_max = sigma.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = rank_cutoff(s_max, max_dim);
    let uty = u.t().dot(&c);
    let mut beta = Array1::<f64>::zeros(p);
    let mut rank = 0;
    for (k, &s) in sigma.iter().enumerate() {
        if s > cutoff {
            beta.scaled_add(uty[k] / s, &vt.row(k));
            rank += 1;
        }
    }
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(ModelError::Numerical {
            message: "coeficientes no finitos".to_owned(),
        });
    }
    Ok((beta, rank))
}

/// Singular values at or below this count as zero: `eps · max(n, p) · σ_max`, the
/// `cond` that scikit-learn passes to `scipy.linalg.lstsq`.
fn rank_cutoff(s_max: f64, max_dim: usize) -> f64 {
    f64::EPSILON * max_dim as f64 * s_max
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

    /// splitmix64 in [0, 1): the same stream as the Python script that produced
    /// the references below (exact integer arithmetic in both languages).
    fn uniform(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn assert_close(m: &LinearRegression, rank: usize, coef: &[f64], intercept: f64) {
        assert_eq!(m.rank(), rank, "{:?}", m.coef());
        let err = m
            .coef()
            .iter()
            .zip(coef)
            .map(|(a, b)| (a - b).abs())
            .fold((m.intercept() - intercept).abs(), f64::max);
        // Tolerance 1e-9: after the rank cut the problem is well conditioned.
        assert!(err < 1e-9, "max error {err:e}: {:?}", m.coef());
    }

    /// Review F2 (high): `x1 = 0.1·x0 + 0.7·x2` and `x3 = x0/3` are not exactly
    /// representable, so the centred matrix has two singular values of ~1e-15·σ_max.
    /// With `cutoff = eps·σ_max` some seeds kept one (rank 3, huge coefficients).
    /// References: scikit-learn 1.7.2, `LinearRegression().fit(X, y)`, n = 200,
    /// `x0 = 10u−5`, `x2 = 4u−2`, `y = 1.5·x0 − 2·x2 + 3 + 0.1·(u−0.5)`.
    #[test]
    fn derived_columns_match_sklearn_rank_and_coefficients() {
        // Seed 6 is the one where our QR + SVD finds a third singular value of
        // 1.07·eps·σ_max: the old cutoff kept it (rank 3); the other seeds were
        // already right and guard against regressions.
        let cases: [(u64, [f64; 4], f64); 8] = [
            (
                6,
                [
                    1.425273887846125,
                    -0.8449731854786442,
                    -1.4107151060903662,
                    0.4750912959487085,
                ],
                2.9983341740295577,
            ),
            (
                55,
                [
                    1.4261274536332675,
                    -0.8439930859067156,
                    -1.4094369018143462,
                    0.47537581787775607,
                ],
                2.9996960189791952,
            ),
            (
                58,
                [
                    1.4264229833638546,
                    -0.8438347773069604,
                    -1.4092529652047803,
                    0.47547432778795157,
                ],
                2.9993483791303817,
            ),
            (
                1,
                [
                    1.424509603088894,
                    -0.8418796987174982,
                    -1.4061866557519824,
                    0.4748365343629646,
                ],
                2.99763188175539,
            ),
            (
                2,
                [
                    1.4260141773455315,
                    -0.8439112137232988,
                    -1.409303759225504,
                    0.4753380591151777,
                ],
                3.001362914484429,
            ),
            (
                3,
                [
                    1.4250804909185109,
                    -0.8442329717883721,
                    -1.4096300298288915,
                    0.47502683030617004,
                ],
                2.996092211887391,
            ),
            (
                42,
                [
                    1.4267958558122638,
                    -0.8436697074719711,
                    -1.4090704186474245,
                    0.475598618604088,
                ],
                2.9997367831101944,
            ),
            (
                7,
                [
                    1.425096405549646,
                    -0.8434484586559104,
                    -1.4085115703012498,
                    0.4750321351832154,
                ],
                2.9990040763670995,
            ),
        ];
        for (seed, coef, intercept) in cases {
            let n = 200;
            let mut st = seed;
            let mut x = Array2::<f64>::zeros((n, 4));
            let mut y = Array1::<f64>::zeros(n);
            for i in 0..n {
                let x0 = uniform(&mut st) * 10.0 - 5.0;
                let x2 = uniform(&mut st) * 4.0 - 2.0;
                let e = uniform(&mut st) - 0.5;
                x.row_mut(i)
                    .assign(&array![x0, 0.1 * x0 + 0.7 * x2, x2, x0 / 3.0]);
                y[i] = 1.5 * x0 - 2.0 * x2 + 3.0 + 0.1 * e;
            }
            let m = LinearRegression::fit(x.view(), y.view(), &TrainControl::none())
                .unwrap_or_else(|e| panic!("{e}"));
            assert_close(&m, 2, &coef, intercept);
        }
    }

    /// More columns than rows (5 × 20, `u·2−1`): rank 4 after centring and the
    /// minimum-norm solution, like scikit-learn 1.7.2.
    #[test]
    fn wide_matrix_matches_sklearn_min_norm() {
        let cases: [(u64, [f64; 20], f64); 2] = [
            (
                5,
                [
                    -0.006937406186262267,
                    -0.017931520463017075,
                    -0.0011792306775358545,
                    0.15934874346557754,
                    0.09574168070857941,
                    -0.015964328633976902,
                    -0.01562562782952863,
                    0.22852846255920226,
                    -0.08220747010577831,
                    -0.012371878522173616,
                    -0.012271174142588279,
                    -0.10233598748150693,
                    0.039512116859616225,
                    0.14125829543685073,
                    0.13959168186074505,
                    0.010668337822759692,
                    -0.08578665091236906,
                    0.035555501748545784,
                    0.19882935081965095,
                    0.04681624938303254,
                ],
                0.04262594834772529,
            ),
            (
                42,
                [
                    -0.1764871102329737,
                    0.14590408914632313,
                    -0.0746463274635426,
                    -0.097203303607895,
                    0.0962696849228083,
                    0.10299820513141035,
                    -0.0720522884721202,
                    -0.09468818313806901,
                    -0.10645824310727287,
                    0.03538757531761694,
                    -0.08345822628787668,
                    -0.11957524918842789,
                    -0.12405183295838206,
                    -0.06817299481114832,
                    -0.016292771162308838,
                    0.03900135844191936,
                    0.08035600713191764,
                    0.10075627126755221,
                    -0.11827753383586882,
                    0.1141483347531041,
                ],
                0.43569639938772126,
            ),
        ];
        for (seed, coef, intercept) in cases {
            let mut st = seed;
            let x = Array2::from_shape_fn((5, 20), |_| uniform(&mut st) * 2.0 - 1.0);
            let y = Array1::from_shape_fn(5, |_| uniform(&mut st) * 2.0 - 1.0);
            let m = LinearRegression::fit(x.view(), y.view(), &TrainControl::none())
                .unwrap_or_else(|e| panic!("{e}"));
            assert_close(&m, 4, &coef, intercept);
        }
    }

    #[test]
    fn cancellation_is_checked() {
        let flag = std::sync::atomic::AtomicBool::new(true);
        let x = array![[0.0], [1.0], [2.0]];
        let y = array![1.0, 3.0, 5.0];
        assert_eq!(
            LinearRegression::fit(x.view(), y.view(), &TrainControl::none().with_cancel(&flag)),
            Err(ModelError::Cancelled)
        );
    }
}
