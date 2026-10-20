//! Minimum-norm least squares through the SVD of a small matrix, as LAPACK's
//! `gelsd` (what `scipy.linalg.lstsq` calls) but without forming `U` or `V`.
//!
//! The OLS solver ([`super::ols`]) first reduces the `n × p` data to the upper
//! trapezoidal `R` (`m × p`, `m = min(n, p)`) of a QR factorisation. When `R` is
//! rank deficient we need `β = V·Σ⁺·Uᵀ·c` with the singular values below the
//! cutoff treated as zero. This module computes it in three steps, all of them
//! cancellable:
//!
//! 1. **Square it.** If `m < p` (more columns than rows), a Householder QR of `Rᵀ`
//!    gives `R = Sᵀ·Q₂ᵀ` with `S` square; `R` and `Sᵀ` have the same singular
//!    values and `β = Q₂·w`, where `w` is the minimum-norm solution of `Sᵀ·w = c`.
//! 2. **Golub–Kahan bidiagonalisation** `A = U_b·B·V_bᵀ` with Householder
//!    reflectors from both sides (`8/3·m³` flops, the dominant cost). The left
//!    reflectors are applied to `c` on the fly and only the right ones are kept.
//! 3. **Implicit QR on the bidiagonal `B`** (Golub & Van Loan, algorithm 8.6.2:
//!    Wilkinson shift, chasing the bulge with Givens rotations). The left
//!    rotations are applied to the vector `g = Uᵀ·c` directly, and the right ones
//!    are recorded (`O(m²)` of them) and applied, in reverse order, to
//!    `z = Σ⁺·g` at the end.
//!
//! Accumulating `U` and `V` as matrices (what a general SVD does) costs another
//! `≈ 12·m³` flops in the rotations; recording them makes the whole solve about
//! as expensive as the bidiagonalisation. 1 500 × 1 500 (rank 1 499): the SVD of
//! `R` takes 0.77 s (0.70 s bidiagonalisation, 0.07 s QR sweeps) and the whole OLS
//! fit about 1.3–1.4 s, against 14 s with the full SVD of linfa-linalg. The price is
//! memory: about `0.85·m²` recorded rotations of 24 bytes (1.9 million, 45 MB, for
//! `m = 1 500`; the work copy of the data is 18 MB), capped at `6·m²`.
//!
//! Singular values are found to absolute accuracy `≈ ε·σ_max` (the same backward
//! error as any Householder-based SVD): a diagonal entry `|d_i| ≤ ε·max|entry of B|`
//! is set to
//! zero and deflated, and a super-diagonal `e_i` is negligible when
//! `|e_i| ≤ ε·(|d_i| + |d_{i+1}|)`. Both thresholds are far below scikit-learn's
//! rank cutoff `ε·max(n, p)·σ_max`, so they never change the rank.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::model::{ModelError, TrainControl};

/// One recorded right rotation: columns `(i, j)` of `B` became
/// `(c·col_i + s·col_j, −s·col_i + c·col_j)`.
#[derive(Debug, Clone, Copy)]
struct Rotation {
    i: u32,
    j: u32,
    c: f64,
    s: f64,
}

/// A Householder reflector `H = I − τ·v·vᵀ` acting on the indices `offset..`.
#[derive(Debug)]
struct Reflector {
    offset: usize,
    v: Vec<f64>,
    tau: f64,
}

impl Reflector {
    /// `x[offset..] ← H·x[offset..]`.
    fn apply(&self, x: &mut [f64]) {
        if self.tau == 0.0 {
            return;
        }
        let tail = &mut x[self.offset..];
        let f = self.tau * dot(&self.v, tail);
        axpy(tail, -f, &self.v);
    }
}

/// The minimum-norm solution of `min ‖R·β − c‖` and the rank of `R` (singular
/// values `> ε·max_dim·σ_max`, scikit-learn's `cond`).
///
/// `r` is `m × p` with `m ≤ p` and `c` has `m` values. The returned `β` has `p`.
pub(super) fn min_norm_lstsq(
    r: ArrayView2<'_, f64>,
    c: ArrayView1<'_, f64>,
    max_dim: usize,
    control: &TrainControl<'_>,
) -> Result<(Array1<f64>, usize), ModelError> {
    let (m, p) = r.dim();
    if m > p || c.len() != m {
        return Err(ModelError::Numerical {
            message: "dimensiones inconsistentes en mínimos cuadrados".to_owned(),
        });
    }
    // f64::max ignores NaN: reject non-finite input before computing amax.
    if r.iter().chain(c.iter()).any(|v| !v.is_finite()) {
        return Err(non_finite());
    }
    // Scale to max |r_ij| = 1, as LAPACK does, so that the squares of the shift and
    // the reflectors' vᵀv cannot overflow or underflow. R' = R/amax, so the
    // solution of R·β = c is (solution of R'·β = c) / amax.
    let amax = r.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if m == 0 || amax == 0.0 {
        return Ok((Array1::zeros(p), 0)); // Every column is constant.
    }
    let g: Vec<f64> = c.to_vec();

    let (z, rank) = if m == p {
        let a: Vec<f64> = r.iter().map(|v| v / amax).collect(); // Row-major m × m.
        square_min_norm(a, m, g, max_dim, control)?
    } else {
        // Rᵀ = Q₂·S: the rows of R are the columns of Rᵀ (contiguous here).
        let mut w: Vec<f64> = r.iter().map(|v| v / amax).collect(); // Row-major m × p.
        let mut q2 = Vec::with_capacity(m);
        for k in 0..m {
            control.check()?;
            let (head, rest) = w.split_at_mut((k + 1) * p);
            let x = &mut head[k * p + k..];
            let (tau, alpha) = make_reflector(x);
            let refl = Reflector {
                offset: k,
                v: x.to_vec(),
                tau,
            };
            x[0] = alpha;
            x[1..].fill(0.0);
            if tau != 0.0 {
                for row in rest.chunks_exact_mut(p) {
                    refl.apply(row);
                }
            }
            q2.push(refl);
        }
        // Sᵀ is the lower triangle of the first m columns of w (w[j][k] = S[k][j]).
        let mut st = vec![0.0; m * m];
        for j in 0..m {
            st[j * m..j * m + j + 1].copy_from_slice(&w[j * p..j * p + j + 1]);
        }
        drop(w);
        let (sol, rank) = square_min_norm(st, m, g, max_dim, control)?;
        // β = Q₂·[w; 0] = H₀·H₁·…·H_{m−1}·[w; 0].
        let mut beta = sol;
        beta.resize(p, 0.0);
        for refl in q2.iter().rev() {
            refl.apply(&mut beta);
        }
        (beta, rank)
    };
    let beta = Array1::from_iter(z.into_iter().map(|v| v / amax));
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(non_finite());
    }
    Ok((beta, rank))
}

fn non_finite() -> ModelError {
    ModelError::Numerical {
        message: "coeficientes no finitos".to_owned(),
    }
}

/// Minimum-norm solution of the square `a·z = g` (`a` row-major `m × m`, entries
/// of magnitude ≤ 1) and the rank. See the module docs.
fn square_min_norm(
    mut a: Vec<f64>,
    m: usize,
    mut g: Vec<f64>,
    max_dim: usize,
    control: &TrainControl<'_>,
) -> Result<(Vec<f64>, usize), ModelError> {
    let Bidiagonal {
        mut d,
        mut e,
        right,
    } = bidiagonalize(&mut a, m, &mut g, control)?;
    drop(a);
    let rotations = bidiagonal_qr(&mut d, &mut e, &mut g, max_rotations(m), control)?;

    let s_max = d.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let cutoff = f64::EPSILON * max_dim as f64 * s_max;
    let mut rank = 0;
    let mut z: Vec<f64> = d
        .iter()
        .zip(&g)
        .map(|(&di, &gi)| {
            if di.abs() > cutoff {
                rank += 1;
                gi / di // A negative d_i carries the sign of its singular vector.
            } else {
                0.0
            }
        })
        .collect();
    control.check()?;
    // z ← R·z with R = G₁·G₂·…·G_T (the right rotations, last one first).
    for rot in rotations.iter().rev() {
        let (i, j) = (rot.i as usize, rot.j as usize);
        let (zi, zj) = (z[i], z[j]);
        z[i] = rot.c * zi - rot.s * zj;
        z[j] = rot.s * zi + rot.c * zj;
    }
    // z ← V_b·z with V_b = P₀·P₁·…: the last reflector first.
    for refl in right.iter().rev() {
        refl.apply(&mut z);
    }
    Ok((z, rank))
}

/// Householder bidiagonalisation of the row-major `m × m` matrix `a`, in place:
/// `a = U_b·B·V_bᵀ` with `B` upper bidiagonal (`d` on the diagonal, `e` above it).
/// `g ← U_bᵀ·g`; returns the right reflectors (`V_b = P₀·P₁·…`).
///
/// Every step updates the trailing rows once: the left reflector's rank-one
/// update and the right reflector's are fused in the same pass over each row
/// (contiguous memory), which halves the memory traffic.
fn bidiagonalize(
    a: &mut [f64],
    m: usize,
    g: &mut [f64],
    control: &TrainControl<'_>,
) -> Result<Bidiagonal, ModelError> {
    let mut d = vec![0.0; m];
    let mut e = vec![0.0; m.saturating_sub(1)];
    let mut right = Vec::with_capacity(m.saturating_sub(1));
    let mut v = Vec::with_capacity(m);
    let mut w = vec![0.0; m];
    for k in 0..m {
        control.check()?;
        // Left reflector: zero column k below the diagonal.
        v.clear();
        v.extend((k..m).map(|i| a[i * m + k]));
        let (tau, alpha) = make_reflector(&mut v);
        d[k] = alpha;
        let cols = k + 1..m;
        let wk = &mut w[..m - k - 1];
        wk.fill(0.0);
        if tau != 0.0 {
            // wᵀ = τ·vᵀ·A[k.., k+1..]
            for (i, &vi) in (k..m).zip(&v) {
                if vi != 0.0 {
                    axpy(wk, vi, &a[i * m + cols.start..(i + 1) * m]);
                }
            }
            wk.iter_mut().for_each(|x| *x *= tau);
            let f = tau * dot(&v, &g[k..]);
            axpy(&mut g[k..], -f, &v);
        }
        // Row k first: it defines the right reflector.
        let row_k = &mut a[k * m + cols.start..(k + 1) * m];
        if tau != 0.0 {
            axpy(row_k, -v[0], wk);
        }
        if k + 1 == m {
            break;
        }
        let mut u = row_k.to_vec();
        let (tau2, alpha2) = make_reflector(&mut u);
        e[k] = alpha2;
        for (i, &vi) in (k + 1..m).zip(&v[1..]) {
            let row = &mut a[i * m + cols.start..(i + 1) * m];
            if tau != 0.0 && vi != 0.0 {
                axpy(row, -vi, wk);
            }
            if tau2 != 0.0 {
                let f = tau2 * dot(row, &u);
                axpy(row, -f, &u);
            }
        }
        right.push(Reflector {
            offset: k + 1,
            v: u,
            tau: tau2,
        });
    }
    Ok(Bidiagonal { d, e, right })
}

/// The result of [`bidiagonalize`]: `B` (diagonal `d`, super-diagonal `e`) and the
/// right reflectors (`V_b = P₀·P₁·…`).
struct Bidiagonal {
    d: Vec<f64>,
    e: Vec<f64>,
    right: Vec<Reflector>,
}

/// Diagonalises the upper bidiagonal `(d, e)` with implicit-shift QR sweeps. The
/// left rotations are applied to `g`; the right ones are returned, in order.
///
/// Fails with `Numerical` ("la SVD no converge") if more than `max_rotations`
/// rotations are needed: the memory stays bounded (see [`max_rotations`]).
fn bidiagonal_qr(
    d: &mut [f64],
    e: &mut [f64],
    g: &mut [f64],
    max_rotations: usize,
    control: &TrainControl<'_>,
) -> Result<Vec<Rotation>, ModelError> {
    let m = d.len();
    if m < 2 {
        return Ok(Vec::new());
    }
    // About 0.85·m² are needed in practice: reserve m² once instead of growing.
    let mut rotations = Vec::with_capacity((m * m).min(max_rotations));
    let eps = f64::EPSILON;
    let bnorm = d
        .iter()
        .chain(e.iter())
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    let small = eps * bnorm;
    let mut sweeps = 0_usize;
    let mut hi = m - 1;
    loop {
        // Negligible super-diagonal entries split B into independent blocks.
        for i in 0..hi {
            if e[i].abs() <= eps * (d[i].abs() + d[i + 1].abs()) {
                e[i] = 0.0;
            }
        }
        while hi > 0 && e[hi - 1] == 0.0 {
            hi -= 1;
        }
        if hi == 0 {
            return Ok(rotations);
        }
        let mut lo = hi - 1;
        while lo > 0 && e[lo - 1] != 0.0 {
            lo -= 1;
        }
        // The active block is lo..=hi; every e[lo..hi] is non-zero.
        if sweeps % 64 == 0 {
            control.check()?;
        }
        sweeps += 1;
        // A sweep or a chase adds at most m rotations, so this bounds the memory
        // to max_rotations + m.
        if rotations.len() > max_rotations {
            return Err(ModelError::Numerical {
                message: "la SVD no converge".to_owned(),
            });
        }
        if let Some(i) = (lo..hi).find(|&i| d[i].abs() <= small) {
            // Zero d_i: chase e_i along row i with left rotations (rows j, i).
            d[i] = 0.0;
            let mut x = e[i];
            e[i] = 0.0;
            for j in i + 1..=hi {
                let (c, s, r) = givens(d[j], x);
                d[j] = r;
                if j < hi {
                    x = -s * e[j];
                    e[j] *= c;
                }
                let (gj, gi) = (g[j], g[i]);
                g[j] = c * gj + s * gi;
                g[i] = -s * gj + c * gi;
            }
            continue;
        }
        if d[hi].abs() <= small {
            // Zero d_hi: chase e_{hi−1} up column hi with right rotations (j, hi).
            d[hi] = 0.0;
            let mut x = e[hi - 1];
            e[hi - 1] = 0.0;
            for j in (lo..hi).rev() {
                let (c, s, r) = givens(d[j], x);
                d[j] = r;
                if j > lo {
                    x = -s * e[j - 1];
                    e[j - 1] *= c;
                }
                rotations.push(rotation(j, hi, c, s));
            }
            continue;
        }
        golub_kahan_step(d, e, g, lo, hi, &mut rotations);
    }
}

/// One implicit QR sweep with Wilkinson shift on the block `lo..=n` (GVL 8.6.2).
fn golub_kahan_step(
    d: &mut [f64],
    e: &mut [f64],
    g: &mut [f64],
    lo: usize,
    n: usize,
    rotations: &mut Vec<Rotation>,
) {
    // Eigenvalue of the trailing 2 × 2 of BᵀB closest to its last entry.
    let prev = if n - 1 > lo { e[n - 2] } else { 0.0 };
    let t11 = d[n - 1] * d[n - 1] + prev * prev;
    let t12 = d[n - 1] * e[n - 1];
    let t22 = d[n] * d[n] + e[n - 1] * e[n - 1];
    let mu = if t12 == 0.0 {
        t22
    } else {
        let delta = (t11 - t22) / 2.0;
        let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
        t22 - t12 * t12 / (delta + sign * delta.hypot(t12))
    };
    let mut y = d[lo] * d[lo] - mu;
    let mut z = d[lo] * e[lo];
    for k in lo..n {
        // Right rotation on columns k, k+1: zeroes z (the bulge in row k−1).
        let (c, s, r) = givens(y, z);
        if k > lo {
            e[k - 1] = r;
        }
        let (dk, ek, dk1) = (d[k], e[k], d[k + 1]);
        d[k] = c * dk + s * ek;
        e[k] = -s * dk + c * ek;
        let bulge = s * dk1;
        d[k + 1] = c * dk1;
        rotations.push(rotation(k, k + 1, c, s));
        // Left rotation on rows k, k+1: zeroes the bulge below the diagonal.
        let (c, s, r) = givens(d[k], bulge);
        d[k] = r;
        let (ek, dk1) = (e[k], d[k + 1]);
        e[k] = c * ek + s * dk1;
        d[k + 1] = -s * ek + c * dk1;
        let (gk, gk1) = (g[k], g[k + 1]);
        g[k] = c * gk + s * gk1;
        g[k + 1] = -s * gk + c * gk1;
        if k + 1 < n {
            y = e[k];
            z = s * e[k + 1];
            e[k + 1] *= c;
        }
    }
}

/// The most right rotations [`bidiagonal_qr`] may record for an `m × m` problem:
/// `6·m²`, like LAPACK's `MAXITR·n²` limit on QR steps in `dbdsqr`. In practice
/// it needs about `0.85·m²` for large `m` (1.25·m² at `m = 30`), 24 bytes each.
fn max_rotations(m: usize) -> usize {
    6 * m * m
}

fn rotation(i: usize, j: usize, c: f64, s: f64) -> Rotation {
    // m ≤ max columns of a table (far below u32::MAX).
    Rotation {
        i: i as u32,
        j: j as u32,
        c,
        s,
    }
}

/// `(c, s, r)` with `c·a + s·b = r` and `−s·a + c·b = 0`.
fn givens(a: f64, b: f64) -> (f64, f64, f64) {
    if b == 0.0 {
        return (1.0, 0.0, a);
    }
    let r = a.hypot(b);
    (a / r, b / r, r)
}

/// LAPACK's `dlarfg`: turns `x` into `v` and returns `(τ, β)` with
/// `(I − τ·v·vᵀ)·x = β·e₁`. `τ = 0` (identity) when `x[1..]` is already zero.
fn make_reflector(x: &mut [f64]) -> (f64, f64) {
    let Some((&x0, tail)) = x.split_first() else {
        return (0.0, 0.0);
    };
    let tail_norm = scaled_norm(tail);
    if tail_norm == 0.0 {
        return (0.0, x0);
    }
    let norm = x0.hypot(tail_norm);
    let beta = if x0 > 0.0 { -norm } else { norm };
    x[0] = x0 - beta;
    // vᵀv = (x0 − β)² + ‖tail‖² = 2·|β|·(|β| + |x0|), without cancellation.
    let vtv = 2.0 * norm * (norm + x0.abs());
    (2.0 / vtv, beta)
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

/// Dot product with eight independent accumulators, so that the compiler can use
/// SIMD; the order of the additions is fixed, so the result is deterministic.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0_f64; 8];
    let split = n - n % 8;
    for (x, y) in a[..split].chunks_exact(8).zip(b[..split].chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f64 = a[split..].iter().zip(&b[split..]).map(|(x, y)| x * y).sum();
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

/// `y ← y + α·x`.
fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
    y.iter_mut().zip(x).for_each(|(y, &x)| *y += alpha * x);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// splitmix64 in [0, 1).
    fn uniform(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Reference: one-sided Jacobi SVD (Hestenes), slow but very accurate and
    /// independent of the code under test. (linfa-linalg's SVD, which OLS used
    /// before, is not a good reference: on one of these 8 × 8 rank-3 matrices it
    /// returns σ₂ = 1.84517 instead of 1.84509.) Also returns `σ_max / σ_min` over
    /// the singular values above the cutoff.
    fn reference(r: &Array2<f64>, c: &Array1<f64>, max_dim: usize) -> (Array1<f64>, usize, f64) {
        let amax = r.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let mut a = r / amax;
        let p = a.ncols();
        let mut v = Array2::<f64>::eye(p);
        let mut converged = false;
        let mut last_off = 0.0;
        // Columns with a norm below ε² (the matrix is scaled to max |entry| = 1)
        // are numerically zero: their mutual cosines are only noise.
        let negligible = f64::EPSILON.powi(4);
        for _ in 0..100 {
            let mut off = 0.0_f64;
            for i in 0..p {
                for j in i + 1..p {
                    let (ci, cj) = (a.column(i), a.column(j));
                    let (al, be, ga) = (ci.dot(&ci), cj.dot(&cj), ci.dot(&cj));
                    if al <= negligible
                        || be <= negligible
                        || ga.abs() <= f64::EPSILON * al.sqrt() * be.sqrt()
                    {
                        continue;
                    }
                    off = off.max(ga.abs() / (al.sqrt() * be.sqrt()));
                    let zeta = (be - al) / (2.0 * ga);
                    let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                    let cs = 1.0 / (1.0 + t * t).sqrt();
                    let sn = cs * t;
                    for mat in [&mut a, &mut v] {
                        for mut row in mat.rows_mut() {
                            let (x, y) = (row[i], row[j]);
                            row[i] = cs * x - sn * y;
                            row[j] = sn * x + cs * y;
                        }
                    }
                }
            }
            // Every pair of columns orthogonal to a cosine of a few ε (exactly ε is
            // not always reachable: rounding keeps rotating near-zero columns).
            last_off = off;
            if off <= 8.0 * f64::EPSILON {
                converged = true;
                break;
            }
        }
        assert!(
            converged,
            "Jacobi reference did not converge: {:?} off {last_off:e}",
            r.dim()
        );
        let sigma: Vec<f64> = a
            .columns()
            .into_iter()
            .map(|col| col.dot(&col).sqrt())
            .collect();
        let s_max = sigma.iter().copied().fold(0.0_f64, f64::max);
        let cutoff = f64::EPSILON * max_dim as f64 * s_max;
        let mut beta = Array1::<f64>::zeros(p);
        let (mut rank, mut s_min) = (0, s_max);
        for (k, &s) in sigma.iter().enumerate() {
            if s > cutoff {
                beta.scaled_add(a.column(k).dot(c) / (s * s), &v.column(k));
                rank += 1;
                s_min = s_min.min(s);
            }
        }
        (beta / amax, rank, s_max / s_min)
    }

    /// An orthogonal matrix: modified Gram–Schmidt, twice, on random columns.
    fn orthogonal(p: usize, seed: u64) -> Array2<f64> {
        let mut q = low_rank(p, p, p, 1.0, seed);
        for _ in 0..2 {
            for j in 0..p {
                for k in 0..j {
                    let f = q.column(k).dot(&q.column(j));
                    let qk = q.column(k).to_owned();
                    q.column_mut(j).scaled_add(-f, &qk);
                }
                let n = q.column(j).dot(&q.column(j)).sqrt();
                q.column_mut(j).mapv_inplace(|x| x / n);
            }
        }
        q
    }

    /// `m × p` with rank `rank` (product of random factors) times `scale`.
    fn low_rank(m: usize, p: usize, rank: usize, scale: f64, seed: u64) -> Array2<f64> {
        let mut s = seed;
        let a = Array2::from_shape_fn((m, rank), |_| uniform(&mut s) * 2.0 - 1.0);
        let b = Array2::from_shape_fn((rank, p), |_| uniform(&mut s) * 2.0 - 1.0);
        a.dot(&b) * scale
    }

    /// Observed errors are at most ≈ 7·κ·ε on the matrices below (2026-10-20).
    const TOL_FACTOR: f64 = 30.0;

    fn check(r: &Array2<f64>, seed: u64) {
        let mut s = seed;
        let c = Array1::from_shape_fn(r.nrows(), |_| uniform(&mut s) - 0.5);
        let max_dim = r.ncols().max(r.nrows()) + 3;
        let (beta, rank) = min_norm_lstsq(r.view(), c.view(), max_dim, &TrainControl::none())
            .unwrap_or_else(|e| panic!("{e}"));
        let (expected, rank_ref, kappa) = reference(r, &c, max_dim);
        assert_eq!(rank, rank_ref, "{:?}", r.dim());
        // Two backward-stable solvers agree to about κ·ε (relative to ‖β‖∞).
        let tol = (TOL_FACTOR * f64::EPSILON * kappa).max(1e-13);
        let scale = expected.iter().fold(1e-300_f64, |a, v| a.max(v.abs()));
        let err = beta
            .iter()
            .zip(&expected)
            .fold(0.0_f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(
            err <= tol * scale,
            "{:?}: error {err:e} (scale {scale:e}, κ {kappa:e})",
            r.dim()
        );
    }

    #[test]
    fn matches_the_full_svd_on_square_wide_and_rank_deficient_matrices() {
        for (seed, (m, p, rank)) in [
            (1_u64, (1, 1, 1)),
            (2, (2, 2, 1)),
            (3, (6, 6, 6)),
            (4, (8, 8, 3)),
            (5, (30, 30, 29)),
            (6, (40, 40, 12)),
            (7, (3, 9, 3)),
            (8, (5, 20, 4)),
            (9, (17, 40, 9)),
            (10, (60, 60, 1)),
        ] {
            check(&low_rank(m, p, rank, 1.0, seed), seed);
            check(&low_rank(m, p, rank, 1e-150, seed + 100), seed);
            check(&low_rank(m, p, rank, 1e150, seed + 200), seed);
        }
    }

    #[test]
    fn handles_zero_rows_columns_and_diagonals() {
        // Upper triangular with zeros on the diagonal (the R of a QR with a
        // constant or collinear column) and an all-zero row.
        let r = ndarray::array![
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        check(&r, 1);
        let r = ndarray::array![[0.0, 1.0], [0.0, 0.0]];
        check(&r, 2);
        let r = Array2::from_shape_fn((5, 5), |(i, j)| if i == j { (i + 1) as f64 } else { 0.0 });
        check(&r, 3);
        let r = Array2::from_shape_fn((5, 5), |(i, j)| if j == i + 1 { 1.0 } else { 0.0 });
        check(&r, 4);
    }

    /// Widely spread singular values (10⁰ … 10⁻¹²) plus two below the cutoff.
    #[test]
    fn graded_singular_values() {
        let mut s = 99;
        for p in [10_usize, 25] {
            let qu = orthogonal(p, s);
            let sigma = Array2::from_shape_fn((p, p), |(i, j)| {
                if i != j {
                    0.0
                } else if i + 2 < p {
                    10f64.powf(-(i as f64) * 12.0 / (p - 3) as f64)
                } else if i + 1 < p {
                    1e-17
                } else {
                    0.0
                }
            });
            let r = qu.dot(&sigma).dot(&qu.t());
            s += 1;
            check(&r, s);
        }
    }

    #[test]
    fn cancellation_is_checked() {
        let flag = std::sync::atomic::AtomicBool::new(true);
        let r = low_rank(8, 8, 4, 1.0, 1);
        let c = Array1::ones(8);
        let ctl = TrainControl::none().with_cancel(&flag);
        assert_eq!(
            min_norm_lstsq(r.view(), c.view(), 8, &ctl),
            Err(ModelError::Cancelled)
        );
    }

    #[test]
    fn rejects_non_finite_input() {
        let ctl = TrainControl::none();
        let mut r = low_rank(4, 4, 4, 1.0, 3);
        let c = Array1::ones(4);
        r[[1, 2]] = f64::NAN;
        assert_eq!(
            min_norm_lstsq(r.view(), c.view(), 4, &ctl),
            Err(non_finite())
        );
        let r = low_rank(4, 6, 3, 1.0, 3);
        let mut c = Array1::ones(4);
        c[3] = f64::INFINITY;
        assert_eq!(
            min_norm_lstsq(r.view(), c.view(), 6, &ctl),
            Err(non_finite())
        );
    }

    #[test]
    fn more_rows_than_columns_is_an_error() {
        // The caller always passes the m × p R with m = min(n, p) ≤ p.
        let r = low_rank(5, 3, 3, 1.0, 4);
        let c = Array1::ones(5);
        let err = min_norm_lstsq(r.view(), c.view(), 5, &TrainControl::none());
        assert!(matches!(err, Err(ModelError::Numerical { .. })), "{err:?}");
    }

    #[test]
    fn non_convergence_fails_instead_of_growing() {
        let mut s = 11;
        let m = 30;
        let fresh = |s: &mut u64| {
            let d: Vec<f64> = (0..m).map(|_| uniform(s) + 0.5).collect();
            let e: Vec<f64> = (0..m - 1).map(|_| uniform(s) + 0.5).collect();
            (d, e, vec![1.0; m])
        };
        let ctl = TrainControl::none();
        // With the real cap it converges, with far fewer rotations than the cap.
        let (mut d, mut e, mut g) = fresh(&mut s);
        let rots = bidiagonal_qr(&mut d, &mut e, &mut g, max_rotations(m), &ctl)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(rots.len() < 2 * m * m, "{}", rots.len()); // 1.25·m² here, 0.85·m² at m = 1500
                                                           // A tiny cap stops it with an error.
        let (mut d, mut e, mut g) = fresh(&mut s);
        let err = bidiagonal_qr(&mut d, &mut e, &mut g, m / 2, &ctl);
        assert!(
            matches!(&err, Err(ModelError::Numerical { message }) if message.contains("no converge")),
            "{err:?}"
        );
    }
}
