//! The multilayer perceptron: dense layers, activations, forward pass and manual
//! backpropagation in float32 (`tools/nn_reference.py::forward` / `gradients`).
//!
//! Matrices are row-major `Vec<f32>`. Every product accumulates in float64 in a
//! fixed order (inner index ascending) and rounds once to float32. Large products
//! are split across threads (rayon) **by output rows**: each output element is still
//! the same sum in the same order, so the result is bit for bit the same with 1 or
//! 64 threads and training stays deterministic.
//!
//! Shapes are checked at the public entry points ([`Mlp::new`], deserialisation,
//! [`Mlp::forward`], [`Mlp::gradients`], [`Mlp::hidden_activations`]): a malformed
//! network or input is a [`ModelError`], never a panic.

use mltutor_core::model::ModelError;
use mltutor_core::nn_data::{Activation, LossKind};
use mltutor_core::nn_stats::DenseView;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// `Dense(units)`: `z = a·W + b` with `kernel` row-major `inputs × units` (Keras
/// layout).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dense {
    pub inputs: usize,
    pub units: usize,
    pub kernel: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Dense {
    /// The layer as the core's statistics see it.
    pub fn view(&self) -> DenseView<'_> {
        DenseView {
            inputs: self.inputs,
            units: self.units,
            kernel: &self.kernel,
            bias: &self.bias,
        }
    }
}

/// A trained (or initial) network: hidden layers with `hidden`, the last one with
/// `output`. Dropout is only applied while training ([`crate::train`]).
///
/// The fields are public (the parity tests build networks from Keras' weights), but
/// every entry point checks the shapes with [`Mlp::validate`], and deserialising
/// refuses an inconsistent network.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "MlpRepr")]
pub struct Mlp {
    pub layers: Vec<Dense>,
    pub hidden: Activation,
    pub output: Activation,
}

/// The serialised form of [`Mlp`], validated by `TryFrom`.
#[derive(Deserialize)]
struct MlpRepr {
    layers: Vec<Dense>,
    hidden: Activation,
    output: Activation,
}

impl TryFrom<MlpRepr> for Mlp {
    type Error = ModelError;

    fn try_from(r: MlpRepr) -> Result<Self, ModelError> {
        Mlp::new(r.layers, r.hidden, r.output)
    }
}

fn shape_error(reason: impl Into<String>) -> ModelError {
    ModelError::InvalidConfig {
        reason: reason.into(),
    }
}

/// Products with at least this many multiply-adds are split across threads (below,
/// the cost of scheduling is larger than the work: Iris never goes parallel; measured
/// on California: 2^15 halves the speed of a [64, 64] network, 2^21 slows 5 × 256).
const PARALLEL_MIN_WORK: usize = 1 << 19;
/// Rows of `aᵀ·d` computed by one task (they share the reads of `d`).
const AT_B_BLOCK: usize = 16;

fn parallel(n: usize, k: usize, m: usize) -> bool {
    n.saturating_mul(k).saturating_mul(m) >= PARALLEL_MIN_WORK && rayon::current_num_threads() > 1
}

/// One output row of [`matmul`]: `c_row = a_row · b`, summing `kk` ascending in f64.
fn matmul_row(a_row: &[f32], b: &[f32], m: usize, acc: &mut [f64], c_row: &mut [f32]) {
    acc.iter_mut().for_each(|v| *v = 0.0);
    for (kk, &aik) in a_row.iter().enumerate() {
        let aik = f64::from(aik);
        let brow = &b[kk * m..(kk + 1) * m];
        for (cj, &bj) in acc.iter_mut().zip(brow) {
            *cj += aik * f64::from(bj);
        }
    }
    for (c, v) in c_row.iter_mut().zip(acc.iter()) {
        *c = *v as f32;
    }
}

/// `c (n×m) = a (n×k) · b (k×m)`. Products of float32 values are accumulated in
/// float64 and rounded once: the error is half an ulp, so the result is as close as
/// possible to Keras' float32 kernels whatever order those use. Rows of `c` are
/// computed in parallel when the product is large (same sums, same order).
pub(crate) fn matmul(a: &[f32], n: usize, k: usize, b: &[f32], m: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; n * m];
    if m == 0 || n == 0 {
        return c;
    }
    if k == 0 {
        return c;
    }
    if parallel(n, k, m) {
        c.par_chunks_mut(m).zip(a.par_chunks(k)).for_each_init(
            || vec![0.0_f64; m],
            |acc, (c_row, a_row)| matmul_row(a_row, b, m, acc, c_row),
        );
    } else {
        let mut acc = vec![0.0_f64; m];
        for (c_row, a_row) in c.chunks_mut(m).zip(a.chunks(k)) {
            matmul_row(a_row, b, m, &mut acc, c_row);
        }
    }
    c
}

/// Rows `kk0..kk0 + g.len()/m` of `aᵀ·d`: `g[kk, :] = Σ_i a[i, kk]·d[i, :]`, `i`
/// ascending, in f64.
fn at_b_block(a: &[f32], n: usize, k: usize, d: &[f32], m: usize, kk0: usize, g: &mut [f32]) {
    let rows = g.len() / m;
    let mut acc = vec![0.0_f64; rows * m];
    for i in 0..n {
        let drow = &d[i * m..(i + 1) * m];
        let arow = &a[i * k + kk0..i * k + kk0 + rows];
        for (r, &aik) in arow.iter().enumerate() {
            let aik = f64::from(aik);
            for (gj, &dj) in acc[r * m..(r + 1) * m].iter_mut().zip(drow) {
                *gj += aik * f64::from(dj);
            }
        }
    }
    for (o, v) in g.iter_mut().zip(&acc) {
        *o = *v as f32;
    }
}

/// `aᵀ (k×n) · d (n×m)` → `k×m` (kernel gradient), accumulated in float64; blocks of
/// output rows in parallel.
fn matmul_at_b(a: &[f32], n: usize, k: usize, d: &[f32], m: usize) -> Vec<f32> {
    let mut g = vec![0.0_f32; k * m];
    if m == 0 || k == 0 {
        return g;
    }
    let block = AT_B_BLOCK * m;
    if parallel(n, k, m) {
        g.par_chunks_mut(block)
            .enumerate()
            .for_each(|(b, rows)| at_b_block(a, n, k, d, m, b * AT_B_BLOCK, rows));
    } else {
        for (b, rows) in g.chunks_mut(block).enumerate() {
            at_b_block(a, n, k, d, m, b * AT_B_BLOCK, rows);
        }
    }
    g
}

/// `d (n×m) · wᵀ (m×k)` → `n×k` (gradient with respect to the layer input),
/// accumulated in float64. `w` (`k×m`) is transposed once so that the inner loop is
/// contiguous: `out[i, kk] = Σ_j d[i, j]·w[kk, j]`, `j` ascending, exactly the sum of
/// the direct dot product.
fn matmul_a_bt(d: &[f32], n: usize, m: usize, w: &[f32], k: usize) -> Vec<f32> {
    let mut wt = vec![0.0_f32; m * k];
    for kk in 0..k {
        for j in 0..m {
            wt[j * k + kk] = w[kk * m + j];
        }
    }
    matmul(d, n, m, &wt, k)
}

/// Applies `act` to a row-major `n × units` matrix in place.
pub(crate) fn activate(z: &mut [f32], units: usize, act: Activation) {
    match act {
        // Not `v.max(0.0)`: `f32::max` returns the non-NaN operand, so it would turn
        // a NaN into 0 and hide a divergence. `tf.nn.relu(NaN)` is NaN.
        Activation::Relu => z.iter_mut().for_each(|v| {
            if *v < 0.0 {
                *v = 0.0;
            }
        }),
        Activation::Tanh => z.iter_mut().for_each(|v| *v = v.tanh()),
        Activation::Sigmoid => z.iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp())),
        Activation::Softmax => {
            for row in z.chunks_mut(units.max(1)) {
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for v in row.iter_mut() {
                    *v = (*v - m).exp();
                    sum += *v;
                }
                row.iter_mut().for_each(|v| *v /= sum);
            }
        }
        Activation::Linear => {}
    }
}

/// Derivative of the activation from its output `a` (and pre-activation `z`).
fn act_grad(a: f32, z: f32, act: Activation) -> f32 {
    match act {
        Activation::Relu => {
            if z > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Activation::Tanh => 1.0 - a * a,
        Activation::Sigmoid => a * (1.0 - a),
        _ => 1.0,
    }
}

/// `(kernel, bias)` gradient of every layer, in layer order.
pub type Gradients = Vec<(Vec<f32>, Vec<f32>)>;

/// Dropout masks of one batch: one `rows × units` 0/1 matrix per hidden layer
/// (inverted dropout: kept activations are divided by `1 − rate`).
#[derive(Debug, Clone, PartialEq)]
pub struct DropoutMasks {
    pub rate: f32,
    pub masks: Vec<Vec<f32>>,
}

/// What the forward pass keeps for backpropagation.
pub(crate) struct ForwardCache {
    /// Pre-activations of every layer.
    pub z: Vec<Vec<f32>>,
    /// Inputs of every layer (`a[0]` = x, after dropout for hidden ones) and the output
    /// (`a[L]`).
    pub a: Vec<Vec<f32>>,
    /// Hidden activations before dropout.
    pub pre: Vec<Vec<f32>>,
}

impl Mlp {
    /// A network from its layers, checked with [`Mlp::validate`].
    pub fn new(
        layers: Vec<Dense>,
        hidden: Activation,
        output: Activation,
    ) -> Result<Self, ModelError> {
        let mlp = Mlp {
            layers,
            hidden,
            output,
        };
        mlp.validate()?;
        Ok(mlp)
    }

    /// At least one layer, every kernel `inputs × units` and every bias `units` long,
    /// at least one neuron per layer, and each layer's inputs = the previous layer's
    /// units.
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.layers.is_empty() {
            return Err(shape_error("la red no tiene capas"));
        }
        for (i, l) in self.layers.iter().enumerate() {
            if l.units == 0 || l.inputs == 0 {
                return Err(shape_error(format!("la capa {} no tiene neuronas", i + 1)));
            }
            if l.kernel.len() != l.inputs.saturating_mul(l.units) || l.bias.len() != l.units {
                return Err(shape_error(format!(
                    "los pesos de la capa {} no tienen la forma ({}, {})",
                    i + 1,
                    l.inputs,
                    l.units
                )));
            }
            if i > 0 && l.inputs != self.layers[i - 1].units {
                return Err(shape_error(format!(
                    "la capa {} recibe {} entradas y la anterior tiene {} neuronas",
                    i + 1,
                    l.inputs,
                    self.layers[i - 1].units
                )));
            }
        }
        Ok(())
    }

    /// Whether every weight and bias is finite (a diverged training leaves NaN).
    pub fn is_finite(&self) -> bool {
        self.layers
            .iter()
            .all(|l| l.kernel.iter().chain(&l.bias).all(|v| v.is_finite()))
    }

    /// Input features.
    pub fn n_inputs(&self) -> usize {
        self.layers.first().map_or(0, |l| l.inputs)
    }

    /// Output neurons.
    pub fn n_outputs(&self) -> usize {
        self.layers.last().map_or(0, |l| l.units)
    }

    /// `[input, hidden…, output]`.
    pub fn architecture(&self) -> Vec<usize> {
        let mut out = vec![self.n_inputs()];
        out.extend(self.layers.iter().map(|l| l.units));
        out
    }

    /// Number of weights and biases (`count_params()`).
    pub fn n_params(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.kernel.len() + l.bias.len())
            .sum()
    }

    /// The layers as the core's statistics see them.
    pub fn views(&self) -> Vec<DenseView<'_>> {
        self.layers.iter().map(Dense::view).collect()
    }

    /// The network is valid and `x` has `n` rows of [`Mlp::n_inputs`] values.
    fn check_input(&self, x: &[f32], n: usize) -> Result<(), ModelError> {
        self.validate()?;
        let p = self.n_inputs();
        if x.len() != n.saturating_mul(p) {
            return Err(if n > 0 && x.len() % n == 0 {
                ModelError::FeatureMismatch {
                    expected: p,
                    found: x.len() / n,
                }
            } else {
                ModelError::InvalidInput {
                    reason: format!("se esperaban {n} filas de {p} valores y hay {}", x.len()),
                }
            });
        }
        Ok(())
    }

    /// Forward pass keeping what backpropagation needs. The caller has checked the
    /// shapes ([`Mlp::check_input`], masks).
    pub(crate) fn forward_cache(
        &self,
        x: &[f32],
        n: usize,
        masks: Option<&DropoutMasks>,
    ) -> ForwardCache {
        let last = self.layers.len().saturating_sub(1);
        let mut zs = Vec::with_capacity(self.layers.len());
        let mut acts = Vec::with_capacity(self.layers.len() + 1);
        let mut pre = Vec::with_capacity(last);
        acts.push(x.to_vec());
        for (i, l) in self.layers.iter().enumerate() {
            let input = &acts[i];
            let mut z = matmul(input, n, l.inputs, &l.kernel, l.units);
            for row in z.chunks_mut(l.units.max(1)) {
                for (v, b) in row.iter_mut().zip(&l.bias) {
                    *v += b;
                }
            }
            let mut a = z.clone();
            activate(
                &mut a,
                l.units,
                if i == last { self.output } else { self.hidden },
            );
            zs.push(z);
            if i < last {
                pre.push(a.clone());
                if let Some(m) = masks.filter(|m| m.rate > 0.0) {
                    let keep = (1.0_f64 - f64::from(m.rate)) as f32;
                    for (v, mask) in a.iter_mut().zip(&m.masks[i]) {
                        *v = *v * mask / keep;
                    }
                }
            }
            acts.push(a);
        }
        ForwardCache {
            z: zs,
            a: acts,
            pre,
        }
    }

    /// Inference on `n` row-major float32 rows (already scaled): `n × n_outputs`.
    pub fn forward(&self, x: &[f32], n: usize) -> Result<Vec<f32>, ModelError> {
        self.check_input(x, n)?;
        let mut a = x.to_vec();
        let last = self.layers.len() - 1;
        for (i, l) in self.layers.iter().enumerate() {
            let mut z = matmul(&a, n, l.inputs, &l.kernel, l.units);
            for row in z.chunks_mut(l.units) {
                for (v, b) in row.iter_mut().zip(&l.bias) {
                    *v += b;
                }
            }
            activate(
                &mut z,
                l.units,
                if i == last { self.output } else { self.hidden },
            );
            a = z;
        }
        Ok(a)
    }

    /// Outputs of every hidden layer (after the activation, before dropout; C26) of
    /// `n` rows already **scaled**: one `n × units` row-major matrix per hidden layer.
    /// [`crate::NeuralNetwork::hidden_activations`] takes original units.
    pub fn hidden_activations(&self, x: &[f32], n: usize) -> Result<Vec<Vec<f32>>, ModelError> {
        self.check_input(x, n)?;
        Ok(self.forward_cache(x, n, None).pre)
    }

    /// Mean loss of a batch, the gradient of every layer `(kernel, bias)` and the
    /// batch output (the predictions before the step, for the metric)
    /// (`nn_reference.gradients`): cross-entropies from the logits, MSE through the
    /// output activation (softmax included).
    ///
    /// Errors: malformed network, `x`/`y` lengths that are not `n` rows, or dropout
    /// masks without one `n × units` matrix per hidden layer.
    pub fn gradients(
        &self,
        x: &[f32],
        y: &[f32],
        n: usize,
        loss: LossKind,
        masks: Option<&DropoutMasks>,
    ) -> Result<(f32, Gradients, Vec<f32>), ModelError> {
        self.check_input(x, n)?;
        let lsz = self.layers.len();
        let k = self.n_outputs();
        if y.len() != n * k {
            return Err(ModelError::LengthMismatch {
                rows: n,
                targets: y.len() / k.max(1),
            });
        }
        if let Some(m) = masks.filter(|m| m.rate > 0.0) {
            let ok = m.masks.len() == lsz - 1
                && m.masks
                    .iter()
                    .zip(&self.layers)
                    .all(|(mask, l)| mask.len() == n * l.units);
            if !ok {
                return Err(shape_error(
                    "las máscaras de dropout no tienen la forma de la red",
                ));
            }
        }
        let f = self.forward_cache(x, n, masks);
        let (z_out, p) = (&f.z[lsz - 1], &f.a[lsz]);
        let per = crate::loss::per_sample_loss(loss, self.output, z_out, p, y, k);
        let loss_value = crate::loss::mean_f32(&per);
        let nk = (n * k) as f32;
        let mut dz: Vec<f32> = match (loss, self.output) {
            (LossKind::BinaryCrossentropy, Activation::Sigmoid) => {
                p.iter().zip(y).map(|(p, y)| (p - y) / nk).collect()
            }
            (LossKind::CategoricalCrossentropy, Activation::Softmax) => {
                p.iter().zip(y).map(|(p, y)| (p - y) / n as f32).collect()
            }
            _ => {
                let dp: Vec<f32> = p.iter().zip(y).map(|(p, y)| 2.0 * (p - y) / nk).collect();
                if self.output == Activation::Softmax {
                    let mut out = vec![0.0_f32; dp.len()];
                    for i in 0..n {
                        let r = i * k..(i + 1) * k;
                        let s: f32 = dp[r.clone()]
                            .iter()
                            .zip(&p[r.clone()])
                            .map(|(d, p)| d * p)
                            .sum();
                        for j in r {
                            out[j] = p[j] * (dp[j] - s);
                        }
                    }
                    out
                } else {
                    dp.iter()
                        .zip(p)
                        .zip(z_out)
                        .map(|((d, a), z)| d * act_grad(*a, *z, self.output))
                        .collect()
                }
            }
        };
        let mut grads = Vec::with_capacity(lsz);
        for i in (0..lsz).rev() {
            let l = &self.layers[i];
            let a_prev = &f.a[i];
            let gw = matmul_at_b(a_prev, n, l.inputs, &dz, l.units);
            let mut gb = vec![0.0_f64; l.units];
            for row in dz.chunks(l.units) {
                for (g, d) in gb.iter_mut().zip(row) {
                    *g += f64::from(*d);
                }
            }
            let gb: Vec<f32> = gb.into_iter().map(|v| v as f32).collect();
            grads.push((gw, gb));
            if i > 0 {
                let mut da = matmul_a_bt(&dz, n, l.units, &l.kernel, l.inputs);
                if let Some(m) = masks.filter(|m| m.rate > 0.0) {
                    let keep = (1.0_f64 - f64::from(m.rate)) as f32;
                    for (v, mask) in da.iter_mut().zip(&m.masks[i - 1]) {
                        *v = *v * mask / keep;
                    }
                }
                let (pre, z) = (&f.pre[i - 1], &f.z[i - 1]);
                for ((d, a), zz) in da.iter_mut().zip(pre).zip(z) {
                    *d *= act_grad(*a, *zz, self.hidden);
                }
                dz = da;
            }
        }
        grads.reverse();
        let out = f.a[lsz].clone();
        Ok((loss_value, grads, out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(inputs: usize, units: usize, seed: u32) -> Dense {
        Dense {
            inputs,
            units,
            kernel: (0..inputs * units)
                .map(|i| {
                    ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) % 1000) as f32
                        / 500.0
                        - 1.0
                })
                .collect(),
            bias: (0..units).map(|j| j as f32 * 0.01).collect(),
        }
    }

    /// The parallel products are the serial ones bit for bit, with any number of
    /// threads (review F5a, finding 2).
    #[test]
    fn products_do_not_depend_on_threads() {
        let (n, k, m) = (70, 130, 90);
        let a = dense(n, k, 1).kernel;
        let b = dense(k, m, 2).kernel;
        let d = dense(n, m, 3).kernel;
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap_or_else(|e| panic!("{e}"));
            pool.install(|| {
                (
                    matmul(&a, n, k, &b, m),
                    matmul_at_b(&a, n, k, &d, m),
                    matmul_a_bt(&d, n, m, &b, k),
                )
            })
        };
        let one = run(1);
        for t in [2, 3, 8] {
            assert_eq!(run(t), one, "{t} threads");
        }
        // And they are the plain definitions.
        let mut naive = vec![0.0_f32; n * k];
        for i in 0..n {
            for kk in 0..k {
                let mut s = 0.0_f64;
                for j in 0..m {
                    s += f64::from(d[i * m + j]) * f64::from(b[kk * m + j]);
                }
                naive[i * k + kk] = s as f32;
            }
        }
        assert_eq!(one.2, naive);
        let mut naive = vec![0.0_f64; k * m];
        for i in 0..n {
            for kk in 0..k {
                for j in 0..m {
                    naive[kk * m + j] += f64::from(a[i * k + kk]) * f64::from(d[i * m + j]);
                }
            }
        }
        let naive: Vec<f32> = naive.into_iter().map(|v| v as f32).collect();
        assert_eq!(one.1, naive);
    }

    /// Review F5a (medium, finding 4): malformed networks and inputs are errors.
    #[test]
    fn shapes_are_checked() {
        let mlp = Mlp::new(
            vec![dense(3, 4, 1), dense(4, 2, 2)],
            Activation::Relu,
            Activation::Softmax,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(mlp.forward(&[0.0; 6], 2).is_ok());
        assert!(matches!(
            mlp.forward(&[0.0; 8], 2),
            Err(ModelError::FeatureMismatch {
                expected: 3,
                found: 4
            })
        ));
        assert!(mlp.forward(&[0.0; 7], 2).is_err());
        assert!(mlp.hidden_activations(&[0.0; 5], 2).is_err());
        assert!(mlp
            .gradients(
                &[0.0; 6],
                &[0.0; 3],
                2,
                LossKind::CategoricalCrossentropy,
                None
            )
            .is_err());
        let masks = DropoutMasks {
            rate: 0.5,
            masks: vec![vec![1.0; 3]],
        };
        assert!(mlp
            .gradients(
                &[0.0; 6],
                &[0.0; 4],
                2,
                LossKind::CategoricalCrossentropy,
                Some(&masks)
            )
            .is_err());
        let empty = Mlp {
            layers: vec![],
            hidden: Activation::Relu,
            output: Activation::Linear,
        };
        assert!(empty.forward(&[], 0).is_err());
        assert!(Mlp::new(
            vec![dense(3, 4, 1), dense(5, 2, 2)],
            Activation::Relu,
            Activation::Linear
        )
        .is_err());
        let mut bad = dense(3, 4, 1);
        bad.bias.pop();
        assert!(Mlp::new(vec![bad], Activation::Relu, Activation::Linear).is_err());
    }

    /// Review F5a (low, finding 12): deserialising validates the network.
    #[test]
    fn deserialisation_is_validated() {
        let mlp = Mlp::new(
            vec![dense(2, 3, 1), dense(3, 1, 2)],
            Activation::Tanh,
            Activation::Sigmoid,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let text = serde_json::to_string(&mlp).unwrap_or_default();
        let back: Mlp = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back, mlp);
        let broken = text.replacen("\"inputs\":3", "\"inputs\":4", 1);
        assert_ne!(broken, text);
        assert!(serde_json::from_str::<Mlp>(&broken).is_err());
    }
}
