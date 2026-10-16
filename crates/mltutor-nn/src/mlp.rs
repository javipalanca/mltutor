//! The multilayer perceptron: dense layers, activations, forward pass and manual
//! backpropagation in float32 (`tools/nn_reference.py::forward` / `gradients`).
//!
//! Matrices are row-major `Vec<f32>`. Every product accumulates in float64 in a
//! fixed order (inner index ascending) and rounds once to float32, so a result never
//! depends on the number of threads or the machine: training is deterministic.

use mltutor_core::nn_data::{Activation, LossKind};
use mltutor_core::nn_stats::DenseView;
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mlp {
    pub layers: Vec<Dense>,
    pub hidden: Activation,
    pub output: Activation,
}

/// `c (n×m) = a (n×k) · b (k×m)`. Products of float32 values are accumulated in
/// float64 and rounded once: the error is half an ulp, so the result is as close as
/// possible to Keras' float32 kernels whatever order those use.
pub(crate) fn matmul(a: &[f32], n: usize, k: usize, b: &[f32], m: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; n * m];
    let mut acc = vec![0.0_f64; m];
    for i in 0..n {
        acc.iter_mut().for_each(|v| *v = 0.0);
        for (kk, &aik) in a[i * k..(i + 1) * k].iter().enumerate() {
            let aik = f64::from(aik);
            let brow = &b[kk * m..(kk + 1) * m];
            for (cj, &bj) in acc.iter_mut().zip(brow) {
                *cj += aik * f64::from(bj);
            }
        }
        for (c, v) in c[i * m..(i + 1) * m].iter_mut().zip(&acc) {
            *c = *v as f32;
        }
    }
    c
}

/// `aᵀ (k×n) · d (n×m)` → `k×m` (kernel gradient), accumulated in float64.
fn matmul_at_b(a: &[f32], n: usize, k: usize, d: &[f32], m: usize) -> Vec<f32> {
    let mut g = vec![0.0_f64; k * m];
    for i in 0..n {
        let drow = &d[i * m..(i + 1) * m];
        for (kk, &aik) in a[i * k..(i + 1) * k].iter().enumerate() {
            let aik = f64::from(aik);
            let grow = &mut g[kk * m..(kk + 1) * m];
            for (gj, &dj) in grow.iter_mut().zip(drow) {
                *gj += aik * f64::from(dj);
            }
        }
    }
    g.into_iter().map(|v| v as f32).collect()
}

/// `d (n×m) · wᵀ (m×k)` → `n×k` (gradient with respect to the layer input),
/// accumulated in float64.
fn matmul_a_bt(d: &[f32], n: usize, m: usize, w: &[f32], k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * k];
    for i in 0..n {
        let drow = &d[i * m..(i + 1) * m];
        for kk in 0..k {
            let wrow = &w[kk * m..(kk + 1) * m];
            let mut s = 0.0_f64;
            for (dj, wj) in drow.iter().zip(wrow) {
                s += f64::from(*dj) * f64::from(*wj);
            }
            out[i * k + kk] = s as f32;
        }
    }
    out
}

/// Applies `act` to a row-major `n × units` matrix in place.
pub(crate) fn activate(z: &mut [f32], units: usize, act: Activation) {
    match act {
        Activation::Relu => z.iter_mut().for_each(|v| *v = v.max(0.0)),
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
    pub fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut a = x.to_vec();
        let last = self.layers.len().saturating_sub(1);
        for (i, l) in self.layers.iter().enumerate() {
            let mut z = matmul(&a, n, l.inputs, &l.kernel, l.units);
            for row in z.chunks_mut(l.units.max(1)) {
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
        a
    }

    /// Outputs of every hidden layer (after the activation, before dropout; C26):
    /// one `n × units` row-major matrix per hidden layer.
    pub fn hidden_activations(&self, x: &[f32], n: usize) -> Vec<Vec<f32>> {
        self.forward_cache(x, n, None).pre
    }

    /// Mean loss of a batch, the gradient of every layer `(kernel, bias)` and the
    /// batch output (the predictions before the step, for the metric)
    /// (`nn_reference.gradients`): cross-entropies from the logits, MSE through the
    /// output activation (softmax included).
    pub fn gradients(
        &self,
        x: &[f32],
        y: &[f32],
        n: usize,
        loss: LossKind,
        masks: Option<&DropoutMasks>,
    ) -> (f32, Gradients, Vec<f32>) {
        let f = self.forward_cache(x, n, masks);
        let lsz = self.layers.len();
        let k = self.n_outputs();
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
            for row in dz.chunks(l.units.max(1)) {
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
        (loss_value, grads, out)
    }
}
