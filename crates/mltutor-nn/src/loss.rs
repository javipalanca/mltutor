//! Losses and metrics as Keras 3.12 computes them (`nn_reference.per_sample_loss`,
//! `metric_values`). Keras recognises a sigmoid/softmax output and computes the
//! cross-entropy **from the logits**, without clipping with ε.

use mltutor_core::nn_data::{Activation, LossKind, MetricKind};

/// Keras `backend.epsilon()`.
const EPS: f32 = 1e-7;

/// Mean of float32 values (`.mean(dtype=float32)`), accumulated in float64 and
/// rounded once.
pub(crate) fn mean_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let s: f64 = v.iter().map(|&x| f64::from(x)).sum();
    (s / v.len() as f64) as f32
}

/// Loss of every row of a batch (`n × k` logits `z`, outputs `p`, targets `y`).
pub fn per_sample_loss(
    loss: LossKind,
    output: Activation,
    z: &[f32],
    p: &[f32],
    y: &[f32],
    k: usize,
) -> Vec<f32> {
    let k = k.max(1);
    let rows = z.len() / k;
    (0..rows)
        .map(|i| {
            let r = i * k..(i + 1) * k;
            let (z, p, y) = (&z[r.clone()], &p[r.clone()], &y[r]);
            match loss {
                LossKind::BinaryCrossentropy if output == Activation::Sigmoid => {
                    // tf.nn.sigmoid_cross_entropy_with_logits
                    let s: f32 = z
                        .iter()
                        .zip(y)
                        .map(|(&z, &y)| z.max(0.0) - z * y + (-z.abs()).exp().ln_1p())
                        .sum();
                    s / k as f32
                }
                LossKind::BinaryCrossentropy => {
                    let s: f32 = p
                        .iter()
                        .zip(y)
                        .map(|(&p, &y)| {
                            let p = p.clamp(EPS, 1.0 - EPS);
                            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                        })
                        .sum();
                    s / k as f32
                }
                LossKind::CategoricalCrossentropy if output == Activation::Softmax => {
                    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let lse = z.iter().map(|v| (v - m).exp()).sum::<f32>().ln() + m;
                    -z.iter().zip(y).map(|(z, y)| y * (z - lse)).sum::<f32>()
                }
                LossKind::CategoricalCrossentropy => {
                    let total: f32 = p.iter().sum();
                    -p.iter()
                        .zip(y)
                        .map(|(p, y)| y * (p / total).clamp(EPS, 1.0 - EPS).ln())
                        .sum::<f32>()
                }
                LossKind::Mse => {
                    p.iter().zip(y).map(|(p, y)| (p - y) * (p - y)).sum::<f32>() / k as f32
                }
            }
        })
        .collect()
}

/// Metric of every row (accuracy as 0/1, or the row's MAE).
pub fn metric_values(metric: MetricKind, p: &[f32], y: &[f32], k: usize) -> Vec<f32> {
    let k = k.max(1);
    let rows = p.len() / k;
    (0..rows)
        .map(|i| {
            let r = i * k..(i + 1) * k;
            let (p, y) = (&p[r.clone()], &y[r]);
            match metric {
                MetricKind::BinaryAccuracy => {
                    let hits = p
                        .iter()
                        .zip(y)
                        .filter(|(p, y)| (if **p > 0.5 { 1.0 } else { 0.0 }) == **y)
                        .count();
                    hits as f32 / k as f32
                }
                MetricKind::CategoricalAccuracy => {
                    if argmax(p) == argmax(y) {
                        1.0
                    } else {
                        0.0
                    }
                }
                MetricKind::Mae => {
                    p.iter().zip(y).map(|(p, y)| (p - y).abs()).sum::<f32>() / k as f32
                }
            }
        })
        .collect()
}

/// First index of the maximum (NumPy `argmax`; NaN counts as the maximum).
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x.is_nan() {
            return i;
        }
        if x > v[best] {
            best = i;
        }
    }
    best
}
