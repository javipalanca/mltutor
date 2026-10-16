//! Adam, SGD and RMSprop with the update formulas of `keras/src/optimizers`
//! (Keras 3.12 defaults), in float32. They differ from burn's: Adam adds ε **after**
//! the bias correction of the step size, and RMSprop puts ε **inside** the square
//! root.

use mltutor_core::nn_data::OptimizerKind;
use serde::{Deserialize, Serialize};

use crate::mlp::Mlp;

const BETA_1: f32 = 0.9;
const BETA_2: f32 = 0.999;
const RHO: f32 = 0.9;
const EPSILON: f32 = 1e-7;

/// Optimizer state: the learning rate (ReduceLROnPlateau changes it), the step
/// counter and two slots per variable (Adam's m and v; RMSprop's average in the
/// second).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Optimizer {
    pub kind: OptimizerKind,
    pub learning_rate: f32,
    /// Steps applied (`optimizer.iterations`).
    pub iterations: u64,
    slots: Vec<[Vec<f32>; 2]>,
}

impl Optimizer {
    pub fn new(kind: OptimizerKind, learning_rate: f32) -> Self {
        Self {
            kind,
            learning_rate,
            iterations: 0,
            slots: Vec::new(),
        }
    }

    /// One step on every variable of `mlp` (kernel then bias of each layer, Keras'
    /// `trainable_weights` order) with the gradients of [`Mlp::gradients`].
    pub fn step(&mut self, mlp: &mut Mlp, grads: &[(Vec<f32>, Vec<f32>)]) {
        if self.slots.is_empty() {
            for l in &mlp.layers {
                self.slots
                    .push([vec![0.0; l.kernel.len()], vec![0.0; l.kernel.len()]]);
                self.slots
                    .push([vec![0.0; l.bias.len()], vec![0.0; l.bias.len()]]);
            }
        }
        let lr = self.learning_rate;
        let t = (self.iterations + 1) as f32;
        let alpha = lr * (1.0 - BETA_2.powf(t)).sqrt() / (1.0 - BETA_1.powf(t));
        let kind = self.kind;
        let mut slot = self.slots.iter_mut();
        for (layer, (gw, gb)) in mlp.layers.iter_mut().zip(grads) {
            for (vars, g) in [(&mut layer.kernel, gw), (&mut layer.bias, gb)] {
                let Some([m, s]) = slot.next() else { return };
                for (((w, g), m), s) in vars.iter_mut().zip(g).zip(m.iter_mut()).zip(s.iter_mut()) {
                    match kind {
                        OptimizerKind::Adam => {
                            *m += (g - *m) * (1.0 - BETA_1);
                            *s += (g * g - *s) * (1.0 - BETA_2);
                            *w -= (*m * alpha) / (s.sqrt() + EPSILON);
                        }
                        OptimizerKind::Rmsprop => {
                            *s = RHO * *s + (1.0 - RHO) * (g * g);
                            *w -= (lr * g) / (*s + EPSILON).sqrt();
                        }
                        OptimizerKind::Sgd => *w -= g * lr,
                    }
                }
            }
        }
        self.iterations += 1;
    }
}
