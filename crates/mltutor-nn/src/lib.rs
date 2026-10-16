//! Neural networks for MLTutor (🧠 Redes Neuronales, fase5-redes.md C21).
//!
//! A float32 multilayer perceptron with manual backpropagation that follows
//! `tools/nn_reference.py` (the executable specification of Keras 3.12's `fit`)
//! line by line. No burn (D68, decided in F5a: see fase5-redes.md §8): the
//! optimizers, the losses from the logits, the initialisation and the callbacks had
//! to be our own anyway, and a 400-line MLP is deterministic without a global RNG
//! mutex, adds nothing to the binary and trains Iris (50 epochs) in about a
//! millisecond.
//!
//! # Public API (stable; used by `src-tauri/src/real_nn.rs`)
//!
//! | Use | Function / type |
//! |---|---|
//! | `train_model` (`ModelConfig::NeuralNet`) | [`train_network`]`(x: ArrayView2<f64> /* original units */, y: &Target, &NnArchitecture, &NnTraining, shuffle: true, init: None, &FitControl { on_epoch, cancel }) -> Result<TrainedNetwork, NnError>`; per-epoch [`EpochLog`] through `on_epoch` (after the callbacks), cancellation checked before every batch ([`NnError::Cancelled`]). Configuration types and C22 in [`mltutor_core::nn_data`] |
//! | `TrainedModelInfo.nn` | [`TrainedNetwork`]`{ network, history: History { epochs, best_epoch, stopped_epoch }, split, sizes, test_loss, test_metric, final_learning_rate, layers }` |
//! | every model view (`evaluate_model`, `predict_one`, `decision_grid`…) | [`NeuralNetwork`] is a [`mltutor_core::model::Predictor`] on rows in original units (scales inside); `predict_proba` = output (`[1 − p, p]` with one neuron); [`NeuralNetwork::raw_output`] |
//! | `nn_diagnostics`, `nn_history` | [`mltutor_core::nn_diagnostics::diagnostics_view`]`(&history.series(), task)`, [`mltutor_core::nn_diagnostics::history_cards`]; `History::lr_drops()` |
//! | `nn_weights` | [`mltutor_core::nn_stats::weight_stats`]`(&mlp.views())` + the raw `kernel`/`bias` of [`Mlp::layers`] for the Plotly histograms |
//! | `nn_activations` | [`NeuralNetwork::hidden_activations`]`(first sample_size rows of trained.x_test_scaled(x), n)` → [`mltutor_core::nn_stats::activation_stats`] per layer, `dead_pct`, `network_map(&mlp.views(), &stats)` |
//! | `decision_grid` (D78) | [`mltutor_core::grid::mean_grid`] |
//! | low level (parity tests) | [`Mlp::forward`], [`Mlp::gradients`], [`Optimizer::step`], [`fit`], [`evaluate`], [`glorot_init`] |

pub mod loss;
pub mod mlp;
pub mod network;
pub mod optim;
pub mod train;

pub use mlp::{Dense, DropoutMasks, Gradients, Mlp};
pub use mltutor_core::nn_data::{
    Activation, EpochLog, History, LossKind, MetricKind, NnArchitecture, NnError, NnTraining,
    OptimizerKind,
};
pub use network::{train_network, validate, NeuralNetwork, TrainedNetwork};
pub use optim::Optimizer;
pub use train::{evaluate, fit, glorot_init, FitControl, FitData, FitResult, FitSpec};

/// Version of the MLTutor workspace this crate was built from.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
