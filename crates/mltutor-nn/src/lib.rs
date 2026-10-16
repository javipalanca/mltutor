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
//! | `train_model` (`ModelConfig::NeuralNet`) | [`train_network`]`(x: ArrayView2<f64> /* original units */, y: &Target, &NnArchitecture, &NnTraining, TrainOptions::default() /* K5: { shuffle, init } */, &FitControl { on_epoch, cancel }) -> Result<TrainedNetwork, NnError>`; per-epoch [`EpochLog`] through `on_epoch` (after the callbacks); cancellation checked before every training batch, every validation batch and in the final `evaluate` ([`NnError::Cancelled`]). K4: `FitControl::from(&TrainControl)` and `TrainControl::from(&FitControl)`. Configuration types and C22 in [`mltutor_core::nn_data`] |
//! | `TrainedModelInfo.nn` | [`TrainedNetwork`]`{ network, history: History { epochs, best_epoch, stopped_epoch, diverged_epoch /* K1 */ }, split, sizes, test_loss, test_metric, final_learning_rate, layers }`; a diverged history (NaN) serialises its non-finite values as `"nan"`/`"inf"`/`"-inf"` |
//! | divergence (K1) | `history.diverged_epoch: Option<usize>` (first epoch with a NaN/∞ batch or validation loss; training continues as in Keras) and [`NeuralNetwork::is_valid`]`()` (finite weights and scaler); an invalid network returns [`mltutor_core::model::ModelError::Numerical`] from every prediction |
//! | cost warning (K6) | [`mltutor_core::nn_data::training_cost`]`(total_params, n_fit, epochs) -> TrainingCost { work, estimated_seconds, level: CostLevel::{Low, High, VeryHigh} }` |
//! | every model view (`evaluate_model`, `predict_one`, `decision_grid`…) | [`NeuralNetwork`] is a [`mltutor_core::model::Predictor`] on rows in original units (scales inside); `predict_proba` = output (`[1 − p, p]` with one neuron); [`NeuralNetwork::output_is_probability`] is `false` with a tanh output (hide or warn in the probability views); [`NeuralNetwork::raw_output`]; accessors `mlp()`, `scaler()`, `class_codes()`, `resolved()` |
//! | `nn_diagnostics`, `nn_history` | [`mltutor_core::nn_diagnostics::diagnostics_view`]`(&history.series(), task, n_val)`, [`mltutor_core::nn_diagnostics::history_cards`]; `History::lr_drops()` |
//! | `nn_weights` (K2) | [`NeuralNetwork::weights_view`]`(weight_bins /* 50 */, bias_bins /* 20 */) -> WeightsView { stats: WeightStats, histograms: Vec<LayerHistograms { weights, biases: Histogram { edges, counts } }> }` (Plotly's `nbinsx` bins, computed in Rust) |
//! | `nn_activations` (K2, K3) | [`TrainedNetwork::test_activations`]`(x_all, &ActivationParams { sample_size, dead_threshold, constant_std, bins })` or [`NeuralNetwork::activations_view`]`(x_test, &params) -> ActivationsView { rows_used, layers: Vec<LayerActivationsView { stats /* dead, constant, neuron_means, neuron_stds… */, histogram }>, dead_pct, map }`; raw values: [`NeuralNetwork::hidden_activations`]`(x_test, sample_size) -> Activations { rows_used, layers }` |
//! | `decision_grid` (D78) | [`NeuralNetwork::mean_grid`]`(x_test, &y_test, fx, fy, &TrainControl) -> MeanGrid` (classification and regression) |
//! | low level (parity tests) | [`Mlp::forward`], [`Mlp::gradients`], [`Mlp::hidden_activations`] (scaled rows, `Result`), [`Optimizer::step`] (`Result`), [`fit`], [`evaluate`], [`glorot_init`] |
//!
//! ```
//! use mltutor_core::dataset::{load_builtin, DatasetId};
//! use mltutor_nn::{train_network, Activation, FitControl, NnArchitecture, NnTraining,
//!     OptimizerKind, TrainOptions};
//!
//! let iris = load_builtin(DatasetId::Iris).unwrap();
//! let arch = NnArchitecture {
//!     hidden: vec![10, 10],
//!     activation: Activation::Relu,
//!     output_activation: Activation::Softmax,
//!     dropout: 0.2,
//!     batch_size: 32,
//!     optimizer: OptimizerKind::Adam,
//! };
//! let training = NnTraining {
//!     learning_rate: 0.01,
//!     epochs: 20,
//!     validation_pct: 20,
//!     patience: Some(10),
//!     lr_factor: Some(0.5),
//!     seed: 42,
//! };
//! let trained = train_network(
//!     iris.features.view(),
//!     &iris.target,
//!     &arch,
//!     &training,
//!     TrainOptions::default(),
//!     &FitControl::default(),
//! )
//! .unwrap();
//! assert!(trained.network.is_valid());
//! assert_eq!(trained.history.diverged_epoch, None);
//! let n_val = trained.sizes.val as usize;
//! let diagnostics = mltutor_core::nn_diagnostics::diagnostics_view(
//!     &trained.history.series(),
//!     mltutor_core::dataset::TaskType::Classification,
//!     Some(n_val),
//! )
//! .unwrap();
//! assert!(!diagnostics.blocks.is_empty());
//! ```

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
pub use network::{
    train_network, validate, Activations, NeuralNetwork, TrainOptions, TrainedNetwork,
};
pub use optim::Optimizer;
pub use train::{evaluate, fit, glorot_init, FitControl, FitData, FitResult, FitSpec};

/// Version of the MLTutor workspace this crate was built from.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
