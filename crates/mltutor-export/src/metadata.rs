//! 📋 Metadatos of the 🧠 network (D75): Python's `neural_network_metadata.json`
//! (`algorithms/export.py::show_neural_network_export`), corrected and completed.
//!
//! Same four blocks and the same keys as Python (`model_info`, `training_info`,
//! `data_info`, `export_info`), so that a student who knows the old file finds
//! everything where it was, plus a `config` block with what Python did not save.
//!
//! Corrections (fase5-redes.md §12.16, D75):
//! - `data_info.num_classes` is the number of classes (2 in binary); Python wrote
//!   `output_size`, i.e. 1.
//! - `export_info.export_date` is replaced by `training_date`, the date of the
//!   **training** (the caller passes it: the core has no clock); Python wrote the
//!   current time.
//! - `export_info.framework` says who trained it (MLTutor's own float32 MLP with
//!   Keras 3 semantics, D68), not "TensorFlow/Keras"; `version` is MLTutor's.
//! - `data_info.target_column` is the real column name; Python always wrote
//!   `"target"` (the name of its internal DataFrame column).
//! - `training_info` also has the MAE in regression (Python only kept the
//!   accuracy), the test loss and metric, the best and the stopping epoch and the
//!   divergence epoch (K1).
//!
//! Deterministic: no clock, no randomness, the key order is fixed and non-finite
//! numbers (a diverged training) are written as `null`. A diverged network can
//! still be described (`model_info.valid = false`): this file is not a model.

use mltutor_core::codegen::f32_as_decimal;
use mltutor_core::dataset::{Dataset, Target, TaskType};
use mltutor_core::model::Predictor;
use mltutor_core::nn_data::{NnArchitecture, NnTraining};
use mltutor_nn::TrainedNetwork;
use serde_json::{json, Value};

use crate::ExportError;

/// Suggested file name (C28).
pub const METADATA_FILE_NAME: &str = "red_neuronal_metadatos.json";
/// `export_info.format`.
pub const METADATA_FORMAT: &str = "mltutor-red-neuronal-metadatos/1";

/// A finite number, or `null`.
fn num(v: f64) -> Value {
    if v.is_finite() {
        json!(v)
    } else {
        Value::Null
    }
}

/// The metadata of a trained network. `dataset` is the one it was trained on
/// (same columns); `arch` and `training` the configuration of 🏗️ and ⚙️ used;
/// `trained_at` the local date and time of the training (`NnSummary.trained_at`).
///
/// Errors: `InvalidModel` when the dataset does not have the network's columns.
pub fn nn_metadata(
    trained: &TrainedNetwork,
    dataset: &Dataset,
    arch: &NnArchitecture,
    training: &NnTraining,
    trained_at: &str,
) -> Result<Value, ExportError> {
    let net = &trained.network;
    let mlp = net.mlp();
    if dataset.n_features() != mlp.n_inputs() {
        return Err(ExportError::InvalidModel {
            reason: format!(
                "la red espera {} características y el dataset tiene {}",
                mlp.n_inputs(),
                dataset.n_features()
            ),
        });
    }
    let classification = net.task() == TaskType::Classification;
    let resolved = net.resolved();
    let history = &trained.history;
    let last = history.epochs.last();
    let metric_key = history.metric.key();
    let mut training_info = serde_json::Map::new();
    training_info.insert(
        "final_loss".into(),
        last.map_or(Value::Null, |e| num(e.loss)),
    );
    training_info.insert(
        "final_val_loss".into(),
        last.map_or(Value::Null, |e| num(e.val_loss)),
    );
    training_info.insert("epochs_trained".into(), json!(history.epochs.len()));
    training_info.insert(
        format!("final_{metric_key}"),
        last.map_or(Value::Null, |e| num(e.metric)),
    );
    training_info.insert(
        format!("final_val_{metric_key}"),
        last.map_or(Value::Null, |e| num(e.val_metric)),
    );
    training_info.insert("test_loss".into(), num(trained.test_loss));
    training_info.insert(format!("test_{metric_key}"), num(trained.test_metric));
    // 1-based, as the UI shows the epochs.
    training_info.insert(
        "best_epoch".into(),
        json!(history.best_epoch.map(|e| e + 1)),
    );
    training_info.insert(
        "stopped_early".into(),
        json!(history.stopped_epoch.is_some_and(|e| e > 0)),
    );
    training_info.insert(
        "diverged_epoch".into(),
        json!(history.diverged_epoch.map(|e| e + 1)),
    );
    training_info.insert(
        "final_learning_rate".into(),
        num(f32_as_decimal(trained.final_learning_rate)),
    );

    let (class_names, num_classes): (Option<Vec<String>>, Option<usize>) = match &dataset.target {
        Target::Classes { class_names, .. } if classification => {
            let codes = net.class_codes();
            let names = codes
                .iter()
                .map(|&c| {
                    class_names
                        .get(c)
                        .cloned()
                        .unwrap_or_else(|| format!("Clase {c}"))
                })
                .collect();
            (Some(names), Some(codes.len()))
        }
        _ => (None, None),
    };
    let scaler = net.scaler();
    Ok(json!({
        "model_info": {
            "type": "Neural Network",
            "task_type": if classification { "Clasificación" } else { "Regresión" },
            "architecture": mlp.architecture(),
            "total_parameters": mlp.n_params(),
            "activation_function": mlp.hidden.as_str(),
            "output_activation": mlp.output.as_str(),
            "optimizer": arch.optimizer.as_str(),
            "dropout_rate": f32_as_decimal(arch.dropout),
            "batch_size": arch.batch_size,
            "loss": resolved.loss.as_str(),
            "metric": metric_key,
            "valid": net.is_valid(),
        },
        "training_info": Value::Object(training_info),
        "data_info": {
            "dataset": dataset.label,
            "feature_names": dataset.feature_display,
            "feature_columns": dataset.feature_columns,
            "target_column": dataset.target_column,
            "num_features": mlp.n_inputs(),
            // 1 in regression, as Python.
            "num_classes": num_classes.unwrap_or(1),
            "class_names": class_names,
            "n_train": trained.sizes.train,
            "n_val": trained.sizes.val,
            "n_test": trained.sizes.test,
            "scaler": {
                "fit_on": "todas las filas (antes de dividir, como la app; D72)",
                "mean": scaler.mean.iter().map(|&v| num(v)).collect::<Vec<_>>(),
                "scale": scaler.scale.iter().map(|&v| num(v)).collect::<Vec<_>>(),
                "n_samples": scaler.n_samples,
            },
        },
        "config": {
            "learning_rate": f32_as_decimal(training.learning_rate),
            "epochs": training.epochs,
            "validation_split": training.validation_split(),
            "early_stopping": training.patience.is_some(),
            "patience": training.patience,
            "reduce_lr": training.lr_factor.is_some(),
            "lr_factor": training.lr_factor.map(f32_as_decimal),
            "reduce_lr_patience": training.lr_factor.map(|_| training.reduce_lr_patience()),
            "seed": training.seed,
            "test_size": mltutor_core::nn_data::LESSON_TEST_SIZE,
            "stratified": classification,
        },
        "export_info": {
            "format": METADATA_FORMAT,
            "training_date": trained_at,
            "version": crate::VERSION,
            "framework": "MLTutor (red neuronal propia en float32 con la semántica de Keras 3)",
        },
    }))
}

/// The file's text: `json.dumps(indent=2, ensure_ascii=False)` style, with a
/// final newline.
pub fn metadata_json_text(doc: &Value) -> Result<String, ExportError> {
    let mut s = serde_json::to_string_pretty(doc).map_err(|e| ExportError::Json {
        message: e.to_string(),
    })?;
    s.push('\n');
    Ok(s)
}
