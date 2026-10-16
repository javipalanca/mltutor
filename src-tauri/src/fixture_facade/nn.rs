//! 🧠 training of the [`super::FixtureFacade`]: replays the Keras history of
//! `nn/exact/<case>.json` epoch by epoch through the observer, as the real engine
//! reports it (phases, then one event per epoch, cancellation between epochs).
//!
//! The case is the one whose controls match the request (architecture, activations,
//! batch, optimizer, LR, validation and callbacks; the fixtures were trained without
//! dropout, which is ignored). Otherwise the `<dataset>_default` case is replayed and
//! truncated to the requested epochs. Only for tests and development.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;

use crate::core_facade::{check_cancelled, ModelHandle, TrainEvent, TrainObserver, TrainSpec};
use crate::ipc::{CommandError, DatasetId, DatasetSummary, EpochLog, TrainPhase};
use crate::nn::finite_log;
use crate::nn_model::{prepare, NnModel, Outcome, Prepared};

fn unavailable(reason: impl std::fmt::Display) -> CommandError {
    CommandError::training(format!("FixtureFacade: {reason}"))
}

fn read(path: &Path) -> Result<Value, CommandError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| unavailable(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| CommandError::internal(format!("{}: {e}", path.display())))
}

/// Whether the controls of `case` are the ones of the request.
fn matches(input: &Value, p: &Prepared) -> bool {
    let config = &input["nn_config"];
    let controls = &input["controls"];
    let layers: Vec<u64> = p.layers.iter().map(|&n| u64::from(n)).collect();
    let close = |v: &Value, x: f32| v.as_f64().is_some_and(|v| (v - f64::from(x)).abs() < 1e-6);
    config["architecture"] == serde_json::json!(layers)
        && config["activation"] == p.arch.activation.keras_name()
        && config["output_activation"] == p.arch.output_activation.keras_name()
        && config["batch_size"].as_u64() == Some(u64::from(p.arch.batch_size))
        && config["optimizer"] == p.arch.optimizer.keras_name()
        && close(&controls["lr"], p.training.learning_rate)
        && controls["val"].as_u64() == Some(u64::from(p.training.validation_pct))
        && controls["es"].as_bool() == Some(p.training.patience.is_some())
        && p.training
            .patience
            .is_none_or(|pat| controls["patience"].as_u64() == Some(u64::from(pat)))
        && controls["rlr"].as_bool() == Some(p.training.lr_factor.is_some())
        && p.training
            .lr_factor
            .is_none_or(|f| close(&controls["factor"], f))
}

/// The fixture case to replay for `id` and the request.
fn case_for(root: &Path, id: DatasetId, p: &Prepared) -> Result<Value, CommandError> {
    let dir = root.join("nn/exact");
    let slug = id.slug();
    let wanted = format!("datasets/{slug}.json");
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .map_err(|e| unavailable(format!("{}: {e}", dir.display())))?
        .filter_map(|entry| {
            entry
                .ok()
                .map(|e| e.file_name().to_string_lossy().into_owned())
        })
        .filter(|name| {
            name.starts_with(slug) && name.ends_with(".json") && !name.contains("dropout_masks")
        })
        .collect();
    names.sort();
    let mut default = None;
    for name in names {
        let case = read(&dir.join(&name))?;
        if case["input"]["dataset"] != wanted.as_str() {
            continue;
        }
        if matches(&case["input"], p) {
            return Ok(case);
        }
        if name.starts_with(&format!("{slug}_default")) {
            default = Some(case);
        }
    }
    default.ok_or_else(|| unavailable(format!("no hay ninguna red de {slug} en nn/exact")))
}

/// The epochs of the Keras history (`accuracy|mae`, `loss`, `val_*`, `learning_rate`).
fn history(case: &Value) -> Vec<EpochLog> {
    let h = &case["expected"]["history"];
    let column = |key: &str| -> Vec<Option<f64>> {
        h[key]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_f64().and_then(finite_log)).collect())
            .unwrap_or_default()
    };
    let loss = column("loss");
    let (metric, val_metric) = if h.get("accuracy").is_some() {
        (column("accuracy"), column("val_accuracy"))
    } else {
        (column("mae"), column("val_mae"))
    };
    let val_loss = column("val_loss");
    let lr = column("learning_rate");
    (0..loss.len())
        .map(|i| EpochLog {
            epoch: u16::try_from(i + 1).unwrap_or(u16::MAX),
            loss: loss[i],
            metric: metric.get(i).copied().flatten(),
            val_loss: val_loss.get(i).copied().flatten(),
            val_metric: val_metric.get(i).copied().flatten(),
            learning_rate: lr.get(i).copied().flatten(),
        })
        .collect()
}

/// Pause between replayed epochs (`MLTUTOR_FIXTURE_EPOCH_MS`, 0 by default): lets a
/// developer watch the live chart and try "Cancelar" with the fixture facade.
fn epoch_pause() -> std::time::Duration {
    std::env::var("MLTUTOR_FIXTURE_EPOCH_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(std::time::Duration::from_millis)
        .unwrap_or_default()
}

pub fn train(
    root: &Path,
    id: DatasetId,
    summary: &DatasetSummary,
    class_counts: Option<Vec<(String, u32)>>,
    spec: &TrainSpec,
    observer: &TrainObserver<'_>,
    cancel: &AtomicBool,
) -> Result<ModelHandle, CommandError> {
    let started = Instant::now();
    let prepared = prepare(summary, spec, class_counts.as_deref())?;
    observer(TrainEvent::Phase(TrainPhase::Preparing));
    let case = case_for(root, id, &prepared)?;
    check_cancelled(cancel)?;
    observer(TrainEvent::Phase(TrainPhase::Building));
    observer(TrainEvent::Phase(TrainPhase::Compiling));
    observer(TrainEvent::Phase(TrainPhase::Training));
    let mut epochs = history(&case);
    epochs.truncate(usize::from(prepared.training.epochs));
    let pause = epoch_pause();
    for log in &epochs {
        check_cancelled(cancel)?;
        if !pause.is_zero() {
            std::thread::sleep(pause);
        }
        observer(TrainEvent::Epoch(log));
    }
    check_cancelled(cancel)?;
    let es = &case["expected"]["callbacks"]["early_stopping"];
    let best_epoch = prepared
        .training
        .patience
        .and(es["best_epoch"].as_u64())
        .and_then(|b| u16::try_from(b + 1).ok())
        .filter(|&b| usize::from(b) <= epochs.len());
    Ok(Arc::new(NnModel::new(
        summary,
        &prepared,
        Outcome {
            epochs,
            best_epoch,
            diverged_epoch: None,
            network_valid: true,
            near_divergence: false,
            started,
        },
        None,
    )))
}
