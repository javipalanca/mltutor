//! 🔍 K-Nearest Neighbors (fase4-knn.md §5): the texts the backend sends, the D59 rule,
//! the palette and limits of the interactive canvas (C18) and the context of the training
//! script, shared by both facades. Every computation lives in `mltutor-core::knn`.

use std::sync::OnceLock;

use serde::Serialize;

use crate::ipc::{CommandError, KnnMetric, KnnWeights, ModelConfig, SeparationVerdict, KNN_MAX_K};

/// `content/knn/mensajes.toml` (the UI reads the same file).
const KNN_MESSAGES: &str = include_str!("../../content/knn/mensajes.toml");

/// A text of `content/knn/mensajes.toml` (`fallback` if the file lacks it, logged).
pub fn knn_text(section: &str, key: &str, fallback: &str) -> String {
    static MESSAGES: OnceLock<Option<mltutor_core::content::Messages>> = OnceLock::new();
    let messages = MESSAGES.get_or_init(|| {
        mltutor_core::content::Messages::parse("content/knn/mensajes.toml", KNN_MESSAGES)
            .map_err(|error| log::error!("contenido: {error}"))
            .ok()
    });
    match messages.as_ref().map(|m| m.text(section, key)) {
        Some(Ok(text)) => text.to_owned(),
        Some(Err(error)) => {
            log::error!("contenido: {error}");
            fallback.to_owned()
        }
        None => fallback.to_owned(),
    }
}

/// D59: `[entrenamiento] error_k_mayor` with `{k}` and `{n}`.
pub fn too_few_samples(k: usize, n_train: usize) -> CommandError {
    let message = knn_text(
        "entrenamiento",
        "error_k_mayor",
        "K = {k} es mayor que el número de muestras de entrenamiento ({n}). Elige un K de {n} o menos.",
    )
    .replace("{k}", &k.to_string())
    .replace("{n}", &n_train.to_string());
    CommandError::too_few_samples(k, n_train, message)
}

/// D59: Python trained with K > n_train and every view failed in English; Rust refuses.
pub fn check_k(config: ModelConfig, n_train: usize) -> Result<(), CommandError> {
    if let ModelConfig::Knn { n_neighbors, .. } = config {
        let k = usize::from(n_neighbors);
        if k > n_train {
            return Err(too_few_samples(k, n_train));
        }
    }
    Ok(())
}

/// Range of "Vecinos (K)" and of the K of the canvas (D55).
pub fn validate_k(k: u8) -> Result<(), CommandError> {
    if (1..=KNN_MAX_K).contains(&k) {
        Ok(())
    } else {
        Err(CommandError::invalid_config(format!(
            "El número de vecinos (K) debe estar entre 1 y {KNN_MAX_K}."
        )))
    }
}

/// `colors` of the Python canvas (`colors[code % 8]`, fixture `interactive.palette`).
pub const PALETTE: [&str; 8] = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA726", "#AB47BC", "#66BB6A", "#EF5350", "#42A5F5",
];

/// Colour of class `code` in the canvas.
pub fn class_color(code: usize) -> String {
    PALETTE[code % PALETTE.len()].to_owned()
}

/// Limits of the canvas: `min − 1`, `max + 1` of each feature (fixture `interactive.bounds`).
pub fn bounds(xs: &[f64], ys: &[f64]) -> [f64; 4] {
    let range = |values: &[f64]| {
        let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lo.is_finite() && hi.is_finite() {
            (lo - 1.0, hi + 1.0)
        } else {
            (-1.0, 1.0)
        }
    };
    let (x0, x1) = range(xs);
    let (y0, y1) = range(ys);
    [x0, x1, y0, y1]
}

/// The test points of one canvas request (more would only slow the drag down).
pub const MAX_PLAYGROUND_POINTS: usize = 200;

/// Validates the points of a canvas request (finite, at most [`MAX_PLAYGROUND_POINTS`]).
pub fn validate_points(points: &[[f64; 2]]) -> Result<(), CommandError> {
    if points.len() > MAX_PLAYGROUND_POINTS {
        return Err(CommandError::invalid_input(format!(
            "El lienzo admite como máximo {MAX_PLAYGROUND_POINTS} puntos de prueba."
        )));
    }
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(CommandError::invalid_input(
            "Las coordenadas del punto de prueba no son números válidos.",
        ));
    }
    Ok(())
}

/// Python's verdict (thresholds > 5 and > 2), with D67 for two classes.
pub fn verdict(n_classes: usize, ratio: f64) -> SeparationVerdict {
    match n_classes {
        0 | 1 => SeparationVerdict::TooFewClasses,
        2 => SeparationVerdict::Binary,
        _ if ratio > 5.0 => SeparationVerdict::Separated,
        _ if ratio > 2.0 => SeparationVerdict::Moderate,
        _ => SeparationVerdict::Close,
    }
}

/// Context of `knn_entrenamiento.py.j2` (every key is always sent, `None` as `null`;
/// fixture `code/knn_training.json` → `context`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KnnTrainingContext {
    pub dataset: String,
    pub dataset_label: String,
    pub csv_file: Option<String>,
    pub csv_target: Option<String>,
    pub csv_sep: Option<String>,
    pub csv_decimal: Option<String>,
    pub csv_encoding: Option<String>,
    /// `"classification"` or `"regression"`.
    pub task: String,
    pub n_neighbors: u32,
    pub weights: String,
    pub metric: String,
    pub test_size: f64,
    pub feature_names: Vec<String>,
    pub class_names: Option<Vec<String>>,
    pub sample: Option<Vec<f64>>,
}

impl KnnTrainingContext {
    pub fn render(&self) -> Result<String, CommandError> {
        mltutor_core::codegen::render_named("knn_entrenamiento.py.j2", self)
            .map_err(|error| CommandError::internal(format!("plantilla: {error}")))
    }
}

/// The hyperparameters of a KNN configuration (`None` for other models).
pub fn params(config: ModelConfig) -> Option<(u8, KnnWeights, KnnMetric)> {
    match config {
        ModelConfig::Knn {
            n_neighbors,
            weights,
            metric,
        } => Some((n_neighbors, weights, metric)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KNN: ModelConfig = ModelConfig::Knn {
        n_neighbors: 20,
        weights: KnnWeights::Uniform,
        metric: KnnMetric::Minkowski,
    };

    /// D59 with the text of the content (fixture `csv_string_target_k20_test05_error`).
    #[test]
    fn k_larger_than_the_train_set_is_an_error_in_spanish() {
        assert!(check_k(KNN, 20).is_ok());
        let Err(CommandError::TooFewSamples {
            k,
            n_train,
            message,
        }) = check_k(KNN, 15)
        else {
            panic!("K = 20 with 15 samples must fail");
        };
        assert_eq!((k, n_train), (20, 15));
        assert!(
            message.contains("K = 20") && message.contains("(15)"),
            "{message}"
        );
        assert!(!message.contains('{'), "{message}");
        assert!(check_k(ModelConfig::Linear, 0).is_ok());
    }

    #[test]
    fn k_range_and_points_are_validated() {
        assert!(validate_k(0).is_err() && validate_k(21).is_err());
        assert!(validate_k(1).is_ok() && validate_k(20).is_ok());
        assert!(validate_points(&[[1.0, 2.0]]).is_ok());
        assert!(validate_points(&[[f64::NAN, 2.0]]).is_err());
        assert!(validate_points(&vec![[0.0, 0.0]; MAX_PLAYGROUND_POINTS + 1]).is_err());
    }

    #[test]
    fn canvas_palette_and_bounds_follow_python() {
        assert_eq!(class_color(0), "#FF6B6B");
        assert_eq!(class_color(9), "#4ECDC4");
        // Iris sepal length/width: fixture `interactive.default.bounds`.
        let b = bounds(&[4.3, 7.9], &[2.0, 4.4]);
        let expected = [3.3, 8.9, 1.0, 5.4];
        for (a, e) in b.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "{b:?}");
        }
    }

    #[test]
    fn verdicts_follow_the_thresholds_and_d67() {
        assert_eq!(verdict(3, 5.01), SeparationVerdict::Separated);
        assert_eq!(verdict(3, 5.0), SeparationVerdict::Moderate);
        assert_eq!(verdict(3, 2.53), SeparationVerdict::Moderate);
        assert_eq!(verdict(3, 2.0), SeparationVerdict::Close);
        assert_eq!(verdict(2, 1.0), SeparationVerdict::Binary);
        assert_eq!(verdict(1, 0.0), SeparationVerdict::TooFewClasses);
    }
}
