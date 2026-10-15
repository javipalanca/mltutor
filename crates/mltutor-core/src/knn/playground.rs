//! C18: 🎮 Visualización Interactiva (fase4-knn.md §2.4.1, D54).
//!
//! Python embedded a 34 KB HTML page whose JavaScript classified the test points
//! with its own rules (stable sort, a point at distance 0 weighs 1 and the others keep
//! `1/d`). Here the points are classified by a real 2-feature KNN, with scikit-learn's
//! rules ([`super`]): the model is refitted on **every row** of the dataset (as the
//! 🌈 Frontera, D63) with the K, weights and metric of the playground's controls.
//!
//! The view gets the 2-D data, the colours of the Python page (`colors[code % 8]`),
//! the bounds of the canvas (`min − 1`, `max + 1`) and, for every test point, the
//! predicted class, its probabilities, the confidence (`max proba`), the K neighbours
//! in order and their mean distance.

use ndarray::{Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use super::{fit, KnnConfig, Neighbor, SearchEngine};
use crate::dataset::Target;
use crate::model::{ModelError, TrainControl};
use crate::stats::pairwise_sum;

/// The palette of the Python canvas (`colors[class_code % 8]`).
pub const PALETTE: [&str; 8] = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA726", "#AB47BC", "#66BB6A", "#EF5350", "#42A5F5",
];

/// What the playground asks for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaygroundRequest {
    /// The two columns (x, y).
    pub features: [usize; 2],
    /// K, weights and metric of the playground's controls.
    pub config: KnnConfig,
    /// Test points in data coordinates.
    pub points: Vec<[f64; 2]>,
}

/// One classified test point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaygroundPoint {
    pub point: [f64; 2],
    /// Predicted class code.
    pub class: usize,
    /// `max(proba)`.
    pub confidence: f64,
    /// Probabilities in [`Playground::classes`] order.
    pub proba: Vec<f64>,
    /// The K neighbours, nearest first (`index` = dataset row).
    pub neighbors: Vec<Neighbor>,
    /// Mean distance to them ("Distancia Promedio").
    pub mean_distance: f64,
    /// Tie at the K-th distance ([`super::KthTie`]).
    pub tie_at_kth: Option<super::KthTie>,
}

/// Everything the playground draws.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Playground {
    pub features: [usize; 2],
    /// Every row of the dataset on the two features (`n × 2`).
    pub data: Array2<f64>,
    /// Class code of each row.
    pub codes: Vec<usize>,
    /// Class codes present, sorted (`classes_` of the 2-D model).
    pub classes: Vec<usize>,
    /// Colour of each class of `classes` (`PALETTE[code % 8]`).
    pub colors: Vec<String>,
    /// `[x_min, x_max, y_min, y_max]` = min − 1 and max + 1 of each feature.
    pub bounds: [f64; 4],
    pub points: Vec<PlaygroundPoint>,
    /// Mean confidence of the points (`None` without points).
    pub mean_confidence: Option<f64>,
    /// Engine of the 2-D model.
    pub engine: SearchEngine,
    pub config: KnnConfig,
}

/// C18 on the whole dataset `x` (all rows, every feature) and its target.
pub fn playground(
    x: ArrayView2<'_, f64>,
    y: &Target,
    request: &PlaygroundRequest,
    control: &TrainControl<'_>,
) -> Result<Playground, ModelError> {
    let [fx, fy] = request.features;
    let p = x.ncols();
    if fx == fy {
        return Err(ModelError::InvalidSelection {
            reason: "elige dos características distintas".to_owned(),
        });
    }
    if fx >= p || fy >= p {
        return Err(ModelError::InvalidSelection {
            reason: format!("el dataset tiene {p} características"),
        });
    }
    let codes = y.class_codes().ok_or(ModelError::NotClassifier)?;
    if request.points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ModelError::InvalidInput {
            reason: "las coordenadas de los puntos deben ser números".to_owned(),
        });
    }
    let data = x.select(Axis(1), &[fx, fy]);
    let model = fit(request.config, data.view(), y, control)?;
    let classes = model.classes.clone().unwrap_or_default();
    let bound = |c: usize| {
        data.column(c)
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            })
    };
    let ((x_lo, x_hi), (y_lo, y_hi)) = (bound(0), bound(1));
    let q = Array2::from_shape_vec(
        (request.points.len(), 2),
        request.points.iter().flatten().copied().collect(),
    )
    .map_err(|e| ModelError::InvalidInput {
        reason: e.to_string(),
    })?;
    let nb = model.kneighbors(q.view(), control)?;
    let points: Vec<PlaygroundPoint> = request
        .points
        .iter()
        .enumerate()
        .map(|(r, &point)| {
            let e = model.explain_row(&nb, r);
            let tie_at_kth = model.kth_tie(&point, nb.row_distances(r));
            let proba = e.proba.unwrap_or_default();
            let confidence = proba.iter().copied().fold(0.0, f64::max);
            let d: Vec<f64> = e.neighbors.iter().map(|n| n.distance).collect();
            PlaygroundPoint {
                point,
                class: e.class.unwrap_or(0),
                confidence,
                proba,
                mean_distance: pairwise_sum(&d) / d.len().max(1) as f64,
                neighbors: e.neighbors,
                tie_at_kth,
            }
        })
        .collect();
    let mean_confidence = (!points.is_empty()).then(|| {
        let c: Vec<f64> = points.iter().map(|p| p.confidence).collect();
        pairwise_sum(&c) / c.len() as f64
    });
    Ok(Playground {
        features: request.features,
        colors: classes
            .iter()
            .map(|&c| PALETTE[c % PALETTE.len()].to_owned())
            .collect(),
        classes,
        codes,
        bounds: [x_lo - 1.0, x_hi + 1.0, y_lo - 1.0, y_hi + 1.0],
        points,
        mean_confidence,
        engine: model.engine(),
        config: request.config,
        data,
    })
}
