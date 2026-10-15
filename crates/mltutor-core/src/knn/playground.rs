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

use super::{fit, Knn, KnnConfig, Neighbor, SearchEngine};
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

/// C18 on the whole dataset `x` (all rows, every feature) and its target: fits the
/// 2-feature model ([`playground_model`]) and classifies the points
/// ([`playground_points`]).
pub fn playground(
    x: ArrayView2<'_, f64>,
    y: &Target,
    request: &PlaygroundRequest,
    control: &TrainControl<'_>,
) -> Result<Playground, ModelError> {
    check_points(&request.points)?;
    let model = playground_model(x, y, request.features, request.config, control)?;
    control.check()?;
    playground_points(&model, request.features, &request.points, control)
}

/// The 2-feature KNN of the playground: columns `features` of every row of `x`,
/// with the playground's K, weights and metric. The caller may keep it (keyed by
/// `(features, K, weights, metric)`) and call [`playground_points`] on every drag
/// without refitting.
pub fn playground_model(
    x: ArrayView2<'_, f64>,
    y: &Target,
    features: [usize; 2],
    config: KnnConfig,
    control: &TrainControl<'_>,
) -> Result<Knn, ModelError> {
    let [fx, fy] = features;
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
    if y.class_codes().is_none() {
        return Err(ModelError::NotClassifier);
    }
    let data = x.select(Axis(1), &[fx, fy]);
    fit(config, data.view(), y, control)
}

fn check_points(points: &[[f64; 2]]) -> Result<(), ModelError> {
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(ModelError::InvalidInput {
            reason: "las coordenadas de los puntos deben ser números".to_owned(),
        });
    }
    Ok(())
}

/// C18 with an already fitted 2-feature classifier `model` (from
/// [`playground_model`]; `features` are the dataset columns it was fitted on, only
/// echoed in the result). `control` is checked before the search and before the
/// O(n) tie count of every point.
pub fn playground_points(
    model: &Knn,
    features: [usize; 2],
    points: &[[f64; 2]],
    control: &TrainControl<'_>,
) -> Result<Playground, ModelError> {
    if model.n_features != 2 {
        return Err(ModelError::FeatureMismatch {
            expected: 2,
            found: model.n_features,
        });
    }
    let codes = model.train_codes().ok_or(ModelError::NotClassifier)?;
    check_points(points)?;
    control.check()?;
    let data = model.train_x().to_owned();
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
        (points.len(), 2),
        points.iter().flatten().copied().collect(),
    )
    .map_err(|e| ModelError::InvalidInput {
        reason: e.to_string(),
    })?;
    let nb = model.kneighbors(q.view(), control)?;
    let mut out = Vec::with_capacity(points.len());
    for (r, &point) in points.iter().enumerate() {
        let e = model.explain_row(&nb, r);
        control.check()?;
        let tie_at_kth = model.kth_tie(&point, nb.row_distances(r));
        let proba = e.proba.unwrap_or_default();
        let confidence = proba.iter().copied().fold(0.0, f64::max);
        let d: Vec<f64> = e.neighbors.iter().map(|n| n.distance).collect();
        out.push(PlaygroundPoint {
            point,
            class: e.class.unwrap_or(0),
            confidence,
            proba,
            mean_distance: pairwise_sum(&d) / d.len().max(1) as f64,
            neighbors: e.neighbors,
            tie_at_kth,
        });
    }
    let points = out;
    let mean_confidence = (!points.is_empty()).then(|| {
        let c: Vec<f64> = points.iter().map(|p| p.confidence).collect();
        pairwise_sum(&c) / c.len() as f64
    });
    Ok(Playground {
        features,
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
        config: *model.config(),
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cached_model_gives_the_same_playground() {
        let x = array![
            [0.0, 0.0, 9.0],
            [1.0, 0.0, 9.0],
            [5.0, 5.0, 9.0],
            [6.0, 5.0, 9.0],
            [0.0, 1.0, 9.0],
            [5.0, 6.0, 9.0]
        ];
        let y = Target::Classes {
            codes: vec![0, 0, 1, 1, 0, 1],
            class_names: vec!["a".into(), "b".into()],
            class_values: vec![0.0, 1.0],
        };
        let req = PlaygroundRequest {
            features: [0, 1],
            config: KnnConfig {
                n_neighbors: 3,
                ..KnnConfig::default()
            },
            points: vec![[0.5, 0.5], [5.5, 5.5]],
        };
        let ctl = TrainControl::none();
        let full = playground(x.view(), &y, &req, &ctl).unwrap_or_else(|e| panic!("{e}"));
        let m = playground_model(x.view(), &y, req.features, req.config, &ctl)
            .unwrap_or_else(|e| panic!("{e}"));
        let cached = playground_points(&m, req.features, &req.points, &ctl)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(full, cached);
        assert_eq!(
            full.points.iter().map(|p| p.class).collect::<Vec<_>>(),
            vec![0, 1]
        );
        let flag = std::sync::atomic::AtomicBool::new(true);
        assert_eq!(
            playground_points(
                &m,
                req.features,
                &req.points,
                &TrainControl::none().with_cancel(&flag)
            )
            .unwrap_err(),
            ModelError::Cancelled
        );
    }
}
