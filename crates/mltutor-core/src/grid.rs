//! Decision boundary and prediction surface on two features (fase2-regresion.md C8,
//! `viz/decision_boundary.py`).
//!
//! The model is **refitted** on the two chosen training columns (same algorithm
//! and hyperparameters, [`Predictor::refit`]); the other features are not fixed at
//! any value. The grid follows the app exactly (D24):
//!
//! 1. `h = 0.02` (KNN: [`knn_step`]); limits = `min − 1`, `max + 1` per axis.
//! 2. If the larger range exceeds 100, `h = max(h, range/200)`; then
//!    `h = min(h, range_x/10, range_y/10)`.
//! 3. Axes = `np.arange(min, max, h)` ([`np_arange`]); an axis with fewer than 2
//!    points becomes `linspace(min, max, 10)`.
//! 4. Above [`MAX_GRID_POINTS`] points, each axis becomes
//!    `linspace(min, max, min(200, len))`.
//!
//! Only the final axes are materialised (Titanic's fare gives 2 M points before
//! step 4).

use ndarray::{Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::dataset::{Target, TaskType};
use crate::model::{check_n_classes, ModelError, Prediction, Predictor, TrainControl};
use crate::stats::linspace;

/// Default grid step (`h = 0.02`).
pub const DEFAULT_STEP: f64 = 0.02;
/// Above this many points the grid is reduced (Python's "límite de seguridad").
pub const MAX_GRID_POINTS: usize = 50_000;
/// Points per axis after the reduction.
pub const REDUCED_AXIS_POINTS: usize = 200;

/// KNN's coarser step (F4): 0.1 with more than 1000 samples, 0.05 with more than
/// 500, 0.02 otherwise. `n` is the size of the **whole dataset**: in 🔍 KNN the
/// boundary and the surface refit on every row (D63), so pass `x`/`y` complete to
/// [`decision_grid`] and `knn_step(n_total)` (Cancer: 569 rows → 0.05).
pub fn knn_step(n: usize) -> f64 {
    if n > 1000 {
        0.1
    } else if n > 500 {
        0.05
    } else {
        DEFAULT_STEP
    }
}

/// Number of values of `np.arange(start, stop, step)`: `ceil((stop − start)/step)`.
pub fn np_arange_len(start: f64, stop: f64, step: f64) -> usize {
    let len = ((stop - start) / step).ceil();
    if len.is_finite() && len > 0.0 {
        len as usize
    } else {
        0
    }
}

/// `np.arange(start, stop, step)` with NumPy's rounding: value `i` is
/// `start + i·((start + step) − start)`, not `start + i·step`.
pub fn np_arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = np_arange_len(start, stop, step);
    let delta = (start + step) - start;
    (0..n)
        .map(|i| {
            if i == 0 {
                start
            } else {
                start + i as f64 * delta
            }
        })
        .collect()
}

/// Informative messages of the grid ("Malla creada: …").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GridNotice {
    /// "Malla creada: {rows} x {cols} = {rows·cols:,} puntos" (rows = y points).
    Created { rows: usize, cols: usize },
    /// "⚠️ La malla es muy grande ({points:,} puntos). Reduciendo resolución..."
    TooLarge { points: usize },
    /// "Nueva resolución: {rows} x {cols} = {rows·cols:,} puntos"
    Resized { rows: usize, cols: usize },
}

/// The two axes of the grid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GridAxes {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub notices: Vec<GridNotice>,
}

fn min_max(values: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    values.fold(None, |acc, v| match acc {
        None => Some((v, v)),
        Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
    })
}

/// The axes for points `(xs, ys)` with initial step `h` (steps 1–4 above).
pub fn decision_axes(xs: &[f64], ys: &[f64], h: f64) -> Result<GridAxes, ModelError> {
    let empty = || ModelError::InvalidSelection {
        reason: "no hay datos para la malla".to_owned(),
    };
    let (x_lo, x_hi) = min_max(xs.iter().copied()).ok_or_else(empty)?;
    let (y_lo, y_hi) = min_max(ys.iter().copied()).ok_or_else(empty)?;
    let (x_min, x_max) = (x_lo - 1.0, x_hi + 1.0);
    let (y_min, y_max) = (y_lo - 1.0, y_hi + 1.0);
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    let max_range = x_range.max(y_range);
    let mut h = h;
    if max_range > 100.0 {
        h = h.max(max_range / 200.0);
    }
    h = h.min(x_range / 10.0).min(y_range / 10.0);

    let axis = |lo: f64, hi: f64| -> (usize, Option<Vec<f64>>) {
        let n = np_arange_len(lo, hi, h);
        if n < 2 {
            (10, Some(linspace(lo, hi, 10)))
        } else {
            (n, None)
        }
    };
    let (nx, fixed_x) = axis(x_min, x_max);
    let (ny, fixed_y) = axis(y_min, y_max);
    let mut notices = vec![GridNotice::Created { rows: ny, cols: nx }];
    let points = nx.saturating_mul(ny);
    let (x, y) = if points > MAX_GRID_POINTS {
        notices.push(GridNotice::TooLarge { points });
        let x = linspace(x_min, x_max, REDUCED_AXIS_POINTS.min(nx));
        let y = linspace(y_min, y_max, REDUCED_AXIS_POINTS.min(ny));
        notices.push(GridNotice::Resized {
            rows: y.len(),
            cols: x.len(),
        });
        (x, y)
    } else {
        (
            fixed_x.unwrap_or_else(|| np_arange(x_min, x_max, h)),
            fixed_y.unwrap_or_else(|| np_arange(y_min, y_max, h)),
        )
    };
    if x.len() < 2 || y.len() < 2 {
        return Err(ModelError::GridTooSmall {
            rows: y.len(),
            cols: x.len(),
        });
    }
    Ok(GridAxes { x, y, notices })
}

/// Predictions on the grid, row-major with rows = `y` values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GridValues {
    /// Class codes (decision boundary).
    Classes { z: Vec<u16> },
    /// Values (prediction surface) and their range.
    Values { z: Vec<f64>, min: f64, max: f64 },
}

/// Everything C8 draws.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionGrid {
    pub axes: GridAxes,
    pub values: GridValues,
    /// The two features (column indices of the dataset).
    pub features: [usize; 2],
    /// Training points on those two features (`n × 2`).
    pub points: Array2<f64>,
    /// Their numeric target, as Python keeps it in `y` (`Target::numeric`): the
    /// class *values* (e.g. 3 and 6 in a CSV) or the regression values.
    pub point_target: Vec<f64>,
    /// Class codes (`0..k`, those of `GridValues::Classes::z`) of the same points
    /// when the target has classes (`y_train.class_codes()`); `None` in regression.
    /// Colour the points with these, not with `point_target`: with classes {3, 6}
    /// the grid says 0/1 and the values say 3/6.
    pub point_codes: Option<Vec<usize>>,
    /// Coefficients of the 2-feature model, when it is linear (tests, code).
    pub model_2d: Option<crate::model::LinearCoefficients>,
    /// `st.warning` "⚠️ Esta visualización solo muestra 2 características…" when
    /// the model has more than 2 features.
    pub show_2d_warning: bool,
}

/// Options of [`decision_grid`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridOptions {
    /// Initial step `h` ([`DEFAULT_STEP`], or [`knn_step`]).
    pub step: f64,
}

impl Default for GridOptions {
    fn default() -> Self {
        Self { step: DEFAULT_STEP }
    }
}

/// Smallest number of mesh cells predicted per call (see [`decision_grid`]): more
/// than scikit-learn's `4 · chunk_size` = 1024 query rows of a one-thread `ArgKmin`.
const GRID_BLOCK_MIN_POINTS: usize = 1025;

/// C8: refits `model` on training columns `fx`/`fy` and predicts on the grid. The
/// cancellation flag is checked before the refit, before every block of rows and,
/// for KNN, inside the prediction ([`Predictor::predict_with_control`]).
pub fn decision_grid(
    model: &dyn Predictor,
    x_train: ArrayView2<'_, f64>,
    y_train: &Target,
    fx: usize,
    fy: usize,
    options: GridOptions,
    control: &TrainControl<'_>,
) -> Result<DecisionGrid, ModelError> {
    let p = x_train.ncols();
    if fx == fy {
        return Err(ModelError::InvalidSelection {
            reason: "elige dos características distintas".to_owned(),
        });
    }
    if fx >= p || fy >= p {
        return Err(ModelError::InvalidSelection {
            reason: format!("el modelo tiene {p} características"),
        });
    }
    if let Some(classes) = model.classes() {
        check_n_classes(classes.len())?;
    }
    control.check()?;
    let points = x_train.select(Axis(1), &[fx, fy]);
    let model_2d = model.refit(points.view(), y_train, control)?;
    let xs = points.column(0).to_vec();
    let ys = points.column(1).to_vec();
    let axes = decision_axes(&xs, &ys, options.step)?;

    let classification = model_2d.task() == TaskType::Classification;
    let (nx, ny) = (axes.x.len(), axes.y.len());
    let total = nx * ny;
    let mut classes = Vec::with_capacity(if classification { total } else { 0 });
    let mut values = Vec::with_capacity(if classification { 0 } else { total });
    // Python predicts the whole mesh in one call. Rows of the mesh are predicted in
    // blocks of at least `GRID_BLOCK_MIN_POINTS` cells (the last block takes the
    // remainder) or the whole mesh if it is smaller, so that every call sees the same
    // brute-force strategy as scikit-learn's single call (a KNN orders tied
    // neighbours differently with ≤ 1024 query rows, `knn::search`).
    let rows_per_block = GRID_BLOCK_MIN_POINTS.div_ceil(nx.max(1)).max(1);
    let n_blocks = (ny / rows_per_block).max(1);
    for b in 0..n_blocks {
        control.check()?;
        let start = b * rows_per_block;
        let end = if b + 1 == n_blocks {
            ny
        } else {
            start + rows_per_block
        };
        let mut block = Array2::<f64>::zeros(((end - start) * nx, 2));
        for (r, &y) in axes.y[start..end].iter().enumerate() {
            for (c, &x) in axes.x.iter().enumerate() {
                block[[r * nx + c, 0]] = x;
                block[[r * nx + c, 1]] = y;
            }
        }
        match model_2d.predict_with_control(block.view(), control)? {
            Prediction::Classes(c) => {
                classes.extend(c.iter().map(|&v| u16::try_from(v).unwrap_or(u16::MAX)));
            }
            Prediction::Values(v) => values.extend(v),
        }
    }
    let values = if classification {
        GridValues::Classes { z: classes }
    } else {
        let (min, max) = min_max(values.iter().copied()).unwrap_or((f64::NAN, f64::NAN));
        GridValues::Values {
            z: values,
            min,
            max,
        }
    };
    Ok(DecisionGrid {
        axes,
        values,
        features: [fx, fy],
        point_target: y_train.numeric(),
        point_codes: y_train.class_codes(),
        model_2d: model_2d.linear_coefficients().cloned(),
        points,
        show_2d_warning: p > 2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arange_like_numpy() {
        let a = np_arange(0.1, 1.0, 0.1);
        assert_eq!(a.len(), 9);
        assert_eq!(a[2], 0.300_000_000_000_000_04);
        assert_eq!(a[6], 0.700_000_000_000_000_1);
        assert!(np_arange(1.0, 0.0, 0.1).is_empty());
    }

    #[test]
    fn huge_grids_are_reduced_without_materialising() {
        let axes = decision_axes(&[0.0, 10_000.0], &[0.0, 1.0], DEFAULT_STEP)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(axes.x.len(), 200);
        assert!(matches!(axes.notices[1], GridNotice::TooLarge { .. }));
    }

    /// Review F2 (high): a CSV whose classes are 3 and 6. The grid predicts codes
    /// 0/1; the points must carry the same codes (and keep 3/6 as values).
    #[test]
    fn csv_classes_3_and_6_have_matching_point_codes() {
        let mut text = String::from("a,b,y\n");
        for i in 0..40 {
            let (a, b) = (f64::from(i) * 0.25, f64::from(i % 7));
            let y = if i < 20 { 3 } else { 6 };
            text.push_str(&format!("{a},{b},{y}\n"));
        }
        let ds = crate::csv::read_csv(text.as_bytes())
            .and_then(|t| t.to_dataset("y", crate::TaskChoice::Classification, "c.csv"))
            .unwrap_or_else(|e| panic!("{e}"));
        let ctl = TrainControl::none();
        let model = crate::linear::fit(
            crate::linear::ModelConfig::Logistic { max_iter: 1000 },
            ds.features.view(),
            &ds.target,
            &ctl,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let grid = decision_grid(
            model.as_ref(),
            ds.features.view(),
            &ds.target,
            0,
            1,
            GridOptions::default(),
            &ctl,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let GridValues::Classes { z } = &grid.values else {
            panic!("expected classes")
        };
        assert!(z.iter().all(|&c| c <= 1));
        assert!(z.contains(&0) && z.contains(&1));
        let codes = grid.point_codes.unwrap_or_else(|| panic!("point_codes"));
        assert_eq!(codes[..20], [0; 20]);
        assert_eq!(codes[20..], [1; 20]);
        assert_eq!(grid.point_target[0], 3.0);
        assert_eq!(grid.point_target[39], 6.0);
    }

    #[test]
    fn regression_grid_has_no_point_codes() {
        let x = ndarray::array![[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]];
        let y = Target::Values {
            values: vec![1.0, 2.0, 3.0, 4.0],
        };
        let ctl = TrainControl::none();
        let model = crate::linear::fit(crate::linear::ModelConfig::Linear, x.view(), &y, &ctl)
            .unwrap_or_else(|e| panic!("{e}"));
        let grid = decision_grid(
            model.as_ref(),
            x.view(),
            &y,
            0,
            1,
            GridOptions::default(),
            &ctl,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(grid.point_codes, None);
    }
}
