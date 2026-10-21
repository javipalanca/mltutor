//! "🔍 Coeficientes" tab of the 📊 Regresión lesson (fase2-regresion.md §2.5), with
//! the corrections of D27: the intercept is the selected class's, the analysis is
//! numbered 1, 2, 3 by the UI and the "Fuerza del Efecto" is symmetric in |β|.

use serde::{Deserialize, Serialize};

use crate::model::{check_n_classes, ModelError, Predictor};
use crate::stats::quantile_linear;
use crate::TaskType;

/// Sign of a coefficient ("📈 Positivo" if `> 0`, else "📉 Negativo").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    Positive,
    Negative,
}

/// A qualitative level. Importance (linear): `High` "🔥 Alta", `Medium` "🔶 Media",
/// `Low` "🔹 Baja". Strength (logistic): `VeryHigh` "🔥 Muy Fuerte", `High`
/// "🔥 Fuerte", `Medium` "🔶 Moderado", `Low` "🔹 Débil". Intensity of the detailed
/// analysis: `High` "🔥 **FUERTE**", `Medium` "🔶 **MODERADO**", `Low` "🔹 **DÉBIL**".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    VeryHigh,
    High,
    Medium,
    Low,
}

/// One row of the table, in order of decreasing |coef|.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoefficientRow {
    pub feature: String,
    /// Column index of the feature.
    pub feature_position: usize,
    pub coef: f64,
    pub abs: f64,
    /// `e^coef` (logistic).
    pub odds_ratio: Option<f64>,
    /// `(e^coef − 1)·100` (logistic).
    pub change_pct: Option<f64>,
    pub effect: Effect,
    /// Linear: `> q75` High, `> q25` Medium, else Low.
    pub importance: Option<Level>,
    /// Logistic (D27, symmetric): `|β| > 2` VeryHigh, `> 1` High, `> 0.5` Medium.
    pub strength: Option<Level>,
    /// "🔍 Análisis Detallado": logistic `OR > 3`/`> 1.5` (positive) or `OR < 0.33`/
    /// `< 0.67` (negative); linear `|β| > q80`/`> q60`.
    pub intensity: Level,
}

/// Quantiles (pandas, linear interpolation) of |coef| of the shown row.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AbsQuantiles {
    pub q25: f64,
    pub q60: f64,
    pub q75: f64,
    pub q80: f64,
    pub q95: f64,
}

/// The whole tab.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoefficientsView {
    pub model_task: TaskType,
    pub rows: Vec<CoefficientRow>,
    /// Selected class (multinomial only).
    pub class_index: Option<usize>,
    /// Number of rows of `coef_` (1 for linear and binary models).
    pub n_rows: usize,
    /// Intercept of the shown row (D27a).
    pub intercept: f64,
    /// `e^intercept` (logistic, "Odds Ratio Base").
    pub intercept_odds: Option<f64>,
    pub quantiles: AbsQuantiles,
    /// Features with |coef| above the 0.95 quantile, in table order.
    pub extreme: Vec<String>,
}

impl CoefficientsView {
    /// Number of positive and negative coefficients (logistic balance message).
    pub fn signs(&self) -> (usize, usize) {
        let pos = self.rows.iter().filter(|r| r.coef > 0.0).count();
        let neg = self.rows.iter().filter(|r| r.coef < 0.0).count();
        (pos, neg)
    }

    /// Up to 3 rows with positive / negative coefficients, in table order.
    pub fn top(&self, effect: Effect) -> Vec<&CoefficientRow> {
        self.rows
            .iter()
            .filter(|r| match effect {
                Effect::Positive => r.coef > 0.0,
                Effect::Negative => r.coef < 0.0,
            })
            .take(3)
            .collect()
    }
}

/// Builds the tab for a linear model. `class_index` selects the row of a
/// multinomial model (0 by default).
pub fn view(
    model: &dyn Predictor,
    feature_names: &[String],
    class_index: Option<usize>,
) -> Result<CoefficientsView, ModelError> {
    let coefs = model
        .linear_coefficients()
        .ok_or_else(|| ModelError::InvalidSelection {
            reason: "el modelo no tiene coeficientes".to_owned(),
        })?;
    if let Some(classes) = model.classes() {
        check_n_classes(classes.len())?;
    }
    let n_rows = coefs.coef.nrows();
    let row = if n_rows > 1 {
        class_index.unwrap_or(0)
    } else {
        0
    };
    if row >= n_rows {
        return Err(ModelError::InvalidSelection {
            reason: format!("la clase {row} no existe"),
        });
    }
    let coef = coefs.coef.row(row);
    if coef.len() != feature_names.len() {
        return Err(ModelError::FeatureMismatch {
            expected: coef.len(),
            found: feature_names.len(),
        });
    }
    let logistic = model.task() == TaskType::Classification;
    let mut abs_sorted: Vec<f64> = coef.iter().map(|c| c.abs()).collect();
    abs_sorted.sort_by(f64::total_cmp);
    let q = |p: f64| quantile_linear(&abs_sorted, p);
    let quantiles = AbsQuantiles {
        q25: q(0.25),
        q60: q(0.6),
        q75: q(0.75),
        q80: q(0.8),
        q95: q(0.95),
    };

    let mut order: Vec<usize> = (0..coef.len()).collect();
    order.sort_by(|&a, &b| coef[b].abs().total_cmp(&coef[a].abs()));
    let rows: Vec<CoefficientRow> = order
        .into_iter()
        .map(|j| {
            let c = coef[j];
            let abs = c.abs();
            let odds = c.exp();
            let effect = if c > 0.0 {
                Effect::Positive
            } else {
                Effect::Negative
            };
            let intensity = if logistic {
                if c > 0.0 {
                    if odds > 3.0 {
                        Level::High
                    } else if odds > 1.5 {
                        Level::Medium
                    } else {
                        Level::Low
                    }
                } else if odds < 0.33 {
                    Level::High
                } else if odds < 0.67 {
                    Level::Medium
                } else {
                    Level::Low
                }
            } else if abs > quantiles.q80 {
                Level::High
            } else if abs > quantiles.q60 {
                Level::Medium
            } else {
                Level::Low
            };
            CoefficientRow {
                feature: feature_names[j].clone(),
                feature_position: j,
                coef: c,
                abs,
                odds_ratio: logistic.then_some(odds),
                change_pct: logistic.then_some((odds - 1.0) * 100.0),
                effect,
                importance: (!logistic).then_some({
                    if abs > quantiles.q75 {
                        Level::High
                    } else if abs > quantiles.q25 {
                        Level::Medium
                    } else {
                        Level::Low
                    }
                }),
                strength: logistic.then_some({
                    if abs > 2.0 {
                        Level::VeryHigh
                    } else if abs > 1.0 {
                        Level::High
                    } else if abs > 0.5 {
                        Level::Medium
                    } else {
                        Level::Low
                    }
                }),
                intensity,
            }
        })
        .collect();
    let extreme = rows
        .iter()
        .filter(|r| r.abs > quantiles.q95)
        .map(|r| r.feature.clone())
        .collect();
    let intercept = coefs.intercept.get(row).copied().unwrap_or(0.0);
    Ok(CoefficientsView {
        model_task: model.task(),
        rows,
        class_index: (n_rows > 1).then_some(row),
        n_rows,
        intercept,
        intercept_odds: logistic.then(|| intercept.exp()),
        quantiles,
        extreme,
    })
}
