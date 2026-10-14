//! Linear models of the 📊 Regresión lesson: [`LinearRegression`] (OLS) and
//! [`LogisticRegression`] (binary and multinomial), plus [`ModelConfig`] and
//! [`fit`], the entry point of `train_model`.

pub mod logistic;
pub mod ols;

use serde::{Deserialize, Serialize};

pub use logistic::{LogisticParams, LogisticRegression};
pub use ols::LinearRegression;

use crate::dataset::Target;
use crate::model::{ModelError, Predictor, TrainControl};
use ndarray::ArrayView2;

/// Model chosen in the Training tab (contract `ModelConfig`, fase2-regresion.md §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelConfig {
    /// `LinearRegression()`.
    Linear,
    /// `LogisticRegression(max_iter=max_iter, random_state=42)`.
    Logistic { max_iter: u32 },
}

/// Trains the configured model on `x`/`y` (the training rows).
///
/// `Linear` accepts any target (classification codes become their numeric class
/// values, as in Python); `Logistic` rejects regression targets (D26).
pub fn fit(
    config: ModelConfig,
    x: ArrayView2<'_, f64>,
    y: &Target,
    control: &TrainControl<'_>,
) -> Result<Box<dyn Predictor>, ModelError> {
    match config {
        ModelConfig::Linear => Ok(Box::new(LinearRegression::fit_target(x, y, control)?)),
        ModelConfig::Logistic { max_iter } => {
            let params = LogisticParams {
                max_iter,
                ..LogisticParams::default()
            };
            Ok(Box::new(LogisticRegression::fit_target(
                x, y, params, control,
            )?))
        }
    }
}
