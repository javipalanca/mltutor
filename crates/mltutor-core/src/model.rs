//! The common interface of every trained model (fase2-regresion.md §3.0).
//!
//! Lessons and shared components (metrics, curves, decision grids, the prediction
//! form, exports) only see a [`Predictor`]. Linear and logistic regression (F2)
//! implement it here; CART (F3), KNN (F4) and the neural network (F5) will too.
//!
//! Class labels are always the dataset's class **codes** (`0..n_classes`, the
//! index into `Target::Classes::class_names`). A model trained on a subset where
//! some class is missing only knows the classes it saw: [`Predictor::classes`]
//! tells which code each column of [`Predictor::predict_proba`] belongs to, like
//! `classes_` in scikit-learn.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::dataset::{Target, TaskType};

/// Output of [`Predictor::predict`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "values", rename_all = "snake_case")]
pub enum Prediction {
    /// One class code per row.
    Classes(Vec<usize>),
    /// One value per row.
    Values(Vec<f64>),
}

impl Prediction {
    /// Number of predicted rows.
    pub fn len(&self) -> usize {
        match self {
            Prediction::Classes(c) => c.len(),
            Prediction::Values(v) => v.len(),
        }
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The predictions as numbers (class codes as `f64`), e.g. for a surface plot.
    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            Prediction::Classes(c) => c.iter().map(|&c| c as f64).collect(),
            Prediction::Values(v) => v.clone(),
        }
    }
}

/// Largest number of classes of a classification problem (I5). Above it the
/// confusion matrix, the curves and the boundary colours stop being readable, and a
/// target with so many distinct values is almost always a number to regress.
pub const MAX_CLASSES: usize = 50;

/// `Err(TooManyClasses)` when `n_classes > MAX_CLASSES`: training and every
/// classification view call it.
pub fn check_n_classes(n_classes: usize) -> Result<(), ModelError> {
    if n_classes > MAX_CLASSES {
        Err(ModelError::TooManyClasses {
            n_classes,
            max: MAX_CLASSES,
        })
    } else {
        Ok(())
    }
}

/// Errors of training, prediction and the model views. Messages are in Spanish
/// because they can reach the student; `src-tauri` maps each variant to its own
/// `CommandError`.
#[derive(Debug, Clone, PartialEq, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelError {
    /// D26: logistic regression on a regression target.
    #[error("La regresión logística necesita una variable objetivo con clases. Elige 📈 Regresión Lineal o un dataset de clasificación.")]
    InvalidTarget,
    /// More than [`MAX_CLASSES`] classes.
    #[error("El objetivo tiene {n_classes} clases y el máximo para clasificación es {max}. Si es una variable numérica, elige 📈 Regresión Lineal (o el tipo de tarea Regresión al cargar el CSV).")]
    TooManyClasses { n_classes: usize, max: usize },
    /// Fewer than two classes in the training data.
    #[error("Se necesitan al menos 2 clases en los datos de entrenamiento y hay {found}")]
    TooFewClasses { found: usize },
    /// No training rows.
    #[error("No hay datos de entrenamiento")]
    Empty,
    /// The number of columns does not match the model.
    #[error("El modelo espera {expected} características y se han recibido {found}")]
    FeatureMismatch { expected: usize, found: usize },
    /// Rows of `X` and length of `y` differ.
    #[error("X tiene {rows} filas y el objetivo {targets} valores")]
    LengthMismatch { rows: usize, targets: usize },
    /// NaN or infinite values in the inputs.
    #[error("Los datos contienen valores no válidos (NaN o infinito)")]
    NonFinite,
    /// The solver failed (it should not happen with finite data).
    #[error("Error numérico al entrenar el modelo: {message}")]
    Numerical { message: String },
    /// The cancellation flag was set.
    #[error("Operación cancelada")]
    Cancelled,
    /// Invalid feature selection or parameter (e.g. the same feature twice).
    #[error("Selección no válida: {reason}")]
    InvalidSelection { reason: String },
    /// The decision grid has fewer than 2 rows or columns.
    #[error("❌ Error: La malla resultante es demasiado pequeña: ({rows}, {cols})")]
    GridTooSmall { rows: usize, cols: usize },
    /// A classification view was requested for a regressor.
    #[error("El modelo no es de clasificación")]
    NotClassifier,
    /// A regression view was requested for a classifier.
    #[error("El modelo no es de regresión")]
    NotRegressor,
    /// Invalid input values (prediction form).
    #[error("Entrada no válida: {reason}")]
    InvalidInput { reason: String },
}

/// Progress of a training run, reported through [`TrainControl`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainProgress {
    /// Iterations done so far (0 at the start).
    pub iteration: u64,
    /// Iteration budget (`max_iter`); 0 for closed-form models.
    pub max_iter: u64,
    /// Current value of the objective, when the algorithm has one.
    pub loss: Option<f64>,
    /// `true` on the last report.
    pub finished: bool,
}

/// Cooperative cancellation and progress reporting for long computations.
///
/// The flag is checked between solver iterations and between rows of a grid; the
/// callback runs on the worker thread.
#[derive(Clone, Copy, Default)]
pub struct TrainControl<'a> {
    cancel: Option<&'a AtomicBool>,
    progress: Option<&'a (dyn Fn(TrainProgress) + Sync)>,
}

impl fmt::Debug for TrainControl<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrainControl")
            .field("cancel", &self.cancel)
            .field("progress", &self.progress.is_some())
            .finish()
    }
}

impl<'a> TrainControl<'a> {
    /// No cancellation, no progress.
    pub fn none() -> Self {
        Self::default()
    }

    /// With a cancellation flag.
    pub fn with_cancel(mut self, cancel: &'a AtomicBool) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// With a progress callback.
    pub fn with_progress(mut self, progress: &'a (dyn Fn(TrainProgress) + Sync)) -> Self {
        self.progress = Some(progress);
        self
    }

    /// The cancellation flag, if any.
    pub fn cancel_flag(&self) -> Option<&'a AtomicBool> {
        self.cancel
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_some_and(|c| c.load(Ordering::Relaxed))
    }

    /// `Err(Cancelled)` if cancellation was requested.
    pub fn check(&self) -> Result<(), ModelError> {
        if self.is_cancelled() {
            Err(ModelError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Reports progress (no-op without a callback).
    pub fn report(&self, progress: TrainProgress) {
        if let Some(cb) = self.progress {
            cb(progress);
        }
    }
}

/// Coefficients of a linear model, with scikit-learn's shapes: `coef` is
/// `n_rows × n_features` (1 row for linear regression and binary logistic
/// regression, one row per class for the multinomial one) and `intercept` has
/// `n_rows` values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearCoefficients {
    pub coef: Array2<f64>,
    pub intercept: Vec<f64>,
}

/// What happened during training (iterative solvers).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitReport {
    /// Iterations used (`n_iter_`); 0 for closed-form solutions.
    pub n_iter: u64,
    /// Iteration budget (`max_iter`); 0 for closed-form solutions.
    pub max_iter: u64,
    /// Whether scikit-learn's stopping criterion holds at the solution
    /// (D19: show "⚠️ El optimizador no ha convergido…" when `false`).
    pub converged: bool,
}

/// A trained model.
pub trait Predictor: Send + Sync + fmt::Debug {
    /// Task of the **model** (a linear regression trained on Iris is a regressor).
    fn task(&self) -> TaskType;

    /// Number of input features.
    fn n_features(&self) -> usize;

    /// Class codes of the columns of [`Predictor::predict_proba`], sorted
    /// (`classes_`). `None` for regressors.
    fn classes(&self) -> Option<&[usize]>;

    /// Predicted class codes or values, one per row of `x`.
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Prediction, ModelError>;

    /// Class probabilities (`n_rows × classes().len()`), or `None` if the model has
    /// no `predict_proba`.
    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Option<Result<Array2<f64>, ModelError>>;

    /// The same algorithm and hyperparameters trained on other data (the 2-feature
    /// models of the decision boundary and the prediction surface).
    fn refit(
        &self,
        x: ArrayView2<'_, f64>,
        y: &Target,
        control: &TrainControl<'_>,
    ) -> Result<Box<dyn Predictor>, ModelError>;

    /// Coefficients of linear models (Coefficients tab, exports).
    fn linear_coefficients(&self) -> Option<&LinearCoefficients> {
        None
    }

    /// Training report of iterative models.
    fn fit_report(&self) -> Option<FitReport> {
        None
    }

    /// Downcast to the concrete model (exports of a specific algorithm).
    fn as_any(&self) -> &dyn Any;
}

/// Validates the shape and values of an input matrix for a model with
/// `n_features` columns.
pub(crate) fn check_input(x: ArrayView2<'_, f64>, n_features: usize) -> Result<(), ModelError> {
    if x.ncols() != n_features {
        return Err(ModelError::FeatureMismatch {
            expected: n_features,
            found: x.ncols(),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ModelError::NonFinite);
    }
    Ok(())
}

/// Index of the largest value; the first one wins ties (NumPy's `argmax`).
pub(crate) fn argmax(values: impl IntoIterator<Item = f64>) -> usize {
    let mut best = 0;
    let mut best_v = f64::NEG_INFINITY;
    for (i, v) in values.into_iter().enumerate() {
        if v > best_v || i == 0 {
            best = i;
            best_v = v;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_keeps_first_tie() {
        assert_eq!(argmax([0.2, 0.5, 0.5]), 1);
        assert_eq!(argmax([1.0]), 0);
    }

    #[test]
    fn class_limit() {
        assert_eq!(check_n_classes(MAX_CLASSES), Ok(()));
        let err = check_n_classes(51).unwrap_err();
        assert_eq!(
            err,
            ModelError::TooManyClasses {
                n_classes: 51,
                max: 50
            }
        );
        assert!(err.to_string().contains("📈 Regresión Lineal"));
    }

    #[test]
    fn control_cancels() {
        let flag = AtomicBool::new(false);
        let ctl = TrainControl::none().with_cancel(&flag);
        assert!(ctl.check().is_ok());
        flag.store(true, Ordering::Relaxed);
        assert_eq!(ctl.check(), Err(ModelError::Cancelled));
    }
}
