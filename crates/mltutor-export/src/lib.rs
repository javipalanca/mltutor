//! ONNX and Python-package exports for MLTutor models (fase2-regresion.md C11,
//! decisions D1 and D28).
//!
//! # Public API used by the IPC contract (`export_model`, fase2-regresion.md §5.4)
//!
//! This list is **stable**; `src-tauri` opens the native "Guardar" dialog and
//! writes the bytes.
//!
//! | Format | Function |
//! |---|---|
//! | ONNX (any F2 model) | [`onnx_model`]`(&dyn Predictor, class_values: Option<&[f64]>) -> Result<onnx::Model, ExportError>`, then [`onnx::Model::to_bytes`]`() -> Vec<u8>`; suggested names [`onnx_file_name`]`(&dyn Predictor)` (`regresion_lineal.onnx` / `regresion_logistica.onnx`) |
//! | ONNX by parts | [`onnx::linear_regressor`]`(coef: &[f64], intercept: f64)`, [`onnx::linear_classifier`]`(coef: &[f64], n_features, intercept: &[f64], classes: &[i64])`; [`onnx::evaluate`] (float32 reference evaluation) |
//! | Python package (.zip) | [`package::python_package`]`(&PackageInput) -> Result<PythonPackage, ExportError>`, then [`package::PythonPackage::to_zip`]`() -> Result<Vec<u8>, ExportError>`; [`package::ZIP_NAME`] = `modelo_regresion.zip`; [`package::model_json`] alone; [`package::python_zip`]`(&[(&str, &[u8])])` |

pub mod onnx;
pub mod package;

use mltutor_core::codegen::CodegenError;
use mltutor_core::linear::{LinearRegression, LogisticRegression};
use mltutor_core::model::{ModelError, Predictor};
use serde::Serialize;

pub use package::{PackageInput, PythonPackage};

/// Version of the MLTutor workspace this crate was built from.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Export errors (Spanish messages; `src-tauri` maps them to `CommandError::Export`).
#[derive(Debug, Clone, PartialEq, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportError {
    /// The model type has no exporter yet.
    #[error("Este tipo de modelo todavía no se puede exportar")]
    Unsupported,
    /// Inconsistent or non-finite parameters.
    #[error("No se puede exportar el modelo: {reason}")]
    InvalidModel { reason: String },
    #[error("Error al generar el archivo ONNX: {message}")]
    Onnx { message: String },
    #[error("Error al generar el archivo ZIP: {message}")]
    Zip { message: String },
    #[error("Error al generar modelo.json: {message}")]
    Json { message: String },
    #[error(transparent)]
    Codegen(#[from] CodegenError),
    #[error(transparent)]
    Model(#[from] ModelError),
}

/// ONNX graph of an F2 model. `class_values` = numeric value of each class code
/// (`classes_` of scikit-learn; the codes themselves when `None`).
///
/// `LinearClassifier` takes integer labels (`classlabels_ints`), so the class
/// values are used only when **all** of them are integers that fit in `i64` and
/// stay distinct; otherwise (e.g. `[2.5, 0.0]`) every class gets its code
/// (`0..k`), so that no two classes share a label. See [`onnx_class_labels`].
pub fn onnx_model(
    model: &dyn Predictor,
    class_values: Option<&[f64]>,
) -> Result<onnx::Model, ExportError> {
    let coefs = model
        .linear_coefficients()
        .ok_or(ExportError::Unsupported)?;
    let any = model.as_any();
    if any.is::<LinearRegression>() {
        let coef = coefs.coef.row(0).to_vec();
        onnx::linear_regressor(&coef, coefs.intercept.first().copied().unwrap_or(0.0))
    } else if any.is::<LogisticRegression>() {
        let classes = onnx_class_labels(model.classes().unwrap_or(&[]), class_values);
        let flat: Vec<f64> = coefs.coef.iter().copied().collect();
        onnx::linear_classifier(&flat, coefs.coef.ncols(), &coefs.intercept, &classes)
    } else {
        Err(ExportError::Unsupported)
    }
}

/// The `classlabels_ints` of a classifier with classes `codes` (see [`onnx_model`]):
/// the class values when all are distinct integers, the codes otherwise.
pub fn onnx_class_labels(codes: &[usize], class_values: Option<&[f64]>) -> Vec<i64> {
    let as_codes = || codes.iter().map(|&c| c as i64).collect::<Vec<_>>();
    let Some(values) = class_values else {
        return as_codes();
    };
    // 2^63 as f64: integral values strictly below it in magnitude fit in i64.
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    let mut labels = Vec::with_capacity(codes.len());
    for &c in codes {
        match values.get(c) {
            Some(&v) if v.is_finite() && v.fract() == 0.0 && v.abs() < LIMIT => {
                labels.push(v as i64);
            }
            _ => return as_codes(),
        }
    }
    let mut sorted = labels.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() == labels.len() {
        labels
    } else {
        as_codes()
    }
}

/// File name suggested by the "Guardar" dialog.
pub fn onnx_file_name(model: &dyn Predictor) -> &'static str {
    if model.as_any().is::<LogisticRegression>() {
        "regresion_logistica.onnx"
    } else {
        "regresion_lineal.onnx"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mltutor_core::model::TrainControl;
    use ndarray::array;

    #[test]
    fn onnx_labels_never_repeat() {
        // Review F2: [2.5, 0.0] used to give [0, 0] (2.5 → code 0, 0.0 → 0).
        assert_eq!(onnx_class_labels(&[0, 1], Some(&[2.5, 0.0])), vec![0, 1]);
        assert_eq!(onnx_class_labels(&[0, 1], Some(&[3.0, 6.0])), vec![3, 6]);
        assert_eq!(
            onnx_class_labels(&[0, 2], Some(&[1.0, 5.0, 7.0])),
            vec![1, 7]
        );
        assert_eq!(onnx_class_labels(&[0, 1], None), vec![0, 1]);
        assert_eq!(
            onnx_class_labels(&[0, 1], Some(&[f64::NAN, 1.0])),
            vec![0, 1]
        );
        assert_eq!(onnx_class_labels(&[0, 1], Some(&[1e300, 1.0])), vec![0, 1]);
        // Codes missing from `class_values`.
        assert_eq!(onnx_class_labels(&[0, 1], Some(&[4.0])), vec![0, 1]);
    }

    #[test]
    fn onnx_model_with_fractional_class_values() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = [0, 0, 1, 0, 1, 1];
        let model = LogisticRegression::fit(
            x.view(),
            &y,
            mltutor_core::linear::LogisticParams::default(),
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let onnx = onnx_model(&model, Some(&[2.5, 0.0])).unwrap_or_else(|e| panic!("{e}"));
        let (labels, _) =
            onnx::evaluate(&onnx, &[vec![0.0], vec![5.0]]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(labels, vec![0, 1]);
    }

    #[test]
    fn version_matches_workspace() {
        assert!(VERSION.starts_with("0.4.0"));
    }
}
