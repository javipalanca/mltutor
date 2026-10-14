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
        let classes: Vec<i64> = model
            .classes()
            .unwrap_or(&[])
            .iter()
            .map(|&c| {
                class_values
                    .and_then(|v| v.get(c))
                    .filter(|v| v.fract() == 0.0)
                    .map_or(c as i64, |&v| v as i64)
            })
            .collect();
        let flat: Vec<f64> = coefs.coef.iter().copied().collect();
        onnx::linear_classifier(&flat, coefs.coef.ncols(), &coefs.intercept, &classes)
    } else {
        Err(ExportError::Unsupported)
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
    use super::VERSION;

    #[test]
    fn version_matches_workspace() {
        assert!(VERSION.starts_with("0.4.0"));
    }
}
