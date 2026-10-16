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
//! | ONNX of a tree (F3) | [`onnx_model`] with a [`mltutor_core::tree::DecisionTree`] → `TreeEnsembleClassifier` / `TreeEnsembleRegressor` (thresholds = largest float32 ≤ t, D47); [`onnx_file_name`] = `arbol_decision.onnx`; by parts [`onnx::tree_ensemble_classifier`]`(&DecisionTree, class_labels: &[i64])`, [`onnx::tree_ensemble_regressor`]`(&DecisionTree)` |
//! | Python package of a tree (F3, D43) | the same [`package::python_package`]: `modelo.json` with `tree_` (rebuilt with `Tree.__setstate__`), `reconstruir_modelo.py` (`paquete_modelo_arbol.py.j2`, writes `decision_tree_model.pkl`) and `LEEME.txt` (`paquete_leeme_arbol.txt.j2`); [`package::zip_name`]`(&dyn Predictor)` = `modelo_arbol.zip` |
//! | ONNX of a KNN (F4, D57) | [`onnx_model`] with a [`mltutor_core::knn::Knn`] → the training rows as an initializer, `Scan` + `TopK` per query row, scikit-learn's distance-0 rule (`Equal` + `Cast` + `Where`), votes or weighted mean; input `float_input` float32 cast to float64; outputs `label` + `probabilities` or `variable`; [`onnx_file_name`] = `knn.onnx`; by parts [`onnx::knn_classifier`]`(&Knn, class_labels: &[i64])`, [`onnx::knn_regressor`]`(&Knn)`; reference evaluation [`onnx::evaluate`] (dispatches to [`onnx::evaluate_knn`]) |
//! | Python package of a KNN (F4, D57) | the same [`package::python_package`]: `modelo.json` with `params` (`get_params()`), `X_train`/`y_train` in fit order and a `check` of the first 5 test rows (neighbours, distances, predictions, probabilities), `reconstruir_modelo.py` (`paquete_modelo_knn.py.j2`, refits and writes `modelo_knn.pkl`) and `LEEME.txt` (`paquete_leeme_knn.txt.j2`); [`package::zip_name`] = `modelo_knn.zip` |
//! | Tree rules (.txt) and code (.py) | core: [`mltutor_core::tree::export_text()`] (`decision_tree_rules.txt`) and [`mltutor_core::codegen::render`]`(CodeTemplate::TreePython, &tree_python_corrected_context(..))` (`arbol_decision.py`) |
//! | ONNX of the 🧠 network (F5, D75) | [`onnx_model`] with a [`mltutor_nn::NeuralNetwork`] → [`onnx::mlp_model`]: `Sub`/`Div` (scaler) + `Gemm` + activation per layer; `label` = dataset class code (`Gather(class_labels)` over `ArgMax`, or over `Greater`→`Cast`→`Squeeze` with one neuron) + `probabilities` (`Concat(1 − p, p)` [N, 2] with one neuron), or `variable`; `class_values` is ignored (the label is the code, as `LEEME.txt` documents); non-finite weights or scaler → `InvalidModel`; [`onnx_file_name`] = `red_neuronal.onnx` |
//! | 📦 Paquete Keras (F5, D75) | [`package::keras::keras_package`]`(&KerasPackageInput { network, arch, dataset_label, feature_names, class_names, final_learning_rate, x_check /* first 5 test rows, original units */, n_train, n_test }) -> Result<KerasPackage { model_json, script, numpy_script, readme }, ExportError>`, then `to_zip()` ([`package::keras::KERAS_ZIP_NAME`] = `modelo_red.zip`); the NumPy predictor alone = `numpy_script` ([`package::keras::NUMPY_FILE_NAME`]); `modelo.json` alone: [`package::keras::keras_model_json`]; `modelo.json` is compact (one line per kernel row, float32 weights with their shortest decimal); non-finite weights, scaler or check outputs → `InvalidModel` |
//! | 📝 Código Python of the network (F5d, D75) | [`package::keras::KerasPackage::numpy_zip`]`()` = `predecir_numpy.py` + the `modelo.json` it reads ([`package::keras::NUMPY_ZIP_NAME`] = `red_predictor_numpy.zip`); the script alone (`numpy_script`) does not run without `modelo.json`. The D74 training script is the core's `generate_code(NnTraining)` |
//! | 📋 Metadatos (F5d, D75, L3) | [`metadata::nn_metadata`]`(&TrainedNetwork, &Dataset, &NnArchitecture, &NnTraining, trained_at: &str) -> Result<serde_json::Value, ExportError>` (Python's `model_info`/`training_info`/`data_info`/`export_info` corrected: `num_classes` = 2 in binary, the training date, MLTutor's version and framework, the real target column, MAE in regression, test metrics, best/stop/divergence epochs; plus `config` with the seed and the scaler); text [`metadata::metadata_json_text`]; [`metadata::METADATA_FILE_NAME`] = `red_neuronal_metadatos.json`. Deterministic; non-finite values → `null`; a diverged network is described (`model_info.valid = false`) but ONNX and the Keras package refuse it (K1) |
//! | Python package (.zip) | [`package::python_package`]`(&PackageInput) -> Result<PythonPackage, ExportError>`, then [`package::PythonPackage::to_zip`]`() -> Result<Vec<u8>, ExportError>`; [`package::ZIP_NAME`] = `modelo_regresion.zip`; [`package::model_json`] alone; [`package::python_zip`]`(&[(&str, &[u8])])` |

pub mod metadata;
pub mod onnx;
pub mod package;

use mltutor_core::codegen::CodegenError;
use mltutor_core::knn::Knn;
use mltutor_core::linear::{LinearRegression, LogisticRegression};
use mltutor_core::model::{ModelError, Predictor};
use mltutor_core::tree::DecisionTree;
use mltutor_nn::NeuralNetwork;
use serde::Serialize;

pub use metadata::nn_metadata;
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

/// The checks every export of a network makes (ONNX and Keras package; review F5a,
/// finding 5): the network was built consistently and every weight, bias and
/// scaler value is finite (a diverged training cannot be exported).
pub(crate) fn check_network(net: &NeuralNetwork) -> Result<(), ExportError> {
    let mlp = net.mlp();
    let scaler = net.scaler();
    if mlp.validate().is_err()
        || scaler.mean.len() != mlp.n_inputs()
        || scaler.scale.len() != mlp.n_inputs()
    {
        return Err(ExportError::InvalidModel {
            reason: "la red y el escalador no coinciden".to_owned(),
        });
    }
    if !mlp.is_finite() {
        return Err(ExportError::InvalidModel {
            reason: "pesos no finitos (el entrenamiento divergió)".to_owned(),
        });
    }
    if !scaler.is_finite() {
        return Err(ExportError::InvalidModel {
            reason: "escalador no finito".to_owned(),
        });
    }
    Ok(())
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
    if let Some(net) = model.as_any().downcast_ref::<NeuralNetwork>() {
        return onnx::mlp_model(net);
    }
    if let Some(tree) = model.as_any().downcast_ref::<DecisionTree>() {
        return match tree.classes() {
            Some(codes) => {
                onnx::tree_ensemble_classifier(tree, &onnx_class_labels(codes, class_values))
            }
            None => onnx::tree_ensemble_regressor(tree),
        };
    }
    if let Some(knn) = model.as_any().downcast_ref::<Knn>() {
        return match knn.classes() {
            Some(codes) => onnx::knn_classifier(knn, &onnx_class_labels(codes, class_values)),
            None => onnx::knn_regressor(knn),
        };
    }
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
    if model.as_any().is::<NeuralNetwork>() {
        package::keras::ONNX_FILE_NAME
    } else if model.as_any().is::<DecisionTree>() {
        "arbol_decision.onnx"
    } else if model.as_any().is::<Knn>() {
        "knn.onnx"
    } else if model.as_any().is::<LogisticRegression>() {
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
