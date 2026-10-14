//! C11 (D1/D28): ONNX and the Python package of a trained model, with `mltutor-export`.
//! The bytes are saved by `export_model` with the native dialog.

use mltutor_core as core;
use mltutor_export::{onnx_model, package, ExportError, PackageInput};

use crate::ipc::{CommandError, ExportFormat};
use crate::real_models::CoreModel;

/// `random_state=42` (D7), recorded in `modelo.json`.
const SEED: u32 = 42;

fn export_error(error: ExportError) -> CommandError {
    CommandError::export(error)
}

pub fn export(model: &CoreModel, format: ExportFormat) -> Result<Vec<u8>, CommandError> {
    let source = model.export_source();
    let class_values = match &source.dataset.target {
        core::Target::Classes { class_values, .. } => Some(class_values.as_slice()),
        core::Target::Values { .. } => None,
    };
    match format {
        ExportFormat::Onnx => onnx_model(source.model, class_values)
            .map(|m| m.to_bytes())
            .map_err(export_error),
        ExportFormat::PythonPackage => {
            let input = PackageInput {
                model: source.model,
                feature_columns: &source.dataset.feature_columns,
                display_names: &source.dataset.feature_display,
                class_values,
                class_names: source.dataset.class_names(),
                dataset_label: &source.spec.dataset_label,
                test_size: source.spec.test_size,
                random_state: SEED,
                n_train: source.n_train,
                n_test: source.x_test.nrows(),
                x_test: source.x_test,
            };
            package::python_package(&input)
                .and_then(|p| p.to_zip())
                .map_err(export_error)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::core_facade::{CoreFacade, TrainSpec};
    use crate::ipc::{DatasetId, ModelConfig};
    use crate::real_facade::RealFacade;

    #[test]
    fn exports_onnx_and_a_zip_package() {
        for (id, config) in [
            (DatasetId::Iris, ModelConfig::Logistic { max_iter: 1000 }),
            (DatasetId::Diabetes, ModelConfig::Linear),
        ] {
            let spec = TrainSpec {
                config,
                test_size: 0.3,
                dataset_label: "dataset".into(),
            };
            let model = RealFacade
                .load_builtin(id)
                .unwrap()
                .train(&spec, &|_, _| {}, &AtomicBool::new(false))
                .unwrap();
            let onnx = model.export(ExportFormat::Onnx).unwrap();
            assert!(mltutor_export::onnx::Model::from_bytes(&onnx).is_ok());
            let zip = model.export(ExportFormat::PythonPackage).unwrap();
            assert_eq!(&zip[..2], b"PK", "{id:?}");
            let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip)).unwrap();
            let mut json = String::new();
            archive
                .by_name("modelo.json")
                .unwrap()
                .read_to_string(&mut json)
                .unwrap();
            assert!(json.contains("\"format\": \"mltutor-model\""), "{json}");
            assert!(archive.by_name("reconstruir_modelo.py").is_ok());
            assert!(archive.by_name("LEEME.txt").is_ok());
        }
    }
}
