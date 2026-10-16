//! 📦 Paquete Keras (.zip) of the 🧠 network (C28, D75): what replaces Python's
//! SavedModel/HDF5/TFLite/pickle exports, which need TensorFlow.
//!
//! - `modelo.json` (`mltutor-red-neuronal/1`, `tools/golden_nn.py::package_model_json`):
//!   task, names, classes, architecture, activations, dropout, loss, metric,
//!   optimizer with the final learning rate, scaler (with `n_samples`), the dense
//!   layers (`name/units/activation/kernel/bias`, Keras names `dense_i` and
//!   `output_layer`), the app's `config` and a `check` with 5 raw rows and the output.
//! - `reconstruir_modelo.py` (`paquete_modelo_red.py.j2`): rebuilds the network in
//!   Keras, checks the outputs and saves .keras, .h5, SavedModel, .tflite and pickles.
//! - `predecir_numpy.py` (`red_predictor_numpy.py.j2`): the same network in NumPy.
//! - `LEEME.txt` (`paquete_leeme_red.txt.j2`).

use mltutor_core::codegen::{
    self, f32_as_decimal, CodeTemplate, NnPackageReadmeContext, NnPackageScriptContext,
    NnReadmeClass,
};
use mltutor_core::dataset::TaskType;
use mltutor_core::nn_data::NnArchitecture;
use mltutor_nn::NeuralNetwork;
use ndarray::ArrayView2;
use serde_json::{json, Value};

use super::{python_zip, MODEL_JSON, README};
use crate::ExportError;

/// File name of the ZIP.
pub const KERAS_ZIP_NAME: &str = "modelo_red.zip";
/// The Keras script inside the ZIP.
pub const KERAS_SCRIPT: &str = "reconstruir_modelo.py";
/// The NumPy predictor inside the ZIP (also the "📝 Código Python" download,
/// `red_predictor_numpy.py`).
pub const NUMPY_SCRIPT: &str = "predecir_numpy.py";
/// Suggested name of the NumPy predictor when downloaded alone.
pub const NUMPY_FILE_NAME: &str = "red_predictor_numpy.py";
/// Suggested name of the ONNX file.
pub const ONNX_FILE_NAME: &str = "red_neuronal.onnx";
/// Format tag of `modelo.json`.
pub const FORMAT: &str = "mltutor-red-neuronal/1";

/// What the package needs besides the network.
#[derive(Debug, Clone, Copy)]
pub struct KerasPackageInput<'a> {
    pub network: &'a NeuralNetwork,
    pub arch: &'a NnArchitecture,
    /// Selector label ("🌸 Iris - Clasificación de flores", "📄 datos.csv").
    pub dataset_label: &'a str,
    /// Readable feature names, in column order.
    pub feature_names: &'a [String],
    /// By class code (classification).
    pub class_names: Option<&'a [String]>,
    /// Learning rate at the end of training (after ReduceLROnPlateau).
    pub final_learning_rate: f32,
    /// The first test rows in original units (`check`, 5 in the app).
    pub x_check: ArrayView2<'a, f64>,
    /// Rows of the training split (fit + validation) and of the test split.
    pub n_train: usize,
    pub n_test: usize,
}

/// The four files of the package.
#[derive(Debug, Clone, PartialEq)]
pub struct KerasPackage {
    pub model_json: String,
    pub script: String,
    pub numpy_script: String,
    pub readme: String,
}

impl KerasPackage {
    /// The ZIP (deterministic).
    pub fn to_zip(&self) -> Result<Vec<u8>, ExportError> {
        python_zip(&[
            (MODEL_JSON, self.model_json.as_bytes()),
            (KERAS_SCRIPT, self.script.as_bytes()),
            (NUMPY_SCRIPT, self.numpy_script.as_bytes()),
            (README, self.readme.as_bytes()),
        ])
    }
}

fn floats(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// `modelo.json` as a value.
pub fn keras_model_json(input: &KerasPackageInput<'_>) -> Result<Value, ExportError> {
    let net = input.network;
    let mlp = &net.mlp;
    let classification = net.task == TaskType::Classification;
    let n_layers = mlp.layers.len();
    let layers: Vec<Value> = mlp
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let last = i + 1 == n_layers;
            let rows: Vec<Vec<f64>> = l.kernel.chunks(l.units.max(1)).map(floats).collect();
            json!({
                "name": if last { "output_layer".to_owned() } else { format!("dense_{}", i + 1) },
                "units": l.units,
                "activation": if last { mlp.output } else { mlp.hidden }.as_str(),
                "kernel": rows,
                "bias": floats(&l.bias),
            })
        })
        .collect();
    let architecture = mlp.architecture();
    let raw = net.raw_output(input.x_check)?;
    let check_x: Vec<Vec<f64>> = input
        .x_check
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();
    let check_out: Vec<Vec<f64>> = raw.rows().into_iter().map(|r| r.to_vec()).collect();
    let dropout = f32_as_decimal(input.arch.dropout);
    Ok(json!({
        "format": FORMAT,
        "task": if classification { "classification" } else { "regression" },
        "feature_names": input.feature_names,
        "class_names": if classification { input.class_names.map(<[String]>::to_vec) } else { None },
        "architecture": architecture,
        "activation": mlp.hidden.as_str(),
        "output_activation": mlp.output.as_str(),
        "dropout_rate": dropout,
        "loss": net.resolved.loss.as_str(),
        "metric": net.resolved.metric.key(),
        "optimizer": {"name": input.arch.optimizer.as_str(), "learning_rate": f64::from(input.final_learning_rate)},
        "scaler": {"mean": net.scaler.mean, "scale": net.scaler.scale, "n_samples": net.scaler.n_samples},
        "layers": layers,
        "config": {
            "architecture": architecture,
            "activation": mlp.hidden.as_str(),
            "output_activation": mlp.output.as_str(),
            "dropout_rate": dropout,
            "batch_size": input.arch.batch_size,
            "optimizer": input.arch.optimizer.as_str(),
            "task_type": if classification { "Clasificación" } else { "Regresión" },
            "input_size": mlp.n_inputs(),
            "output_size": mlp.n_outputs(),
        },
        "check": {"X": check_x, "output": check_out},
    }))
}

/// The four files.
pub fn keras_package(input: &KerasPackageInput<'_>) -> Result<KerasPackage, ExportError> {
    let doc = keras_model_json(input)?;
    let net = input.network;
    let classification = net.task == TaskType::Classification;
    let task = if classification {
        "classification"
    } else {
        "regression"
    };
    let script_ctx = NnPackageScriptContext {
        dataset: input.dataset_label.to_owned(),
        task: task.to_owned(),
        json_name: MODEL_JSON.to_owned(),
    };
    let classes = classification.then(|| {
        input
            .class_names
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(value, name)| NnReadmeClass {
                value,
                name: name.clone(),
            })
            .collect()
    });
    let readme = codegen::render(
        CodeTemplate::NnPackageReadme,
        &NnPackageReadmeContext {
            dataset: input.dataset_label.to_owned(),
            task: task.to_owned(),
            architecture: net.mlp.architecture(),
            activation: net.mlp.hidden.as_str().to_owned(),
            output_activation: net.mlp.output.as_str().to_owned(),
            dropout_rate: f32_as_decimal(input.arch.dropout),
            n_params: net.mlp.n_params(),
            feature_names: input.feature_names.to_vec(),
            classes,
            n_train: input.n_train,
            n_test: input.n_test,
        },
    )?;
    let mut model_json = serde_json::to_string_pretty(&doc).map_err(|e| ExportError::Json {
        message: e.to_string(),
    })?;
    model_json.push('\n');
    Ok(KerasPackage {
        model_json,
        script: codegen::render(CodeTemplate::NnPackageScript, &script_ctx)?,
        numpy_script: codegen::render(CodeTemplate::NnNumpyPredictor, &script_ctx)?,
        readme,
    })
}
