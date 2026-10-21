//! 📦 Paquete Keras (.zip) of the 🧠 network (C28, D75): what replaces Python's
//! SavedModel/HDF5/TFLite/pickle exports, which need TensorFlow.
//!
//! - `modelo.json` (`mltutor-red-neuronal/1`, `tools/golden_nn.py::package_model_json`,
//!   written compactly by [`model_json_text`]):
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
use mltutor_core::model::Predictor;
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
/// "📝 Código Python" as a ZIP: the NumPy predictor reads `modelo.json` (it has
/// the weights), so the script alone cannot run; [`KerasPackage::numpy_zip`] puts
/// both files together.
pub const NUMPY_ZIP_NAME: &str = "red_predictor_numpy.zip";
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

    /// The NumPy predictor with the `modelo.json` it reads ([`NUMPY_ZIP_NAME`];
    /// deterministic). `numpy_script` alone ([`NUMPY_FILE_NAME`]) only runs next to
    /// a `modelo.json`, e.g. inside the Keras package.
    pub fn numpy_zip(&self) -> Result<Vec<u8>, ExportError> {
        python_zip(&[
            (NUMPY_SCRIPT, self.numpy_script.as_bytes()),
            (MODEL_JSON, self.model_json.as_bytes()),
        ])
    }
}

/// A float32 as the f64 of its shortest decimal (`0.1f32` → `0.1`, not
/// `0.10000000149011612`), so that `modelo.json` is about three times smaller and
/// still reads back as exactly the same float32 (checked; otherwise the exact value
/// is kept).
fn f32_short(x: f32) -> f64 {
    match format!("{x}").parse::<f64>() {
        Ok(d) if d as f32 == x || (d.is_nan() && x.is_nan()) => d,
        _ => f64::from(x),
    }
}

fn floats(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f32_short(x)).collect()
}

fn not_finite(what: &str) -> ExportError {
    ExportError::InvalidModel {
        reason: format!("{what} no finitos"),
    }
}

/// `modelo.json` as a value.
pub fn keras_model_json(input: &KerasPackageInput<'_>) -> Result<Value, ExportError> {
    let net = input.network;
    crate::check_network(net)?;
    let mlp = net.mlp();
    let scaler = net.scaler();
    let classification = net.task() == TaskType::Classification;
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
    if input.x_check.iter().any(|v| !v.is_finite()) {
        return Err(not_finite("datos de comprobación"));
    }
    let raw = net.raw_output(input.x_check)?;
    if raw.iter().any(|v| !v.is_finite()) {
        return Err(not_finite("resultados de comprobación"));
    }
    let check_x: Vec<Vec<f64>> = input
        .x_check
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();
    // The outputs come from the float32 forward: shortest float32 decimal too.
    let check_out: Vec<Vec<f64>> = raw
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|&v| f32_short(v as f32)).collect())
        .collect();
    let dropout = f32_as_decimal(input.arch.dropout);
    let resolved = net.resolved();
    Ok(json!({
        "format": FORMAT,
        "task": if classification { "classification" } else { "regression" },
        "feature_names": input.feature_names,
        "class_names": if classification { input.class_names.map(<[String]>::to_vec) } else { None },
        "architecture": architecture,
        "activation": mlp.hidden.as_str(),
        "output_activation": mlp.output.as_str(),
        "dropout_rate": dropout,
        "loss": resolved.loss.as_str(),
        "metric": resolved.metric.key(),
        "optimizer": {"name": input.arch.optimizer.as_str(), "learning_rate": f64::from(input.final_learning_rate)},
        "scaler": {"mean": scaler.mean, "scale": scaler.scale, "n_samples": scaler.n_samples},
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

/// `modelo.json` as text: indented like `json.dumps(indent=2)` except that arrays of
/// numbers or strings go on one line, so a kernel takes one line per row (a 5 × 256
/// network: about 3 MB instead of 10 MB).
pub fn model_json_text(doc: &Value) -> Result<String, ExportError> {
    let mut out = String::new();
    write_value(doc, 0, &mut out)?;
    out.push('\n');
    Ok(out)
}

fn scalar(v: &Value) -> bool {
    !matches!(v, Value::Array(_) | Value::Object(_))
}

fn write_value(v: &Value, indent: usize, out: &mut String) -> Result<(), ExportError> {
    let pad = |n: usize| "  ".repeat(n);
    match v {
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Array(items) if items.iter().all(scalar) => {
            out.push_str(&compact(v)?);
        }
        Value::Array(items) => {
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                out.push_str(&pad(indent + 1));
                write_value(item, indent + 1, out)?;
                out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad(indent));
            out.push(']');
        }
        Value::Object(map) if map.is_empty() => out.push_str("{}"),
        Value::Object(map) => {
            out.push_str("{\n");
            for (i, (k, item)) in map.iter().enumerate() {
                out.push_str(&pad(indent + 1));
                out.push_str(&compact(&Value::String(k.clone()))?);
                out.push_str(": ");
                write_value(item, indent + 1, out)?;
                out.push_str(if i + 1 < map.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad(indent));
            out.push('}');
        }
        _ => out.push_str(&compact(v)?),
    }
    Ok(())
}

fn compact(v: &Value) -> Result<String, ExportError> {
    serde_json::to_string(v).map_err(|e| ExportError::Json {
        message: e.to_string(),
    })
}

/// The four files.
pub fn keras_package(input: &KerasPackageInput<'_>) -> Result<KerasPackage, ExportError> {
    let doc = keras_model_json(input)?;
    let net = input.network;
    let classification = net.task() == TaskType::Classification;
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
            architecture: net.mlp().architecture(),
            activation: net.mlp().hidden.as_str().to_owned(),
            output_activation: net.mlp().output.as_str().to_owned(),
            dropout_rate: f32_as_decimal(input.arch.dropout),
            n_params: net.mlp().n_params(),
            feature_names: input.feature_names.to_vec(),
            classes,
            n_train: input.n_train,
            n_test: input.n_test,
        },
    )?;
    let model_json = model_json_text(&doc)?;
    Ok(KerasPackage {
        model_json,
        script: codegen::render(CodeTemplate::NnPackageScript, &script_ctx)?,
        numpy_script: codegen::render(CodeTemplate::NnNumpyPredictor, &script_ctx)?,
        readme,
    })
}
