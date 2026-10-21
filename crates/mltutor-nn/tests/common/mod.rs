//! Loading of `tests/parity/nn/**` fixtures for the network tests.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use base64::Engine;
use mltutor_core::csv::read_csv;
use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::{Dataset, TaskChoice};
use mltutor_nn::{Activation, Dense, Mlp, NnArchitecture, NnTraining, OptimizerKind};
use serde_json::Value;

pub const EXACT: &[&str] = &[
    "breast_cancer_default",
    "breast_cancer_sgd_tanh_output",
    "california_housing_default_e20",
    "csv_numeric_regression_default",
    "csv_string_target_default",
    "diabetes_adam_lr01_b32",
    "diabetes_default",
    "diabetes_sgd_lr001_tanh",
    "diabetes_sigmoid_output",
    "diabetes_softmax_output",
    "iris_default",
    "iris_relu_3layers_lr03",
    "iris_rmsprop_sigmoid_1layer",
    "iris_sgd_tanh_lr01_b16",
    "iris_tanh_output",
    "moons_default",
    "wine_default",
    "wine_rmsprop_lr03_chaotic",
];

pub fn parity_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity")
}

pub fn fixture(rel: &str) -> Value {
    let path = parity_dir().join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// A number (`"nan"`, `"inf"` strings as in the fixtures).
pub fn num(v: &Value) -> f64 {
    match v {
        Value::String(s) if s == "nan" => f64::NAN,
        Value::String(s) if s == "inf" => f64::INFINITY,
        Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
        _ => v.as_f64().unwrap_or(f64::NAN),
    }
}

pub fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

/// A matrix (list of rows) flattened row-major.
pub fn flat(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|r| if r.is_array() { nums(r) } else { vec![num(r)] })
        .collect()
}

/// An `_array_block` (values or base64 float32), flattened.
pub fn block(v: &Value) -> Vec<f32> {
    if let Some(b) = v["base64_f32_le"].as_str() {
        let bytes = base64::engine::general_purpose::STANDARD.decode(b).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        flat(&v["values"]).into_iter().map(|x| x as f32).collect()
    }
}

/// The dataset of a fixture `input`.
pub fn dataset(input: &Value) -> Dataset {
    if let Some(rel) = input["dataset"].as_str() {
        let slug = rel
            .trim_start_matches("datasets/")
            .trim_end_matches(".json");
        load_builtin(DatasetId::from_slug(slug).unwrap()).unwrap()
    } else {
        let csv_fx = fixture(input["csv_fixture"].as_str().unwrap());
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(csv_fx["input"]["csv_base64"].as_str().unwrap())
            .unwrap();
        let table = read_csv(&bytes).unwrap();
        let file = input["app_label"]
            .as_str()
            .unwrap()
            .trim_start_matches("📄 ");
        table
            .to_dataset(input["target"].as_str().unwrap(), TaskChoice::Auto, file)
            .unwrap()
    }
}

pub fn activation(v: &Value) -> Activation {
    Activation::parse(v.as_str().unwrap()).unwrap()
}

/// Keras weights (`[{kernel, bias}]`) as an [`Mlp`].
pub fn mlp_of(weights: &Value, hidden: Activation, output: Activation) -> Mlp {
    let layers = weights
        .as_array()
        .unwrap()
        .iter()
        .map(|l| {
            let rows = l["kernel"].as_array().unwrap();
            let inputs = rows.len();
            let kernel: Vec<f32> = flat(&l["kernel"]).into_iter().map(|x| x as f32).collect();
            let bias: Vec<f32> = nums(&l["bias"]).into_iter().map(|x| x as f32).collect();
            Dense {
                inputs,
                units: bias.len(),
                kernel,
                bias,
            }
        })
        .collect();
    Mlp {
        layers,
        hidden,
        output,
    }
}

/// `(architecture, training)` of an `nn/exact` case (dropout 0, its epochs).
pub fn config(fx: &Value) -> (NnArchitecture, NnTraining) {
    let input = &fx["input"];
    let cfg = &input["nn_config"];
    let c = &input["controls"];
    let arch_sizes: Vec<u16> = cfg["architecture"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u16)
        .collect();
    let arch = NnArchitecture {
        hidden: arch_sizes[1..arch_sizes.len() - 1].to_vec(),
        activation: activation(&cfg["activation"]),
        output_activation: activation(&cfg["output_activation"]),
        dropout: cfg["dropout_rate"].as_f64().unwrap() as f32,
        batch_size: cfg["batch_size"].as_u64().unwrap() as u16,
        optimizer: OptimizerKind::parse(cfg["optimizer"].as_str().unwrap()).unwrap(),
    };
    let training = NnTraining {
        learning_rate: c["lr"].as_f64().unwrap() as f32,
        epochs: input["fit_kwargs"]["epochs"].as_u64().unwrap() as u16,
        validation_pct: c["val"].as_u64().unwrap() as u8,
        patience: c["es"]
            .as_bool()
            .unwrap()
            .then(|| c["patience"].as_u64().unwrap() as u8),
        lr_factor: c["rlr"]
            .as_bool()
            .unwrap()
            .then(|| c["factor"].as_f64().unwrap() as f32),
        seed: 42,
    };
    (arch, training)
}

/// Largest absolute difference between our layers and Keras' `[{kernel, bias}]`.
pub fn max_weight_diff(mlp: &Mlp, weights: &Value) -> f64 {
    let theirs = mlp_of(weights, mlp.hidden, mlp.output);
    let mut m = 0.0_f64;
    for (a, b) in mlp.layers.iter().zip(&theirs.layers) {
        for (x, y) in a
            .kernel
            .iter()
            .zip(&b.kernel)
            .chain(a.bias.iter().zip(&b.bias))
        {
            m = m.max((f64::from(*x) - f64::from(*y)).abs());
        }
    }
    m
}
