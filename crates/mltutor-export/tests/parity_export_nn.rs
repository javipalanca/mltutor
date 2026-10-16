//! F5 parity (C28, D75): the network's ONNX graph and its Keras package against
//! `tests/parity/nn/exact/*.json` (`onnx_proposal_D75`, `export_package`), built
//! from Keras' final weights and the fixture's scaler.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "../../mltutor-nn/tests/common/mod.rs"]
mod common;

use common::*;
use mltutor_core::dataset::TaskType;
use mltutor_core::nn_data::{output_size, resolve_training, Scaler};
use mltutor_export::onnx::{self, Model};
use mltutor_export::package::keras::{keras_package, KerasPackageInput};
use mltutor_nn::NeuralNetwork;
use ndarray::Axis;
use serde_json::Value;

struct Loaded {
    fx: Value,
    net: NeuralNetwork,
    ds: mltutor_core::Dataset,
    test: Vec<usize>,
    arch: mltutor_nn::NnArchitecture,
}

fn load(name: &str) -> Loaded {
    let fx = fixture(&format!("nn/exact/{name}.json"));
    let (arch, _) = config(&fx);
    let ds = dataset(&fx["input"]);
    let e = &fx["expected"];
    let task = if fx["input"]["nn_config"]["task_type"] == "Clasificación" {
        TaskType::Classification
    } else {
        TaskType::Regression
    };
    let n_classes = ds.target.n_classes().unwrap_or(0);
    let k = output_size(task, n_classes);
    let net = NeuralNetwork {
        mlp: mlp_of(&e["final_weights"], arch.activation, arch.output_activation),
        scaler: Scaler {
            mean: nums(&e["scaler"]["mean"]),
            scale: nums(&e["scaler"]["scale"]),
            n_samples: ds.n_rows(),
        },
        task,
        classes: (0..n_classes).collect(),
        resolved: resolve_training(task, arch.output_activation, k).unwrap(),
    };
    let codes = ds.target.class_codes();
    let split =
        mltutor_core::nn_data::lesson_split(task, codes.as_deref(), &[], ds.n_rows()).unwrap();
    Loaded {
        fx,
        net,
        ds,
        test: split.test,
        arch,
    }
}

/// A float32 interpreter of the MLP graph (the operators `mlp_model` writes).
fn run(model: &Model, x: &[f32], n: usize) -> (Option<Vec<i64>>, Vec<f32>, usize) {
    use std::collections::HashMap;
    let g = model.graph.as_ref().unwrap();
    let mut t: HashMap<String, (Vec<f32>, usize)> = HashMap::new();
    let mut ints: HashMap<String, Vec<i64>> = HashMap::new();
    for init in &g.initializer {
        if init.data_type == onnx::ELEM_FLOAT {
            let v: Vec<f32> = init
                .raw_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let cols = init.dims.last().copied().unwrap_or(1).max(1) as usize;
            t.insert(
                init.name.clone(),
                (
                    v,
                    if init.dims.len() == 2 {
                        cols
                    } else {
                        usize::MAX
                    },
                ),
            );
        }
    }
    let f = x.len() / n;
    t.insert(onnx::INPUT_NAME.to_owned(), (x.to_vec(), f));
    for node in &g.node {
        let inp = |i: usize| t[&node.input[i]].clone();
        match node.op_type.as_str() {
            "Sub" | "Div" => {
                let (a, cols) = inp(0);
                let (b, _) = inp(1);
                let out: Vec<f32> = a
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        if node.op_type == "Sub" {
                            v - b[i % cols]
                        } else {
                            v / b[i % cols]
                        }
                    })
                    .collect();
                t.insert(node.output[0].clone(), (out, cols));
            }
            "Gemm" => {
                let (a, k) = inp(0);
                let (w, m) = inp(1);
                let (b, _) = inp(2);
                let rows = a.len() / k;
                let mut out = vec![0.0_f32; rows * m];
                for r in 0..rows {
                    for j in 0..m {
                        let mut s = 0.0_f64;
                        for kk in 0..k {
                            s += f64::from(a[r * k + kk]) * f64::from(w[kk * m + j]);
                        }
                        out[r * m + j] = s as f32 + b[j];
                    }
                }
                t.insert(node.output[0].clone(), (out, m));
            }
            "Relu" | "Tanh" | "Sigmoid" | "Identity" | "Softmax" => {
                let (mut a, cols) = inp(0);
                match node.op_type.as_str() {
                    "Relu" => a.iter_mut().for_each(|v| *v = v.max(0.0)),
                    "Tanh" => a.iter_mut().for_each(|v| *v = v.tanh()),
                    "Sigmoid" => a.iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp())),
                    "Softmax" => {
                        for row in a.chunks_mut(cols) {
                            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let s: f32 = row.iter().map(|v| (v - m).exp()).sum();
                            row.iter_mut().for_each(|v| *v = (*v - m).exp() / s);
                        }
                    }
                    _ => {}
                }
                t.insert(node.output[0].clone(), (a, cols));
            }
            "ArgMax" => {
                let (a, cols) = inp(0);
                ints.insert(
                    node.output[0].clone(),
                    a.chunks(cols)
                        .map(|r| mltutor_nn::loss::argmax(r) as i64)
                        .collect(),
                );
            }
            "Greater" => {
                let (a, _) = inp(0);
                let (b, _) = inp(1);
                ints.insert(
                    node.output[0].clone(),
                    a.iter().map(|v| i64::from(*v > b[0])).collect(),
                );
            }
            "Cast" | "Squeeze" => {
                let v = ints[&node.input[0]].clone();
                ints.insert(node.output[0].clone(), v);
            }
            other => panic!("operator {other}"),
        }
    }
    let outs: Vec<String> = g.output.iter().map(|o| o.name.clone()).collect();
    let main = outs.iter().find(|o| t.contains_key(*o)).unwrap();
    let (v, cols) = t[main].clone();
    (ints.get("label").cloned(), v, cols)
}

fn attrs_json(node: &onnx::Node) -> Value {
    let mut m = serde_json::Map::new();
    for a in &node.attribute {
        m.insert(a.name.clone(), Value::from(a.i));
    }
    Value::Object(m)
}

#[test]
fn onnx_graph_and_outputs() {
    let mut worst = 0.0_f64;
    for &name in EXACT {
        let l = load(name);
        let e = &l.fx["expected"];
        let p = &e["onnx_proposal_D75"];
        let model = mltutor_export::onnx_model(&l.net, None).unwrap();
        assert_eq!(mltutor_export::onnx_file_name(&l.net), "red_neuronal.onnx");
        let g = model.graph.as_ref().unwrap();
        let theirs = p["nodes"].as_array().unwrap();
        assert_eq!(g.node.len(), theirs.len(), "{name}");
        for (a, b) in g.node.iter().zip(theirs) {
            assert_eq!(a.op_type, b["op"].as_str().unwrap(), "{name}");
            assert_eq!(a.name, b["name"].as_str().unwrap(), "{name}");
            assert_eq!(
                serde_json::to_value(&a.input).unwrap(),
                b["inputs"],
                "{name}"
            );
            assert_eq!(
                serde_json::to_value(&a.output).unwrap(),
                b["outputs"],
                "{name}"
            );
            assert_eq!(attrs_json(a), b["attributes"], "{name} {}", a.name);
        }
        let outs: Vec<&str> = g.output.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(serde_json::to_value(&outs).unwrap(), p["outputs"], "{name}");
        assert_eq!(model.ir_version, 8);
        // Raw test rows (original units) → test.raw_output.
        let x_test = l.ds.features.select(Axis(0), &l.test);
        let x: Vec<f32> = x_test.iter().map(|&v| v as f32).collect();
        let (labels, out, _) = run(
            &Model::from_bytes(&model.to_bytes()).unwrap(),
            &x,
            l.test.len(),
        );
        let raw = flat(&e["test"]["raw_output"]);
        let scale = raw.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for (a, b) in out.iter().zip(&raw) {
            let d = (f64::from(*a) - b).abs();
            assert!(d <= 1e-5 * scale, "{name}: {a} vs {b}");
            worst = worst.max(d / scale);
        }
        if let Some(labels) = labels {
            let yp: Vec<i64> = e["test"]["y_pred"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            assert_eq!(labels, yp, "{name}: labels");
        }
    }
    eprintln!("ONNX: max |Δ| / scale vs Keras = {worst:e}");
}

/// Writes the ONNX of every case to `$MLTUTOR_NN_ONNX_DIR` for the onnxruntime and
/// `onnx.checker.check_model(full_check=True)` validation (manual, F6 in CI).
#[test]
#[ignore = "writes files: MLTUTOR_NN_ONNX_DIR=… cargo test -- --ignored"]
fn write_onnx_for_onnxruntime() {
    let dir = std::env::var("MLTUTOR_NN_ONNX_DIR").expect("MLTUTOR_NN_ONNX_DIR");
    for &name in EXACT {
        let l = load(name);
        let model = mltutor_export::onnx_model(&l.net, None).unwrap();
        std::fs::write(format!("{dir}/{name}.onnx"), model.to_bytes()).unwrap();
        let x_test = l.ds.features.select(Axis(0), &l.test);
        let rows: Vec<Vec<f64>> = x_test.rows().into_iter().map(|r| r.to_vec()).collect();
        std::fs::write(format!("{dir}/{name}.json"), serde_json::to_string(&serde_json::json!({
            "x": rows, "raw_output": l.fx["expected"]["test"]["raw_output"], "y_pred": l.fx["expected"]["test"]["y_pred"],
        })).unwrap()).unwrap();
    }
}

/// Floats within 1e-5 (the check outputs come from the float32 forward), the rest exact.
fn same_json(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!(
                (x - y).abs() <= 1e-5 * y.abs().max(1.0),
                "{path}: {x} vs {y}"
            );
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}");
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                same_json(u, v, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            let kx: Vec<&String> = x.keys().collect();
            let ky: Vec<&String> = y.keys().collect();
            assert_eq!(kx, ky, "{path}: keys");
            for (k, v) in x {
                same_json(v, &y[k], &format!("{path}.{k}"));
            }
        }
        _ => assert_eq!(a, b, "{path}"),
    }
}

#[test]
fn keras_package_matches() {
    let mut n = 0;
    for &name in EXACT {
        let l = load(name);
        let e = &l.fx["expected"];
        let pkg = &e["export_package"];
        if !pkg.is_object() {
            continue;
        }
        let m = &pkg["model_json"];
        let feature_names = strings_of(&m["feature_names"]);
        let class_names = m["class_names"]
            .is_array()
            .then(|| strings_of(&m["class_names"]));
        let x_test = l.ds.features.select(Axis(0), &l.test);
        let check = x_test.slice(ndarray::s![..5, ..]);
        let input = KerasPackageInput {
            network: &l.net,
            arch: &l.arch,
            dataset_label: l.fx["input"]["app_label"].as_str().unwrap(),
            feature_names: &feature_names,
            class_names: class_names.as_deref(),
            final_learning_rate: m["optimizer"]["learning_rate"].as_f64().unwrap() as f32,
            x_check: check,
            n_train: l.ds.n_rows() - l.test.len(),
            n_test: l.test.len(),
        };
        let out = keras_package(&input).unwrap();
        let ours: Value = serde_json::from_str(&out.model_json).unwrap();
        same_json(&ours, m, name);
        assert_eq!(
            out.script,
            pkg["script"].as_str().unwrap(),
            "{name}: script"
        );
        assert_eq!(
            out.numpy_script,
            pkg["numpy_script"].as_str().unwrap(),
            "{name}: numpy"
        );
        assert_eq!(out.readme, pkg["leeme"].as_str().unwrap(), "{name}: LEEME");
        let zip = out.to_zip().unwrap();
        assert_eq!(zip, out.to_zip().unwrap());
        n += 1;
    }
    assert!(n >= 9, "{n}");
}

fn strings_of(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_owned())
        .collect()
}
