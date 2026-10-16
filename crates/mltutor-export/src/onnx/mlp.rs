//! ONNX of the 🧠 network (C28, D75): the graph of
//! `tools/golden_nn.py::onnx_proposal_model`, node for node.
//!
//! `float_input` [N, F] float32 in **original units** → `Sub(scaler_mean)` →
//! `Div(scaler_scale)` → per layer `Gemm(W{i}, B{i})` + `Relu | Tanh | Sigmoid |
//! Softmax(axis=1) | Identity`. Classification: outputs `label` (int64: `ArgMax(axis=1,
//! keepdims=0)`, or with one neuron `Greater(0.5) → Cast(int64) → Squeeze(axes=[1])`,
//! the axes as an input since opset 13) and `probabilities` (the network output);
//! regression: `variable`. The label is the class **index** (the app's codes when
//! every class reached the training set). IR 8, opset 15, unique node names.

use mltutor_core::dataset::TaskType;
use mltutor_core::nn_data::Activation;
use mltutor_nn::NeuralNetwork;

use super::knn::{i64_tensor, int_attr, node, to_i64, value_info, Dim};
use super::{model, Graph, Model, Node, Tensor, ELEM_FLOAT, ELEM_INT64, INPUT_NAME};
use crate::ExportError;

fn f32_tensor(name: &str, dims: &[i64], values: &[f32]) -> Tensor {
    Tensor {
        dims: dims.to_vec(),
        data_type: ELEM_FLOAT,
        name: name.to_owned(),
        raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn op_of(act: Activation) -> &'static str {
    match act {
        Activation::Relu => "Relu",
        Activation::Tanh => "Tanh",
        Activation::Sigmoid => "Sigmoid",
        Activation::Softmax => "Softmax",
        Activation::Linear => "Identity",
    }
}

/// The ONNX graph of a trained network (scaler included).
pub fn mlp_model(net: &NeuralNetwork) -> Result<Model, ExportError> {
    let mlp = &net.mlp;
    let f = mlp.n_inputs();
    if net.scaler.mean.len() != f || net.scaler.scale.len() != f || mlp.layers.is_empty() {
        return Err(ExportError::InvalidModel {
            reason: "la red y el escalador no coinciden".to_owned(),
        });
    }
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
    if mlp
        .layers
        .iter()
        .any(|l| !finite(&l.kernel) || !finite(&l.bias))
    {
        return Err(ExportError::InvalidModel {
            reason: "pesos no finitos".to_owned(),
        });
    }
    let as_f32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let fi = to_i64(f)?;
    let mut inits = vec![
        f32_tensor("scaler_mean", &[fi], &as_f32(&net.scaler.mean)),
        f32_tensor("scaler_scale", &[fi], &as_f32(&net.scaler.scale)),
    ];
    let mut nodes: Vec<Node> = vec![
        node(
            "Sub",
            "scaler_sub",
            &[INPUT_NAME, "scaler_mean"],
            &["centred"],
            vec![],
        ),
        node(
            "Div",
            "scaler_div",
            &["centred", "scaler_scale"],
            &["scaled"],
            vec![],
        ),
    ];
    let classification = net.task == TaskType::Classification;
    let n_layers = mlp.layers.len();
    let mut prev = "scaled".to_owned();
    for (i, l) in mlp.layers.iter().enumerate() {
        let (w, b, z) = (format!("W{i}"), format!("B{i}"), format!("z{i}"));
        inits.push(f32_tensor(
            &w,
            &[to_i64(l.inputs)?, to_i64(l.units)?],
            &l.kernel,
        ));
        inits.push(f32_tensor(&b, &[to_i64(l.units)?], &l.bias));
        nodes.push(node(
            "Gemm",
            &format!("gemm_{i}"),
            &[&prev, &w, &b],
            &[&z],
            vec![],
        ));
        let last = i + 1 == n_layers;
        let act = if last { mlp.output } else { mlp.hidden };
        let out = if !last {
            format!("a{i}")
        } else if classification {
            "probabilities".to_owned()
        } else {
            "variable".to_owned()
        };
        let attrs = if act == Activation::Softmax {
            vec![int_attr("axis", 1)]
        } else {
            vec![]
        };
        nodes.push(node(op_of(act), &format!("act_{i}"), &[&z], &[&out], attrs));
        prev = out;
    }
    let k = mlp.n_outputs();
    let mut outputs = vec![value_info(
        &prev,
        ELEM_FLOAT,
        &[Dim::Unknown, Dim::Fixed(to_i64(k)?)],
    )];
    if classification {
        if k > 1 {
            nodes.push(node(
                "ArgMax",
                "argmax",
                &[&prev],
                &["label"],
                vec![int_attr("axis", 1), int_attr("keepdims", 0)],
            ));
        } else {
            inits.push(f32_tensor("threshold", &[], &[0.5]));
            inits.push(i64_tensor("squeeze_axes", &[1], &[1]));
            nodes.push(node(
                "Greater",
                "greater",
                &[&prev, "threshold"],
                &["is_positive"],
                vec![],
            ));
            nodes.push(node(
                "Cast",
                "cast",
                &["is_positive"],
                &["label_2d"],
                vec![int_attr("to", i64::from(ELEM_INT64))],
            ));
            nodes.push(node(
                "Squeeze",
                "squeeze",
                &["label_2d", "squeeze_axes"],
                &["label"],
                vec![],
            ));
        }
        outputs.insert(0, value_info("label", ELEM_INT64, &[Dim::Unknown]));
    }
    Ok(model(Graph {
        node: nodes,
        name: "mltutor_mlp".to_owned(),
        initializer: inits,
        input: vec![value_info(
            INPUT_NAME,
            ELEM_FLOAT,
            &[Dim::Unknown, Dim::Fixed(fi)],
        )],
        output: outputs,
    }))
}
