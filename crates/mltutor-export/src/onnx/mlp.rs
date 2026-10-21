//! ONNX of the 🧠 network (C28, D75): the graph of
//! `tools/golden_nn.py::onnx_proposal_model`, node for node.
//!
//! `float_input` [N, F] float32 in **original units** → `Sub(scaler_mean)` →
//! `Div(scaler_scale)` → per layer `Gemm(W{i}, B{i})` + `Relu | Tanh | Sigmoid |
//! Softmax(axis=1) | Identity`. Classification: outputs `label` (int64) and
//! `probabilities` (float32); regression: `variable`. IR 8, opset 15, unique node
//! names.
//!
//! - `label` is the **dataset code** of the class (review F5a, finding 9):
//!   `Gather(class_labels, index)` over `ArgMax(axis=1, keepdims=0)`, or with one
//!   neuron over `Greater(0.5) → Cast(int64) → Squeeze(axes=[1])` (the axes as an
//!   input since opset 13). `class_labels` = [`NeuralNetwork::class_codes`].
//! - `probabilities`: the network output with several neurons; with one neuron
//!   (output `positive_probability`) `Concat(1 − p, p)` [N, 2], like the KNN and tree
//!   ONNX and `predict_proba`.
//!
//! The scaler is applied in float32 (the app scales in float64 before converting):
//! with large, off-centre columns (years, timestamps) the outputs can differ in the
//! last digits.

use mltutor_core::dataset::TaskType;
use mltutor_core::model::Predictor;
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
    crate::check_network(net)?;
    let mlp = net.mlp();
    let scaler = net.scaler();
    let f = mlp.n_inputs();
    let as_f32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let fi = to_i64(f)?;
    let mut inits = vec![
        f32_tensor("scaler_mean", &[fi], &as_f32(&scaler.mean)),
        f32_tensor("scaler_scale", &[fi], &as_f32(&scaler.scale)),
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
    let classification = net.task() == TaskType::Classification;
    let k = mlp.n_outputs();
    let binary = classification && k == 1;
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
        } else if !classification {
            "variable".to_owned()
        } else if binary {
            "positive_probability".to_owned()
        } else {
            "probabilities".to_owned()
        };
        let attrs = if act == Activation::Softmax {
            vec![int_attr("axis", 1)]
        } else {
            vec![]
        };
        nodes.push(node(op_of(act), &format!("act_{i}"), &[&z], &[&out], attrs));
        prev = out;
    }
    if !classification {
        return Ok(finish(
            nodes,
            inits,
            fi,
            vec![value_info(
                &prev,
                ELEM_FLOAT,
                &[Dim::Unknown, Dim::Fixed(to_i64(k)?)],
            )],
        ));
    }
    let codes: Vec<i64> = net
        .class_codes()
        .iter()
        .map(|&c| to_i64(c))
        .collect::<Result<_, _>>()?;
    let n_classes = to_i64(codes.len())?;
    inits.push(i64_tensor("class_labels", &[n_classes], &codes));
    if !binary {
        nodes.push(node(
            "ArgMax",
            "argmax",
            &[&prev],
            &["label_index"],
            vec![int_attr("axis", 1), int_attr("keepdims", 0)],
        ));
    } else {
        inits.push(f32_tensor("threshold", &[], &[0.5]));
        inits.push(i64_tensor("squeeze_axes", &[1], &[1]));
        inits.push(f32_tensor("one", &[], &[1.0]));
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
            &["label_index"],
            vec![],
        ));
    }
    nodes.push(node(
        "Gather",
        "gather_label",
        &["class_labels", "label_index"],
        &["label"],
        vec![int_attr("axis", 0)],
    ));
    if binary {
        nodes.push(node(
            "Sub",
            "proba_negative",
            &["one", &prev],
            &["negative_probability"],
            vec![],
        ));
        nodes.push(node(
            "Concat",
            "proba_concat",
            &["negative_probability", &prev],
            &["probabilities"],
            vec![int_attr("axis", 1)],
        ));
    }
    Ok(finish(
        nodes,
        inits,
        fi,
        vec![
            value_info("label", ELEM_INT64, &[Dim::Unknown]),
            value_info(
                "probabilities",
                ELEM_FLOAT,
                &[Dim::Unknown, Dim::Fixed(n_classes)],
            ),
        ],
    ))
}

fn finish(
    node: Vec<Node>,
    initializer: Vec<Tensor>,
    fi: i64,
    output: Vec<super::ValueInfo>,
) -> Model {
    model(Graph {
        node,
        name: "mltutor_mlp".to_owned(),
        initializer,
        input: vec![value_info(
            INPUT_NAME,
            ELEM_FLOAT,
            &[Dim::Unknown, Dim::Fixed(fi)],
        )],
        output,
    })
}
