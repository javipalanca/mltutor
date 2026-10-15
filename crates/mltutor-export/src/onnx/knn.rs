//! ONNX of a KNN (D57, fase4-knn.md §5): the graph `tools/golden_knn.py::
//! _onnx_proposal_model` builds, checked with `onnx.checker`, `onnx.reference` and
//! onnxruntime.
//!
//! There is no ONNX operator for nearest neighbours, so the **training rows go into
//! the graph** (`train_X`, float64) and the search is spelled out:
//!
//! 1. `Cast(float_input → double)`: the input is float32 like every MLTutor export;
//!    distances are computed in float64 like scikit-learn.
//! 2. `Scan` over the query rows. Its body takes one row and returns the K smallest
//!    reduced distances and their indices: `Sub`, `Mul` (Euclidean) or `Abs`
//!    (Manhattan), `ReduceSum` and `TopK(largest=0, sorted=1)`. **TopK keeps the lower
//!    index on ties**, scikit-learn's brute force rule; the kd-tree keeps the first
//!    point it visits, so labels can differ only when the K-th neighbour is tied
//!    (Titanic: 2 rows).
//! 3. `Sqrt` (Euclidean) or `Identity`.
//! 4. Weights: `uniform` → `ConstantOfShape(1)`; `distance` → `1/d`, and on the rows
//!    where some distance is 0, scikit-learn's rule (only those weigh 1) with
//!    `Equal` + `Cast` + `ReduceMax` + `Where`.
//! 5. Classifier: the labels of the neighbours (`Gather`), one-hot with `Equal`
//!    against `0..C` (onnxruntime has no `OneHot` with double output), the weighted
//!    votes (`Mul`, `ReduceSum`), `probabilities = votes / Σ votes` (float32) and
//!    `label = classes[ArgMax(votes)]` (int64; the first maximum = the smallest
//!    class, like scikit-learn). Regressor: `variable = Σ w·y / Σ w` (float32).
//!
//! IR version 8, opset 15 (no `ai.onnx.ml` operators). Every node has a unique name
//! (onnxruntime rejects duplicates, MEMORY.md).

use mltutor_core::knn::{EffectiveMetric, Knn, KnnWeights};
use mltutor_core::model::Predictor;

use super::{
    model, Attribute, AttributeType, DimValue, Dimension, Graph, Model, Node, Tensor, TensorShape,
    TensorType, TypeProto, ValueInfo, ELEM_BOOL, ELEM_DOUBLE, ELEM_FLOAT, ELEM_INT64, INPUT_NAME,
};
use crate::ExportError;

fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str], attrs: Vec<Attribute>) -> Node {
    Node {
        input: inputs.iter().map(|s| (*s).to_owned()).collect(),
        output: outputs.iter().map(|s| (*s).to_owned()).collect(),
        name: name.to_owned(),
        op_type: op.to_owned(),
        attribute: attrs,
        domain: String::new(),
    }
}

fn int_attr(name: &str, v: i64) -> Attribute {
    Attribute {
        name: name.to_owned(),
        i: v,
        r#type: AttributeType::Int as i32,
        ..Attribute::default()
    }
}

fn ints_attr(name: &str, v: Vec<i64>) -> Attribute {
    Attribute {
        name: name.to_owned(),
        ints: v,
        r#type: AttributeType::Ints as i32,
        ..Attribute::default()
    }
}

fn tensor_attr(name: &str, t: Tensor) -> Attribute {
    Attribute {
        name: name.to_owned(),
        t: Some(t),
        r#type: AttributeType::Tensor as i32,
        ..Attribute::default()
    }
}

fn graph_attr(name: &str, g: Graph) -> Attribute {
    Attribute {
        name: name.to_owned(),
        g: Some(Box::new(g)),
        r#type: AttributeType::Graph as i32,
        ..Attribute::default()
    }
}

fn f64_tensor(name: &str, dims: &[i64], values: &[f64]) -> Tensor {
    Tensor {
        dims: dims.to_vec(),
        data_type: ELEM_DOUBLE,
        name: name.to_owned(),
        raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn i64_tensor(name: &str, dims: &[i64], values: &[i64]) -> Tensor {
    Tensor {
        dims: dims.to_vec(),
        data_type: ELEM_INT64,
        name: name.to_owned(),
        raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// A dimension: fixed, symbolic (`"N"`) or unknown (`None` in `onnx.helper`).
enum Dim<'a> {
    Fixed(i64),
    Param(&'a str),
    Unknown,
}

fn value_info(name: &str, elem_type: i32, dims: &[Dim<'_>]) -> ValueInfo {
    ValueInfo {
        name: name.to_owned(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorType {
                elem_type,
                shape: Some(TensorShape {
                    dim: dims
                        .iter()
                        .map(|d| Dimension {
                            value: match d {
                                Dim::Fixed(v) => Some(DimValue::Value(*v)),
                                Dim::Param(p) => Some(DimValue::Param((*p).to_owned())),
                                Dim::Unknown => None,
                            },
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

fn to_i64(v: usize) -> Result<i64, ExportError> {
    i64::try_from(v).map_err(|_| ExportError::InvalidModel {
        reason: "modelo demasiado grande".to_owned(),
    })
}

/// The `Scan`, the distances and the weights (steps 1–4), shared by both graphs.
fn search_nodes(knn: &Knn) -> Result<(Vec<Node>, Vec<Tensor>), ExportError> {
    let x = knn.train_x();
    let (n, p) = x.dim();
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ExportError::InvalidModel {
            reason: "datos de entrenamiento no finitos".to_owned(),
        });
    }
    let (n64, p64) = (to_i64(n)?, to_i64(p)?);
    let k = to_i64(knn.config().n_neighbors as usize)?;
    let euclidean = knn.effective_metric() == EffectiveMetric::Euclidean;
    let train: Vec<f64> = x.iter().copied().collect();
    let mut inits = vec![
        f64_tensor("train_X", &[n64, p64], &train),
        i64_tensor("axes1", &[1], &[1]),
    ];
    let body = Graph {
        node: vec![
            node(
                "Constant",
                "body_k",
                &[],
                &["k_in"],
                vec![tensor_attr("value", i64_tensor("kc", &[1], &[k]))],
            ),
            node(
                "Constant",
                "body_axes",
                &[],
                &["axes1_in"],
                vec![tensor_attr("value", i64_tensor("ac", &[1], &[1]))],
            ),
            node(
                "Sub",
                "body_sub",
                &["x_row", "train_X_in"],
                &["diff"],
                vec![],
            ),
            if euclidean {
                node("Mul", "body_sq", &["diff", "diff"], &["term"], vec![])
            } else {
                node("Abs", "body_abs", &["diff"], &["term"], vec![])
            },
            node(
                "ReduceSum",
                "body_sum",
                &["term", "axes1_in"],
                &["r"],
                vec![int_attr("keepdims", 0)],
            ),
            node(
                "TopK",
                "body_topk",
                &["r", "k_in"],
                &["r_k", "i_k"],
                vec![
                    int_attr("axis", 0),
                    int_attr("largest", 0),
                    int_attr("sorted", 1),
                ],
            ),
            node(
                "Identity",
                "body_id_x",
                &["train_X_in"],
                &["train_X_out"],
                vec![],
            ),
        ],
        name: "knn_row".to_owned(),
        initializer: Vec::new(),
        input: vec![
            value_info(
                "train_X_in",
                ELEM_DOUBLE,
                &[Dim::Param("N"), Dim::Fixed(p64)],
            ),
            value_info("x_row", ELEM_DOUBLE, &[Dim::Fixed(p64)]),
        ],
        output: vec![
            value_info(
                "train_X_out",
                ELEM_DOUBLE,
                &[Dim::Param("N"), Dim::Fixed(p64)],
            ),
            value_info("r_k", ELEM_DOUBLE, &[Dim::Fixed(k)]),
            value_info("i_k", ELEM_INT64, &[Dim::Fixed(k)]),
        ],
    };
    let mut nodes = vec![
        node(
            "Cast",
            "cast_in",
            &[INPUT_NAME],
            &["X64"],
            vec![int_attr("to", i64::from(ELEM_DOUBLE))],
        ),
        node(
            "Scan",
            "scan_rows",
            &["train_X", "X64"],
            &["train_X_final", "rdist_k", "idx_k"],
            vec![graph_attr("body", body), int_attr("num_scan_inputs", 1)],
        ),
        if euclidean {
            node("Sqrt", "sqrt", &["rdist_k"], &["dist_k"], vec![])
        } else {
            node("Identity", "dist_id", &["rdist_k"], &["dist_k"], vec![])
        },
    ];
    match knn.config().weights {
        KnnWeights::Distance => {
            inits.push(f64_tensor("zero", &[1], &[0.0]));
            nodes.extend([
                node("Equal", "eq0", &["dist_k", "zero"], &["is_zero"], vec![]),
                node(
                    "Cast",
                    "c0",
                    &["is_zero"],
                    &["zero_w"],
                    vec![int_attr("to", i64::from(ELEM_DOUBLE))],
                ),
                node(
                    "ReduceMax",
                    "rowmax",
                    &["zero_w"],
                    &["row_has_zero_f"],
                    vec![ints_attr("axes", vec![1]), int_attr("keepdims", 1)],
                ),
                node(
                    "Cast",
                    "c1",
                    &["row_has_zero_f"],
                    &["row_has_zero"],
                    vec![int_attr("to", i64::from(ELEM_BOOL))],
                ),
                node("Reciprocal", "inv", &["dist_k"], &["inv_d"], vec![]),
                node(
                    "Where",
                    "where",
                    &["row_has_zero", "zero_w", "inv_d"],
                    &["w"],
                    vec![],
                ),
            ]);
        }
        KnnWeights::Uniform => {
            nodes.extend([
                node("Shape", "shape_w", &["dist_k"], &["dk_shape"], vec![]),
                node(
                    "ConstantOfShape",
                    "ones",
                    &["dk_shape"],
                    &["w"],
                    vec![tensor_attr("value", f64_tensor("one", &[1], &[1.0]))],
                ),
            ]);
        }
    }
    Ok((nodes, inits))
}

fn input_info(p: usize) -> Result<ValueInfo, ExportError> {
    Ok(value_info(
        INPUT_NAME,
        ELEM_FLOAT,
        &[Dim::Unknown, Dim::Fixed(to_i64(p)?)],
    ))
}

/// `KNeighborsClassifier` → `label` (int64, `class_labels[ArgMax]`) and
/// `probabilities` (float32, one column per class of `classes()`). `class_labels` =
/// the label of each class, in `classes()` order ([`crate::onnx_class_labels`]).
pub fn knn_classifier(knn: &Knn, class_labels: &[i64]) -> Result<Model, ExportError> {
    let classes = knn.classes().ok_or_else(|| ExportError::InvalidModel {
        reason: "el KNN no es de clasificación".to_owned(),
    })?;
    let c = classes.len();
    if c == 0 || class_labels.len() != c {
        return Err(ExportError::InvalidModel {
            reason: "número de clases".to_owned(),
        });
    }
    let codes = knn.train_codes().unwrap_or_default();
    let y_index: Vec<i64> = codes
        .iter()
        .map(|code| classes.binary_search(code).unwrap_or(0) as i64)
        .collect();
    let (mut nodes, mut inits) = search_nodes(knn)?;
    let c64 = to_i64(c)?;
    let range: Vec<i64> = (0..c64).collect();
    inits.extend([
        i64_tensor("train_y_code", &[to_i64(y_index.len())?], &y_index),
        i64_tensor("classes", &[c64], class_labels),
        i64_tensor("class_range", &[c64], &range),
        i64_tensor("axes2", &[1], &[2]),
    ]);
    nodes.extend([
        node(
            "Gather",
            "g_lab",
            &["train_y_code", "idx_k"],
            &["lab_k"],
            vec![int_attr("axis", 0)],
        ),
        node(
            "Unsqueeze",
            "unsq_lab",
            &["lab_k", "axes2"],
            &["lab3"],
            vec![],
        ),
        node(
            "Equal",
            "eq_cls",
            &["lab3", "class_range"],
            &["oh_b"],
            vec![],
        ),
        node(
            "Cast",
            "cast_oh",
            &["oh_b"],
            &["oh"],
            vec![int_attr("to", i64::from(ELEM_DOUBLE))],
        ),
        node("Unsqueeze", "unsq", &["w", "axes2"], &["w3"], vec![]),
        node("Mul", "mulv", &["oh", "w3"], &["votes3"], vec![]),
        node(
            "ReduceSum",
            "sumv",
            &["votes3", "axes1"],
            &["votes"],
            vec![int_attr("keepdims", 0)],
        ),
        node(
            "ReduceSum",
            "sumn",
            &["votes", "axes1"],
            &["norm"],
            vec![int_attr("keepdims", 1)],
        ),
        node("Div", "div", &["votes", "norm"], &["proba64"], vec![]),
        node(
            "ArgMax",
            "argmax",
            &["votes"],
            &["arg"],
            vec![int_attr("axis", 1), int_attr("keepdims", 0)],
        ),
        node(
            "Gather",
            "g_cls",
            &["classes", "arg"],
            &["label"],
            vec![int_attr("axis", 0)],
        ),
        node(
            "Cast",
            "cast_p",
            &["proba64"],
            &["probabilities"],
            vec![int_attr("to", i64::from(ELEM_FLOAT))],
        ),
    ]);
    Ok(model_without_ml(Graph {
        node: nodes,
        name: "mltutor_knn".to_owned(),
        initializer: inits,
        input: vec![input_info(knn.n_features())?],
        output: vec![
            value_info("label", ELEM_INT64, &[Dim::Unknown]),
            value_info(
                "probabilities",
                ELEM_FLOAT,
                &[Dim::Unknown, Dim::Fixed(c64)],
            ),
        ],
    }))
}

/// `KNeighborsRegressor` → `variable` (float32 `[N, 1]`).
pub fn knn_regressor(knn: &Knn) -> Result<Model, ExportError> {
    let y = knn
        .train_values()
        .ok_or_else(|| ExportError::InvalidModel {
            reason: "el KNN no es de regresión".to_owned(),
        })?;
    if y.iter().any(|v| !v.is_finite()) {
        return Err(ExportError::InvalidModel {
            reason: "valores objetivo no finitos".to_owned(),
        });
    }
    let (mut nodes, mut inits) = search_nodes(knn)?;
    inits.push(f64_tensor("train_y", &[to_i64(y.len())?], y));
    nodes.extend([
        node(
            "Gather",
            "g_y",
            &["train_y", "idx_k"],
            &["y_k"],
            vec![int_attr("axis", 0)],
        ),
        node("Mul", "mul_yw", &["y_k", "w"], &["yw"], vec![]),
        node(
            "ReduceSum",
            "sum_num",
            &["yw", "axes1"],
            &["num"],
            vec![int_attr("keepdims", 1)],
        ),
        node(
            "ReduceSum",
            "sum_den",
            &["w", "axes1"],
            &["den"],
            vec![int_attr("keepdims", 1)],
        ),
        node("Div", "div_y", &["num", "den"], &["pred64"], vec![]),
        node(
            "Cast",
            "cast_y",
            &["pred64"],
            &["variable"],
            vec![int_attr("to", i64::from(ELEM_FLOAT))],
        ),
    ]);
    Ok(model_without_ml(Graph {
        node: nodes,
        name: "mltutor_knn".to_owned(),
        initializer: inits,
        input: vec![input_info(knn.n_features())?],
        output: vec![value_info(
            "variable",
            ELEM_FLOAT,
            &[Dim::Unknown, Dim::Fixed(1)],
        )],
    }))
}

/// The common model header with only the default opset (no `ai.onnx.ml` node).
fn model_without_ml(graph: Graph) -> Model {
    let mut m = model(graph);
    m.opset_import.retain(|o| o.domain.is_empty());
    m
}

fn tensor_f64(t: &Tensor) -> Vec<f64> {
    t.raw_data
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect()
}

fn tensor_i64(t: &Tensor) -> Vec<i64> {
    t.raw_data
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect()
}

/// Reference evaluation of the KNN graphs (tests and export checks): reads the
/// initializers and the options back from the graph and applies the ONNX semantics
/// (float32 input cast to float64, brute force, `TopK` ties by lower index, votes or
/// weighted mean, outputs cast to float32). Returns `(labels, scores)` like
/// [`super::evaluate`] (`labels` are 0 for the regressor).
pub fn evaluate_knn(m: &Model, x: &[Vec<f32>]) -> Result<(Vec<i64>, Vec<Vec<f32>>), ExportError> {
    let err = |msg: &str| ExportError::Onnx {
        message: msg.to_owned(),
    };
    let g = m.graph.as_ref().ok_or_else(|| err("sin grafo"))?;
    let init = |name: &str| g.initializer.iter().find(|t| t.name == name);
    let train_t = init("train_X").ok_or_else(|| err("sin train_X"))?;
    let (n, p) = match train_t.dims.as_slice() {
        [n, p] => (
            usize::try_from(*n).map_err(|_| err("dims"))?,
            usize::try_from(*p).map_err(|_| err("dims"))?,
        ),
        _ => return Err(err("dims de train_X")),
    };
    let train = tensor_f64(train_t);
    if train.len() != n * p {
        return Err(err("tamaño de train_X"));
    }
    let scan = g
        .node
        .iter()
        .find(|nd| nd.op_type == "Scan")
        .ok_or_else(|| err("sin Scan"))?;
    let body = scan
        .attribute
        .iter()
        .find_map(|a| a.g.as_deref())
        .ok_or_else(|| err("sin cuerpo"))?;
    let k = body
        .node
        .iter()
        .find(|nd| nd.name == "body_k")
        .and_then(|nd| nd.attribute.first())
        .and_then(|a| a.t.as_ref())
        .map(tensor_i64)
        .and_then(|v| v.first().copied())
        .ok_or_else(|| err("sin K"))?;
    let k = usize::try_from(k).map_err(|_| err("K"))?.min(n);
    let manhattan = body.node.iter().any(|nd| nd.op_type == "Abs");
    let distance = g.node.iter().any(|nd| nd.op_type == "Where");
    let labels_t = init("train_y_code").map(tensor_i64);
    let classes = init("classes").map(tensor_i64);
    let values = init("train_y").map(tensor_f64);
    let (mut out_l, mut out_s) = (Vec::with_capacity(x.len()), Vec::with_capacity(x.len()));
    let mut order: Vec<(f64, usize)> = Vec::with_capacity(n);
    for row in x {
        if row.len() != p {
            return Err(err("número de columnas"));
        }
        let q: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
        order.clear();
        for (i, t) in train.chunks_exact(p).enumerate() {
            let mut s = 0.0;
            for (a, b) in q.iter().zip(t) {
                let d = a - b;
                s += if manhattan { d.abs() } else { d * d };
            }
            order.push((s, i));
        }
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let nb = &order[..k];
        let dist: Vec<f64> = nb
            .iter()
            .map(|&(r, _)| if manhattan { r } else { r.sqrt() })
            .collect();
        let has_zero = distance && dist.contains(&0.0);
        let w: Vec<f64> = dist
            .iter()
            .map(|&d| match (distance, has_zero) {
                (false, _) => 1.0,
                (true, true) => f64::from(u8::from(d == 0.0)),
                (true, false) => 1.0 / d,
            })
            .collect();
        match (&labels_t, &classes, &values) {
            (Some(lab), Some(cls), _) => {
                let mut votes = vec![0.0; cls.len()];
                for (&(_, i), &wi) in nb.iter().zip(&w) {
                    let l = usize::try_from(lab[i]).map_err(|_| err("etiqueta"))?;
                    if let Some(v) = votes.get_mut(l) {
                        *v += wi;
                    }
                }
                let total: f64 = votes.iter().sum();
                let best = votes
                    .iter()
                    .enumerate()
                    .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv {
                            (i, v)
                        } else {
                            (bi, bv)
                        }
                    })
                    .0;
                out_l.push(cls[best]);
                out_s.push(votes.iter().map(|v| (v / total) as f32).collect());
            }
            (_, _, Some(y)) => {
                let num: f64 = nb.iter().zip(&w).map(|(&(_, i), &wi)| y[i] * wi).sum();
                let den: f64 = w.iter().sum();
                out_l.push(0);
                out_s.push(vec![(num / den) as f32]);
            }
            _ => return Err(err("sin etiquetas ni valores")),
        }
    }
    Ok((out_l, out_s))
}
