//! ONNX export of linear models (fase2-regresion.md C11, D1).
//!
//! The protobuf messages are the subset of `onnx.proto` (onnx 1.17) that these
//! graphs need, written by hand with `prost-derive` (no `protoc`; MEMORY spike
//! 2026-10-13). Field numbers are the official ones. Graphs follow skl2onnx's
//! (`tests/parity/linear_models/*.json` → `onnx_reference.nodes`): IR version 8,
//! opsets `ai.onnx` 15 and `ai.onnx.ml` 3, input `float_input` `[None, n]` float32.
//!
//! * [`linear_regressor`]: one `ai.onnx.ml.LinearRegressor` → `variable` `[None, 1]`.
//! * [`linear_classifier`]: `ai.onnx.ml.LinearClassifier` → `label` (int64) and
//!   probabilities (no ZipMap). Multinomial: `post_transform="SOFTMAX"`,
//!   `multi_class=1`, followed by `Normalizer(norm="L1")` → `probabilities`.
//!   Binary: like skl2onnx, two rows `[−w, w]`, intercepts `[−b, b]` and
//!   `post_transform="LOGISTIC"` (σ(−z), σ(z)), a single node.
//! * [`tree_ensemble_classifier`] / [`tree_ensemble_regressor`] (F3): one
//!   `TreeEnsembleClassifier` (→ `label` int64 + `probabilities`, no ZipMap) or
//!   `TreeEnsembleRegressor` (→ `variable`) with one tree. Thresholds are the
//!   **largest float32 ≤ the threshold** (D47): ONNX compares float32 inputs with
//!   float32 thresholds, and `x ≤ floor32(t) ⇔ x ≤ t` for every float32 `x`, which is
//!   exactly scikit-learn's rule (skl2onnx rounds to nearest and can misroute).
//!   Binary classifiers follow skl2onnx (one weight per leaf, P(class 1) under
//!   class id 0), which onnxruntime evaluates with its binary rule; listing both
//!   classes makes onnxruntime return wrong labels (checked: 537/891 in Titanic).

use mltutor_core::codegen::f32_floor;
use mltutor_core::tree::DecisionTree;
use prost::Message;

use crate::ExportError;

mod knn;

pub use knn::{evaluate_knn, knn_classifier, knn_regressor};

/// `TensorProto.DataType.FLOAT`.
pub const ELEM_FLOAT: i32 = 1;
/// `TensorProto.DataType.INT64`.
pub const ELEM_INT64: i32 = 7;
/// `TensorProto.DataType.BOOL`.
pub const ELEM_BOOL: i32 = 9;
/// `TensorProto.DataType.DOUBLE`.
pub const ELEM_DOUBLE: i32 = 11;
/// IR version written in the model.
pub const IR_VERSION: i64 = 8;
/// Opset of the default domain.
pub const OPSET_ONNX: i64 = 15;
/// Opset of `ai.onnx.ml`.
pub const OPSET_ML: i64 = 3;
/// Name of the input.
pub const INPUT_NAME: &str = "float_input";

/// `OperatorSetIdProto`.
#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetId {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// `AttributeProto.AttributeType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum AttributeType {
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    Floats = 6,
    Ints = 7,
    Strings = 8,
}

/// `AttributeProto` (only scalar and list fields).
#[derive(Clone, PartialEq, Message)]
pub struct Attribute {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes = "vec", tag = "4")]
    pub s: Vec<u8>,
    /// `t`: a tensor (`Constant`, `ConstantOfShape`).
    #[prost(message, optional, tag = "5")]
    pub t: Option<Tensor>,
    /// `g`: a subgraph (the body of `Scan`).
    #[prost(message, optional, boxed, tag = "6")]
    pub g: Option<Box<Graph>>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(bytes = "vec", repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
}

impl Attribute {
    fn floats(name: &str, v: Vec<f32>) -> Self {
        Self {
            name: name.to_owned(),
            floats: v,
            r#type: AttributeType::Floats as i32,
            ..Self::default()
        }
    }
    fn ints(name: &str, v: Vec<i64>) -> Self {
        Self {
            name: name.to_owned(),
            ints: v,
            r#type: AttributeType::Ints as i32,
            ..Self::default()
        }
    }
    fn int(name: &str, v: i64) -> Self {
        Self {
            name: name.to_owned(),
            i: v,
            r#type: AttributeType::Int as i32,
            ..Self::default()
        }
    }
    fn string(name: &str, v: &str) -> Self {
        Self {
            name: name.to_owned(),
            s: v.as_bytes().to_vec(),
            r#type: AttributeType::String as i32,
            ..Self::default()
        }
    }
}

/// `TensorProto` (initializers and tensor attributes; values in `raw_data`,
/// little-endian, as `onnx.numpy_helper.from_array` writes them).
#[derive(Clone, PartialEq, Message)]
pub struct Tensor {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
}

/// `NodeProto`.
#[derive(Clone, PartialEq, Message)]
pub struct Node {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<Attribute>,
    #[prost(string, tag = "7")]
    pub domain: String,
}

/// `TensorShapeProto.Dimension` (`dim_value` or `dim_param`).
#[derive(Clone, PartialEq, Message)]
pub struct Dimension {
    #[prost(oneof = "DimValue", tags = "1, 2")]
    pub value: Option<DimValue>,
}

/// The oneof of [`Dimension`].
#[derive(Clone, PartialEq, prost::Oneof)]
pub enum DimValue {
    #[prost(int64, tag = "1")]
    Value(i64),
    #[prost(string, tag = "2")]
    Param(String),
}

/// `TensorShapeProto`.
#[derive(Clone, PartialEq, Message)]
pub struct TensorShape {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<Dimension>,
}

/// `TypeProto.Tensor`.
#[derive(Clone, PartialEq, Message)]
pub struct TensorType {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShape>,
}

/// `TypeProto` (tensor types only).
#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TensorType>,
}

/// `ValueInfoProto`.
#[derive(Clone, PartialEq, Message)]
pub struct ValueInfo {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

/// `GraphProto`.
#[derive(Clone, PartialEq, Message)]
pub struct Graph {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<Node>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<Tensor>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfo>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfo>,
}

/// `ModelProto`.
#[derive(Clone, PartialEq, Message)]
pub struct Model {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<Graph>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetId>,
}

impl Model {
    /// Protobuf bytes (the `.onnx` file).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    /// Parses bytes written by [`Model::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ExportError> {
        Self::decode(bytes).map_err(|e| ExportError::Onnx {
            message: e.to_string(),
        })
    }
}

fn tensor_value(name: &str, elem_type: i32, cols: Option<i64>) -> ValueInfo {
    let mut dim = vec![Dimension { value: None }];
    if let Some(c) = cols {
        dim.push(Dimension {
            value: Some(DimValue::Value(c)),
        });
    }
    ValueInfo {
        name: name.to_owned(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorType {
                elem_type,
                shape: Some(TensorShape { dim }),
            }),
        }),
    }
}

pub(crate) fn model(graph: Graph) -> Model {
    Model {
        ir_version: IR_VERSION,
        producer_name: "MLTutor".to_owned(),
        producer_version: crate::VERSION.to_owned(),
        graph: Some(graph),
        opset_import: vec![
            OperatorSetId {
                domain: String::new(),
                version: OPSET_ONNX,
            },
            OperatorSetId {
                domain: "ai.onnx.ml".to_owned(),
                version: OPSET_ML,
            },
        ],
    }
}

fn check_finite(values: &[f64]) -> Result<Vec<f32>, ExportError> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(ExportError::InvalidModel {
            reason: "coeficientes no finitos".to_owned(),
        });
    }
    Ok(values.iter().map(|&v| v as f32).collect())
}

/// `LinearRegression` → `ai.onnx.ml.LinearRegressor`.
pub fn linear_regressor(coef: &[f64], intercept: f64) -> Result<Model, ExportError> {
    let n = i64::try_from(coef.len()).map_err(|_| ExportError::InvalidModel {
        reason: "demasiadas características".to_owned(),
    })?;
    let node = Node {
        input: vec![INPUT_NAME.to_owned()],
        output: vec!["variable".to_owned()],
        name: "LinearRegressor".to_owned(),
        op_type: "LinearRegressor".to_owned(),
        attribute: vec![
            Attribute::floats("coefficients", check_finite(coef)?),
            Attribute::floats("intercepts", check_finite(&[intercept])?),
        ],
        domain: "ai.onnx.ml".to_owned(),
    };
    Ok(model(Graph {
        node: vec![node],
        name: "mltutor_linear_regression".to_owned(),
        input: vec![tensor_value(INPUT_NAME, ELEM_FLOAT, Some(n))],
        output: vec![tensor_value("variable", ELEM_FLOAT, Some(1))],
        initializer: Vec::new(),
    }))
}

/// `LogisticRegression` → `LinearClassifier` (+ `Normalizer(L1)` if multinomial).
///
/// `coef` is `n_rows × n_features` row-major (1 row if binary), `classes` the
/// labels (`classes_`), `intercept` one value per row.
pub fn linear_classifier(
    coef: &[f64],
    n_features: usize,
    intercept: &[f64],
    classes: &[i64],
) -> Result<Model, ExportError> {
    let rows = intercept.len();
    let binary = classes.len() == 2;
    if n_features == 0
        || coef.len() != rows * n_features
        || (binary && rows != 1)
        || (!binary && rows != classes.len())
    {
        return Err(ExportError::InvalidModel {
            reason: format!(
                "forma de coeficientes no válida ({} valores, {rows} filas, {} clases)",
                coef.len(),
                classes.len()
            ),
        });
    }
    let n = i64::try_from(n_features).map_err(|_| ExportError::InvalidModel {
        reason: "demasiadas características".to_owned(),
    })?;
    // Binary: skl2onnx's two rows [−w, w] so that LOGISTIC gives [1 − p, p].
    let (coef_rows, intercepts): (Vec<f64>, Vec<f64>) = if binary {
        (
            coef.iter()
                .map(|v| -v)
                .chain(coef.iter().copied())
                .collect(),
            vec![-intercept[0], intercept[0]],
        )
    } else {
        (coef.to_vec(), intercept.to_vec())
    };
    let proba_out = if binary {
        "probabilities"
    } else {
        "probability_tensor"
    };
    let classifier = Node {
        input: vec![INPUT_NAME.to_owned()],
        output: vec!["label".to_owned(), proba_out.to_owned()],
        name: "LinearClassifier".to_owned(),
        op_type: "LinearClassifier".to_owned(),
        attribute: vec![
            Attribute::ints("classlabels_ints", classes.to_vec()),
            Attribute::floats("coefficients", check_finite(&coef_rows)?),
            Attribute::floats("intercepts", check_finite(&intercepts)?),
            Attribute::int("multi_class", i64::from(!binary)),
            Attribute::string(
                "post_transform",
                if binary { "LOGISTIC" } else { "SOFTMAX" },
            ),
        ],
        domain: "ai.onnx.ml".to_owned(),
    };
    let mut nodes = vec![classifier];
    if !binary {
        nodes.push(Node {
            input: vec!["probability_tensor".to_owned()],
            output: vec!["probabilities".to_owned()],
            name: "Normalizer".to_owned(),
            op_type: "Normalizer".to_owned(),
            attribute: vec![Attribute::string("norm", "L1")],
            domain: "ai.onnx.ml".to_owned(),
        });
    }
    let k = i64::try_from(classes.len()).unwrap_or(i64::MAX);
    Ok(model(Graph {
        node: nodes,
        name: "mltutor_logistic_regression".to_owned(),
        input: vec![tensor_value(INPUT_NAME, ELEM_FLOAT, Some(n))],
        output: vec![
            tensor_value("label", ELEM_INT64, None),
            tensor_value("probabilities", ELEM_FLOAT, Some(k)),
        ],
        initializer: Vec::new(),
    }))
}

fn floats_attr(name: &str, v: Vec<f32>) -> Attribute {
    Attribute::floats(name, v)
}

fn strings_attr(name: &str, v: &[&str]) -> Attribute {
    Attribute {
        name: name.to_owned(),
        strings: v.iter().map(|s| s.as_bytes().to_vec()).collect(),
        r#type: AttributeType::Strings as i32,
        ..Attribute::default()
    }
}

fn to_i64(v: usize) -> Result<i64, ExportError> {
    i64::try_from(v).map_err(|_| ExportError::InvalidModel {
        reason: "árbol demasiado grande".to_owned(),
    })
}

/// The `nodes_*` attributes shared by both tree ensembles (tree id 0).
fn tree_node_attributes(tree: &DecisionTree) -> Result<Vec<Attribute>, ExportError> {
    let nodes = tree.nodes();
    let n = nodes.len();
    let (mut ids, mut feat, mut modes, mut values) = (vec![], vec![], vec![], vec![]);
    let (mut t_ids, mut f_ids) = (vec![], vec![]);
    for (i, node) in nodes.iter().enumerate() {
        ids.push(to_i64(i)?);
        match node.split {
            Some(s) => {
                if !s.threshold.is_finite() {
                    return Err(ExportError::InvalidModel {
                        reason: "umbral no finito".to_owned(),
                    });
                }
                feat.push(to_i64(s.feature)?);
                modes.push("BRANCH_LEQ");
                values.push(f32_floor(s.threshold));
                t_ids.push(to_i64(s.left)?);
                f_ids.push(to_i64(s.right)?);
            }
            None => {
                feat.push(0);
                modes.push("LEAF");
                values.push(0.0);
                t_ids.push(0);
                f_ids.push(0);
            }
        }
    }
    Ok(vec![
        Attribute::ints("nodes_falsenodeids", f_ids),
        Attribute::ints("nodes_featureids", feat),
        floats_attr("nodes_hitrates", vec![1.0; n]),
        Attribute::ints("nodes_missing_value_tracks_true", vec![0; n]),
        strings_attr("nodes_modes", &modes),
        Attribute::ints("nodes_nodeids", ids),
        Attribute::ints("nodes_treeids", vec![0; n]),
        Attribute::ints("nodes_truenodeids", t_ids),
        floats_attr("nodes_values", values),
        Attribute::string("post_transform", "NONE"),
    ])
}

/// `DecisionTreeClassifier` → `ai.onnx.ml.TreeEnsembleClassifier`.
/// `class_labels` = `classlabels_int64s`, one per class of the tree.
pub fn tree_ensemble_classifier(
    tree: &DecisionTree,
    class_labels: &[i64],
) -> Result<Model, ExportError> {
    let p = mltutor_core::model::Predictor::n_features(tree);
    let k = class_labels.len();
    if !tree.is_classifier() || k == 0 {
        return Err(ExportError::InvalidModel {
            reason: "el árbol no es de clasificación".to_owned(),
        });
    }
    let mut attrs = tree_node_attributes(tree)?;
    let (mut cids, mut cnodes, mut cweights) = (vec![], vec![], vec![]);
    for (i, node) in tree.nodes().iter().enumerate() {
        if node.is_leaf() {
            if node.value.len() != k {
                return Err(ExportError::InvalidModel {
                    reason: "número de clases".to_owned(),
                });
            }
            if k == 2 {
                // Binary, as skl2onnx: one weight per leaf, P(class 1) under class
                // id 0. onnxruntime's binary rule then gives [1 − p, p] and the
                // label `classlabels[1]` only when p > 0.5 (ties → class 0, like
                // scikit-learn's argmax).
                cids.push(0);
                cnodes.push(to_i64(i)?);
                cweights.push(node.value[1] as f32);
            } else {
                for (c, &v) in node.value.iter().enumerate() {
                    cids.push(to_i64(c)?);
                    cnodes.push(to_i64(i)?);
                    cweights.push(v as f32);
                }
            }
        }
    }
    let n_leaf_entries = cids.len();
    attrs.push(Attribute::ints("class_ids", cids));
    attrs.push(Attribute::ints("class_nodeids", cnodes));
    attrs.push(Attribute::ints("class_treeids", vec![0; n_leaf_entries]));
    attrs.push(floats_attr("class_weights", cweights));
    attrs.push(Attribute::ints("classlabels_int64s", class_labels.to_vec()));
    attrs.sort_by(|a, b| a.name.cmp(&b.name));
    let node = Node {
        input: vec![INPUT_NAME.to_owned()],
        output: vec!["label".to_owned(), "probabilities".to_owned()],
        name: "TreeEnsembleClassifier".to_owned(),
        op_type: "TreeEnsembleClassifier".to_owned(),
        attribute: attrs,
        domain: "ai.onnx.ml".to_owned(),
    };
    Ok(model(Graph {
        node: vec![node],
        name: "mltutor_decision_tree_classifier".to_owned(),
        input: vec![tensor_value(INPUT_NAME, ELEM_FLOAT, Some(to_i64(p)?))],
        output: vec![
            tensor_value("label", ELEM_INT64, None),
            tensor_value("probabilities", ELEM_FLOAT, Some(to_i64(k)?)),
        ],
        initializer: Vec::new(),
    }))
}

/// `DecisionTreeRegressor` → `ai.onnx.ml.TreeEnsembleRegressor`.
pub fn tree_ensemble_regressor(tree: &DecisionTree) -> Result<Model, ExportError> {
    if tree.is_classifier() {
        return Err(ExportError::InvalidModel {
            reason: "el árbol no es de regresión".to_owned(),
        });
    }
    let p = mltutor_core::model::Predictor::n_features(tree);
    let mut attrs = tree_node_attributes(tree)?;
    let (mut tnodes, mut tweights) = (vec![], vec![]);
    for (i, node) in tree.nodes().iter().enumerate() {
        if node.is_leaf() {
            let v = node.value.first().copied().unwrap_or(f64::NAN);
            if !v.is_finite() {
                return Err(ExportError::InvalidModel {
                    reason: "valor de hoja no finito".to_owned(),
                });
            }
            tnodes.push(to_i64(i)?);
            tweights.push(v as f32);
        }
    }
    let n_leaves = tnodes.len();
    attrs.push(Attribute::int("n_targets", 1));
    attrs.push(Attribute::ints("target_ids", vec![0; n_leaves]));
    attrs.push(Attribute::ints("target_nodeids", tnodes));
    attrs.push(Attribute::ints("target_treeids", vec![0; n_leaves]));
    attrs.push(floats_attr("target_weights", tweights));
    attrs.sort_by(|a, b| a.name.cmp(&b.name));
    let node = Node {
        input: vec![INPUT_NAME.to_owned()],
        output: vec!["variable".to_owned()],
        name: "TreeEnsembleRegressor".to_owned(),
        op_type: "TreeEnsembleRegressor".to_owned(),
        attribute: attrs,
        domain: "ai.onnx.ml".to_owned(),
    };
    Ok(model(Graph {
        node: vec![node],
        name: "mltutor_decision_tree_regressor".to_owned(),
        input: vec![tensor_value(INPUT_NAME, ELEM_FLOAT, Some(to_i64(p)?))],
        output: vec![tensor_value("variable", ELEM_FLOAT, Some(1))],
        initializer: Vec::new(),
    }))
}

/// float32 evaluation of a one-tree `TreeEnsemble*` node (BRANCH_LEQ only).
fn evaluate_tree_ensemble(
    node: &Node,
    x: &[Vec<f32>],
) -> Result<(Vec<i64>, Vec<Vec<f32>>), ExportError> {
    let err = |m: &str| ExportError::Onnx {
        message: m.to_owned(),
    };
    let attr = |name: &str| node.attribute.iter().find(|a| a.name == name);
    let ints = |name: &str| attr(name).map(|a| a.ints.clone()).ok_or_else(|| err(name));
    let floats = |name: &str| {
        attr(name)
            .map(|a| a.floats.clone())
            .ok_or_else(|| err(name))
    };
    let modes: Vec<Vec<u8>> = attr("nodes_modes")
        .map(|a| a.strings.clone())
        .ok_or_else(|| err("nodes_modes"))?;
    let feat = ints("nodes_featureids")?;
    let vals = floats("nodes_values")?;
    let tn = ints("nodes_truenodeids")?;
    let fnn = ints("nodes_falsenodeids")?;
    let idx = |v: i64| usize::try_from(v).map_err(|_| err("índice negativo"));
    let classifier = node.op_type == "TreeEnsembleClassifier";
    let (leaf_ids, leaf_cls, leaf_w, labels) = if classifier {
        (
            ints("class_nodeids")?,
            ints("class_ids")?,
            floats("class_weights")?,
            ints("classlabels_int64s")?,
        )
    } else {
        let n = ints("target_nodeids")?;
        let z = vec![0; n.len()];
        (n, z, floats("target_weights")?, vec![0])
    };
    let k = labels.len();
    let mut out_l = Vec::with_capacity(x.len());
    let mut out_s = Vec::with_capacity(x.len());
    for row in x {
        let mut cur = 0usize;
        let mut guard = 0;
        while modes.get(cur).map(Vec::as_slice) == Some(b"BRANCH_LEQ".as_slice()) {
            let f = idx(*feat.get(cur).ok_or_else(|| err("nodo"))?)?;
            let v = *row.get(f).ok_or_else(|| err("número de columnas"))?;
            cur = if v <= vals[cur] {
                idx(tn[cur])?
            } else {
                idx(fnn[cur])?
            };
            guard += 1;
            if guard > modes.len() {
                return Err(err("árbol con ciclos"));
            }
        }
        let mut scores = vec![0.0_f32; k];
        for ((&n, &c), &w) in leaf_ids.iter().zip(&leaf_cls).zip(&leaf_w) {
            if idx(n)? == cur {
                if let Some(s) = scores.get_mut(idx(c)?) {
                    *s += w;
                }
            }
        }
        if classifier && k == 2 && leaf_cls.iter().all(|&c| c == 0) {
            // onnxruntime's binary case: the single score is P(class 1).
            let p = scores[0];
            out_l.push(if p > 0.5 { labels[1] } else { labels[0] });
            out_s.push(vec![1.0 - p, p]);
            continue;
        }
        let best = scores
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        out_l.push(if classifier {
            labels.get(best).copied().unwrap_or(0)
        } else {
            0
        });
        out_s.push(scores);
    }
    Ok((out_l, out_s))
}

/// Reference evaluation of the graphs above in float32, like onnxruntime, for
/// tests and for the export check: returns `(labels, scores)` with one row of
/// scores per sample (`variable` for the regressor, `probabilities` for the
/// classifier).
pub fn evaluate(model: &Model, x: &[Vec<f32>]) -> Result<(Vec<i64>, Vec<Vec<f32>>), ExportError> {
    let err = |m: &str| ExportError::Onnx {
        message: m.to_owned(),
    };
    let graph = model.graph.as_ref().ok_or_else(|| err("sin grafo"))?;
    let first = graph.node.first().ok_or_else(|| err("sin nodos"))?;
    if graph.node.iter().any(|n| n.op_type == "Scan") {
        return evaluate_knn(model, x);
    }
    if first.op_type.starts_with("TreeEnsemble") {
        return evaluate_tree_ensemble(first, x);
    }
    let attr = |name: &str| first.attribute.iter().find(|a| a.name == name);
    let coef = &attr("coefficients")
        .ok_or_else(|| err("sin coefficients"))?
        .floats;
    let intercepts = &attr("intercepts")
        .ok_or_else(|| err("sin intercepts"))?
        .floats;
    let rows = intercepts.len();
    let p = coef.len().checked_div(rows).unwrap_or(0);
    let mut labels = Vec::with_capacity(x.len());
    let mut scores = Vec::with_capacity(x.len());
    for sample in x {
        if sample.len() != p {
            return Err(err("número de columnas"));
        }
        let z: Vec<f32> = (0..rows)
            .map(|r| {
                let mut acc = intercepts[r];
                for j in 0..p {
                    acc += coef[r * p + j] * sample[j];
                }
                acc
            })
            .collect();
        if first.op_type == "LinearRegressor" {
            scores.push(z);
            labels.push(0);
            continue;
        }
        let classes = &attr("classlabels_ints")
            .ok_or_else(|| err("sin clases"))?
            .ints;
        let post = attr("post_transform")
            .map(|a| a.s.as_slice())
            .unwrap_or(b"NONE");
        let probs: Vec<f32> = if post == b"LOGISTIC" {
            z.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
        } else if post == b"SOFTMAX" {
            let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
            let s: f32 = e.iter().sum();
            e.iter().map(|v| v / s).collect()
        } else {
            z
        };
        let s: f32 = probs.iter().map(|v| v.abs()).sum();
        let probs: Vec<f32> = probs.iter().map(|v| v / s).collect();
        let best = probs
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        labels.push(classes.get(best).copied().unwrap_or(0));
        scores.push(probs);
    }
    Ok((labels, scores))
}
