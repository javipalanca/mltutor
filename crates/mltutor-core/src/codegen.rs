//! Python code shown to the student (fase2-regresion.md C12): the templates of
//! `content/code/*.j2`, embedded at build time (`build.rs`, loaded by file name)
//! and rendered with minijinja exactly like jinja2 in `tools/export_golden.py`:
//! `keep_trailing_newline`, strict undefined variables, a `pyrepr` filter that
//! reproduces Python's `repr()` (`'texto'`, `[1, 2.5, None]`, `0.3`, `2.5e-07`)
//! and a `pycomment` filter ([`py_comment`]) for user text inside a `#` comment or
//! a docstring.

use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Serde, Value, ValueKind};
use minijinja::{Environment, UndefinedBehavior};
use serde::{Deserialize, Serialize};

use crate::pyfmt::py_float_repr;

pub mod tree;

pub use tree::{
    corrected_context as tree_python_corrected_context, f32_floor, f64_switch_threshold,
    literal_context as tree_python_literal_context, py_identifiers, TreePythonContext,
    TreePythonCorrectedContext,
};

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/templates.rs"));
}

/// Errors of [`render`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CodegenError {
    /// No `content/code/<name>` was embedded at build time.
    #[error("No existe la plantilla de código '{name}'")]
    TemplateNotFound { name: String },
    /// minijinja failed (syntax error, missing variable…).
    #[error("Error al generar el código con '{name}': {message}")]
    Render { name: String, message: String },
}

/// The templates the lessons use (contract `CodeTemplate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeTemplate {
    /// `matriz_confusion.py.j2` (no variables).
    ConfusionMatrix,
    /// `precision_por_clase.py.j2` (no variables).
    PrecisionByClass,
    /// `predicciones_vs_reales.py.j2` (no variables).
    PredVsReal,
    /// `distribucion_errores.py.j2` (no variables).
    ErrorDistribution,
    /// `frontera_decision.py.j2`: [`DecisionBoundaryContext`] (D30).
    DecisionBoundary,
    /// `regresion_codigo.py.j2`: Python's literal pickle snippet (no variables).
    RegressionCode,
    /// `regresion_entrenamiento.py.j2`: [`RegressionTrainingContext`] (D29).
    RegressionTraining,
    /// `paquete_modelo_lineal.py.j2`: script of the Python package (D1).
    PackageScript,
    /// `paquete_leeme.txt.j2`: `LEEME.txt` of the Python package (D28).
    PackageReadme,
    /// 🌲 `visualizar_arbol.py.j2` (no variables).
    TreeViz,
    /// 🌲 `texto_arbol.py.j2` (no variables).
    TreeText,
    /// 🌲 `camino_decision.py.j2` (no variables).
    DecisionPath,
    /// 🌲 `usar_modelo_arbol.py.j2`: [`TreeUsageContext`].
    TreeUsage,
    /// 🌲 `arbol_python_corregido.py.j2` (D44): [`TreePythonCorrectedContext`].
    TreePython,
    /// 🌲 `arbol_python.py.j2` (Python's literal text): [`TreePythonContext`].
    TreePythonLiteral,
    /// 🌲 `paquete_modelo_arbol.py.j2`: script of the tree's Python package (D43).
    TreePackageScript,
    /// 🌲 `paquete_leeme_arbol.txt.j2`: `LEEME.txt` of the tree's package.
    TreePackageReadme,
    /// 🌲 `arbol_entrenamiento.py.j2`: [`TreeTrainingContext`] (training script).
    TreeTraining,
    /// 🔍 `knn_entrenamiento.py.j2`: [`KnnTrainingContext`] (training script).
    KnnTraining,
    /// 🔍 `paquete_modelo_knn.py.j2`: script of the KNN's Python package (D57).
    KnnPackageScript,
    /// 🔍 `paquete_leeme_knn.txt.j2`: `LEEME.txt` of the KNN's package.
    KnnPackageReadme,
    /// 🧠 `red_entrenamiento.py.j2` (D74): [`NnTrainingContext`] (Keras training script,
    /// also shown in 🏗️ without a model).
    NnTraining,
    /// 🧠 `paquete_modelo_red.py.j2`: script of the Keras package (D75),
    /// [`NnPackageScriptContext`].
    NnPackageScript,
    /// 🧠 `paquete_leeme_red.txt.j2`: `LEEME.txt` of the Keras package,
    /// [`NnPackageReadmeContext`].
    NnPackageReadme,
    /// 🧠 `red_predictor_numpy.py.j2`: NumPy predictor with the real weights,
    /// [`NnPackageScriptContext`].
    NnNumpyPredictor,
}

impl CodeTemplate {
    /// File name in `content/code/`.
    pub fn file_name(self) -> &'static str {
        match self {
            CodeTemplate::ConfusionMatrix => "matriz_confusion.py.j2",
            CodeTemplate::PrecisionByClass => "precision_por_clase.py.j2",
            CodeTemplate::PredVsReal => "predicciones_vs_reales.py.j2",
            CodeTemplate::ErrorDistribution => "distribucion_errores.py.j2",
            CodeTemplate::DecisionBoundary => "frontera_decision.py.j2",
            CodeTemplate::RegressionCode => "regresion_codigo.py.j2",
            CodeTemplate::RegressionTraining => "regresion_entrenamiento.py.j2",
            CodeTemplate::PackageScript => "paquete_modelo_lineal.py.j2",
            CodeTemplate::PackageReadme => "paquete_leeme.txt.j2",
            CodeTemplate::TreeViz => "visualizar_arbol.py.j2",
            CodeTemplate::TreeText => "texto_arbol.py.j2",
            CodeTemplate::DecisionPath => "camino_decision.py.j2",
            CodeTemplate::TreeUsage => "usar_modelo_arbol.py.j2",
            CodeTemplate::TreePython => "arbol_python_corregido.py.j2",
            CodeTemplate::TreePythonLiteral => "arbol_python.py.j2",
            CodeTemplate::TreePackageScript => "paquete_modelo_arbol.py.j2",
            CodeTemplate::TreePackageReadme => "paquete_leeme_arbol.txt.j2",
            CodeTemplate::TreeTraining => "arbol_entrenamiento.py.j2",
            CodeTemplate::KnnTraining => "knn_entrenamiento.py.j2",
            CodeTemplate::KnnPackageScript => "paquete_modelo_knn.py.j2",
            CodeTemplate::KnnPackageReadme => "paquete_leeme_knn.txt.j2",
            CodeTemplate::NnTraining => "red_entrenamiento.py.j2",
            CodeTemplate::NnPackageScript => "paquete_modelo_red.py.j2",
            CodeTemplate::NnPackageReadme => "paquete_leeme_red.txt.j2",
            CodeTemplate::NnNumpyPredictor => "red_predictor_numpy.py.j2",
        }
    }
}

/// Context of `frontera_decision.py.j2`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionBoundaryContext {
    /// The two column indices of the chosen pair.
    pub feature_indices: [usize; 2],
    /// Their friendly names.
    pub feature_names: Vec<String>,
    pub class_names: Option<Vec<String>>,
    /// D63 (F4): `"todos"` in 🔍 KNN (the 2-D model is fitted on every row of the
    /// dataset, `X`/`y`); `None` or `"entrenamiento"` elsewhere (`X_train`/`y_train`).
    /// `None` is not sent, so the render is byte for byte the one of F2/F3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub datos: Option<String>,
}

impl DecisionBoundaryContext {
    /// `datos` value of 🔍 KNN (D63).
    pub const ALL_ROWS: &'static str = "todos";
    /// `datos` value of Regresión and Árboles.
    pub const TRAIN_ROWS: &'static str = "entrenamiento";

    /// The context of Regresión and Árboles (`datos` unset).
    pub fn new(
        feature_indices: [usize; 2],
        feature_names: Vec<String>,
        class_names: Option<Vec<String>>,
    ) -> Self {
        Self {
            feature_indices,
            feature_names,
            class_names,
            datos: None,
        }
    }

    /// The same context with `datos = "todos"` (🔍 KNN, D63).
    pub fn all_rows(mut self) -> Self {
        self.datos = Some(Self::ALL_ROWS.to_owned());
        self
    }
}

/// Context of `regresion_entrenamiento.py.j2` (every key is always sent, `None`
/// as `null` when it does not apply).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionTrainingContext {
    /// Built-in slug (`"iris"`…) or `"csv"`.
    pub dataset: String,
    pub dataset_label: String,
    pub csv_file: Option<String>,
    pub csv_target: Option<String>,
    pub csv_sep: Option<String>,
    pub csv_decimal: Option<String>,
    pub csv_encoding: Option<String>,
    /// `"linear"` or `"logistic"`.
    pub model: String,
    pub max_iter: Option<u32>,
    pub test_size: f64,
    pub feature_names: Vec<String>,
    pub class_names: Option<Vec<String>>,
    /// Form values (codes, column order); `None` = first test row.
    pub sample: Option<Vec<f64>>,
}

/// Context of `arbol_entrenamiento.py.j2` (every key is always sent, `None` as
/// `null` when it does not apply), the tree's equivalent of
/// [`RegressionTrainingContext`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeTrainingContext {
    /// Built-in slug (`"iris"`…) or `"csv"`.
    pub dataset: String,
    pub dataset_label: String,
    pub csv_file: Option<String>,
    pub csv_target: Option<String>,
    pub csv_sep: Option<String>,
    pub csv_decimal: Option<String>,
    pub csv_encoding: Option<String>,
    /// `"classification"` or `"regression"` (the model's task).
    pub task: String,
    /// scikit-learn's criterion name ([`crate::tree::TreeCriterion::as_str`]).
    pub criterion: String,
    pub max_depth: u32,
    pub min_samples_split: u32,
    pub test_size: f64,
    pub feature_names: Vec<String>,
    /// By class code; `None` in regression.
    pub class_names: Option<Vec<String>>,
    /// Form values (codes, column order); `None` = first test row.
    pub sample: Option<Vec<f64>>,
}

/// Context of `knn_entrenamiento.py.j2` (every key is always sent, `None` as `null`
/// when it does not apply), the KNN's equivalent of [`TreeTrainingContext`]
/// (`tests/parity/code/knn_training.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnnTrainingContext {
    /// Built-in slug (`"iris"`…) or `"csv"`.
    pub dataset: String,
    pub dataset_label: String,
    pub csv_file: Option<String>,
    pub csv_target: Option<String>,
    pub csv_sep: Option<String>,
    pub csv_decimal: Option<String>,
    pub csv_encoding: Option<String>,
    /// `"classification"` or `"regression"` (the dataset's task).
    pub task: String,
    pub n_neighbors: u32,
    /// [`crate::knn::KnnWeights::as_str`].
    pub weights: String,
    /// [`crate::knn::KnnMetric::as_str`].
    pub metric: String,
    pub test_size: f64,
    pub feature_names: Vec<String>,
    /// By class code; `None` in regression.
    pub class_names: Option<Vec<String>>,
    /// Form values (codes, column order); `None` = first test row.
    pub sample: Option<Vec<f64>>,
}

/// An f32 control value as the student reads it (`0.2`, not `0.20000000298…`):
/// the shortest decimal of the f32, read as f64 (what Python had as a float).
pub fn f32_as_decimal(v: f32) -> f64 {
    v.to_string().parse().unwrap_or(f64::from(v))
}

/// Context of `red_entrenamiento.py.j2` (🧠, D74). Every key is always sent (null
/// when it does not apply): the template was verified with minijinja in strict mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NnTrainingContext {
    /// Built-in slug or `"csv"`.
    pub dataset: String,
    pub dataset_label: String,
    pub csv_file: Option<String>,
    pub csv_target: Option<String>,
    pub csv_sep: Option<String>,
    pub csv_decimal: Option<String>,
    pub csv_encoding: Option<String>,
    /// `"classification"` or `"regression"`.
    pub task: String,
    /// `[input, hidden…, output]`.
    pub architecture: Vec<usize>,
    pub activation: String,
    pub output_activation: String,
    /// Keras name ([`crate::nn_data::LossKind::as_str`]).
    pub loss: String,
    pub one_hot: bool,
    pub dropout_rate: f64,
    pub optimizer: String,
    pub learning_rate: f64,
    pub batch_size: u32,
    pub epochs: u32,
    pub validation_split: f64,
    pub early_stopping: bool,
    pub patience: Option<u32>,
    pub reduce_lr: bool,
    pub lr_factor: Option<f64>,
    pub reduce_lr_patience: Option<u32>,
    pub feature_names: Vec<String>,
    pub class_names: Option<Vec<String>>,
    /// Form values (column order); `None` = first test row.
    pub sample: Option<Vec<f64>>,
}

/// The dataset part of the 🧠 contexts (how the script loads the data).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptDataset {
    pub dataset: String,
    pub dataset_label: String,
    pub csv_file: Option<String>,
    pub csv_target: Option<String>,
    pub csv_sep: Option<String>,
    pub csv_decimal: Option<String>,
    pub csv_encoding: Option<String>,
    pub feature_names: Vec<String>,
    pub class_names: Option<Vec<String>>,
}

impl NnTrainingContext {
    /// The context of the current 🏗️/⚙️ configuration (also without a model, D74).
    /// `class_names` is only used in classification.
    pub fn new(
        data: ScriptDataset,
        task: crate::dataset::TaskType,
        n_classes: usize,
        arch: &crate::nn_data::NnArchitecture,
        training: &crate::nn_data::NnTraining,
    ) -> Result<Self, crate::nn_data::NnError> {
        use crate::nn_data::{output_size, resolve_training};
        let k = output_size(task, n_classes);
        let resolved = resolve_training(task, arch.output_activation, k)?;
        let mut architecture = vec![data.feature_names.len()];
        architecture.extend(arch.hidden.iter().map(|&h| usize::from(h)));
        architecture.push(k);
        let classification = task == crate::dataset::TaskType::Classification;
        Ok(Self {
            dataset: data.dataset,
            dataset_label: data.dataset_label,
            csv_file: data.csv_file,
            csv_target: data.csv_target,
            csv_sep: data.csv_sep,
            csv_decimal: data.csv_decimal,
            csv_encoding: data.csv_encoding,
            task: if classification {
                "classification"
            } else {
                "regression"
            }
            .to_owned(),
            architecture,
            activation: arch.activation.as_str().to_owned(),
            output_activation: arch.output_activation.as_str().to_owned(),
            loss: resolved.loss.as_str().to_owned(),
            one_hot: resolved.one_hot,
            dropout_rate: f32_as_decimal(arch.dropout),
            optimizer: arch.optimizer.as_str().to_owned(),
            learning_rate: f32_as_decimal(training.learning_rate),
            batch_size: u32::from(arch.batch_size),
            epochs: u32::from(training.epochs),
            validation_split: training.validation_split(),
            early_stopping: training.patience.is_some(),
            patience: training.patience.map(u32::from),
            reduce_lr: training.lr_factor.is_some(),
            lr_factor: training.lr_factor.map(f32_as_decimal),
            reduce_lr_patience: training
                .lr_factor
                .map(|_| u32::try_from(training.reduce_lr_patience()).unwrap_or(u32::MAX)),
            feature_names: data.feature_names,
            class_names: if classification {
                data.class_names
            } else {
                None
            },
            sample: None,
        })
    }
}

/// Context of `paquete_modelo_red.py.j2` and `red_predictor_numpy.py.j2`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NnPackageScriptContext {
    pub dataset: String,
    /// `"classification"` or `"regression"`.
    pub task: String,
    pub json_name: String,
}

/// One class of the 🧠 `LEEME.txt`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NnReadmeClass {
    pub value: usize,
    pub name: String,
}

/// Context of `paquete_leeme_red.txt.j2`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NnPackageReadmeContext {
    pub dataset: String,
    pub task: String,
    pub architecture: Vec<usize>,
    pub activation: String,
    pub output_activation: String,
    pub dropout_rate: f64,
    pub n_params: usize,
    pub feature_names: Vec<String>,
    pub classes: Option<Vec<NnReadmeClass>>,
    /// Rows of the training split (fit + validation).
    pub n_train: usize,
    pub n_test: usize,
}

/// Context of `usar_modelo_arbol.py.j2` ("Código para Usar el Modelo").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeUsageContext {
    /// Readable feature names, in column order.
    pub feature_names: Vec<String>,
}

/// Names of the embedded templates.
pub fn template_names() -> Vec<&'static str> {
    embedded::TEMPLATES.iter().map(|(n, _)| *n).collect()
}

/// Source of an embedded template.
pub fn template_source(name: &str) -> Option<&'static str> {
    embedded::TEMPLATES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
}

/// Python's `repr()` of a string: single quotes unless the text contains `'` and
/// no `"`; escapes `\\`, the quote, `\n`, `\r`, `\t` and other control characters.
pub fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr()` of a template value (str, int, float, bool, None, list, dict).
pub fn py_repr(value: &Value) -> String {
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => "None".to_owned(),
        ValueKind::Bool => if value.is_true() { "True" } else { "False" }.to_owned(),
        ValueKind::String => py_str_repr(value.as_str().unwrap_or_default()),
        ValueKind::Number => {
            if value.is_integer() {
                value.to_string()
            } else {
                f64::try_from(value.clone()).map_or_else(|_| value.to_string(), py_float_repr)
            }
        }
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<String> = value
                .try_iter()
                .map(|it| it.map(|v| py_repr(&v)).collect())
                .unwrap_or_default();
            format!("[{}]", items.join(", "))
        }
        ValueKind::Map => {
            let items: Vec<String> = value
                .try_iter()
                .map(|it| {
                    it.map(|k| {
                        let v = value.get_item(&k).unwrap_or_default();
                        format!("{}: {}", py_repr(&k), py_repr(&v))
                    })
                    .collect()
                })
                .unwrap_or_default();
            format!("{{{}}}", items.join(", "))
        }
        _ => value.to_string(),
    }
}

/// `json.dumps(v, ensure_ascii=False)` (the `tojson_compact` filter of
/// `export_golden.py`).
fn py_json(value: &Value) -> String {
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => "null".to_owned(),
        ValueKind::Bool => if value.is_true() { "true" } else { "false" }.to_owned(),
        ValueKind::String => {
            serde_json::to_string(value.as_str().unwrap_or_default()).unwrap_or_default()
        }
        ValueKind::Number => py_repr(value),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<String> = value
                .try_iter()
                .map(|it| it.map(|v| py_json(&v)).collect())
                .unwrap_or_default();
            format!("[{}]", items.join(", "))
        }
        ValueKind::Map => {
            let items: Vec<String> = value
                .try_iter()
                .map(|it| {
                    it.map(|k| {
                        let v = value.get_item(&k).unwrap_or_default();
                        format!("{}: {}", py_json(&Value::from(k.to_string())), py_json(&v))
                    })
                    .collect()
                })
                .unwrap_or_default();
            format!("{{{}}}", items.join(", "))
        }
        _ => value.to_string(),
    }
}

/// The `pycomment` filter (interface I3): user text (a CSV file name, a dataset
/// label) made safe inside a Python `#` comment or a `"""` docstring.
///
/// 1. Removes `\r` and `\n`, which would end the comment.
/// 2. Removes `"""` repeatedly, so that a removal cannot form a new one
///    (`""\n"` → `"""` after step 1 → nothing).
/// 3. Doubles every backslash: `\N` or `\x` inside a docstring is a `SyntaxError`
///    (an invalid escape); `\\` is read back as one `\`.
pub fn py_comment(text: &str) -> String {
    let mut out: String = text.chars().filter(|&c| c != '\r' && c != '\n').collect();
    while out.contains("\"\"\"") {
        out = out.replace("\"\"\"", "");
    }
    out.replace('\\', "\\\\")
}

fn environment() -> Environment<'static> {
    let mut env = Environment::new();
    if let Ok(syntax) = SyntaxConfig::builder().keep_trailing_newline(true).build() {
        env.set_syntax(syntax);
    }
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    // Strict like jinja2's StrictUndefined: a missing variable is an error even
    // inside a filter.
    env.add_filter("pyrepr", |v: Value| -> Result<String, minijinja::Error> {
        if v.is_undefined() {
            return Err(minijinja::Error::new(
                minijinja::ErrorKind::UndefinedError,
                "variable sin definir en pyrepr",
            ));
        }
        Ok(py_repr(&v))
    });
    env.add_filter("tojson_compact", |v: Value| py_json(&v));
    env.add_filter(
        "pycomment",
        |v: Value| -> Result<String, minijinja::Error> {
            if v.is_undefined() {
                return Err(minijinja::Error::new(
                    minijinja::ErrorKind::UndefinedError,
                    "variable sin definir en pycomment",
                ));
            }
            // `str(value)` in Python: strings as they are, anything else as its text.
            Ok(match v.as_str() {
                Some(s) => py_comment(s),
                None => py_comment(&v.to_string()),
            })
        },
    );
    for (name, source) in embedded::TEMPLATES {
        // A broken template only fails when it is rendered (see `render_named`).
        let _ = env.add_template(name, source);
    }
    env
}

/// Renders `content/code/<name>` with `context` (any serde value: a struct of this
/// module or a `serde_json::Value`).
pub fn render_named<S: Serialize>(name: &str, context: &S) -> Result<String, CodegenError> {
    if template_source(name).is_none() {
        return Err(CodegenError::TemplateNotFound {
            name: name.to_owned(),
        });
    }
    let env = environment();
    let render_err = |e: minijinja::Error| CodegenError::Render {
        name: name.to_owned(),
        message: e.to_string(),
    };
    let tpl = env.get_template(name).map_err(render_err)?;
    tpl.render(Value::from(Serde(context))).map_err(render_err)
}

/// Renders one of the [`CodeTemplate`]s.
pub fn render<S: Serialize>(template: CodeTemplate, context: &S) -> Result<String, CodegenError> {
    render_named(template.file_name(), context)
}

/// Renders a template without variables.
pub fn render_static(template: CodeTemplate) -> Result<String, CodegenError> {
    render(template, &serde_json::json!({}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_like_python() {
        let v = Value::from(Serde(serde_json::json!([
            "a'b", "tulipán", 0.3, 2.5e-7, 3, null, true, 1.0, "x\"y'z"
        ])));
        assert_eq!(
            py_repr(&v),
            r#"["a'b", 'tulipán', 0.3, 2.5e-07, 3, None, True, 1.0, 'x"y\'z']"#
        );
    }

    #[test]
    fn pycomment_is_safe_in_comments_and_docstrings() {
        assert_eq!(py_comment("datos.csv"), "datos.csv");
        assert_eq!(py_comment("a\r\nb\nc"), "abc");
        assert_eq!(py_comment(r#"x"""y"#), "xy");
        // Removing the line break forms a triple quote, and removing one forms another.
        assert_eq!(py_comment("\"\"\n\""), "");
        assert_eq!(py_comment(&"\"".repeat(5)), "\"\""); // like Python's str.replace
        assert_eq!(py_comment(&"\"".repeat(6)), "");
        assert_eq!(py_comment(r"C:\Nuevos\x.csv"), r"C:\\Nuevos\\x.csv");
        let env = environment();
        let out = env
            .render_str(
                "{{ a | pycomment }}|{{ n | pycomment }}",
                Value::from(Serde(serde_json::json!({"a": "p\"\"\"q\n", "n": 3}))),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out, "pq|3");
        assert!(env
            .render_str("{{ missing | pycomment }}", Value::from(()))
            .is_err());
    }

    #[test]
    fn missing_template() {
        assert!(matches!(
            render_named("nope.py.j2", &serde_json::json!({})),
            Err(CodegenError::TemplateNotFound { .. })
        ));
    }

    /// Python's `repr(0.0)` is `0.0`: whole floats stay floats (minijinja's Python
    /// binding passes them as ints; serde_json keeps them as f64).
    #[test]
    fn pyrepr_keeps_whole_floats() {
        let env = environment();
        let out = env
            .render_str(
                "{{ a | pyrepr }} {{ b | pyrepr }} {{ c | pyrepr }} {{ d | pyrepr }}",
                Value::from(Serde(
                    serde_json::json!({"a": 0.0, "b": 1.0, "c": [0.0, 2], "d": -0.0}),
                )),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out, "0.0 1.0 [0.0, 2] -0.0");
        assert_eq!(f32_as_decimal(0.2), 0.2);
        assert_eq!(f32_as_decimal(0.001), 0.001);
    }
}
