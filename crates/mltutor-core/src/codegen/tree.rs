//! C15: the decision tree as Python `if/else` code (💾 Exportar).
//!
//! * [`literal_context`]: Python's `export_tree_as_python_code` byte for byte
//!   (`arbol_python.py.j2`), with its errors: it does not compile with names such
//!   as `sepal length (cm)` and returns the class code.
//! * [`corrected_context`] (D44, the one the app shows): valid identifiers
//!   ([`py_identifiers`]), class names, the body inside the function and exact
//!   thresholds ([`f64_switch_threshold`]): scikit-learn compares `float32(x)`, so
//!   the code writes the largest float64 `w` with `x <= w ⇔ float32(x) <= t`.

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use super::py_str_repr;
use crate::model::{argmax, Predictor};
use crate::pyfmt::py_float_repr;
use crate::stats::pairwise_sum;
use crate::tree::DecisionTree;

/// Context of `arbol_python.py.j2` (literal) — every key is required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreePythonContext {
    /// `"predecir_clase"` or `"predecir_valor"`.
    pub function_name: String,
    /// Feature names as they are (not valid identifiers in general).
    pub params: Vec<String>,
    pub classifier: bool,
    pub n_features: usize,
    /// `tree_.max_depth` (the real depth).
    pub max_depth: usize,
    pub node_count: usize,
    pub body: String,
}

/// Context of `arbol_python_corregido.py.j2` (D44).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreePythonCorrectedContext {
    pub function_name: String,
    /// Python identifiers ([`py_identifiers`]).
    pub params: Vec<String>,
    /// Readable names, same order.
    pub display_names: Vec<String>,
    pub classifier: bool,
    pub n_features: usize,
    pub max_depth: usize,
    pub node_count: usize,
    pub body: String,
}

/// Next float32 towards +∞ (finite input).
fn f32_next_up(f: f32) -> f32 {
    if f.is_nan() || f == f32::INFINITY {
        return f;
    }
    if f == 0.0 {
        return f32::from_bits(1);
    }
    let b = f.to_bits();
    f32::from_bits(if f > 0.0 { b + 1 } else { b - 1 })
}

fn f32_next_down(f: f32) -> f32 {
    -f32_next_up(-f)
}

fn f64_next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    let b = x.to_bits();
    f64::from_bits(if x > 0.0 { b - 1 } else { b + 1 })
}

/// The largest float32 `f` with `f <= t` (D47: the ONNX thresholds).
pub fn f32_floor(t: f64) -> f32 {
    let mut f = t as f32;
    if f64::from(f) > t {
        f = f32_next_down(f);
    }
    f
}

/// D44: the largest float64 `w` such that `x <= w ⇔ float32(x) <= t` for every
/// float64 `x`. With `f` = the largest float32 ≤ `t`, every float64 below the
/// rounding boundary `m = (f + next(f))/2` (exact in f64) rounds to ≤ `f`; at the
/// boundary itself round-half-even decides.
pub fn f64_switch_threshold(t: f64) -> f64 {
    let f = f32_floor(t);
    let next = f32_next_up(f);
    let m = (f64::from(f) + f64::from(next)) / 2.0;
    if m as f32 == f {
        m
    } else {
        f64_next_down(m)
    }
}

/// `keyword.iskeyword` of Python 3.11 (plus `True`/`False`/`None`).
fn is_python_keyword(s: &str) -> bool {
    matches!(
        s,
        "False"
            | "None"
            | "True"
            | "and"
            | "as"
            | "assert"
            | "async"
            | "await"
            | "break"
            | "class"
            | "continue"
            | "def"
            | "del"
            | "elif"
            | "else"
            | "except"
            | "finally"
            | "for"
            | "from"
            | "global"
            | "if"
            | "import"
            | "in"
            | "is"
            | "lambda"
            | "nonlocal"
            | "not"
            | "or"
            | "pass"
            | "raise"
            | "return"
            | "try"
            | "while"
            | "with"
            | "yield"
    )
}

/// D44: valid, unique Python identifiers for the feature names, following the
/// identifier rules of Python 3 (PEP 3131):
///
/// 1. NFKC, which is what Python applies to identifiers when it parses them
///    (`ﬁn` and `fin`, or `m²` and `m2`, are the same name to Python);
/// 2. every run of characters that are not `XID_Continue` → one `_`, then the
///    `_` at both ends are stripped (`"superficie (m²)"` → `superficie_m2`);
/// 3. `"x"` if empty, an `x_` prefix if the first character is not
///    `XID_Start` (a digit, a combining mark), a `_` suffix for keywords;
/// 4. `_2`, `_3`… when the normalized name is already taken.
///
/// `py_identifier` of `v1.0.0:tools/golden_trees.py` is the same rule.
pub fn py_identifiers(names: &[String]) -> Vec<String> {
    let mut used: Vec<String> = Vec::with_capacity(names.len());
    for name in names {
        let mut s = String::with_capacity(name.len());
        let mut in_run = false;
        for c in name.nfkc() {
            if unicode_ident::is_xid_continue(c) {
                s.push(c);
                in_run = false;
            } else if !in_run {
                s.push('_');
                in_run = true;
            }
        }
        let mut s = s.trim_matches('_').to_owned();
        if s.is_empty() {
            s = "x".to_owned();
        }
        if !s.chars().next().is_some_and(unicode_ident::is_xid_start) {
            s = format!("x_{s}");
        }
        if is_python_keyword(&s) {
            s.push('_');
        }
        // Joining NFKC pieces with `_` keeps them NFKC, but normalize again so
        // the uniqueness check compares exactly what Python will compare.
        let s: String = s.nfkc().collect();
        let base = s.clone();
        let mut k = 2;
        let mut s = s;
        while used.contains(&s) {
            s = format!("{base}_{k}");
            k += 1;
        }
        used.push(s);
    }
    used
}

/// How the body is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Style {
    Literal,
    Corrected,
}

/// The recursive `if/else` body (`tree_python_body` of `v1.0.0:tools/golden_trees.py`).
/// `names` are the names written in the comparisons; `leaf_labels[k]` the text
/// returned for tree class `k` (already quoted).
fn body(
    tree: &DecisionTree,
    names: &[String],
    leaf_labels: Option<&[String]>,
    style: Style,
) -> String {
    fn rec(
        tree: &DecisionTree,
        names: &[String],
        labels: Option<&[String]>,
        style: Style,
        node: usize,
        depth: usize,
        out: &mut String,
    ) {
        let ind = "    ".repeat(depth);
        let n = &tree.nodes()[node];
        match n.split {
            None => {
                if let Some(labels) = labels {
                    let k = argmax(n.value.iter().copied());
                    let p = n.value.get(k).copied().unwrap_or(0.0) / pairwise_sum(&n.value);
                    let label = labels.get(k).cloned().unwrap_or_default();
                    out.push_str(&format!("{ind}return {label}  # Probabilidad: {p:.3}"));
                } else {
                    let v = n.value.first().copied().unwrap_or(f64::NAN);
                    match style {
                        Style::Literal => out.push_str(&format!("{ind}return {v:.6}")),
                        Style::Corrected => {
                            out.push_str(&format!("{ind}return {}", py_float_repr(v)));
                        }
                    }
                }
            }
            Some(s) => {
                let fallback = format!("feature_{}", s.feature);
                let name = names.get(s.feature).unwrap_or(&fallback);
                let t = match style {
                    Style::Literal => format!("{:.6}", s.threshold),
                    Style::Corrected => py_float_repr(f64_switch_threshold(s.threshold)),
                };
                out.push_str(&format!("{ind}if {name} <= {t}:\n"));
                rec(tree, names, labels, style, s.left, depth + 1, out);
                out.push('\n');
                out.push_str(&format!("{ind}else:\n"));
                rec(tree, names, labels, style, s.right, depth + 1, out);
            }
        }
    }
    let mut out = String::new();
    if !tree.nodes().is_empty() {
        let base = if style == Style::Corrected { 1 } else { 0 };
        rec(tree, names, leaf_labels, style, 0, base, &mut out);
    }
    out
}

fn function_name(tree: &DecisionTree) -> &'static str {
    if tree.is_classifier() {
        "predecir_clase"
    } else {
        "predecir_valor"
    }
}

/// Context of the literal template. `class_labels` = `str(classes_[k])` by
/// dataset code (`None`: the codes, as for the built-in datasets).
pub fn literal_context(
    tree: &DecisionTree,
    feature_names: &[String],
    class_labels: Option<&[String]>,
) -> TreePythonContext {
    let labels: Option<Vec<String>> = tree.classes().map(|classes| {
        classes
            .iter()
            .map(|&c| {
                let s = class_labels
                    .and_then(|l| l.get(c).cloned())
                    .unwrap_or_else(|| c.to_string());
                format!("\"{s}\"")
            })
            .collect()
    });
    let s = tree.summary();
    TreePythonContext {
        function_name: function_name(tree).to_owned(),
        params: feature_names.to_vec(),
        classifier: tree.is_classifier(),
        n_features: feature_names.len(),
        max_depth: s.max_depth,
        node_count: s.node_count,
        body: body(tree, feature_names, labels.as_deref(), Style::Literal),
    }
}

/// Context of the corrected template (D44). `class_names` by dataset code.
pub fn corrected_context(
    tree: &DecisionTree,
    feature_names: &[String],
    class_names: Option<&[String]>,
) -> TreePythonCorrectedContext {
    let idents = py_identifiers(feature_names);
    let labels: Option<Vec<String>> = tree.classes().map(|classes| {
        classes
            .iter()
            .map(|&c| {
                let name = class_names
                    .and_then(|n| n.get(c).cloned())
                    .unwrap_or_else(|| c.to_string());
                py_str_repr(&name)
            })
            .collect()
    });
    let s = tree.summary();
    TreePythonCorrectedContext {
        function_name: function_name(tree).to_owned(),
        body: body(tree, &idents, labels.as_deref(), Style::Corrected),
        params: idents,
        display_names: feature_names.to_vec(),
        classifier: tree.is_classifier(),
        n_features: feature_names.len(),
        max_depth: s.max_depth,
        node_count: s.node_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_like_python() {
        let names: Vec<String> = ["sepal length (cm)", "1st", "class", "a b", "a-b", "", "año"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            py_identifiers(&names),
            vec![
                "sepal_length_cm",
                "x_1st",
                "class_",
                "a_b",
                "a_b_2",
                "x",
                "año"
            ]
        );
    }

    /// Python identifiers are XID_Start XID_Continue* after NFKC: `²` is
    /// alphanumeric but not XID_Continue (`superficie_m²` does not compile), and
    /// `ﬁn` is `fin` to Python. Checked with `str.isidentifier` and `compile`
    /// in `v1.0.0:tools/golden_trees.py` (`py_identifier`).
    #[test]
    fn identifiers_are_valid_python_3() {
        let names: Vec<String> = [
            "superficie (m²)",
            "ﬁn",
            "fin",
            "e\u{301}",
            "é",
            "\u{301}acento",
            "½ taza",
            "x\u{b7}y",
            "Ⅸ",
            "数据",
            "€",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            py_identifiers(&names),
            vec![
                "superficie_m2",
                "fin",
                "fin_2",
                "é",
                "é_2",
                "x_\u{301}acento",
                "x_1_2_taza",
                "x\u{b7}y",
                "IX",
                "数据",
                "x"
            ]
        );
    }

    #[test]
    fn switch_threshold_routes_like_float32() {
        for &t in &[
            2.449999988079071,
            0.005111072678118944,
            -117.9,
            1.5,
            -0.03205987624824047,
        ] {
            let w = f64_switch_threshold(t);
            assert!((w as f32) as f64 <= t, "{t}");
            assert!(f64::from((f64_next_up(w)) as f32) > t, "{t}");
        }
        assert_eq!(f64_switch_threshold(2.449999988079071), 2.4499999284744263);
    }

    fn f64_next_up(x: f64) -> f64 {
        -f64_next_down(-x)
    }
}
