//! C14: `sklearn.tree.export_text`, character for character.
//!
//! ```text
//! |--- petal length (cm) <= 2.45
//! |   |--- weights: [31.00, 0.00, 0.00] class: 0
//! |--- petal length (cm) >  2.45
//! ```
//!
//! Internal nodes write `"{name} <= {t:.2f}"` and `"{name} >  {t:.2f}"` (two
//! spaces); leaves `"class: {c}"`, `"weights: [..] class: {c}"` (`show_weights`,
//! counts = fraction × samples) or, in regression, `"value: [v]"`. Python formats
//! with `"{:.2f}"`, which rounds the exact binary value half to even like Rust's
//! `{:.2}`.

use super::DecisionTree;
use crate::model::{argmax, Predictor};

/// Options of [`export_text`] (scikit-learn's defaults except the class names).
#[derive(Debug, Clone, PartialEq)]
pub struct ExportTextOptions<'a> {
    /// Names of the classes **by dataset code** (D40: "class: setosa"). `None`
    /// prints the codes, like scikit-learn without `class_names` ("class: 0").
    pub class_names: Option<&'a [String]>,
    /// `show_weights` (📝 Texto: `true`; the exported .txt: `false`).
    pub show_weights: bool,
    pub decimals: usize,
    pub spacing: usize,
    pub max_depth: usize,
}

impl Default for ExportTextOptions<'_> {
    fn default() -> Self {
        Self {
            class_names: None,
            show_weights: false,
            decimals: 2,
            spacing: 3,
            max_depth: 10,
        }
    }
}

/// `export_text(tree, feature_names=..., class_names=..., show_weights=...)`.
pub fn export_text(
    tree: &DecisionTree,
    feature_names: &[String],
    options: &ExportTextOptions<'_>,
) -> String {
    let mut out = String::new();
    if !tree.nodes().is_empty() {
        recurse(tree, feature_names, options, 0, 1, &mut out);
    }
    out
}

fn class_label(tree: &DecisionTree, node: usize, options: &ExportTextOptions<'_>) -> String {
    let value = &tree.nodes()[node].value;
    let k = argmax(value.iter().copied());
    let code = tree.classes().and_then(|c| c.get(k).copied()).unwrap_or(k);
    match options.class_names.and_then(|n| n.get(code)) {
        Some(name) => name.clone(),
        None => code.to_string(),
    }
}

fn add_leaf(
    tree: &DecisionTree,
    node: usize,
    indent: &str,
    o: &ExportTextOptions<'_>,
    out: &mut String,
) {
    let n = &tree.nodes()[node];
    let d = o.decimals;
    if tree.is_classifier() {
        let class = class_label(tree, node, o);
        if o.show_weights {
            let w = n.n_samples as f64;
            let vals: Vec<String> = n.value.iter().map(|v| format!("{:.d$}", v * w)).collect();
            out.push_str(&format!(
                "{indent} weights: [{}] class: {class}\n",
                vals.join(", ")
            ));
        } else {
            out.push_str(&format!("{indent} class: {class}\n"));
        }
    } else {
        let vals: Vec<String> = n.value.iter().map(|v| format!("{v:.d$}")).collect();
        out.push_str(&format!("{indent} value: [{}]\n", vals.join(", ")));
    }
}

/// `_compute_depth`: depth of the subtree rooted at `node` (a leaf is 1).
fn subtree_depth(tree: &DecisionTree, node: usize) -> usize {
    match tree.nodes()[node].split {
        Some(s) => 1 + subtree_depth(tree, s.left).max(subtree_depth(tree, s.right)),
        None => 1,
    }
}

fn recurse(
    tree: &DecisionTree,
    names: &[String],
    o: &ExportTextOptions<'_>,
    node: usize,
    depth: usize,
    out: &mut String,
) {
    let unit = format!("|{}", " ".repeat(o.spacing));
    let full = unit.repeat(depth);
    let cut = full.len().saturating_sub(o.spacing);
    let indent = format!("{}{}", &full[..cut], "-".repeat(o.spacing));
    if depth <= o.max_depth + 1 {
        match tree.nodes()[node].split {
            Some(s) => {
                let fallback = format!("feature_{}", s.feature);
                let name = names.get(s.feature).unwrap_or(&fallback);
                let d = o.decimals;
                let t = format!("{:.d$}", s.threshold);
                out.push_str(&format!("{indent} {name} <= {t}\n"));
                recurse(tree, names, o, s.left, depth + 1, out);
                out.push_str(&format!("{indent} {name} >  {t}\n"));
                recurse(tree, names, o, s.right, depth + 1, out);
            }
            None => add_leaf(tree, node, &indent, o, out),
        }
    } else {
        let sd = subtree_depth(tree, node);
        if sd == 1 {
            add_leaf(tree, node, &indent, o, out);
        } else {
            out.push_str(&format!("{indent} truncated branch of depth {sd}\n"));
        }
    }
}
