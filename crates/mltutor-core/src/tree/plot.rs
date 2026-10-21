//! C13: the boxes of `plot_tree(model, feature_names, class_names, filled=True,
//! rounded=True, proportion=True, impurity=True)` — text, fill colour and position —
//! so that the UI can draw them as an SVG (D41).
//!
//! Text of a node (lines joined by `\n` in Python):
//!
//! * internal nodes: `"{feature} <= {round(threshold, 3)}"`;
//! * `"{criterion} = {round(impurity, 3)}"`;
//! * `"samples = {round(100·n/n_root, 1)}%"`;
//! * `"value = [0.295, 0.352, 0.352]"` (`np.around(value, 3)`; regression without
//!   brackets);
//! * `"class = {name}"` (classification with class names).
//!
//! `str(round(x, 3))` is the correctly rounded decimal printed with the shortest
//! repr; `np.around` is `rint(x·1000)/1000`, which is not always the same number.
//!
//! Colours (`filled`): classification, `_color_brew(n_classes)[argmax]` blended with
//! white by `alpha = (v1 − v2)/(1 − v2)` (the two largest fractions); regression,
//! `_color_brew(1)[0]` with `alpha = (value − min)/(max − min)` over every node.

use serde::{Deserialize, Serialize};

use super::layout::{buchheim, TreeLayout};
use super::{DecisionTree, TreeSummary};
use crate::grid::np_arange;
use crate::model::{argmax, Predictor};
use crate::pyfmt::py_float_repr;

/// One box of the diagram (contract `TreeDiagramNode`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagramNode {
    pub id: usize,
    pub parent: Option<usize>,
    /// Whether it is the left ("True") child of its parent.
    pub is_left: bool,
    pub depth: usize,
    /// Buchheim coordinates (box units; `y` = depth).
    pub x: f64,
    pub y: f64,
    /// Position in the axes of `plot_tree` (`[0, 1]²`, `y` upwards).
    pub xy_axes: [f64; 2],
    /// Text lines of the box, exactly as `plot_tree`.
    pub lines: Vec<String>,
    /// `"#rrggbb"`.
    pub fill: String,
    pub leaf: bool,
    pub feature: Option<usize>,
    pub threshold: Option<f64>,
    pub impurity: f64,
    pub n_samples: usize,
    /// `tree_.value` (fractions or `[value]`).
    pub value: Vec<f64>,
    /// Class counts (classification).
    pub value_counts: Option<Vec<u32>>,
    /// Majority class code (classification).
    pub class_code: Option<usize>,
}

/// "True  " / "  False" on the two edges of the root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeLabel {
    pub text: String,
    /// Midpoint of the edge, axes coordinates.
    pub xy_axes: [f64; 2],
    /// `"right"` (True) or `"left"` (False).
    pub ha: String,
}

/// The whole diagram (contract `TreeDiagram`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreeDiagram {
    /// In node order (= pre-order = `plot_tree`'s drawing order).
    pub nodes: Vec<DiagramNode>,
    pub edge_labels: Vec<EdgeLabel>,
    pub max_x: f64,
    pub max_y: f64,
    /// Criterion name written in the boxes.
    pub criterion: String,
    pub summary: TreeSummary,
}

/// `str(round(x, ndigits))` of Python for a float.
pub fn py_round_str(x: f64, ndigits: usize) -> String {
    if !x.is_finite() {
        return py_float_repr(x);
    }
    let s = format!("{x:.ndigits$}");
    py_float_repr(s.parse::<f64>().unwrap_or(x))
}

/// `np.around(x, decimals)` for `decimals ≥ 0`: `rint(x·10^d)/10^d`.
pub fn np_around(x: f64, decimals: i32) -> f64 {
    let f = 10f64.powi(decimals);
    (x * f).round_ties_even() / f
}

/// `_color_brew(n)`: `n` RGB colours with equally spaced hues.
pub fn color_brew(n: usize) -> Vec<[u8; 3]> {
    let (s, v) = (0.75_f64, 0.9_f64);
    let c = s * v;
    let m = v - c;
    let n = n.max(1);
    np_arange(25.0, 385.0, 360.0 / n as f64)
        .into_iter()
        .map(|h| {
            let h = h.trunc(); // .astype(int)
            let h_bar = h / 60.0;
            let x = c * (1.0 - ((h_bar % 2.0) - 1.0).abs());
            let rgb = [
                (c, x, 0.0),
                (x, c, 0.0),
                (0.0, c, x),
                (0.0, x, c),
                (x, 0.0, c),
                (c, 0.0, x),
                (c, x, 0.0),
            ];
            let (r, g, b) = rgb[(h_bar as usize).min(6)];
            let to = |u: f64| (255.0 * (u + m)) as u8;
            [to(r), to(g), to(b)]
        })
        .collect()
}

fn blend(color: [u8; 3], alpha: f64) -> String {
    let ch = |c: u8| -> u8 {
        let v = (alpha * f64::from(c) + (1.0 - alpha) * 255.0).round_ties_even();
        v.clamp(0.0, 255.0) as u8
    };
    format!(
        "#{:02x}{:02x}{:02x}",
        ch(color[0]),
        ch(color[1]),
        ch(color[2])
    )
}

/// `get_color` for a classification node (`value` = fractions).
fn class_color(palette: &[[u8; 3]], value: &[f64]) -> String {
    let k = argmax(value.iter().copied());
    let color = palette.get(k).copied().unwrap_or([255, 255, 255]);
    let mut sorted = value.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let alpha = if sorted.len() < 2 {
        0.0
    } else {
        (sorted[0] - sorted[1]) / (1.0 - sorted[1])
    };
    blend(color, alpha)
}

/// `str(np.array(words, dtype="S32"))`: NumPy's 1-D array printing (line width 75,
/// continuation lines indented by one space) of byte strings `b'...'`.
fn numpy_bytes_array_str(items: &[String]) -> String {
    const LINE_WIDTH: usize = 75;
    let words: Vec<String> = items.iter().map(|s| format!("b'{s}'")).collect();
    let prefix = " ";
    let elem_width = LINE_WIDTH - 1; // len("]")
    let mut s = String::new();
    let mut line = prefix.to_owned();
    for (i, word) in words.iter().enumerate() {
        let last = i + 1 == words.len();
        let needs_wrap = line.len() + word.len() > elem_width && line.len() > prefix.len();
        if needs_wrap {
            s.push_str(line.trim_end());
            s.push('\n');
            line = prefix.to_owned();
        }
        line.push_str(word);
        if !last {
            line.push(' ');
        }
    }
    s.push_str(&line);
    format!("[{}]", &s[prefix.len()..])
}

/// The `value = …` text of a node.
fn value_text(value: &[f64], regression: bool) -> String {
    let items: Vec<String> = value
        .iter()
        .map(|&v| py_float_repr(np_around(v, 3)))
        .collect();
    let mut t = numpy_bytes_array_str(&items)
        .replace("b'", "'")
        .replace("' '", ", ")
        .replace('\'', "");
    if regression {
        t = t.replace(['[', ']'], "");
    }
    t.replace("\n ", "\n")
}

/// Rounding noise below zero shown as 0 (see [`node_lines`]).
fn display_impurity(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

/// The text lines of node `id` (`node_to_str`).
pub fn node_lines(
    tree: &DecisionTree,
    id: usize,
    feature_names: &[String],
    class_names: Option<&[String]>,
) -> Vec<String> {
    let nodes = tree.nodes();
    let node = &nodes[id];
    let mut text = String::new();
    if let Some(s) = node.split {
        let fallback = format!("x[{}]", s.feature);
        let name = feature_names.get(s.feature).unwrap_or(&fallback);
        text.push_str(&format!("{name} <= {}\n", py_round_str(s.threshold, 3)));
    }
    // An impurity is never negative; MSE computed as `Σy²/n − ȳ²` can give
    // −3e-14 for a constant leaf, which Python prints as "-0.0" (the FMA of the
    // macOS arm64 wheel happens to give 0.0). Shown as 0.
    text.push_str(&format!(
        "{} = {}\n",
        tree.config().criterion.as_str(),
        py_round_str(display_impurity(node.impurity), 3)
    ));
    let n0 = nodes.first().map_or(1, |n| n.n_samples) as f64;
    let percent = 100.0 * node.n_samples as f64 / n0;
    text.push_str(&format!("samples = {}%\n", py_round_str(percent, 1)));
    text.push_str(&format!(
        "value = {}\n",
        value_text(&node.value, !tree.is_classifier())
    ));
    if let (Some(names), Some(classes)) = (class_names, tree.classes()) {
        let k = argmax(node.value.iter().copied());
        let code = classes.get(k).copied().unwrap_or(k);
        let name = names.get(code).cloned().unwrap_or_else(|| code.to_string());
        text.push_str(&format!("class = {name}"));
    }
    let text = text.strip_suffix('\n').unwrap_or(&text);
    text.split('\n').map(str::to_owned).collect()
}

/// Fill colours of every node (`get_fill_color`).
pub fn fill_colors(tree: &DecisionTree) -> Vec<String> {
    let nodes = tree.nodes();
    match tree.classes() {
        Some(classes) => {
            let palette = color_brew(classes.len());
            nodes
                .iter()
                .map(|n| class_color(&palette, &n.value))
                .collect()
        }
        None => {
            let palette = color_brew(1);
            let values: Vec<f64> = nodes.iter().flat_map(|n| n.value.iter().copied()).collect();
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if max.is_nan() || min.is_nan() || max <= min {
                // A single distinct value: scikit-learn falls back to the
                // classification rule with one value, alpha 0 (white).
                return nodes.iter().map(|_| blend(palette[0], 0.0)).collect();
            }
            nodes
                .iter()
                .map(|n| {
                    let v = n.value.first().copied().unwrap_or(min);
                    blend(palette[0], (v - min) / (max - min))
                })
                .collect()
        }
    }
}

/// C13: the diagram of `tree`. `class_names` by dataset code (Python passes them in
/// classification only; with `None` the boxes have no `class = …` line).
///
/// Difference with Python: when a class is missing from the training rows, Python
/// indexes the dataset's names with the tree's class index (a wrong name); here
/// the code is mapped to its own name.
pub fn diagram(
    tree: &DecisionTree,
    feature_names: &[String],
    class_names: Option<&[String]>,
) -> TreeDiagram {
    let layout: TreeLayout = buchheim(tree);
    let fills = fill_colors(tree);
    let nodes = tree.nodes();
    let mut parent: Vec<Option<(usize, bool)>> = vec![None; nodes.len()];
    for (i, n) in nodes.iter().enumerate() {
        if let Some(s) = n.split {
            parent[s.left] = Some((i, true));
            parent[s.right] = Some((i, false));
        }
    }
    let mut edge_labels = Vec::new();
    let out: Vec<DiagramNode> = nodes
        .iter()
        .enumerate()
        .map(|(id, n)| {
            let xy = layout.axes_xy(id);
            if let Some((0, is_left)) = parent[id] {
                let p = layout.axes_xy(0);
                let mid = [(p[0] + xy[0]) / 2.0, (p[1] + xy[1]) / 2.0];
                let (text, ha) = if is_left {
                    ("True  ", "right")
                } else {
                    ("  False", "left")
                };
                edge_labels.push(EdgeLabel {
                    text: text.to_owned(),
                    xy_axes: mid,
                    ha: ha.to_owned(),
                });
            }
            DiagramNode {
                id,
                parent: parent[id].map(|(p, _)| p),
                is_left: parent[id].is_some_and(|(_, l)| l),
                depth: n.depth,
                x: layout.x[id],
                y: layout.y[id] as f64,
                xy_axes: xy,
                lines: node_lines(tree, id, feature_names, class_names),
                fill: fills
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| "#ffffff".to_owned()),
                leaf: n.is_leaf(),
                feature: n.split.map(|s| s.feature),
                threshold: n.split.map(|s| s.threshold),
                impurity: display_impurity(n.impurity),
                n_samples: n.n_samples,
                value: n.value.clone(),
                value_counts: tree.is_classifier().then(|| n.counts.clone()),
                class_code: tree
                    .classes()
                    .and_then(|c| c.get(argmax(n.value.iter().copied())).copied()),
            }
        })
        .collect();
    TreeDiagram {
        nodes: out,
        edge_labels,
        max_x: layout.max_x,
        max_y: layout.max_y,
        criterion: tree.config().criterion.as_str().to_owned(),
        summary: tree.summary(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brew_like_sklearn() {
        assert_eq!(
            color_brew(3),
            vec![[229, 129, 57], [57, 229, 129], [129, 57, 229]]
        );
        assert_eq!(color_brew(1)[0], [229, 129, 57]);
    }

    #[test]
    fn python_rounding() {
        assert_eq!(py_round_str(0.6644897959183673, 3), "0.664");
        assert_eq!(py_round_str(0.0, 3), "0.0");
        assert_eq!(py_round_str(100.0, 1), "100.0");
        assert_eq!(py_round_str(2.449999988079071, 3), "2.45");
        assert_eq!(np_around(0.29523809523809524, 3), 0.295);
    }

    #[test]
    fn numpy_wraps_long_arrays() {
        let items: Vec<String> = (0..30).map(|_| "0.0".to_owned()).collect();
        let s = numpy_bytes_array_str(&items);
        assert!(s.contains('\n'));
        assert!(s.lines().all(|l| l.len() <= 75));
    }
}
