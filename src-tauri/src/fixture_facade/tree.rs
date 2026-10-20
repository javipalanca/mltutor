//! Decision trees of the [`super::FixtureFacade`] (fase3-arboles.md §5–§6).
//!
//! Training looks up `trees/<case>.json` and replays it: the split, the arrays of `tree_`,
//! predictions, probabilities, metrics, curves, `export_text`, the boxes of `plot_tree`,
//! the permutation importance and the generated code are scikit-learn's exact values.
//! Predictions of new samples (form, grid) walk the stored tree with scikit-learn's rule
//! (`float32(x) <= threshold`). The views shared with F2 come from [`FixtureModel`].

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde_json::Value;

use super::frame::Frame;
use super::model::{indices, FixtureModel};
use super::{floats, strings};
use crate::core_facade::{
    check_cancelled, ModelHandle, ModelSummary, ModelView, Progress, TrainSpec,
};
use crate::ipc::{
    to_u32, ClassProbability, CodeTemplate, CoefficientsView, CommandError, DatasetId,
    DecisionGrid, DecisionPath, Evaluation, ExportFormat, FeatureImportanceView, GridValues,
    ImportanceMethod, ImportanceRow, Matrix, ModelConfig, PathStep, PredictionForm,
    PredictionResult, ProbabilityView, ResidualsView, RocData, RulesKind, TaskType, TreeCriterion,
    TreeDiagram, TreeDiagramNode, TreeSummary, TreeTask,
};

/// Name of the fixture of a configuration (`iris_gini_d3`, `wine_entropy_d3_mss10`…).
pub fn case_name(id: DatasetId, config: ModelConfig) -> Option<String> {
    let ModelConfig::Tree {
        task,
        criterion,
        max_depth,
        min_samples_split,
    } = config
    else {
        return None;
    };
    if id == DatasetId::Iris
        && task == TreeTask::Regression
        && criterion == TreeCriterion::SquaredError
        && (max_depth, min_samples_split) == (3, 2)
    {
        return Some("iris_regression_mismatch".to_owned());
    }
    let mss = if min_samples_split == 2 {
        String::new()
    } else {
        format!("_mss{min_samples_split}")
    };
    Some(format!(
        "{}_{}_d{max_depth}{mss}",
        id.slug(),
        criterion.sklearn_name()
    ))
}

fn unavailable(reason: impl std::fmt::Display) -> CommandError {
    CommandError::training(format!("FixtureFacade: {reason}"))
}

/// The arrays of `tree_` (children −1 = leaf).
#[derive(Debug, Clone)]
pub struct FixtureTree {
    left: Vec<i64>,
    right: Vec<i64>,
    feature: Vec<i64>,
    threshold: Vec<f64>,
    impurity: Vec<f64>,
    n_samples: Vec<u32>,
    /// Class fractions or `[value]`, per node.
    value: Vec<Vec<f64>>,
    value_counts: Option<Vec<Vec<u32>>>,
    max_depth: u32,
}

fn ints(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(-1)).collect())
        .unwrap_or_default()
}

fn node_values(value: &Value) -> Vec<Vec<f64>> {
    value
        .as_array()
        .map(|nodes| {
            nodes
                .iter()
                .map(|n| match n.as_array() {
                    // `[[v]]` (regression, n_outputs = 1) or `[f0, f1…]`.
                    Some(inner) if inner.first().is_some_and(Value::is_array) => floats(&inner[0]),
                    Some(_) => floats(n),
                    None => vec![n.as_f64().unwrap_or(f64::NAN)],
                })
                .collect()
        })
        .unwrap_or_default()
}

impl FixtureTree {
    pub fn parse(tree: &Value) -> Self {
        let counts = tree["value_counts"].as_array().map(|nodes| {
            nodes
                .iter()
                .map(|n| {
                    floats(n)
                        .iter()
                        .map(|c| c.round().max(0.0) as u32)
                        .collect()
                })
                .collect()
        });
        Self {
            left: ints(&tree["children_left"]),
            right: ints(&tree["children_right"]),
            feature: ints(&tree["feature"]),
            threshold: floats(&tree["threshold"]),
            impurity: floats(&tree["impurity"]),
            n_samples: floats(&tree["n_node_samples"])
                .iter()
                .map(|n| *n as u32)
                .collect(),
            value: node_values(&tree["value"]),
            value_counts: counts,
            max_depth: tree["max_depth"].as_u64().map_or(0, |d| d as u32),
        }
    }

    fn len(&self) -> usize {
        self.left.len()
    }

    fn is_leaf(&self, node: usize) -> bool {
        self.left.get(node).is_none_or(|&l| l < 0)
    }

    /// Nodes from the root to the leaf of `x` (sklearn: `float32(x) <= threshold`).
    pub fn path(&self, x: &[f64]) -> Vec<usize> {
        let mut node = 0;
        let mut nodes = vec![0];
        while !self.is_leaf(node) && node < self.len() {
            let feature = self.feature[node].max(0) as usize;
            let value = f64::from(x.get(feature).copied().unwrap_or(f64::NAN) as f32);
            node = if value <= self.threshold[node] {
                self.left[node]
            } else {
                self.right[node]
            } as usize;
            nodes.push(node);
        }
        nodes
    }

    fn leaf(&self, x: &[f64]) -> usize {
        self.path(x).last().copied().unwrap_or(0)
    }

    fn argmax(&self, node: usize) -> usize {
        let mut best = 0;
        for (i, v) in self.value[node].iter().enumerate() {
            if *v > self.value[node][best] {
                best = i;
            }
        }
        best
    }

    fn summary(&self, n_features: usize) -> TreeSummary {
        TreeSummary {
            node_count: to_u32(self.len()),
            n_leaves: to_u32((0..self.len()).filter(|&n| self.is_leaf(n)).count()),
            max_depth: self.max_depth,
            n_features: to_u32(n_features),
        }
    }

    /// Depth and parent of each node (pre-order arrays).
    fn parents(&self) -> Vec<(u32, Option<u32>, bool)> {
        let mut out = vec![(0, None, false); self.len()];
        for node in 0..self.len() {
            if self.is_leaf(node) {
                continue;
            }
            let depth = out[node].0 + 1;
            for (child, is_left) in [(self.left[node], true), (self.right[node], false)] {
                if let Some(slot) = usize::try_from(child).ok().and_then(|c| out.get_mut(c)) {
                    *slot = (depth, Some(to_u32(node)), is_left);
                }
            }
        }
        out
    }

    /// Simple tidy layout (leaves at consecutive x, parents centred) for fixtures without
    /// `plot_tree` boxes.
    fn fallback_layout(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.len()];
        let mut next = 0.0;
        fn walk(tree: &FixtureTree, node: usize, x: &mut [f64], next: &mut f64) {
            if tree.is_leaf(node) {
                x[node] = *next;
                *next += 1.0;
                return;
            }
            let (l, r) = (tree.left[node] as usize, tree.right[node] as usize);
            walk(tree, l, x, next);
            walk(tree, r, x, next);
            x[node] = (x[l] + x[r]) / 2.0;
        }
        if self.len() > 0 {
            walk(self, 0, &mut x, &mut next);
        }
        x
    }
}

/// A tree replayed from `trees/<case>.json`.
pub struct TreeFixtureModel {
    base: FixtureModel,
    id: DatasetId,
    label: String,
    test_size: f64,
    config: ModelConfig,
    tree: FixtureTree,
    /// `classes_` (codes) of a classifier.
    classes: Option<Vec<usize>>,
}

/// `FixtureDataset::train` for `ModelConfig::Tree`.
pub fn train(
    root: &Path,
    id: DatasetId,
    frame: &Frame,
    dataset_task: TaskType,
    spec: &TrainSpec,
    progress: &Progress<'_>,
    cancel: &AtomicBool,
) -> Result<ModelHandle, CommandError> {
    progress("Entrenando modelo...", Some(10.0));
    let name = case_name(id, spec.config.clone()).ok_or_else(|| unavailable("no es un árbol"))?;
    let read = |relative: &str| -> Result<Value, CommandError> {
        let path = root.join(relative);
        let text = std::fs::read_to_string(&path)
            .map_err(|_| unavailable(format!("no hay fixture {relative}")))?;
        serde_json::from_str(&text).map_err(|e| CommandError::internal(format!("{relative}: {e}")))
    };
    let fixture = read(&format!("trees/{name}.json"))?;
    if (fixture["input"]["test_size"].as_f64().unwrap_or(0.3) - spec.test_size).abs() > 1e-9 {
        return Err(unavailable("solo hay fixtures con test_size = 0.3"));
    }
    check_cancelled(cancel)?;
    let mut expected = fixture["expected"].clone();
    if expected["trained"] != Value::Bool(true) {
        return Err(unavailable(format!("{name}.json no entrena")));
    }
    // The F2 snippets (confusion matrix, boundary…) of the shared views.
    let code = read("code/regression.json")?["expected"].take();
    let tree = FixtureTree::parse(&expected["tree"]);
    let classifier = spec.config.task_on(dataset_task) == TaskType::Classification;
    let class_names = expected["class_names"]
        .as_array()
        .map(|_| strings(&expected["class_names"]))
        .or_else(|| match &frame.target {
            super::frame::Target::Classes { names, .. } => Some(names.clone()),
            super::frame::Target::Numeric(_) => None,
        });
    let classes = classifier.then(|| indices(&expected["classes"]));
    // The evaluation of regression trees reads `evaluation`; Iris in regression keeps it
    // under the same key.
    if expected["evaluation"].is_null() {
        expected["evaluation"] = Value::Object(serde_json::Map::new());
    }
    let base = FixtureModel::for_tree(
        spec.config.clone(),
        dataset_task,
        frame,
        expected,
        code,
        class_names,
        Vec::new(),
    );
    Ok(Arc::new(TreeFixtureModel {
        base,
        id,
        label: spec.dataset_label.clone(),
        test_size: spec.test_size,
        config: spec.config.clone(),
        tree,
        classes,
    }))
}

/// Replaces the codes of `class: N` at the end of the lines by the class names (D40).
pub fn with_class_names(text: &str, names: &[String]) -> String {
    text.lines()
        .map(|line| match line.rsplit_once("class: ") {
            Some((head, code)) => {
                match code.trim().parse::<usize>().ok().and_then(|c| names.get(c)) {
                    Some(name) => format!("{head}class: {name}"),
                    None => line.to_owned(),
                }
            }
            None => line.to_owned(),
        })
        .collect::<Vec<_>>()
        .join("\n")
        + if text.ends_with('\n') { "\n" } else { "" }
}

impl TreeFixtureModel {
    fn expected(&self) -> &Value {
        self.base.expected()
    }

    fn classifier(&self) -> bool {
        self.classes.is_some()
    }

    fn name_of(&self, code: usize) -> String {
        self.base
            .class_names_ref()
            .and_then(|n| n.get(code).cloned())
            .unwrap_or_else(|| code.to_string())
    }

    fn criterion(&self) -> &'static str {
        match self.config {
            ModelConfig::Tree { criterion, .. } => criterion.sklearn_name(),
            _ => "",
        }
    }

    /// Class code of node `node` (`classes_[argmax]`).
    fn class_of(&self, node: usize) -> Option<usize> {
        let classes = self.classes.as_ref()?;
        classes.get(self.tree.argmax(node)).copied()
    }

    fn decision_path(&self, values: &[f64], form: Option<&PredictionForm>) -> DecisionPath {
        let nodes = self.tree.path(values);
        let names = &self.base.frame().feature_display;
        let leaf = nodes.last().copied().unwrap_or(0);
        let steps = nodes
            .windows(2)
            .map(|pair| {
                let (node, next) = (pair[0], pair[1]);
                let feature = self.tree.feature[node].max(0) as usize;
                let value = f64::from(values.get(feature).copied().unwrap_or(f64::NAN) as f32);
                PathStep {
                    node: to_u32(node),
                    feature: to_u32(feature),
                    name: names.get(feature).cloned().unwrap_or_default(),
                    value,
                    threshold: self.tree.threshold[node],
                    goes_left: self.tree.left[node] == next as i64,
                    display: form.and_then(|f| crate::models::path_display(f, feature, value)),
                }
            })
            .collect();
        DecisionPath {
            steps,
            leaf: to_u32(leaf),
            leaf_samples: self.tree.n_samples.get(leaf).copied().unwrap_or(0),
            leaf_value: self.tree.value.get(leaf).cloned().unwrap_or_default(),
            leaf_class: self.class_of(leaf).map(to_u32),
        }
    }
}

impl ModelView for TreeFixtureModel {
    fn summary(&self) -> ModelSummary {
        let mut summary = self.base.summary();
        summary.model_classes = self.classes.clone();
        summary.tree = Some(self.tree.summary(self.base.frame().feature_columns.len()));
        summary
    }

    fn evaluate(&self) -> Result<Evaluation, CommandError> {
        self.base.evaluate()
    }

    fn roc_curves(&self) -> Result<RocData, CommandError> {
        self.base.roc_curves()
    }

    fn probability_view(&self) -> Result<ProbabilityView, CommandError> {
        self.base.probability_view()
    }

    fn residuals_view(&self) -> Result<ResidualsView, CommandError> {
        self.base.residuals_view()
    }

    /// The 2D tree of `two_feature_models` evaluated on the grid of §C8.
    fn decision_grid(
        &self,
        x: usize,
        y: usize,
        _progress: &Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<DecisionGrid, CommandError> {
        let models = &self.expected()["two_feature_models"];
        let fitted = ["default", "pair"]
            .iter()
            .map(|key| &models[*key])
            .find(|m| indices(&m["features"]) == [x, y])
            .ok_or_else(|| {
                CommandError::invalid_selection(
                    "FixtureFacade: este par de características no está en el fixture.",
                )
            })?;
        let tree2 = FixtureTree::parse(&fitted["tree"]);
        let frame = self.base.frame();
        let train = self.base.train_rows();
        let xs: Vec<f64> = train.iter().map(|&i| frame.features[x][i]).collect();
        let ys: Vec<f64> = train.iter().map(|&i| frame.features[y][i]).collect();
        let (ax, ay, notices) = FixtureModel::axes(&xs, &ys)?;
        let mut classes = Vec::new();
        let mut values = Vec::new();
        for gy in &ay {
            check_cancelled(cancel)?;
            for gx in &ax {
                let leaf = tree2.leaf(&[*gx, *gy]);
                if let Some(model_classes) = &self.classes {
                    let code = model_classes.get(tree2.argmax(leaf)).copied().unwrap_or(0);
                    classes.push(u16::try_from(code).unwrap_or(u16::MAX));
                } else {
                    values.push(tree2.value[leaf].first().copied().unwrap_or(f64::NAN));
                }
            }
        }
        let grid_values = if self.classifier() {
            GridValues::Classes { z: classes }
        } else {
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            GridValues::Values {
                z: values,
                min,
                max,
            }
        };
        let target = frame.target.as_f64();
        Ok(DecisionGrid {
            x: ax,
            y: ay,
            values: grid_values,
            notices,
            points: Matrix::from_f64(
                xs.len(),
                2,
                xs.iter()
                    .zip(&ys)
                    .flat_map(|(a, b)| [*a, *b])
                    .collect::<Vec<_>>(),
            ),
            point_target: train.iter().map(|&i| target[i]).collect(),
            point_codes: match &frame.target {
                super::frame::Target::Classes { codes, .. } if self.classifier() => {
                    Some(train.iter().map(|&i| u32::from(codes[i])).collect())
                }
                _ => None,
            },
            feature_names: [
                frame.feature_display[x].clone(),
                frame.feature_display[y].clone(),
            ],
            class_names: self
                .base
                .class_names_ref()
                .map(<[String]>::to_vec)
                .filter(|_| self.classifier()),
            show_2d_warning: frame.feature_columns.len() > 2,
            fixed: None,
        })
    }

    fn coefficients(&self, _class_index: Option<usize>) -> Result<CoefficientsView, CommandError> {
        Err(CommandError::invalid_selection(
            "Los árboles de decisión no tienen coeficientes.",
        ))
    }

    fn prediction_form(&self) -> Result<PredictionForm, CommandError> {
        self.base.prediction_form()
    }

    fn predict_one(&self, values: &[f64]) -> Result<PredictionResult, CommandError> {
        let form = self.base.prediction_form().ok();
        let path = self.decision_path(values, form.as_ref());
        let leaf = path.leaf as usize;
        match &self.classes {
            Some(classes) => {
                let code = self.class_of(leaf).unwrap_or(0);
                let proba = classes
                    .iter()
                    .zip(&self.tree.value[leaf])
                    .map(|(&c, &p)| ClassProbability {
                        code: to_u32(c),
                        name: self.name_of(c),
                        probability: p,
                    })
                    .collect();
                Ok(PredictionResult::Class {
                    index: to_u32(code),
                    name: self.name_of(code),
                    proba: Some(proba),
                    path: Some(path),
                    neighbors: None,
                })
            }
            None => Ok(PredictionResult::Value {
                value: self.tree.value[leaf].first().copied().unwrap_or(f64::NAN),
                coefficients: None,
                path: Some(path),
                neighbors: None,
            }),
        }
    }

    fn export(&self, format: ExportFormat) -> Result<Vec<u8>, CommandError> {
        match format {
            ExportFormat::Onnx => Err(CommandError::export(
                "FixtureFacade no genera ONNX (usa mltutor-export)",
            )),
            ExportFormat::PythonPackage => {
                serde_json::to_vec_pretty(&self.expected()["export_package"]["model_json"])
                    .map_err(CommandError::export)
            }
            ExportFormat::TreeRules => Ok(self.tree_rules(RulesKind::Export, true)?.into_bytes()),
            ExportFormat::TreePython => Ok(self.code(CodeTemplate::TreePython, None)?.into_bytes()),
            ExportFormat::KerasPackage
            | ExportFormat::NnNumpyPredictor
            | ExportFormat::NnMetadata => Err(CommandError::not_nn()),
        }
    }

    fn code(
        &self,
        template: CodeTemplate,
        _pair: Option<(usize, usize)>,
    ) -> Result<String, CommandError> {
        let text = |value: &Value, what: &str| {
            value
                .as_str()
                .map(|t| t.trim_start_matches('\n').to_owned())
                .ok_or_else(|| CommandError::internal(format!("falta {what} en el fixture")))
        };
        let python = &self.expected()["python_code"];
        match template {
            CodeTemplate::TreePython => text(&python["corrected_text"], "corrected_text"),
            CodeTemplate::TreePythonLiteral => text(&python["app_text"], "app_text"),
            template if template.needs_tree() => {
                let frame = self.base.frame();
                crate::tree_code::render(
                    template,
                    &crate::tree_code::TreeCodeInput {
                        dataset: self.id.slug(),
                        dataset_label: &self.label,
                        csv: None,
                        config: self.config.clone(),
                        test_size: self.test_size,
                        feature_names: &frame.feature_display,
                        class_names: self.base.class_names_ref(),
                    },
                )
            }
            other => self.base.code(other, None),
        }
    }

    fn tree_diagram(&self) -> Result<TreeDiagram, CommandError> {
        let boxes = self.expected()["plot_tree"]["boxes_in_drawing_order"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let parents = self.tree.parents();
        let max_y = f64::from(self.tree.max_depth + 1);
        // Inverts `(x + 0.5) / max_x` with `max_x = max(x) + 1`.
        let max_xy = boxes
            .iter()
            .filter_map(|b| b["xy_axes"][0].as_f64())
            .fold(0.0_f64, f64::max);
        let (xs, max_x) = if boxes.len() == self.tree.len() && max_xy < 1.0 {
            let max_x = 0.5 / (1.0 - max_xy);
            let xs = boxes
                .iter()
                .map(|b| b["xy_axes"][0].as_f64().unwrap_or(0.0) * max_x - 0.5)
                .collect();
            (xs, max_x)
        } else {
            let xs = self.tree.fallback_layout();
            let max_x = xs.iter().copied().fold(0.0, f64::max) + 1.0;
            (xs, max_x)
        };
        let nodes = (0..self.tree.len())
            .map(|node| {
                let (depth, parent, is_left) = parents[node];
                let leaf = self.tree.is_leaf(node);
                let b = boxes.get(node);
                let lines = b
                    .and_then(|b| b["text"].as_str())
                    .map(|t| t.split('\n').map(str::to_owned).collect())
                    .unwrap_or_else(|| {
                        vec![
                            format!("{} = {:.3}", self.criterion(), self.tree.impurity[node]),
                            format!("samples = {}", self.tree.n_samples[node]),
                        ]
                    });
                TreeDiagramNode {
                    id: to_u32(node),
                    parent,
                    is_left,
                    depth,
                    x: xs[node],
                    y: f64::from(depth),
                    lines,
                    fill: b
                        .and_then(|b| b["facecolor"].as_str())
                        .unwrap_or("#ffffff")
                        .to_owned(),
                    leaf,
                    feature: (!leaf).then(|| to_u32(self.tree.feature[node].max(0) as usize)),
                    threshold: (!leaf).then(|| self.tree.threshold[node]),
                    impurity: self.tree.impurity[node],
                    n_samples: self.tree.n_samples[node],
                    value: self.tree.value[node].clone(),
                    value_counts: self
                        .tree
                        .value_counts
                        .as_ref()
                        .filter(|_| self.classifier())
                        .and_then(|c| c.get(node).cloned()),
                    class_index: self.class_of(node).map(to_u32),
                }
            })
            .collect();
        Ok(TreeDiagram {
            nodes,
            max_x,
            max_y,
            criterion: self.criterion().to_owned(),
            summary: self.tree.summary(self.base.frame().feature_columns.len()),
            class_names: self
                .base
                .class_names_ref()
                .map(<[String]>::to_vec)
                .filter(|_| self.classifier()),
        })
    }

    fn tree_rules(&self, kind: RulesKind, class_names: bool) -> Result<String, CommandError> {
        let export = &self.expected()["export_text"];
        let key = match kind {
            RulesKind::View => "view_text",
            RulesKind::Export => "rules_txt",
        };
        let text = export[key]
            .as_str()
            .ok_or_else(|| CommandError::internal(format!("falta export_text.{key}")))?;
        match (class_names, self.base.class_names_ref()) {
            (true, Some(names)) if self.classifier() => Ok(with_class_names(text, names)),
            _ => Ok(text.to_owned()),
        }
    }

    fn feature_importance(
        &self,
        method: ImportanceMethod,
        progress: &Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<FeatureImportanceView, CommandError> {
        let names = &self.base.frame().feature_display;
        let (importances, std, n_repeats, baseline, seed) = match method {
            ImportanceMethod::Native => (
                floats(&self.expected()["feature_importances"]),
                None,
                None,
                None,
                None,
            ),
            ImportanceMethod::Permutation => {
                progress("Calculando importancia por permutación...", Some(0.0));
                check_cancelled(cancel)?;
                let p = &self.expected()["permutation_importance"];
                if p.is_null() {
                    return Err(CommandError::internal("falta permutation_importance"));
                }
                (
                    floats(&p["importances_mean"]),
                    Some(floats(&p["importances_std"])),
                    p["n_repeats"].as_u64().map(|n| n as u32),
                    p["baseline_score"].as_f64(),
                    p["seed"].as_u64(),
                )
            }
        };
        let mut rows: Vec<ImportanceRow> = importances
            .iter()
            .enumerate()
            .map(|(i, &importance)| ImportanceRow {
                feature_index: to_u32(i),
                name: names.get(i).cloned().unwrap_or_default(),
                importance,
                std: std.as_ref().and_then(|s| s.get(i).copied()),
            })
            .collect();
        rows.sort_by(|a, b| {
            a.importance
                .total_cmp(&b.importance)
                .then(a.feature_index.cmp(&b.feature_index))
        });
        Ok(FeatureImportanceView {
            method,
            rows,
            n_repeats,
            baseline,
            seed,
            task: self.config.task_on(self.base.summary().dataset_task),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_facade::CoreFacade;
    use crate::fixture_facade::FixtureFacade;

    fn tree(task: TreeTask, criterion: TreeCriterion, depth: u8, mss: u8) -> ModelConfig {
        ModelConfig::Tree {
            task,
            criterion,
            max_depth: depth,
            min_samples_split: mss,
        }
    }

    const IRIS: ModelConfig = ModelConfig::Tree {
        task: TreeTask::Classification,
        criterion: TreeCriterion::Gini,
        max_depth: 3,
        min_samples_split: 2,
    };

    fn trained(id: DatasetId, config: ModelConfig) -> ModelHandle {
        let spec = TrainSpec {
            config,
            test_size: 0.3,
            dataset_label: String::new(),
        };
        FixtureFacade::from_checkout()
            .load_builtin(id)
            .unwrap()
            .train(&spec, &|_, _| {}, &AtomicBool::new(false))
            .unwrap()
    }

    fn fixture(name: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../tests/parity/trees/{name}.json"));
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn case_names_follow_the_fixtures() {
        assert_eq!(
            case_name(DatasetId::Iris, IRIS).as_deref(),
            Some("iris_gini_d3")
        );
        assert_eq!(
            case_name(
                DatasetId::Wine,
                tree(TreeTask::Classification, TreeCriterion::Entropy, 3, 10)
            )
            .as_deref(),
            Some("wine_entropy_d3_mss10")
        );
        assert_eq!(
            case_name(
                DatasetId::Iris,
                tree(TreeTask::Regression, TreeCriterion::SquaredError, 3, 2)
            )
            .as_deref(),
            Some("iris_regression_mismatch")
        );
        assert_eq!(case_name(DatasetId::Iris, ModelConfig::Linear), None);
    }

    /// The stored tree reproduces scikit-learn's decision paths (float32 rule).
    #[test]
    fn walking_the_tree_reproduces_sklearn() {
        for name in [
            "iris_gini_d3",
            "california_housing_squared_error_d8_mss10",
            "titanic_gini_d4",
            "diabetes_squared_error_d10",
        ] {
            let e = fixture(name)["expected"].clone();
            let tree = FixtureTree::parse(&e["tree"]);
            let n = e["feature_columns"].as_array().unwrap().len();
            for path in e["decision_paths_test_head"].as_array().unwrap() {
                let mut x = vec![f64::NAN; n];
                for step in path["steps"].as_array().unwrap() {
                    let f = step["feature"].as_u64().unwrap() as usize;
                    x[f] = step["value_float32"].as_f64().unwrap();
                }
                assert_eq!(tree.path(&x), indices(&path["nodes"]), "{name}");
            }
        }
    }

    #[test]
    fn iris_views_match_the_fixture() {
        let model = trained(DatasetId::Iris, IRIS);
        let e = fixture("iris_gini_d3")["expected"].clone();
        let summary = model.summary();
        assert_eq!(
            summary.tree,
            Some(TreeSummary {
                node_count: 9,
                n_leaves: 5,
                max_depth: 3,
                n_features: 4
            })
        );
        let d = model.tree_diagram().unwrap();
        assert_eq!(d.nodes.len(), 9);
        assert_eq!(d.nodes[0].fill, "#ffffff");
        assert_eq!(d.nodes[1].fill, "#e58139");
        assert_eq!(d.nodes[0].lines[0], "petal length (cm) <= 2.45");
        assert_eq!(d.nodes[1].parent, Some(0));
        assert!(d.nodes[1].is_left && !d.nodes[2].is_left);
        // Positions: (x + 0.5) / max_x of the fixture.
        for (node, b) in d
            .nodes
            .iter()
            .zip(e["plot_tree"]["boxes_in_drawing_order"].as_array().unwrap())
        {
            let xy = floats(&b["xy_axes"]);
            assert!(((node.x + 0.5) / d.max_x - xy[0]).abs() < 1e-12);
            assert!(((d.max_y - node.y - 0.5) / d.max_y - xy[1]).abs() < 1e-12);
        }
        assert_eq!(
            model.tree_rules(RulesKind::View, false).unwrap(),
            e["export_text"]["view_text"].as_str().unwrap()
        );
        assert_eq!(
            model.tree_rules(RulesKind::View, true).unwrap(),
            e["export_text"]["view_text_with_class_names_D40"]
                .as_str()
                .unwrap()
        );
        let native = model
            .feature_importance(
                ImportanceMethod::Native,
                &|_, _| {},
                &AtomicBool::new(false),
            )
            .unwrap();
        let order: Vec<u32> = native.rows.iter().map(|r| r.feature_index).collect();
        assert_eq!(order, vec![0, 1, 3, 2], "ascending, ties by index");
        let perm = model
            .feature_importance(
                ImportanceMethod::Permutation,
                &|_, _| {},
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(perm.n_repeats, Some(10));
        assert_eq!(perm.seed, Some(1_608_637_542));
        // Default form values → leaf 4 (fixture prediction_form.default).
        let x = floats(&e["prediction_form"]["default"]["input_float32"]);
        match model.predict_one(&x).unwrap() {
            PredictionResult::Class {
                name, proba, path, ..
            } => {
                assert_eq!(name, "versicolor");
                let path = path.unwrap();
                assert_eq!(path.leaf, 4);
                assert_eq!(path.steps.len(), 3);
                assert!(!path.steps[0].goes_left && path.steps[1].goes_left);
                assert_eq!(proba.unwrap()[1].probability, 1.0);
            }
            other => panic!("{other:?}"),
        }
        assert!(model
            .code(CodeTemplate::TreePython, None)
            .unwrap()
            .starts_with("def predecir_clase("));
        let grid = model
            .decision_grid(0, 1, &|_, _| {}, &AtomicBool::new(false))
            .unwrap();
        assert!(matches!(grid.values, GridValues::Classes { .. }));
    }

    #[test]
    fn categorical_steps_show_their_label() {
        let model = trained(
            DatasetId::Titanic,
            tree(TreeTask::Classification, TreeCriterion::Gini, 4, 2),
        );
        let e = fixture("titanic_gini_d4")["expected"].clone();
        let x = floats(&e["prediction_form"]["default"]["input_float32"]);
        let PredictionResult::Class { path, .. } = model.predict_one(&x).unwrap() else {
            panic!("clasificación");
        };
        let path = path.unwrap();
        let expected = indices(&e["prediction_form"]["default"]["path"]["nodes"]);
        let mut nodes: Vec<usize> = path.steps.iter().map(|s| s.node as usize).collect();
        nodes.push(path.leaf as usize);
        assert_eq!(nodes, expected);
        assert!(
            path.steps.iter().any(|s| s.display.is_some()),
            "{:?}",
            path.steps
        );
    }

    #[test]
    fn class_names_replace_the_codes() {
        let names = vec!["a".to_owned(), "b".to_owned()];
        assert_eq!(
            with_class_names("|--- x <= 1.00\n|   |--- class: 1\n", &names),
            "|--- x <= 1.00\n|   |--- class: b\n"
        );
    }

    #[test]
    fn regression_trees_serve_their_views() {
        let model = trained(
            DatasetId::Diabetes,
            tree(TreeTask::Regression, TreeCriterion::SquaredError, 3, 2),
        );
        assert!(matches!(
            model.evaluate().unwrap(),
            Evaluation::Regression(_)
        ));
        assert!(model.residuals_view().is_ok());
        let d = model.tree_diagram().unwrap();
        assert_eq!(d.nodes[0].fill, "#f8dcc8");
        assert!(d.class_names.is_none());
        let grid = model
            .decision_grid(2, 8, &|_, _| {}, &AtomicBool::new(false))
            .unwrap();
        assert!(matches!(grid.values, GridValues::Values { .. }));
        let rules = String::from_utf8(model.export(ExportFormat::TreeRules).unwrap()).unwrap();
        assert!(rules.contains("value: ["), "{rules}");
        let x = vec![0.0; 10];
        assert!(matches!(
            model.predict_one(&x).unwrap(),
            PredictionResult::Value { path: Some(_), .. }
        ));
    }
}
