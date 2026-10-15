//! 🔍 K-Nearest Neighbors of the [`super::FixtureFacade`] (fase4-knn.md §5–§6).
//!
//! Training looks up `knn/<case>.json` and replays it: the split, predictions,
//! probabilities, metrics, curves, the grids of the 2D models, the distance analysis and
//! the permutation importance are scikit-learn's exact values. New samples (form, canvas)
//! are answered with a plain brute-force search (ties: lowest index), which differs from
//! scikit-learn's kd-tree only on ties at the K-th distance: the parity tests of
//! `mltutor-core::knn` check the exact search. The views shared with F2 come from
//! [`FixtureModel`].

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde_json::Value;

use super::frame::{Frame, Target};
use super::model::{indices, FixtureModel};
use super::{floats, strings};
use crate::core_facade::{
    check_cancelled, ModelHandle, ModelSummary, ModelView, Progress, TrainSpec,
};
use crate::ipc::{
    to_u32, ClassDistancesView, ClassProbability, CodeTemplate, CoefficientsView, CommandError,
    DatasetId, DecisionGrid, Evaluation, ExportFormat, FeatureImportanceView, GridValues,
    ImportanceMethod, ImportanceRow, KnnMetric, KnnNeighbors, KnnSummary, KnnWeights, Matrix,
    ModelConfig, NeighborRow, PlaygroundPoint, PlaygroundRequest, PlaygroundView, PredictionForm,
    PredictionResult, ProbabilityView, ResidualsView, RocData, SearchEngine, TaskType, TieAtKth,
};
use crate::knn::{bounds, class_color, verdict, KnnTrainingContext};

/// `iris_k5_uniform_minkowski`, `…_test02` for `test_size = 0.2`.
pub fn case_name(id: DatasetId, config: ModelConfig, test_size: f64) -> Option<String> {
    let (k, weights, metric) = crate::knn::params(config)?;
    let suffix = if (test_size - 0.3).abs() < 1e-9 {
        String::new()
    } else {
        format!("_test{:02}", (test_size * 10.0).round() as u32)
    };
    Some(format!(
        "{}_k{k}_{}_{}{suffix}",
        id.slug(),
        weights.sklearn_name(),
        metric.sklearn_name()
    ))
}

fn unavailable(reason: impl std::fmt::Display) -> CommandError {
    CommandError::training(format!("FixtureFacade: {reason}"))
}

/// A KNN replayed from `knn/<case>.json`.
pub struct KnnFixtureModel {
    base: FixtureModel,
    id: DatasetId,
    label: String,
    test_size: f64,
    k: usize,
    weights: KnnWeights,
    metric: KnnMetric,
    engine: SearchEngine,
    /// `classes_` (codes) of a classifier.
    classes: Option<Vec<usize>>,
}

/// `FixtureDataset::train` for `ModelConfig::Knn`.
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
    let (k, weights, metric) =
        crate::knn::params(spec.config).ok_or_else(|| unavailable("no es un KNN"))?;
    let name =
        case_name(id, spec.config, spec.test_size).ok_or_else(|| unavailable("no es un KNN"))?;
    let read = |relative: &str| -> Result<Value, CommandError> {
        let path = root.join(relative);
        let text = std::fs::read_to_string(&path)
            .map_err(|_| unavailable(format!("no hay fixture {relative}")))?;
        serde_json::from_str(&text).map_err(|e| CommandError::internal(format!("{relative}: {e}")))
    };
    let fixture = read(&format!("knn/{name}.json"))?;
    check_cancelled(cancel)?;
    let mut expected = fixture["expected"].clone();
    let n_train = indices(&expected["train_indices"]).len();
    // D59 (Python trained and failed later): the fixture of the error has no model.
    crate::knn::check_k(spec.config, n_train)?;
    if expected["trained"] != Value::Bool(true) {
        return Err(unavailable(format!("{name}.json no entrena")));
    }
    let code = read("code/regression.json")?["expected"].take();
    let classifier = spec.config.task_on(dataset_task) == TaskType::Classification;
    let class_names = match &frame.target {
        Target::Classes { names, .. } => Some(names.clone()),
        Target::Numeric(_) => None,
    };
    let classes = classifier.then(|| indices(&expected["classes"]));
    if expected["evaluation"].is_null() {
        expected["evaluation"] = Value::Object(serde_json::Map::new());
    }
    let engine = match expected["neighbours_test"]["fit_method"].as_str() {
        Some("brute") => SearchEngine::Brute,
        _ => SearchEngine::KdTree,
    };
    let base = FixtureModel::for_tree(
        spec.config,
        dataset_task,
        frame,
        expected,
        code,
        class_names,
        Vec::new(),
    );
    Ok(Arc::new(KnnFixtureModel {
        base,
        id,
        label: spec.dataset_label.clone(),
        test_size: spec.test_size,
        k: usize::from(k),
        weights,
        metric,
        engine,
        classes,
    }))
}

/// Distance between two points (`minkowski` with p = 2 is the Euclidean one).
fn distance(metric: KnnMetric, a: &[f64], b: &[f64]) -> f64 {
    match metric {
        KnnMetric::Manhattan => a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum(),
        KnnMetric::Minkowski | KnnMetric::Euclidean => a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt(),
    }
}

/// The `k` nearest of `rows` to `query`: brute force, ties by the lowest index.
fn nearest(rows: &[Vec<f64>], query: &[f64], k: usize, metric: KnnMetric) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| (i, distance(metric, row, query)))
        .collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

/// sklearn's `_get_weights`: 1, `1/d`, or only the neighbours at distance 0.
fn weights_of(weights: KnnWeights, distances: &[f64]) -> Vec<f64> {
    match weights {
        KnnWeights::Uniform => vec![1.0; distances.len()],
        KnnWeights::Distance if distances.contains(&0.0) => distances
            .iter()
            .map(|d| if *d == 0.0 { 1.0 } else { 0.0 })
            .collect(),
        KnnWeights::Distance => distances.iter().map(|d| 1.0 / d).collect(),
    }
}

/// `predict_proba` over `classes` (votes in neighbour order, divided by their sum) and
/// the predicted code (`argmax`, first on ties).
fn vote(classes: &[usize], codes: &[usize], w: &[f64]) -> (usize, Vec<f64>) {
    let mut votes = vec![0.0; classes.len()];
    for (code, weight) in codes.iter().zip(w) {
        if let Some(slot) = classes.iter().position(|c| c == code) {
            votes[slot] += weight;
        }
    }
    let total: f64 = votes.iter().sum();
    let proba: Vec<f64> = votes
        .iter()
        .map(|v| if total > 0.0 { v / total } else { 0.0 })
        .collect();
    let mut best = 0;
    for (i, p) in proba.iter().enumerate() {
        if *p > proba[best] {
            best = i;
        }
    }
    (classes.get(best).copied().unwrap_or(0), proba)
}

impl KnnFixtureModel {
    fn expected(&self) -> &Value {
        self.base.expected()
    }

    fn frame(&self) -> &Frame {
        self.base.frame()
    }

    fn name_of(&self, code: usize) -> String {
        self.base
            .class_names_ref()
            .and_then(|n| n.get(code).cloned())
            .unwrap_or_else(|| format!("Clase {code}"))
    }

    fn codes(&self) -> Option<&[u16]> {
        match &self.frame().target {
            Target::Classes { codes, .. } => Some(codes),
            Target::Numeric(_) => None,
        }
    }

    fn row(&self, i: usize) -> Vec<f64> {
        self.frame().features.iter().map(|col| col[i]).collect()
    }

    fn neighbor_rows(
        &self,
        found: &[(usize, f64)],
        rows: &[usize],
        features: bool,
    ) -> Vec<NeighborRow> {
        let distances: Vec<f64> = found.iter().map(|(_, d)| *d).collect();
        let w = weights_of(self.weights, &distances);
        let target = self.frame().target.as_f64();
        found
            .iter()
            .zip(w)
            .map(|(&(pos, d), weight)| {
                let dataset_row = rows[pos];
                NeighborRow {
                    train_row: to_u32(pos),
                    dataset_row: to_u32(dataset_row),
                    distance: d,
                    weight,
                    class_index: self
                        .codes()
                        .filter(|_| self.classes.is_some())
                        .map(|c| u32::from(c[dataset_row])),
                    value: self.classes.is_none().then(|| target[dataset_row]),
                    features: if features {
                        self.row(dataset_row)
                    } else {
                        Vec::new()
                    },
                }
            })
            .collect()
    }
}

impl ModelView for KnnFixtureModel {
    fn summary(&self) -> ModelSummary {
        let mut summary = self.base.summary();
        summary.model_classes = self.classes.clone();
        summary.knn = Some(KnnSummary {
            n_neighbors: to_u32(self.k),
            weights: self.weights,
            metric: self.metric,
            engine: self.engine,
            n_train: to_u32(summary.n_train),
        });
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

    /// The 2D model of `two_feature_models` (all the rows of the dataset, D63).
    fn decision_grid(
        &self,
        x: usize,
        y: usize,
        _progress: &Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<DecisionGrid, CommandError> {
        check_cancelled(cancel)?;
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
        let ax = floats(&fitted["x"]);
        let ay = floats(&fitted["y"]);
        let z: Vec<f64> = fitted["z"]
            .as_array()
            .map(|rows| rows.iter().flat_map(floats).collect())
            .unwrap_or_default();
        let frame = self.frame();
        let n = frame.n_rows();
        let (xs, ys) = (&frame.features[x], &frame.features[y]);
        let notices = FixtureModel::axes(xs, ys)?.2;
        let values = if self.classes.is_some() {
            GridValues::Classes {
                z: z.iter().map(|v| v.round().max(0.0) as u16).collect(),
            }
        } else {
            let min = z.iter().copied().fold(f64::INFINITY, f64::min);
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            GridValues::Values { z, min, max }
        };
        Ok(DecisionGrid {
            x: ax,
            y: ay,
            values,
            notices,
            points: Matrix::from_f64(
                n,
                2,
                xs.iter()
                    .zip(ys)
                    .flat_map(|(a, b)| [*a, *b])
                    .collect::<Vec<_>>(),
            ),
            point_target: frame.target.as_f64(),
            point_codes: self
                .codes()
                .filter(|_| self.classes.is_some())
                .map(|c| c.iter().map(|&v| u32::from(v)).collect()),
            feature_names: [
                frame.feature_display[x].clone(),
                frame.feature_display[y].clone(),
            ],
            class_names: self
                .base
                .class_names_ref()
                .map(<[String]>::to_vec)
                .filter(|_| self.classes.is_some()),
            show_2d_warning: frame.feature_columns.len() > 2,
        })
    }

    fn coefficients(&self, _class_index: Option<usize>) -> Result<CoefficientsView, CommandError> {
        Err(CommandError::invalid_selection(
            "Los modelos KNN no tienen coeficientes.",
        ))
    }

    fn prediction_form(&self) -> Result<PredictionForm, CommandError> {
        self.base.prediction_form()
    }

    fn predict_one(&self, values: &[f64]) -> Result<PredictionResult, CommandError> {
        let train = self.base.train_rows();
        let rows: Vec<Vec<f64>> = train.iter().map(|&i| self.row(i)).collect();
        let found = nearest(&rows, values, self.k, self.metric);
        let tie_at_kth = found.last().and_then(|&(_, kth)| {
            let n_tied = rows
                .iter()
                .filter(|r| distance(self.metric, r, values) == kth)
                .count();
            let slots = found.iter().filter(|(_, d)| *d == kth).count();
            (n_tied > slots).then(|| TieAtKth {
                n_tied: to_u32(n_tied),
                slots: to_u32(slots),
                distance: kth,
            })
        });
        let neighbors = self.neighbor_rows(&found, train, true);
        let w: Vec<f64> = neighbors.iter().map(|n| n.weight).collect();
        match &self.classes {
            Some(classes) => {
                let codes: Vec<usize> = neighbors
                    .iter()
                    .map(|n| n.class_index.unwrap_or(0) as usize)
                    .collect();
                let (code, proba) = vote(classes, &codes, &w);
                Ok(PredictionResult::Class {
                    index: to_u32(code),
                    name: self.name_of(code),
                    proba: Some(
                        classes
                            .iter()
                            .zip(proba)
                            .map(|(&c, probability)| ClassProbability {
                                name: self.name_of(c),
                                probability,
                            })
                            .collect(),
                    ),
                    path: None,
                    neighbors: Some(KnnNeighbors {
                        rows: neighbors,
                        tie_at_kth,
                    }),
                })
            }
            None => {
                let total: f64 = w.iter().sum();
                let value = neighbors
                    .iter()
                    .zip(&w)
                    .map(|(n, w)| n.value.unwrap_or(0.0) * w)
                    .sum::<f64>()
                    / total;
                Ok(PredictionResult::Value {
                    value,
                    coefficients: None,
                    path: None,
                    neighbors: Some(KnnNeighbors {
                        rows: neighbors,
                        tie_at_kth,
                    }),
                })
            }
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
            ExportFormat::TreeRules | ExportFormat::TreePython => Err(CommandError::not_tree()),
        }
    }

    fn code(
        &self,
        template: CodeTemplate,
        pair: Option<(usize, usize)>,
    ) -> Result<String, CommandError> {
        let frame = self.frame();
        if template == CodeTemplate::DecisionBoundary {
            let (x, y) = pair.unwrap_or((0, 1.min(frame.feature_columns.len().saturating_sub(1))));
            let context = mltutor_core::codegen::DecisionBoundaryContext::new(
                [x, y],
                vec![
                    frame.feature_display[x].clone(),
                    frame.feature_display[y].clone(),
                ],
                self.base.class_names_ref().map(<[String]>::to_vec),
            )
            .all_rows();
            return mltutor_core::codegen::render_named("frontera_decision.py.j2", &context)
                .map_err(|error| CommandError::internal(format!("plantilla: {error}")));
        }
        if template != CodeTemplate::KnnTraining {
            return self.base.code(template, pair);
        }
        KnnTrainingContext {
            dataset: self.id.slug().to_owned(),
            dataset_label: self.label.clone(),
            csv_file: None,
            csv_target: None,
            csv_sep: None,
            csv_decimal: None,
            csv_encoding: None,
            task: if self.classes.is_some() {
                "classification"
            } else {
                "regression"
            }
            .to_owned(),
            n_neighbors: to_u32(self.k),
            weights: self.weights.sklearn_name().to_owned(),
            metric: self.metric.sklearn_name().to_owned(),
            test_size: self.test_size,
            feature_names: frame.feature_display.clone(),
            class_names: self.base.class_names_ref().map(<[String]>::to_vec),
            sample: None,
        }
        .render()
    }

    fn feature_importance(
        &self,
        method: ImportanceMethod,
        progress: &Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<FeatureImportanceView, CommandError> {
        if method != ImportanceMethod::Permutation {
            return Err(CommandError::not_tree());
        }
        progress("Calculando importancia por permutación...", Some(0.0));
        check_cancelled(cancel)?;
        let p = &self.expected()["permutation_importance"];
        if p.is_null() {
            return Err(CommandError::internal("falta permutation_importance"));
        }
        let names = &self.frame().feature_display;
        let std = floats(&p["importances_std"]);
        let mut rows: Vec<ImportanceRow> = floats(&p["importances_mean"])
            .into_iter()
            .enumerate()
            .map(|(i, importance)| ImportanceRow {
                feature_index: to_u32(i),
                name: names.get(i).cloned().unwrap_or_default(),
                importance,
                std: std.get(i).copied(),
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
            n_repeats: p["n_repeats"].as_u64().map(|n| n as u32),
            baseline: p["baseline_score"].as_f64(),
            seed: p["seed"].as_u64(),
            task: self.base.summary().model_task,
        })
    }

    /// `distance_analysis.app` (test 0.3) or `with_model_split_D56`; J2 separation
    /// (`separation_J2`, or computed from the matrix and the diagonal of the fixture).
    fn class_distances(&self, cancel: &AtomicBool) -> Result<ClassDistancesView, CommandError> {
        check_cancelled(cancel)?;
        if self.classes.is_none() {
            return Err(CommandError::not_classifier());
        }
        let analysis = &self.expected()["distance_analysis"];
        let app = if analysis["with_model_split_D56"].is_object() {
            &analysis["with_model_split_D56"]
        } else {
            &analysis["app"]
        };
        let classes = indices(&app["classes"]);
        let matrix: Vec<Vec<f64>> = app["matrix"]
            .as_array()
            .map(|rows| rows.iter().map(floats).collect())
            .unwrap_or_default();
        let ratio = app["ratio"].as_f64().unwrap_or(0.0);
        let within = app["within_class_mean_D67"]
            .as_array()
            .or_else(|| analysis["app"]["within_class_mean_D67"].as_array())
            .map(|v| v.iter().map(Value::as_f64).collect())
            .unwrap_or_else(|| vec![None; classes.len()]);
        let labels = strings(&app["class_labels"]);
        let separation = app["separation_J2"]
            .as_f64()
            .or_else(|| separation(&matrix, &within));
        let n_samples = self.base.train_rows().len();
        let sample_limit = mltutor_core::knn::class_distances::MAX_ROWS_PER_CLASS;
        Ok(ClassDistancesView {
            class_names: if labels.len() == classes.len() {
                labels
            } else {
                classes.iter().map(|&c| self.name_of(c)).collect()
            },
            verdict: verdict(classes.len(), separation, ratio),
            separation: separation.filter(|s| s.is_finite()),
            separation_unbounded: separation.is_some_and(f64::is_infinite),
            sampled: n_samples > sample_limit
                && self.codes().is_some_and(|codes| {
                    let rows = self.base.train_rows();
                    classes.iter().any(|&c| {
                        rows.iter()
                            .filter(|&&r| codes.get(r).map(|&v| usize::from(v)) == Some(c))
                            .count()
                            > sample_limit
                    })
                }),
            sample_limit: to_u32(sample_limit),
            classes: classes.iter().map(|&c| to_u32(c)).collect(),
            matrix,
            within,
            min: app["min_distance"].as_f64(),
            max: app["max_distance"].as_f64(),
            ratio,
            metric: self.metric,
            n_samples: to_u32(n_samples),
        })
    }

    fn knn_playground(
        &self,
        req: &PlaygroundRequest,
        cancel: &AtomicBool,
    ) -> Result<PlaygroundView, CommandError> {
        let Some(codes) = self.codes().filter(|_| self.classes.is_some()) else {
            return Err(CommandError::not_classifier());
        };
        let frame = self.frame();
        let (fx, fy) = (req.feature_x as usize, req.feature_y as usize);
        let (xs, ys) = (&frame.features[fx], &frame.features[fy]);
        let n = frame.n_rows();
        let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![xs[i], ys[i]]).collect();
        let all: Vec<usize> = (0..n).collect();
        let mut classes: Vec<usize> = codes.iter().map(|&c| usize::from(c)).collect();
        classes.sort_unstable();
        classes.dedup();
        let k = usize::from(req.n_neighbors).min(n);
        let canvas = bounds(xs, ys);
        let requested = crate::knn::validate_points(&req.points, canvas)?;
        let mut points = Vec::with_capacity(requested.len());
        for point in &requested {
            check_cancelled(cancel)?;
            let found = nearest(&rows, point, k, req.metric);
            let distances: Vec<f64> = found.iter().map(|(_, d)| *d).collect();
            let w = weights_of(req.weights, &distances);
            let neighbor_codes: Vec<usize> =
                found.iter().map(|(i, _)| usize::from(codes[*i])).collect();
            let (code, proba) = vote(&classes, &neighbor_codes, &w);
            let mut neighbors = self.neighbor_rows(&found, &all, false);
            for (row, weight) in neighbors.iter_mut().zip(&w) {
                row.weight = *weight;
                row.train_row = row.dataset_row;
            }
            points.push(PlaygroundPoint {
                point: *point,
                class_index: to_u32(code),
                confidence: proba.iter().copied().fold(0.0, f64::max),
                proba,
                mean_distance: distances.iter().sum::<f64>() / distances.len().max(1) as f64,
                neighbors,
            });
        }
        let mean_confidence = (!points.is_empty())
            .then(|| points.iter().map(|p| p.confidence).sum::<f64>() / points.len() as f64);
        let n_names = self.base.class_names_ref().map_or(0, <[String]>::len);
        let n_codes = classes.last().map_or(0, |c| c + 1).max(n_names);
        Ok(PlaygroundView {
            data: req.include_data.then(|| {
                Matrix::from_f64(
                    n,
                    2,
                    xs.iter()
                        .zip(ys)
                        .flat_map(|(a, b)| [*a, *b])
                        .collect::<Vec<_>>(),
                )
            }),
            codes: req
                .include_data
                .then(|| codes.iter().map(|&c| u32::from(c)).collect()),
            classes: classes.iter().map(|&c| to_u32(c)).collect(),
            class_names: (0..n_codes).map(|c| self.name_of(c)).collect(),
            colors: (0..n_codes).map(class_color).collect(),
            bounds: canvas,
            feature_names: [
                frame.feature_display[fx].clone(),
                frame.feature_display[fy].clone(),
            ],
            points,
            mean_confidence,
            engine: if frame.n_rows() / 2 <= k {
                SearchEngine::Brute
            } else {
                SearchEngine::KdTree
            },
            n_neighbors: to_u32(k),
        })
    }
}

/// J2: `min_{i≠j} M[i, j] / max(W_i, W_j)` (`None` with < 2 classes or some `W_i`
/// missing or 0), the rule of `mltutor_core::knn::class_distances`.
fn separation(matrix: &[Vec<f64>], within: &[Option<f64>]) -> Option<f64> {
    let w: Vec<f64> = within
        .iter()
        .map(|w| w.filter(|&v| v > 0.0))
        .collect::<Option<_>>()?;
    let mut best: Option<f64> = None;
    for (i, row) in matrix.iter().enumerate() {
        for (j, &m) in row.iter().enumerate() {
            if i != j {
                let s = m / w.get(i)?.max(*w.get(j)?);
                best = Some(best.map_or(s, |b| b.min(s)));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_facade::CoreFacade;
    use crate::fixture_facade::FixtureFacade;

    fn knn(k: u8, weights: KnnWeights, metric: KnnMetric) -> ModelConfig {
        ModelConfig::Knn {
            n_neighbors: k,
            weights,
            metric,
        }
    }

    fn trained(id: DatasetId, config: ModelConfig, test_size: f64) -> ModelHandle {
        let spec = TrainSpec {
            config,
            test_size,
            dataset_label: "🌸 Iris - Clasificación de flores".into(),
        };
        FixtureFacade::from_checkout()
            .load_builtin(id)
            .unwrap()
            .train(&spec, &|_, _| {}, &AtomicBool::new(false))
            .unwrap()
    }

    fn fixture(name: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/parity/knn")
            .join(format!("{name}.json"));
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    const DEFAULT: (KnnWeights, KnnMetric) = (KnnWeights::Uniform, KnnMetric::Minkowski);

    #[test]
    fn case_names_follow_the_fixtures() {
        let (w, m) = DEFAULT;
        assert_eq!(
            case_name(DatasetId::Iris, knn(5, w, m), 0.3).as_deref(),
            Some("iris_k5_uniform_minkowski")
        );
        assert_eq!(
            case_name(DatasetId::Iris, knn(5, w, m), 0.2).as_deref(),
            Some("iris_k5_uniform_minkowski_test02")
        );
        assert_eq!(
            case_name(
                DatasetId::BreastCancer,
                knn(10, KnnWeights::Distance, KnnMetric::Euclidean),
                0.3
            )
            .as_deref(),
            Some("breast_cancer_k10_distance_euclidean")
        );
        assert_eq!(case_name(DatasetId::Iris, ModelConfig::Linear, 0.3), None);
    }

    #[test]
    fn iris_views_match_the_fixture() {
        let (w, m) = DEFAULT;
        let model = trained(DatasetId::Iris, knn(5, w, m), 0.3);
        let e = &fixture("iris_k5_uniform_minkowski")["expected"];
        let summary = model.summary();
        assert_eq!(summary.knn.map(|k| k.engine), Some(SearchEngine::KdTree));
        assert_eq!(summary.model_classes, Some(vec![0, 1, 2]));
        let Evaluation::Classification(eval) = model.evaluate().unwrap() else {
            panic!("Iris is classification");
        };
        assert!((eval.accuracy - e["evaluation"]["accuracy"].as_f64().unwrap()).abs() < 1e-12);
        let distances = model.class_distances(&AtomicBool::new(false)).unwrap();
        assert_eq!(distances.classes, vec![0, 1, 2]);
        assert_eq!(distances.within.len(), 3);
        assert_eq!(distances.verdict, crate::ipc::SeparationVerdict::Moderate);
        // The default form sample: the 5 neighbours of the fixture (no ties there).
        let x = floats(&e["prediction_form"]["default"]["input_float32"]);
        let PredictionResult::Class {
            index, neighbors, ..
        } = model.predict_one(&x).unwrap()
        else {
            panic!("Iris predicts a class");
        };
        assert_eq!(index, 1);
        let rows: Vec<usize> = neighbors
            .unwrap()
            .rows
            .iter()
            .map(|n| n.train_row as usize)
            .collect();
        assert_eq!(
            rows,
            indices(&e["prediction_form"]["default"]["neighbours"])
        );
        let grid = model
            .decision_grid(0, 1, &|_, _| {}, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(grid.points.rows, 150, "D63: every row of the dataset");
    }

    /// C18 against `interactive.default.samples[*].sklearn_2d_all_data` (the predicted
    /// class and the probabilities; the neighbours only differ on ties).
    #[test]
    fn playground_follows_sklearn_on_the_samples() {
        let (w, m) = DEFAULT;
        let model = trained(DatasetId::Iris, knn(5, w, m), 0.3);
        let e = &fixture("iris_k5_uniform_minkowski")["expected"]["interactive"]["default"];
        let samples = e["samples"].as_array().unwrap();
        let points: Vec<[f64; 2]> = samples
            .iter()
            .map(|s| {
                let p = floats(&s["point"]);
                [p[0], p[1]]
            })
            .collect();
        let req = PlaygroundRequest {
            model_id: 1,
            feature_x: 0,
            feature_y: 1,
            n_neighbors: 5,
            weights: w,
            metric: m,
            points,
            include_data: true,
            task_id: 1,
        };
        let view = model.knn_playground(&req, &AtomicBool::new(false)).unwrap();
        assert_eq!(view.data.as_ref().map(|d| d.rows), Some(150));
        assert_eq!(view.colors[..3], ["#FF6B6B", "#4ECDC4", "#45B7D1"]);
        let mut compared = 0;
        for (point, sample) in view.points.iter().zip(samples) {
            let sk = &sample["sklearn_2d_all_data"];
            let mut ours: Vec<usize> = point
                .neighbors
                .iter()
                .map(|n| n.dataset_row as usize)
                .collect();
            let mut theirs = indices(&sk["neighbours"]);
            ours.sort_unstable();
            theirs.sort_unstable();
            if ours != theirs {
                // A tie at the K-th distance: the brute force of the fixture keeps the
                // lowest index, sklearn's kd-tree the first point it visits.
                continue;
            }
            compared += 1;
            assert_eq!(
                u64::from(point.class_index),
                sk["predict"].as_u64().unwrap()
            );
            for (a, b) in point.proba.iter().zip(floats(&sk["proba"])) {
                assert!((a - b).abs() < 1e-12, "{:?}", point.point);
            }
        }
        assert!(compared >= 3, "{compared}");
        // Distance 0 with `distance` weights: only those neighbours count (sklearn).
        let zero = PlaygroundRequest {
            weights: KnnWeights::Distance,
            points: vec![[5.1, 3.5]],
            include_data: false,
            ..req
        };
        let view = model
            .knn_playground(&zero, &AtomicBool::new(false))
            .unwrap();
        let weights: Vec<f64> = view.points[0].neighbors.iter().map(|n| n.weight).collect();
        assert_eq!(weights, vec![1.0, 1.0, 0.0, 0.0, 0.0]);
        assert!(view.data.is_none());
    }

    #[test]
    fn regression_and_d56_and_d59() {
        let (w, m) = DEFAULT;
        let model = trained(DatasetId::Diabetes, knn(5, w, m), 0.3);
        assert!(model.residuals_view().is_ok());
        assert!(matches!(
            model.class_distances(&AtomicBool::new(false)),
            Err(CommandError::NotClassifier { .. })
        ));
        let grid = model
            .decision_grid(0, 1, &|_, _| {}, &AtomicBool::new(false))
            .unwrap();
        assert!(matches!(grid.values, GridValues::Values { .. }));
        let x = vec![0.0; 10];
        assert!(matches!(
            model.predict_one(&x).unwrap(),
            PredictionResult::Value {
                neighbors: Some(_),
                ..
            }
        ));
        assert!(model
            .code(CodeTemplate::KnnTraining, None)
            .unwrap()
            .contains("KNeighborsRegressor"));
        // D56: the distances of the 0.2 split of the model.
        let iris = trained(DatasetId::Iris, knn(5, w, m), 0.2);
        assert_eq!(
            iris.class_distances(&AtomicBool::new(false))
                .unwrap()
                .n_samples,
            120
        );
    }

    /// The training script is the one of `code/knn_training.json` (same context).
    #[test]
    fn training_script_matches_the_fixture() {
        let (w, m) = DEFAULT;
        let model = trained(DatasetId::Iris, knn(5, w, m), 0.3);
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/parity/code/knn_training.json");
        let code: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let expected = code["expected"]["iris_k5_uniform_minkowski"]["text"]
            .as_str()
            .unwrap();
        assert_eq!(
            model.code(CodeTemplate::KnnTraining, None).unwrap(),
            expected
        );
    }
}
