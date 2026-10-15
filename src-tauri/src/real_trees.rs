//! 🌲 Decision trees of [`crate::real_facade::RealFacade`] (fase3-arboles.md §5 and §8):
//! conversions of `mltutor_core::tree`, `importance` and `codegen` to the IPC DTOs. Every
//! computation is a core function; here only the orchestration lives.

use std::sync::atomic::AtomicBool;

use mltutor_core as core;
use mltutor_core::model::{ModelError, Predictor, TrainControl, TrainProgress};
use mltutor_core::tree::DecisionTree;
use ndarray::ArrayView2;

use crate::core_facade::{ModelView, Progress};
use crate::ipc::{self, to_u32, CodeTemplate, CommandError, ModelConfig};
use crate::real_models::{model_error, CoreModel};

fn tree_task(task: ipc::TreeTask) -> core::tree::TreeTask {
    match task {
        ipc::TreeTask::Classification => core::tree::TreeTask::Classification,
        ipc::TreeTask::Regression => core::tree::TreeTask::Regression,
    }
}

fn tree_criterion(criterion: ipc::TreeCriterion) -> core::tree::TreeCriterion {
    match criterion {
        ipc::TreeCriterion::Gini => core::tree::TreeCriterion::Gini,
        ipc::TreeCriterion::Entropy => core::tree::TreeCriterion::Entropy,
        ipc::TreeCriterion::SquaredError => core::tree::TreeCriterion::SquaredError,
        ipc::TreeCriterion::FriedmanMse => core::tree::TreeCriterion::FriedmanMse,
        ipc::TreeCriterion::AbsoluteError => core::tree::TreeCriterion::AbsoluteError,
    }
}

/// `train_model` with `ModelConfig::Tree` (`random_state = 42`).
pub fn fit(
    config: ModelConfig,
    x_train: ArrayView2<'_, f64>,
    y_train: &core::Target,
    control: &TrainControl<'_>,
) -> Result<Box<dyn Predictor>, ModelError> {
    let ModelConfig::Tree {
        task,
        criterion,
        max_depth,
        min_samples_split,
    } = config
    else {
        return Err(ModelError::InvalidConfig {
            reason: "no es un árbol de decisión".to_owned(),
        });
    };
    let config = core::tree::TreeConfig::new(
        tree_task(task),
        tree_criterion(criterion),
        u32::from(max_depth),
        u32::from(min_samples_split),
    );
    core::tree::fit_boxed(config, x_train, y_train, control)
}

pub fn summary(tree: &DecisionTree) -> ipc::TreeSummary {
    let s = tree.summary();
    ipc::TreeSummary {
        node_count: to_u32(s.node_count),
        n_leaves: to_u32(s.n_leaves),
        max_depth: to_u32(s.max_depth),
        n_features: to_u32(s.n_features),
    }
}

fn require_tree(model: &CoreModel) -> Result<&DecisionTree, CommandError> {
    model.tree().ok_or_else(CommandError::not_tree)
}

/// Class names for the boxes and rules (classifiers only).
fn class_names(model: &CoreModel, tree: &DecisionTree) -> Option<Vec<String>> {
    model.class_names().filter(|_| tree.is_classifier())
}

/// C13: `plot_tree` boxes and Buchheim positions.
pub fn diagram(model: &CoreModel) -> Result<ipc::TreeDiagram, CommandError> {
    let tree = require_tree(model)?;
    let names = class_names(model, tree);
    let d = core::tree::plot::diagram(tree, &model.dataset.feature_display, names.as_deref());
    Ok(ipc::TreeDiagram {
        nodes: d
            .nodes
            .into_iter()
            .map(|n| ipc::TreeDiagramNode {
                id: to_u32(n.id),
                parent: n.parent.map(to_u32),
                is_left: n.is_left,
                depth: to_u32(n.depth),
                x: n.x,
                y: n.y,
                lines: n.lines,
                fill: n.fill,
                leaf: n.leaf,
                feature: n.feature.map(to_u32),
                threshold: n.threshold,
                impurity: n.impurity,
                n_samples: to_u32(n.n_samples),
                value: n.value,
                value_counts: n.value_counts,
                class_index: n.class_code.map(to_u32),
            })
            .collect(),
        max_x: d.max_x,
        max_y: d.max_y,
        criterion: d.criterion,
        summary: summary(tree),
        class_names: names,
    })
}

/// C14: `export_text` with weights (📝 Texto) or without (`decision_tree_rules.txt`).
pub fn rules(
    model: &CoreModel,
    kind: ipc::RulesKind,
    with_class_names: bool,
) -> Result<String, CommandError> {
    let tree = require_tree(model)?;
    let names = class_names(model, tree).filter(|_| with_class_names);
    let options = core::tree::ExportTextOptions {
        class_names: names.as_deref(),
        show_weights: kind == ipc::RulesKind::View,
        ..core::tree::ExportTextOptions::default()
    };
    Ok(core::tree::export_text(
        tree,
        &model.dataset.feature_display,
        &options,
    ))
}

/// C10: native importance or permutation importance on the test set.
pub fn importance(
    model: &CoreModel,
    method: ipc::ImportanceMethod,
    progress: &Progress<'_>,
    cancel: &AtomicBool,
) -> Result<ipc::FeatureImportanceView, CommandError> {
    let names = &model.dataset.feature_display;
    let (rows, n_repeats, baseline, seed) = match method {
        ipc::ImportanceMethod::Native => {
            let tree = require_tree(model)?;
            let native = core::importance::native(tree);
            (
                core::importance::rows(names, &native, None),
                None,
                None,
                None,
            )
        }
        ipc::ImportanceMethod::Permutation => {
            let report = |p: TrainProgress| {
                let percent = (p.max_iter > 0)
                    .then(|| (p.iteration as f32 / p.max_iter as f32 * 100.0).min(99.0));
                progress("Calculando importancia por permutación...", percent);
            };
            let control = TrainControl::none()
                .with_cancel(cancel)
                .with_progress(&report);
            let options = core::importance::PermutationOptions::default();
            let x_test = model.x_test();
            let p = core::importance::permutation(
                model.model.as_ref(),
                x_test.view(),
                &model.y_test,
                options,
                &control,
            )
            .map_err(model_error)?;
            (
                core::importance::rows(names, &p.mean, Some(&p.std)),
                Some(to_u32(options.n_repeats)),
                Some(p.baseline),
                Some(u64::from(p.seed)),
            )
        }
    };
    Ok(ipc::FeatureImportanceView {
        method,
        rows: rows
            .into_iter()
            .map(|r| ipc::ImportanceRow {
                feature_index: to_u32(r.feature_index),
                name: r.name,
                importance: r.importance,
                std: r.std,
            })
            .collect(),
        n_repeats,
        baseline,
        seed,
        task: crate::real_facade::task(model.model.task()),
    })
}

/// C16 (+ D46): the path of the sample of the form and the leaf it reaches.
pub fn decision_path(
    model: &CoreModel,
    tree: &DecisionTree,
    values: &[f64],
) -> Result<ipc::DecisionPath, CommandError> {
    let path = tree.decision_path(values).map_err(model_error)?;
    let form = model.prediction_form().ok();
    let names = &model.dataset.feature_display;
    Ok(ipc::DecisionPath {
        steps: path
            .steps
            .into_iter()
            .map(|s| ipc::PathStep {
                node: to_u32(s.node),
                feature: to_u32(s.feature),
                name: names.get(s.feature).cloned().unwrap_or_default(),
                value: s.value,
                threshold: s.threshold,
                goes_left: s.goes_left,
                display: form
                    .as_ref()
                    .and_then(|f| crate::models::path_display(f, s.feature, s.value)),
            })
            .collect(),
        leaf: to_u32(path.leaf),
        leaf_samples: to_u32(path.leaf_samples),
        leaf_value: path.leaf_value,
        leaf_class: path.leaf_class.map(to_u32),
    })
}

/// The Python code of the tree lesson (§7).
pub fn code(model: &CoreModel, template: CodeTemplate) -> Result<String, CommandError> {
    use core::codegen::{render, CodeTemplate as T};
    let tree = require_tree(model)?;
    let features = &model.dataset.feature_display;
    let names = class_names(model, tree);
    let template_error =
        |e: core::codegen::CodegenError| CommandError::internal(format!("plantilla: {e}"));
    match template {
        CodeTemplate::TreePython => render(
            T::TreePython,
            &core::codegen::tree_python_corrected_context(tree, features, names.as_deref()),
        )
        .map_err(template_error),
        CodeTemplate::TreePythonLiteral => render(
            T::TreePythonLiteral,
            &core::codegen::tree_python_literal_context(tree, features, None),
        )
        .map_err(template_error),
        _ => {
            // The CSV is read in the script as the app read it (as in F2, I1).
            let context = model.training_context();
            let csv = context
                .csv_file
                .clone()
                .map(|file| crate::tree_code::CsvSource {
                    file,
                    target: context.csv_target.clone().unwrap_or_default(),
                    sep: context.csv_sep.clone().unwrap_or_else(|| ",".to_owned()),
                    decimal: context
                        .csv_decimal
                        .clone()
                        .unwrap_or_else(|| ".".to_owned()),
                    encoding: context
                        .csv_encoding
                        .clone()
                        .unwrap_or_else(|| "utf-8".to_owned()),
                });
            crate::tree_code::render(
                template,
                &crate::tree_code::TreeCodeInput {
                    dataset: &context.dataset,
                    dataset_label: &context.dataset_label,
                    csv,
                    config: model.spec.config,
                    test_size: model.spec.test_size,
                    feature_names: features,
                    class_names: names.as_deref(),
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use crate::core_facade::{CoreFacade, ModelHandle, TrainSpec};
    use crate::fixture_facade::FixtureFacade;
    use crate::ipc::{
        CodeTemplate, DatasetId, Evaluation, ExportFormat, GridValues, ImportanceMethod,
        ModelConfig, PredictionResult, RulesKind, TreeCriterion, TreeTask,
    };
    use crate::real_facade::RealFacade;

    fn tree(task: TreeTask, criterion: TreeCriterion, depth: u8, mss: u8) -> ModelConfig {
        ModelConfig::Tree {
            task,
            criterion,
            max_depth: depth,
            min_samples_split: mss,
        }
    }

    fn trained(facade: &dyn CoreFacade, id: DatasetId, config: ModelConfig) -> ModelHandle {
        let spec = TrainSpec {
            config,
            test_size: 0.3,
            dataset_label: "dataset".into(),
        };
        facade
            .load_builtin(id)
            .unwrap()
            .train(&spec, &|_, _| {}, &AtomicBool::new(false))
            .unwrap()
    }

    const C: TreeTask = TreeTask::Classification;
    const R: TreeTask = TreeTask::Regression;

    /// The views of the app (core) and scikit-learn's values (fixtures) coincide: same
    /// boxes, texts, colours and positions, rules, importances, paths and grids.
    #[test]
    fn real_trees_match_the_fixture_facade() {
        let cases = [
            (DatasetId::Iris, tree(C, TreeCriterion::Gini, 3, 2)),
            (DatasetId::Iris, tree(C, TreeCriterion::Entropy, 5, 5)),
            (DatasetId::Wine, tree(C, TreeCriterion::Gini, 4, 2)),
            (
                DatasetId::BreastCancer,
                tree(C, TreeCriterion::Entropy, 10, 2),
            ),
            (DatasetId::Titanic, tree(C, TreeCriterion::Gini, 4, 2)),
            (
                DatasetId::Diabetes,
                tree(R, TreeCriterion::SquaredError, 3, 2),
            ),
            (
                DatasetId::Diabetes,
                tree(R, TreeCriterion::AbsoluteError, 4, 2),
            ),
        ];
        compare(&cases);
    }

    /// California d8 (407 nodes). Needs the sklearn train size `n − ceil(t·n)`: with
    /// `floor((1 − t)·n)` it lost a row (0.7·20640 = 14447.999…).
    #[test]
    fn real_california_d8_matches_the_fixture_facade() {
        compare(&[(
            DatasetId::CaliforniaHousing,
            tree(R, TreeCriterion::SquaredError, 8, 10),
        )]);
    }

    fn compare(cases: &[(DatasetId, ModelConfig)]) {
        let go = AtomicBool::new(false);
        for &(id, config) in cases {
            let a = trained(&RealFacade, id, config);
            let b = trained(&FixtureFacade::from_checkout(), id, config);
            let what = format!("{id:?} {config:?}");
            assert_eq!(a.summary().tree, b.summary().tree, "{what}");
            assert_eq!(
                a.summary().model_classes,
                b.summary().model_classes,
                "{what}"
            );
            let (da, db) = (a.tree_diagram().unwrap(), b.tree_diagram().unwrap());
            assert_eq!(da.nodes.len(), db.nodes.len(), "{what}");
            for (x, y) in da.nodes.iter().zip(&db.nodes) {
                // The core prints a negative-zero impurity as "0.0"; the arm64 sklearn wheel
                // (FMA) writes "-0.0" in some constant leaves (documented deviation, F3 §8.4).
                let unsigned = |l: &Vec<String>| -> Vec<String> {
                    l.iter().map(|t| t.replace("= -0.0", "= 0.0")).collect()
                };
                assert_eq!(
                    unsigned(&x.lines),
                    unsigned(&y.lines),
                    "{what} nodo {}",
                    x.id
                );
                assert_eq!(x.fill, y.fill, "{what} nodo {}", x.id);
                assert_eq!(
                    (x.parent, x.is_left, x.depth),
                    (y.parent, y.is_left, y.depth)
                );
                assert!(
                    ((x.x + 0.5) / da.max_x - (y.x + 0.5) / db.max_x).abs() < 1e-9,
                    "{what} nodo {}",
                    x.id
                );
            }
            if id != DatasetId::CaliforniaHousing {
                assert_eq!(
                    a.tree_rules(RulesKind::View, false).unwrap(),
                    b.tree_rules(RulesKind::View, false).unwrap(),
                    "{what}"
                );
                assert_eq!(
                    a.tree_rules(RulesKind::View, true).unwrap(),
                    b.tree_rules(RulesKind::View, true).unwrap(),
                    "{what}"
                );
            }
            for method in [ImportanceMethod::Native, ImportanceMethod::Permutation] {
                let (ia, ib) = (
                    a.feature_importance(method, &|_, _| {}, &go).unwrap(),
                    b.feature_importance(method, &|_, _| {}, &go).unwrap(),
                );
                assert_eq!(ia.seed, ib.seed, "{what}");
                for (x, y) in ia.rows.iter().zip(&ib.rows) {
                    assert!(
                        (x.importance - y.importance).abs() < 1e-12,
                        "{what} {method:?}"
                    );
                    assert_eq!(x.feature_index, y.feature_index, "{what} {method:?}");
                }
            }
            if id != DatasetId::CaliforniaHousing {
                match (a.evaluate().unwrap(), b.evaluate().unwrap()) {
                    (Evaluation::Classification(x), Evaluation::Classification(y)) => {
                        assert!((x.accuracy - y.accuracy).abs() < 1e-12, "{what}");
                        assert_eq!(x.confusion, y.confusion, "{what}");
                    }
                    (Evaluation::Regression(x), Evaluation::Regression(y)) => {
                        assert!((x.r2 - y.r2).abs() < 1e-12, "{what}");
                    }
                    _ => panic!("{what}"),
                }
                let form = b.prediction_form().unwrap();
                let values: Vec<f64> = form
                    .controls
                    .iter()
                    .map(|c| match &c.control {
                        crate::ipc::FormControlKind::Checkbox { default } => {
                            f64::from(u8::from(*default))
                        }
                        crate::ipc::FormControlKind::Select {
                            options,
                            default_index,
                        } => options[*default_index as usize].code,
                        crate::ipc::FormControlKind::Slider { default, .. } => *default,
                    })
                    .collect();
                let path = |r: PredictionResult| match r {
                    PredictionResult::Class { path, .. } | PredictionResult::Value { path, .. } => {
                        path.unwrap()
                    }
                };
                let (pa, pb) = (
                    path(a.predict_one(&values).unwrap()),
                    path(b.predict_one(&values).unwrap()),
                );
                assert_eq!(pa.leaf, pb.leaf, "{what}");
                assert_eq!(pa.steps.len(), pb.steps.len(), "{what}");
                let grid = |m: &ModelHandle| m.decision_grid(0, 1, &|_, _| {}, &go).unwrap();
                let (ga, gb) = (grid(&a), grid(&b));
                assert_eq!((ga.x.len(), ga.y.len()), (gb.x.len(), gb.y.len()), "{what}");
                if let (GridValues::Classes { z: za }, GridValues::Classes { z: zb }) =
                    (&ga.values, &gb.values)
                {
                    assert_eq!(za, zb, "{what}");
                }
            }
            for template in [
                CodeTemplate::TreeViz,
                CodeTemplate::TreeText,
                CodeTemplate::DecisionPath,
                CodeTemplate::TreeUsage,
                CodeTemplate::TreeTraining,
            ] {
                assert_eq!(
                    a.code(template, None).unwrap(),
                    b.code(template, None).unwrap(),
                    "{what} {template:?}"
                );
            }
            if id != DatasetId::CaliforniaHousing {
                assert_eq!(
                    a.code(CodeTemplate::TreePython, None).unwrap().trim(),
                    b.code(CodeTemplate::TreePython, None).unwrap().trim(),
                    "{what}"
                );
            }
            for format in [
                ExportFormat::TreeRules,
                ExportFormat::TreePython,
                ExportFormat::Onnx,
                ExportFormat::PythonPackage,
            ] {
                assert!(!a.export(format).unwrap().is_empty(), "{what} {format:?}");
            }
        }
    }

    #[test]
    fn classification_on_a_continuous_target_is_d37() {
        let spec = TrainSpec {
            config: tree(C, TreeCriterion::Gini, 3, 2),
            test_size: 0.3,
            dataset_label: String::new(),
        };
        let error = RealFacade
            .load_builtin(DatasetId::CaliforniaHousing)
            .unwrap()
            .train(&spec, &|_, _| {}, &AtomicBool::new(false))
            .err()
            .unwrap();
        assert_eq!(
            error,
            crate::ipc::CommandError::invalid_target(crate::models::tree_needs_classes_message())
        );
    }
}
