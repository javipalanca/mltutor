//! C12: `content/code/*.j2` rendered with minijinja equals jinja2's output
//! (`tests/parity/code/regression.json`, `regression_training.json`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::fixture;
use mltutor_core::codegen::{self, CodeTemplate, RegressionTrainingContext};

#[test]
fn regression_templates_match_jinja2() {
    let fx = fixture("code/regression.json");
    let mut n = 0;
    for (name, case) in fx["expected"].as_object().unwrap() {
        let text = case["text"].as_str().unwrap();
        let rendered = codegen::render_named(name, &case["context"]).unwrap();
        if case["compare"].as_str().unwrap().contains("strip") {
            assert_eq!(
                rendered.trim_matches('\n'),
                text.trim_matches('\n'),
                "{name}"
            );
        } else {
            assert_eq!(rendered, text, "{name}");
        }
        n += 1;
    }
    assert_eq!(n, 6);
}

#[test]
fn training_script_matches_jinja2_exactly() {
    let fx = fixture("code/regression_training.json");
    for (case, v) in fx["expected"].as_object().unwrap() {
        // Through the typed context: every key, nulls included (strict mode).
        let ctx: RegressionTrainingContext = serde_json::from_value(v["context"].clone()).unwrap();
        let rendered = codegen::render(CodeTemplate::RegressionTraining, &ctx).unwrap();
        assert_eq!(rendered, v["text"].as_str().unwrap(), "{case}");
    }
}

#[test]
fn every_template_name_is_embedded() {
    for t in [
        CodeTemplate::ConfusionMatrix,
        CodeTemplate::PrecisionByClass,
        CodeTemplate::PredVsReal,
        CodeTemplate::ErrorDistribution,
        CodeTemplate::DecisionBoundary,
        CodeTemplate::RegressionCode,
        CodeTemplate::RegressionTraining,
        CodeTemplate::PackageScript,
        CodeTemplate::PackageReadme,
    ] {
        assert!(codegen::template_source(t.file_name()).is_some(), "{t:?}");
    }
    // Strict mode: a missing variable is an error, not an empty string.
    assert!(codegen::render(CodeTemplate::DecisionBoundary, &serde_json::json!({})).is_err());
}

#[test]
fn tree_training_script_matches_jinja2_exactly() {
    use mltutor_core::codegen::TreeTrainingContext;
    let fx = fixture("code/trees_training.json");
    let mut n = 0;
    for (case, v) in fx["expected"].as_object().unwrap() {
        let ctx: TreeTrainingContext = serde_json::from_value(v["context"].clone()).unwrap();
        let rendered = codegen::render(CodeTemplate::TreeTraining, &ctx).unwrap();
        assert_eq!(rendered, v["text"].as_str().unwrap(), "{case}");
        n += 1;
    }
    assert_eq!(n, 16);
}

#[test]
fn tree_templates_are_embedded() {
    for t in [
        CodeTemplate::TreeViz,
        CodeTemplate::TreeText,
        CodeTemplate::DecisionPath,
        CodeTemplate::TreeUsage,
        CodeTemplate::TreePython,
        CodeTemplate::TreePythonLiteral,
        CodeTemplate::TreePackageScript,
        CodeTemplate::TreePackageReadme,
        CodeTemplate::TreeTraining,
    ] {
        assert!(codegen::template_source(t.file_name()).is_some(), "{t:?}");
    }
}

#[test]
fn knn_templates_are_embedded() {
    for t in [
        CodeTemplate::KnnTraining,
        CodeTemplate::KnnPackageScript,
        CodeTemplate::KnnPackageReadme,
    ] {
        assert!(codegen::template_source(t.file_name()).is_some(), "{t:?}");
    }
}

/// F4: `knn_entrenamiento.py.j2` rendered from the typed context gives exactly the
/// script that `tools/golden_knn.py` executed and checked against the fixtures.
#[test]
fn knn_training_script_matches_jinja2_exactly() {
    use mltutor_core::codegen::KnnTrainingContext;
    let fx = fixture("code/knn_training.json");
    let mut n = 0;
    for (case, v) in fx["expected"].as_object().unwrap() {
        let ctx: KnnTrainingContext = serde_json::from_value(v["context"].clone()).unwrap();
        // Round trip: every key is sent (strict templates).
        assert_eq!(serde_json::to_value(&ctx).unwrap(), v["context"], "{case}");
        let rendered = codegen::render(CodeTemplate::KnnTraining, &ctx).unwrap();
        assert_eq!(rendered, v["text"].as_str().unwrap(), "{case}");
        n += 1;
    }
    assert_eq!(n, 19);
}

/// D63: `frontera_decision.py.j2` with `datos = "todos"` (🔍 KNN), executed after the
/// KNN training script in `tools/golden_knn.py`; without `datos` the text is F2/F3's.
#[test]
fn knn_boundary_all_rows_matches_jinja2() {
    use mltutor_core::codegen::DecisionBoundaryContext;
    let fx = fixture("code/knn_boundary.json");
    let mut n = 0;
    for (case, v) in fx["expected"].as_object().unwrap() {
        let ctx: DecisionBoundaryContext = serde_json::from_value(v["context"].clone()).unwrap();
        assert_eq!(
            ctx.datos.as_deref(),
            Some(DecisionBoundaryContext::ALL_ROWS)
        );
        let rendered = codegen::render(CodeTemplate::DecisionBoundary, &ctx).unwrap();
        assert_eq!(rendered, v["text"].as_str().unwrap(), "{case}");
        let plain = DecisionBoundaryContext::new(
            ctx.feature_indices,
            ctx.feature_names.clone(),
            ctx.class_names.clone(),
        );
        let rendered = codegen::render(CodeTemplate::DecisionBoundary, &plain).unwrap();
        assert_eq!(rendered, v["default_text"].as_str().unwrap(), "{case}");
        let again = plain.clone().all_rows();
        assert_eq!(again, ctx);
        n += 1;
    }
    assert_eq!(n, 4);
}

/// 🧠 D74: `red_entrenamiento.py.j2` with the typed context built from the lesson's
/// configuration equals the script `export_golden.py` executed (`code/nn_training.json`).
#[test]
fn nn_training_script_matches_exactly() {
    use mltutor_core::codegen::{NnTrainingContext, ScriptDataset};
    use mltutor_core::dataset::TaskType;
    use mltutor_core::nn_data::{Activation, NnArchitecture, NnTraining, OptimizerKind};
    let fx = fixture("code/nn_training.json");
    let mut n = 0;
    for (case, v) in fx["expected"].as_object().unwrap() {
        let c = &v["context"];
        let stat = fixture(&format!("nn/stat/{case}.json"));
        let input = &stat["input"];
        let cfg = &input["nn_config"];
        let sizes: Vec<u16> = cfg["architecture"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u16)
            .collect();
        let arch = NnArchitecture {
            hidden: sizes[1..sizes.len() - 1].to_vec(),
            activation: Activation::parse(cfg["activation"].as_str().unwrap()).unwrap(),
            output_activation: Activation::parse(cfg["output_activation"].as_str().unwrap())
                .unwrap(),
            dropout: cfg["dropout_rate"].as_f64().unwrap() as f32,
            batch_size: cfg["batch_size"].as_u64().unwrap() as u16,
            optimizer: OptimizerKind::parse(cfg["optimizer"].as_str().unwrap()).unwrap(),
        };
        let training = NnTraining {
            learning_rate: input["lr"].as_f64().unwrap() as f32,
            epochs: input["epochs"].as_u64().unwrap() as u16,
            validation_pct: (input["validation_split"].as_f64().unwrap() * 100.0).round() as u8,
            patience: input["early_stopping"]
                .as_bool()
                .unwrap()
                .then(|| input["patience"].as_u64().unwrap() as u8),
            lr_factor: input["reduce_lr"]
                .as_bool()
                .unwrap()
                .then(|| input["lr_factor"].as_f64().unwrap() as f32),
            seed: 42,
        };
        let s = |k: &str| c[k].as_str().map(str::to_owned);
        let data = ScriptDataset {
            dataset: s("dataset").unwrap(),
            dataset_label: s("dataset_label").unwrap(),
            csv_file: s("csv_file"),
            csv_target: s("csv_target"),
            csv_sep: s("csv_sep"),
            csv_decimal: s("csv_decimal"),
            csv_encoding: s("csv_encoding"),
            feature_names: common::strings(&c["feature_names"]),
            class_names: c["class_names"]
                .is_array()
                .then(|| common::strings(&c["class_names"])),
        };
        let task = if c["task"] == "classification" {
            TaskType::Classification
        } else {
            TaskType::Regression
        };
        let n_classes = c["class_names"].as_array().map_or(0, Vec::len);
        let ctx = NnTrainingContext::new(data, task, n_classes, &arch, &training).unwrap();
        assert_eq!(&serde_json::to_value(&ctx).unwrap(), c, "{case}: context");
        let rendered = codegen::render(CodeTemplate::NnTraining, &ctx).unwrap();
        assert_eq!(rendered, v["script"].as_str().unwrap(), "{case}: script");
        n += 1;
    }
    assert_eq!(n, 6);
}
