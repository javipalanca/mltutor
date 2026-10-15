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
