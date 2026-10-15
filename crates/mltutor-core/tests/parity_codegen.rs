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
