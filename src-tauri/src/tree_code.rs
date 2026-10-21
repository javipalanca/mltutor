//! Python code of 🌲 Árboles that only depends on the dataset and the configuration
//! (fase3-arboles.md §7): the snippets of 🌲 Árbol, 📝 Texto, 🔮 Predicciones and 💾 Exportar,
//! and the training script (`arbol_entrenamiento.py.j2`). Both facades render them with
//! the templates embedded in `mltutor-core` (`codegen::render_named`); the functions of the
//! tree itself (`arbol_python*.py.j2`) need the tree and live in each facade.

use serde::Serialize;

use crate::ipc::{CodeTemplate, CommandError, ModelConfig};

/// How the app read a user CSV (`pd.read_csv` of the script must read the same table).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CsvSource {
    pub file: String,
    pub target: String,
    pub sep: String,
    pub decimal: String,
    pub encoding: String,
}

/// What the templates need from a trained tree.
pub struct TreeCodeInput<'a> {
    /// `DatasetId::slug()` or `"csv"`.
    pub dataset: &'a str,
    pub dataset_label: &'a str,
    pub csv: Option<CsvSource>,
    pub config: ModelConfig,
    pub test_size: f64,
    pub feature_names: &'a [String],
    pub class_names: Option<&'a [String]>,
}

/// Context of `arbol_entrenamiento.py.j2`: every variable, `null` when it does not apply
/// (the template is tested with minijinja in strict mode).
#[derive(Debug, Serialize)]
struct TrainingContext<'a> {
    dataset: &'a str,
    dataset_label: &'a str,
    csv_file: Option<&'a str>,
    csv_target: Option<&'a str>,
    csv_sep: Option<&'a str>,
    csv_decimal: Option<&'a str>,
    csv_encoding: Option<&'a str>,
    task: &'static str,
    criterion: &'static str,
    max_depth: u8,
    min_samples_split: u8,
    test_size: f64,
    feature_names: &'a [String],
    class_names: Option<&'a [String]>,
    sample: Option<Vec<f64>>,
}

fn template_error(error: impl std::fmt::Display) -> CommandError {
    CommandError::internal(format!("plantilla: {error}"))
}

/// Renders `template` (one of the tree snippets or the training script).
pub fn render(template: CodeTemplate, input: &TreeCodeInput<'_>) -> Result<String, CommandError> {
    use mltutor_core::codegen::render_named;
    let empty = serde_json::json!({});
    let text = match template {
        CodeTemplate::TreeViz | CodeTemplate::TreeText | CodeTemplate::DecisionPath => {
            render_named(template.source().file_name(), &empty)
        }
        CodeTemplate::TreeUsage => render_named(
            template.source().file_name(),
            &serde_json::json!({ "feature_names": input.feature_names }),
        ),
        CodeTemplate::TreeTraining => {
            let ModelConfig::Tree {
                task,
                criterion,
                max_depth,
                min_samples_split,
            } = input.config
            else {
                return Err(CommandError::not_tree());
            };
            let csv = input.csv.as_ref();
            let classifier = task == crate::ipc::TreeTask::Classification;
            render_named(
                template.source().file_name(),
                &TrainingContext {
                    dataset: input.dataset,
                    dataset_label: input.dataset_label,
                    csv_file: csv.map(|c| c.file.as_str()),
                    csv_target: csv.map(|c| c.target.as_str()),
                    csv_sep: csv.map(|c| c.sep.as_str()),
                    csv_decimal: csv.map(|c| c.decimal.as_str()),
                    csv_encoding: csv.map(|c| c.encoding.as_str()),
                    task: if classifier {
                        "classification"
                    } else {
                        "regression"
                    },
                    criterion: criterion.sklearn_name(),
                    max_depth,
                    min_samples_split,
                    test_size: input.test_size,
                    feature_names: input.feature_names,
                    class_names: input.class_names.filter(|_| classifier),
                    sample: None,
                },
            )
        }
        _ => return Err(CommandError::not_tree()),
    };
    text.map(|t| t.trim_start_matches('\n').to_owned())
        .map_err(template_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::{TreeCriterion, TreeTask};

    fn fixture() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/parity/code/trees_training.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    /// The training script is exactly the one verified by `tools/export_golden.py`.
    #[test]
    fn training_script_matches_the_golden_text() {
        let golden = fixture();
        let mut checked = 0;
        for (name, case) in golden["expected"].as_object().unwrap() {
            let c = &case["context"];
            if !c["sample"].is_null() || c["dataset"] == "csv" {
                continue;
            }
            let strings = |v: &serde_json::Value| -> Vec<String> {
                v.as_array()
                    .map(|a| a.iter().map(|s| s.as_str().unwrap().to_owned()).collect())
                    .unwrap_or_default()
            };
            let features = strings(&c["feature_names"]);
            let classes = strings(&c["class_names"]);
            let criterion: TreeCriterion = serde_json::from_value(c["criterion"].clone()).unwrap();
            let task: TreeTask = serde_json::from_value(c["task"].clone()).unwrap();
            let input = TreeCodeInput {
                dataset: c["dataset"].as_str().unwrap(),
                dataset_label: c["dataset_label"].as_str().unwrap(),
                csv: None,
                config: ModelConfig::Tree {
                    task,
                    criterion,
                    max_depth: c["max_depth"].as_u64().unwrap() as u8,
                    min_samples_split: c["min_samples_split"].as_u64().unwrap() as u8,
                },
                test_size: c["test_size"].as_f64().unwrap(),
                feature_names: &features,
                class_names: (!classes.is_empty()).then_some(classes.as_slice()),
            };
            let text = render(CodeTemplate::TreeTraining, &input).unwrap();
            mltutor_testkit::assert_template_text_trimmed(
                "code/trees_training.json",
                &format!("/expected/{name}/text"),
                &text,
            );
            checked += 1;
        }
        assert!(checked >= 5, "{checked}");
    }
}
