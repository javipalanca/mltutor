//! Phase 6: the corpus of everything the app exports, for `tools/verify_exports.py`.
//!
//! An `#[ignore]` test (it writes files). For every model family and a set of
//! representative datasets it goes through the same domain layer as the commands:
//! [`DataState`] (built-in selection, or `load_csv_bytes` + `set_csv_target`),
//! [`ModelState::train`] (`validate_train` and the checks of `train_model`), and then
//! [`predict_with`], [`code_of`] and [`export_bytes`] (the bodies of `predict_one`,
//! `generate_code` and `export_model`, with their validations). It writes each export
//! and code template next to the inputs (every test row) and Rust's predictions:
//!
//! ```sh
//! MLTUTOR_EXPORT_CORPUS=/tmp/corpus cargo test -p mltutor --bin mltutor \
//!     export_corpus::export_corpus -- --ignored --exact
//! uv run … python tools/verify_exports.py /tmp/corpus
//! ```
//!
//! Without `MLTUTOR_EXPORT_CORPUS` it writes to `target/export-corpus`. The output is
//! deterministic (fixed seeds, fixed metadata date, `serde_json` in insertion order):
//! two runs give the same bytes.
//!
//! Before writing, the generator checks that the model's split is the D7 split (or the
//! stratified one of 🧠, D72) **row by row**: the rows of the packages' `check`, the
//! confusion matrix or `y_true` of 📈 Evaluación, and the training rows of the KNN
//! neighbours must all come from those indices.
//!
//! Layout: `index.json` (the case names) and one folder per case with `case.json`
//! (configuration, names, class values, the inputs, Rust's predictions and metrics,
//! and which `content/code/*.j2` produced each file), `modelo.onnx`, the ZIP package
//! (`paquete.zip`), the rendered templates in `code/<key>.py` and, for CSV cases, the
//! CSV the scripts read.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use mltutor_core as core;
use ndarray::Axis;
use serde_json::{json, Value};

use crate::data::DataState;
use crate::ipc::{
    CodeExtra, CodeTemplate, DatasetId, DatasetRef, Evaluation, ExportFormat, KnnMetric,
    KnnWeights, LessonId, ModelConfig, NnActivation, NnArchitecture, NnOptimizer, NnTraining,
    PredictionResult, RulesKind, TaskChoice, TrainRequest, TreeCriterion, TreeTask,
};
use crate::models::{code_of, export_bytes, predict_with, ModelState, TrainedModel};
use crate::real_facade::RealFacade;

/// `random_state=42` of the split (D7), as in `real_models`.
const SEED: u32 = 42;
/// `test_size` of the classic lessons in the corpus (the default of the sliders).
const TEST_SIZE: f64 = 0.3;
/// Fixed date of the 🧠 metadata (the UI sends the local time).
const TRAINED_AT: &str = "2026-01-01 12:00";
/// Moons is generated once per session with this seed (`real_facade`, D9).
const MOONS_SEED: u32 = 42;

enum Source {
    Builtin(DatasetId),
    Csv {
        file: &'static str,
        target: &'static str,
        choice: TaskChoice,
        bytes: Vec<u8>,
    },
}

struct Case {
    name: &'static str,
    lesson: LessonId,
    source: Source,
    config: ModelConfig,
}

impl Case {
    fn is_nn(&self) -> bool {
        self.lesson == LessonId::Nn
    }

    fn test_size(&self) -> f64 {
        if self.is_nn() {
            core::nn_data::LESSON_TEST_SIZE
        } else {
            TEST_SIZE
        }
    }

    /// Whether the scripts can load the dataset offline: California needs a download
    /// in scikit-learn (and Diamonds is not in `tools/data/seaborn`). The verifier skips
    /// their templates; ONNX and packages do not load data.
    fn templates_offline(&self) -> bool {
        !matches!(
            self.source,
            Source::Builtin(DatasetId::CaliforniaHousing | DatasetId::Diamonds)
        )
    }
}

fn lesson_slug(lesson: LessonId) -> &'static str {
    match lesson {
        LessonId::Regression => "regresion",
        LessonId::Trees => "arboles",
        LessonId::Knn => "knn",
        LessonId::Nn => "redes",
    }
}

fn builtin_dataset(id: DatasetId) -> core::Dataset {
    match id {
        DatasetId::Moons => core::dataset::moons_dataset(MOONS_SEED),
        other => core::dataset::load_builtin(core::DatasetId::from_slug(other.slug()).unwrap()),
    }
    .unwrap_or_else(|e| panic!("{id:?}: {e}"))
}

fn class_codes(data: &core::Dataset) -> &[u32] {
    match &data.target {
        core::Target::Classes { codes, .. } => codes,
        core::Target::Values { .. } => panic!("{} is not a classification dataset", data.label),
    }
}

/// A CSV made from a built-in dataset: separator `sep`, decimal comma if
/// `decimal_comma`, header `header` (the feature columns by default) and the class of
/// each row from `label(row, code)`.
fn csv_from(
    id: DatasetId,
    sep: char,
    decimal_comma: bool,
    header: Option<&[&str]>,
    label: impl Fn(usize, u32) -> String,
) -> String {
    let data = builtin_dataset(id);
    let codes = class_codes(&data);
    let mut columns: Vec<String> = match header {
        Some(h) => h.iter().map(|s| (*s).to_owned()).collect(),
        None => data.feature_columns.clone(),
    };
    columns.truncate(data.feature_columns.len() + 1);
    let mut text = columns.join(&sep.to_string());
    text.push('\n');
    for (i, (row, &code)) in data.features.axis_iter(Axis(0)).zip(codes).enumerate() {
        for v in row {
            let v = format!("{v}");
            text.push_str(&if decimal_comma {
                v.replace('.', ",")
            } else {
                v
            });
            text.push(sep);
        }
        text.push_str(&label(i, code));
        text.push('\n');
    }
    text
}

/// Every character is in Latin-1, where Windows-1252 and Unicode coincide.
fn cp1252(text: &str) -> Vec<u8> {
    text.chars()
        .map(|c| u8::try_from(u32::from(c)).unwrap_or_else(|_| panic!("{c} is not Latin-1")))
        .collect()
}

/// `;`, decimal comma, Windows-1252, text classes and accented headers (Iris): it
/// exercises `csv_sep`, `csv_decimal` and `csv_encoding` of every training script.
fn medidas_csv() -> Vec<u8> {
    let header = [
        "longitud_sépalo",
        "anchura_sépalo",
        "longitud_pétalo",
        "tamaño",
        "especie",
    ];
    let names = ["lirio", "rosa", "tulipán"];
    cp1252(&csv_from(
        DatasetId::Iris,
        ';',
        true,
        Some(&header),
        |_, c| names[c as usize].to_owned(),
    ))
}

/// Integer classes that are NOT the codes (`code * scale + offset`): the ONNX labels are
/// the class values, and the scripts must map them.
fn numeric_classes_csv(id: DatasetId, target: &str, scale: u32, offset: u32) -> Vec<u8> {
    let data = builtin_dataset(id);
    let mut header: Vec<&str> = data.feature_columns.iter().map(String::as_str).collect();
    header.push(target);
    csv_from(id, ',', false, Some(&header), |_, c| {
        (c * scale + offset).to_string()
    })
    .into_bytes()
}

/// Tab-separated, text classes (Pingüinos).
fn tab_csv() -> Vec<u8> {
    let data = builtin_dataset(DatasetId::Penguins);
    let names = data.class_names().unwrap().to_vec();
    let mut header: Vec<&str> = data.feature_columns.iter().map(String::as_str).collect();
    header.push("especie");
    csv_from(DatasetId::Penguins, '\t', false, Some(&header), |_, c| {
        names[c as usize].clone()
    })
    .into_bytes()
}

/// Iris with two rare classes: `rara_prueba` on ONE test row (absent from the train set,
/// so the model never predicts it) and `rara_entrenamiento` on ONE train row (absent from
/// the test set). Both must keep their place in the confusion matrix and the report.
fn rare_classes_csv() -> Vec<u8> {
    let iris = builtin_dataset(DatasetId::Iris);
    let names = iris.class_names().unwrap().to_vec();
    let split = core::split::train_test_split(iris.n_rows(), TEST_SIZE, SEED).unwrap();
    let (test_row, train_row) = (split.test[0], split.train[0]);
    let mut header: Vec<&str> = iris.feature_columns.iter().map(String::as_str).collect();
    header.push("clase");
    csv_from(DatasetId::Iris, ',', false, Some(&header), |i, c| {
        if i == test_row {
            "rara_prueba".to_owned()
        } else if i == train_row {
            "rara_entrenamiento".to_owned()
        } else {
            names[c as usize].clone()
        }
    })
    .into_bytes()
}

/// Wine with cultivars 2, 3, 4 (code + 2) and a cultivar 1 on ONE test row, absent from
/// the train set. The class value, the dataset code and the position in `classes_` then
/// all differ (value 2 = code 1 = position 0), the model sees 3 of the 4 classes and the
/// ONNX labels are the class values of `model_classes` only.
fn shifted_rare_classes_csv() -> Vec<u8> {
    let wine = builtin_dataset(DatasetId::Wine);
    let split = core::split::train_test_split(wine.n_rows(), TEST_SIZE, SEED).unwrap();
    let test_row = split.test[0];
    let mut header: Vec<&str> = wine.feature_columns.iter().map(String::as_str).collect();
    header.push("cultivar");
    csv_from(DatasetId::Wine, ',', false, Some(&header), |i, c| {
        if i == test_row {
            "1".to_owned()
        } else {
            (c + 2).to_string()
        }
    })
    .into_bytes()
}

/// `tests/parity/csv/<name>.json` → the CSV bytes it describes.
fn parity_csv(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/parity/csv")
        .join(format!("{name}.json"));
    let doc: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    doc["input"]["csv_text"]
        .as_str()
        .unwrap()
        .as_bytes()
        .to_vec()
}

fn logistic() -> ModelConfig {
    ModelConfig::Logistic { max_iter: 1000 }
}

fn tree(task: TreeTask, criterion: TreeCriterion, max_depth: u8, mss: u8) -> ModelConfig {
    ModelConfig::Tree {
        task,
        criterion,
        max_depth,
        min_samples_split: mss,
    }
}

fn knn(k: u8, weights: KnnWeights, metric: KnnMetric) -> ModelConfig {
    ModelConfig::Knn {
        n_neighbors: k,
        weights,
        metric,
    }
}

fn nn(hidden: Vec<u16>, output: NnActivation, optimizer: NnOptimizer, lr: f32) -> ModelConfig {
    ModelConfig::NeuralNet {
        arch: NnArchitecture {
            hidden,
            activation: NnActivation::Relu,
            output_activation: output,
            dropout: 0.2,
            batch_size: 32,
            optimizer,
        },
        training: NnTraining {
            learning_rate: lr,
            epochs: 30,
            validation_pct: 20,
            patience: Some(10),
            lr_factor: Some(0.5),
            seed: 42,
        },
    }
}

#[rustfmt::skip]
fn cases() -> Vec<Case> {
    use DatasetId as D;
    use KnnMetric::{Euclidean, Manhattan, Minkowski};
    use KnnWeights::{Distance, Uniform};
    use LessonId as L;
    use TreeCriterion::{AbsoluteError, Entropy, Gini, SquaredError};
    use TreeTask::{Classification as TC, Regression as TR};
    let b = |name, lesson, id, config| Case { name, lesson, source: Source::Builtin(id), config };
    let csv = |file: &'static str, target: &'static str, choice, bytes: Vec<u8>| Source::Csv { file, target, choice, bytes };
    let c = |name, lesson, source, config| Case { name, lesson, source, config };
    let (cls, reg) = (TaskChoice::Classification, TaskChoice::Regression);
    let medidas = || csv("medidas_ñ.csv", "especie", cls, medidas_csv());
    let numeric = || csv("numeric_regression.csv", "y", reg, parity_csv("numeric_regression"));
    // Wine with cultivars 1, 2, 3 and Breast Cancer with 2 = malignant and 4 = benign
    // (scikit-learn's codes 0/1, times 2 plus 2).
    let vinos = || csv("vinos.csv", "cultivar", cls, numeric_classes_csv(D::Wine, "cultivar", 1, 1));
    let cancer = || csv("cancer.csv", "diagnostico", cls, numeric_classes_csv(D::BreastCancer, "diagnostico", 2, 2));
    let raras = || csv("raras.csv", "clase", cls, rare_classes_csv());
    let vinos_raros = || csv("vinos_raros.csv", "cultivar", cls, shifted_rare_classes_csv());
    let tab = || csv("pinguinos.tsv", "especie", cls, tab_csv());
    let relu = |h: Vec<u16>, out, opt, lr| nn(h, out, opt, lr);
    vec![
        // 📊 Regresión (lineal y logística)
        b("lr_diabetes_linear", L::Regression, D::Diabetes, ModelConfig::Linear),
        b("lr_california_linear", L::Regression, D::CaliforniaHousing, ModelConfig::Linear),
        b("lr_mpg_linear", L::Regression, D::Mpg, ModelConfig::Linear),
        b("lr_iris_logistic", L::Regression, D::Iris, logistic()),
        b("lr_breast_cancer_logistic", L::Regression, D::BreastCancer, logistic()),
        b("lr_titanic_logistic", L::Regression, D::Titanic, logistic()),
        b("lr_penguins_logistic", L::Regression, D::Penguins, logistic()),
        b("lr_digits_logistic", L::Regression, D::Digits, logistic()),
        c("lr_csv_medidas_logistic", L::Regression, medidas(), logistic()),
        c("lr_csv_vinos_logistic", L::Regression, vinos(), logistic()),
        c("lr_csv_cancer_2_4_logistic", L::Regression, cancer(), logistic()),
        c("lr_csv_raras_logistic", L::Regression, raras(), logistic()),
        c("lr_csv_vinos_raros_logistic", L::Regression, vinos_raros(), logistic()),
        c("lr_csv_tab_logistic", L::Regression, tab(), logistic()),
        c("lr_csv_numeric_linear", L::Regression, numeric(), ModelConfig::Linear),
        // 🌲 Árboles
        b("tree_iris_gini_d3", L::Trees, D::Iris, tree(TC, Gini, 3, 2)),
        b("tree_breast_cancer_entropy_d10", L::Trees, D::BreastCancer, tree(TC, Entropy, 10, 2)),
        b("tree_titanic_gini_d4", L::Trees, D::Titanic, tree(TC, Gini, 4, 2)),
        b("tree_moons_gini_d5", L::Trees, D::Moons, tree(TC, Gini, 5, 2)),
        b("tree_diabetes_squared_error_d3", L::Trees, D::Diabetes, tree(TR, SquaredError, 3, 2)),
        b("tree_diabetes_absolute_error_d2", L::Trees, D::Diabetes, tree(TR, AbsoluteError, 2, 2)),
        b("tree_tips_squared_error_d4", L::Trees, D::Tips, tree(TR, SquaredError, 4, 5)),
        b("tree_california_squared_error_d8", L::Trees, D::CaliforniaHousing, tree(TR, SquaredError, 8, 10)),
        c("tree_csv_vinos_gini_d3", L::Trees, vinos(), tree(TC, Gini, 3, 2)),
        c("tree_csv_cancer_2_4_gini_d3", L::Trees, cancer(), tree(TC, Gini, 3, 2)),
        c("tree_csv_medidas_entropy_d4", L::Trees, medidas(), tree(TC, Entropy, 4, 2)),
        c("tree_csv_raras_gini_d4", L::Trees, raras(), tree(TC, Gini, 4, 2)),
        c("tree_csv_vinos_raros_gini_d3", L::Trees, vinos_raros(), tree(TC, Gini, 3, 2)),
        c("tree_csv_tab_gini_d3", L::Trees, tab(), tree(TC, Gini, 3, 2)),
        // 🔍 KNN
        b("knn_iris_k5_uniform_minkowski", L::Knn, D::Iris, knn(5, Uniform, Minkowski)),
        b("knn_wine_k7_distance_manhattan", L::Knn, D::Wine, knn(7, Distance, Manhattan)),
        b("knn_breast_cancer_k10_distance_euclidean", L::Knn, D::BreastCancer, knn(10, Distance, Euclidean)),
        b("knn_titanic_k5_uniform_minkowski", L::Knn, D::Titanic, knn(5, Uniform, Minkowski)),
        b("knn_titanic_k7_distance_euclidean", L::Knn, D::Titanic, knn(7, Distance, Euclidean)),
        // 64 features: brute force (algorithm='auto' only uses kd_tree up to 15).
        b("knn_digits_k5_uniform_minkowski", L::Knn, D::Digits, knn(5, Uniform, Minkowski)),
        b("knn_moons_k15_distance_euclidean", L::Knn, D::Moons, knn(15, Distance, Euclidean)),
        b("knn_diabetes_k10_distance_manhattan", L::Knn, D::Diabetes, knn(10, Distance, Manhattan)),
        c("knn_csv_medidas_k15_distance_euclidean", L::Knn, medidas(), knn(15, Distance, Euclidean)),
        c("knn_csv_cancer_2_4_k5_uniform", L::Knn, cancer(), knn(5, Uniform, Minkowski)),
        c("knn_csv_raras_k3_uniform", L::Knn, raras(), knn(3, Uniform, Minkowski)),
        c("knn_csv_vinos_raros_k5_distance", L::Knn, vinos_raros(), knn(5, Distance, Manhattan)),
        c("knn_csv_numeric_k5_uniform", L::Knn, numeric(), knn(5, Uniform, Minkowski)),
        // 🧠 Redes neuronales
        b("nn_iris_softmax", L::Nn, D::Iris, relu(vec![10, 10], NnActivation::Softmax, NnOptimizer::Adam, 0.01)),
        b("nn_breast_cancer_sigmoid", L::Nn, D::BreastCancer, relu(vec![16, 8], NnActivation::Sigmoid, NnOptimizer::Rmsprop, 0.001)),
        // D87: a tanh output in binary is not a probability (output_is_probability = false).
        b("nn_breast_cancer_tanh", L::Nn, D::BreastCancer, relu(vec![16], NnActivation::Tanh, NnOptimizer::Adam, 0.01)),
        b("nn_diabetes_linear", L::Nn, D::Diabetes, relu(vec![32, 16], NnActivation::Linear, NnOptimizer::Adam, 0.01)),
        c("nn_csv_cancer_2_4_sigmoid", L::Nn, cancer(), relu(vec![16], NnActivation::Sigmoid, NnOptimizer::Adam, 0.01)),
        c("nn_csv_medidas_softmax", L::Nn, medidas(), relu(vec![8], NnActivation::Softmax, NnOptimizer::Sgd, 0.1)),
    ]
}

/// The core dataset the app builds from `source` (for the inputs: the domain layer only
/// exposes views).
fn core_dataset(source: &Source) -> core::Dataset {
    match source {
        Source::Builtin(id) => builtin_dataset(*id),
        Source::Csv {
            file,
            target,
            choice,
            bytes,
        } => {
            let choice = match choice {
                TaskChoice::Auto => core::TaskChoice::Auto,
                TaskChoice::Classification => core::TaskChoice::Classification,
                TaskChoice::Regression => core::TaskChoice::Regression,
            };
            core::csv::read_csv(bytes)
                .and_then(|t| t.to_dataset(target, choice, file))
                .unwrap()
        }
    }
}

/// Selects or registers the dataset like the UI does and trains through
/// [`ModelState::train`].
fn train(case: &Case) -> Arc<TrainedModel> {
    let data = DataState::new(Arc::new(RealFacade));
    let cancel = AtomicBool::new(false);
    let dataset = match &case.source {
        Source::Builtin(id) => DatasetRef::Builtin { id: *id },
        Source::Csv {
            file,
            target,
            choice,
            bytes,
        } => {
            let preview = data.load_csv_bytes(file, bytes, &cancel).unwrap();
            let loaded = data
                .set_csv_target(preview.draft_id, target, *choice, 1, &cancel)
                .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
            loaded.dataset.dataset
        }
    };
    let models = ModelState::default();
    let req = TrainRequest {
        lesson: case.lesson,
        dataset,
        config: case.config.clone(),
        test_size: case.test_size(),
        seq: 1,
        task_id: 1,
    };
    let info = models
        .train(
            &data,
            &req,
            &crate::tasks::CancelToken::default(),
            &|_, _| {},
        )
        .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
    models.model(info.model_id).unwrap()
}

/// The templates each lesson offers for this model (the `ModelCode` blocks of the UI).
fn templates(lesson: LessonId, classifier: bool) -> Vec<CodeTemplate> {
    use CodeTemplate as T;
    let mut out = match lesson {
        LessonId::Regression => vec![T::RegressionCode],
        LessonId::Trees => vec![
            T::TreeTraining,
            T::TreeViz,
            T::TreeText,
            T::DecisionPath,
            T::TreeUsage,
            T::TreePython,
        ],
        LessonId::Knn => vec![T::KnnTraining],
        LessonId::Nn => vec![T::NnTraining, T::NnNumpyPredictor],
    };
    if classifier {
        out.extend([T::ConfusionMatrix, T::PrecisionByClass]);
        // The 🧠 surface hides the code (it fixes the other features at their means).
        if lesson != LessonId::Nn {
            out.push(T::DecisionBoundary);
        }
    } else {
        out.extend([T::PredVsReal, T::ErrorDistribution]);
    }
    out
}

/// Templates inside the ZIP packages (rendered by `mltutor-export`).
fn package_templates(lesson: LessonId) -> Vec<&'static str> {
    use core::codegen::CodeTemplate as T;
    match lesson {
        LessonId::Regression => vec![T::PackageScript, T::PackageReadme],
        LessonId::Trees => vec![T::TreePackageScript, T::TreePackageReadme],
        LessonId::Knn => vec![T::KnnPackageScript, T::KnnPackageReadme],
        LessonId::Nn => vec![T::NnPackageScript, T::NnPackageReadme, T::NnNumpyPredictor],
    }
    .into_iter()
    .map(T::file_name)
    .collect()
}

fn template_key(template: CodeTemplate) -> String {
    serde_json::to_value(template)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Rust's prediction of one row, from the 🔮 form path. Probabilities are in the order
/// of `model_classes`, matched by class code.
fn prediction(
    m: &TrainedModel,
    row: &[f64],
    model_classes: &[u32],
    split_train: &[usize],
) -> Value {
    let result = predict_with(m, row).unwrap();
    let (path, neighbors) = match &result {
        PredictionResult::Class {
            path, neighbors, ..
        }
        | PredictionResult::Value {
            path, neighbors, ..
        } => (path, neighbors),
    };
    let path = path.as_ref().map(|p| {
        json!({
            "steps": p.steps.iter().map(|s| json!({
                "feature": s.feature, "name": s.name, "value": s.value,
                "threshold": s.threshold, "goes_left": s.goes_left,
            })).collect::<Vec<_>>(),
            "leaf_value": p.leaf_value,
        })
    });
    let knn = neighbors.as_ref().map(|n| {
        for r in &n.rows {
            // The neighbour's row of the dataset is the train row of the D7 split.
            assert_eq!(
                split_train[r.train_row as usize], r.dataset_row as usize,
                "KNN neighbour outside the split"
            );
        }
        json!({
            "train_rows": n.rows.iter().map(|r| r.train_row).collect::<Vec<_>>(),
            "distances": n.rows.iter().map(|r| r.distance).collect::<Vec<_>>(),
            "tie_at_kth": n.tie_at_kth.as_ref().map(|t| json!({
                "n_tied": t.n_tied, "slots": t.slots, "distance": t.distance,
            })),
        })
    });
    match result {
        PredictionResult::Class {
            index, name, proba, ..
        } => {
            let proba = proba.map(|p| {
                model_classes
                    .iter()
                    .map(|&code| {
                        p.iter()
                            .find(|c| c.code == code)
                            .map(|c| c.probability)
                            .unwrap_or_else(|| panic!("no probability for class {code}"))
                    })
                    .collect::<Vec<_>>()
            });
            json!({ "code": index, "name": name, "proba": proba, "knn": knn, "path": path })
        }
        PredictionResult::Value { value, .. } => {
            json!({ "value": value, "knn": knn, "path": path })
        }
    }
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// `modelo.json` of a ZIP package.
fn package_model_json(zip_bytes: &[u8]) -> Value {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip_bytes)).unwrap();
    let mut text = String::new();
    archive
        .by_name("modelo.json")
        .unwrap()
        .read_to_string(&mut text)
        .unwrap();
    serde_json::from_str(&text).unwrap()
}

/// The split of the model, checked row by row against what the app exposes.
fn check_split(
    case: &Case,
    m: &TrainedModel,
    dataset: &core::Dataset,
    split: &core::split::TrainTestSplit,
    package: &[u8],
    predictions: &[Value],
) {
    let info = m.info();
    assert_eq!(
        (split.train.len(), split.test.len()),
        (info.n_train as usize, info.n_test as usize),
        "{}: split sizes",
        case.name
    );
    // The packages check their first test rows: they must be the rows of the split.
    let check = package_model_json(package)["check"]["X"].clone();
    let rows = check.as_array().unwrap();
    assert!(!rows.is_empty(), "{}: empty package check", case.name);
    for (i, row) in rows.iter().enumerate() {
        let want: Vec<f64> = dataset.features.row(split.test[i]).to_vec();
        let got: Vec<f64> = serde_json::from_value(row.clone()).unwrap();
        assert_eq!(got, want, "{}: package check row {i}", case.name);
    }
    if case.is_nn() {
        return;
    }
    // 📈 Evaluación of the same rows.
    match m.model.evaluate().unwrap() {
        Evaluation::Classification(e) => {
            let codes = class_codes(dataset);
            let k = e.confusion.cols as usize;
            let mut confusion = vec![0.0; (e.confusion.rows as usize) * k];
            for (i, &row) in split.test.iter().enumerate() {
                let predicted = predictions[i]["code"].as_u64().unwrap() as usize;
                confusion[codes[row] as usize * k + predicted] += 1.0;
            }
            let got: Vec<f64> = e
                .confusion
                .values
                .iter()
                .map(|v| v.unwrap_or(0.0))
                .collect();
            assert_eq!(got, confusion, "{}: confusion of the split", case.name);
        }
        Evaluation::Regression(e) => {
            let core::Target::Values { values } = &dataset.target else {
                panic!("{}: regression evaluation of classes", case.name);
            };
            let want: Vec<f64> = split.test.iter().map(|&r| values[r]).collect();
            assert_eq!(e.y_true, want, "{}: y_true of the split", case.name);
            for (i, (&p, r)) in e.y_pred.iter().zip(predictions).enumerate() {
                let v = r["value"].as_f64().unwrap();
                assert!(
                    (p - v).abs() <= 1e-9 * v.abs().max(1.0),
                    "{}: y_pred[{i}] = {p} but predict_one = {v}",
                    case.name
                );
            }
        }
    }
}

fn rust_metrics(m: &TrainedModel) -> Value {
    match m.model.evaluate() {
        Ok(Evaluation::Classification(e)) => json!({ "accuracy": e.accuracy }),
        Ok(Evaluation::Regression(e)) => json!({ "r2": e.r2, "mse": e.mse, "mae": e.mae }),
        Err(e) => panic!("evaluate: {e:?}"),
    }
}

fn generate(case: &Case, root: &Path) -> String {
    let dir = root.join(case.name);
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir_all(&dir).unwrap();
    let m = train(case);
    let dataset = core_dataset(&case.source);
    let info = m.info();
    let classifier = info.model_classes.is_some();
    let model_classes = info.model_classes.clone().unwrap_or_default();

    // Inputs: every test row of the split (D7; 🧠: the stratified split of D72).
    let split = if case.is_nn() {
        let codes: Option<Vec<usize>> = match &dataset.target {
            core::Target::Classes { codes, .. } => {
                Some(codes.iter().map(|&c| c as usize).collect())
            }
            core::Target::Values { .. } => None,
        };
        let task = if codes.is_some() {
            core::TaskType::Classification
        } else {
            core::TaskType::Regression
        };
        let names = dataset.class_names().unwrap_or_default().to_vec();
        core::nn_data::lesson_split(task, codes.as_deref(), &names, dataset.n_rows()).unwrap()
    } else {
        core::split::train_test_split(dataset.n_rows(), case.test_size(), SEED).unwrap()
    };
    let inputs: Vec<Vec<f64>> = split
        .test
        .iter()
        .map(|&r| dataset.features.row(r).to_vec())
        .collect();
    let predictions: Vec<Value> = inputs
        .iter()
        .map(|row| prediction(&m, row, &model_classes, &split.train))
        .collect();

    let mut files = serde_json::Map::new();
    write(
        &dir,
        "modelo.onnx",
        &export_bytes(&m, ExportFormat::Onnx, None).unwrap(),
    );
    files.insert("onnx".into(), json!("modelo.onnx"));
    let package_format = if case.is_nn() {
        ExportFormat::KerasPackage
    } else {
        ExportFormat::PythonPackage
    };
    let package = export_bytes(&m, package_format, None).unwrap();
    write(&dir, "paquete.zip", &package);
    files.insert("package".into(), json!("paquete.zip"));
    check_split(case, &m, &dataset, &split, &package, &predictions);
    let mut metrics = rust_metrics(&m);
    if case.is_nn() {
        let numpy = export_bytes(&m, ExportFormat::NnNumpyPredictor, None).unwrap();
        write(&dir, "red_predictor_numpy.zip", &numpy);
        files.insert("numpy_predictor".into(), json!("red_predictor_numpy.zip"));
        let metadata = export_bytes(&m, ExportFormat::NnMetadata, Some(TRAINED_AT)).unwrap();
        write(&dir, "metadatos.json", &metadata);
        files.insert("metadata".into(), json!("metadatos.json"));
        let meta: Value = serde_json::from_slice(&metadata).unwrap();
        for key in ["test_loss", "test_accuracy"] {
            if let Some(v) = meta["training_info"].get(key) {
                metrics[key] = v.clone();
            }
        }
    }
    if case.lesson == LessonId::Trees {
        let rules = export_bytes(&m, ExportFormat::TreeRules, None).unwrap();
        write(&dir, "reglas.txt", &rules);
        files.insert("rules".into(), json!("reglas.txt"));
        // 📝 Texto (with the weights and the class names, D40).
        let view = m.model.tree_rules(RulesKind::View, true).unwrap();
        write(&dir, "reglas_pesos.txt", view.as_bytes());
        files.insert("rules_with_weights".into(), json!("reglas_pesos.txt"));
        let python = export_bytes(&m, ExportFormat::TreePython, None).unwrap();
        write(&dir, "arbol_decision.py", &python);
        files.insert("tree_python".into(), json!("arbol_decision.py"));
    }
    // A non-trivial pair for the decision boundary: columns 2 and 3 (or the last and the
    // first with fewer than 4 features).
    let n_features = info.feature_columns.len();
    let pair = if n_features >= 4 {
        (2u32, 3u32)
    } else {
        (crate::ipc::to_u32(n_features - 1), 0)
    };
    let mut code = serde_json::Map::new();
    for template in templates(case.lesson, classifier) {
        let extra = (template == CodeTemplate::DecisionBoundary).then_some(CodeExtra {
            feature_x: pair.0,
            feature_y: pair.1,
        });
        let text = code_of(&m, template, extra)
            .unwrap_or_else(|e| panic!("{}: {template:?}: {e:?}", case.name));
        let key = template_key(template);
        let file = format!("code/{key}.py");
        write(&dir, &file, text.as_bytes());
        code.insert(
            key,
            json!({ "file": file, "template": template.source().file_name() }),
        );
    }
    let csv_file = match &case.source {
        Source::Csv { file, bytes, .. } => {
            write(&dir, file, bytes);
            Some(*file)
        }
        Source::Builtin(_) => None,
    };
    let class_values = match &dataset.target {
        core::Target::Classes { class_values, .. } => Some(class_values.clone()),
        core::Target::Values { .. } => None,
    };
    let doc = json!({
        "name": case.name,
        "lesson": lesson_slug(case.lesson),
        "dataset": match &case.source {
            Source::Builtin(id) => id.slug(),
            Source::Csv { .. } => "csv",
        },
        "dataset_label": info.dataset_label,
        "csv_file": csv_file,
        "target_column": dataset.target_column,
        "config": serde_json::to_value(&case.config).unwrap(),
        "test_size": case.test_size(),
        "task": if classifier { "classification" } else { "regression" },
        "feature_columns": info.feature_columns,
        "feature_names": info.feature_names,
        "class_names": info.class_names,
        "class_values": class_values,
        "model_classes": info.model_classes,
        "n_train": info.n_train,
        "n_test": info.n_test,
        "templates_offline": case.templates_offline(),
        "boundary_pair": [pair.0, pair.1],
        "input_rows": split.test,
        // The real class code or value of each input (y_test).
        "targets": match &dataset.target {
            core::Target::Classes { codes, .. } => {
                json!(split.test.iter().map(|&r| codes[r]).collect::<Vec<_>>())
            }
            core::Target::Values { values } => {
                json!(split.test.iter().map(|&r| values[r]).collect::<Vec<_>>())
            }
        },
        "inputs": inputs,
        "rust": predictions,
        "rust_metrics": metrics,
        "files": files,
        "code": code,
        "package_templates": package_templates(case.lesson),
    });
    write(
        &dir,
        "case.json",
        serde_json::to_string_pretty(&doc).unwrap().as_bytes(),
    );
    case.name.to_owned()
}

fn output_dir() -> PathBuf {
    std::env::var_os("MLTUTOR_EXPORT_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/export-corpus"))
}

#[test]
#[ignore = "writes the export corpus for tools/verify_exports.py (MLTUTOR_EXPORT_CORPUS)"]
fn export_corpus() {
    let root = output_dir();
    fs::create_dir_all(&root).unwrap();
    let names: Vec<String> = cases().iter().map(|case| generate(case, &root)).collect();
    let index = json!({
        "generator": "src-tauri/src/export_corpus.rs",
        "mltutor_version": env!("CARGO_PKG_VERSION"),
        "cases": names,
    });
    write(
        &root,
        "index.json",
        serde_json::to_string_pretty(&index).unwrap().as_bytes(),
    );
    eprintln!("export corpus: {} cases in {}", names.len(), root.display());
}

/// The CSVs of the corpus are read back with their dialect and classes.
#[test]
fn the_corpus_csvs_are_read_with_their_dialect() {
    let read = |bytes: &[u8], target: &str| {
        core::csv::read_csv(bytes)
            .and_then(|t| t.to_dataset(target, core::TaskChoice::Classification, "f.csv"))
            .unwrap()
    };
    let bytes = medidas_csv();
    assert!(std::str::from_utf8(&bytes).is_err());
    let medidas = read(&bytes, "especie");
    assert_eq!(medidas.n_rows(), 150);
    assert_eq!(medidas.feature_columns[3], "tamaño");
    assert_eq!(medidas.class_names().unwrap()[2], "tulipán");
    assert!((medidas.features[[0, 0]] - 5.1).abs() < 1e-12);
    let tab = read(&tab_csv(), "especie");
    assert_eq!(tab.n_rows(), builtin_dataset(DatasetId::Penguins).n_rows());
    let raras = read(&rare_classes_csv(), "clase");
    assert_eq!(raras.class_names().unwrap().len(), 5);
    let split = core::split::train_test_split(150, TEST_SIZE, SEED).unwrap();
    let codes = class_codes(&raras);
    let train: std::collections::BTreeSet<u32> = split.train.iter().map(|&r| codes[r]).collect();
    let test: std::collections::BTreeSet<u32> = split.test.iter().map(|&r| codes[r]).collect();
    assert_eq!(
        (train.len(), test.len()),
        (4, 4),
        "one class missing from each side"
    );
    let vinos = read(&shifted_rare_classes_csv(), "cultivar");
    let core::Target::Classes {
        class_values,
        codes,
        ..
    } = &vinos.target
    else {
        panic!("classes");
    };
    assert_eq!(class_values, &[1.0, 2.0, 3.0, 4.0]);
    let split = core::split::train_test_split(vinos.n_rows(), TEST_SIZE, SEED).unwrap();
    assert!(
        split.train.iter().all(|&r| codes[r] != 0),
        "cultivar 1 only in the test set"
    );
}

/// Every template of the lessons maps to a file of `content/code/`.
#[test]
fn every_code_template_has_a_source_file() {
    use CodeTemplate as T;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../content/code");
    for t in [
        T::ConfusionMatrix,
        T::PrecisionByClass,
        T::PredVsReal,
        T::ErrorDistribution,
        T::DecisionBoundary,
        T::RegressionCode,
        T::TreeViz,
        T::TreeText,
        T::DecisionPath,
        T::TreeUsage,
        T::TreePython,
        T::TreePythonLiteral,
        T::TreeTraining,
        T::KnnTraining,
        T::NnTraining,
        T::NnNumpyPredictor,
    ] {
        assert!(dir.join(t.source().file_name()).is_file(), "{t:?}");
    }
}
