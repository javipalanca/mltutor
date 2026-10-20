//! Phase 6: the corpus of everything the app exports, for `tools/verify_exports.py`.
//!
//! An `#[ignore]` test (it writes files): for every model family and a few
//! representative datasets it trains through [`RealFacade`] exactly like the
//! `train_model` command, then asks the [`ModelView`] for each export (`export`) and
//! each code template (`code`) the lessons offer, and writes them next to the inputs
//! and the predictions Rust computes for them (`predict_one`, the 🔮 form path):
//!
//! ```sh
//! MLTUTOR_EXPORT_CORPUS=/tmp/corpus cargo test -p mltutor export_corpus -- --ignored
//! uv run … python tools/verify_exports.py /tmp/corpus
//! ```
//!
//! Without `MLTUTOR_EXPORT_CORPUS` it writes to `target/export-corpus`. The output is
//! deterministic (fixed seeds, fixed metadata date, `serde_json` in insertion order):
//! two runs give the same bytes.
//!
//! Layout: `index.json` (the case names) and one folder per case with `case.json`
//! (configuration, names, class values, the inputs in original units and Rust's
//! predictions), `modelo.onnx`, the ZIP package (`paquete.zip`), the rendered
//! templates in `code/<file>.py` and, for CSV cases, the CSV the scripts read.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use mltutor_core as core;
use ndarray::Axis;
use serde_json::{json, Value};

use crate::core_facade::{CoreFacade, DatasetHandle, ModelHandle, TrainSpec};
use crate::ipc::{
    CodeTemplate, DatasetId, ExportFormat, KnnMetric, KnnWeights, ModelConfig, NnActivation,
    NnArchitecture, NnOptimizer, NnTraining, PredictionResult, TaskChoice, TreeCriterion, TreeTask,
};
use crate::real_facade::RealFacade;

/// `random_state=42` of the split (D7), as in `real_models`.
const SEED: u32 = 42;
/// At most this many test rows per case go to `inputs` (California has 6192).
const MAX_INPUTS: usize = 1000;
/// Fixed date of the 🧠 metadata (the UI sends the local time).
const TRAINED_AT: &str = "2026-01-01 12:00";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Lesson {
    Regression,
    Trees,
    Knn,
    Nn,
}

impl Lesson {
    fn slug(self) -> &'static str {
        match self {
            Lesson::Regression => "regresion",
            Lesson::Trees => "arboles",
            Lesson::Knn => "knn",
            Lesson::Nn => "redes",
        }
    }
}

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
    lesson: Lesson,
    source: Source,
    config: ModelConfig,
    test_size: f64,
    /// Whether the dataset can be loaded offline by the scripts (California needs a
    /// download in scikit-learn; the verifier skips its templates).
    templates_offline: bool,
}

/// A semicolon-separated, decimal-comma, Windows-1252 CSV with text classes and
/// accented column names (Iris measurements): it exercises `csv_sep`,
/// `csv_decimal` and `csv_encoding` of every training script.
fn iris_cp1252_csv() -> Vec<u8> {
    let iris =
        core::dataset::load_builtin(core::DatasetId::Iris).unwrap_or_else(|e| panic!("iris: {e}"));
    let names = ["lirio", "rosa", "tulipán"];
    let core::Target::Classes { codes, .. } = &iris.target else {
        panic!("iris is a classification dataset");
    };
    let mut text = String::from("longitud_sépalo;anchura_sépalo;longitud_pétalo;tamaño;especie\n");
    for (row, &code) in iris.features.axis_iter(Axis(0)).zip(codes) {
        for v in row {
            text.push_str(&format!("{v}").replace('.', ","));
            text.push(';');
        }
        text.push_str(names[code as usize]);
        text.push('\n');
    }
    // Every character is in Latin-1, where Windows-1252 and Unicode coincide.
    text.chars()
        .map(|c| u8::try_from(u32::from(c)).unwrap_or_else(|_| panic!("{c} is not Latin-1")))
        .collect()
}

/// A comma-separated UTF-8 CSV of a built-in dataset whose integer classes are NOT the
/// codes (`class + offset`, times `scale`): the ONNX labels are then the class values
/// (`onnx_class_labels`), and the scripts must map them (binary: the positive class is
/// the larger value; 🧠: `LabelEncoder`).
fn numeric_classes_csv(id: core::DatasetId, target: &str, scale: u32, offset: u32) -> Vec<u8> {
    let data = core::dataset::load_builtin(id).unwrap_or_else(|e| panic!("{id:?}: {e}"));
    let core::Target::Classes { codes, .. } = &data.target else {
        panic!("{id:?} is not a classification dataset");
    };
    let mut text = data.feature_columns.join(",");
    text.push_str(&format!(",{target}\n"));
    for (row, &code) in data.features.axis_iter(Axis(0)).zip(codes) {
        for v in row {
            text.push_str(&format!("{v},"));
        }
        text.push_str(&format!("{}\n", code * scale + offset));
    }
    text.into_bytes()
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

fn cases() -> Vec<Case> {
    use DatasetId as D;
    use Lesson as L;
    let builtin = |name, lesson, id, config, test_size| Case {
        name,
        lesson,
        source: Source::Builtin(id),
        config,
        test_size,
        templates_offline: id != D::CaliforniaHousing,
    };
    let csv = |name, lesson, file, target, choice, bytes, config, test_size| Case {
        name,
        lesson,
        source: Source::Csv {
            file,
            target,
            choice,
            bytes,
        },
        config,
        test_size,
        templates_offline: true,
    };
    let medidas = iris_cp1252_csv;
    let numeric = || parity_csv("numeric_regression");
    // Wine with cultivars 1, 2, 3 and Breast Cancer with 2 = malignant and 4 = benign
    // (scikit-learn's codes 0/1, times 2 plus 2).
    let vinos = || numeric_classes_csv(core::DatasetId::Wine, "cultivar", 1, 1);
    let cancer = || numeric_classes_csv(core::DatasetId::BreastCancer, "diagnostico", 2, 2);
    let (cls, reg) = (TaskChoice::Classification, TaskChoice::Regression);
    vec![
        // 📊 Regresión (lineal y logística)
        builtin(
            "lr_diabetes_linear",
            L::Regression,
            D::Diabetes,
            ModelConfig::Linear,
            0.3,
        ),
        builtin(
            "lr_california_linear",
            L::Regression,
            D::CaliforniaHousing,
            ModelConfig::Linear,
            0.3,
        ),
        builtin("lr_iris_logistic", L::Regression, D::Iris, logistic(), 0.3),
        builtin(
            "lr_breast_cancer_logistic",
            L::Regression,
            D::BreastCancer,
            logistic(),
            0.3,
        ),
        builtin(
            "lr_titanic_logistic",
            L::Regression,
            D::Titanic,
            logistic(),
            0.2,
        ),
        csv(
            "lr_csv_medidas_logistic",
            L::Regression,
            "medidas_ñ.csv",
            "especie",
            cls,
            medidas(),
            logistic(),
            0.3,
        ),
        csv(
            "lr_csv_vinos_logistic",
            L::Regression,
            "vinos.csv",
            "cultivar",
            cls,
            vinos(),
            logistic(),
            0.3,
        ),
        csv(
            "lr_csv_cancer_2_4_logistic",
            L::Regression,
            "cancer.csv",
            "diagnostico",
            cls,
            cancer(),
            logistic(),
            0.3,
        ),
        csv(
            "lr_csv_numeric_linear",
            L::Regression,
            "numeric_regression.csv",
            "y",
            reg,
            numeric(),
            ModelConfig::Linear,
            0.3,
        ),
        // 🌲 Árboles
        builtin(
            "tree_iris_gini_d3",
            L::Trees,
            D::Iris,
            tree(TreeTask::Classification, TreeCriterion::Gini, 3, 2),
            0.3,
        ),
        builtin(
            "tree_breast_cancer_entropy_d10",
            L::Trees,
            D::BreastCancer,
            tree(TreeTask::Classification, TreeCriterion::Entropy, 10, 2),
            0.3,
        ),
        builtin(
            "tree_titanic_gini_d4",
            L::Trees,
            D::Titanic,
            tree(TreeTask::Classification, TreeCriterion::Gini, 4, 2),
            0.3,
        ),
        builtin(
            "tree_diabetes_squared_error_d3",
            L::Trees,
            D::Diabetes,
            tree(TreeTask::Regression, TreeCriterion::SquaredError, 3, 2),
            0.3,
        ),
        builtin(
            "tree_california_squared_error_d8",
            L::Trees,
            D::CaliforniaHousing,
            tree(TreeTask::Regression, TreeCriterion::SquaredError, 8, 10),
            0.3,
        ),
        csv(
            "tree_csv_vinos_gini_d3",
            L::Trees,
            "vinos.csv",
            "cultivar",
            cls,
            vinos(),
            tree(TreeTask::Classification, TreeCriterion::Gini, 3, 2),
            0.3,
        ),
        csv(
            "tree_csv_medidas_entropy_d4",
            L::Trees,
            "medidas_ñ.csv",
            "especie",
            cls,
            medidas(),
            tree(TreeTask::Classification, TreeCriterion::Entropy, 4, 2),
            0.3,
        ),
        // 🔍 KNN
        builtin(
            "knn_iris_k5_uniform_minkowski",
            L::Knn,
            D::Iris,
            knn(5, KnnWeights::Uniform, KnnMetric::Minkowski),
            0.3,
        ),
        builtin(
            "knn_wine_k7_distance_manhattan",
            L::Knn,
            D::Wine,
            knn(7, KnnWeights::Distance, KnnMetric::Manhattan),
            0.3,
        ),
        builtin(
            "knn_breast_cancer_k10_distance_euclidean",
            L::Knn,
            D::BreastCancer,
            knn(10, KnnWeights::Distance, KnnMetric::Euclidean),
            0.3,
        ),
        builtin(
            "knn_titanic_k5_uniform_minkowski",
            L::Knn,
            D::Titanic,
            knn(5, KnnWeights::Uniform, KnnMetric::Minkowski),
            0.3,
        ),
        builtin(
            "knn_diabetes_k10_distance_manhattan",
            L::Knn,
            D::Diabetes,
            knn(10, KnnWeights::Distance, KnnMetric::Manhattan),
            0.3,
        ),
        csv(
            "knn_csv_medidas_k15_distance_euclidean",
            L::Knn,
            "medidas_ñ.csv",
            "especie",
            cls,
            medidas(),
            knn(15, KnnWeights::Distance, KnnMetric::Euclidean),
            0.3,
        ),
        csv(
            "knn_csv_cancer_2_4_k5_uniform",
            L::Knn,
            "cancer.csv",
            "diagnostico",
            cls,
            cancer(),
            knn(5, KnnWeights::Uniform, KnnMetric::Minkowski),
            0.3,
        ),
        csv(
            "knn_csv_numeric_k5_uniform",
            L::Knn,
            "numeric_regression.csv",
            "y",
            reg,
            numeric(),
            knn(5, KnnWeights::Uniform, KnnMetric::Minkowski),
            0.3,
        ),
        // 🧠 Redes neuronales
        builtin(
            "nn_iris_softmax",
            L::Nn,
            D::Iris,
            nn(vec![10, 10], NnActivation::Softmax, NnOptimizer::Adam, 0.01),
            0.2,
        ),
        builtin(
            "nn_breast_cancer_sigmoid",
            L::Nn,
            D::BreastCancer,
            nn(
                vec![16, 8],
                NnActivation::Sigmoid,
                NnOptimizer::Rmsprop,
                0.001,
            ),
            0.2,
        ),
        builtin(
            "nn_diabetes_linear",
            L::Nn,
            D::Diabetes,
            nn(vec![32, 16], NnActivation::Linear, NnOptimizer::Adam, 0.01),
            0.2,
        ),
        csv(
            "nn_csv_cancer_2_4_sigmoid",
            L::Nn,
            "cancer.csv",
            "diagnostico",
            cls,
            cancer(),
            nn(vec![16], NnActivation::Sigmoid, NnOptimizer::Adam, 0.01),
            0.2,
        ),
        csv(
            "nn_csv_medidas_softmax",
            L::Nn,
            "medidas_ñ.csv",
            "especie",
            cls,
            medidas(),
            nn(vec![8], NnActivation::Softmax, NnOptimizer::Sgd, 0.1),
            0.2,
        ),
    ]
}

/// The dataset through the facade (what the app trains) and the core dataset with
/// the same rows (for the inputs: the facade only exposes views).
fn load(source: &Source) -> (DatasetHandle, core::Dataset) {
    let cancel = AtomicBool::new(false);
    match source {
        Source::Builtin(id) => {
            let handle = RealFacade.load_builtin(*id).unwrap();
            let core_id = core::DatasetId::from_slug(id.slug()).unwrap();
            (handle, core::dataset::load_builtin(core_id).unwrap())
        }
        Source::Csv {
            file,
            target,
            choice,
            bytes,
        } => {
            let draft = RealFacade.read_csv(file, bytes, &cancel).unwrap();
            let done = draft.finalize(target, *choice, &cancel).unwrap();
            let core_choice = match choice {
                TaskChoice::Auto => core::TaskChoice::Auto,
                TaskChoice::Classification => core::TaskChoice::Classification,
                TaskChoice::Regression => core::TaskChoice::Regression,
            };
            let dataset = core::csv::read_csv(bytes)
                .and_then(|t| t.to_dataset(target, core_choice, file))
                .unwrap();
            (done.dataset, dataset)
        }
    }
}

/// The templates each lesson offers for this model (the `ModelCode` blocks of the UI).
fn templates(lesson: Lesson, classifier: bool) -> Vec<CodeTemplate> {
    use CodeTemplate as T;
    let mut out = match lesson {
        Lesson::Regression => vec![T::RegressionCode],
        Lesson::Trees => vec![
            T::TreeTraining,
            T::TreeViz,
            T::TreeText,
            T::DecisionPath,
            T::TreeUsage,
            T::TreePython,
        ],
        Lesson::Knn => vec![T::KnnTraining],
        Lesson::Nn => vec![T::NnTraining, T::NnNumpyPredictor],
    };
    if classifier {
        out.extend([T::ConfusionMatrix, T::PrecisionByClass]);
        // The 🧠 surface hides the code (it fixes the other features at their means).
        if lesson != Lesson::Nn {
            out.push(T::DecisionBoundary);
        }
    } else {
        out.extend([T::PredVsReal, T::ErrorDistribution]);
    }
    out
}

fn template_key(template: CodeTemplate) -> String {
    serde_json::to_value(template)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Rust's prediction of one row, from the 🔮 form path.
fn prediction(
    model: &ModelHandle,
    row: &[f64],
    model_classes: &[usize],
    names: &[String],
) -> Value {
    match model.predict_one(row).unwrap() {
        PredictionResult::Class {
            index,
            name,
            proba,
            neighbors,
            ..
        } => {
            let proba = proba.map(|p| {
                model_classes
                    .iter()
                    .map(|&code| {
                        p.iter()
                            .find(|c| names.get(code) == Some(&c.name))
                            .map_or(f64::NAN, |c| c.probability)
                    })
                    .collect::<Vec<_>>()
            });
            json!({ "code": index, "name": name, "proba": proba, "knn": knn_info(neighbors) })
        }
        PredictionResult::Value {
            value, neighbors, ..
        } => json!({ "value": value, "knn": knn_info(neighbors) }),
    }
}

fn knn_info(neighbors: Option<crate::ipc::KnnNeighbors>) -> Value {
    match neighbors {
        None => Value::Null,
        Some(n) => json!({
            "train_rows": n.rows.iter().map(|r| r.train_row).collect::<Vec<_>>(),
            "distances": n.rows.iter().map(|r| r.distance).collect::<Vec<_>>(),
            "tie_at_kth": n.tie_at_kth.is_some(),
        }),
    }
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

fn generate(case: &Case, root: &Path) -> String {
    let dir = root.join(case.name);
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir_all(&dir).unwrap();
    let (handle, dataset) = load(&case.source);
    let summary = handle.summary();
    let spec = TrainSpec {
        config: case.config.clone(),
        test_size: case.test_size,
        dataset_label: summary.label.clone(),
    };
    let model = handle
        .train_observed(&spec, &|_, _| {}, &|_| {}, &AtomicBool::new(false))
        .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
    let ms = model.summary();
    let classifier = ms.model_classes.is_some();
    let names = ms.class_names.clone().unwrap_or_default();
    let model_classes = ms.model_classes.clone().unwrap_or_default();

    // Inputs: the test rows of the D7 split (classic models), or the first rows of the
    // dataset (🧠: the network uses its own stratified split, D72).
    let rows: Vec<usize> = if case.lesson == Lesson::Nn {
        (0..dataset.n_rows().min(300)).collect()
    } else {
        let split = core::split::train_test_split(dataset.n_rows(), case.test_size, SEED).unwrap();
        assert_eq!(split.test.len(), ms.n_test, "{}: split", case.name);
        split.test.into_iter().take(MAX_INPUTS).collect()
    };
    let inputs: Vec<Vec<f64>> = rows
        .iter()
        .map(|&r| dataset.features.row(r).to_vec())
        .collect();
    let predictions: Vec<Value> = inputs
        .iter()
        .map(|row| prediction(&model, row, &model_classes, &names))
        .collect();

    let mut files = serde_json::Map::new();
    let onnx = model.export(ExportFormat::Onnx).unwrap();
    write(&dir, "modelo.onnx", &onnx);
    files.insert("onnx".into(), json!("modelo.onnx"));
    let package = if case.lesson == Lesson::Nn {
        ExportFormat::KerasPackage
    } else {
        ExportFormat::PythonPackage
    };
    write(&dir, "paquete.zip", &model.export(package).unwrap());
    files.insert("package".into(), json!("paquete.zip"));
    if case.lesson == Lesson::Nn {
        let numpy = model.export(ExportFormat::NnNumpyPredictor).unwrap();
        write(&dir, "red_predictor_numpy.zip", &numpy);
        files.insert("numpy_predictor".into(), json!("red_predictor_numpy.zip"));
        let metadata = model.nn_metadata(TRAINED_AT).unwrap();
        write(&dir, "metadatos.json", metadata.as_bytes());
        files.insert("metadata".into(), json!("metadatos.json"));
    }
    if case.lesson == Lesson::Trees {
        write(
            &dir,
            "reglas.txt",
            &model.export(ExportFormat::TreeRules).unwrap(),
        );
        files.insert("rules".into(), json!("reglas.txt"));
        write(
            &dir,
            "arbol_decision.py",
            &model.export(ExportFormat::TreePython).unwrap(),
        );
        files.insert("tree_python".into(), json!("arbol_decision.py"));
    }
    let mut code = serde_json::Map::new();
    for template in templates(case.lesson, classifier) {
        let text = model
            .code(template, None)
            .unwrap_or_else(|e| panic!("{}: {template:?}: {e:?}", case.name));
        let file = format!("code/{}.py", template_key(template));
        write(&dir, &file, text.as_bytes());
        code.insert(template_key(template), json!(file));
    }
    let csv_file = match &case.source {
        Source::Csv { file, bytes, .. } => {
            write(&dir, file, bytes);
            Some(*file)
        }
        Source::Builtin(_) => None,
    };
    let (class_values, target_column) = match &dataset.target {
        core::Target::Classes { class_values, .. } => {
            (Some(class_values.clone()), &dataset.target_column)
        }
        core::Target::Values { .. } => (None, &dataset.target_column),
    };
    let doc = json!({
        "name": case.name,
        "lesson": case.lesson.slug(),
        "dataset": match &case.source {
            Source::Builtin(id) => id.slug(),
            Source::Csv { .. } => "csv",
        },
        "dataset_label": summary.label,
        "csv_file": csv_file,
        "target_column": target_column,
        "config": serde_json::to_value(&case.config).unwrap(),
        "test_size": case.test_size,
        "task": if classifier { "classification" } else { "regression" },
        "feature_columns": ms.feature_columns,
        "feature_names": ms.feature_names,
        "class_names": ms.class_names,
        "class_values": class_values,
        "model_classes": ms.model_classes,
        "n_train": ms.n_train,
        "n_test": ms.n_test,
        "templates_offline": case.templates_offline,
        "input_rows": rows,
        "inputs": inputs,
        "rust": predictions,
        "files": files,
        "code": code,
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

/// The CSV of the corpus is read back as Windows-1252 with `;` and decimal comma.
#[test]
fn the_cp1252_csv_is_read_with_its_dialect() {
    let bytes = iris_cp1252_csv();
    assert!(std::str::from_utf8(&bytes).is_err());
    let dataset = core::csv::read_csv(&bytes)
        .and_then(|t| t.to_dataset("especie", core::TaskChoice::Classification, "medidas_ñ.csv"))
        .unwrap();
    assert_eq!(dataset.n_rows(), 150);
    assert_eq!(dataset.feature_columns[3], "tamaño");
    assert_eq!(dataset.class_names().unwrap()[2], "tulipán");
    assert!((dataset.features[[0, 0]] - 5.1).abs() < 1e-12);
}
