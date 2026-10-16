//! 📋 Metadatos (D75) and the K1 refusals of the network exports with real
//! trainings (Iris, Cáncer, Diabetes and a Diabetes run that diverges).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::nn_data::{Activation, NnArchitecture, NnTraining, OptimizerKind};
use mltutor_core::Dataset;
use mltutor_export::metadata::{metadata_json_text, nn_metadata};
use mltutor_export::package::keras::{keras_package, KerasPackageInput, NUMPY_SCRIPT};
use mltutor_export::ExportError;
use mltutor_nn::{train_network, FitControl, TrainOptions, TrainedNetwork};
use ndarray::Axis;
use serde_json::Value;

fn arch(output: Activation, activation: Activation, optimizer: OptimizerKind) -> NnArchitecture {
    NnArchitecture {
        hidden: vec![10, 8],
        activation,
        output_activation: output,
        dropout: 0.2,
        batch_size: 32,
        optimizer,
    }
}

fn training(epochs: u16, lr: f32, callbacks: bool) -> NnTraining {
    NnTraining {
        learning_rate: lr,
        epochs,
        validation_pct: 20,
        patience: callbacks.then_some(10),
        lr_factor: callbacks.then_some(0.5),
        seed: 7,
    }
}

fn train(ds: &Dataset, a: &NnArchitecture, t: &NnTraining) -> TrainedNetwork {
    train_network(
        ds.features.view(),
        &ds.target,
        a,
        t,
        TrainOptions::default(),
        &FitControl::default(),
    )
    .unwrap()
}

#[test]
fn metadata_is_complete_corrected_and_deterministic() {
    let iris = load_builtin(DatasetId::Iris).unwrap();
    let a = arch(Activation::Softmax, Activation::Relu, OptimizerKind::Adam);
    let t = training(15, 0.01, true);
    let trained = train(&iris, &a, &t);
    let m = nn_metadata(&trained, &iris, &a, &t, "2026-10-16 12:00").unwrap();
    let again = nn_metadata(&train(&iris, &a, &t), &iris, &a, &t, "2026-10-16 12:00").unwrap();
    assert_eq!(
        metadata_json_text(&m).unwrap(),
        metadata_json_text(&again).unwrap()
    );
    let mi = &m["model_info"];
    assert_eq!(mi["architecture"], serde_json::json!([4, 10, 8, 3]));
    assert_eq!(mi["total_parameters"], 4 * 10 + 10 + 10 * 8 + 8 + 8 * 3 + 3);
    assert_eq!(mi["dropout_rate"], 0.2);
    assert_eq!(mi["loss"], "categorical_crossentropy");
    assert_eq!(mi["valid"], true);
    let ti = &m["training_info"];
    assert_eq!(ti["epochs_trained"], trained.history.epochs.len());
    assert!(ti["final_accuracy"].is_number() && ti["final_val_accuracy"].is_number());
    assert!(ti["test_accuracy"].is_number() && ti["final_learning_rate"].is_number());
    // M7: with EarlyStopping the model has the weights of best_epoch, whose
    // metrics are written next to the final ones.
    let best = trained.history.best_epoch.unwrap();
    let e = trained.history.epochs[best];
    assert_eq!(ti["weights_epoch"], "best");
    assert_eq!(ti["best_epoch"], best + 1);
    assert_eq!(ti["best_val_loss"], e.val_loss);
    assert_eq!(ti["best_val_accuracy"], e.val_metric);
    assert_eq!(ti["best_loss"], e.loss);
    assert_eq!(ti["best_accuracy"], e.metric);
    assert_eq!(ti["near_divergence"], false);
    let di = &m["data_info"];
    assert_eq!(di["num_classes"], 3);
    assert_eq!(di["class_names"].as_array().unwrap().len(), 3);
    assert_eq!(di["num_features"], 4);
    assert_eq!(di["target_column"], iris.target_column.as_str());
    assert_eq!(m["config"]["seed"], 7);
    assert_eq!(m["config"]["learning_rate"], 0.01);
    assert_eq!(m["config"]["lr_factor"], 0.5);
    assert_eq!(m["export_info"]["training_date"], "2026-10-16 12:00");
    assert_eq!(m["export_info"]["version"], mltutor_export::VERSION);

    // Binary: 2 classes (Python wrote output_size = 1).
    let cancer = load_builtin(DatasetId::BreastCancer).unwrap();
    let a = arch(Activation::Sigmoid, Activation::Relu, OptimizerKind::Adam);
    let tb = training(5, 0.01, false);
    let m = nn_metadata(&train(&cancer, &a, &tb), &cancer, &a, &tb, "x").unwrap();
    assert_eq!(m["data_info"]["num_classes"], 2);
    assert_eq!(m["config"]["patience"], Value::Null);
    // Without EarlyStopping the model is the last epoch: no best_* metrics.
    assert_eq!(m["training_info"]["weights_epoch"], "last");
    assert_eq!(m["training_info"]["best_val_loss"], Value::Null);
    assert_eq!(m["training_info"]["best_accuracy"], Value::Null);

    // Regression: MAE, num_classes null (Python wrote 1), no class names; a dataset
    // of other width fails.
    let diabetes = load_builtin(DatasetId::Diabetes).unwrap();
    let a = arch(Activation::Linear, Activation::Relu, OptimizerKind::Adam);
    let tr = train(&diabetes, &a, &tb);
    let m = nn_metadata(&tr, &diabetes, &a, &tb, "x").unwrap();
    assert_eq!(m["data_info"]["num_classes"], Value::Null);
    assert_eq!(m["data_info"]["class_names"], Value::Null);
    assert!(m["training_info"]["final_mae"].is_number());
    assert!(matches!(
        nn_metadata(&tr, &iris, &a, &tb, "x"),
        Err(ExportError::InvalidModel { .. })
    ));
}

/// K1 with a real divergence (MEMORY F5c: Diabetes, tanh, SGD, LR 0.3, no
/// callbacks): the metadata still describe it (nulls, `valid: false`), ONNX and the
/// Keras package refuse it.
#[test]
fn diverged_training_is_described_but_not_exported() {
    let diabetes = load_builtin(DatasetId::Diabetes).unwrap();
    let mut a = arch(Activation::Linear, Activation::Tanh, OptimizerKind::Sgd);
    a.hidden = vec![10, 10];
    a.dropout = 0.0;
    a.batch_size = 64;
    let t = NnTraining {
        learning_rate: 0.3,
        epochs: 15,
        validation_pct: 20,
        patience: None,
        lr_factor: None,
        seed: 42,
    };
    let trained = train(&diabetes, &a, &t);
    assert!(trained.history.diverged_epoch.is_some(), "did not diverge");
    let m = nn_metadata(&trained, &diabetes, &a, &t, "x").unwrap();
    assert!(!trained.network.is_valid());
    assert_eq!(m["model_info"]["valid"], false);
    assert!(m["training_info"]["diverged_epoch"].is_number());
    assert_eq!(m["training_info"]["final_loss"], Value::Null);
    metadata_json_text(&m).unwrap();
    assert!(matches!(
        mltutor_export::onnx_model(&trained.network, None),
        Err(ExportError::InvalidModel { .. })
    ));
    let x_test = trained.x_test(diabetes.features.view());
    let names = diabetes.feature_display.clone();
    let pkg = keras_package(&KerasPackageInput {
        network: &trained.network,
        arch: &a,
        dataset_label: &diabetes.label,
        feature_names: &names,
        class_names: None,
        final_learning_rate: trained.final_learning_rate,
        x_check: x_test.slice(ndarray::s![..5, ..]),
        n_train: trained.split.train.len(),
        n_test: trained.split.test.len(),
    });
    assert!(pkg.is_err(), "the Keras package of a diverged network");
}

/// Finite but huge weights: the network is "valid", yet its check outputs overflow
/// to ∞, and the Keras package refuses to write them. Infinite weights are refused
/// by both exports.
#[test]
fn overflowing_and_infinite_weights_are_refused() {
    let iris = load_builtin(DatasetId::Iris).unwrap();
    let a = arch(Activation::Linear, Activation::Relu, OptimizerKind::Adam);
    let t = training(2, 0.01, false);
    let mut values = iris.clone();
    values.target = mltutor_core::Target::Values {
        values: iris.features.column(0).to_vec(),
    };
    let trained = train(&values, &a, &t);
    let rebuild = |f: &dyn Fn(&mut mltutor_nn::Mlp)| {
        let mut mlp = trained.network.mlp().clone();
        f(&mut mlp);
        mltutor_nn::NeuralNetwork::new(
            mlp,
            trained.network.scaler().clone(),
            mltutor_core::TaskType::Regression,
            vec![],
            trained.network.resolved(),
        )
        .unwrap()
    };
    let x = values.features.select(Axis(0), &trained.split.test[..5]);
    let names: Vec<String> = values.feature_display.clone();
    let input = |net| KerasPackageInput {
        network: net,
        arch: &a,
        dataset_label: "🌸",
        feature_names: &names,
        class_names: None,
        final_learning_rate: 0.01,
        x_check: x.view(),
        n_train: 120,
        n_test: 30,
    };
    let huge = rebuild(&|m| {
        for l in &mut m.layers {
            l.kernel.iter_mut().for_each(|w| *w = 3e38);
            l.bias.iter_mut().for_each(|b| *b = 3e38);
        }
    });
    assert!(huge.is_valid());
    assert!(matches!(
        keras_package(&input(&huge)),
        Err(ExportError::InvalidModel { .. }) | Err(ExportError::Model(_))
    ));
    let inf = rebuild(&|m| m.layers[1].bias[0] = f32::INFINITY);
    assert!(!inf.is_valid());
    assert!(keras_package(&input(&inf)).is_err());
    assert!(matches!(
        mltutor_export::onnx_model(&inf, None),
        Err(ExportError::InvalidModel { .. })
    ));
    // 📝 Código Python: the NumPy predictor travels with its modelo.json.
    let pkg = keras_package(&input(&trained.network)).unwrap();
    let zip = pkg.numpy_zip().unwrap();
    assert_eq!(zip, pkg.numpy_zip().unwrap());
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip)).unwrap();
    let names: Vec<String> = archive
        .file_names()
        .map(|n| n.unwrap().into_owned())
        .collect();
    assert_eq!(
        names,
        vec![NUMPY_SCRIPT.to_owned(), "modelo.json".to_owned()]
    );
    let mut text = String::new();
    std::io::Read::read_to_string(&mut archive.by_name("modelo.json").unwrap(), &mut text).unwrap();
    assert_eq!(text, pkg.model_json);
}
