//! 🔍 K-Nearest Neighbors of [`crate::real_facade::RealFacade`] (fase4-knn.md §5 and §8):
//! conversions of `mltutor_core::knn` to the IPC DTOs. Every computation is a core
//! function; here only the orchestration lives.

use std::sync::atomic::AtomicBool;

use mltutor_core as core;
use mltutor_core::knn::Knn;
use mltutor_core::model::{ModelError, Predictor, TrainControl};
use ndarray::{ArrayView2, Axis};

use crate::ipc::{
    self, to_u32, ClassDistancesView, CommandError, KnnMetric, KnnNeighbors, KnnWeights,
    ModelConfig, NeighborRow, PlaygroundRequest, PlaygroundView, SearchEngine, TaskType, TieAtKth,
};
use crate::knn::KnnTrainingContext;
use crate::real_models::{model_error, CoreModel};

fn weights(w: KnnWeights) -> core::knn::KnnWeights {
    match w {
        KnnWeights::Uniform => core::knn::KnnWeights::Uniform,
        KnnWeights::Distance => core::knn::KnnWeights::Distance,
    }
}

fn metric(m: KnnMetric) -> core::knn::KnnMetric {
    match m {
        KnnMetric::Minkowski => core::knn::KnnMetric::Minkowski,
        KnnMetric::Euclidean => core::knn::KnnMetric::Euclidean,
        KnnMetric::Manhattan => core::knn::KnnMetric::Manhattan,
    }
}

fn ipc_weights(w: core::knn::KnnWeights) -> KnnWeights {
    match w {
        core::knn::KnnWeights::Uniform => KnnWeights::Uniform,
        core::knn::KnnWeights::Distance => KnnWeights::Distance,
    }
}

fn ipc_metric(m: core::knn::KnnMetric) -> KnnMetric {
    match m {
        core::knn::KnnMetric::Minkowski => KnnMetric::Minkowski,
        core::knn::KnnMetric::Euclidean => KnnMetric::Euclidean,
        core::knn::KnnMetric::Manhattan => KnnMetric::Manhattan,
    }
}

fn engine(e: core::knn::SearchEngine) -> SearchEngine {
    match e {
        core::knn::SearchEngine::KdTree => SearchEngine::KdTree,
        core::knn::SearchEngine::Brute => SearchEngine::Brute,
    }
}

fn config(k: u8, w: KnnWeights, m: KnnMetric) -> core::knn::KnnConfig {
    core::knn::KnnConfig {
        n_neighbors: u32::from(k),
        weights: weights(w),
        metric: metric(m),
    }
}

/// `train_model` with `ModelConfig::Knn` (`algorithm='auto'`, `leaf_size=30`).
pub fn fit(
    model_config: ModelConfig,
    x_train: ArrayView2<'_, f64>,
    y_train: &core::Target,
    control: &TrainControl<'_>,
) -> Result<Box<dyn Predictor>, ModelError> {
    let Some((k, w, m)) = crate::knn::params(model_config) else {
        return Err(ModelError::InvalidConfig {
            reason: "no es un KNN".to_owned(),
        });
    };
    core::knn::fit_boxed(config(k, w, m), x_train, y_train, control)
}

pub fn summary(knn: &Knn) -> ipc::KnnSummary {
    let s = knn.summary();
    ipc::KnnSummary {
        n_neighbors: s.n_neighbors,
        weights: ipc_weights(s.weights),
        metric: ipc_metric(s.metric),
        engine: engine(s.engine),
        n_train: to_u32(s.n_train),
    }
}

fn require_knn(model: &CoreModel) -> Result<&Knn, CommandError> {
    model.knn().ok_or_else(CommandError::not_knn)
}

fn tie(t: Option<core::knn::KthTie>) -> Option<TieAtKth> {
    t.map(|t| TieAtKth {
        n_tied: to_u32(t.n_tied),
        slots: to_u32(t.slots),
        distance: t.distance,
    })
}

/// C20 (D58): the K neighbours of the sample of the form, in the order of the search.
pub fn neighbors(model: &CoreModel, values: &[f64]) -> Result<KnnNeighbors, CommandError> {
    let knn = require_knn(model)?;
    let e = knn.explain(values).map_err(model_error)?;
    let rows = e
        .neighbors
        .iter()
        .map(|n| {
            let dataset_row = model.split.train.get(n.index).copied().unwrap_or(n.index);
            NeighborRow {
                train_row: to_u32(n.index),
                dataset_row: to_u32(dataset_row),
                distance: n.distance,
                weight: n.weight,
                class_index: n.class.map(to_u32),
                value: n.value,
                features: model.dataset.features.row(dataset_row).to_vec(),
            }
        })
        .collect();
    Ok(KnnNeighbors {
        rows,
        tie_at_kth: tie(e.tie_at_kth),
    })
}

/// C19: the distances between the classes of the train set of the model (D56).
pub fn class_distances(model: &CoreModel) -> Result<ClassDistancesView, CommandError> {
    let knn = require_knn(model)?;
    let codes = knn.train_codes().ok_or_else(CommandError::not_classifier)?;
    let m = knn.config().metric;
    let d = core::knn::class_distances(knn.train_x(), &codes, m, &TrainControl::none())
        .map_err(model_error)?;
    Ok(ClassDistancesView {
        class_names: d
            .classes
            .iter()
            .map(|&c| model.display_name_of(c))
            .collect(),
        classes: d.classes.iter().map(|&c| to_u32(c)).collect(),
        matrix: d.matrix,
        within: d.within,
        min: d.min,
        max: d.max,
        ratio: d.ratio,
        verdict: match d.verdict {
            core::knn::SeparationVerdict::Separated => ipc::SeparationVerdict::Separated,
            core::knn::SeparationVerdict::Moderate => ipc::SeparationVerdict::Moderate,
            core::knn::SeparationVerdict::Close => ipc::SeparationVerdict::Close,
            core::knn::SeparationVerdict::Binary => ipc::SeparationVerdict::Binary,
            core::knn::SeparationVerdict::TooFewClasses => ipc::SeparationVerdict::TooFewClasses,
        },
        metric: ipc_metric(d.metric),
        n_samples: to_u32(d.n_samples),
    })
}

/// C18: the 2D KNN of the canvas over every row of the dataset (D63).
pub fn playground(
    model: &CoreModel,
    req: &PlaygroundRequest,
    cancel: &AtomicBool,
) -> Result<PlaygroundView, CommandError> {
    require_knn(model)?;
    let dataset = &model.dataset;
    let n = dataset.n_rows();
    // A canvas K larger than the dataset would be D59: use every sample instead.
    let k = u8::try_from(usize::from(req.n_neighbors).min(n)).unwrap_or(req.n_neighbors);
    let (fx, fy) = (req.feature_x as usize, req.feature_y as usize);
    let request = core::knn::PlaygroundRequest {
        features: [fx, fy],
        config: config(k, req.weights, req.metric),
        points: req.points.clone(),
    };
    let control = TrainControl::none().with_cancel(cancel);
    let p = core::knn::playground(dataset.features.view(), &dataset.target, &request, &control)
        .map_err(model_error)?;
    let n_codes = model
        .class_names()
        .map_or(0, |names| names.len())
        .max(p.classes.last().map_or(0, |c| c + 1));
    let points = p
        .points
        .into_iter()
        .map(|point| ipc::PlaygroundPoint {
            point: point.point,
            class_index: to_u32(point.class),
            confidence: point.confidence,
            proba: point.proba,
            mean_distance: point.mean_distance,
            neighbors: point
                .neighbors
                .into_iter()
                .map(|nb| NeighborRow {
                    train_row: to_u32(nb.index),
                    dataset_row: to_u32(nb.index),
                    distance: nb.distance,
                    weight: nb.weight,
                    class_index: nb.class.map(to_u32),
                    value: nb.value,
                    features: Vec::new(),
                })
                .collect(),
        })
        .collect();
    let (rows, _) = p.data.dim();
    Ok(PlaygroundView {
        data: req
            .include_data
            .then(|| ipc::Matrix::from_f64(rows, 2, p.data.iter().copied().collect::<Vec<_>>())),
        codes: req
            .include_data
            .then(|| p.codes.iter().map(|&c| to_u32(c)).collect()),
        classes: p.classes.iter().map(|&c| to_u32(c)).collect(),
        class_names: (0..n_codes).map(|c| model.display_name_of(c)).collect(),
        colors: (0..n_codes).map(crate::knn::class_color).collect(),
        bounds: p.bounds,
        feature_names: [
            dataset.feature_display[fx].clone(),
            dataset.feature_display[fy].clone(),
        ],
        points,
        mean_confidence: p.mean_confidence,
        engine: engine(p.engine),
        n_neighbors: u32::from(k),
    })
}

/// D63: the rows the 2D views of KNN use (every row of the dataset).
pub fn all_rows(model: &CoreModel) -> (ndarray::Array2<f64>, core::Target) {
    let rows: Vec<usize> = (0..model.dataset.n_rows()).collect();
    (
        model.dataset.features.select(Axis(0), &rows),
        model.dataset.target.select(&rows),
    )
}

/// 💾 "📄 Código de Entrenamiento" (`knn_entrenamiento.py.j2`).
pub fn code(model: &CoreModel) -> Result<String, CommandError> {
    let (k, weights, metric) =
        crate::knn::params(model.spec.config).ok_or_else(CommandError::not_knn)?;
    let base = model.training_context();
    KnnTrainingContext {
        dataset: base.dataset,
        dataset_label: base.dataset_label,
        csv_file: base.csv_file,
        csv_target: base.csv_target,
        csv_sep: base.csv_sep,
        csv_decimal: base.csv_decimal,
        csv_encoding: base.csv_encoding,
        task: match crate::real_facade::task(model.model.task()) {
            TaskType::Classification => "classification",
            TaskType::Regression => "regression",
        }
        .to_owned(),
        n_neighbors: u32::from(k),
        weights: weights.sklearn_name().to_owned(),
        metric: metric.sklearn_name().to_owned(),
        test_size: base.test_size,
        feature_names: base.feature_names,
        class_names: base.class_names,
        sample: None,
    }
    .render()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use super::*;
    use crate::core_facade::{CoreFacade, ModelHandle, TrainSpec};
    use crate::fixture_facade::FixtureFacade;
    use crate::ipc::{DatasetId, Evaluation, GridValues, PredictionResult};
    use crate::real_facade::RealFacade;

    fn knn(k: u8, w: KnnWeights, m: KnnMetric) -> ModelConfig {
        ModelConfig::Knn {
            n_neighbors: k,
            weights: w,
            metric: m,
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

    fn fixture(name: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/parity/knn")
            .join(format!("{name}.json"));
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn floats(v: &Value) -> Vec<f64> {
        v.as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default()
    }

    fn ints(v: &Value) -> Vec<u64> {
        v.as_array()
            .map(|a| a.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default()
    }

    const CASES: [(DatasetId, u8, KnnWeights, KnnMetric, &str); 4] = [
        (
            DatasetId::Iris,
            5,
            KnnWeights::Uniform,
            KnnMetric::Minkowski,
            "iris_k5_uniform_minkowski",
        ),
        (
            DatasetId::BreastCancer,
            10,
            KnnWeights::Distance,
            KnnMetric::Euclidean,
            "breast_cancer_k10_distance_euclidean",
        ),
        (
            DatasetId::Diabetes,
            5,
            KnnWeights::Uniform,
            KnnMetric::Minkowski,
            "diabetes_k5_uniform_minkowski",
        ),
        (
            DatasetId::Titanic,
            5,
            KnnWeights::Uniform,
            KnnMetric::Minkowski,
            "titanic_k5_uniform_minkowski",
        ),
    ];

    /// The views of the real KNN against scikit-learn's values (fixtures) and against the
    /// FixtureFacade: evaluation, the D63 grid (exact `z`), C19, the form sample (C20).
    #[test]
    fn real_knn_matches_the_fixtures() {
        let go = AtomicBool::new(false);
        for (id, k, w, m, name) in CASES {
            let config = knn(k, w, m);
            let real = trained(&RealFacade, id, config);
            let fake = trained(&FixtureFacade::from_checkout(), id, config);
            let e = &fixture(name)["expected"];
            let summary = real.summary();
            let engine = match e["neighbours_test"]["fit_method"].as_str() {
                Some("brute") => SearchEngine::Brute,
                _ => SearchEngine::KdTree,
            };
            assert_eq!(summary.knn.map(|s| s.engine), Some(engine), "{name}");
            assert_eq!(summary.knn, fake.summary().knn, "{name}");
            match (real.evaluate().unwrap(), fake.evaluate().unwrap()) {
                (Evaluation::Classification(a), Evaluation::Classification(b)) => {
                    assert_eq!(a.confusion, b.confusion, "{name}");
                }
                (Evaluation::Regression(a), Evaluation::Regression(b)) => {
                    assert!((a.r2 - b.r2).abs() < 1e-12, "{name}");
                }
                _ => panic!("{name}: tareas distintas"),
            }
            // D63: the default 2D model over every row, `z` exact.
            let pair = &e["two_feature_models"]["default"];
            let f = ints(&pair["features"]);
            let grid = real
                .decision_grid(f[0] as usize, f[1] as usize, &|_, _| {}, &go)
                .unwrap();
            assert_eq!(grid.x, floats(&pair["x"]), "{name}: eje x");
            assert_eq!(grid.points.rows as usize, real_rows(id), "{name}");
            let z: Vec<f64> = pair["z"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(floats)
                .collect();
            match grid.values {
                GridValues::Classes { z: got } => {
                    let got: Vec<f64> = got.iter().map(|&c| f64::from(c)).collect();
                    assert_eq!(got, z, "{name}: frontera");
                }
                GridValues::Values { z: got, .. } => {
                    for (a, b) in got.iter().zip(&z) {
                        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{name}");
                    }
                }
            }
            if summary.model_task == TaskType::Classification {
                let d = real.class_distances().unwrap();
                let app = &e["distance_analysis"]["app"];
                for (row, expected) in d.matrix.iter().zip(app["matrix"].as_array().unwrap()) {
                    for (a, b) in row.iter().zip(floats(expected)) {
                        assert!((a - b).abs() <= 1e-10 * b.abs().max(1.0), "{name}");
                    }
                }
            }
            // C20: the default sample of the form (input float32 in Python).
            let sample = &e["prediction_form"]["default"];
            let x = floats(&sample["input_float32"]);
            let result = real.predict_one(&x).unwrap();
            let (PredictionResult::Class { neighbors, .. }
            | PredictionResult::Value { neighbors, .. }) = result;
            let rows: Vec<u64> = neighbors
                .unwrap()
                .rows
                .iter()
                .map(|n| u64::from(n.train_row))
                .collect();
            assert_eq!(rows, ints(&sample["neighbours"]), "{name}: vecinos");
            let code = real
                .code(crate::ipc::CodeTemplate::KnnTraining, None)
                .unwrap();
            assert!(code.contains("KNeighbors"), "{name}");
        }
    }

    fn real_rows(id: DatasetId) -> usize {
        RealFacade.load_builtin(id).unwrap().summary().n_rows as usize
    }

    /// C18 against `interactive.default.samples[*].sklearn_2d_all_data` (exact ties too).
    #[test]
    fn playground_reproduces_sklearn() {
        let real = trained(
            &RealFacade,
            DatasetId::Iris,
            knn(5, KnnWeights::Uniform, KnnMetric::Minkowski),
        );
        let e = &fixture("iris_k5_uniform_minkowski")["expected"]["interactive"]["default"];
        let samples = e["samples"].as_array().unwrap();
        let req = PlaygroundRequest {
            model_id: 1,
            feature_x: 0,
            feature_y: 1,
            n_neighbors: 5,
            weights: KnnWeights::Uniform,
            metric: KnnMetric::Minkowski,
            points: samples
                .iter()
                .map(|s| {
                    let p = floats(&s["point"]);
                    [p[0], p[1]]
                })
                .collect(),
            include_data: true,
            task_id: 1,
        };
        let started = std::time::Instant::now();
        let view = real.knn_playground(&req, &AtomicBool::new(false)).unwrap();
        let elapsed = started.elapsed();
        assert_eq!(view.data.as_ref().map(|d| d.rows), Some(150));
        let b = &e["bounds"];
        let expected =
            [&b["xMin"], &b["xMax"], &b["yMin"], &b["yMax"]].map(|v| v.as_f64().unwrap());
        for (a, x) in view.bounds.iter().zip(expected) {
            assert!((a - x).abs() < 1e-12);
        }
        for (point, sample) in view.points.iter().zip(samples) {
            let sk = &sample["sklearn_2d_all_data"];
            let rows: Vec<u64> = point
                .neighbors
                .iter()
                .map(|n| u64::from(n.dataset_row))
                .collect();
            assert_eq!(rows, ints(&sk["neighbours"]), "{:?}", point.point);
            assert_eq!(
                u64::from(point.class_index),
                sk["predict"].as_u64().unwrap()
            );
            assert_eq!(point.proba, floats(&sk["proba"]));
        }
        assert!(elapsed.as_millis() < 500, "{elapsed:?}");
    }

    /// D59 in the real facade: K = 20 with 15 training rows is an error in Spanish.
    #[test]
    fn k_larger_than_the_train_set_is_rejected() {
        let mut bytes = b"x,z,y\n".to_vec();
        for i in 0..21 {
            bytes.extend(format!("{i},{},{}\n", i % 3, i % 2).as_bytes());
        }
        let dataset = core::csv::read_csv(&bytes)
            .and_then(|t| t.to_dataset("y", core::TaskChoice::Classification, "f.csv"))
            .unwrap();
        let spec = TrainSpec {
            config: knn(20, KnnWeights::Uniform, KnnMetric::Minkowski),
            test_size: 0.3,
            dataset_label: String::new(),
        };
        let error = crate::real_models::train(
            &std::sync::Arc::new(dataset),
            None,
            &spec,
            &|_, _| {},
            &AtomicBool::new(false),
        )
        .err()
        .unwrap();
        assert!(
            matches!(
                error,
                CommandError::TooFewSamples {
                    k: 20,
                    n_train: 14,
                    ..
                }
            ),
            "{error:?}"
        );
    }
}
