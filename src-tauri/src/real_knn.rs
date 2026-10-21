//! 🔍 K-Nearest Neighbors of [`crate::real_facade::RealFacade`] (fase4-knn.md §5 and §8):
//! conversions of `mltutor_core::knn` to the IPC DTOs. Every computation is a core
//! function; here only the orchestration lives.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

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

/// C19: the distances between the classes of the train set of the model (D56), with
/// the J2 verdict; checks `cancel` (O(n²) pairs, a newer request supersedes it).
pub fn class_distances(
    model: &CoreModel,
    cancel: &AtomicBool,
) -> Result<ClassDistancesView, CommandError> {
    let knn = require_knn(model)?;
    let codes = knn.train_codes().ok_or_else(CommandError::not_classifier)?;
    let m = knn.config().metric;
    let control = TrainControl::none().with_cancel(cancel);
    let d = core::knn::class_distances(knn.train_x(), &codes, m, &control).map_err(model_error)?;
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
        separation: d.separation.filter(|s| s.is_finite()),
        separation_unbounded: d.separation.is_some_and(f64::is_infinite),
        verdict: crate::knn::ipc_verdict(d.verdict, d.ratio_verdict),
        metric: ipc_metric(d.metric),
        n_samples: to_u32(d.n_samples),
        sampled: d.sampled,
        sample_limit: to_u32(core::knn::class_distances::MAX_ROWS_PER_CLASS),
    })
}

/// What identifies the 2-feature KNN of the canvas: the pair, K, weights and metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlaygroundKey {
    features: [usize; 2],
    k: u8,
    weights: KnnWeights,
    metric: KnnMetric,
}

/// A fitted 2-feature KNN of the canvas and the bounds of its plot.
struct PlaygroundModel {
    knn: Knn,
    bounds: [f64; 4],
}

/// The last 2-feature KNN of the canvas of a trained model. Dragging a point only
/// changes the points, so every frame reuses it instead of refitting (the kd-tree of
/// California, 20 640 rows, takes several ms). The lock is held only to read or
/// replace the entry, never while fitting or searching.
#[derive(Default)]
pub struct PlaygroundCache {
    entry: Mutex<Option<(PlaygroundKey, Arc<PlaygroundModel>)>>,
    #[cfg(test)]
    fits: std::sync::atomic::AtomicUsize,
}

impl PlaygroundCache {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<(PlaygroundKey, Arc<PlaygroundModel>)>> {
        // Each update replaces the whole entry: a poisoned lock still holds a valid one.
        self.entry.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn get_or_fit(
        &self,
        key: PlaygroundKey,
        fit: impl FnOnce() -> Result<PlaygroundModel, CommandError>,
    ) -> Result<Arc<PlaygroundModel>, CommandError> {
        if let Some((cached, model)) = self.lock().as_ref() {
            if *cached == key {
                return Ok(model.clone());
            }
        }
        // Two requests with a new key may both fit; the last one stays (same result).
        let model = Arc::new(fit()?);
        #[cfg(test)]
        self.fits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        *self.lock() = Some((key, model.clone()));
        Ok(model)
    }

    #[cfg(test)]
    fn fits(&self) -> usize {
        self.fits.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// C18: the 2D KNN of the canvas over every row of the dataset (D63), fitted once per
/// pair and controls ([`PlaygroundCache`]); the points are kept inside the canvas.
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
    let control = TrainControl::none().with_cancel(cancel);
    let key = PlaygroundKey {
        features: [fx, fy],
        k,
        weights: req.weights,
        metric: req.metric,
    };
    let fitted = model.playground.get_or_fit(key, || {
        let knn = core::knn::playground::playground_model(
            dataset.features.view(),
            &dataset.target,
            [fx, fy],
            config(k, req.weights, req.metric),
            &control,
        )
        .map_err(model_error)?;
        let train = knn.train_x();
        let column = |c: usize| train.column(c).to_vec();
        let bounds = crate::knn::bounds(&column(0), &column(1));
        Ok(PlaygroundModel { knn, bounds })
    })?;
    let points = crate::knn::validate_points(&req.points, fitted.bounds)?;
    let p = core::knn::playground::playground_points(&fitted.knn, [fx, fy], &points, &control)
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
        crate::knn::params(model.spec.config.clone()).ok_or_else(CommandError::not_knn)?;
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
            let real = trained(&RealFacade, id, config.clone());
            let fake = trained(&FixtureFacade::from_checkout(), id, config.clone());
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
                let d = real.class_distances(&AtomicBool::new(false)).unwrap();
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

    /// J2: a verdict with its separation for binary and multiclass, equal in both
    /// facades, and the analysis is cancellable.
    #[test]
    fn class_distances_have_the_separation_verdict_and_cancel() {
        let go = AtomicBool::new(false);
        for (id, separation, verdict) in [
            (DatasetId::Iris, 1.58, ipc::SeparationVerdict::Moderate),
            (
                DatasetId::BreastCancer,
                1.26,
                ipc::SeparationVerdict::Moderate,
            ),
        ] {
            let config = knn(5, KnnWeights::Uniform, KnnMetric::Minkowski);
            let real = trained(&RealFacade, id, config.clone());
            let fake = trained(&FixtureFacade::from_checkout(), id, config.clone());
            let d = real.class_distances(&go).unwrap();
            let s = d.separation.unwrap();
            assert!((s - separation).abs() < 5e-3, "{id:?}: {s}");
            assert_eq!(d.verdict, verdict, "{id:?}");
            assert!(!d.sampled && d.sample_limit == 5000, "{id:?}");
            let f = fake.class_distances(&go).unwrap();
            assert!((f.separation.unwrap() - s).abs() <= 1e-10 * s, "{id:?}");
            assert_eq!(f.verdict, d.verdict, "{id:?}");
            assert!(matches!(
                real.class_distances(&AtomicBool::new(true)),
                Err(CommandError::Cancelled { .. })
            ));
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

    /// C18: the 2D KNN is fitted once per (pair, K, weights, metric); a drag (other
    /// points) reuses it with the same result, and typed points stay in the canvas.
    #[test]
    fn playground_reuses_the_2d_model_and_clamps_the_points() {
        let dataset =
            std::sync::Arc::new(core::dataset::load_builtin(core::DatasetId::Iris).unwrap());
        let spec = TrainSpec {
            config: knn(5, KnnWeights::Uniform, KnnMetric::Minkowski),
            test_size: 0.3,
            dataset_label: String::new(),
        };
        let go = AtomicBool::new(false);
        let model = crate::real_models::fit(&dataset, None, &spec, &|_, _| {}, &go).unwrap();
        let mut req = PlaygroundRequest {
            model_id: 1,
            feature_x: 0,
            feature_y: 1,
            n_neighbors: 5,
            weights: KnnWeights::Uniform,
            metric: KnnMetric::Minkowski,
            points: vec![[5.0, 3.0]],
            include_data: false,
            task_id: 1,
        };
        let first = playground(&model, &req, &go).unwrap();
        req.points = vec![[6.5, 3.1], [1e300, -1e300]];
        let dragged = playground(&model, &req, &go).unwrap();
        assert_eq!(model.playground.fits(), 1, "the drag must not refit");
        // The same as a fresh fit (the cache does not change the result).
        let fresh = crate::real_models::fit(&dataset, None, &spec, &|_, _| {}, &go).unwrap();
        assert_eq!(playground(&fresh, &req, &go).unwrap(), dragged);
        // Clamped to [min − 1, max + 1] of sepal length (4.3–7.9) and width (2.0–4.4).
        let far = dragged.points[1].point;
        assert!(
            (far[0] - 8.9).abs() < 1e-12 && (far[1] - 1.0).abs() < 1e-12,
            "{far:?}"
        );
        assert_eq!(first.bounds, dragged.bounds);
        // Another K, metric or pair refits.
        req.n_neighbors = 3;
        playground(&model, &req, &go).unwrap();
        req.metric = KnnMetric::Manhattan;
        playground(&model, &req, &go).unwrap();
        req.feature_y = 2;
        playground(&model, &req, &go).unwrap();
        assert_eq!(model.playground.fits(), 4);
        playground(&model, &req, &go).unwrap();
        assert_eq!(model.playground.fits(), 4);
        // Cancelled before the fit: nothing is cached.
        req.n_neighbors = 7;
        assert!(matches!(
            playground(&model, &req, &AtomicBool::new(true)),
            Err(CommandError::Cancelled { .. })
        ));
        assert_eq!(model.playground.fits(), 4);
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
