//! F4 parity: the KNN of `mltutor_core::knn` against scikit-learn 1.7.2
//! (`tests/parity/knn/*.json`, fase4-knn.md §6).
//!
//! * `kd_tree` cases: the neighbours of every test row in the same order and the
//!   distances bit for bit (D52, D53), plus the tree itself (`idx_array`, nodes).
//! * `brute` cases: the same neighbours (sklearn ranks the Euclidean distance with
//!   BLAS: distances to 1e-9 relative).
//! * predictions exact (classes) / 1e-12 (values); probabilities 1e-15; the grids of
//!   the 2-D models; permutation importance 1e-12; the class distances 1e-10; the
//!   interactive samples and the prediction form.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::knn::{config_of, dataset, Case, ERROR_CASE, TRAINED};
use common::{fixture, nums, rel_err, MaxErr};
use mltutor_core::grid::{decision_grid, knn_step, GridOptions, GridValues};
use mltutor_core::importance::{permutation, PermutationOptions};
use mltutor_core::knn::{
    self, class_distances, playground, Knn, PlaygroundRequest, SearchEngine, SeparationVerdict,
};
use mltutor_core::metrics;
use mltutor_core::model::{ModelError, Prediction, Predictor, TrainControl};
use mltutor_core::split::train_test_split;
use ndarray::{Array2, Axis};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn rows_of<T>(v: &Value, f: impl Fn(&Value) -> Vec<T>) -> Vec<Vec<T>> {
    v.as_array().unwrap().iter().map(f).collect()
}

/// Relative error, absolute below 1 (the fixtures' rule for values).
fn tol_err(a: f64, b: f64) -> f64 {
    if b.abs() < 1.0 {
        (a - b).abs()
    } else {
        rel_err(a, b)
    }
}

fn sha256_i64(values: &[usize]) -> String {
    let mut h = Sha256::new();
    for &v in values {
        h.update((v as i64).to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn bits_equal(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Distances of one row: bit for bit with the kd-tree; with brute force and the
/// Euclidean metric scikit-learn returns `sqrt` of its BLAS surrogate (1e-9 rel).
fn distances_match(engine: SearchEngine, euclidean: bool, a: &[f64], b: &[f64]) -> bool {
    if engine == SearchEngine::Brute && euclidean {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| rel_err(*x, *y) <= 1e-9)
    } else {
        bits_equal(a, b)
    }
}

/// Tolerance of the probabilities: 1e-15, or 1e-9 when brute-force Euclidean
/// distances (BLAS in scikit-learn) feed `weights='distance'`.
fn proba_tol(m: &Knn) -> f64 {
    use mltutor_core::knn::{EffectiveMetric, KnnWeights};
    if m.engine() == SearchEngine::Brute
        && m.effective_metric() == EffectiveMetric::Euclidean
        && m.config().weights == KnnWeights::Distance
    {
        1e-9
    } else {
        1e-15
    }
}

#[test]
fn neighbours_and_kd_tree_match_sklearn() {
    let (mut kd_rows, mut brute_err) = (0, MaxErr::default());
    for name in TRAINED {
        let case = Case::load(name);
        let nt = &case.expected()["neighbours_test"];
        let m = case.fit();
        assert_eq!(
            m.engine().as_str(),
            nt["fit_method"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            m.effective_metric().as_str(),
            nt["effective_metric"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(m.n_train() as u64, nt["n_samples_fit"].as_u64().unwrap());
        let nb = m
            .kneighbors(case.x_test().view(), &TrainControl::none())
            .unwrap();
        let want_i = rows_of(&nt["indices"], usizes);
        let want_d = rows_of(&nt["distances"], nums);
        let near: Vec<usize> = usizes(&nt["ties"]["rows_with_near_tie_at_kth_rel_1e-9"]);
        for (r, (wi, wd)) in want_i.iter().zip(&want_d).enumerate() {
            let (gi, gd) = (nb.row_indices(r), nb.row_distances(r));
            match m.engine() {
                SearchEngine::KdTree => {
                    assert_eq!(gi, &wi[..], "{name}: neighbours of test row {r}");
                    assert!(
                        bits_equal(gd, wd),
                        "{name}: distances of row {r}: {gd:?} vs {wd:?}"
                    );
                    kd_rows += 1;
                }
                SearchEngine::Brute => {
                    if near.contains(&r) {
                        let (mut a, mut b) = (gi.to_vec(), wi.clone());
                        a.sort_unstable();
                        b.sort_unstable();
                        assert_eq!(a, b, "{name}: neighbour set of row {r}");
                    } else {
                        assert_eq!(gi, &wi[..], "{name}: neighbours of test row {r}");
                    }
                    for (a, b) in gd.iter().zip(wd) {
                        let e = rel_err(*a, *b);
                        assert!(e <= 1e-9, "{name}: brute distance {a} vs {b}");
                        brute_err.add(e);
                    }
                }
            }
        }
        if let Some(tree) = m.tree() {
            let kd = &nt["reference_impl"]["kd_tree"];
            let nodes = tree.nodes();
            assert_eq!(
                nodes.len() as u64,
                kd["n_nodes"].as_u64().unwrap(),
                "{name}"
            );
            let starts: Vec<usize> = nodes.iter().map(|n| n.idx_start).collect();
            let ends: Vec<usize> = nodes.iter().map(|n| n.idx_end).collect();
            let leaves: Vec<bool> = nodes.iter().map(|n| n.is_leaf).collect();
            let want_leaves: Vec<bool> = kd["is_leaf"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect();
            // Nodes below a node that became a leaf early are never initialised.
            for (i, &leaf) in want_leaves.iter().enumerate() {
                assert_eq!(leaves[i], leaf, "{name}: is_leaf of node {i}");
                assert_eq!(
                    starts[i],
                    usizes(&kd["idx_start"])[i],
                    "{name}: idx_start {i}"
                );
                assert_eq!(ends[i], usizes(&kd["idx_end"])[i], "{name}: idx_end {i}");
            }
            if let Some(arr) = kd.get("idx_array") {
                assert_eq!(tree.idx_array(), &usizes(arr)[..], "{name}: idx_array");
            } else {
                assert_eq!(
                    sha256_i64(tree.idx_array()),
                    kd["idx_array_sha256_int64"].as_str().unwrap(),
                    "{name}: idx_array hash"
                );
            }
        }
    }
    println!(
        "kd_tree test rows bit-exact: {kd_rows}; brute distance max rel err {:e} ({})",
        brute_err.max, brute_err.n
    );
}

#[test]
fn predictions_probabilities_and_metrics() {
    let (mut val, mut prob, mut met) = (MaxErr::default(), MaxErr::default(), MaxErr::default());
    for name in TRAINED {
        let case = Case::load(name);
        let e = case.expected();
        let m = case.fit();
        let pred = m.predict(case.x_test().view()).unwrap();
        let y_test = case.y_test();
        match &pred {
            Prediction::Classes(c) => {
                let classes: Vec<usize> = usizes(&e["classes"]);
                assert_eq!(m.classes().unwrap(), &classes[..], "{name}: classes_");
                assert_eq!(c, &usizes(&e["y_pred"]), "{name}: y_pred");
                let proba = m.predict_proba(case.x_test().view()).unwrap().unwrap();
                let tol = proba_tol(&m);
                for (r, row) in rows_of(&e["proba"], nums).iter().enumerate() {
                    for (a, b) in proba.row(r).iter().zip(row) {
                        let d = (a - b).abs();
                        assert!(d <= tol, "{name}: proba row {r}: {a} vs {b}");
                        prob.add(d);
                    }
                }
                let n_classes = y_test.n_classes().unwrap();
                let ev = metrics::classification(&y_test.class_codes().unwrap(), c, n_classes);
                let acc = e["evaluation"]["accuracy"].as_f64().unwrap();
                met.add((ev.accuracy - acc).abs());
                assert!((ev.accuracy - acc).abs() <= 1e-12, "{name}: accuracy");
            }
            Prediction::Values(v) => {
                for (a, b) in v.iter().zip(nums(&e["y_pred"])) {
                    let d = tol_err(*a, b);
                    assert!(d <= 1e-12, "{name}: y_pred {a} vs {b}");
                    val.add(d);
                }
                let (mse, rmse, mae, r2) = metrics::regression_scores(&y_test.numeric(), v);
                for (a, k) in [(mse, "mse"), (rmse, "rmse"), (mae, "mae"), (r2, "r2")] {
                    let d = tol_err(a, e["evaluation"][k].as_f64().unwrap());
                    assert!(d <= 1e-12, "{name}: {k}");
                    met.add(d);
                }
            }
        }
    }
    println!(
        "values max err {:e} ({}); proba max abs err {:e} ({}); metrics {:e} ({})",
        val.max, val.n, prob.max, prob.n, met.max, met.n
    );
}

#[test]
fn two_feature_grids_match() {
    let (mut cells, mut err) = (0usize, MaxErr::default());
    for name in TRAINED {
        let case = Case::load(name);
        let m = case.fit();
        let x_all = &case.ds.features;
        let y_all = &case.ds.target;
        let models = case.expected()["two_feature_models"].as_object().unwrap();
        for (key, t) in models {
            let f = usizes(&t["features"]);
            // `step_h` of the fixture is the final step (after the range rules).
            let opts = GridOptions {
                step: knn_step(x_all.nrows()),
            };
            let ctl = TrainControl::none();
            let grid = decision_grid(&m, x_all.view(), y_all, f[0], f[1], opts, &ctl).unwrap();
            let refit = knn::fit(
                *m.config(),
                x_all.select(Axis(1), &[f[0], f[1]]).view(),
                y_all,
                &ctl,
            )
            .unwrap();
            assert_eq!(refit.engine().as_str(), t["fit_method"].as_str().unwrap());
            assert!(
                bits_equal(&grid.axes.x, &nums(&t["x"])),
                "{name}/{key}: x axis"
            );
            assert!(
                bits_equal(&grid.axes.y, &nums(&t["y"])),
                "{name}/{key}: y axis"
            );
            let want: Vec<f64> = t["z"].as_array().unwrap().iter().flat_map(nums).collect();
            match &grid.values {
                GridValues::Classes { z } => {
                    assert_eq!(z.len(), want.len());
                    for (i, (a, b)) in z.iter().zip(&want).enumerate() {
                        assert_eq!(f64::from(*a), *b, "{name}/{key}: cell {i}");
                    }
                }
                GridValues::Values { z, .. } => {
                    assert_eq!(z.len(), want.len());
                    for (i, (a, b)) in z.iter().zip(&want).enumerate() {
                        let d = tol_err(*a, *b);
                        assert!(d <= 1e-12, "{name}/{key}: cell {i}: {a} vs {b}");
                        err.add(d);
                    }
                }
            }
            cells += want.len();
        }
    }
    println!("grid cells checked: {cells}; values max err {:e}", err.max);
}

#[test]
fn permutation_importance_matches() {
    let mut err = MaxErr::default();
    for name in TRAINED {
        let case = Case::load(name);
        let e = &case.expected()["permutation_importance"];
        let m = case.fit();
        let pi = permutation(
            &m,
            case.x_test().view(),
            &case.y_test(),
            PermutationOptions::default(),
            &TrainControl::none(),
        )
        .unwrap();
        assert_eq!(u64::from(pi.seed), e["seed"].as_u64().unwrap());
        let mut add = |a: f64, b: f64, what: &str| {
            let d = (a - b).abs();
            assert!(d <= 1e-12, "{name}: {what} {a} vs {b}");
            err.add(d);
        };
        add(
            pi.baseline,
            e["baseline_score"].as_f64().unwrap(),
            "baseline",
        );
        for (c, row) in e["importances"].as_array().unwrap().iter().enumerate() {
            for (r, b) in nums(row).iter().enumerate() {
                add(pi.importances[[c, r]], *b, "importance");
            }
        }
        for (a, b) in pi.mean.iter().zip(nums(&e["importances_mean"])) {
            add(*a, b, "mean");
        }
        for (a, b) in pi.std.iter().zip(nums(&e["importances_std"])) {
            add(*a, b, "std");
        }
    }
    println!("permutation max abs err {:e} ({})", err.max, err.n);
}

fn check_distance_block(name: &str, d: &knn::ClassDistances, e: &Value, err: &mut MaxErr) {
    assert_eq!(d.classes, usizes(&e["classes"]), "{name}");
    let mut add = |a: f64, b: f64, what: &str| {
        let r = rel_err(a, b);
        assert!(r <= 1e-10, "{name}: {what} {a} vs {b}");
        err.add(r);
    };
    for (i, row) in rows_of(&e["matrix"], nums).iter().enumerate() {
        for (j, b) in row.iter().enumerate() {
            add(d.matrix[i][j], *b, "matrix");
        }
    }
    add(d.min.unwrap(), e["min_distance"].as_f64().unwrap(), "min");
    add(d.max.unwrap(), e["max_distance"].as_f64().unwrap(), "max");
    add(d.ratio, e["ratio"].as_f64().unwrap(), "ratio");
    for (a, b) in d
        .within
        .iter()
        .zip(e["within_class_mean_D67"].as_array().unwrap())
    {
        match b.as_f64() {
            Some(b) => add(a.unwrap(), b, "within"),
            None => assert!(a.is_none()),
        }
    }
    let shown = &e["metrics_shown"];
    assert_eq!(
        format!("{:.3}", d.min.unwrap()),
        shown["Distancia Mínima"].as_str().unwrap()
    );
    assert_eq!(
        format!("{:.3}", d.max.unwrap()),
        shown["Distancia Máxima"].as_str().unwrap()
    );
    assert_eq!(
        format!("{:.2}", d.ratio),
        shown["Ratio Max/Min"].as_str().unwrap()
    );
    let python = match e["verdict"].as_str().unwrap() {
        "success" => SeparationVerdict::Separated,
        "warning" => SeparationVerdict::Moderate,
        _ => SeparationVerdict::Close,
    };
    assert_eq!(d.ratio_verdict, python, "{name}: verdict");
    if d.classes.len() == 2 {
        assert_eq!(d.verdict, SeparationVerdict::Binary);
    } else {
        assert_eq!(d.verdict, python);
    }
}

#[test]
fn class_distances_match() {
    let mut err = MaxErr::default();
    for name in TRAINED {
        let case = Case::load(name);
        let Some(da) = case.expected().get("distance_analysis") else {
            continue;
        };
        let n = case.ds.n_rows();
        let ctl = TrainControl::none();
        // Python: always test_size = 0.3.
        let split = train_test_split(n, 0.3, 42).unwrap();
        let x = case.ds.features.select(Axis(0), &split.train);
        let codes = case.ds.target.select(&split.train).class_codes().unwrap();
        let d = class_distances(x.view(), &codes, case.config.metric, &ctl).unwrap();
        check_distance_block(name, &d, &da["app"], &mut err);
        // D56: the model's split.
        if let Some(own) = da.get("with_model_split_D56") {
            let codes = case.y_train().class_codes().unwrap();
            let d =
                class_distances(case.x_train().view(), &codes, case.config.metric, &ctl).unwrap();
            assert_eq!(d.n_samples, case.train.len());
            check_distance_block(name, &d, own, &mut err);
        }
    }
    println!("class distances max rel err {:e} ({})", err.max, err.n);
}

#[test]
fn interactive_samples_match_the_2d_model() {
    let mut n = 0;
    for name in TRAINED {
        let case = Case::load(name);
        let Some(inter) = case.expected().get("interactive") else {
            continue;
        };
        for (key, view) in inter.as_object().unwrap() {
            let f = usizes(&view["features"]);
            let samples = view["samples"].as_array().unwrap();
            let req = PlaygroundRequest {
                features: [f[0], f[1]],
                config: case.config,
                points: samples
                    .iter()
                    .map(|s| {
                        let p = nums(&s["point"]);
                        [p[0], p[1]]
                    })
                    .collect(),
            };
            let pg = playground(
                case.ds.features.view(),
                &case.ds.target,
                &req,
                &TrainControl::none(),
            )
            .unwrap();
            let b = &view["bounds"];
            assert_eq!(
                pg.bounds,
                [
                    b["xMin"].as_f64().unwrap(),
                    b["xMax"].as_f64().unwrap(),
                    b["yMin"].as_f64().unwrap(),
                    b["yMax"].as_f64().unwrap()
                ],
                "{name}/{key}: bounds"
            );
            assert_eq!(pg.data.nrows() as u64, view["n_points"].as_u64().unwrap());
            let palette: Vec<String> = view["palette"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect();
            assert_eq!(knn::PALETTE.to_vec(), palette);
            for (ci, info) in view["class_info"].as_array().unwrap().iter().enumerate() {
                assert_eq!(pg.classes[ci] as u64, info["id"].as_u64().unwrap());
                assert_eq!(pg.colors[ci], info["color"].as_str().unwrap());
            }
            for (s, got) in samples.iter().zip(&pg.points) {
                let sk = &s["sklearn_2d_all_data"];
                assert_eq!(
                    got.class as u64,
                    sk["predict"].as_u64().unwrap(),
                    "{name}/{key}"
                );
                assert_eq!(
                    got.neighbors.iter().map(|n| n.index).collect::<Vec<_>>(),
                    usizes(&sk["neighbours"]),
                    "{name}/{key}: neighbours"
                );
                let d: Vec<f64> = got.neighbors.iter().map(|n| n.distance).collect();
                let euclid = case.config.metric.effective() == knn::EffectiveMetric::Euclidean;
                assert!(
                    distances_match(pg.engine, euclid, &d, &nums(&sk["distances"])),
                    "{name}/{key}"
                );
                let tol = if pg.engine == SearchEngine::Brute && euclid {
                    1e-9
                } else {
                    1e-15
                };
                for (a, b) in got.proba.iter().zip(nums(&sk["proba"])) {
                    assert!((a - b).abs() <= tol, "{name}/{key}: proba");
                }
                let conf = sk["confidence_max_proba"].as_f64().unwrap();
                assert!((got.confidence - conf).abs() <= tol);
                n += 1;
            }
        }
    }
    println!("interactive samples checked: {n}");
}

#[test]
fn prediction_form_neighbours() {
    let mut n = 0;
    for name in TRAINED {
        let case = Case::load(name);
        let Some(pf) = case.expected().get("prediction_form") else {
            continue;
        };
        let m = case.fit();
        for key in ["default", "custom"] {
            let want = &pf[key];
            if want.is_null() {
                continue;
            }
            let e = m.explain(&nums(&want["input_float32"])).unwrap();
            assert_eq!(
                e.neighbors.iter().map(|n| n.index).collect::<Vec<_>>(),
                usizes(&want["neighbours"]),
                "{name}/{key}"
            );
            let d: Vec<f64> = e.neighbors.iter().map(|n| n.distance).collect();
            let euclid = m.effective_metric() == knn::EffectiveMetric::Euclidean;
            assert!(
                distances_match(m.engine(), euclid, &d, &nums(&want["distances"])),
                "{name}/{key}"
            );
            match (e.class, e.value) {
                (Some(c), _) => {
                    assert_eq!(c as u64, want["predict"].as_u64().unwrap(), "{name}/{key}");
                    let tol = proba_tol(&m);
                    for (a, b) in e.proba.unwrap().iter().zip(nums(&want["predict_proba"])) {
                        assert!((a - b).abs() <= tol);
                    }
                }
                (None, Some(v)) => {
                    assert!(tol_err(v, want["predict"].as_f64().unwrap()) <= 1e-12);
                }
                _ => panic!("{name}: empty explanation"),
            }
            n += 1;
        }
    }
    println!("prediction form samples checked: {n}");
}

#[test]
fn k_larger_than_train_is_an_error() {
    let fx = fixture(&format!("knn/{ERROR_CASE}.json"));
    let (ds, _) = dataset(&fx["input"]);
    let split =
        train_test_split(ds.n_rows(), fx["input"]["test_size"].as_f64().unwrap(), 42).unwrap();
    let n_train = fx["expected"]["n_train"].as_u64().unwrap() as usize;
    assert_eq!(split.train.len(), n_train);
    let x = ds.features.select(Axis(0), &split.train);
    let err = knn::fit(
        config_of(&fx["input"]),
        x.view(),
        &ds.target.select(&split.train),
        &TrainControl::none(),
    )
    .unwrap_err();
    assert_eq!(err, ModelError::TooFewSamples { k: 20, n_train });
}

/// The 2-feature models with classes refitted through `Predictor::refit` keep the
/// engine rule (Cancer: brute with 30 features, kd_tree in 2-D).
#[test]
fn refit_switches_engine() {
    let case = Case::load("breast_cancer_k5_uniform_minkowski");
    let m = case.fit();
    assert_eq!(m.engine(), SearchEngine::Brute);
    let x2: Array2<f64> = case.x_train().select(Axis(1), &[0, 1]);
    let r = m
        .refit(x2.view(), &case.y_train(), &TrainControl::none())
        .unwrap();
    let r = r.as_any().downcast_ref::<Knn>().unwrap();
    assert_eq!(r.engine(), SearchEngine::KdTree);
}

/// `tie_at_kth` against `neighbours_test.ties.rows_with_tie_at_kth` (the rows of the
/// test set where scikit-learn had to choose among equally distant samples).
#[test]
fn ties_at_the_kth_neighbour_match() {
    let mut n = 0;
    for name in TRAINED {
        let case = Case::load(name);
        let ties = &case.expected()["neighbours_test"]["ties"]["rows_with_tie_at_kth"];
        let m = case.fit();
        let x = case.x_test();
        let want: Vec<(usize, usize, usize, f64)> = ties
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                (
                    t["row"].as_u64().unwrap() as usize,
                    t["n_at_kth_distance"].as_u64().unwrap() as usize,
                    t["slots"].as_u64().unwrap() as usize,
                    t["kth_distance"].as_f64().unwrap(),
                )
            })
            .collect();
        for r in 0..x.nrows() {
            let e = m.explain(x.row(r).as_slice().unwrap()).unwrap();
            let w = want.iter().find(|t| t.0 == r);
            match (e.tie_at_kth, w) {
                (None, None) => {}
                (Some(t), Some(&(_, n_tied, slots, d))) => {
                    assert_eq!((t.n_tied, t.slots), (n_tied, slots), "{name}: row {r}");
                    assert!(rel_err(t.distance, d) <= 1e-12, "{name}: row {r}");
                    n += 1;
                }
                (got, want) => panic!("{name}: row {r}: {got:?} vs {want:?}"),
            }
        }
    }
    println!("ties at the K-th neighbour checked: {n}");
}
