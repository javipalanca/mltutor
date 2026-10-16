//! Level (a) of D69: with Keras' initial weights, no shuffling and no dropout, the
//! Rust `fit` reproduces Keras 3.12 (`tests/parity/nn/exact/*.json`) within float32
//! rounding: first-batch gradients, the first steps, the whole history while the
//! reference agrees (`reference.agree_epochs_rel_1e-3`), the callbacks' decisions,
//! the final weights and every view computed from the final weights.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::*;
use mltutor_core::dataset::TaskType;
use mltutor_core::nn_data::{output_size, resolve_training};
use mltutor_core::nn_stats::{activation_stats, weight_stats};
use mltutor_nn::{train_network, DropoutMasks, FitControl, Optimizer, TrainOptions};
use serde_json::Value;

struct Case {
    name: &'static str,
    fx: Value,
}

impl Case {
    fn load(name: &'static str) -> Self {
        Self {
            name,
            fx: fixture(&format!("nn/exact/{name}.json")),
        }
    }
    fn e(&self) -> &Value {
        &self.fx["expected"]
    }
    fn task(&self) -> TaskType {
        if self.fx["input"]["nn_config"]["task_type"] == "Clasificación" {
            TaskType::Classification
        } else {
            TaskType::Regression
        }
    }
    /// Diabetes and California: unscaled target, larger gradients.
    fn grad_tol(&self) -> f64 {
        if self.name.starts_with("diabetes") || self.name.starts_with("california") {
            5e-5
        } else {
            1e-6
        }
    }
}

#[test]
fn first_batch_gradients_and_steps() {
    let mut worst = (0.0_f64, 0.0_f64);
    for &name in EXACT {
        let c = Case::load(name);
        let (arch, training) = config(&c.fx);
        let task = c.task();
        let cfg = &c.fx["input"]["nn_config"];
        let k = cfg["output_size"].as_u64().unwrap() as usize;
        let res = resolve_training(task, arch.output_activation, k).unwrap();
        let mut mlp = mlp_of(
            &c.e()["initial_weights"],
            arch.activation,
            arch.output_activation,
        );
        let x = block(&c.e()["data"]["x_fit"]);
        let y = block(&c.e()["data"]["y_fit"]);
        let p = mlp.n_inputs();
        let rows = c.e()["first_batch"]["rows"].as_u64().unwrap() as usize;
        let (loss, grads, _) = mlp
            .gradients(&x[..rows * p], &y[..rows * k], rows, res.loss, None)
            .unwrap();
        let kl = c.e()["first_batch"]["keras_loss"].as_f64().unwrap();
        assert!(
            (f64::from(loss) - kl).abs() <= 1e-6 * kl.abs().max(1.0),
            "{name}: loss {loss} vs {kl}"
        );
        let mut g = 0.0_f64;
        for ((gw, gb), kg) in grads
            .iter()
            .zip(c.e()["first_batch"]["keras_gradients"].as_array().unwrap())
        {
            for (a, b) in gw
                .iter()
                .zip(flat(&kg["kernel"]))
                .chain(gb.iter().zip(nums(&kg["bias"])))
            {
                g = g.max((f64::from(*a) - b).abs());
            }
        }
        assert!(g <= c.grad_tol(), "{name}: gradients {g:e}");
        worst.0 = worst.0.max(g);
        // Steps 1..3 of epoch 0 (consecutive batches).
        let bs = usize::from(arch.batch_size);
        let split = c.e()["data"]["split_at"].as_u64().unwrap() as usize;
        let mut opt = Optimizer::new(arch.optimizer, training.learning_rate);
        for (i, step) in c.e()["steps"].as_array().unwrap().iter().enumerate() {
            let s = i * bs;
            let e = (s + bs).min(split);
            let (_, grads, _) = mlp
                .gradients(&x[s * p..e * p], &y[s * k..e * k], e - s, res.loss, None)
                .unwrap();
            opt.step(&mut mlp, &grads).unwrap();
            let d = max_weight_diff(&mlp, &step["weights"]);
            // 1e-6 (meta.tolerance), or 4× what the NumPy reference itself differs
            // from Keras where that is larger (Adam with LR 0.1–0.3: the first steps
            // move each weight by ≈ ±lr, sensitive to the sign of tiny gradients).
            let ref_d = step["max_abs_diff_reference"].as_f64().unwrap_or(0.0);
            let tol = 1e-6_f64.max(4.0 * ref_d);
            assert!(
                d <= tol,
                "{name}: step {} {d:e} (reference {ref_d:e})",
                i + 1
            );
            worst.1 = worst.1.max(d);
        }
    }
    eprintln!(
        "max |Δ gradient| = {:e}, max |Δ weights after steps 1-3| = {:e}",
        worst.0, worst.1
    );
}

/// Margins of the callbacks' decisions over the compared epochs: decisions with a
/// relative margin below 1e-5 are not normative.
fn decisions_are_firm(c: &Case, epochs: usize) -> bool {
    nums(&c.e()["callbacks"]["val_loss_relative_margins"])
        .iter()
        .take(epochs)
        .all(|m| *m >= 1e-5)
}

#[test]
fn history_callbacks_and_final_weights() {
    let mut report = Vec::new();
    for &name in EXACT {
        let c = Case::load(name);
        let (arch, training) = config(&c.fx);
        let ds = dataset(&c.fx["input"]);
        let init = mlp_of(
            &c.e()["initial_weights"],
            arch.activation,
            arch.output_activation,
        );
        let trained = train_network(
            ds.features.view(),
            &ds.target,
            &arch,
            &training,
            TrainOptions {
                shuffle: false,
                init: Some(init),
            },
            &FitControl::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let h = &c.e()["history"];
        let keras_epochs = nums(&h["loss"]).len();
        let agree = c.e()["reference"]["agree_epochs_rel_1e-3"]
            .as_u64()
            .unwrap() as usize;
        let metric = trained.history.metric.key();
        let mut max_rel = 0.0_f64;
        let mut max_rel_loose = 0.0_f64;
        let n_cmp = agree.min(trained.history.epochs.len());
        for (i, log) in trained.history.epochs.iter().enumerate().take(n_cmp) {
            let pairs = [
                ("loss", log.loss),
                ("val_loss", log.val_loss),
                (metric, log.metric),
                (&*format!("val_{metric}"), log.val_metric),
            ];
            for (key, ours) in pairs {
                let theirs = num(&h[key][i]);
                let rel = (ours - theirs).abs() / theirs.abs().max(1e-6);
                // meta.tolerance: 1e-4 where the NumPy reference itself is within 1e-5
                // of Keras; elsewhere (long float32 trajectories, California) the
                // reference agrees at 1e-3, so we require 1e-3 too.
                let ref_rel = num(&c.e()["reference"]["relative_diff_by_epoch"][key][i]);
                let tol = if ref_rel <= 1e-5 { 1e-4 } else { 1e-3 };
                assert!(
                    rel <= tol,
                    "{name} epoch {i} {key}: {ours} vs {theirs} (ref {ref_rel:e})"
                );
                if ref_rel <= 1e-5 {
                    max_rel = max_rel.max(rel);
                } else {
                    max_rel_loose = max_rel_loose.max(rel);
                }
            }
            if let Some(lr) = log.learning_rate {
                let theirs = num(&h["learning_rate"][i]);
                assert!(
                    (lr - theirs).abs() <= 1e-9 * theirs,
                    "{name} epoch {i} lr {lr} vs {theirs}"
                );
            }
        }
        let comparable = agree >= keras_epochs && decisions_are_firm(&c, keras_epochs);
        if comparable {
            assert_eq!(
                trained.history.epochs.len(),
                keras_epochs,
                "{name}: epochs trained"
            );
            let es = &c.e()["callbacks"]["early_stopping"];
            if es.is_object() {
                assert_eq!(
                    trained.history.best_epoch,
                    Some(es["best_epoch"].as_u64().unwrap() as usize),
                    "{name}"
                );
                assert_eq!(
                    trained.history.stopped_epoch,
                    Some(es["stopped_epoch"].as_u64().unwrap() as usize),
                    "{name}"
                );
            }
        }
        let ref_diff = c.e()["reference"]["final_weights_max_abs_diff"]
            .as_f64()
            .unwrap_or(f64::INFINITY);
        let mut wd = f64::NAN;
        if ref_diff < 1e-4 && comparable {
            wd = max_weight_diff(trained.network.mlp(), &c.e()["final_weights"]);
            assert!(wd <= 1e-4, "{name}: final weights {wd:e}");
        }
        // test.evaluate (loss, metric) with the final weights.
        if ref_diff < 1e-4 && comparable {
            let ev = nums(&c.e()["test"]["evaluate"]);
            let rel = (trained.test_loss - ev[0]).abs() / ev[0].abs().max(1e-6);
            assert!(
                rel <= 1e-4,
                "{name}: test loss {} vs {}",
                trained.test_loss,
                ev[0]
            );
        }
        report.push(format!(
            "{name}: epochs {}/{keras_epochs}, compared {n_cmp}, max rel {max_rel:.1e} (1e-4 epochs), {max_rel_loose:.1e} (1e-3 epochs), final Δw {wd:.1e}",
            trained.history.epochs.len()
        ));
    }
    for r in report {
        eprintln!("{r}");
    }
}

/// The chaotic case: Rust also becomes unstable (it is not normative beyond epoch 1).
#[test]
fn chaotic_case_first_epoch() {
    let c = Case::load("wine_rmsprop_lr03_chaotic");
    let (arch, training) = config(&c.fx);
    let ds = dataset(&c.fx["input"]);
    let init = mlp_of(
        &c.e()["initial_weights"],
        arch.activation,
        arch.output_activation,
    );
    let t = train_network(
        ds.features.view(),
        &ds.target,
        &arch,
        &training,
        TrainOptions {
            shuffle: false,
            init: Some(init),
        },
        &FitControl::default(),
    )
    .unwrap();
    let theirs = num(&c.e()["history"]["loss"][0]);
    assert!((t.history.epochs[0].loss - theirs).abs() <= 1e-4 * theirs);
    let val: Vec<f64> = t.history.epochs.iter().map(|e| e.val_loss).collect();
    let spread = val.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        / val.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        spread > 2.0,
        "RMSprop with LR 0.3 should be unstable: {val:?}"
    );
}

#[test]
fn views_from_final_weights() {
    for &name in EXACT {
        let c = Case::load(name);
        let (arch, _) = config(&c.fx);
        let e = c.e();
        let mlp = mlp_of(&e["final_weights"], arch.activation, arch.output_activation);
        let k = mlp.n_outputs();
        // test.raw_output
        let xt = block(&e["test"]["x_test"]);
        let n = xt.len() / mlp.n_inputs();
        let out = mlp.forward(&xt, n).unwrap();
        let raw = flat(&e["test"]["raw_output"]);
        let scale = raw.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for (a, b) in out.iter().zip(&raw) {
            assert!(
                (f64::from(*a) - b).abs() <= 1e-5 * scale,
                "{name}: raw {a} vs {b}"
            );
        }
        if c.task() == TaskType::Classification {
            let yp: Vec<usize> = e["test"]["y_pred"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            for (i, row) in out.chunks(k).enumerate() {
                let ours = if k == 1 {
                    usize::from(row[0] > 0.5)
                } else {
                    mltutor_nn::loss::argmax(row)
                };
                let edge = if k == 1 {
                    (row[0] - 0.5).abs() < 1e-5
                } else {
                    false
                };
                assert!(ours == yp[i] || edge, "{name}: y_pred[{i}]");
            }
        }
        if !e["weights"].is_object() {
            continue; // cases without views
        }
        // 🧠 weights
        let ws = weight_stats(&mlp.views());
        let ew = &e["weights"];
        assert!(
            (ws.global_mean - ew["global_mean"].as_f64().unwrap()).abs() <= 1e-5,
            "{name}"
        );
        assert!(
            (ws.global_std - ew["global_std"].as_f64().unwrap()).abs() <= 1e-5,
            "{name}"
        );
        for (ours, theirs) in ws.layers.iter().zip(ew["layers"].as_array().unwrap()) {
            let o = serde_json::to_value(ours).unwrap();
            for key in [
                "mean_w",
                "std_w",
                "near_zero_pct",
                "mean_b",
                "std_b",
                "min_w",
                "max_w",
                "min_b",
                "max_b",
            ] {
                let (a, b) = (o[key].as_f64().unwrap(), theirs[key].as_f64().unwrap());
                assert!((a - b).abs() <= 1e-5, "{name} {key}: {a} vs {b}");
            }
            assert_eq!(o["n_params"], theirs["n_params"]);
        }
        // 📉 activations (default and options)
        for key in ["activations_default", "activations_options"] {
            let a = &e[key];
            if !a.is_object() {
                continue;
            }
            let rows = a["params"]["rows_used"].as_u64().unwrap() as usize;
            let dead = a["params"]["dead_threshold"].as_f64().unwrap();
            let cst = a["params"]["constant_std"].as_f64().unwrap();
            let acts = mlp
                .hidden_activations(&xt[..rows * mlp.n_inputs()], rows)
                .unwrap();
            let stats: Vec<_> = acts
                .iter()
                .zip(&mlp.layers)
                .map(|(v, l)| activation_stats(v, l.units, dead, cst))
                .collect();
            for (s, t) in stats.iter().zip(a["layers"].as_array().unwrap()) {
                let ids = |v: &Value| -> Vec<usize> {
                    v.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect()
                };
                assert_eq!(s.dead, ids(&t["dead"]), "{name} {key} dead");
                assert_eq!(s.constant, ids(&t["constant"]), "{name} {key} constant");
                for (field, ours) in [
                    ("mean_act", s.mean_act),
                    ("std_act", s.std_act),
                    ("saturation_pct", s.saturation_pct),
                    ("act_max", s.act_max),
                ] {
                    let theirs = t[field].as_f64().unwrap();
                    assert!(
                        (ours - theirs).abs() <= 1e-5 * theirs.abs().max(1.0),
                        "{name} {key} {field}: {ours} vs {theirs}"
                    );
                }
            }
            let dp = mltutor_core::nn_stats::dead_pct(&stats);
            assert!(
                (dp - a["dead_pct"].as_f64().unwrap()).abs() < 1e-9,
                "{name} dead_pct"
            );
        }
        // 🎯 surface: the sampled cells (scaled space, the rest at the X_test mean).
        for key in ["surface_default", "surface_pair"] {
            let s = &e[key];
            if !s.is_object() {
                continue;
            }
            let pair: Vec<usize> = s["pair"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let means = nums(&s["mean_values"]);
            let xs = nums(&s["sample"]["x"]);
            let ys = nums(&s["sample"]["y"]);
            let mut grid = Vec::new();
            for (x, y) in xs.iter().zip(&ys) {
                let mut row = means.clone();
                row[pair[0]] = *x;
                row[pair[1]] = *y;
                grid.extend(row.iter().map(|&v| v as f32));
            }
            let out = mlp.forward(&grid, xs.len()).unwrap();
            let raw = flat(&s["sample"]["raw_output"]);
            let scale = raw.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            for (a, b) in out.iter().zip(&raw) {
                assert!(
                    (f64::from(*a) - b).abs() <= 1e-5 * scale,
                    "{name} {key}: {a} vs {b}"
                );
            }
        }
        // 🔮 prediction form: float32 input -> scaler -> network.
        let pf = &e["prediction_form"];
        let scaled: Vec<f32> = nums(&pf["scaled"]).into_iter().map(|v| v as f32).collect();
        let out = mlp.forward(&scaled, 1).unwrap();
        for (a, b) in out.iter().zip(flat(&pf["raw_output"])) {
            assert!(
                (f64::from(*a) - b).abs() <= 1e-5 * b.abs().max(1.0),
                "{name}: form {a} vs {b}"
            );
        }
        let _ = output_size;
    }
}

#[test]
fn dropout_with_given_masks() {
    // `iris_dropout_masks.json` stores as `initial_weights` the weights AFTER its 5
    // steps (golden_nn's reference optimizer updated `w0` in place); the `_w0` case,
    // added in F5a, stores the real initial ones.
    let fx = fixture("nn/exact/iris_dropout_masks_w0.json");
    let e = &fx["expected"];
    let input = &fx["input"];
    let rate = input["rate"].as_f64().unwrap() as f32;
    let bs = input["batch_size"].as_u64().unwrap() as usize;
    let ds = dataset(input);
    let codes = ds.target.class_codes().unwrap();
    let split = mltutor_core::nn_data::stratified_split(&codes, &[], 0.2, 42).unwrap();
    let sc = mltutor_core::nn_data::Scaler::fit(ds.features.view());
    let xtr = sc.transform_f32(ds.features.select(ndarray::Axis(0), &split.train).view());
    let ytr: Vec<f32> = split
        .train
        .iter()
        .flat_map(|&r| {
            let c = codes[r];
            (0..3).map(move |j| if c == j { 1.0 } else { 0.0 })
        })
        .collect();
    let mut mlp = mlp_of(
        &e["initial_weights"],
        mltutor_nn::Activation::Relu,
        mltutor_nn::Activation::Softmax,
    );
    let mut opt = Optimizer::new(
        mltutor_nn::OptimizerKind::Adam,
        input["lr"].as_f64().unwrap() as f32,
    );
    let mut worst = 0.0_f64;
    for step in e["steps"].as_array().unwrap() {
        let r = usizes(&step["rows"]);
        let masks = DropoutMasks {
            rate,
            masks: vec![
                flat(&step["mask_1"])
                    .into_iter()
                    .map(|v| v as f32)
                    .collect(),
                flat(&step["mask_2"])
                    .into_iter()
                    .map(|v| v as f32)
                    .collect(),
            ],
        };
        let (loss, grads, _) = mlp
            .gradients(
                &xtr[r[0] * 4..r[1] * 4],
                &ytr[r[0] * 3..r[1] * 3],
                bs,
                mltutor_nn::LossKind::CategoricalCrossentropy,
                Some(&masks),
            )
            .unwrap();
        let kl = step["keras_loss"].as_f64().unwrap();
        assert!(
            (f64::from(loss) - kl).abs() <= 1e-6 * kl,
            "loss {loss} vs {kl}"
        );
        opt.step(&mut mlp, &grads).unwrap();
        let d = max_weight_diff(&mlp, &step["weights"]);
        assert!(d <= 1e-6, "weights {d:e}");
        worst = worst.max(d);
    }
    // Keras' own Dropout: values {0, 1/(1-rate)}, identity at inference.
    let sem = &e["keras_dropout_semantics"];
    assert_eq!(nums(&sem["values"]), vec![0.0, 2.0]);
    assert_eq!(sem["inference_is_identity"], true);
    eprintln!("dropout masks: max |Δw| = {worst:e}");
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

/// D78: `grid::mean_grid` with the network reproduces Python's surface when both axes
/// have ≤ 200 points (the two CSV cases): the same cells and the same predictions.
#[test]
fn mean_grid_matches_python_surface() {
    use mltutor_core::grid::{mean_grid, GridValues};
    use mltutor_core::model::TrainControl;
    use mltutor_core::nn_data::Scaler;
    for name in [
        "csv_string_target_default",
        "csv_numeric_regression_default",
    ] {
        let c = Case::load(name);
        let (arch, _) = config(&c.fx);
        let e = c.e();
        let ds = dataset(&c.fx["input"]);
        let task = c.task();
        let mlp = mlp_of(&e["final_weights"], arch.activation, arch.output_activation);
        let n_classes = ds.target.n_classes().unwrap_or(0);
        let k = output_size(task, n_classes);
        let net = mltutor_nn::NeuralNetwork::new(
            mlp,
            Scaler {
                mean: nums(&e["scaler"]["mean"]),
                scale: nums(&e["scaler"]["scale"]),
                n_samples: ds.n_rows(),
            },
            task,
            (0..n_classes).collect(),
            resolve_training(task, arch.output_activation, k).unwrap(),
        )
        .unwrap();
        let test: Vec<usize> = e["split"]["test_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let x_test = ds.features.select(ndarray::Axis(0), &test);
        let y_test = ds.target.select(&test);
        let s = &e["surface_default"];
        let g = mean_grid(
            &net,
            x_test.view(),
            &y_test,
            net.scaler(),
            0,
            1,
            &TrainControl::none(),
        )
        .unwrap();
        assert_eq!(
            g.grid.axes.x.len() as u64,
            s["x_axis"]["count"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(
            g.grid.axes.y.len() as u64,
            s["y_axis"]["count"].as_u64().unwrap(),
            "{name}"
        );
        let idx: Vec<usize> = s["sample"]["index"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let z = nums(&s["sample"]["z"]);
        match &g.grid.values {
            GridValues::Classes { z: ours } => {
                let mut counts = std::collections::BTreeMap::new();
                for v in ours {
                    *counts.entry(v.to_string()).or_insert(0u64) += 1;
                }
                for (cls, n) in s["z_counts"].as_object().unwrap() {
                    let o = counts.get(cls).copied().unwrap_or(0) as i64;
                    assert!(
                        (o - n.as_i64().unwrap()).abs() <= 2,
                        "{name} class {cls}: {o} vs {n}"
                    );
                }
                let same = idx
                    .iter()
                    .zip(&z)
                    .filter(|(i, v)| f64::from(ours[**i]) == **v)
                    .count();
                assert!(same + 1 >= idx.len(), "{name}: {same}/{}", idx.len());
            }
            GridValues::Values { z: ours, .. } => {
                for (i, v) in idx.iter().zip(&z) {
                    assert!(
                        (ours[*i] - v).abs() <= 1e-5 * v.abs().max(1.0),
                        "{name} cell {i}: {} vs {v}",
                        ours[*i]
                    );
                }
            }
        }
    }
}
