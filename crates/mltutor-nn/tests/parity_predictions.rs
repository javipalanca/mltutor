//! C29 parity (D81): 🎲 Exploración interactiva and 📊 Predicción por lotes against
//! `tests/parity/nn/predictions.json`, which renders Python's (unreachable)
//! `viz/nn.py::show_neural_network_predictions` with the final weights of
//! `nn/exact/<case>` and the StandardScaler on all rows.
//!
//! Tolerances (`meta.tolerance`): xs exact; outputs 1e-5 relative to max(1, |v|)
//! (float32 forward, Keras' matmul order); classes exact unless the output is
//! within 1e-5 of the threshold or of a tie; the texts Python formats (`{:.3f}`,
//! `{:.6f}`, `{:.1f}%`) exact.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::*;
use mltutor_core::csv::read_csv;
use mltutor_core::dataset::TaskType;
use mltutor_core::model::TrainControl;
use mltutor_core::nn_data::{output_size, resolve_training, Scaler};
use mltutor_core::nn_predict::{
    batch_columns, BatchPredictions, ExploreCurve, Influence, RowPrediction, EXPLORE_DEFAULT_POINTS,
};
use mltutor_nn::NeuralNetwork;
use ndarray::Array2;
use serde_json::Value;

fn network(name: &str, ds: &mltutor_core::Dataset) -> NeuralNetwork {
    let fx = fixture(&format!("nn/exact/{name}.json"));
    let (arch, _) = config(&fx);
    let e = &fx["expected"];
    let task = ds.task();
    let n_classes = ds.target.n_classes().unwrap_or(0);
    let k = output_size(task, n_classes);
    NeuralNetwork::new(
        mlp_of(&e["final_weights"], arch.activation, arch.output_activation),
        Scaler {
            mean: nums(&e["scaler"]["mean"]),
            scale: nums(&e["scaler"]["scale"]),
            n_samples: ds.n_rows(),
        },
        task,
        if task == TaskType::Classification {
            (0..n_classes).collect()
        } else {
            Vec::new()
        },
        resolve_training(task, arch.output_activation, k).unwrap(),
    )
    .unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-5 * b.abs().max(1.0)
}

/// Every `notice`/`metric` text of a rendered view, depth first.
fn texts(nodes: &Value, out: &mut Vec<String>) {
    for n in nodes.as_array().into_iter().flatten() {
        match n["kind"].as_str() {
            Some("notice") => out.push(n["text"].as_str().unwrap().to_owned()),
            Some("metric") => out.push(format!(
                "{}={}",
                n["label"].as_str().unwrap(),
                n["value"].as_str().unwrap()
            )),
            _ => {}
        }
        texts(&n["children"], out);
    }
}

/// The number right after `prefix` in some text of `view`, compared with `value`
/// within the rounding of `decimals` plus the output tolerance (one float32 ulp
/// may change the last printed digit).
fn has_num(view: &[String], prefix: &str, value: f64, decimals: i32) -> bool {
    view.iter().any(|t| {
        let Some(rest) = t.strip_prefix(prefix) else {
            return false;
        };
        let num: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        num.parse::<f64>().is_ok_and(|p| {
            (p - value).abs() <= 0.5 * 10f64.powi(-decimals) + 1e-5 * value.abs().max(1.0)
        })
    })
}

struct Worst {
    output: f64,
    stat: f64,
}

#[test]
fn exploration_matches_python() {
    let fx = fixture("nn/predictions.json");
    let mut worst = Worst {
        output: 0.0,
        stat: 0.0,
    };
    let mut n = 0;
    for (name, case) in fx["expected"].as_object().unwrap() {
        let exact = fixture(&format!("nn/exact/{name}.json"));
        let ds = dataset(&exact["input"]);
        let net = network(name, &ds);
        for e in case["explorations"].as_array().unwrap() {
            let idx = e["sample_index"].as_u64().unwrap() as usize;
            let feature = e["feature"].as_u64().unwrap() as usize;
            let v = net
                .explore(
                    ds.features.view(),
                    idx,
                    feature,
                    EXPLORE_DEFAULT_POINTS,
                    &TrainControl::none(),
                )
                .unwrap();
            let tag = format!("{name} #{idx} f{feature}");
            assert_eq!(v.base, nums(&e["base"]), "{tag}: base");
            assert_eq!(v.xs, nums(&e["xs"]), "{tag}: xs");
            let base_raw = nums(&e["base_raw_output"]);
            let raw = flat(&e["raw_output"]);
            let k = base_raw.len();
            let traces = e["traces"].as_array().unwrap();
            let trace = |i: usize| nums(&traces[i]["y"]);
            let mut view = Vec::new();
            texts(&e["view"], &mut view);
            let mut cmp = |a: f64, b: f64, what: &str| {
                worst.output = worst.output.max((a - b).abs() / b.abs().max(1.0));
                assert!(close(a, b), "{tag}: {what} {a} vs {b}");
            };
            match (&v.base_prediction, &v.curve) {
                (
                    RowPrediction::Binary {
                        class, probability, ..
                    },
                    ExploreCurve::Binary { probability: curve },
                ) => {
                    cmp(*probability, base_raw[0], "base p");
                    for (a, b) in curve.iter().zip(trace(0)) {
                        cmp(*a, b, "p");
                    }
                    assert!(has_num(
                        &view,
                        &format!("🎯 **Predicción Base:** {class} (Probabilidad: "),
                        *probability,
                        3
                    ));
                    assert!(has_num(&view, "Rango de Probabilidades=", v.stats.range, 3));
                    assert!(has_num(&view, "Volatilidad=", v.stats.std, 3));
                }
                (
                    RowPrediction::Multiclass {
                        class, confidence, ..
                    },
                    ExploreCurve::Multiclass {
                        classes,
                        confidence: conf,
                        base_class,
                        base_class_probability,
                    },
                ) => {
                    cmp(*confidence, base_raw[*class], "base confidence");
                    assert_eq!(base_class, class);
                    let py_classes = trace(0);
                    for (i, (c, py)) in classes.iter().zip(&py_classes).enumerate() {
                        let row = &raw[i * k..(i + 1) * k];
                        let mut sorted = row.to_vec();
                        sorted.sort_by(|a, b| b.total_cmp(a));
                        if (sorted[0] - sorted[1]).abs() > 1e-5 {
                            assert_eq!(*c as f64, *py, "{tag}: class at {i}");
                        }
                        cmp(base_class_probability[i], row[*base_class], "base-class p");
                    }
                    for (a, b) in conf.iter().zip(trace(1)) {
                        cmp(*a, b, "confidence");
                    }
                    assert!(has_num(
                        &view,
                        &format!("🎯 **Predicción Base:** {class} (Confianza: "),
                        *confidence,
                        3
                    ));
                }
                (RowPrediction::Regression { value }, ExploreCurve::Regression { values }) => {
                    cmp(*value, base_raw[0], "base value");
                    for (a, b) in values.iter().zip(trace(0)) {
                        cmp(*a, b, "value");
                    }
                    assert!(
                        has_num(&view, "🎯 **Predicción Base:** ", *value, 6),
                        "{tag}: {value} {view:?}"
                    );
                    assert!(has_num(&view, "Rango de Predicciones=", v.stats.range, 6));
                    let impact = v.stats.relative_impact_pct.unwrap();
                    assert!(
                        has_num(&view, "Impacto Relativo=", impact, 1),
                        "{tag}: {view:?}"
                    );
                }
                other => panic!("{tag}: {other:?}"),
            }
            // The verdict (Python has none in multiclass: NEW there).
            let fname = e["feature_name"].as_str().unwrap();
            let verdict = view.iter().find(|t| t.contains("**Característica"));
            match verdict {
                Some(t) => {
                    let expected = match v.influence {
                        Influence::High => "🎯 **Característica muy influyente:**",
                        Influence::Moderate => "📊 **Característica moderadamente influyente:**",
                        Influence::Low => "📉 **Característica poco influyente:**",
                    };
                    assert!(t.starts_with(expected) && t.contains(fname), "{tag}: {t}");
                }
                None => assert!(matches!(v.curve, ExploreCurve::Multiclass { .. }), "{tag}"),
            }
            // Python's statistics in float32 against ours (range in f32 too).
            if let ExploreCurve::Binary { probability } = &v.curve {
                let py = trace(0);
                let r = py.iter().cloned().fold(f64::MIN, f64::max)
                    - py.iter().cloned().fold(f64::MAX, f64::min);
                worst.stat = worst.stat.max((r - v.stats.range).abs());
                assert_eq!(probability.len(), py.len());
            }
            n += 1;
        }
    }
    assert_eq!(n, 16);
    eprintln!(
        "exploration: max relative output diff {:.2e} (tol 1e-5), binary range diff {:.2e}",
        worst.output, worst.stat
    );
}

#[test]
fn batch_matches_python() {
    let fx = fixture("nn/predictions.json");
    let mut worst = 0.0_f64;
    for (name, case) in fx["expected"].as_object().unwrap() {
        let exact = fixture(&format!("nn/exact/{name}.json"));
        let ds = dataset(&exact["input"]);
        let net = network(name, &ds);
        let b = &case["batch"];
        let features: Vec<String> = case["feature_columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect();
        assert_eq!(features, ds.feature_columns, "{name}");
        // Python's results CSV: the features + Predicción [+ Probabilidad] [+ Confianza].
        let table = read_csv(b["results_csv"].as_str().unwrap().as_bytes()).unwrap();
        let names: Vec<String> = table.columns.iter().map(|c| c.name.clone()).collect();
        let mut with_extra = names.clone();
        with_extra.push(b["extra_column"].as_str().unwrap().to_owned());
        let cols = batch_columns(&features, &with_extra);
        assert!(cols.complete());
        assert_eq!(cols.extra.len(), with_extra.len() - features.len());
        assert!(cols
            .extra
            .contains(&b["extra_column"].as_str().unwrap().to_owned()));
        let notices: Vec<&str> = b["notices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["text"].as_str().unwrap())
            .collect();
        assert!(
            notices.contains(&"⚠️ Características adicionales (serán ignoradas): columna_extra")
        );
        let missing = batch_columns(&features, &features[1..]);
        let miss: Vec<&str> = b["missing_first_feature_notices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["text"].as_str().unwrap())
            .collect();
        assert!(miss.contains(
            &format!("❌ Faltan características: {}", missing.missing.join(", ")).as_str()
        ));

        let n = table.n_rows;
        let mut x = Array2::zeros((n, features.len()));
        for (j, pos) in cols.positions.iter().enumerate() {
            let col = table.columns[pos.unwrap()].data.to_f64().unwrap();
            x.column_mut(j).assign(&ndarray::aview1(&col));
        }
        let view = net.predict_batch(x.view(), &TrainControl::none()).unwrap();
        assert_eq!(view.rows, n);
        assert!(view.finite.iter().all(|&f| f), "{name}: finite");
        let col = |name: &str| table.column(name).unwrap().data.to_f64().unwrap();
        let raw = flat(&b["raw_output"]);
        let k = raw.len() / n;
        let mut cmp = |a: f64, b: f64, what: &str| {
            worst = worst.max((a - b).abs() / b.abs().max(1.0));
            assert!(close(a, b), "{name}: {what} {a} vs {b}");
        };
        match &view.predictions {
            BatchPredictions::Binary {
                predicted,
                probability,
                confidence,
            } => {
                let (pc, pp, pf) = (col("Predicción"), col("Probabilidad"), col("Confianza"));
                for i in 0..n {
                    if (pp[i] - 0.5).abs() > 1e-5 {
                        assert_eq!(
                            predicted[i].map(|c| c as f64),
                            Some(pc[i]),
                            "{name}: row {i}"
                        );
                    }
                    cmp(probability[i], pp[i], "p");
                    cmp(confidence[i], pf[i], "confidence");
                }
            }
            BatchPredictions::Multiclass {
                predicted,
                confidence,
                probabilities,
            } => {
                let (pc, pf) = (col("Predicción"), col("Confianza"));
                for i in 0..n {
                    let row = &raw[i * k..(i + 1) * k];
                    let mut s = row.to_vec();
                    s.sort_by(|a, b| b.total_cmp(a));
                    if s[0] - s[1] > 1e-5 {
                        assert_eq!(
                            predicted[i].map(|c| c as f64),
                            Some(pc[i]),
                            "{name}: row {i}"
                        );
                    }
                    cmp(confidence[i], pf[i], "confidence");
                    for (a, b) in probabilities[i].iter().zip(row) {
                        cmp(*a, *b, "probabilities");
                    }
                }
            }
            BatchPredictions::Regression { values } => {
                for (a, b) in values.iter().zip(col("Predicción")) {
                    cmp(*a, b, "value");
                }
            }
        }
    }
    eprintln!("batch: max relative output diff {worst:.2e} (tol 1e-5)");
}

/// `training_ranges` uses the unscaled training rows of the split.
#[test]
fn training_ranges_follow_the_split() {
    let iris = mltutor_core::dataset::load_builtin(mltutor_core::DatasetId::Iris).unwrap();
    let fx = fixture("nn/exact/iris_default.json");
    let (arch, mut training) = config(&fx);
    training.epochs = 2;
    let t = mltutor_nn::train_network(
        iris.features.view(),
        &iris.target,
        &arch,
        &training,
        mltutor_nn::TrainOptions::default(),
        &mltutor_nn::FitControl::default(),
    )
    .unwrap();
    let r = t.training_ranges(iris.features.view()).unwrap();
    assert_eq!(r.len(), 4);
    for (j, range) in r.iter().enumerate() {
        let vals: Vec<f64> = t
            .split
            .train
            .iter()
            .map(|&i| iris.features[[i, j]])
            .collect();
        assert_eq!(range.min, vals.iter().cloned().fold(f64::MAX, f64::min));
        assert_eq!(range.max, vals.iter().cloned().fold(f64::MIN, f64::max));
    }
    let narrow = iris.features.slice(ndarray::s![.., ..3]);
    assert!(t.training_ranges(narrow).is_err());
}

/// 🔮 Predicción individual: `form::predict_one` on the network scales inside and
/// gives Keras' output for the form's default values (`expected.prediction_form`
/// of the `full` cases: float32 input → scaler → network).
#[test]
fn predict_one_scales_inside() {
    use mltutor_core::form::{predict_one, PredictionOutcome};
    let mut n = 0;
    for &name in EXACT {
        let fx = fixture(&format!("nn/exact/{name}.json"));
        let pf = &fx["expected"]["prediction_form"];
        if !pf.is_object() {
            continue;
        }
        let ds = dataset(&fx["input"]);
        let net = network(name, &ds);
        let raw = flat(&pf["raw_output"]);
        match predict_one(&net, &nums(&pf["input_float32"])).unwrap() {
            PredictionOutcome::Class { proba, .. } => {
                let proba = proba.unwrap();
                let ours: Vec<f64> = if raw.len() == 1 {
                    vec![proba[1].1]
                } else {
                    proba.iter().map(|p| p.1).collect()
                };
                for (a, b) in ours.iter().zip(&raw) {
                    assert!(close(*a, *b), "{name}: {a} vs {b}");
                }
            }
            PredictionOutcome::Value { value } => assert!(close(value, raw[0]), "{name}"),
        }
        n += 1;
    }
    assert!(n >= 3, "{n}");
}
