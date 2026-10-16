//! Parity of the 🧠 Redes Neuronales core (C22 data preparation, C23 diagnostics,
//! architecture numbers) against `tests/parity/nn/*.json`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{fixture, knn::dataset, nums};
use mltutor_core::content::Messages;
use mltutor_core::dataset::{load_builtin, DatasetId, TaskType};
use mltutor_core::nn_data::{
    self, default_epochs, output_check, split_sizes, stratified_split, suggested_neurons,
    total_parameters, Activation, HistorySeries, NnError, Scaler,
};
use mltutor_core::nn_diagnostics::{diagnostics_view, history_cards};
use serde_json::Value;

/// The 18 level-(a) cases (`nn/exact`, without the dropout masks).
pub const EXACT: &[&str] = &[
    "breast_cancer_default",
    "breast_cancer_sgd_tanh_output",
    "california_housing_default_e20",
    "csv_numeric_regression_default",
    "csv_string_target_default",
    "diabetes_adam_lr01_b32",
    "diabetes_default",
    "diabetes_sgd_lr001_tanh",
    "diabetes_sigmoid_output",
    "diabetes_softmax_output",
    "iris_default",
    "iris_relu_3layers_lr03",
    "iris_rmsprop_sigmoid_1layer",
    "iris_sgd_tanh_lr01_b16",
    "iris_tanh_output",
    "moons_default",
    "wine_default",
    "wine_rmsprop_lr03_chaotic",
];

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn stratified_split_is_exact() {
    let fx = fixture("nn/split_stratified.json");
    let mut checked = 0;
    for (key, exp) in fx["expected"].as_object().unwrap() {
        let input = if let Some(csv) = key.strip_prefix("csv_") {
            serde_json::json!({
                "csv_fixture": format!("csv/{csv}.json"),
                "target": "especie",
                "app_label": format!("📄 {csv}.csv"),
            })
        } else {
            serde_json::json!({ "dataset": format!("datasets/{key}.json") })
        };
        let (ds, _) = dataset(&input);
        let codes = ds.target.class_codes().unwrap();
        let names = match &ds.target {
            mltutor_core::Target::Classes { class_names, .. } => class_names.clone(),
            _ => unreachable!(),
        };
        let split = stratified_split(&codes, &names, 0.2, 42).unwrap();
        assert_eq!(
            split.train.len(),
            exp["n_train"].as_u64().unwrap() as usize,
            "{key}"
        );
        if exp["train_indices"].is_array() {
            assert_eq!(split.train, usizes(&exp["train_indices"]), "{key}");
            assert_eq!(split.test, usizes(&exp["test_indices"]), "{key}");
        } else {
            assert_eq!(
                split.train[..20],
                usizes(&exp["train_indices_head"])[..],
                "{key}"
            );
            assert_eq!(
                split.test[..20],
                usizes(&exp["test_indices_head"])[..],
                "{key}"
            );
            use sha2::Digest;
            let sha = |v: &[usize]| {
                let bytes: Vec<u8> = v.iter().flat_map(|&i| (i as i64).to_le_bytes()).collect();
                format!("{:x}", sha2::Sha256::digest(&bytes))
            };
            assert_eq!(
                sha(&split.train),
                exp["train_indices_sha256_i64"].as_str().unwrap()
            );
            assert_eq!(
                sha(&split.test),
                exp["test_indices_sha256_i64"].as_str().unwrap()
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 8);
}

#[test]
fn exact_cases_split_and_scaler() {
    for name in EXACT {
        let fx = fixture(&format!("nn/exact/{name}.json"));
        let (ds, _) = dataset(&fx["input"]);
        let e = &fx["expected"];
        let task = ds.task();
        let codes = ds.target.class_codes();
        let split = nn_data::lesson_split(task, codes.as_deref(), &[], ds.n_rows()).unwrap();
        if e["split"]["train_indices"].is_array() {
            assert_eq!(split.train, usizes(&e["split"]["train_indices"]), "{name}");
            assert_eq!(split.test, usizes(&e["split"]["test_indices"]), "{name}");
        } else {
            assert_eq!(
                split.train[..20],
                usizes(&e["split"]["train_indices_head"])[..],
                "{name}"
            );
        }
        let sc = Scaler::fit(ds.features.view());
        for (a, b) in sc.mean.iter().zip(nums(&e["scaler"]["mean"])) {
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().max(1e-300),
                "{name} mean {a} {b}"
            );
        }
        for (a, b) in sc.scale.iter().zip(nums(&e["scaler"]["scale"])) {
            assert!((a - b).abs() <= 1e-12 * b.abs(), "{name} scale {a} {b}");
        }
        // x_fit = the first split_at scaled training rows, in f32 (1 ulp).
        let x_train = ds.features.select(ndarray::Axis(0), &split.train);
        let xs = sc.transform_f32(x_train.view());
        if let Some(rows) = e["data"]["x_fit"]["values"].as_array() {
            let p = ds.features.ncols();
            for (i, row) in rows.iter().enumerate() {
                for (j, v) in nums(row).into_iter().enumerate() {
                    let ours = xs[i * p + j];
                    let theirs = v as f32;
                    let ulp = (ours.to_bits() as i64 - theirs.to_bits() as i64).abs();
                    assert!(ulp <= 1, "{name} x_fit[{i},{j}] {ours} {theirs}");
                }
            }
        }
        // Real split sizes (D73).
        let sizes = split_sizes(
            ds.n_rows(),
            (e["data"]["validation_split"].as_f64().unwrap() * 100.0).round() as u8,
        );
        assert_eq!(
            sizes.train as u64,
            e["data"]["split_at"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(
            sizes.val as u64,
            e["data"]["n_val"].as_u64().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn errors_are_detected_before_training() {
    let fx = fixture("nn/errors.json");
    let mut seen = 0;
    for (key, case) in fx["expected"].as_object().unwrap() {
        let spec = &case["spec"];
        let input = if let Some(csv) = spec["csv"].as_str() {
            serde_json::json!({
                "csv_fixture": format!("csv/{csv}.json"),
                "target": csv_target(csv),
                "app_label": format!("📄 {csv}.csv"),
            })
        } else {
            serde_json::json!({ "dataset": format!("datasets/{}.json", spec["dataset"].as_str().unwrap()) })
        };
        let (ds, _) = dataset(&input);
        let task = ds.task();
        let n_classes = ds.target.n_classes().unwrap_or(0);
        let k = nn_data::output_size(task, n_classes);
        let out = spec["out"]
            .as_str()
            .map(|s| Activation::parse(s).unwrap())
            .unwrap_or_else(|| Activation::recommended_output(task, n_classes));
        let resolved = nn_data::resolve_training(task, out, k);
        if key == "diabetes_salida_softmax" {
            // Python trains it (the output is always 1): Rust too.
            assert!(resolved.is_ok());
        } else if key == "csv_una_muestra_por_clase" {
            assert!(resolved.is_ok());
            let codes = ds.target.class_codes().unwrap();
            let err = stratified_split(&codes, &[], 0.2, 42).unwrap_err();
            assert!(
                matches!(err, NnError::TooFewPerClass { count: 1, .. }),
                "{err:?}"
            );
        } else {
            let err = resolved.unwrap_err();
            assert!(matches!(err, NnError::UntrainableOutput { .. }), "{key}");
            assert!(output_check(task, n_classes, out).untrainable_key.is_some());
        }
        seen += 1;
    }
    assert_eq!(seen, 6);
}

fn csv_target(csv: &str) -> String {
    let fx = fixture(&format!("csv/{csv}.json"));
    let text = fx["input"]["csv_text"].as_str().unwrap();
    text.lines()
        .next()
        .unwrap()
        .split(',')
        .next_back()
        .unwrap()
        .trim()
        .to_owned()
}

#[test]
fn architecture_numbers_match_controls() {
    let fx = fixture("nn/controls.json");
    for (key, case) in fx["expected"]["datasets"].as_object().unwrap() {
        if key.starts_with("csv_") || key == "mpg" {
            continue; // MPG: Rust drops `name` (D8); CSV defaults checked by the UI tests.
        }
        let id = DatasetId::from_slug(key).unwrap();
        let ds = load_builtin(id).unwrap();
        let cfg = &case["nn_config"];
        let arch = usizes(&cfg["architecture"]);
        let task = ds.task();
        let n_classes = ds.target.n_classes().unwrap_or(0);
        let multi = (task == TaskType::Classification && n_classes > 2).then_some(n_classes);
        let input = ds.features.ncols();
        let hidden = nn_data::default_neurons(&suggested_neurons(input, 2, multi));
        let mut ours = vec![input];
        ours.extend(hidden.iter().map(|&h| usize::from(h)));
        ours.push(nn_data::output_size(task, n_classes));
        assert_eq!(ours, arch, "{key}");
        assert_eq!(
            usize::from(default_epochs(ds.n_rows()).1),
            case["epochs_default"].as_u64().unwrap() as usize,
            "{key}"
        );
        assert_eq!(
            Activation::recommended_output(task, n_classes).as_str(),
            cfg["output_activation"].as_str().unwrap()
        );
        // "🔢 Total de Parámetros" of the view.
        let total = total_parameters(&arch);
        let shown = find_metric(&case["view"], "🔢 Total de Parámetros").unwrap();
        assert_eq!(
            shown,
            mltutor_core::pyfmt::py_format(
                "{t:,}",
                &[("t", mltutor_core::pyfmt::FmtArg::Int(total as i64))]
            ),
            "{key}"
        );
    }
}

fn find_metric(node: &Value, label: &str) -> Option<String> {
    if node["kind"] == "metric" && node["label"] == label {
        return node["value"].as_str().map(str::to_owned);
    }
    node["children"]
        .as_array()
        .into_iter()
        .flatten()
        .find_map(|c| find_metric(c, label))
}

fn series_of(history: &Value, task: TaskType) -> HistorySeries {
    let key = if task == TaskType::Classification {
        "accuracy"
    } else {
        "mae"
    };
    let opt = |k: &str| history.get(k).map(nums);
    HistorySeries {
        loss: nums(&history["loss"]),
        val_loss: opt("val_loss"),
        metric_key: key.to_owned(),
        metric: opt(key),
        val_metric: opt(&format!("val_{key}")),
    }
}

/// The texts that differ from Python on purpose (`CORREGIDO`/`NUEVO` in
/// `content/redes/mensajes.toml`): the fixture's Python literal is replaced by the
/// current text before comparing, so every other text must still be identical.
fn corrected(mut expected: Value, task: TaskType, d: &Value, n_val: Option<usize>) -> Value {
    let m = Messages::neural_network().unwrap();
    let t = |k: &str| m.text("diagnostico", k).unwrap().to_owned();
    let li = |k: &str, i: usize| m.lines("diagnostico", k).unwrap()[i].to_owned();
    let mut subs: Vec<(String, String)> = vec![
        (
            "❌ Modelo no convergió suficientemente".into(),
            t("no_convergio"),
        ),
        (
            "✅ Modelo convergió con buen rendimiento".into(),
            t("convergio"),
        ),
        (
            "⚠️ Pérdida convergió pero rendimiento bajo".into(),
            t("convergio_bajo"),
        ),
        (
            "**Modelo no convergió suficientemente**".into(),
            format!("**{}**", t("alerta_no_convergio")),
        ),
        (
            "✅ Error de regresión aceptable".into(),
            t("regresion_aceptable"),
        ),
        ("❌ Error de regresión alto".into(), t("regresion_alto")),
        (
            "🌟 Modelo listo para producción".into(),
            t("estado_excelente"),
        ),
        (
            "Pérdida aumentando - posible sobreajuste".into(),
            li("medio_ayuda", 3),
        ),
        (
            "Pérdida aumentando - detener entrenamiento".into(),
            li("final_ayuda", 3),
        ),
        (
            "- 🚨 Sobreajuste".into(),
            format!("- {}", li("final_calidad", 3)),
        ),
        (
            "🚨 Patrón de sobreajuste detectado".into(),
            t("patron_sobreajuste"),
        ),
        (
            "• ✅ El modelo está listo para producción".into(),
            format!("• {}", li("rec_excelente_acciones", 0)),
        ),
        (
            "• 🚀 Puedes proceder a hacer predicciones con confianza".into(),
            format!("• {}", li("rec_excelente_acciones", 2)),
        ),
    ];
    if task == TaskType::Regression {
        let old = "⚠️ CRÍTICO: El modelo tiene muy bajo rendimiento (accuracy ≤ 70%). Las curvas pueden verse bien pero el modelo no está aprendiendo correctamente.";
        subs.push((old.into(), t("rec_critico_regresion")));
    }
    // Flat loss with a poor model (approved F5a deviation).
    if d["loss_stable"] == true && d["overall_quality"] == "poor" {
        let cv = d["stability"].as_f64().unwrap() * 100.0;
        let f = |k: &str| {
            mltutor_core::pyfmt::py_format(&t(k), &[("cv", mltutor_core::pyfmt::FmtArg::Float(cv))])
        };
        let py = |tpl: &str| {
            mltutor_core::pyfmt::py_format(tpl, &[("cv", mltutor_core::pyfmt::FmtArg::Float(cv))])
        };
        subs.push((
            py("⚠️ Entrenamiento inestable (CV: {cv:.1f}%)"),
            f("estable_rendimiento_bajo"),
        ));
        subs.push((
            py("📈 **Entrenamiento inestable:** Variabilidad del {cv:.1f}%"),
            f("patron_rendimiento_bajo"),
        ));
        subs.push((
            "• 🔧 Reducir learning rate o aumentar batch size".into(),
            format!("• {}", t("patron_rendimiento_bajo_rec")),
        ));
        subs.push((
            py("⚠️ **Entrenamiento inestable (variabilidad: {cv:.1f}%)**\n💡 Reducir learning rate o aumentar batch size"),
            format!("⚠️ **{}**\n💡 {}", f("alerta_rendimiento_bajo"), t("alerta_rendimiento_bajo_accion")),
        ));
    }
    for (old, new) in &subs {
        replace_text(&mut expected, old, new);
    }
    // The long texts are content (`content/redes/*.md`, reviewed by ml-lesson-author):
    // the structure is compared, the body is the current file.
    use mltutor_core::content::{
        markdown_body, NN_DIAG_CONTEXT_MD, NN_DIAG_SHORT_MD, NN_DIAG_TRENDS_MD,
    };
    for node in expected.as_array_mut().unwrap() {
        let label = node["label"].as_str().unwrap_or("").to_owned();
        if label == t("fases_expander") {
            node["children"][0]["text"] = markdown_body(NN_DIAG_TRENDS_MD).into();
        } else if label == t("contexto_expander") {
            node["children"][0]["text"] = markdown_body(NN_DIAG_CONTEXT_MD).into();
        } else if node["kind"] == "text"
            && node["text"]
                .as_str()
                .is_some_and(|s| s.starts_with("**🔍 Para un análisis completo"))
        {
            node["text"] = markdown_body(NN_DIAG_SHORT_MD).into();
        }
    }
    // nota_validacion_pequena after the accuracy-overfitting alarm.
    if let (Some(n), true) = (n_val, d["accuracy_overfitting"] == true) {
        if n < mltutor_core::nn_diagnostics::SMALL_VALIDATION {
            let note = mltutor_core::pyfmt::py_format(
                &t("nota_validacion_pequena"),
                &[
                    ("n", mltutor_core::pyfmt::FmtArg::Int(n as i64)),
                    ("pp", mltutor_core::pyfmt::FmtArg::Float(100.0 / n as f64)),
                ],
            );
            let col = &mut expected[2]["children"][0]["children"];
            let arr = col.as_array_mut().unwrap();
            let pos = arr
                .iter()
                .position(|x| {
                    x["text"]
                        .as_str()
                        .is_some_and(|s| s.starts_with("❌ Sobreajuste en accuracy"))
                })
                .unwrap();
            arr.insert(
                pos + 1,
                serde_json::json!({"kind": "notice", "text": note, "severity": "info"}),
            );
        }
    }
    expected
}

fn replace_text(v: &mut Value, old: &str, new: &str) {
    match v {
        Value::String(s) if s.contains(old) => *s = s.replace(old, new),
        Value::Array(a) => a.iter_mut().for_each(|x| replace_text(x, old, new)),
        Value::Object(o) => o.values_mut().for_each(|x| replace_text(x, old, new)),
        _ => {}
    }
}

fn check_numbers(name: &str, ours: &Value, exp: &Value) {
    for (k, v) in exp.as_object().unwrap() {
        let o = &ours[k];
        match v {
            Value::Number(n) => {
                let (a, b) = (o.as_f64().unwrap(), n.as_f64().unwrap());
                assert!(
                    (a - b).abs() <= 1e-12 * b.abs().max(1e-12),
                    "{name} {k}: {a} vs {b}"
                );
            }
            Value::String(s) if s == "inf" => assert!(
                o.is_null() || o.as_f64() == Some(f64::INFINITY) || o == "inf",
                "{name} {k}"
            ),
            Value::String(s) if s == "True" || s == "False" => {
                assert_eq!(o.as_bool(), Some(s == "True"), "{name} {k}")
            }
            Value::String(s) => assert_eq!(o.as_str(), Some(s.as_str()), "{name} {k}"),
            Value::Bool(b) => assert_eq!(o.as_bool(), Some(*b), "{name} {k}"),
            Value::Null => assert!(o.is_null(), "{name} {k}"),
            _ => {}
        }
    }
}

fn check_diagnostics(
    name: &str,
    history: &Value,
    task: TaskType,
    exp: &Value,
    n_val: Option<usize>,
) -> bool {
    let h = series_of(history, task);
    let view = diagnostics_view(&h, task, n_val).unwrap();
    let ours = serde_json::to_value(&view.blocks).unwrap();
    let dj = serde_json::to_value(&view.diagnostics).unwrap();
    let expected = corrected(exp["view"]["children"].clone(), task, &dj, n_val);
    let (a, b) = (ours.as_array().unwrap(), expected.as_array().unwrap());
    // Known difference: on an exactly flat third np.polyfit returns LAPACK rounding
    // noise (±1e-17 relative to the losses) whose sign picks "Descendente" or
    // "Ascendente"; Rust's closed form gives 0. Those two rows (trend metrics and the
    // comparative analysis that reads the same slopes) are then not compared.
    let scale = h.loss.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    // The same happens with exact ties of |slope| (a perfectly linear loss: the
    // fastest phase is decided by the noise).
    let flat = exp["trends"]["slopes"].as_array().is_some_and(|s| {
        let v: Vec<f64> = s.iter().map(|x| x.as_f64().unwrap()).collect();
        let tie = |a: f64, b: f64| (a.abs() - b.abs()).abs() <= 1e-10 * scale;
        v.iter().any(|x| x.abs() <= 1e-10 * scale)
            || tie(v[0], v[1])
            || tie(v[1], v[2])
            || tie(v[0], v[2])
    });
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if flat && x["kind"] == "row" && (i == 6 || i == 8) {
            continue;
        }
        assert_eq!(x, y, "{name}: block {i}");
    }
    assert_eq!(a.len(), b.len(), "{name}: blocks");
    let d = serde_json::to_value(&view.diagnostics).unwrap();
    check_numbers(name, &d, &exp["diagnostics"]);
    if let Some(tr) = &view.trends {
        // np.polyfit solves a scaled least-squares problem with LAPACK's SVD: with a
        // nearly flat third the slope cancels, so the error is relative to the scale
        // of the losses, not of the slope (≤ 3e-12 relative to the slope observed).
        for (a, b) in tr.slopes.iter().zip(nums(&exp["trends"]["slopes"])) {
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().max(scale.max(1.0)),
                "{name} slope {a} {b}"
            );
        }
    }
    if let Some(tab) = exp.get("history_tab") {
        let cards = history_cards(&h).unwrap();
        let row = serde_json::to_value(&cards.row).unwrap();
        let expected_row = tab["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["kind"] == "row")
            .unwrap();
        assert_eq!(&row, expected_row, "{name}: history cards");
    }
    flat
}

#[test]
fn diagnostics_synthetic_histories() {
    let fx = fixture("nn/diagnostics.json");
    let cases = fx["expected"]["cases"].as_object().unwrap();
    let mut degenerate = Vec::new();
    for (name, case) in cases {
        let task = if case["config"]["task_type"] == "Clasificación" {
            TaskType::Classification
        } else {
            TaskType::Regression
        };
        if check_diagnostics(name, &case["history"], task, case, None) {
            degenerate.push(name.clone());
        }
    }
    assert_eq!(cases.len(), 14);
    eprintln!("trend rows not compared (flat or tied slopes): {degenerate:?}");
}

#[test]
fn diagnostics_of_real_histories() {
    for name in EXACT {
        let fx = fixture(&format!("nn/exact/{name}.json"));
        let task = if fx["input"]["nn_config"]["task_type"] == "Clasificación" {
            TaskType::Classification
        } else {
            TaskType::Regression
        };
        let e = &fx["expected"];
        let n_val = e["data"]["n_val"].as_u64().map(|v| v as usize);
        if check_diagnostics(name, &e["history"], task, &e["diagnostics"], n_val) {
            eprintln!("{name}: trend rows not compared (flat or tied slopes)");
        }
    }
}
