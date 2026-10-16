//! 🏗️ and ⚙️ against `tests/parity/nn/controls.json` (the Python views of every dataset and
//! the variations of the controls) and the scripts of `code/nn_training.json`.

use std::path::Path;

use serde_json::Value;

use super::*;
use crate::core_facade::CoreFacade;
use crate::fixture_facade::FixtureFacade;
use crate::ipc::{DatasetId, NnOptimizer};

fn fixture(relative: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/parity")
        .join(relative);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn summary(id: DatasetId) -> DatasetSummary {
    let dataset = FixtureFacade::from_checkout().load_builtin(id).unwrap();
    let mut summary = dataset.summary();
    summary.dataset = DatasetRef::Builtin { id };
    summary
}

/// Texts marked CORREGIDO/NUEVO in `content/redes/mensajes.toml` differ from Python on
/// purpose: the parity of texts skips them (MEMORY, F2).
fn corrected(key: &str) -> bool {
    let lines: Vec<&str> = MESSAGES.lines().collect();
    lines.iter().enumerate().any(|(i, line)| {
        line.starts_with(&format!("{key} ="))
            && lines[..i]
                .iter()
                .rev()
                .take_while(|l| l.starts_with('#'))
                .any(|l| l.contains("CORREGIDO") || l.contains("NUEVO"))
    })
}

fn id_of(slug: &str) -> Option<DatasetId> {
    DatasetId::ALL.into_iter().find(|id| id.slug() == slug)
}

/// Every node of a rendered Python page, depth first.
fn nodes(value: &Value, out: &mut Vec<Value>) {
    out.push(value.clone());
    if let Some(children) = value["children"].as_array() {
        for child in children {
            nodes(child, out);
        }
    }
}

fn find<'a>(all: &'a [Value], kind: &str, label: &str) -> Option<&'a Value> {
    all.iter()
        .find(|n| n["kind"] == kind && n["label"].as_str() == Some(label))
}

fn default_arch(view: &ArchitectureView, layers: usize) -> NnArchitecture {
    NnArchitecture {
        hidden: view.default_neurons[..layers].to_vec(),
        activation: NnActivation::Relu,
        output_activation: view.recommended_output,
        dropout: 0.2,
        batch_size: 64,
        optimizer: NnOptimizer::Adam,
    }
}

fn view_of(summary: &DatasetSummary, arch: &NnArchitecture) -> ArchitectureView {
    architecture_view(summary.dataset, summary, arch, 20, None)
}

/// The default network of every built-in dataset: sizes, parameters, complexity,
/// captions, epochs and the selected output (§2.2, nn/controls.json → datasets).
#[test]
fn defaults_of_every_dataset_match_python() {
    let controls = fixture("nn/controls.json");
    let datasets = controls["expected"]["datasets"].as_object().unwrap();
    let mut checked = 0;
    for (slug, case) in datasets {
        let Some(id) = id_of(slug) else {
            continue; // the two CSV cases
        };
        let summary = summary(id);
        let probe = view_of(&summary, &default_arch_probe());
        let arch = default_arch(&probe, 2);
        let view = view_of(&summary, &arch);
        let config = &case["nn_config"];
        let mut expected: Vec<u32> = config["architecture"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        if id == DatasetId::Mpg {
            // D8: `name` is no longer a feature (8 → 7 inputs).
            expected[0] = 7;
        }
        let sizes: Vec<u32> = view.layers.iter().map(|l| l.neurons).collect();
        assert_eq!(sizes, expected, "{slug}");
        assert_eq!(
            view.recommended_output.keras_name(),
            config["output_activation"],
            "{slug}"
        );
        assert_eq!(
            view.n_rows,
            case["n_rows"].as_u64().unwrap() as u32,
            "{slug}"
        );
        assert_eq!(
            u64::from(view.default_epochs),
            case["epochs_default"].as_u64().unwrap(),
            "{slug}"
        );
        let mut all = Vec::new();
        nodes(&case["view"], &mut all);
        let total = find(&all, "metric", "🔢 Total de Parámetros").unwrap();
        let shown = total["value"].as_str().unwrap().replace(',', "");
        if id != DatasetId::Mpg {
            assert_eq!(view.total_params.to_string(), shown, "{slug}");
            assert_eq!(
                view.layers.iter().map(|l| l.params).sum::<u64>(),
                view.total_params,
                "{slug}: D76, the rows add up to the total"
            );
            let complexity = find(&all, "metric", "⚖️ Complejidad").unwrap();
            let level = match view.complexity {
                Complexity::Low => "Baja",
                Complexity::Medium => "Media",
                Complexity::High => "Alta",
            };
            assert_eq!(complexity["value"], level, "{slug}");
            assert_eq!(
                complexity["help"].as_str().unwrap(),
                format!("Ratio parámetros/datos: {:.2}", view.complexity_ratio),
                "{slug}"
            );
        }
        // Captions of the batch hint and the sliders of the neurons.
        let captions: Vec<&str> = all
            .iter()
            .filter(|n| n["kind"] == "caption")
            .filter_map(|n| n["text"].as_str())
            .collect();
        let batch_key = match view.batch_hint {
            BatchHint::Large => "batch_grande",
            BatchHint::Optimal => "batch_optimo",
            BatchHint::Small => "batch_pequeno",
        };
        let batch_text = nn_text("arquitectura", batch_key, "");
        assert!(
            captions.contains(&batch_text.as_str()),
            "{slug}: {batch_text}"
        );
        for i in 0..2 {
            let slider = find(
                &all,
                "slider",
                &format!("Neuronas en capa oculta {}", i + 1),
            )
            .unwrap();
            assert_eq!(
                slider["value"].as_u64().unwrap(),
                u64::from(view.default_neurons[i]),
                "{slug}"
            );
            assert_eq!(
                slider["help"].as_str().unwrap(),
                format!(
                    "💡 Sugerencia para capa {}: {} neuronas",
                    i + 1,
                    view.suggested_neurons[i]
                ),
                "{slug}"
            );
        }
        // The output selector: options and the message of the default combination.
        let output = find(&all, "select", "Función de activación de salida").unwrap();
        let options: Vec<&str> = view.output_options.iter().map(|a| a.keras_name()).collect();
        assert_eq!(output["options"], serde_json::json!(options), "{slug}");
        let notice = nn_text("arquitectura", &view.output_check.key, "");
        assert!(
            all.iter()
                .any(|n| n["kind"] == "notice" && n["text"] == notice.as_str()),
            "{slug}: {notice}"
        );
        assert_eq!(view.output_check.untrainable, None, "{slug}");
        // arquitectura-tamano.md: the numbers of the expander.
        let size = &all
            .iter()
            .find(|n| {
                n["kind"] == "expander" && n["label"] == "💡 ¿Cómo decidir el tamaño de la red?"
            })
            .unwrap()["children"][0]["text"];
        let text = size.as_str().unwrap();
        if id != DatasetId::Mpg {
            let tips = view.size_tips;
            assert!(
                text.contains(&format!("Pequeño: {} - {} neuronas", tips.half, view.input)),
                "{slug}"
            );
            assert!(
                text.contains(&format!(
                    "Grande: {} - {} neuronas",
                    tips.double, tips.quadruple
                )),
                "{slug}"
            );
        }
        assert!(
            text.contains(&format!(
                "mantén parámetros < {}",
                view.size_tips.max_params
            )),
            "{slug}"
        );
        let tips = &all
            .iter()
            .find(|n| {
                n["kind"] == "expander" && n["label"] == "💡 Tips para optimizar tu configuración"
            })
            .unwrap()["children"][0]["text"];
        assert!(
            tips.as_str().unwrap().contains(&format!(
                "→ Recomendado: {}",
                view.size_tips.batch_recommended
            )),
            "{slug}"
        );
        checked += 1;
    }
    assert_eq!(checked, 10);
}

fn default_arch_probe() -> NnArchitecture {
    NnArchitecture {
        hidden: vec![10, 10],
        activation: NnActivation::Relu,
        output_activation: NnActivation::Relu,
        dropout: 0.2,
        batch_size: 64,
        optimizer: NnOptimizer::Adam,
    }
}

/// The variations of Iris, Cáncer and Diabetes: parameters, layer table (corrected, D76)
/// and the message of each output activation.
#[test]
fn variations_match_python() {
    let controls = fixture("nn/controls.json");
    let variations = controls["expected"]["variations"].as_object().unwrap();
    for (slug, cases) in variations {
        let summary = summary(id_of(slug).unwrap());
        for (name, page) in cases.as_object().unwrap() {
            let mut all = Vec::new();
            nodes(page, &mut all);
            let slider =
                |label: &str| find(&all, "slider", label).and_then(|n| n["value"].as_f64());
            let select = |label: &str| find(&all, "select", label).map(|n| n["value"].clone());
            let layers = slider("Número de capas ocultas").unwrap() as usize;
            let hidden: Vec<u16> = (0..layers)
                .map(|i| slider(&format!("Neuronas en capa oculta {}", i + 1)).unwrap() as u16)
                .collect();
            let act = |v: Value| -> NnActivation { serde_json::from_value(v).unwrap() };
            let arch = NnArchitecture {
                hidden,
                activation: act(select("Función de activación").unwrap()),
                output_activation: act(select("Función de activación de salida").unwrap()),
                dropout: slider("Tasa de Dropout").unwrap() as f32,
                batch_size: select("Tamaño de Batch").unwrap().as_u64().unwrap() as u16,
                optimizer: serde_json::from_value(select("Optimizador").unwrap()).unwrap(),
            };
            let view = view_of(&summary, &arch);
            let total = find(&all, "metric", "🔢 Total de Parámetros").unwrap()["value"]
                .as_str()
                .unwrap()
                .replace(',', "");
            assert_eq!(view.total_params.to_string(), total, "{slug}/{name}");
            let notice = nn_text("arquitectura", &view.output_check.key, "");
            let severity = match view.output_check.level {
                OutputLevel::Success => "success",
                OutputLevel::Warning => "warning",
                OutputLevel::Error => "error",
            };
            assert!(
                // The severity always matches; the text unless the content corrected it.
                all.iter().any(|n| n["kind"] == "notice"
                    && n["severity"] == severity
                    && (corrected(&view.output_check.key) || n["text"] == notice.as_str())),
                "{slug}/{name}: {notice} ({severity})"
            );
            // Python's table is shifted (D76): our rows are the true layers.
            let rows = find_table(&all);
            assert_eq!(
                rows.len(),
                view.layers.len() - 1,
                "{slug}/{name}: Python lacks a row"
            );
            assert_eq!(view.layers.len(), layers + 2);
        }
    }
}

fn find_table(all: &[Value]) -> Vec<Value> {
    all.iter()
        .find(|n| n["kind"] == "table")
        .and_then(|n| n["table"]["rows"].as_array().cloned())
        .unwrap_or_default()
}

/// D73: the real sizes of Keras for every validation percentage (Iris, Cáncer, Diabetes).
#[test]
fn split_sizes_are_the_real_ones() {
    let controls = fixture("nn/controls.json");
    for (slug, cases) in controls["expected"]["training"].as_object().unwrap() {
        let summary = summary(id_of(slug).unwrap());
        let table = cases["distribucion_real_vs_caption"].as_object().unwrap();
        assert_eq!(table.len(), 31);
        for (pct, row) in table {
            let s = split_sizes(summary.n_rows, pct.parse().unwrap());
            let real: Vec<u32> = row["real"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(vec![s.train, s.val, s.test], real, "{slug} {pct}%");
        }
    }
    assert_eq!(
        split_sizes(150, 20),
        SplitSizes {
            train: 96,
            val: 24,
            test: 30
        }
    );
    assert_eq!(split_sizes(569, 20).test, 114);
}

/// D71: exactly the four combinations of `nn/errors.json` cannot be trained.
#[test]
fn untrainable_outputs_are_blocked_with_a_spanish_message() {
    use NnActivation::*;
    let multi = NnTask::Multiclass(3);
    for (task, output, key) in [
        (multi, Sigmoid, "no_entrenable_multiclase_sigmoid"),
        (multi, Linear, "no_entrenable_linear"),
        (NnTask::Binary, Softmax, "no_entrenable_binaria_softmax"),
        (NnTask::Binary, Linear, "no_entrenable_linear"),
    ] {
        let Err(CommandError::UntrainableOutput {
            key: k, message, ..
        }) = check_trainable(task, output)
        else {
            panic!("{task:?} {output:?}");
        };
        assert_eq!(k, key);
        assert!(message.starts_with("🚫"), "{message}");
        assert!(!message.contains('{'), "{message}");
    }
    assert!(nn_text("arquitectura", "no_entrenable_multiclase_sigmoid", "").contains("{clases}"));
    for (task, output) in [
        (multi, Softmax),
        (multi, Tanh),
        (NnTask::Binary, Sigmoid),
        (NnTask::Binary, Tanh),
        (NnTask::Regression, Softmax),
        (NnTask::Regression, Linear),
        (NnTask::Regression, Sigmoid),
    ] {
        assert!(check_trainable(task, output).is_ok(), "{task:?} {output:?}");
    }
    // Losses of §2.2.1.
    assert_eq!(resolve(NnTask::Binary, Tanh).loss, "mse");
    assert!(!resolve(NnTask::Binary, Sigmoid).one_hot);
    assert!(resolve(multi, Tanh).one_hot);
    assert_eq!(resolve(multi, Softmax).loss, "categorical_crossentropy");
    assert_eq!(resolve(NnTask::Regression, Softmax).metric, NnMetric::Mae);
}

fn training() -> NnTraining {
    NnTraining {
        learning_rate: 0.001,
        epochs: 50,
        validation_pct: 20,
        patience: Some(10),
        lr_factor: Some(0.5),
        seed: 42,
    }
}

#[test]
fn controls_are_validated() {
    let arch = default_arch_probe();
    let arch = NnArchitecture {
        output_activation: NnActivation::Softmax,
        ..arch
    };
    assert!(validate(&arch, &training()).is_ok());
    let bad_arch = [
        NnArchitecture {
            hidden: vec![],
            ..arch.clone()
        },
        NnArchitecture {
            hidden: vec![10; 6],
            ..arch.clone()
        },
        NnArchitecture {
            hidden: vec![0],
            ..arch.clone()
        },
        NnArchitecture {
            hidden: vec![257],
            ..arch.clone()
        },
        NnArchitecture {
            activation: NnActivation::Softmax,
            ..arch.clone()
        },
        NnArchitecture {
            dropout: 0.9,
            ..arch.clone()
        },
        NnArchitecture {
            dropout: f32::NAN,
            ..arch.clone()
        },
        NnArchitecture {
            batch_size: 48,
            ..arch.clone()
        },
    ];
    for a in &bad_arch {
        assert!(
            matches!(
                validate(a, &training()),
                Err(CommandError::InvalidConfig { .. })
            ),
            "{a:?}"
        );
    }
    let bad_training = [
        NnTraining {
            learning_rate: 0.05,
            ..training()
        },
        NnTraining {
            epochs: 9,
            ..training()
        },
        NnTraining {
            epochs: 501,
            ..training()
        },
        NnTraining {
            validation_pct: 41,
            ..training()
        },
        NnTraining {
            patience: Some(4),
            ..training()
        },
        NnTraining {
            lr_factor: Some(0.95),
            ..training()
        },
        NnTraining {
            seed: 0,
            ..training()
        },
    ];
    for t in &bad_training {
        assert!(validate(&arch, t).is_err(), "{t:?}");
    }
    // The f32 of the selectors and sliders are accepted.
    let ok = NnTraining {
        learning_rate: 0.3,
        lr_factor: Some(0.1 + 0.2),
        patience: None,
        ..training()
    };
    assert!(validate(
        &NnArchitecture {
            dropout: 0.1 * 8.0,
            ..arch.clone()
        },
        &ok
    )
    .is_ok());
    assert_eq!(reduce_lr_patience(&training()), 5);
    assert_eq!(
        reduce_lr_patience(&NnTraining {
            patience: Some(5),
            ..training()
        }),
        2
    );
    assert_eq!(
        reduce_lr_patience(&NnTraining {
            patience: None,
            ..training()
        }),
        5
    );
}

/// D74: `red_entrenamiento.py.j2` with the controls renders the scripts that
/// `export_golden.py` executed (text exact) for the built-in datasets.
#[test]
fn training_script_matches_the_fixture() {
    let code = fixture("code/nn_training.json");
    let mut checked = 0;
    for (name, case) in code["expected"].as_object().unwrap() {
        let context = &case["context"];
        let Some(id) = context["dataset"].as_str().and_then(id_of) else {
            continue; // the CSV cases need a registered CSV
        };
        let summary = summary(id);
        let architecture: Vec<u16> = context["architecture"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u16)
            .collect();
        let arch = NnArchitecture {
            hidden: architecture[1..architecture.len() - 1].to_vec(),
            activation: serde_json::from_value(context["activation"].clone()).unwrap(),
            output_activation: serde_json::from_value(context["output_activation"].clone())
                .unwrap(),
            dropout: context["dropout_rate"].as_f64().unwrap() as f32,
            batch_size: context["batch_size"].as_u64().unwrap() as u16,
            optimizer: serde_json::from_value(context["optimizer"].clone()).unwrap(),
        };
        let training = NnTraining {
            learning_rate: context["learning_rate"].as_f64().unwrap() as f32,
            epochs: context["epochs"].as_u64().unwrap() as u16,
            validation_pct: (context["validation_split"].as_f64().unwrap() * 100.0).round() as u8,
            patience: context["patience"].as_u64().map(|p| p as u8),
            lr_factor: context["lr_factor"].as_f64().map(|f| f as f32),
            seed: 42,
        };
        let dataset = FixtureFacade::from_checkout().load_builtin(id).unwrap();
        let base = dataset.script_base(context["dataset_label"].as_str().unwrap());
        let ours = NnTrainingContext::new(base, &summary, &arch, &training).unwrap();
        assert_eq!(
            serde_json::to_value(&ours).unwrap(),
            *context,
            "{name}: context"
        );
        assert_eq!(
            ours.render().unwrap(),
            case["script"].as_str().unwrap(),
            "{name}"
        );
        checked += 1;
    }
    assert!(checked >= 4, "{checked}");
}

#[test]
fn epochs_are_batched_but_never_dropped() {
    let batcher = EpochBatcher::default();
    let t0 = Instant::now();
    let log = |epoch| EpochLog {
        epoch,
        loss: Some(1.0),
        metric: None,
        val_loss: None,
        val_metric: None,
        learning_rate: None,
    };
    let mut sent = Vec::new();
    let mut events = 0;
    // 500 epochs in 100 ms: at most 1 + 100/50 events, the last one included.
    for e in 1..=500u16 {
        let now = t0 + Duration::from_micros(u64::from(e) * 200);
        if let Some(batch) = batcher.push(log(e), e == 500, now) {
            events += 1;
            sent.extend(batch);
        }
    }
    assert!(events <= 4, "{events}");
    assert_eq!(sent.len(), 500);
    assert_eq!(sent.last().map(|l| l.epoch), Some(500));
    assert!(batcher.flush().is_empty());
    // A cancelled training flushes what was left.
    assert!(batcher
        .push(log(1), false, t0 + Duration::from_secs(1))
        .is_some());
    assert!(batcher
        .push(log(2), false, t0 + Duration::from_secs(1))
        .is_none());
    assert_eq!(batcher.flush().len(), 1);
    assert_eq!(finite_log(f64::NAN), None);
    assert_eq!(control_f64(0.1_f32), 0.1);
}
