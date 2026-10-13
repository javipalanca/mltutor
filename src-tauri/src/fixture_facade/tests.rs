//! Checks the fixture facade against `tests/parity`: the computed views of every built-in
//! dataset must match the golden stats, and the CSV reader the 14 CSV cases.

use super::*;
use crate::core_facade::CsvDraft;
use crate::ipc::{CsvTargetStats, DiagPanel, Hue, TaskChoice};

fn facade() -> FixtureFacade {
    FixtureFacade::from_checkout()
}

fn load(id: DatasetId) -> FixtureDataset {
    let slug = id.slug();
    let f = facade();
    let meta = f.read_json(&format!("datasets/{slug}.json")).unwrap();
    let stats = f.read_json(&format!("stats/{slug}.json")).unwrap();
    let csv = std::fs::read_to_string(f.root.join(format!("datasets/data/{slug}.csv"))).unwrap();
    FixtureDataset::parse(id, slug.into(), &meta["expected"], stats, &csv).unwrap()
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= rel * b.abs().max(1.0)
}

fn assert_values(actual: &[Option<f64>], expected: &[f64], rel: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let a = a.unwrap_or(f64::NAN);
        assert!(close(a, *e, rel), "{what}[{i}]: {a} != {e}");
    }
}

#[test]
fn computed_views_match_the_golden_stats() {
    for id in DatasetId::ALL {
        let dataset = load(id);
        let slug = id.slug();
        let computed_describe = dataset.frame.describe();
        let golden_describe = dataset.describe();
        assert_eq!(computed_describe.columns, golden_describe.columns, "{slug}");
        assert_eq!(
            computed_describe.excluded_columns, golden_describe.excluded_columns,
            "{slug}"
        );
        let golden: Vec<f64> = golden_describe
            .values
            .values
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();
        assert_values(&computed_describe.values.values, &golden, 1e-9, slug);

        let computed_corr = dataset.frame.correlation();
        let golden_corr = dataset.correlation();
        assert_eq!(computed_corr.labels, golden_corr.labels, "{slug}");
        let golden: Vec<f64> = golden_corr
            .values
            .values
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();
        assert_values(&computed_corr.values.values, &golden, 1e-9, slug);

        match (
            dataset.frame.target_distribution(),
            dataset.target_distribution(),
        ) {
            (
                TargetDistribution::Bar { labels, counts, .. },
                TargetDistribution::Bar {
                    labels: l2,
                    counts: c2,
                    ..
                },
            ) => {
                assert_eq!((labels, counts), (l2, c2), "{slug}");
            }
            (
                TargetDistribution::Histogram { edges, counts, .. },
                TargetDistribution::Histogram {
                    edges: e2,
                    counts: c2,
                    ..
                },
            ) => {
                assert_eq!(counts, c2, "{slug}");
                let edges: Vec<Option<f64>> = edges.into_iter().map(Some).collect();
                assert_values(&edges, &e2, 1e-12, slug);
            }
            _ => panic!("{slug}: distinto tipo de distribución"),
        }

        let preview = dataset.frame.preview(10);
        assert_eq!(preview.columns, dataset.preview(10).columns, "{slug}");
    }
}

#[test]
fn pairplot_diagonals_match_seaborn() {
    for id in DatasetId::ALL {
        let dataset = load(id);
        let slug = id.slug();
        let expected = &dataset.stats["expected"]["pairplot"];
        let features = strings(&expected["default_selected"]);
        let kde = dataset.pairplot(&features, DiagKind::Kde).unwrap();
        assert_eq!(
            kde.sampled,
            expected["sampled"].as_bool().unwrap(),
            "{slug}"
        );
        assert_eq!(
            u64::from(kde.n_points),
            expected["n_points"].as_u64().unwrap()
        );
        if let Hue::Categorical { levels, .. } = &kde.hue {
            assert_eq!(levels, &strings(&expected["hue_order"]), "{slug}");
        }
        for (feature, panel) in features.iter().zip(&kde.diagonal) {
            let DiagPanel::Kde { curves } = panel else {
                panic!("{slug}: se esperaba KDE");
            };
            for curve in curves {
                let golden = &expected["kde"][feature][&curve.level];
                match golden["density"].as_array() {
                    None => assert!(curve.density.is_empty(), "{slug}/{feature}"),
                    Some(_) => {
                        let what = format!("{slug}/{feature}/{}", curve.level);
                        assert!(
                            close(
                                curve.bandwidth.unwrap(),
                                golden["bandwidth"].as_f64().unwrap(),
                                1e-9
                            ),
                            "{what}"
                        );
                        let density: Vec<Option<f64>> =
                            curve.density.iter().copied().map(Some).collect();
                        assert_values(&density, &floats(&golden["density"]), 1e-7, &what);
                    }
                }
            }
        }
        let hist = dataset.pairplot(&features, DiagKind::Hist).unwrap();
        for (feature, panel) in features.iter().zip(&hist.diagonal) {
            let DiagPanel::Hist { edges, series } = panel else {
                panic!("{slug}: se esperaba histograma");
            };
            let golden = &expected["hist"][feature];
            let edges: Vec<Option<f64>> = edges.iter().copied().map(Some).collect();
            assert_values(&edges, &floats(&golden["edges"]), 1e-9, slug);
            for s in series {
                assert_eq!(
                    s.counts,
                    counts(&golden["counts"][&s.level]),
                    "{slug}/{feature}"
                );
            }
        }
    }
}

#[test]
fn large_datasets_use_the_fixture_sample() {
    let housing = load(DatasetId::CaliforniaHousing);
    let features = vec!["Ingresos_Medios".to_owned(), "Edad_Casa".to_owned()];
    let plot = housing.pairplot(&features, DiagKind::Kde).unwrap();
    assert!(plot.sampled && plot.high_quality);
    assert_eq!((plot.n_points, plot.n_total), (2000, 20640));
    assert!(plot.title.ends_with(" (Muestra: 2000/20640 puntos)"));
    assert!(matches!(plot.hue, Hue::Continuous { .. }));
}

#[test]
fn mpg_keeps_the_car_name_as_tooltip_and_preview_column() {
    let mpg = load(DatasetId::Mpg);
    assert_eq!(mpg.summary.n_features, 7);
    assert_eq!(mpg.summary.label_columns, ["name"]);
    let plot = mpg
        .pairplot(&["cylinders".to_owned()], DiagKind::Hist)
        .unwrap();
    assert_eq!(
        plot.tooltips.unwrap()[0],
        "chevrolet chevelle malibu".to_owned()
    );
    assert_eq!(mpg.preview(10).columns[7], "name");
}

#[test]
fn csv_cases_match_python() {
    let dir = facade().root.join("csv");
    let mut cases: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().collect();
    cases.sort_by_key(|e| e.path());
    assert_eq!(cases.len(), 14);
    for case in cases {
        let name = case.file_name().to_string_lossy().into_owned();
        let json: Value =
            serde_json::from_str(&std::fs::read_to_string(case.path()).unwrap()).unwrap();
        let text = json["input"]["csv_text"].as_str().unwrap_or("");
        let bytes = base64_decode(json["input"]["csv_base64"].as_str().unwrap_or(""))
            .unwrap_or_else(|| text.as_bytes().to_vec());
        let expected = &json["expected"];
        let result = csv::read(&name, &bytes);
        if let Some(message) = expected["validation_error"].as_str() {
            match name.as_str() {
                // D10: the separator is detected, so this file is read correctly.
                "semicolon_separator.json" => {
                    let table = result.unwrap();
                    assert!(table.columns.len() >= 2);
                    assert!(table.notices[0].contains(';'));
                }
                _ => assert_eq!(result.unwrap_err().message(), message, "{name}"),
            }
            continue;
        }
        if expected.get("read_error").is_some() {
            match name.as_str() {
                "empty_file.json" => {
                    assert!(matches!(result, Err(CommandError::CsvEmpty { .. })));
                }
                // D10: Windows-1252 fallback with a notice.
                "latin1_encoding.json" => {
                    let table = result.unwrap();
                    assert!(table.notices[0].contains("Windows-1252"));
                }
                other => panic!("caso inesperado {other}"),
            }
            continue;
        }
        let table = result.unwrap_or_else(|e| panic!("{name}: {e}"));
        let preview = table.preview();
        assert_eq!(
            [u64::from(preview.n_rows), u64::from(preview.n_cols)],
            [
                expected["shape"][0].as_u64().unwrap(),
                expected["shape"][1].as_u64().unwrap()
            ],
            "{name}"
        );
        assert_eq!(
            preview.head.columns,
            strings(&expected["columns"]),
            "{name}"
        );
        for column in &preview.columns {
            assert_eq!(
                column.dtype, expected["dtypes"][&column.name],
                "{name}/{}",
                column.name
            );
        }
        assert_eq!(
            u64::from(preview.missing_values),
            expected["missing_values"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(preview.default_target, expected["default_target"], "{name}");
        let target = &preview.target;
        assert_eq!(
            u64::from(target.nunique),
            expected["target_nunique"].as_u64().unwrap()
        );
        assert_eq!(target.dtype, expected["target_dtype"], "{name}");
        let suggested = if expected["suggested_task"] == "Clasificación" {
            TaskType::Classification
        } else {
            TaskType::Regression
        };
        assert_eq!(target.suggested_task, suggested, "{name}");
        if let Some(counts) = expected["target_value_counts"].as_object() {
            let mut ours: Vec<(String, u32)> = target.value_counts.clone().unwrap();
            let mut theirs: Vec<(String, u32)> = counts
                .iter()
                .map(|(k, v)| (k.clone(), u32::try_from(v.as_u64().unwrap()).unwrap()))
                .collect();
            ours.sort();
            theirs.sort();
            assert_eq!(ours, theirs, "{name}");
        }
        let describe = table.describe();
        assert_eq!(
            describe.columns,
            strings(&expected["describe"]["columns"]),
            "{name}"
        );
        assert_values(
            &describe.values.values,
            &floats(&expected["describe"]["values"]),
            1e-9,
            &name,
        );
        for (choice, key) in [
            (TaskChoice::Auto, "auto"),
            (TaskChoice::Classification, "Clasificación"),
            (TaskChoice::Regression, "Regresión"),
        ] {
            let golden = &expected["load_dataset_from_file"][key];
            let done = table.finalize(&preview.default_target, choice).unwrap();
            let task = if golden["task_type"] == "Clasificación" {
                TaskType::Classification
            } else {
                TaskType::Regression
            };
            assert_eq!(done.resolved_task, task, "{name}/{key}");
            let summary = done.dataset.summary();
            assert_eq!(summary.feature_columns, strings(&golden["feature_names"]));
            assert_eq!(summary.description, golden["dataset_info"], "{name}/{key}");
            if task == TaskType::Classification {
                let golden_names = strings(&golden["class_names"]);
                let ours = summary.class_names.unwrap();
                assert_eq!(ours.len(), golden_names.len(), "{name}/{key}");
                if target.dtype != "object" {
                    // D10 keeps original text names; numbers and bools follow Python.
                    assert_eq!(ours, golden_names, "{name}/{key}");
                }
            }
            if let CsvTargetStats::Numeric { min, max, .. } = done.target_stats {
                assert!(min <= max);
            }
        }
    }
}

fn base64_decode(text: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let (mut buffer, mut bits) = (0u32, 0u32);
    for byte in text
        .bytes()
        .filter(|b| *b != b'=' && !b.is_ascii_whitespace())
    {
        let value = ALPHABET.iter().position(|a| *a == byte)?;
        buffer = (buffer << 6) | u32::try_from(value).ok()?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(u8::try_from((buffer >> bits) & 0xFF).ok()?);
        }
    }
    Some(out)
}
