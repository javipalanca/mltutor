//! Parity of the CSV loader with `tests/parity/csv/*.json` (exact), plus the D10
//! deviations: Windows-1252, `;` separator and original class names of text targets.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use base64::Engine;
use common::{fixture, nums, rel_err, strings, MaxErr};
use mltutor_core::csv::{read_csv, ColumnData, CsvNotice, CsvTargetStats, RawTable};
use mltutor_core::{CsvError, TaskChoice, TaskType};
use serde_json::Value;

const CASES: [&str; 14] = [
    "bool_target",
    "categorical_and_nan",
    "decimal_comma_quoted",
    "empty_file",
    "header_only",
    "int_target_many_values",
    "latin1_encoding",
    "numeric_regression",
    "one_class_per_row",
    "semicolon_separator",
    "seven_levels_100_rows",
    "single_column",
    "string_target",
    "utf8_bom",
];

fn input_bytes(fx: &Value) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(fx["input"]["csv_base64"].as_str().unwrap())
        .unwrap()
}

fn task_from_es(s: &str) -> TaskType {
    match s {
        "Clasificación" => TaskType::Classification,
        "Regresión" => TaskType::Regression,
        other => panic!("task {other}"),
    }
}

fn check_table(name: &str, table: &RawTable, exp: &Value, describe_err: &mut MaxErr) {
    let shape = exp["shape"].as_array().unwrap();
    assert_eq!(
        table.n_rows as u64,
        shape[0].as_u64().unwrap(),
        "{name} rows"
    );
    assert_eq!(
        table.n_cols() as u64,
        shape[1].as_u64().unwrap(),
        "{name} cols"
    );
    let names: Vec<String> = table.columns.iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, strings(&exp["columns"]), "{name} columns");
    for col in &table.columns {
        assert_eq!(
            col.data.kind().dtype_name(),
            exp["dtypes"][&col.name].as_str().unwrap(),
            "{name} {}",
            col.name
        );
    }
    assert_eq!(
        table.missing_values() as u64,
        exp["missing_values"].as_u64().unwrap(),
        "{name}"
    );

    let preview = table.preview().unwrap();
    assert_eq!(
        preview.default_target,
        exp["default_target"].as_str().unwrap()
    );
    assert_eq!(preview.head.rows.len(), table.n_rows.min(5));
    let info = &preview.target;
    assert_eq!(
        info.nunique as u64,
        exp["target_nunique"].as_u64().unwrap(),
        "{name} nunique"
    );
    assert_eq!(info.dtype, exp["target_dtype"].as_str().unwrap(), "{name}");
    assert_eq!(
        info.suggested_task,
        task_from_es(exp["suggested_task"].as_str().unwrap()),
        "{name} suggestion"
    );
    match exp["target_value_counts"].as_object() {
        Some(vc) => {
            let expected: Vec<(String, u32)> = vc
                .iter()
                .map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32))
                .collect();
            assert_eq!(
                info.value_counts.as_ref().unwrap(),
                &expected,
                "{name} value_counts"
            );
        }
        None => assert!(info.value_counts.is_none(), "{name}"),
    }

    let d = table.describe();
    let ed = &exp["describe"];
    assert_eq!(
        d.columns,
        strings(&ed["columns"]),
        "{name} describe columns"
    );
    let ev = nums(&ed["values"]);
    for (i, v) in d.values.iter().enumerate() {
        let e = rel_err(*v, ev[i]);
        describe_err.add(e);
        assert!(e == 0.0, "{name} describe[{i}] {v} vs {}", ev[i]);
    }

    let target = preview.default_target.clone();
    for (task_es, load) in exp["load_dataset_from_file"].as_object().unwrap() {
        let choice = match task_es.as_str() {
            "auto" => TaskChoice::Auto,
            "Clasificación" => TaskChoice::Classification,
            _ => TaskChoice::Regression,
        };
        let ctx = format!("{name} load {task_es}");
        assert!(
            load.get("error").is_none(),
            "{ctx}: Python failed, update the test"
        );
        let ds = table.to_dataset(&target, choice, "datos.csv").unwrap();
        assert_eq!(ds.label, "📄 datos.csv");
        assert_eq!(
            ds.task(),
            task_from_es(load["task_type"].as_str().unwrap()),
            "{ctx} task"
        );
        assert_eq!(ds.feature_columns, strings(&load["feature_names"]), "{ctx}");
        assert_eq!(
            ds.description,
            load["dataset_info"].as_str().unwrap(),
            "{ctx} info"
        );
        for (c, col) in ds.feature_columns.iter().enumerate() {
            assert_eq!(
                ds.feature_kinds[c].dtype_name(),
                load["X_dtypes"][col].as_str().unwrap(),
                "{ctx} {col}"
            );
        }
        let ex = nums(&load["X"]);
        let got: Vec<f64> = ds.features.iter().copied().collect();
        assert_eq!(got.len(), ex.len());
        for (a, b) in got.iter().zip(&ex) {
            assert!(a == b || (a.is_nan() && b.is_nan()), "{ctx} X {a} vs {b}");
        }
        let ey = nums(&load["y"]);
        for (a, b) in ds.target_numeric().iter().zip(&ey) {
            assert!(a == b || (a.is_nan() && b.is_nan()), "{ctx} y {a} vs {b}");
        }
        match load["class_names"].as_array() {
            None => assert!(ds.class_names().is_none(), "{ctx}"),
            Some(_) => {
                let got = ds.class_names().unwrap();
                let col = table.column(&target).unwrap();
                if let ColumnData::Text(values) = &col.data {
                    // D10: original names (sorted like LabelEncoder) instead of "Clase i".
                    let mut names: Vec<String> = values
                        .iter()
                        .map(|v| v.unwrap_or("nan").to_owned())
                        .collect();
                    names.sort();
                    names.dedup();
                    assert_eq!(got, names.as_slice(), "{ctx} class names (D10)");
                    assert_eq!(got.len(), load["class_names"].as_array().unwrap().len());
                } else {
                    assert_eq!(
                        got,
                        strings(&load["class_names"]).as_slice(),
                        "{ctx} class names"
                    );
                }
            }
        }
    }
}

#[test]
fn csv_cases_match_python() {
    let mut describe_err = MaxErr::default();
    for name in CASES {
        let fx = fixture(&format!("csv/{name}.json"));
        assert_eq!(fx["meta"]["tolerance"]["values"], "exact");
        let exp = &fx["expected"];
        let result = read_csv(&input_bytes(&fx));
        match name {
            // pandas: "No columns to parse from file" → "❌ Error al procesar el archivo";
            // Rust maps it to CsvError::Empty ("❌ El archivo está vacío", §4).
            "empty_file" => {
                assert!(exp.get("read_error").is_some());
                assert_eq!(result.err(), Some(CsvError::Empty));
            }
            "header_only" => {
                assert_eq!(exp["validation_error"], "❌ El archivo está vacío");
                assert_eq!(result.err(), Some(CsvError::Empty));
            }
            "single_column" => {
                assert!(exp["validation_error"]
                    .as_str()
                    .unwrap()
                    .contains("al menos 2 columnas"));
                assert_eq!(result.err(), Some(CsvError::TooFewColumns { found: 1 }));
            }
            // D10: Python fails with UnicodeDecodeError; Rust reads Windows-1252.
            "latin1_encoding" => {
                assert_eq!(exp["read_error"]["type"], "UnicodeDecodeError");
                let t = result.unwrap();
                assert_eq!(t.notices, vec![CsvNotice::Windows1252]);
                assert_eq!(t.columns[1].name, "año");
                assert_eq!(
                    t.columns[0].data,
                    ColumnData::Text(vec![Some("Peña"), Some("Núñez")].into())
                );
                assert_eq!(t.columns[2].data, ColumnData::Float(vec![1.5, 2.5]));
            }
            // D10: Python sees one column; Rust detects the `;` separator.
            "semicolon_separator" => {
                assert!(exp["validation_error"]
                    .as_str()
                    .unwrap()
                    .contains("al menos 2 columnas"));
                let t = result.unwrap();
                assert_eq!(t.separator, ';');
                assert_eq!(t.notices, vec![CsvNotice::Separator { separator: ';' }]);
                assert_eq!(t.n_cols(), 3);
                assert_eq!(t.columns[2].data, ColumnData::Int(vec![3, 6]));
            }
            _ => {
                let t = result.unwrap_or_else(|e| panic!("{name}: {e}"));
                check_table(name, &t, exp, &mut describe_err);
            }
        }
    }
    println!(
        "csv describe: max rel error {:e} over {} values (exact)",
        describe_err.max, describe_err.n
    );
}

#[test]
fn decimal_comma_is_accepted_with_semicolons() {
    let t = read_csv("a;b;c\n\"1,5\";2;x\n2,5;3;y\n".as_bytes()).unwrap();
    assert_eq!(t.columns[0].data, ColumnData::Float(vec![1.5, 2.5]));
    assert!(t.notices.contains(&CsvNotice::DecimalComma {
        columns: vec!["a".into()]
    }));
    // With commas as separator "1,5" stays text, as in Python.
    let t = read_csv(b"a,b,c\n\"1,5\",2,x\n\"2,5\",3,y\n").unwrap();
    assert_eq!(t.columns[0].data.kind().dtype_name(), "object");
}

#[test]
fn target_stats_follow_the_task_choice() {
    let fx = fixture("csv/numeric_regression.json");
    let t = read_csv(&input_bytes(&fx)).unwrap();
    let CsvTargetStats::Numeric {
        hist_counts, mean, ..
    } = t.target_stats("y", TaskChoice::Auto).unwrap()
    else {
        panic!("numeric")
    };
    assert_eq!(hist_counts.iter().sum::<u32>(), 40);
    let d = &fx["expected"]["describe"];
    // describe is stats × columns, row-major: mean of `y` is [1, 2].
    assert_eq!(mean, nums(&d["values"])[5]);
    assert!(matches!(
        t.target_stats("y", TaskChoice::Classification).unwrap(),
        CsvTargetStats::Classes { .. }
    ));
    assert_eq!(
        t.target_stats("nope", TaskChoice::Auto).err(),
        Some(CsvError::UnknownColumn {
            column: "nope".into()
        })
    );
    assert!(t.to_dataset("nope", TaskChoice::Auto, "x.csv").is_err());
}

#[test]
fn malformed_rows_and_limits() {
    assert!(matches!(
        read_csv(b"a,b\n1\n"),
        Err(CsvError::Malformed {
            line: 2,
            found: 1,
            expected: 2
        })
    ));
    assert_eq!(read_csv(b"\n\n").err(), Some(CsvError::Empty));
    // Blank lines are skipped like pandas.
    let t = read_csv(b"a,b\n\n1,2\n\n3,4\n").unwrap();
    assert_eq!(t.n_rows, 2);
    // Duplicate names are mangled like pandas.
    let t = read_csv(b"a,a,b\n1,2,3\n").unwrap();
    let names: Vec<&str> = t.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, vec!["a", "a.1", "b"]);
}
