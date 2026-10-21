//! Property tests: the CSV loader and the exploration never panic, whatever the
//! input (PLAN stability criterion; fase1-shell-datos.md §8.1 item 8).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use mltutor_core::csv::{read_csv, RawTable};
use mltutor_core::explore::{self, DiagKind};
use mltutor_core::TaskChoice;
use proptest::prelude::*;

/// Runs everything the UI can call on a parsed table.
fn exercise(table: &RawTable) {
    let preview = table.preview().unwrap();
    assert_eq!(preview.n_cols, table.n_cols());
    let _ = table.describe();
    let _ = table.column_info();
    for col in &table.columns {
        let _ = table.target_info(&col.name).unwrap();
        for choice in [
            TaskChoice::Auto,
            TaskChoice::Classification,
            TaskChoice::Regression,
        ] {
            let _ = table.target_stats(&col.name, choice).unwrap();
            let ds = table.to_dataset(&col.name, choice, "f.csv").unwrap();
            assert_eq!(ds.n_rows(), table.n_rows);
            let _ = explore::preview(&ds, 10);
            let _ = explore::describe(&ds);
            let _ = explore::target_distribution(&ds);
            let _ = explore::correlation_matrix(&ds);
            let defaults = explore::pairplot_defaults(&ds);
            if !defaults.default_selected.is_empty() {
                let _ =
                    explore::pairplot_data(&ds, &defaults.default_selected, DiagKind::Kde).unwrap();
                let _ = explore::pairplot_data(&ds, &defaults.default_selected, DiagKind::Hist)
                    .unwrap();
            }
        }
    }
}

/// CSV-looking text: separators, quotes, numbers, NA strings, booleans, accents.
fn csvish() -> impl Strategy<Value = String> {
    let token = prop_oneof![
        Just(",".to_owned()),
        Just(";".to_owned()),
        Just("\t".to_owned()),
        Just("\n".to_owned()),
        Just("\r\n".to_owned()),
        Just("\"".to_owned()),
        Just("NaN".to_owned()),
        Just("".to_owned()),
        Just("True".to_owned()),
        Just("false".to_owned()),
        Just("inf".to_owned()),
        Just("-0".to_owned()),
        Just("1e308".to_owned()),
        Just("1e999".to_owned()),
        Just("ñ".to_owned()),
        "-?[0-9]{1,6}(\\.[0-9]{0,4})?",
        "-?[0-9]{1,3},[0-9]{1,3}",
        "[a-zA-Z ]{1,5}",
    ];
    prop::collection::vec(token, 0..80).prop_map(|t| t.concat())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn random_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        if let Ok(table) = read_csv(&bytes) {
            exercise(&table);
        }
    }

    #[test]
    fn csv_like_text_never_panics(text in csvish()) {
        if let Ok(table) = read_csv(text.as_bytes()) {
            exercise(&table);
        }
    }

    #[test]
    fn rectangular_tables_always_parse(
        header in prop::collection::vec("[a-z]{0,3}", 2..5),
        cells in prop::collection::vec(prop::collection::vec("-?[0-9]{1,4}(\\.[0-9]{1,2})?|[a-c]|NA|", 5), 1..40),
    ) {
        let width = header.len();
        let mut text = header.join(",");
        text.push('\n');
        for row in &cells {
            text.push_str(&row[..width].join(","));
            text.push('\n');
        }
        let table = read_csv(text.as_bytes()).unwrap();
        prop_assert_eq!(table.n_cols(), width);
        prop_assert_eq!(table.n_rows, cells.len());
        exercise(&table);
    }
}
