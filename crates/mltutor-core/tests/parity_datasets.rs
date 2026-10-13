//! Parity of the embedded datasets with `tests/parity/datasets/*.json` (exact).
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{fixture, num, nums, strings, MaxErr, SLUGS};
use mltutor_core::dataset::{self, make_moons, moons_dataset, DatasetId, Target, TaskType};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn every_dataset_matches_its_fixture_exactly() {
    for slug in SLUGS {
        let fx = fixture(&format!("datasets/{slug}.json"));
        let exp = &fx["expected"];
        let id = DatasetId::from_slug(slug).unwrap();

        // Bytes embedded with include_bytes! are the golden CSV.
        let csv = id.csv_bytes();
        assert_eq!(
            hex(&Sha256::digest(csv)),
            exp["csv_sha256"].as_str().unwrap(),
            "{slug} csv_sha256"
        );
        assert_eq!(
            csv.len() as u64,
            exp["csv_bytes"].as_u64().unwrap(),
            "{slug} csv_bytes"
        );

        let ds = dataset::load_builtin(id).unwrap();
        let shape = exp["shape"].as_array().unwrap();
        assert_eq!(
            ds.n_rows() as u64,
            shape[0].as_u64().unwrap(),
            "{slug} rows"
        );
        assert_eq!(
            ds.n_features() as u64,
            shape[1].as_u64().unwrap(),
            "{slug} cols"
        );
        assert_eq!(
            ds.feature_columns,
            strings(&exp["columns"]),
            "{slug} columns"
        );
        assert_eq!(
            ds.feature_display,
            strings(&exp["feature_names"]),
            "{slug} feature_names"
        );
        let labels: Vec<String> = ds.label_columns.iter().map(|c| c.name.clone()).collect();
        assert_eq!(
            labels,
            strings(&exp["label_columns"]),
            "{slug} label_columns"
        );
        assert_eq!(
            ds.target_column,
            exp["target_name"].as_str().unwrap(),
            "{slug} target"
        );
        // dataset_info is a dict for sklearn-style datasets and a plain string for the rest.
        let info = &exp["dataset_info"];
        let description = info["description"]
            .as_str()
            .or_else(|| info.as_str())
            .unwrap();
        assert_eq!(ds.description, description, "{slug} description");
        assert_eq!(
            ds.task().label_es(),
            exp["task_type"].as_str().unwrap(),
            "{slug} task"
        );
        match exp["class_names"].as_array() {
            Some(_) => assert_eq!(
                ds.class_names().unwrap(),
                strings(&exp["class_names"]).as_slice()
            ),
            None => assert!(ds.class_names().is_none(), "{slug}"),
        }
        for (i, col) in ds.feature_columns.iter().enumerate() {
            assert_eq!(
                ds.feature_kinds[i].dtype_name(),
                exp["dtypes"][col].as_str().unwrap(),
                "{slug} dtype of {col}"
            );
        }

        // data_sha256_f64: row-major LE f64 of the features then the target.
        let y = ds.target_numeric();
        let mut hasher = Sha256::new();
        for (row, target) in ds.features.rows().into_iter().zip(&y) {
            for v in row {
                hasher.update(v.to_le_bytes());
            }
            hasher.update(target.to_le_bytes());
        }
        assert_eq!(
            hex(&hasher.finalize()),
            exp["data_sha256_f64"].as_str().unwrap(),
            "{slug} data hash"
        );

        // head
        for (r, row) in exp["head"].as_array().unwrap().iter().enumerate() {
            let feats = nums(&row["features"]);
            assert_eq!(ds.features.row(r).to_vec(), feats, "{slug} head row {r}");
            assert_eq!(y[r], num(&row["target"]), "{slug} head target {r}");
        }

        // target_summary
        let summary = &exp["target_summary"];
        match &ds.target {
            Target::Classes { codes, .. } => {
                for (code, count) in summary["class_counts"].as_object().unwrap() {
                    let code: u32 = code.parse().unwrap();
                    let n = codes.iter().filter(|&&c| c == code).count() as u64;
                    assert_eq!(n, count.as_u64().unwrap(), "{slug} class {code}");
                }
            }
            Target::Values { values } => {
                let d = mltutor_core::stats::describe_column(values);
                assert_eq!(d[3], num(&summary["min"]), "{slug} min");
                assert_eq!(d[7], num(&summary["max"]), "{slug} max");
                assert_eq!(d[5], num(&summary["median"]), "{slug} median");
                assert!(
                    (d[1] - num(&summary["mean"])).abs() <= 1e-12 * d[1].abs(),
                    "{slug} mean"
                );
                assert!(
                    (d[2] - num(&summary["std_ddof1"])).abs() <= 1e-12 * d[2].abs(),
                    "{slug} std"
                );
            }
        }
    }
}

#[test]
fn metadata_matches_fixtures_and_selector_order() {
    let entries = dataset::list_builtin().unwrap();
    assert_eq!(entries.len(), 12);
    for (entry, slug) in entries.iter().zip(SLUGS) {
        let fx = fixture(&format!("datasets/{slug}.json"));
        assert_eq!(entry.id.slug(), slug);
        assert_eq!(entry.label, fx["input"]["app_label"].as_str().unwrap());
        let meta = entry.id.metadata().unwrap();
        // The friendly names of the TOML are the fixture's feature_names, in order.
        assert_eq!(
            meta.display_names(),
            strings(&fx["expected"]["feature_names"]),
            "{slug}"
        );
        assert_eq!(
            meta.samples as u64,
            fx["expected"]["shape"][0].as_u64().unwrap()
        );
        let task = if fx["expected"]["task_type"] == "Clasificación" {
            TaskType::Classification
        } else {
            TaskType::Regression
        };
        assert_eq!(entry.task, task, "{slug}");
        assert_eq!(entry.synthetic, slug == "moons");
    }
    // Digits, Diabetes, Diamonds and MPG have no line under the selector.
    let no_blurb: Vec<&str> = entries
        .iter()
        .filter(|e| e.blurb.is_none())
        .map(|e| e.id.slug())
        .collect();
    assert_eq!(no_blurb, vec!["digits", "diabetes", "diamonds", "mpg"]);
}

#[test]
fn mpg_keeps_the_car_name_as_a_label_column() {
    let ds = dataset::load_builtin(DatasetId::Mpg).unwrap();
    assert_eq!(ds.label_columns[0].values[0], "chevrolet chevelle malibu");
    assert_eq!(ds.summary().n_features + 1, 8); // "📊 Columnas" (D8)
}

#[test]
fn make_moons_reproduces_the_embedded_csv() {
    let embedded = dataset::load_builtin(DatasetId::Moons).unwrap();
    let (x, y) = make_moons(500, 0.2, 42);
    let mut err = MaxErr::default();
    for r in 0..500 {
        for c in 0..2 {
            err.add((x[[r, c]] - embedded.features[[r, c]]).abs());
        }
    }
    let Target::Classes { codes, .. } = &embedded.target else {
        panic!("moons is classification")
    };
    assert_eq!(&y, codes);
    println!(
        "make_moons vs moons.csv: max abs error {:e} over {} values",
        err.max, err.n
    );
    // Same algorithm and RNG; only libm cos/sin/log may differ by an ulp.
    assert!(err.max <= 1e-15, "{}", err.max);

    let generated = moons_dataset(42).unwrap();
    assert_eq!(generated.feature_columns, embedded.feature_columns);
    assert_eq!(generated.class_names(), embedded.class_names());
    assert_ne!(moons_dataset(7).unwrap().features, generated.features);
}
