//! D7: the split equals scikit-learn's `train_test_split(test_size=0.3, random_state=42)`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{fixture, nums};
use mltutor_core::dataset::{load_builtin, DatasetId};
use mltutor_core::sampling::RandomState;
use mltutor_core::split::train_test_split;

fn usizes(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn iris_split_is_exact() {
    let fx = fixture("split/iris.json");
    let exp = &fx["expected"];
    let ds = load_builtin(DatasetId::Iris).unwrap();
    let n = ds.n_rows();

    assert_eq!(
        RandomState::new(42).permutation(n),
        usizes(&exp["permutation"])
    );
    let split = train_test_split(n, 0.3, 42).unwrap();
    assert_eq!(split.train.len() as u64, exp["n_train"].as_u64().unwrap());
    assert_eq!(split.test.len() as u64, exp["n_test"].as_u64().unwrap());
    assert_eq!(split.train, usizes(&exp["train_indices"]));
    assert_eq!(split.test, usizes(&exp["test_indices"]));

    let y = ds.target_numeric();
    for (set, key) in [
        (&split.train, "train_class_counts"),
        (&split.test, "test_class_counts"),
    ] {
        for (class, count) in exp[key].as_object().unwrap() {
            let c: f64 = class.parse().unwrap();
            let n = set.iter().filter(|&&i| y[i] == c).count() as u64;
            assert_eq!(n, count.as_u64().unwrap(), "{key} {class}");
        }
    }
    for (k, row) in exp["first_test_rows"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_eq!(ds.features.row(split.test[k]).to_vec(), nums(row));
    }
}
