//! Helpers shared by the parity tests (fixtures in `tests/parity/`).
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

pub mod lm;

use std::path::PathBuf;

use serde_json::Value;

/// `tests/parity/` at the repository root.
pub fn parity_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity")
}

/// Parses `tests/parity/<rel>`.
pub fn fixture(rel: &str) -> Value {
    let path = parity_dir().join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A JSON number, or NaN for `null`.
pub fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

pub fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(num)
        .collect()
}

pub fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|s| match s {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .collect()
}

pub fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an array: {v}"))
        .iter()
        .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}

/// Relative error with an absolute floor at 1 ulp-ish scale: |a-b| / max(|b|, tiny).
/// NaN == NaN counts as 0.
pub fn rel_err(a: f64, b: f64) -> f64 {
    if (a.is_nan() && b.is_nan()) || a == b {
        return 0.0;
    }
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

pub fn abs_err(a: f64, b: f64) -> f64 {
    if (a.is_nan() && b.is_nan()) || a == b {
        0.0
    } else {
        (a - b).abs()
    }
}

/// Tracks the largest error of a group of comparisons.
#[derive(Debug, Default, Clone, Copy)]
pub struct MaxErr {
    pub max: f64,
    pub n: usize,
}

impl MaxErr {
    pub fn add(&mut self, err: f64) {
        assert!(!err.is_nan(), "NaN mismatch");
        self.n += 1;
        if err > self.max {
            self.max = err;
        }
    }
}

/// Fixture slugs of the 12 built-in datasets, in selector order.
pub const SLUGS: [&str; 12] = [
    "iris",
    "wine",
    "breast_cancer",
    "digits",
    "diabetes",
    "diamonds",
    "mpg",
    "moons",
    "titanic",
    "tips",
    "california_housing",
    "penguins",
];
