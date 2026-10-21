//! `f64` fields that may be NaN or infinite, serialised so that they survive JSON.
//!
//! serde_json writes NaN and ±∞ as `null`, which cannot be read back into an `f64`.
//! A diverged training (fase5-redes.md K1) has exactly those values in its history,
//! so [`crate::nn_data::EpochLog`] uses this module: finite values stay numbers and
//! the others become the strings `"nan"`, `"inf"` and `"-inf"` (the convention of the
//! parity fixtures, `v1.0.0:tools/golden_nn.py::_clean`). Reading accepts both forms, and
//! also `null` as NaN.
//!
//! ```
//! #[derive(serde::Serialize, serde::Deserialize)]
//! struct Log {
//!     #[serde(with = "mltutor_core::serde_nonfinite")]
//!     loss: f64,
//! }
//! let text = serde_json::to_string(&Log { loss: f64::NAN }).unwrap();
//! assert_eq!(text, r#"{"loss":"nan"}"#);
//! let back: Log = serde_json::from_str(&text).unwrap();
//! assert!(back.loss.is_nan());
//! ```

use serde::de::{self, Visitor};
use serde::{Deserializer, Serializer};

/// The string of a non-finite value (`None` when it is finite).
fn label(v: f64) -> Option<&'static str> {
    if v.is_nan() {
        Some("nan")
    } else if v == f64::INFINITY {
        Some("inf")
    } else if v == f64::NEG_INFINITY {
        Some("-inf")
    } else {
        None
    }
}

/// Serialises `v` as a number, or as `"nan"`/`"inf"`/`"-inf"`.
pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
    match label(*v) {
        Some(text) => s.serialize_str(text),
        None => s.serialize_f64(*v),
    }
}

struct F64Visitor;

impl Visitor<'_> for F64Visitor {
    type Value = f64;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a number or \"nan\", \"inf\", \"-inf\"")
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<f64, E> {
        Ok(v)
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<f64, E> {
        Ok(v as f64)
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<f64, E> {
        Ok(v as f64)
    }

    fn visit_unit<E: de::Error>(self) -> Result<f64, E> {
        Ok(f64::NAN)
    }

    fn visit_none<E: de::Error>(self) -> Result<f64, E> {
        Ok(f64::NAN)
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<f64, E> {
        match v {
            "nan" | "NaN" => Ok(f64::NAN),
            "inf" | "Infinity" => Ok(f64::INFINITY),
            "-inf" | "-Infinity" => Ok(f64::NEG_INFINITY),
            other => other
                .parse()
                .map_err(|_| E::invalid_value(de::Unexpected::Str(other), &self)),
        }
    }
}

/// Reads a number, `"nan"`/`"inf"`/`"-inf"` or `null` (NaN).
pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    d.deserialize_any(F64Visitor)
}

/// The same for `Option<f64>` (`None` is `null`; use it with `#[serde(default)]`).
pub mod option {
    use serde::{Deserialize, Deserializer, Serializer};

    /// `None` → `null`; `Some(v)` → like [`super::serialize`].
    pub fn serialize<S: Serializer>(v: &Option<f64>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(v) => super::serialize(v, s),
            None => s.serialize_none(),
        }
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Value(#[serde(deserialize_with = "super::deserialize")] f64),
    }

    /// `null` → `None`; otherwise like [`super::deserialize`].
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
        Ok(Option::<Repr>::deserialize(d)?.map(|Repr::Value(v)| v))
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct Row {
        #[serde(with = "super")]
        a: f64,
        #[serde(default, with = "super::option")]
        b: Option<f64>,
    }

    #[test]
    fn round_trip_of_every_kind() {
        for (a, b) in [
            (1.5, Some(0.25)),
            (f64::NAN, None),
            (f64::INFINITY, Some(f64::NEG_INFINITY)),
            (-0.0, Some(f64::NAN)),
        ] {
            let text = serde_json::to_string(&Row { a, b }).unwrap_or_default();
            let back: Row = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(back.a.to_bits(), a.to_bits(), "{text}");
            match (back.b, b) {
                (Some(x), Some(y)) => assert!(x == y || (x.is_nan() && y.is_nan()), "{text}"),
                (None, None) => {}
                other => panic!("{text}: {other:?}"),
            }
        }
        let r: Row = serde_json::from_str(r#"{"a": null}"#).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.a.is_nan() && r.b.is_none());
        let r: Row =
            serde_json::from_str(r#"{"a": "-inf", "b": 3}"#).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.a, f64::NEG_INFINITY);
        assert_eq!(r.b, Some(3.0));
        assert!(serde_json::from_str::<Row>(r#"{"a": "abc"}"#).is_err());
    }
}
