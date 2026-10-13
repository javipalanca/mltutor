//! Python-compatible text formatting of numbers.
//!
//! The Python version turns values into labels with `str()`/`repr()` (class names
//! of a float target, `value_counts` keys). Rust's `Display` differs (`3` vs
//! `3.0`, `0.00001` vs `1e-05`), so labels are produced here.

/// `repr(float)` of CPython 3 (shortest round-trip digits).
///
/// Scientific notation is used when the decimal exponent is below -4 or at
/// least 16, as in CPython: `1e-05`, `0.0001`, `1e+16`, `3.0`.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    // `{:e}` gives the shortest round-trip digits: "-1.2345e-7", "3e0".
    let sci = format!("{x:e}");
    let (mantissa, exp) = match sci.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (sci.as_str(), 0),
    };
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let n_digits = i32::try_from(digits.len()).unwrap_or(i32::MAX);

    if !(-4..16).contains(&exp) {
        let (head, tail) = digits.split_at(1);
        let frac = if tail.is_empty() {
            String::new()
        } else {
            format!(".{tail}")
        };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{head}{frac}e{exp_sign}{:02}", exp.unsigned_abs());
    }
    if exp >= 0 {
        let int_len = exp + 1;
        if n_digits <= int_len {
            let zeros = "0".repeat(usize::try_from(int_len - n_digits).unwrap_or(0));
            format!("{sign}{digits}{zeros}.0")
        } else {
            let (int_part, frac) = digits.split_at(usize::try_from(int_len).unwrap_or(0));
            format!("{sign}{int_part}.{frac}")
        }
    } else {
        let zeros = "0".repeat(usize::try_from(-exp - 1).unwrap_or(0));
        format!("{sign}0.{zeros}{digits}")
    }
}

#[cfg(test)]
mod tests {
    use super::py_float_repr;

    #[test]
    fn matches_cpython_repr() {
        let cases = [
            (3.0, "3.0"),
            (-2.0069, "-2.0069"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123.456, "123.456"),
            (1.5e-7, "1.5e-07"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (2.5e20, "2.5e+20"),
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "-inf"),
            (0.30000000000000004, "0.30000000000000004"),
        ];
        for (x, expected) in cases {
            assert_eq!(py_float_repr(x), expected, "{x}");
        }
    }
}
