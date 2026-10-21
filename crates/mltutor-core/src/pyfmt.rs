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
    // `{:e}` gives the shortest round-trip digits ("-1.2345e-7", "3e0"), but when
    // two strings of that length round-trip it may not pick the nearest one.
    // CPython (David Gay's dtoa) writes the nearest, ties to even: that is the
    // correctly rounded value with the same number of digits, which `{:.Ne}`
    // gives (1400.50006103515625 → "1400.5000610351562", not "…563").
    let shortest = format!("{x:e}");
    let n_sig = shortest
        .split('e')
        .next()
        .map_or(1, |m| m.chars().filter(char::is_ascii_digit).count())
        .max(1);
    let nearest = format!("{:.*e}", n_sig - 1, x);
    // At a power of two the round-trip interval is asymmetric: keep the shortest
    // string if the nearest one does not read back as `x`.
    let sci = if nearest.parse::<f64>().ok() == Some(x) {
        nearest
    } else {
        shortest
    };
    let (mantissa, exp) = match sci.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (sci.as_str(), 0),
    };
    let mantissa = if mantissa.contains('.') {
        mantissa.trim_end_matches('0').trim_end_matches('.')
    } else {
        mantissa
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

/// A value for [`py_format`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FmtArg<'a> {
    Float(f64),
    Int(i64),
    Str(&'a str),
}

/// Python's `str.format` for the few format specs used by the messages in
/// `content/**/mensajes.toml`: `{name}`, `{name:.Nf}`, `{name:+.Nf}`,
/// `{name:.N%}` and `{name:,}` (thousands separator for integers). Unknown names
/// are left untouched; `{{` and `}}` are literal braces.
pub fn py_format(template: &str, args: &[(&str, FmtArg<'_>)]) -> String {
    let mut out = String::with_capacity(template.len() + 16);
    let mut rest = template;
    while let Some(pos) = rest.find(['{', '}']) {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        if tail.starts_with("{{") || tail.starts_with("}}") {
            out.push_str(&tail[..1]);
            rest = &tail[2..];
            continue;
        }
        if let Some(after) = tail.strip_prefix('}') {
            out.push('}');
            rest = after;
            continue;
        }
        let Some(end) = tail.find('}') else {
            out.push_str(tail);
            return out;
        };
        let field = &tail[1..end];
        let (name, spec) = field.split_once(':').unwrap_or((field, ""));
        match args.iter().find(|(n, _)| *n == name) {
            Some((_, value)) => out.push_str(&format_spec(*value, spec)),
            None => out.push_str(&tail[..=end]),
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    out
}

/// One `format(value, spec)`.
pub fn format_spec(value: FmtArg<'_>, spec: &str) -> String {
    match value {
        FmtArg::Str(s) => s.to_owned(),
        FmtArg::Int(i) => {
            if spec == "," {
                group_thousands(&i.to_string())
            } else if let Some(prec) = parse_precision(spec, 'f') {
                format_fixed(i as f64, prec, spec.starts_with('+'))
            } else {
                i.to_string()
            }
        }
        FmtArg::Float(x) => {
            if let Some(prec) = parse_precision(spec, '%') {
                format!("{}%", format_fixed(x * 100.0, prec, spec.starts_with('+')))
            } else if let Some(prec) = parse_precision(spec, 'f') {
                format_fixed(x, prec, spec.starts_with('+'))
            } else if spec == "," {
                group_thousands(&py_float_repr(x))
            } else {
                py_float_repr(x)
            }
        }
    }
}

/// `"+.3f"` / `".3f"` → 3 (with `kind = 'f'`), `".1%"` → 1 (with `'%'`).
fn parse_precision(spec: &str, kind: char) -> Option<usize> {
    let s = spec.strip_prefix('+').unwrap_or(spec);
    let s = s.strip_prefix('.')?;
    let s = s.strip_suffix(kind)?;
    s.parse().ok()
}

/// `f"{x:.{prec}f}"` (Rust and Python both round the exact binary value half to
/// even). `nan`/`inf` like Python.
pub fn format_fixed(x: f64, prec: usize, plus: bool) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            if plus {
                "+inf"
            } else {
                "inf"
            }
        } else {
            "-inf"
        }
        .to_owned();
    }
    if plus {
        format!("{x:+.prec$}")
    } else {
        format!("{x:.prec$}")
    }
}

/// `"1234567"` → `"1,234,567"` (keeps a sign and a fractional part).
fn group_thousands(digits: &str) -> String {
    let (sign, body) = match digits.strip_prefix('-') {
        Some(b) => ("-", b),
        None => ("", digits),
    };
    let (int, frac) = match body.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (body, None),
    };
    let mut out = String::new();
    for (i, ch) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    match frac {
        Some(f) => format!("{sign}{out}.{f}"),
        None => format!("{sign}{out}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{py_float_repr, py_format, FmtArg};

    #[test]
    fn formats_like_python() {
        let s = py_format(
            "R2 {a:.1f}% mae {b:.2f} {c:,} {d:+.1f}% {e:.1%} {f} {g} {{x}} {missing}",
            &[
                ("a", FmtArg::Float(47.72897)),
                ("b", FmtArg::Float(41.919378)),
                ("c", FmtArg::Int(56700)),
                ("d", FmtArg::Float(-12.34)),
                ("e", FmtArg::Float(0.029239766)),
                ("f", FmtArg::Str("texto")),
                ("g", FmtArg::Int(3)),
            ],
        );
        assert_eq!(
            s,
            "R2 47.7% mae 41.92 56,700 -12.3% 2.9% texto 3 {x} {missing}"
        );
        assert_eq!(py_format("{x:.2f}", &[("x", FmtArg::Float(0.125))]), "0.12");
        assert_eq!(
            py_format("{x:,}", &[("x", FmtArg::Int(1_234_567))]),
            "1,234,567"
        );
    }

    #[test]
    fn matches_cpython_repr() {
        let cases = [
            (3.0, "3.0"),
            // Two 17-digit strings round-trip; CPython writes the nearest (tie → even).
            (f64::from_bits(0x4095_e200_1000_0000), "1400.5000610351562"),
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
