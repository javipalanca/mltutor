//! Test helpers shared by the MLTutor crates (dev-dependency only).
//!
//! The parity fixtures in `tests/parity/` were generated once with the Python/Qt app,
//! which is no longer in the tree (tag `v1.0.0`). They are frozen references, with one
//! exception: the **texts rendered from `content/code/*.j2`** (training scripts, package
//! scripts, LEEME files…). When a template changes on purpose, those fields can be
//! rewritten from the Rust output with
//!
//! ```sh
//! MLTUTOR_BLESS=1 cargo test --workspace --all-targets
//! git diff tests/parity   # review: only template texts may change
//! ```
//!
//! Only the string at the given JSON pointer is replaced, in place, with the same
//! encoding as Python's `json.dumps(ensure_ascii=False)`; the rest of the file (numbers,
//! layout) is left byte for byte. Numeric fixtures are never blessed: they go through the
//! usual `assert!`s. `tools/verify_exports.py` (CI job `exports`) is what checks that the
//! blessed scripts still run and agree with scikit-learn/Keras.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

/// Environment variable that turns template-text mismatches into fixture rewrites.
pub const BLESS_VAR: &str = "MLTUTOR_BLESS";

/// Serialises the read-modify-write of fixture files between the tests of one binary.
static WRITE_LOCK: Mutex<()> = Mutex::new(());

/// `tests/parity/` at the repository root.
pub fn parity_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity")
}

/// True when `MLTUTOR_BLESS=1`.
pub fn blessing() -> bool {
    std::env::var(BLESS_VAR).is_ok_and(|v| v == "1")
}

/// Asserts that the template text `actual` equals the string at `pointer` (RFC 6901) in
/// `tests/parity/<fixture>`; with `MLTUTOR_BLESS=1` it rewrites that string instead.
#[track_caller]
pub fn assert_template_text(fixture: &str, pointer: &str, actual: &str) {
    check(fixture, pointer, actual, false);
}

/// Like [`assert_template_text`], ignoring the leading and trailing newlines (the app
/// shows some snippets stripped). Blessing keeps the fixture's own surrounding newlines.
#[track_caller]
pub fn assert_template_text_trimmed(fixture: &str, pointer: &str, actual: &str) {
    check(fixture, pointer, actual, true);
}

#[track_caller]
fn check(fixture: &str, pointer: &str, actual: &str, trim: bool) {
    let path = parity_dir().join(fixture);
    let text = read(&path);
    let expected = string_at(&text, pointer)
        .unwrap_or_else(|e| panic!("{fixture}#{pointer}: {e}"))
        .1;
    let core = |s: &str| {
        if trim {
            s.trim_matches('\n').to_owned()
        } else {
            s.to_owned()
        }
    };
    if core(actual) == core(&expected) {
        return;
    }
    if !blessing() {
        assert_eq!(
            core(actual),
            core(&expected),
            "{fixture}#{pointer}: the rendered template differs from the fixture. If the \
             template changed on purpose, run the tests with {BLESS_VAR}=1 and review the diff."
        );
        return;
    }
    let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Re-read under the lock: another test of this binary may have blessed the same file.
    let text = read(&path);
    let (span, old) =
        string_at(&text, pointer).unwrap_or_else(|e| panic!("{fixture}#{pointer}: {e}"));
    let new = if trim {
        let lead = old.len() - old.trim_start_matches('\n').len();
        let trail = old.len() - old.trim_end_matches('\n').len();
        let body = old.trim_matches('\n');
        if body.is_empty() {
            actual.to_owned()
        } else {
            format!(
                "{}{}{}",
                &old[..lead],
                actual.trim_matches('\n'),
                &old[old.len() - trail..]
            )
        }
    } else {
        actual.to_owned()
    };
    let encoded = serde_json::to_string(&new)
        .unwrap_or_else(|e| panic!("{fixture}#{pointer}: cannot encode: {e}"));
    let mut out = String::with_capacity(text.len() + encoded.len());
    out.push_str(&text[..span.0]);
    out.push_str(&encoded);
    out.push_str(&text[span.1..]);
    fs::write(&path, out).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    eprintln!("{BLESS_VAR}: rewrote {fixture}#{pointer}");
}

fn read(path: &std::path::Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Byte span of the JSON string at `pointer` and its decoded value.
fn string_at(text: &str, pointer: &str) -> Result<((usize, usize), String), String> {
    let tokens = pointer_tokens(pointer)?;
    let bytes = text.as_bytes();
    let mut pos = skip_ws(bytes, 0);
    for token in &tokens {
        pos = child(text, pos, token)?;
    }
    let end = value_end(bytes, pos)?;
    let raw = &text[pos..end];
    if !raw.starts_with('"') {
        return Err(format!("not a string: {}", truncate(raw)));
    }
    let value: String = serde_json::from_str(raw).map_err(|e| e.to_string())?;
    Ok(((pos, end), value))
}

fn pointer_tokens(pointer: &str) -> Result<Vec<String>, String> {
    if pointer.is_empty() {
        return Ok(Vec::new());
    }
    let rest = pointer
        .strip_prefix('/')
        .ok_or_else(|| format!("a JSON pointer starts with '/': {pointer}"))?;
    Ok(rest
        .split('/')
        .map(|t| t.replace("~1", "/").replace("~0", "~"))
        .collect())
}

/// Start of the member `token` of the object (or element of the array) at `pos`.
fn child(text: &str, pos: usize, token: &str) -> Result<usize, String> {
    let bytes = text.as_bytes();
    match bytes.get(pos) {
        Some(b'{') => {
            let mut p = skip_ws(bytes, pos + 1);
            if bytes.get(p) == Some(&b'}') {
                return Err(format!("no key {token:?}"));
            }
            loop {
                let key_end = value_end(bytes, p)?;
                let key: String =
                    serde_json::from_str(&text[p..key_end]).map_err(|e| e.to_string())?;
                p = skip_ws(bytes, key_end);
                if bytes.get(p) != Some(&b':') {
                    return Err(format!("expected ':' at byte {p}"));
                }
                p = skip_ws(bytes, p + 1);
                if key == token {
                    return Ok(p);
                }
                p = skip_ws(bytes, value_end(bytes, p)?);
                match bytes.get(p) {
                    Some(b',') => p = skip_ws(bytes, p + 1),
                    _ => return Err(format!("no key {token:?}")),
                }
            }
        }
        Some(b'[') => {
            let index: usize = token
                .parse()
                .map_err(|_| format!("{token:?} is not an array index"))?;
            let mut p = skip_ws(bytes, pos + 1);
            for _ in 0..index {
                p = skip_ws(bytes, value_end(bytes, p)?);
                match bytes.get(p) {
                    Some(b',') => p = skip_ws(bytes, p + 1),
                    _ => return Err(format!("index {index} out of range")),
                }
            }
            if bytes.get(p) == Some(&b']') {
                return Err(format!("index {index} out of range"));
            }
            Ok(p)
        }
        _ => Err(format!("no container at byte {pos} for {token:?}")),
    }
}

/// End (exclusive) of the JSON value that starts at `pos`.
fn value_end(bytes: &[u8], pos: usize) -> Result<usize, String> {
    match bytes.get(pos) {
        Some(b'"') => {
            let mut p = pos + 1;
            while let Some(&b) = bytes.get(p) {
                match b {
                    b'\\' => p += 2,
                    b'"' => return Ok(p + 1),
                    _ => p += 1,
                }
            }
            Err("unterminated string".into())
        }
        Some(&open @ (b'{' | b'[')) => {
            let close = if open == b'{' { b'}' } else { b']' };
            let mut p = skip_ws(bytes, pos + 1);
            if bytes.get(p) == Some(&close) {
                return Ok(p + 1);
            }
            loop {
                if open == b'{' {
                    p = skip_ws(bytes, value_end(bytes, p)?);
                    if bytes.get(p) != Some(&b':') {
                        return Err(format!("expected ':' at byte {p}"));
                    }
                    p = skip_ws(bytes, p + 1);
                }
                p = skip_ws(bytes, value_end(bytes, p)?);
                match bytes.get(p) {
                    Some(b',') => p = skip_ws(bytes, p + 1),
                    Some(&b) if b == close => return Ok(p + 1),
                    _ => return Err(format!("malformed container at byte {p}")),
                }
            }
        }
        Some(_) => {
            let mut p = pos;
            while let Some(&b) = bytes.get(p) {
                if matches!(b, b',' | b']' | b'}') || b.is_ascii_whitespace() {
                    break;
                }
                p += 1;
            }
            if p == pos {
                Err(format!("empty value at byte {pos}"))
            } else {
                Ok(p)
            }
        }
        None => Err("unexpected end of file".into()),
    }
}

fn skip_ws(bytes: &[u8], mut pos: usize) -> usize {
    while bytes.get(pos).is_some_and(u8::is_ascii_whitespace) {
        pos += 1;
    }
    pos
}

fn truncate(s: &str) -> &str {
    match s.char_indices().nth(60) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"{
  "meta": {"a/b": [1, 2.5e-3, null]},
  "expected": {
    "case": {"text": "x = 1\n\"ñ\"", "n": [1, 2]},
    "list": ["p", {"text": "q"}]
  }
}"#;

    #[test]
    fn finds_the_string_at_a_pointer() {
        let ((s, e), v) = string_at(DOC, "/expected/case/text").unwrap();
        assert_eq!(v, "x = 1\n\"ñ\"");
        assert_eq!(&DOC[s..e], r#""x = 1\n\"ñ\"""#);
        assert_eq!(string_at(DOC, "/expected/list/1/text").unwrap().1, "q");
        assert_eq!(string_at(DOC, "/expected/list/0").unwrap().1, "p");
        assert!(string_at(DOC, "/expected/case/n").is_err());
        assert!(string_at(DOC, "/expected/missing").is_err());
        assert!(string_at(DOC, "/expected/list/2").is_err());
        assert!(string_at(DOC, "/meta/a~1b/1").is_err());
    }

    #[test]
    fn encoding_matches_python_json_dumps() {
        // json.dumps("a\"\\\n\t\u{1}ñ€😀", ensure_ascii=False)
        let s = serde_json::to_string("a\"\\\n\t\u{1}ñ€😀\u{7f}").unwrap();
        assert_eq!(s, "\"a\\\"\\\\\\n\\t\\u0001ñ€😀\u{7f}\"");
    }
}
