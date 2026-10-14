//! Embeds every template of `content/code/` (`*.j2`) so that `codegen` can load
//! them by name, including templates added after this crate was written.

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let dir = manifest.join("../../content/code");
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut names: Vec<(String, PathBuf)> = fs::read_dir(&dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "j2"))
                .filter_map(|p| {
                    let name = p.file_name()?.to_str()?.to_owned();
                    Some((name, p))
                })
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    let out = PathBuf::from(env::var("OUT_DIR").unwrap_or_default()).join("templates.rs");
    let mut code = String::from("/// `(file name, source)` of every `content/code/*.j2`.\npub const TEMPLATES: &[(&str, &str)] = &[\n");
    for (name, path) in &names {
        println!("cargo:rerun-if-changed={}", path.display());
        let abs = path.canonicalize().unwrap_or_else(|_| path.clone());
        code.push_str(&format!(
            "    ({name:?}, include_str!({:?})),\n",
            abs.display().to_string()
        ));
    }
    code.push_str("];\n");
    if let Ok(mut f) = fs::File::create(&out) {
        let _ = f.write_all(code.as_bytes());
    }
}
