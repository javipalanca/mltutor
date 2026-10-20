//! Embeds every template of `content/code/` (`*.j2`) so that `codegen` can load
//! them by name, including templates added after this crate was written, and
//! deflates the built-in datasets (`tests/parity/datasets/data/*.csv`) into
//! `OUT_DIR/datasets/<slug>.csv.deflate` for `dataset::catalog` (F7: 3.0 MB of
//! CSV become about 1 MB in the binary).
//!
//! A missing or unreadable directory fails the build (a panic is how a build
//! script reports an error): an empty template list would only show up at run time.

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env_var("CARGO_MANIFEST_DIR"));
    compress_datasets(&manifest);
    let dir = manifest.join("../../content/code");
    println!("cargo:rerun-if-changed={}", dir.display());
    let entries = fs::read_dir(&dir).unwrap_or_else(|e| {
        panic!("cannot read {}: {e}", dir.display());
    });
    let mut names: Vec<(String, PathBuf)> = Vec::new();
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
            .path();
        if path.extension().is_some_and(|x| x == "j2") {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                panic!("non UTF-8 template name: {}", path.display());
            };
            names.push((name.to_owned(), path.clone()));
        }
    }
    if names.is_empty() {
        panic!("no *.j2 templates in {}", dir.display());
    }
    names.sort();
    let out = PathBuf::from(env_var("OUT_DIR")).join("templates.rs");
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
    fs::File::create(&out)
        .and_then(|mut f| f.write_all(code.as_bytes()))
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", out.display()));
}

/// `<slug>.csv.deflate` = original length (u32, little endian) + raw deflate
/// (level 9). The parity test `every_dataset_matches_its_fixture_exactly` checks
/// the SHA-256 of the inflated bytes, so a broken round trip cannot go unnoticed.
fn compress_datasets(manifest: &std::path::Path) {
    use flate2::{write::DeflateEncoder, Compression};
    let dir = manifest.join("../../tests/parity/datasets/data");
    println!("cargo:rerun-if-changed={}", dir.display());
    let out_dir = PathBuf::from(env_var("OUT_DIR")).join("datasets");
    fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", out_dir.display()));
    let entries =
        fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
            .path();
        if path.extension().is_none_or(|x| x != "csv") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let raw = fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let len = u32::try_from(raw.len())
            .unwrap_or_else(|_| panic!("{} is larger than 4 GiB", path.display()));
        let mut enc = DeflateEncoder::new(len.to_le_bytes().to_vec(), Compression::best());
        let packed = enc
            .write_all(&raw)
            .and_then(|()| enc.finish())
            .unwrap_or_else(|e| panic!("cannot deflate {}: {e}", path.display()));
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            panic!("non UTF-8 dataset name: {}", path.display());
        };
        let target = out_dir.join(format!("{name}.deflate"));
        fs::write(&target, packed)
            .unwrap_or_else(|e| panic!("cannot write {}: {e}", target.display()));
    }
}

fn env_var(name: &str) -> String {
    env::var(name).unwrap_or_else(|e| panic!("missing variable {name}: {e}"))
}
