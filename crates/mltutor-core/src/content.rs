//! Embedded pedagogical texts that the core needs to build messages
//! (`content/componentes/mensajes.toml`). The texts are data (AGENTS.md rule 8);
//! the core only chooses and fills them.

use crate::error::ContentError;

/// `content/componentes/mensajes.toml`: texts of the shared components (C1–C10).
pub const COMPONENT_MESSAGES: &str = include_str!("../../../content/componentes/mensajes.toml");

/// `content/regresion/mensajes.toml`: texts of the 📊 Regresión lesson.
pub const REGRESSION_MESSAGES: &str = include_str!("../../../content/regresion/mensajes.toml");

/// `content/redes/mensajes.toml`: texts of the 🧠 Redes Neuronales lesson.
pub const NN_MESSAGES: &str = include_str!("../../../content/redes/mensajes.toml");

/// `content/redes/diagnostico-tendencias.md` (expander of 📊 Tendencias).
pub const NN_DIAG_TRENDS_MD: &str =
    include_str!("../../../content/redes/diagnostico-tendencias.md");
/// `content/redes/diagnostico-contexto.md` (expander "🎓 Contexto Educativo").
pub const NN_DIAG_CONTEXT_MD: &str = include_str!("../../../content/redes/diagnostico-contexto.md");
/// `content/redes/diagnostico-historial-corto.md` (history shorter than 10 epochs).
pub const NN_DIAG_SHORT_MD: &str =
    include_str!("../../../content/redes/diagnostico-historial-corto.md");
/// `content/redes/historial-interpretar.md` (expander of 📊 Historial).
pub const NN_HISTORY_MD: &str = include_str!("../../../content/redes/historial-interpretar.md");

/// The Markdown body of a `content/**/*.md` file: without the leading `<!-- origin -->`
/// comments and the trailing newlines (what `ui.markdown` shows after
/// `textwrap.dedent().strip()`).
pub fn markdown_body(text: &str) -> &str {
    let mut rest = text.trim_start();
    while let Some(after) = rest.strip_prefix("<!--") {
        match after.find("-->") {
            Some(end) => rest = after[end + 3..].trim_start_matches(['\r', '\n']),
            None => break,
        }
    }
    rest.trim_end_matches(['\r', '\n'])
}

/// Parsed `[section]` of a messages file.
#[derive(Debug, Clone)]
pub struct Messages {
    file: &'static str,
    table: toml::Table,
}

impl Messages {
    /// Parses one of the embedded files (`name` is used in error messages).
    pub fn parse(file: &'static str, text: &str) -> Result<Self, ContentError> {
        let table: toml::Table = toml::from_str(text).map_err(|e| ContentError {
            file: file.to_owned(),
            key: String::new(),
            message: e.to_string(),
        })?;
        Ok(Self { file, table })
    }

    /// `content/componentes/mensajes.toml`.
    pub fn components() -> Result<Self, ContentError> {
        Self::parse("content/componentes/mensajes.toml", COMPONENT_MESSAGES)
    }

    /// `content/regresion/mensajes.toml`.
    pub fn regression() -> Result<Self, ContentError> {
        Self::parse("content/regresion/mensajes.toml", REGRESSION_MESSAGES)
    }

    /// `content/redes/mensajes.toml`.
    pub fn neural_network() -> Result<Self, ContentError> {
        Self::parse("content/redes/mensajes.toml", NN_MESSAGES)
    }

    fn value(&self, section: &str, key: &str) -> Result<&toml::Value, ContentError> {
        self.table
            .get(section)
            .and_then(|s| s.get(key))
            .ok_or_else(|| self.missing(section, key))
    }

    fn missing(&self, section: &str, key: &str) -> ContentError {
        ContentError {
            file: self.file.to_owned(),
            key: format!("{section}.{key}"),
            message: "falta o no tiene el tipo esperado".to_owned(),
        }
    }

    /// A string value.
    pub fn text(&self, section: &str, key: &str) -> Result<&str, ContentError> {
        self.value(section, key)?
            .as_str()
            .ok_or_else(|| self.missing(section, key))
    }

    /// An array of `[a, b]` string pairs (e.g. `[["🎯", "Estable"], …]`).
    pub fn table_pairs(
        &self,
        section: &str,
        key: &str,
    ) -> Result<Vec<(String, String)>, ContentError> {
        let arr = self
            .value(section, key)?
            .as_array()
            .ok_or_else(|| self.missing(section, key))?;
        arr.iter()
            .map(|row| {
                let pair = row.as_array().ok_or_else(|| self.missing(section, key))?;
                match (
                    pair.first().and_then(|v| v.as_str()),
                    pair.get(1).and_then(|v| v.as_str()),
                ) {
                    (Some(a), Some(b)) => Ok((a.to_owned(), b.to_owned())),
                    _ => Err(self.missing(section, key)),
                }
            })
            .collect()
    }

    /// An array of strings.
    pub fn lines(&self, section: &str, key: &str) -> Result<Vec<&str>, ContentError> {
        let arr = self
            .value(section, key)?
            .as_array()
            .ok_or_else(|| self.missing(section, key))?;
        arr.iter()
            .map(|v| v.as_str().ok_or_else(|| self.missing(section, key)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Messages;

    #[test]
    fn embedded_files_parse() {
        let m = Messages::components().unwrap_or_else(|e| panic!("{e}"));
        assert!(m.text("interpretacion", "titulo").is_ok());
        assert!(Messages::regression().is_ok());
        let nn = Messages::neural_network().unwrap_or_else(|e| panic!("{e}"));
        assert!(nn.text("diagnostico", "titulo").is_ok());
        assert!(super::markdown_body(super::NN_DIAG_SHORT_MD).starts_with("**"));
    }
}
