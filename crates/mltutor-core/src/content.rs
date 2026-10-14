//! Embedded pedagogical texts that the core needs to build messages
//! (`content/componentes/mensajes.toml`). The texts are data (AGENTS.md rule 8);
//! the core only chooses and fills them.

use crate::error::ContentError;

/// `content/componentes/mensajes.toml`: texts of the shared components (C1–C10).
pub const COMPONENT_MESSAGES: &str = include_str!("../../../content/componentes/mensajes.toml");

/// `content/regresion/mensajes.toml`: texts of the 📊 Regresión lesson.
pub const REGRESSION_MESSAGES: &str = include_str!("../../../content/regresion/mensajes.toml");

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
    }
}
