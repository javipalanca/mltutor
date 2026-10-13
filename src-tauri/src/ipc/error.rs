//! Errors returned by the data commands, with a Spanish message ready for a `Notice`.

use serde::Serialize;
use ts_rs::TS;

/// Generic message shown when something unexpected happens (same text as the Qt version).
pub const INTERNAL_MESSAGE: &str =
    "No se pudo completar la operación. Puedes volver a intentarlo o elegir otra sección.";

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum CommandError {
    DatasetNotFound {
        message: String,
    },
    CsvEmpty {
        message: String,
    },
    CsvTooFewColumns {
        message: String,
    },
    CsvEncoding {
        message: String,
    },
    CsvMalformed {
        #[ts(type = "number")]
        line: u64,
        message: String,
    },
    CsvTooLarge {
        #[ts(type = "number")]
        bytes: u64,
        #[ts(type = "number")]
        limit: u64,
        message: String,
    },
    CsvIo {
        message: String,
    },
    UnknownColumn {
        column: String,
        message: String,
    },
    InvalidSelection {
        message: String,
    },
    /// The user cancelled the operation, or a newer request superseded it. The UI does
    /// not show it as an error.
    Cancelled {
        message: String,
    },
    /// Writing a file chosen in the save dialog failed.
    FileSave {
        message: String,
    },
    Internal {
        message: String,
    },
}

impl CommandError {
    pub fn dataset_not_found(name: impl std::fmt::Display) -> Self {
        Self::DatasetNotFound {
            message: format!("Conjunto de datos '{name}' no reconocido"),
        }
    }

    pub fn csv_empty() -> Self {
        Self::CsvEmpty {
            message: "❌ El archivo está vacío".into(),
        }
    }

    pub fn csv_too_few_columns() -> Self {
        Self::CsvTooFewColumns {
            message: "❌ El archivo debe tener al menos 2 columnas (características + objetivo)"
                .into(),
        }
    }

    pub fn csv_encoding() -> Self {
        Self::CsvEncoding {
            message: "❌ El archivo no es UTF-8 ni Windows-1252".into(),
        }
    }

    pub fn csv_malformed(line: u64, fields: usize, header: usize) -> Self {
        Self::CsvMalformed {
            line,
            message: format!("❌ La línea {line} tiene {fields} campos y la cabecera {header}"),
        }
    }

    pub fn csv_too_large(bytes: u64, limit: u64) -> Self {
        Self::CsvTooLarge {
            bytes,
            limit,
            message: format!(
                "❌ El archivo ocupa {:.1} MB y el máximo es {:.0} MB",
                bytes as f64 / 1_048_576.0,
                limit as f64 / 1_048_576.0
            ),
        }
    }

    pub fn csv_io(error: impl std::fmt::Display) -> Self {
        Self::CsvIo {
            message: format!("❌ Error al procesar el archivo: {error}"),
        }
    }

    pub fn unknown_column(column: &str) -> Self {
        Self::UnknownColumn {
            column: column.to_owned(),
            message: format!("Columna objetivo '{column}' no encontrada en el dataset"),
        }
    }

    pub fn invalid_selection(message: impl Into<String>) -> Self {
        Self::InvalidSelection {
            message: message.into(),
        }
    }

    pub fn cancelled() -> Self {
        Self::Cancelled {
            message: "Operación cancelada".into(),
        }
    }

    pub fn file_save(error: impl std::fmt::Display) -> Self {
        Self::FileSave {
            message: format!("❌ No se pudo guardar el archivo: {error}"),
        }
    }

    /// Unexpected failure: logs the technical detail and returns the generic message.
    pub fn internal(detail: impl std::fmt::Display) -> Self {
        log::error!("error interno en un comando: {detail}");
        Self::Internal {
            message: INTERNAL_MESSAGE.into(),
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::DatasetNotFound { message }
            | Self::CsvEmpty { message }
            | Self::CsvTooFewColumns { message }
            | Self::CsvEncoding { message }
            | Self::CsvMalformed { message, .. }
            | Self::CsvTooLarge { message, .. }
            | Self::CsvIo { message }
            | Self::UnknownColumn { message, .. }
            | Self::InvalidSelection { message }
            | Self::Cancelled { message }
            | Self::FileSave { message }
            | Self::Internal { message } => message,
        }
    }
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for CommandError {}

/// A panic (or cancellation) of a blocking worker becomes an `Internal` error: the panic
/// hook has already logged it, and the window stays open.
impl From<tauri::Error> for CommandError {
    fn from(error: tauri::Error) -> Self {
        Self::internal(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_with_kind_and_spanish_message() {
        let json = serde_json::to_value(CommandError::csv_empty()).unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "kind": "csv_empty", "message": "❌ El archivo está vacío" })
        );
        let json = serde_json::to_value(CommandError::csv_malformed(4, 3, 2)).unwrap();
        assert_eq!(json["kind"], "csv_malformed");
        assert_eq!(json["line"], 4);
        assert_eq!(
            json["message"],
            "❌ La línea 4 tiene 3 campos y la cabecera 2"
        );
    }

    #[test]
    fn messages_match_the_python_texts() {
        assert_eq!(
            CommandError::csv_too_few_columns().message(),
            "❌ El archivo debe tener al menos 2 columnas (características + objetivo)"
        );
        assert_eq!(
            CommandError::unknown_column("y").to_string(),
            "Columna objetivo 'y' no encontrada en el dataset"
        );
        assert_eq!(
            CommandError::csv_io("boom").message(),
            "❌ Error al procesar el archivo: boom"
        );
        assert_eq!(
            CommandError::dataset_not_found("x").message(),
            "Conjunto de datos 'x' no reconocido"
        );
        assert_eq!(
            CommandError::internal("detalle").message(),
            INTERNAL_MESSAGE
        );
        assert!(
            CommandError::csv_too_large(150 * 1_048_576, 100 * 1_048_576)
                .message()
                .contains("150.0 MB")
        );
    }
}
