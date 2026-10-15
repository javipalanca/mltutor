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
    /// The model a view asked for is no longer the lesson's model (it was retrained).
    /// The UI reloads the view instead of showing an error.
    ModelStale {
        message: String,
    },
    /// D26: the target cannot be used by this model (logistic on a regression target).
    InvalidTarget {
        message: String,
    },
    /// The view needs a classifier (ROC, probabilities, boundary).
    NotClassifier {
        message: String,
    },
    /// The view needs a regressor (residuals, surface).
    NotRegressor {
        message: String,
    },
    /// The view needs a decision tree (diagram, rules, native importance, tree code).
    NotTree {
        message: String,
    },
    /// Hyperparameters out of range or incompatible (a regression criterion for a
    /// classification tree).
    InvalidConfig {
        message: String,
    },
    /// The grid of the decision boundary/surface has fewer than 2 rows or columns.
    GridTooSmall {
        message: String,
    },
    /// Values of the prediction form that cannot be used (length, NaN).
    InvalidInput {
        message: String,
    },
    /// The target has more classes than a classifier of the app accepts (the column
    /// is probably an identifier or a continuous value read as classes).
    TooManyClasses {
        n_classes: u32,
        max: u32,
        message: String,
    },
    /// The dataset has missing (NaN) or infinite values in these feature columns: no
    /// model of the app trains with them (sklearn 1.7 trees would).
    MissingValues {
        columns: Vec<String>,
        message: String,
    },
    /// Training failed (singular system, optimizer error…).
    Training {
        message: String,
    },
    /// Building an ONNX file or a Python package failed.
    Export {
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

    pub fn model_stale() -> Self {
        Self::ModelStale {
            message: "El modelo ha cambiado. Vuelve a cargar la vista.".into(),
        }
    }

    pub fn invalid_target(message: impl Into<String>) -> Self {
        Self::InvalidTarget {
            message: message.into(),
        }
    }

    pub fn not_classifier() -> Self {
        Self::NotClassifier {
            message: "Esta visualización necesita un modelo de clasificación.".into(),
        }
    }

    pub fn not_regressor() -> Self {
        Self::NotRegressor {
            message: "Esta visualización necesita un modelo de regresión.".into(),
        }
    }

    pub fn not_tree() -> Self {
        Self::NotTree {
            message: "Esta vista necesita un árbol de decisión.".into(),
        }
    }

    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    pub fn grid_too_small(rows: usize, cols: usize) -> Self {
        Self::GridTooSmall {
            message: format!(
                "❌ Error: La malla resultante es demasiado pequeña: ({rows}, {cols})"
            ),
        }
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// `message` is the core's (`ModelError::TooManyClasses`), already in Spanish.
    pub fn too_many_classes(n_classes: usize, max: usize, message: impl Into<String>) -> Self {
        Self::TooManyClasses {
            n_classes: u32::try_from(n_classes).unwrap_or(u32::MAX),
            max: u32::try_from(max).unwrap_or(u32::MAX),
            message: message.into(),
        }
    }

    pub fn training(error: impl std::fmt::Display) -> Self {
        Self::Training {
            message: format!("Error al entrenar el modelo: {error}"),
        }
    }

    pub fn export(error: impl std::fmt::Display) -> Self {
        Self::Export {
            message: format!("❌ No se pudo exportar el modelo: {error}"),
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
            | Self::ModelStale { message }
            | Self::InvalidTarget { message }
            | Self::NotClassifier { message }
            | Self::NotRegressor { message }
            | Self::NotTree { message }
            | Self::InvalidConfig { message }
            | Self::GridTooSmall { message }
            | Self::InvalidInput { message }
            | Self::TooManyClasses { message, .. }
            | Self::MissingValues { message, .. }
            | Self::Training { message }
            | Self::Export { message }
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
        let error = CommandError::too_many_classes(192, 50, "192 clases");
        let json = serde_json::to_value(&error).unwrap();
        assert_eq!(
            (&json["kind"], &json["n_classes"], &json["max"]),
            (
                &serde_json::json!("too_many_classes"),
                &serde_json::json!(192),
                &serde_json::json!(50)
            )
        );
        assert!(
            CommandError::csv_too_large(150 * 1_048_576, 100 * 1_048_576)
                .message()
                .contains("150.0 MB")
        );
    }
}
