//! Error types of the crate.
//!
//! Messages are in Spanish because they may reach the student; the Tauri layer
//! still maps each variant to its own `CommandError` (fase1-shell-datos.md §5.1).

use serde::Serialize;

/// Errors while reading or interpreting a CSV file (user uploads and embedded data).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CsvError {
    /// No columns, or a header without data rows (pandas: `df.empty`).
    #[error("El archivo está vacío")]
    Empty,
    /// Fewer than two columns: there must be at least one feature and the target.
    #[error("El archivo debe tener al menos 2 columnas (características + objetivo)")]
    TooFewColumns { found: usize },
    /// The bytes are neither UTF-8 nor Windows-1252 text (e.g. UTF-16 or binary data).
    #[error("El archivo no es UTF-8 ni Windows-1252")]
    Encoding,
    /// A row whose number of fields differs from the header. `line` is 1-based and
    /// counts the header as line 1.
    #[error("La línea {line} tiene {found} campos y la cabecera {expected}")]
    Malformed {
        line: u64,
        found: usize,
        expected: usize,
    },
    /// The file exceeds [`crate::csv::MAX_CSV_BYTES`] (D10).
    #[error("El archivo ocupa {bytes} bytes y el límite es {limit}")]
    TooLarge { bytes: u64, limit: u64 },
    /// The file could not be read from disk.
    #[error("Error al procesar el archivo: {message}")]
    Io { message: String },
    /// The requested target column does not exist.
    #[error("Columna objetivo '{column}' no encontrada en el dataset")]
    UnknownColumn { column: String },
    /// The cancellation flag was set (`*_cancellable` functions).
    #[error("Operación cancelada")]
    Cancelled,
}

/// Errors while loading or exploring a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatasetError {
    /// The dataset slug or name is not one of the built-in datasets.
    #[error("Conjunto de datos '{name}' no reconocido")]
    UnknownDataset { name: String },
    /// A column (feature, label or target) does not exist in the dataset.
    #[error("Columna '{column}' no encontrada en el dataset")]
    UnknownColumn { column: String },
    /// An invalid selection of features (empty, too many, unknown or repeated names).
    #[error("Selección no válida: {reason}")]
    InvalidSelection { reason: String },
    /// The embedded metadata (`content/datasets/*.toml`) is invalid. Covered by tests.
    #[error("Metadatos no válidos del dataset '{dataset}': {message}")]
    Metadata { dataset: String, message: String },
    /// The embedded CSV does not match its metadata. Covered by tests.
    #[error("Datos no válidos del dataset '{dataset}': {message}")]
    Data { dataset: String, message: String },
    /// Reading a CSV failed.
    #[error(transparent)]
    Csv(#[from] CsvError),
    /// The cancellation flag was set (`*_cancellable` functions).
    #[error("Operación cancelada")]
    Cancelled,
}

/// An embedded content file (`content/**/*.toml`, templates) is invalid or lacks a
/// key. Covered by tests; it would only happen with a broken build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, thiserror::Error)]
#[error("Contenido no válido en {file} ({key}): {message}")]
pub struct ContentError {
    pub file: String,
    pub key: String,
    pub message: String,
}
