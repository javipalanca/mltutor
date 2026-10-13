//! Algorithms, metrics, preprocessing and embedded datasets for MLTutor (no UI).
//!
//! Pure functions over `ndarray` and serde structs: no Tauri, no I/O besides
//! [`csv::read_csv_file`], no global state. Everything random takes an explicit
//! seed (42 by default, like `random_state=42` in the Python version).
//!
//! # Public API used by the IPC contract (fase1-shell-datos.md §5.4–5.5)
//!
//! This list is **stable**: `src-tauri` programs against it and converts these
//! domain types into its DTOs (`src-tauri/src/ipc/`). Session concepts
//! (`DatasetRef`, `CsvId`, `draft_id`) live in `src-tauri`.
//!
//! | Command | Core function |
//! |---|---|
//! | `list_datasets` | [`dataset::list_builtin`]`() -> Result<Vec<DatasetEntry>, DatasetError>` (12, selector order) |
//! | `select_dataset`, `dataset_summary` | [`dataset::load_builtin`]`(DatasetId) -> Result<Dataset, DatasetError>`, [`Dataset::summary`]`() -> DatasetSummary` |
//! | Moons (D9) | [`dataset::moons::moons_dataset`]`(seed: u32) -> Result<Dataset, DatasetError>` (`42` = embedded); [`dataset::make_moons`]`(n, noise, seed) -> (Array2<f64>, Vec<u32>)` |
//! | `dataset_preview` | [`explore::preview`]`(&Dataset, rows: usize) -> TablePreview` |
//! | `dataset_describe` | [`explore::describe`]`(&Dataset) -> DescribeTable` |
//! | `target_distribution` | [`explore::target_distribution`]`(&Dataset) -> TargetDistribution` |
//! | `correlation_matrix` | [`explore::correlation_matrix`]`(&Dataset) -> CorrelationMatrix` |
//! | pairplot options | [`explore::pairplot_defaults`]`(&Dataset) -> PairplotDefaults` |
//! | `pairplot_data` | [`explore::pairplot_data`]`(&Dataset, features: &[String], DiagKind) -> Result<PairplotData, DatasetError>` |
//! | `load_csv` | [`csv::read_csv_file`]`(&Path) -> Result<RawTable, CsvError>` (or [`csv::read_csv`]`(&[u8])`), then [`csv::RawTable::preview`]`() -> Result<CsvPreview, CsvError>` |
//! | `csv_target_info` | [`csv::RawTable::target_info`]`(&str) -> Result<TargetInfo, CsvError>` |
//! | `set_csv_target` | [`csv::RawTable::to_dataset`]`(target: &str, TaskChoice, file_name: &str) -> Result<Dataset, CsvError>`, [`csv::RawTable::describe`]`() -> DescribeTable`, [`csv::RawTable::target_stats`]`(&str, TaskChoice) -> Result<CsvTargetStats, CsvError>`; the resolved task is [`Dataset::task`] |
//! | cancellation | [`csv::read_csv_cancellable`]`(&[u8], &AtomicBool)`, [`csv::read_csv_file_cancellable`]`(&Path, &AtomicBool)`, [`csv::RawTable::to_dataset_cancellable`]`(…, &AtomicBool)` → [`CsvError::Cancelled`]; [`explore::correlation_matrix_cancellable`]`(&Dataset, &AtomicBool) -> Result<CorrelationMatrix, DatasetError>` → [`DatasetError::Cancelled`] |
//! | F2+ (D7) | [`split::train_test_split`]`(n_rows, test_size: f64, seed: u32) -> Result<TrainTestSplit, DatasetError>` |
//!
//! Other public modules: [`sampling`] (NumPy `RandomState`, `DataFrame.sample`),
//! [`stats`] (describe, value counts, histograms, Pearson, KDE) and [`pyfmt`].
//!
//! Matrices are `ndarray::Array2<f64>` with NaN for missing values; the DTOs turn
//! them into the contract's `Matrix { rows, cols, values: Vec<Option<f64>> }`.

pub mod csv;
pub mod dataset;
pub mod error;
pub mod explore;
pub mod frame;
pub mod pyfmt;
pub mod sampling;
pub mod split;
pub mod stats;

pub use dataset::{
    ColumnKind, Dataset, DatasetEntry, DatasetId, DatasetSummary, LabelColumn, Target, TaskChoice,
    TaskType,
};
pub use error::{CsvError, DatasetError};
pub use frame::{Cell, DescribeTable, TablePreview};

/// Version of the MLTutor workspace this crate was built from.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_matches_workspace() {
        assert!(VERSION.starts_with("0.4.0"));
    }
}
