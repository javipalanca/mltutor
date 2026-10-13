//! Boundary between the Tauri commands and the computation (fase1-shell-datos.md §5.5).
//!
//! The commands only talk to [`CoreFacade`]. Two implementations exist:
//! - [`crate::real_facade::RealFacade`], over `mltutor-core` (used by the app);
//! - [`crate::fixture_facade::FixtureFacade`], over the golden fixtures of
//!   `tests/parity/` (tests, and development while the core was being written).
//!
//! Every method may be slow (parsing, KDE…): the commands call them from
//! `spawn_blocking`, never on the main thread. Implementations return IPC DTOs directly
//! so that the conversion from core types lives in one place per implementation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::ipc::{
    CommandError, CorrelationMatrix, CsvPreview, CsvTargetStats, DatasetEntry, DatasetId,
    DatasetSummary, DescribeTable, DiagKind, PairplotData, TablePreview, TargetDistribution,
    TargetInfo, TaskChoice, TaskType,
};

/// Maximum number of pairplot features (slider limit of §2.6).
pub const MAX_PAIRPLOT_FEATURES: usize = 6;

/// A parsed dataset, ready for the 📊 Datos views.
pub trait DatasetView: Send + Sync {
    /// `dataset` and `label` are filled in by the caller (they depend on registration).
    fn summary(&self) -> DatasetSummary;
    fn preview(&self, rows: usize) -> TablePreview;
    fn describe(&self) -> DescribeTable;
    fn target_distribution(&self) -> TargetDistribution;
    /// Checks `cancel` cooperatively (`CommandError::Cancelled` when set).
    fn correlation(&self, cancel: &AtomicBool) -> Result<CorrelationMatrix, CommandError>;
    /// `features` are friendly names; already validated as non-empty, ≤ 6 and unique.
    fn pairplot(&self, features: &[String], diag: DiagKind) -> Result<PairplotData, CommandError>;
}

pub type DatasetHandle = Arc<dyn DatasetView>;

/// Result of confirming the target of a CSV draft.
pub struct CsvFinal {
    pub dataset: DatasetHandle,
    pub resolved_task: TaskType,
    pub n_rows: usize,
    pub n_cols: usize,
    pub describe: DescribeTable,
    pub target_stats: CsvTargetStats,
}

/// A CSV that has been read but whose target is not confirmed yet.
pub trait CsvDraft: Send + Sync {
    /// Name of the file it was read from (without directories).
    fn file_name(&self) -> &str;
    /// `draft_id` is filled in by the caller.
    fn preview(&self) -> CsvPreview;
    fn target_info(&self, target: &str) -> Result<TargetInfo, CommandError>;
    /// Checks `cancel` cooperatively (`CommandError::Cancelled` when set).
    fn finalize(
        &self,
        target: &str,
        task: TaskChoice,
        cancel: &AtomicBool,
    ) -> Result<CsvFinal, CommandError>;
}

pub type CsvDraftHandle = Arc<dyn CsvDraft>;

pub trait CoreFacade: Send + Sync {
    /// Implementation name, for the log.
    fn name(&self) -> &'static str;
    /// The 12 built-in datasets, in selector order. Callers cache it (`DataState`).
    fn builtin_catalog(&self) -> Result<Vec<DatasetEntry>, CommandError>;
    /// Parses a built-in dataset. Moons is generated with seed 42 (D9); the session caches
    /// the result, so it stays fixed for the whole session.
    fn load_builtin(&self, id: DatasetId) -> Result<DatasetHandle, CommandError>;
    /// Reads a user CSV (`bytes` = whole file; size already checked). Checks `cancel`
    /// cooperatively (`CommandError::Cancelled` when set).
    fn read_csv(
        &self,
        file_name: &str,
        bytes: &[u8],
        cancel: &AtomicBool,
    ) -> Result<CsvDraftHandle, CommandError>;
}

/// `Err(Cancelled)` once `cancel` is set; for the checkpoints of the operations whose
/// core functions are not cancellable themselves.
pub fn check_cancelled(cancel: &AtomicBool) -> Result<(), CommandError> {
    if cancel.load(Ordering::Relaxed) {
        Err(CommandError::cancelled())
    } else {
        Ok(())
    }
}

/// Checks a pairplot selection against the dataset's friendly names.
pub fn validate_pairplot_features(
    features: &[String],
    available: &[String],
) -> Result<(), CommandError> {
    if features.is_empty() {
        return Err(CommandError::invalid_selection(
            "Selecciona al menos una característica para el pairplot.",
        ));
    }
    if features.len() > MAX_PAIRPLOT_FEATURES {
        return Err(CommandError::invalid_selection(format!(
            "El pairplot admite como máximo {MAX_PAIRPLOT_FEATURES} características."
        )));
    }
    for (index, feature) in features.iter().enumerate() {
        if !available.contains(feature) {
            return Err(CommandError::invalid_selection(format!(
                "La característica '{feature}' no existe en este dataset."
            )));
        }
        if features[..index].contains(feature) {
            return Err(CommandError::invalid_selection(format!(
                "La característica '{feature}' está repetida."
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_owned()).collect()
    }

    #[test]
    fn pairplot_selection_rules() {
        let available = names(&["a", "b", "c", "d", "e", "f", "g"]);
        assert!(validate_pairplot_features(&names(&["a", "b"]), &available).is_ok());
        for bad in [
            names(&[]),
            names(&["a", "b", "c", "d", "e", "f", "g"]),
            names(&["a", "zz"]),
            names(&["a", "a"]),
        ] {
            assert!(matches!(
                validate_pairplot_features(&bad, &available),
                Err(CommandError::InvalidSelection { .. })
            ));
        }
    }
}
