//! Data commands of fase1-shell-datos.md §5.4.
//!
//! Each command is `async` and runs its work in `spawn_blocking`, so neither parsing nor a
//! panic can block or abort the main thread. The session lock is never held while a
//! dataset is being computed.

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use tauri::State;

use crate::core_facade::{validate_pairplot_features, CoreFacade, DatasetHandle};
use crate::ipc::{
    CommandError, CorrelationMatrix, CsvLoaded, CsvPreview, DatasetEntry, DatasetRef,
    DatasetSummary, DescribeTable, PairplotData, PairplotRequest, TablePreview, TargetDistribution,
    TargetInfo, TaskChoice,
};
use crate::session::{CsvEntry, Session};

/// D10: larger CSV files are rejected before reading them.
pub const CSV_MAX_BYTES: u64 = 100 * 1024 * 1024;
/// Maximum rows of a preview request.
const MAX_PREVIEW_ROWS: u32 = 1000;

/// Managed state of the data commands.
#[derive(Clone)]
pub struct DataState {
    facade: Arc<dyn CoreFacade>,
    session: Arc<Mutex<Session>>,
}

impl DataState {
    pub fn new(facade: Arc<dyn CoreFacade>) -> Self {
        log::info!("datos: usando {}", facade.name());
        Self {
            facade,
            session: Arc::new(Mutex::new(Session::default())),
        }
    }

    fn session(&self) -> MutexGuard<'_, Session> {
        // A panic while holding the lock leaves the session usable: every update is a
        // single assignment or insertion.
        self.session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn list_datasets(&self) -> Vec<DatasetEntry> {
        let mut entries = self.facade.builtin_catalog();
        entries.extend(self.session().csv_entries().map(|e| e.entry.clone()));
        entries
    }

    pub fn selected(&self) -> DatasetRef {
        self.session().selected
    }

    /// The dataset behind `dataset` and its label, parsing a built-in one on first use.
    fn resolve(&self, dataset: DatasetRef) -> Result<(DatasetHandle, String), CommandError> {
        match dataset {
            DatasetRef::Csv { id } => self
                .session()
                .csv_entry(id)
                .map(|e| (e.dataset.clone(), e.entry.label.clone()))
                .ok_or_else(|| CommandError::dataset_not_found(format!("CSV {id}"))),
            DatasetRef::Builtin { id } => {
                let label = self
                    .facade
                    .builtin_catalog()
                    .into_iter()
                    .find(|e| e.dataset == dataset)
                    .map(|e| e.label)
                    .unwrap_or_else(|| id.slug().to_owned());
                if let Some(cached) = self.session().cached(id) {
                    return Ok((cached, label));
                }
                let loaded = self.facade.load_builtin(id)?;
                Ok((self.session().cache_builtin(id, loaded), label))
            }
        }
    }

    pub fn summary(&self, dataset: DatasetRef) -> Result<DatasetSummary, CommandError> {
        let (handle, label) = self.resolve(dataset)?;
        let mut summary = handle.summary();
        summary.dataset = dataset;
        summary.label = label;
        Ok(summary)
    }

    pub fn select(&self, dataset: DatasetRef) -> Result<DatasetSummary, CommandError> {
        let summary = self.summary(dataset)?;
        if !self.session().select(dataset) {
            return Err(CommandError::dataset_not_found(format!("{dataset:?}")));
        }
        Ok(summary)
    }

    pub fn preview(&self, dataset: DatasetRef, rows: u32) -> Result<TablePreview, CommandError> {
        let (handle, _) = self.resolve(dataset)?;
        Ok(handle.preview(rows.min(MAX_PREVIEW_ROWS) as usize))
    }

    pub fn describe(&self, dataset: DatasetRef) -> Result<DescribeTable, CommandError> {
        Ok(self.resolve(dataset)?.0.describe())
    }

    pub fn target_distribution(
        &self,
        dataset: DatasetRef,
    ) -> Result<TargetDistribution, CommandError> {
        Ok(self.resolve(dataset)?.0.target_distribution())
    }

    pub fn correlation(&self, dataset: DatasetRef) -> Result<CorrelationMatrix, CommandError> {
        Ok(self.resolve(dataset)?.0.correlation())
    }

    pub fn pairplot(&self, request: &PairplotRequest) -> Result<PairplotData, CommandError> {
        let (handle, _) = self.resolve(request.dataset)?;
        validate_pairplot_features(&request.features, &handle.summary().feature_display)?;
        handle.pairplot(&request.features, request.diag)
    }

    /// Reads a CSV chosen in the file dialog. It is kept as a draft until its target is
    /// confirmed with [`Self::set_csv_target`].
    pub fn load_csv(&self, path: &Path) -> Result<CsvPreview, CommandError> {
        let metadata = std::fs::metadata(path).map_err(CommandError::csv_io)?;
        if metadata.len() > CSV_MAX_BYTES {
            return Err(CommandError::csv_too_large(metadata.len(), CSV_MAX_BYTES));
        }
        let bytes = std::fs::read(path).map_err(CommandError::csv_io)?;
        let file_name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "datos.csv".to_owned());
        self.load_csv_bytes(&file_name, &bytes)
    }

    pub fn load_csv_bytes(
        &self,
        file_name: &str,
        bytes: &[u8],
    ) -> Result<CsvPreview, CommandError> {
        let draft = self.facade.read_csv(file_name, bytes)?;
        let mut preview = draft.preview();
        preview.draft_id = self.session().store_draft(draft);
        log::info!(
            "CSV leído: {file_name} ({} filas, {} columnas)",
            preview.n_rows,
            preview.n_cols
        );
        Ok(preview)
    }

    fn draft(&self, draft_id: u32) -> Result<crate::core_facade::CsvDraftHandle, CommandError> {
        self.session().draft(draft_id).ok_or_else(|| {
            CommandError::csv_io("el archivo ya no está disponible; vuelve a cargarlo")
        })
    }

    pub fn csv_target_info(&self, draft_id: u32, target: &str) -> Result<TargetInfo, CommandError> {
        self.draft(draft_id)?.target_info(target)
    }

    pub fn set_csv_target(
        &self,
        draft_id: u32,
        target: &str,
        task: TaskChoice,
    ) -> Result<CsvLoaded, CommandError> {
        let draft = self.draft(draft_id)?;
        let file_name = draft.preview().file_name;
        let done = draft.finalize(target, task)?;
        let label = format!("📄 {file_name}");
        let mut session = self.session();
        let registered = session.register_csv(&label, |dataset| CsvEntry {
            entry: DatasetEntry {
                dataset,
                label: label.clone(),
                blurb: None,
                task: done.resolved_task,
                synthetic: false,
            },
            file_name: file_name.clone(),
            target: target.to_owned(),
            task,
            dataset: done.dataset.clone(),
        });
        log::info!("CSV registrado y seleccionado: {label} (objetivo {target}, {task:?})");
        Ok(CsvLoaded {
            dataset: registered.entry.clone(),
            task_choice: task,
            resolved_task: done.resolved_task,
            n_rows: crate::ipc::to_u32(done.n_rows),
            n_cols: crate::ipc::to_u32(done.n_cols),
            target: registered.target.clone(),
            describe: done.describe,
            target_stats: done.target_stats,
        })
    }
}

/// Runs `work` on a blocking worker; a panic there becomes `CommandError::Internal`.
async fn blocking<T, F>(work: F) -> Result<T, CommandError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, CommandError> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(work)
        .await
        .map_err(|error| CommandError::internal(format!("el cálculo falló: {error}")))?
}

fn state(data: &State<'_, DataState>) -> DataState {
    data.inner().clone()
}

#[tauri::command]
pub async fn list_datasets(data: State<'_, DataState>) -> Result<Vec<DatasetEntry>, CommandError> {
    let data = state(&data);
    blocking(move || Ok(data.list_datasets())).await
}

#[tauri::command]
pub async fn get_selected_dataset(data: State<'_, DataState>) -> Result<DatasetRef, CommandError> {
    Ok(data.selected())
}

#[tauri::command]
pub async fn select_dataset(
    data: State<'_, DataState>,
    dataset: DatasetRef,
) -> Result<DatasetSummary, CommandError> {
    let data = state(&data);
    blocking(move || data.select(dataset)).await
}

#[tauri::command]
pub async fn dataset_summary(
    data: State<'_, DataState>,
    dataset: DatasetRef,
) -> Result<DatasetSummary, CommandError> {
    let data = state(&data);
    blocking(move || data.summary(dataset)).await
}

#[tauri::command]
pub async fn dataset_preview(
    data: State<'_, DataState>,
    dataset: DatasetRef,
    rows: u32,
) -> Result<TablePreview, CommandError> {
    let data = state(&data);
    blocking(move || data.preview(dataset, rows)).await
}

#[tauri::command]
pub async fn dataset_describe(
    data: State<'_, DataState>,
    dataset: DatasetRef,
) -> Result<DescribeTable, CommandError> {
    let data = state(&data);
    blocking(move || data.describe(dataset)).await
}

#[tauri::command]
pub async fn target_distribution(
    data: State<'_, DataState>,
    dataset: DatasetRef,
) -> Result<TargetDistribution, CommandError> {
    let data = state(&data);
    blocking(move || data.target_distribution(dataset)).await
}

#[tauri::command]
pub async fn correlation_matrix(
    data: State<'_, DataState>,
    dataset: DatasetRef,
) -> Result<CorrelationMatrix, CommandError> {
    let data = state(&data);
    blocking(move || data.correlation(dataset)).await
}

#[tauri::command]
pub async fn pairplot_data(
    data: State<'_, DataState>,
    req: PairplotRequest,
) -> Result<PairplotData, CommandError> {
    let data = state(&data);
    blocking(move || data.pairplot(&req)).await
}

#[tauri::command]
pub async fn load_csv(
    data: State<'_, DataState>,
    path: String,
) -> Result<CsvPreview, CommandError> {
    let data = state(&data);
    blocking(move || data.load_csv(Path::new(&path))).await
}

#[tauri::command]
pub async fn csv_target_info(
    data: State<'_, DataState>,
    draft_id: u32,
    target: String,
) -> Result<TargetInfo, CommandError> {
    let data = state(&data);
    blocking(move || data.csv_target_info(draft_id, &target)).await
}

#[tauri::command]
pub async fn set_csv_target(
    data: State<'_, DataState>,
    draft_id: u32,
    target: String,
    task: TaskChoice,
) -> Result<CsvLoaded, CommandError> {
    let data = state(&data);
    blocking(move || data.set_csv_target(draft_id, &target, task)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_facade::FixtureFacade;
    use crate::ipc::{DatasetId, DiagKind, TaskType};

    fn data() -> DataState {
        DataState::new(Arc::new(FixtureFacade::from_checkout()))
    }

    const TIPS_CSV: &[u8] = b"total_bill,tip,sex,size\n16.99,1.01,Female,2\n10.34,1.66,Male,3\n21.01,3.5,Male,3\n23.68,3.31,Male,2\n";

    #[test]
    fn lists_the_builtin_datasets_in_selector_order() {
        let entries = data().list_datasets();
        let ids: Vec<DatasetRef> = entries.iter().map(|e| e.dataset).collect();
        let expected: Vec<DatasetRef> = DatasetId::ALL
            .iter()
            .map(|id| DatasetRef::Builtin { id: *id })
            .collect();
        assert_eq!(ids, expected);
        assert_eq!(entries[0].label, "🌸 Iris - Clasificación de flores");
        assert!(entries[0].blurb.as_deref().unwrap().contains("**Iris**"));
        assert!(entries[3].blurb.is_none(), "Dígitos no tiene línea");
        assert!(entries[7].synthetic, "Moons");
        assert_eq!(entries[4].task, TaskType::Regression);
    }

    #[test]
    fn selection_defaults_to_iris_and_is_shared() {
        let data = data();
        assert_eq!(data.selected(), DatasetRef::default());
        let wine = DatasetRef::Builtin {
            id: DatasetId::Wine,
        };
        let summary = data.select(wine).unwrap();
        assert_eq!(summary.dataset, wine);
        assert_eq!(summary.label, "🍷 Vino - Clasificación de vinos");
        assert_eq!(data.selected(), wine);
        assert!(data.select(DatasetRef::Csv { id: 9 }).is_err());
        assert_eq!(data.selected(), wine);
    }

    #[test]
    fn builtin_datasets_are_cached() {
        let data = data();
        let iris = DatasetRef::default();
        let (a, _) = data.resolve(iris).unwrap();
        let (b, _) = data.resolve(iris).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn registered_csv_becomes_selected_and_same_name_replaces_it() {
        let data = data();
        let preview = data.load_csv_bytes("tips.csv", TIPS_CSV).unwrap();
        assert_eq!(preview.default_target, "size");
        let loaded = data
            .set_csv_target(preview.draft_id, "size", TaskChoice::Classification)
            .unwrap();
        assert_eq!(loaded.dataset.label, "📄 tips.csv");
        assert_eq!(data.selected(), loaded.dataset.dataset);
        assert_eq!(data.list_datasets().len(), 13);

        // Same file name again, another target: the entry is replaced, not duplicated.
        let again = data.load_csv_bytes("tips.csv", TIPS_CSV).unwrap();
        let replaced = data
            .set_csv_target(again.draft_id, "tip", TaskChoice::Regression)
            .unwrap();
        assert_eq!(replaced.dataset.dataset, loaded.dataset.dataset);
        assert_eq!(replaced.resolved_task, TaskType::Regression);
        let entries = data.list_datasets();
        assert_eq!(entries.len(), 13);
        assert_eq!(entries[12].task, TaskType::Regression);

        // Another file is appended after it.
        let other = data.load_csv_bytes("otro.csv", TIPS_CSV).unwrap();
        data.set_csv_target(other.draft_id, "size", TaskChoice::Auto)
            .unwrap();
        let labels: Vec<String> = data.list_datasets().into_iter().map(|e| e.label).collect();
        assert_eq!(&labels[12..], ["📄 tips.csv", "📄 otro.csv"]);

        let summary = data.summary(replaced.dataset.dataset).unwrap();
        assert_eq!(summary.label, "📄 tips.csv");
        assert_eq!(summary.n_features, 3);
    }

    #[test]
    fn stale_drafts_and_unknown_columns_are_errors() {
        let data = data();
        let first = data.load_csv_bytes("a.csv", TIPS_CSV).unwrap();
        let second = data.load_csv_bytes("b.csv", TIPS_CSV).unwrap();
        assert!(matches!(
            data.csv_target_info(first.draft_id, "tip"),
            Err(CommandError::CsvIo { .. })
        ));
        assert!(matches!(
            data.csv_target_info(second.draft_id, "nope"),
            Err(CommandError::UnknownColumn { .. })
        ));
        assert_eq!(
            data.csv_target_info(second.draft_id, "sex")
                .unwrap()
                .nunique,
            2
        );
    }

    #[test]
    fn csv_errors_are_translated() {
        let data = data();
        assert!(matches!(
            data.load_csv_bytes("x.csv", b""),
            Err(CommandError::CsvEmpty { .. })
        ));
        assert!(matches!(
            data.load_csv_bytes("x.csv", b"a\n1\n"),
            Err(CommandError::CsvTooFewColumns { .. })
        ));
        assert!(matches!(
            data.load_csv_bytes("x.csv", b"a,b\n1,2\n1,2,3\n"),
            Err(CommandError::CsvMalformed { line: 3, .. })
        ));
        assert!(matches!(
            data.load_csv(Path::new("/no/existe.csv")),
            Err(CommandError::CsvIo { .. })
        ));
    }

    #[test]
    fn pairplot_validates_the_selection() {
        let data = data();
        let request = |features: &[&str]| PairplotRequest {
            dataset: DatasetRef::default(),
            features: features.iter().map(|f| (*f).to_owned()).collect(),
            diag: DiagKind::Kde,
        };
        assert!(matches!(
            data.pairplot(&request(&[])),
            Err(CommandError::InvalidSelection { .. })
        ));
        assert!(matches!(
            data.pairplot(&request(&["petal width (cm)", "nope"])),
            Err(CommandError::InvalidSelection { .. })
        ));
        let plot = data
            .pairplot(&request(&["petal width (cm)", "sepal length (cm)"]))
            .unwrap();
        assert_eq!(plot.points.cols, 2);
        assert_eq!(plot.n_points, 150);
    }

    #[test]
    fn worker_panics_become_internal_errors() {
        let result = tauri::async_runtime::block_on(blocking::<(), _>(|| {
            panic!("panic de prueba en un worker de datos")
        }));
        assert!(matches!(result, Err(CommandError::Internal { .. })));
    }
}
