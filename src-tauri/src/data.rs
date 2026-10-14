//! Data commands of fase1-shell-datos.md §5.4.
//!
//! Each command is `async` and runs its work in `spawn_blocking`, so neither parsing nor a
//! panic can block or abort the main thread. The session lock is never held while a
//! dataset is being computed.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::cli::Mode;
use crate::core_facade::DatasetView;
use crate::core_facade::{
    check_cancelled, validate_pairplot_features, CoreFacade, CsvDraftHandle, DatasetHandle,
};
use crate::ipc::{
    CommandError, CorrelationMatrix, CsvLoaded, CsvPreview, DatasetEntry, DatasetRef,
    DatasetSummary, DescribeTable, DiagKind, PairplotData, PairplotRequest, SelectedDataset,
    TablePreview, TargetDistribution, TargetInfo, TaskChoice,
};
use crate::session::{CsvEntry, RegisterError, Session};
use crate::state::AppState;
use crate::tasks::TaskRegistry;

/// D10: larger CSV files are rejected before reading them.
pub const CSV_MAX_BYTES: u64 = 100 * 1024 * 1024;
/// Maximum rows of a preview request.
const MAX_PREVIEW_ROWS: u32 = 1000;
/// The CSV is read in chunks of this size, checking the cancel flag between them.
const READ_CHUNK: usize = 1 << 20;
/// Longest filter label accepted for the open dialog.
const MAX_FILTER_LABEL: usize = 80;

/// Managed state of the data commands.
#[derive(Clone)]
pub struct DataState {
    facade: Arc<dyn CoreFacade>,
    session: Arc<Mutex<Session>>,
    /// The built-in catalogue never changes during a session (embedded metadata).
    catalog: Arc<OnceLock<Result<Vec<DatasetEntry>, CommandError>>>,
    pub tasks: TaskRegistry,
}

impl DataState {
    pub fn new(facade: Arc<dyn CoreFacade>) -> Self {
        log::info!("datos: usando {}", facade.name());
        Self {
            facade,
            session: Arc::new(Mutex::new(Session::default())),
            catalog: Arc::new(OnceLock::new()),
            tasks: TaskRegistry::default(),
        }
    }

    fn session(&self) -> MutexGuard<'_, Session> {
        // A panic while holding the lock leaves the session usable: every update is a
        // single assignment or insertion.
        self.session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn catalog(&self) -> Result<&[DatasetEntry], CommandError> {
        self.catalog
            .get_or_init(|| self.facade.builtin_catalog())
            .as_deref()
            .map_err(Clone::clone)
    }

    pub fn list_datasets(&self) -> Result<Vec<DatasetEntry>, CommandError> {
        let mut entries = self.catalog()?.to_vec();
        entries.extend(self.session().csv_entries().map(|e| e.entry.clone()));
        Ok(entries)
    }

    #[cfg(test)]
    pub fn selected(&self) -> DatasetRef {
        self.session().selected
    }

    pub fn selection(&self) -> SelectedDataset {
        let session = self.session();
        SelectedDataset {
            dataset: session.selected,
            last_seq: session.last_seq(),
        }
    }

    /// The dataset behind `dataset` and its label, parsing a built-in one on first use.
    pub fn resolve(&self, dataset: DatasetRef) -> Result<(DatasetHandle, String), CommandError> {
        match dataset {
            DatasetRef::Csv { id } => self
                .session()
                .csv_entry(id)
                .map(|e| (e.dataset.clone(), e.entry.label.clone()))
                .ok_or_else(|| CommandError::dataset_not_found(format!("CSV {id}"))),
            DatasetRef::Builtin { id } => {
                let label = self
                    .catalog()?
                    .iter()
                    .find(|e| e.dataset == dataset)
                    .map(|e| e.label.clone())
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

    /// Selects `dataset` for request `seq` (the newest request wins) and returns its
    /// summary. The selection is validated and applied under the lock *before* the
    /// (possibly slow) summary, so a slow dataset cannot overwrite a later choice.
    pub fn select(&self, dataset: DatasetRef, seq: u64) -> Result<DatasetSummary, CommandError> {
        if !self.session().select(dataset, seq) {
            return Err(CommandError::dataset_not_found(format!("{dataset:?}")));
        }
        self.summary(dataset)
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

    pub fn correlation(
        &self,
        dataset: DatasetRef,
        cancel: &AtomicBool,
    ) -> Result<CorrelationMatrix, CommandError> {
        let (handle, _) = self.resolve(dataset)?;
        check_cancelled(cancel)?;
        handle.correlation(cancel)
    }

    pub fn pairplot(&self, request: &PairplotRequest) -> Result<PairplotData, CommandError> {
        let (handle, _) = self.resolve(request.dataset)?;
        validate_pairplot_features(&request.features, &handle.summary().feature_display)?;
        handle.pairplot(&request.features, request.diag)
    }

    /// Reads a CSV chosen in the file dialog (or given with `--scenario`; never a path
    /// from the frontend). It is kept as a draft until its target is confirmed with
    /// [`Self::set_csv_target`].
    pub fn load_csv(&self, path: &Path, cancel: &AtomicBool) -> Result<CsvPreview, CommandError> {
        // The previous draft is released first: only one file is kept in memory.
        let previous = self.session().clear_draft();
        drop(previous); // Outside the lock: freeing a big table takes a while.
        let bytes = read_capped(path, CSV_MAX_BYTES, cancel)?;
        let file_name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "datos.csv".to_owned());
        self.load_csv_bytes(&file_name, &bytes, cancel)
    }

    pub fn load_csv_bytes(
        &self,
        file_name: &str,
        bytes: &[u8],
        cancel: &AtomicBool,
    ) -> Result<CsvPreview, CommandError> {
        let previous = self.session().clear_draft();
        drop(previous);
        let draft = self.facade.read_csv(file_name, bytes, cancel)?;
        check_cancelled(cancel)?;
        let mut preview = draft.preview();
        let (draft_id, previous) = self.session().store_draft(draft);
        drop(previous); // Another read raced this one: its draft goes, outside the lock.
        preview.draft_id = draft_id;
        log::info!(
            "CSV leído: {file_name} ({} filas, {} columnas)",
            preview.n_rows,
            preview.n_cols
        );
        Ok(preview)
    }

    fn draft(&self, draft_id: u32) -> Result<CsvDraftHandle, CommandError> {
        self.session().draft(draft_id).ok_or_else(stale_draft)
    }

    pub fn csv_target_info(&self, draft_id: u32, target: &str) -> Result<TargetInfo, CommandError> {
        self.draft(draft_id)?.target_info(target)
    }

    /// "Quitar archivo": releases the draft. The registered CSV stays selectable, as in
    /// Python. Returns whether `draft_id` was still the current draft. The draft is
    /// dropped after releasing the lock (callers run this on a blocking worker).
    pub fn discard_csv(&self, draft_id: u32) -> bool {
        let discarded = self.session().discard_draft(draft_id);
        discarded.is_some()
    }

    /// Confirms the target of draft `draft_id` and registers it for request `seq`.
    /// Fails with `CsvIo` if another file replaced the draft meanwhile, and with
    /// `Cancelled` if a newer registration was applied first.
    pub fn set_csv_target(
        &self,
        draft_id: u32,
        target: &str,
        task: TaskChoice,
        seq: u64,
        cancel: &AtomicBool,
    ) -> Result<CsvLoaded, CommandError> {
        let draft = self.draft(draft_id)?;
        let file_name = draft.file_name().to_owned();
        let done = draft.finalize(target, task, cancel)?;
        // Not needed any more: the session keeps its own reference while it is current.
        drop(draft);
        let label = format!("📄 {file_name}");
        let view: DatasetHandle = Arc::new(CachedViews::new(done.dataset.clone()));
        let mut session = self.session();
        let (registered, released) = session
            .register_csv(draft_id, seq, &label, |dataset| CsvEntry {
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
                dataset: view,
            })
            .map_err(|error| match error {
                RegisterError::StaleDraft => stale_draft(),
                RegisterError::Superseded => CommandError::cancelled(),
            })?;
        let entry = registered.entry.clone();
        let target = registered.target.clone();
        drop(session);
        // The previous CSV (another file, or this one with another target) is freed
        // outside the lock.
        drop(released);
        log::info!("CSV registrado y seleccionado: {label} (objetivo {target}, {task:?})");
        Ok(CsvLoaded {
            dataset: entry,
            task_choice: task,
            resolved_task: done.resolved_task,
            n_rows: crate::ipc::to_u32(done.n_rows),
            n_cols: crate::ipc::to_u32(done.n_cols),
            target,
            describe: done.describe,
            target_stats: done.target_stats,
        })
    }
}

/// The view of a registered CSV: `describe` and `target_distribution` are computed once
/// (the 📊 Datos tab asks for them every time it is shown, in every lesson).
struct CachedViews {
    inner: DatasetHandle,
    describe: OnceLock<DescribeTable>,
    target_distribution: OnceLock<TargetDistribution>,
}

impl CachedViews {
    fn new(inner: DatasetHandle) -> Self {
        Self {
            inner,
            describe: OnceLock::new(),
            target_distribution: OnceLock::new(),
        }
    }
}

impl DatasetView for CachedViews {
    fn summary(&self) -> DatasetSummary {
        self.inner.summary()
    }

    fn preview(&self, rows: usize) -> TablePreview {
        self.inner.preview(rows)
    }

    fn describe(&self) -> DescribeTable {
        self.describe.get_or_init(|| self.inner.describe()).clone()
    }

    fn target_distribution(&self) -> TargetDistribution {
        self.target_distribution
            .get_or_init(|| self.inner.target_distribution())
            .clone()
    }

    fn correlation(&self, cancel: &AtomicBool) -> Result<CorrelationMatrix, CommandError> {
        self.inner.correlation(cancel)
    }

    fn pairplot(&self, features: &[String], diag: DiagKind) -> Result<PairplotData, CommandError> {
        self.inner.pairplot(features, diag)
    }

    fn train(
        &self,
        spec: &crate::core_facade::TrainSpec,
        progress: &crate::core_facade::Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<crate::core_facade::ModelHandle, CommandError> {
        self.inner.train(spec, progress, cancel)
    }
}

fn stale_draft() -> CommandError {
    CommandError::csv_io("el archivo ya no está disponible; vuelve a cargarlo")
}

/// Reads at most `limit` bytes of a regular file, checking `cancel` between chunks.
///
/// FIFOs, devices and directories are rejected before opening them (opening a FIFO would
/// block). The size is checked on the opened handle and the read itself is capped with
/// `take(limit + 1)`, so a file that grows after the check cannot exceed the limit.
pub fn read_capped(path: &Path, limit: u64, cancel: &AtomicBool) -> Result<Vec<u8>, CommandError> {
    let not_regular = || CommandError::csv_io("no es un archivo normal");
    if !std::fs::metadata(path)
        .map_err(CommandError::csv_io)?
        .is_file()
    {
        return Err(not_regular());
    }
    let file = File::open(path).map_err(CommandError::csv_io)?;
    let metadata = file.metadata().map_err(CommandError::csv_io)?;
    if !metadata.is_file() {
        return Err(not_regular());
    }
    if metadata.len() > limit {
        return Err(CommandError::csv_too_large(metadata.len(), limit));
    }
    let capacity = usize::try_from(metadata.len()).unwrap_or(0);
    let mut bytes = Vec::with_capacity(capacity);
    let mut reader = file.take(limit.saturating_add(1));
    let mut chunk = vec![0_u8; READ_CHUNK];
    loop {
        check_cancelled(cancel)?;
        let read = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(CommandError::csv_io(error)),
        };
        bytes.extend_from_slice(&chunk[..read]);
    }
    let read = bytes.len() as u64;
    if read > limit {
        return Err(CommandError::csv_too_large(read.max(metadata.len()), limit));
    }
    Ok(bytes)
}

/// Runs `work` on a blocking worker; a panic there becomes `CommandError::Internal`.
pub async fn blocking<T, F>(work: F) -> Result<T, CommandError>
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
    blocking(move || data.list_datasets()).await
}

#[tauri::command]
pub async fn get_selected_dataset(
    data: State<'_, DataState>,
) -> Result<SelectedDataset, CommandError> {
    Ok(data.selection())
}

#[tauri::command]
pub async fn select_dataset(
    data: State<'_, DataState>,
    dataset: DatasetRef,
    seq: u64,
) -> Result<DatasetSummary, CommandError> {
    let data = state(&data);
    blocking(move || data.select(dataset, seq)).await
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
    task_id: u32,
) -> Result<CorrelationMatrix, CommandError> {
    let data = state(&data);
    let task = data.tasks.start(task_id);
    blocking(move || data.correlation(dataset, &task.token())).await
}

#[tauri::command]
pub async fn pairplot_data(
    data: State<'_, DataState>,
    req: PairplotRequest,
) -> Result<PairplotData, CommandError> {
    let data = state(&data);
    blocking(move || data.pairplot(&req)).await
}

/// Cancels a running operation started with `task_id`; `false` if it already finished.
#[tauri::command]
pub async fn cancel_task(data: State<'_, DataState>, task_id: u32) -> Result<bool, CommandError> {
    let cancelled = data.tasks.cancel(task_id);
    if cancelled {
        log::info!("operación {task_id} cancelada por el usuario");
    }
    Ok(cancelled)
}

/// Opens the native "Abrir" dialog and reads the chosen CSV. The path never comes from
/// the frontend; `Ok(None)` if the user closes the dialog.
#[tauri::command]
pub async fn pick_and_load_csv(
    app: AppHandle,
    data: State<'_, DataState>,
    filter_name: String,
    task_id: u32,
) -> Result<Option<CsvPreview>, CommandError> {
    let data = state(&data);
    let task = data.tasks.start(task_id);
    let filter: String = filter_name.chars().take(MAX_FILTER_LABEL).collect();
    blocking(move || {
        let mut dialog = app.dialog().file().add_filter(filter, &["csv"]);
        if let Some(window) = app.get_webview_window("main") {
            dialog = dialog.set_parent(&window);
        }
        let Some(picked) = dialog.blocking_pick_file() else {
            return Ok(None);
        };
        let path = picked.into_path().map_err(CommandError::csv_io)?;
        data.load_csv(&path, &task.token()).map(Some)
    })
    .await
}

/// `--screenshot --scenario csv_…:<ruta.csv>`: reads the CSV named on the command line.
/// Only available in that mode, and the path comes from the command line, not from JS.
#[tauri::command]
pub async fn load_scenario_csv(
    app_state: State<'_, AppState>,
    data: State<'_, DataState>,
    task_id: u32,
) -> Result<CsvPreview, CommandError> {
    let Some(path) = scenario_csv_path(&app_state.mode) else {
        return Err(CommandError::invalid_selection(
            "Este escenario necesita la ruta del CSV en la línea de órdenes: --scenario <nombre>:<ruta.csv>",
        ));
    };
    let data = state(&data);
    let task = data.tasks.start(task_id);
    blocking(move || data.load_csv(Path::new(&path), &task.token())).await
}

/// The `<ruta.csv>` of `--scenario <nombre>:<ruta.csv>` in `--screenshot` mode.
fn scenario_csv_path(mode: &Mode) -> Option<String> {
    let Mode::Screenshot(options) = mode else {
        return None;
    };
    let (_, path) = options.scenario.as_deref()?.split_once(':')?;
    (!path.is_empty()).then(|| path.to_owned())
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
    seq: u64,
    task_id: u32,
) -> Result<CsvLoaded, CommandError> {
    let data = state(&data);
    let guard = data.tasks.start(task_id);
    blocking(move || data.set_csv_target(draft_id, &target, task, seq, &guard.token())).await
}

/// Runs on a blocking worker: the released draft may be a big table.
#[tauri::command]
pub async fn discard_csv(data: State<'_, DataState>, draft_id: u32) -> Result<bool, CommandError> {
    let data = state(&data);
    blocking(move || Ok(data.discard_csv(draft_id))).await
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc;
    use std::sync::Weak;

    use super::*;
    use crate::core_facade::{CsvDraft, CsvFinal};
    use crate::fixture_facade::FixtureFacade;
    use crate::ipc::{DatasetId, DiagKind, TaskType};

    fn data() -> DataState {
        DataState::new(Arc::new(FixtureFacade::from_checkout()))
    }

    fn go() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn builtin(id: DatasetId) -> DatasetRef {
        DatasetRef::Builtin { id }
    }

    const TIPS_CSV: &[u8] = b"total_bill,tip,sex,size\n16.99,1.01,Female,2\n10.34,1.66,Male,3\n21.01,3.5,Male,3\n23.68,3.31,Male,2\n";

    #[test]
    fn lists_the_builtin_datasets_in_selector_order() {
        let entries = data().list_datasets().unwrap();
        let ids: Vec<DatasetRef> = entries.iter().map(|e| e.dataset).collect();
        let expected: Vec<DatasetRef> = DatasetId::ALL.iter().map(|id| builtin(*id)).collect();
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
        let wine = builtin(DatasetId::Wine);
        let summary = data.select(wine, 1).unwrap();
        assert_eq!(summary.dataset, wine);
        assert_eq!(summary.label, "🍷 Vino - Clasificación de vinos");
        assert_eq!(data.selected(), wine);
        assert!(data.select(DatasetRef::Csv { id: 9 }, 2).is_err());
        assert_eq!(data.selected(), wine);
    }

    /// Review scenario A: Iris → California → Vino in quick succession. California is
    /// slow to parse; whatever order the workers reach the session in, Vino wins.
    #[test]
    fn the_latest_selection_wins_whatever_the_completion_order() {
        let data = data();
        let requests = [
            (builtin(DatasetId::Iris), 1),
            (builtin(DatasetId::CaliforniaHousing), 2),
            (builtin(DatasetId::Wine), 3),
        ];
        std::thread::scope(|scope| {
            for (dataset, seq) in requests {
                let data = data.clone();
                scope.spawn(move || data.select(dataset, seq).unwrap());
            }
        });
        assert_eq!(data.selected(), builtin(DatasetId::Wine));

        // Worst case, deterministic: the older requests arrive after the newest one.
        let data = self::data();
        data.select(builtin(DatasetId::Wine), 3).unwrap();
        let late = data
            .select(builtin(DatasetId::CaliforniaHousing), 2)
            .unwrap();
        assert_eq!(late.dataset, builtin(DatasetId::CaliforniaHousing));
        data.select(builtin(DatasetId::Iris), 1).unwrap();
        assert_eq!(data.selected(), builtin(DatasetId::Wine));
    }

    #[test]
    fn builtin_datasets_and_the_catalog_are_cached() {
        let data = data();
        let iris = DatasetRef::default();
        let (a, _) = data.resolve(iris).unwrap();
        let (b, _) = data.resolve(iris).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let first = data.catalog().unwrap().as_ptr();
        assert_eq!(data.catalog().unwrap().as_ptr(), first);
    }

    #[test]
    fn registered_csv_becomes_selected_and_replaces_the_previous_one() {
        let data = data();
        let preview = data.load_csv_bytes("tips.csv", TIPS_CSV, &go()).unwrap();
        assert_eq!(preview.default_target, "size");
        let loaded = data
            .set_csv_target(
                preview.draft_id,
                "size",
                TaskChoice::Classification,
                1,
                &go(),
            )
            .unwrap();
        assert_eq!(loaded.dataset.label, "📄 tips.csv");
        assert_eq!(data.selected(), loaded.dataset.dataset);
        assert_eq!(data.list_datasets().unwrap().len(), 13);

        // Same file name again, another target: the entry is replaced, not duplicated.
        let again = data.load_csv_bytes("tips.csv", TIPS_CSV, &go()).unwrap();
        let replaced = data
            .set_csv_target(again.draft_id, "tip", TaskChoice::Regression, 2, &go())
            .unwrap();
        assert_eq!(replaced.dataset.dataset, loaded.dataset.dataset);
        assert_eq!(replaced.resolved_task, TaskType::Regression);
        let entries = data.list_datasets().unwrap();
        assert_eq!(entries.len(), 13);
        assert_eq!(entries[12].task, TaskType::Regression);
        let summary = data.summary(replaced.dataset.dataset).unwrap();
        assert_eq!(summary.label, "📄 tips.csv");
        assert_eq!(summary.n_features, 3);

        // Another file releases the previous one: the session keeps a single CSV.
        let other = data.load_csv_bytes("otro.csv", TIPS_CSV, &go()).unwrap();
        let other = data
            .set_csv_target(other.draft_id, "size", TaskChoice::Auto, 3, &go())
            .unwrap();
        let labels: Vec<String> = data
            .list_datasets()
            .unwrap()
            .into_iter()
            .map(|e| e.label)
            .collect();
        assert_eq!(&labels[12..], ["📄 otro.csv"]);
        assert_eq!(data.selected(), other.dataset.dataset);
        assert!(data.summary(replaced.dataset.dataset).is_err());
    }

    /// Review scenario C: two quick changes of task. The older registration cannot
    /// overwrite the newer one even if it finishes last.
    #[test]
    fn an_older_registration_does_not_overwrite_a_newer_one() {
        let data = data();
        let preview = data.load_csv_bytes("tips.csv", TIPS_CSV, &go()).unwrap();
        let newer = data
            .set_csv_target(preview.draft_id, "size", TaskChoice::Regression, 5, &go())
            .unwrap();
        assert!(matches!(
            data.set_csv_target(
                preview.draft_id,
                "size",
                TaskChoice::Classification,
                4,
                &go()
            ),
            Err(CommandError::Cancelled { .. })
        ));
        let entries = data.list_datasets().unwrap();
        assert_eq!(entries[12].task, TaskType::Regression);
        assert_eq!(data.selected(), newer.dataset.dataset);
    }

    #[test]
    fn a_registration_does_not_override_a_newer_selection() {
        let data = data();
        let preview = data.load_csv_bytes("tips.csv", TIPS_CSV, &go()).unwrap();
        data.select(builtin(DatasetId::Wine), 7).unwrap();
        let loaded = data
            .set_csv_target(preview.draft_id, "size", TaskChoice::Auto, 6, &go())
            .unwrap();
        assert_eq!(data.list_datasets().unwrap().len(), 13);
        assert_ne!(data.selected(), loaded.dataset.dataset);
        assert_eq!(data.selected(), builtin(DatasetId::Wine));
    }

    /// Wraps the fixture facade: drafts of `gated.csv` block in `finalize` until the
    /// test releases them, and every draft is tracked with a `Weak` to check it is freed.
    struct GatedFacade {
        inner: FixtureFacade,
        gate: Mutex<Option<mpsc::Receiver<()>>>,
        entered: Mutex<Option<mpsc::Sender<()>>>,
        drafts: Mutex<Vec<(String, Weak<dyn CsvDraft>)>>,
    }

    struct GatedDraft {
        inner: CsvDraftHandle,
        gate: Option<Mutex<mpsc::Receiver<()>>>,
        entered: Option<mpsc::Sender<()>>,
    }

    impl CsvDraft for GatedDraft {
        fn file_name(&self) -> &str {
            self.inner.file_name()
        }
        fn preview(&self) -> CsvPreview {
            self.inner.preview()
        }
        fn target_info(&self, target: &str) -> Result<TargetInfo, CommandError> {
            self.inner.target_info(target)
        }
        fn finalize(
            &self,
            target: &str,
            task: TaskChoice,
            cancel: &AtomicBool,
        ) -> Result<CsvFinal, CommandError> {
            if let (Some(entered), Some(gate)) = (&self.entered, &self.gate) {
                entered.send(()).unwrap();
                gate.lock().unwrap().recv().unwrap();
            }
            self.inner.finalize(target, task, cancel)
        }
    }

    impl CoreFacade for GatedFacade {
        fn name(&self) -> &'static str {
            "GatedFacade"
        }
        fn builtin_catalog(&self) -> Result<Vec<DatasetEntry>, CommandError> {
            self.inner.builtin_catalog()
        }
        fn load_builtin(&self, id: DatasetId) -> Result<DatasetHandle, CommandError> {
            self.inner.load_builtin(id)
        }
        fn read_csv(
            &self,
            file_name: &str,
            bytes: &[u8],
            cancel: &AtomicBool,
        ) -> Result<CsvDraftHandle, CommandError> {
            let gated = file_name == "gated.csv";
            let draft: CsvDraftHandle = Arc::new(GatedDraft {
                inner: self.inner.read_csv(file_name, bytes, cancel)?,
                gate: if gated {
                    self.gate.lock().unwrap().take().map(Mutex::new)
                } else {
                    None
                },
                entered: if gated {
                    self.entered.lock().unwrap().take()
                } else {
                    None
                },
            });
            self.drafts
                .lock()
                .unwrap()
                .push((file_name.to_owned(), Arc::downgrade(&draft)));
            Ok(draft)
        }
    }

    impl GatedFacade {
        fn alive(&self, file_name: &str) -> bool {
            self.drafts
                .lock()
                .unwrap()
                .iter()
                .any(|(name, weak)| name == file_name && weak.strong_count() > 0)
        }
    }

    fn gated() -> (Arc<GatedFacade>, mpsc::Sender<()>, mpsc::Receiver<()>) {
        let (release, gate) = mpsc::channel();
        let (entered, entered_rx) = mpsc::channel();
        let facade = Arc::new(GatedFacade {
            inner: FixtureFacade::from_checkout(),
            gate: Mutex::new(Some(gate)),
            entered: Mutex::new(Some(entered)),
            drafts: Mutex::new(Vec::new()),
        });
        (facade, release, entered_rx)
    }

    /// Review scenario B: the big CSV A is still being registered when the small B is
    /// read. A must not be registered (nor selected) once B replaced its draft.
    #[test]
    fn a_draft_replaced_meanwhile_is_not_registered() {
        let (facade, release, entered) = gated();
        let data = DataState::new(facade.clone());
        let a = data.load_csv_bytes("gated.csv", TIPS_CSV, &go()).unwrap();
        let worker = {
            let data = data.clone();
            std::thread::spawn(move || {
                data.set_csv_target(a.draft_id, "size", TaskChoice::Auto, 1, &go())
            })
        };
        entered.recv().unwrap();
        let b = data.load_csv_bytes("b.csv", TIPS_CSV, &go()).unwrap();
        release.send(()).unwrap();
        assert!(matches!(
            worker.join().unwrap(),
            Err(CommandError::CsvIo { .. })
        ));
        assert_eq!(data.list_datasets().unwrap().len(), 12);
        assert_eq!(data.selected(), DatasetRef::default());
        let loaded = data
            .set_csv_target(b.draft_id, "size", TaskChoice::Auto, 2, &go())
            .unwrap();
        assert_eq!(data.selected(), loaded.dataset.dataset);
        assert!(!facade.alive("gated.csv"), "the draft of A is released");
    }

    #[test]
    fn drafts_are_released_when_replaced_or_discarded() {
        let (facade, _release, _entered) = gated();
        let data = DataState::new(facade.clone());
        let a = data.load_csv_bytes("a.csv", TIPS_CSV, &go()).unwrap();
        data.set_csv_target(a.draft_id, "size", TaskChoice::Auto, 1, &go())
            .unwrap();
        assert!(
            facade.alive("a.csv"),
            "kept while the CSV page can change it"
        );
        let b = data.load_csv_bytes("b.csv", TIPS_CSV, &go()).unwrap();
        assert!(!facade.alive("a.csv"));
        assert!(!data.discard_csv(a.draft_id));
        assert!(data.discard_csv(b.draft_id));
        assert!(!facade.alive("b.csv"));
        assert!(matches!(
            data.csv_target_info(b.draft_id, "tip"),
            Err(CommandError::CsvIo { .. })
        ));
        // The registered CSV stays in the selector, as in Python.
        assert_eq!(data.list_datasets().unwrap().len(), 13);
    }

    /// Counts drops of drafts and CSV datasets, and whether the session lock was held then.
    #[derive(Default)]
    struct LockProbe {
        session: OnceLock<Arc<Mutex<Session>>>,
        drops: std::sync::atomic::AtomicUsize,
        drops_while_locked: std::sync::atomic::AtomicUsize,
        describes: std::sync::atomic::AtomicUsize,
        distributions: std::sync::atomic::AtomicUsize,
    }

    struct Probed<T> {
        inner: T,
        probe: Arc<LockProbe>,
    }

    impl<T> Drop for Probed<T> {
        fn drop(&mut self) {
            self.probe.drops.fetch_add(1, Ordering::SeqCst);
            if let Some(session) = self.probe.session.get() {
                if session.try_lock().is_err() {
                    self.probe.drops_while_locked.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
    }

    impl CsvDraft for Probed<CsvDraftHandle> {
        fn file_name(&self) -> &str {
            self.inner.file_name()
        }
        fn preview(&self) -> CsvPreview {
            self.inner.preview()
        }
        fn target_info(&self, target: &str) -> Result<TargetInfo, CommandError> {
            self.inner.target_info(target)
        }
        fn finalize(
            &self,
            target: &str,
            task: TaskChoice,
            cancel: &AtomicBool,
        ) -> Result<CsvFinal, CommandError> {
            let mut done = self.inner.finalize(target, task, cancel)?;
            done.dataset = Arc::new(Probed {
                inner: done.dataset,
                probe: self.probe.clone(),
            });
            Ok(done)
        }
    }

    impl DatasetView for Probed<DatasetHandle> {
        fn summary(&self) -> DatasetSummary {
            self.inner.summary()
        }
        fn preview(&self, rows: usize) -> TablePreview {
            self.inner.preview(rows)
        }
        fn describe(&self) -> DescribeTable {
            self.probe.describes.fetch_add(1, Ordering::SeqCst);
            self.inner.describe()
        }
        fn target_distribution(&self) -> TargetDistribution {
            self.probe.distributions.fetch_add(1, Ordering::SeqCst);
            self.inner.target_distribution()
        }
        fn correlation(&self, cancel: &AtomicBool) -> Result<CorrelationMatrix, CommandError> {
            self.inner.correlation(cancel)
        }
        fn pairplot(
            &self,
            features: &[String],
            diag: DiagKind,
        ) -> Result<PairplotData, CommandError> {
            self.inner.pairplot(features, diag)
        }
        fn train(
            &self,
            spec: &crate::core_facade::TrainSpec,
            progress: &crate::core_facade::Progress<'_>,
            cancel: &AtomicBool,
        ) -> Result<crate::core_facade::ModelHandle, CommandError> {
            self.inner.train(spec, progress, cancel)
        }
    }

    struct ProbeFacade {
        inner: FixtureFacade,
        probe: Arc<LockProbe>,
    }

    impl CoreFacade for ProbeFacade {
        fn name(&self) -> &'static str {
            "ProbeFacade"
        }
        fn builtin_catalog(&self) -> Result<Vec<DatasetEntry>, CommandError> {
            self.inner.builtin_catalog()
        }
        fn load_builtin(&self, id: DatasetId) -> Result<DatasetHandle, CommandError> {
            self.inner.load_builtin(id)
        }
        fn read_csv(
            &self,
            file_name: &str,
            bytes: &[u8],
            cancel: &AtomicBool,
        ) -> Result<CsvDraftHandle, CommandError> {
            Ok(Arc::new(Probed {
                inner: self.inner.read_csv(file_name, bytes, cancel)?,
                probe: self.probe.clone(),
            }))
        }
    }

    fn probed() -> (DataState, Arc<LockProbe>) {
        let probe = Arc::new(LockProbe::default());
        let data = DataState::new(Arc::new(ProbeFacade {
            inner: FixtureFacade::from_checkout(),
            probe: probe.clone(),
        }));
        assert!(probe.session.set(data.session.clone()).is_ok());
        (data, probe)
    }

    /// Big drafts and CSV tables are freed after releasing the session lock, so other
    /// commands are not blocked meanwhile.
    #[test]
    fn released_drafts_and_csvs_are_dropped_outside_the_lock() {
        let (data, probe) = probed();
        let a = data.load_csv_bytes("a.csv", TIPS_CSV, &go()).unwrap();
        data.set_csv_target(a.draft_id, "size", TaskChoice::Auto, 1, &go())
            .unwrap();
        // Same file again: the draft and the registered entry are replaced.
        let again = data.load_csv_bytes("a.csv", TIPS_CSV, &go()).unwrap();
        data.set_csv_target(again.draft_id, "tip", TaskChoice::Auto, 2, &go())
            .unwrap();
        // Another file: the previous CSV goes.
        let b = data.load_csv_bytes("b.csv", TIPS_CSV, &go()).unwrap();
        data.set_csv_target(b.draft_id, "size", TaskChoice::Auto, 3, &go())
            .unwrap();
        assert!(data.discard_csv(b.draft_id));
        // a's draft, a's dataset, again's draft, again's dataset (replaced by b), b's draft.
        assert_eq!(probe.drops.load(Ordering::SeqCst), 5);
        assert_eq!(probe.drops_while_locked.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn describe_and_target_distribution_of_a_csv_are_cached() {
        let (data, probe) = probed();
        let a = data.load_csv_bytes("a.csv", TIPS_CSV, &go()).unwrap();
        let loaded = data
            .set_csv_target(a.draft_id, "size", TaskChoice::Auto, 1, &go())
            .unwrap();
        let csv = loaded.dataset.dataset;
        let first = data.describe(csv).unwrap();
        assert_eq!(data.describe(csv).unwrap(), first);
        let dist = data.target_distribution(csv).unwrap();
        assert_eq!(data.target_distribution(csv).unwrap(), dist);
        assert_eq!(probe.describes.load(Ordering::SeqCst), 1);
        assert_eq!(probe.distributions.load(Ordering::SeqCst), 1);
        // A new registration (another target) computes them again.
        data.set_csv_target(a.draft_id, "tip", TaskChoice::Auto, 2, &go())
            .unwrap();
        assert_ne!(data.target_distribution(csv).unwrap(), dist);
        assert_eq!(probe.distributions.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn the_last_applied_seq_is_exposed() {
        let data = data();
        assert_eq!(data.selection().last_seq, 0);
        data.select(builtin(DatasetId::Wine), 10).unwrap();
        data.select(builtin(DatasetId::Iris), 4).unwrap();
        assert_eq!(
            data.selection(),
            SelectedDataset {
                dataset: builtin(DatasetId::Wine),
                last_seq: 10
            }
        );
        let a = data.load_csv_bytes("a.csv", TIPS_CSV, &go()).unwrap();
        data.set_csv_target(a.draft_id, "size", TaskChoice::Auto, 20, &go())
            .unwrap();
        assert_eq!(data.selection().last_seq, 20);
    }

    #[test]
    fn stale_drafts_and_unknown_columns_are_errors() {
        let data = data();
        let first = data.load_csv_bytes("a.csv", TIPS_CSV, &go()).unwrap();
        let second = data.load_csv_bytes("b.csv", TIPS_CSV, &go()).unwrap();
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
            data.load_csv_bytes("x.csv", b"", &go()),
            Err(CommandError::CsvEmpty { .. })
        ));
        assert!(matches!(
            data.load_csv_bytes("x.csv", b"a\n1\n", &go()),
            Err(CommandError::CsvTooFewColumns { .. })
        ));
        assert!(matches!(
            data.load_csv_bytes("x.csv", b"a,b\n1,2\n1,2,3\n", &go()),
            Err(CommandError::CsvMalformed { line: 3, .. })
        ));
        assert!(matches!(
            data.load_csv(Path::new("/no/existe.csv"), &go()),
            Err(CommandError::CsvIo { .. })
        ));
    }

    #[test]
    fn reads_regular_files_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("propinas.csv");
        std::fs::write(&path, TIPS_CSV).unwrap();
        let data = data();
        let preview = data.load_csv(&path, &go()).unwrap();
        assert_eq!(preview.file_name, "propinas.csv");
        assert_eq!(preview.n_rows, 4);
    }

    #[test]
    fn only_regular_files_within_the_limit_are_read() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            read_capped(dir.path(), CSV_MAX_BYTES, &go()),
            Err(CommandError::CsvIo { .. })
        ));
        let path = dir.path().join("big.csv");
        std::fs::write(&path, vec![b'a'; 64]).unwrap();
        assert!(matches!(
            read_capped(&path, 63, &go()),
            Err(CommandError::CsvTooLarge {
                bytes: 64,
                limit: 63,
                ..
            })
        ));
        assert_eq!(read_capped(&path, 64, &go()).unwrap().len(), 64);
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            read_capped(&path, 64, &cancelled),
            Err(CommandError::Cancelled { .. })
        ));
    }

    /// A FIFO would block `open` forever; it is rejected from its metadata.
    #[cfg(unix)]
    #[test]
    fn fifos_are_rejected_without_blocking() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("tuberia.csv");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(made.success());
        assert!(matches!(
            read_capped(&fifo, CSV_MAX_BYTES, &go()),
            Err(CommandError::CsvIo { .. })
        ));
        assert!(matches!(
            read_capped(Path::new("/dev/zero"), CSV_MAX_BYTES, &go()),
            Err(CommandError::CsvIo { .. })
        ));
    }

    #[test]
    fn cancelled_operations_report_cancelled() {
        let data = data();
        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            data.load_csv_bytes("tips.csv", TIPS_CSV, &cancelled),
            Err(CommandError::Cancelled { .. })
        ));
        assert!(matches!(
            data.correlation(DatasetRef::default(), &cancelled),
            Err(CommandError::Cancelled { .. })
        ));
        let preview = data.load_csv_bytes("tips.csv", TIPS_CSV, &go()).unwrap();
        assert!(matches!(
            data.set_csv_target(preview.draft_id, "size", TaskChoice::Auto, 1, &cancelled),
            Err(CommandError::Cancelled { .. })
        ));
        assert_eq!(data.list_datasets().unwrap().len(), 12, "not registered");
        let task = data.tasks.start(3);
        assert!(data.tasks.cancel(3));
        assert!(task.token().load(Ordering::SeqCst));
        assert!(data.correlation(DatasetRef::default(), &go()).is_ok());
    }

    #[test]
    fn the_scenario_path_only_comes_from_the_screenshot_command_line() {
        use crate::cli::Screenshot;
        let shot = |scenario: Option<&str>| {
            let mut options = Screenshot::new("x.png");
            options.scenario = scenario.map(str::to_owned);
            Mode::Screenshot(options)
        };
        assert_eq!(
            scenario_csv_path(&shot(Some("csv_cargado_clasificacion:/tmp/a b.csv"))),
            Some("/tmp/a b.csv".to_owned())
        );
        assert_eq!(scenario_csv_path(&shot(Some("csv_vacio"))), None);
        assert_eq!(
            scenario_csv_path(&shot(Some("csv_cargado_clasificacion:"))),
            None
        );
        assert_eq!(scenario_csv_path(&shot(None)), None);
        assert_eq!(scenario_csv_path(&Mode::Normal), None);
        assert_eq!(scenario_csv_path(&Mode::SmokeTest), None);
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
