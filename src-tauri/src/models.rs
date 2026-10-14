//! Model commands of fase2-regresion.md §5.4: training and the views of tabs 2–6.
//!
//! Each lesson keeps at most one trained model ([`LessonModels`]). Every training gets a
//! new `model_id`; the views receive it and fail with `ModelStale` once the lesson has
//! been retrained, so an old answer never reaches a newer screen. Training requests carry
//! the `seq` of the UI: a newer training cancels the older one, which ends with
//! `Cancelled` whatever the order in which they finish.
//!
//! All the work runs in `spawn_blocking` (the facade may be slow) and the lock is never
//! held while computing. A replaced model is dropped after releasing the lock.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard};

use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

use crate::core_facade::{check_cancelled, ModelHandle, TrainSpec};
use crate::data::{blocking, DataState};
use crate::ipc::{
    CodeExtra, CodeTemplate, CoefficientsView, CommandError, DatasetRef, DecisionGrid, Evaluation,
    ExportFormat, GridRequest, LessonId, ModelConfig, PredictRequest, PredictionForm,
    PredictionResult, ProbabilityView, ResidualsView, RocData, TaskProgress, TaskType, TrainNotice,
    TrainRequest, TrainedModelInfo,
};
use crate::tasks::CancelToken;

/// Event with the progress of long tasks (training, grids).
pub const PROGRESS_EVENT: &str = "task-progress";

/// Message of D26 (logistic regression on a regression target).
pub const LOGISTIC_NEEDS_CLASSES: &str = "La regresión logística necesita una variable objetivo con clases. Elige 📈 Regresión Lineal o un dataset de clasificación.";

/// Message of D19: `[entrenamiento] aviso_no_convergencia` of
/// `content/regresion/mensajes.toml` (the same text the UI shows), with `{max_iter}`.
pub fn not_converged_message(max_iter: u32) -> String {
    static TEMPLATE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let template = TEMPLATE.get_or_init(|| {
        mltutor_core::content::Messages::regression()
            .and_then(|m| {
                m.text("entrenamiento", "aviso_no_convergencia")
                    .map(str::to_owned)
            })
            .unwrap_or_else(|error| {
                log::error!("contenido: {error}");
                "⚠️ El optimizador no ha convergido en {max_iter} iteraciones.".to_owned()
            })
    });
    template.replace("{max_iter}", &max_iter.to_string())
}

/// A model of a lesson and what it was trained on.
pub struct TrainedModel {
    pub id: u32,
    pub lesson: LessonId,
    pub dataset: DatasetRef,
    pub dataset_label: String,
    pub config: ModelConfig,
    pub test_size: f64,
    pub model: ModelHandle,
}

impl TrainedModel {
    pub fn info(&self) -> TrainedModelInfo {
        let summary = self.model.summary();
        TrainedModelInfo {
            model_id: self.id,
            lesson: self.lesson,
            dataset: self.dataset,
            dataset_label: self.dataset_label.clone(),
            config: self.config,
            test_size: self.test_size,
            model_task: summary.model_task,
            dataset_task: summary.dataset_task,
            n_train: crate::ipc::to_u32(summary.n_train),
            n_test: crate::ipc::to_u32(summary.n_test),
            feature_columns: summary.feature_columns,
            feature_names: summary.feature_names,
            class_names: summary.class_names,
            model_classes: summary
                .model_classes
                .map(|c| c.into_iter().map(crate::ipc::to_u32).collect()),
            notices: with_mismatch(summary.notices, summary.model_task, summary.dataset_task),
        }
    }
}

/// Adds the warning of a model type that does not match the dataset (as in Python, the
/// model is trained anyway: Lineal on Iris, D25).
fn with_mismatch(
    mut notices: Vec<TrainNotice>,
    model_task: TaskType,
    dataset_task: TaskType,
) -> Vec<TrainNotice> {
    if model_task != dataset_task
        && !notices
            .iter()
            .any(|n| matches!(n, TrainNotice::ModelTaskMismatch { .. }))
    {
        let message = match dataset_task {
            TaskType::Classification => "Este dataset parece ser más adecuado para Regresión Logística. La selección actual podría no ofrecer resultados óptimos.",
            TaskType::Regression => "Este dataset parece ser más adecuado para Regresión Lineal. La selección actual podría no ofrecer resultados óptimos.",
        };
        notices.push(TrainNotice::ModelTaskMismatch {
            message: message.to_owned(),
        });
    }
    notices
}

/// Why a finished training was not stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    /// A newer training of the same lesson started meanwhile.
    Superseded,
}

/// Trained models by lesson and the training requests in flight (§5.2).
#[derive(Default)]
pub struct LessonModels {
    slots: HashMap<LessonId, Arc<TrainedModel>>,
    next_id: u32,
    /// Newest `seq` ever accepted per lesson. Never removed (like `session.select`): a
    /// request that arrives after a newer one has finished must still be rejected.
    last_seq: HashMap<LessonId, u64>,
    /// The training in flight per lesson (its `seq` and cancel token).
    pending: HashMap<LessonId, (u64, CancelToken)>,
}

impl LessonModels {
    /// Registers a training request. A newer request cancels the one in flight; an older
    /// or repeated one (it arrived late) is rejected with `false`.
    pub fn begin(&mut self, lesson: LessonId, seq: u64, token: CancelToken) -> bool {
        if self.last_seq.get(&lesson).is_some_and(|&last| seq <= last) {
            return false;
        }
        self.last_seq.insert(lesson, seq);
        if let Some((_, old)) = self.pending.insert(lesson, (seq, token)) {
            old.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        true
    }

    /// Stores the model of request `seq` if it is still the newest one. Returns the new
    /// model and the replaced one (to be dropped outside the lock).
    pub fn finish(
        &mut self,
        lesson: LessonId,
        seq: u64,
        make: impl FnOnce(u32) -> TrainedModel,
    ) -> Result<(Arc<TrainedModel>, Option<Arc<TrainedModel>>), StoreError> {
        let newest = self.last_seq.get(&lesson) == Some(&seq);
        if !newest || self.pending.get(&lesson).map(|(s, _)| *s) != Some(seq) {
            return Err(StoreError::Superseded);
        }
        self.pending.remove(&lesson);
        self.next_id = self.next_id.wrapping_add(1).max(1);
        let model = Arc::new(make(self.next_id));
        let old = self.slots.insert(lesson, model.clone());
        Ok((model, old))
    }

    /// The training of `seq` ended without a model (error or cancellation). `last_seq`
    /// is kept, so the same or an older request cannot start again.
    pub fn abandon(&mut self, lesson: LessonId, seq: u64) {
        if self.pending.get(&lesson).map(|(s, _)| *s) == Some(seq) {
            self.pending.remove(&lesson);
        }
    }

    pub fn current(&self, lesson: LessonId) -> Option<Arc<TrainedModel>> {
        self.slots.get(&lesson).cloned()
    }

    /// The model with `id` if it is still the current model of its lesson.
    pub fn by_id(&self, id: u32) -> Option<Arc<TrainedModel>> {
        self.slots.values().find(|m| m.id == id).cloned()
    }
}

#[derive(Clone, Default)]
pub struct ModelState {
    models: Arc<Mutex<LessonModels>>,
}

impl ModelState {
    fn lock(&self) -> MutexGuard<'_, LessonModels> {
        // Every update is a single insertion or removal: a poisoned map is still valid.
        self.models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn model(&self, id: u32) -> Result<Arc<TrainedModel>, CommandError> {
        self.lock().by_id(id).ok_or_else(CommandError::model_stale)
    }

    pub fn lesson_model(&self, lesson: LessonId) -> Option<TrainedModelInfo> {
        let model = self.lock().current(lesson);
        model.map(|m| m.info())
    }

    /// Validates and trains `req` (blocking: call it from a worker).
    pub fn train(
        &self,
        data: &DataState,
        req: &TrainRequest,
        token: &CancelToken,
        progress: &(dyn Fn(&str, Option<f32>) + Sync),
    ) -> Result<TrainedModelInfo, CommandError> {
        validate_train(req)?;
        if !self.lock().begin(req.lesson, req.seq, token.clone()) {
            return Err(CommandError::cancelled());
        }
        let result = self.train_registered(data, req, token, progress);
        match result {
            Ok(info) => Ok(info),
            Err(error) => {
                self.lock().abandon(req.lesson, req.seq);
                Err(error)
            }
        }
    }

    fn train_registered(
        &self,
        data: &DataState,
        req: &TrainRequest,
        cancel: &AtomicBool,
        progress: &(dyn Fn(&str, Option<f32>) + Sync),
    ) -> Result<TrainedModelInfo, CommandError> {
        let (dataset, label) = data.resolve(req.dataset)?;
        if matches!(req.config, ModelConfig::Logistic { .. })
            && dataset.summary().task == TaskType::Regression
        {
            return Err(CommandError::invalid_target(LOGISTIC_NEEDS_CLASSES));
        }
        check_cancelled(cancel)?;
        let spec = TrainSpec {
            config: req.config,
            test_size: req.test_size,
            dataset_label: label.clone(),
        };
        let started = std::time::Instant::now();
        let model = dataset.train(&spec, progress, cancel)?;
        check_cancelled(cancel)?;
        let stored = self.lock().finish(req.lesson, req.seq, |id| TrainedModel {
            id,
            lesson: req.lesson,
            dataset: req.dataset,
            dataset_label: label,
            config: req.config,
            test_size: req.test_size,
            model,
        });
        let (model, old) = stored.map_err(|StoreError::Superseded| CommandError::cancelled())?;
        drop(old); // Outside the lock.
        log::info!(
            "modelo {} entrenado ({:?}, {:?}) en {} ms",
            model.id,
            req.lesson,
            req.config,
            started.elapsed().as_millis()
        );
        Ok(model.info())
    }
}

/// Range of the "Máximo de Iteraciones" slider.
const MIN_MAX_ITER: u32 = 100;
const MAX_MAX_ITER: u32 = 2000;

fn validate_train(req: &TrainRequest) -> Result<(), CommandError> {
    // Same ranges as the sliders and the messages (contract §5.3); a small epsilon
    // accepts the float steps of the slider (0.1 + 0.1 + 0.1…).
    const EPS: f64 = 1e-9;
    if !(req.test_size.is_finite() && (0.1 - EPS..=0.5 + EPS).contains(&req.test_size)) {
        return Err(CommandError::invalid_selection(
            "El tamaño del conjunto de prueba debe estar entre 0,1 y 0,5.",
        ));
    }
    if let ModelConfig::Logistic { max_iter } = req.config {
        if !(MIN_MAX_ITER..=MAX_MAX_ITER).contains(&max_iter) {
            return Err(CommandError::invalid_selection(
                "El máximo de iteraciones debe estar entre 100 y 2000.",
            ));
        }
    }
    Ok(())
}

fn validate_pair(model: &TrainedModel, x: u32, y: u32) -> Result<(usize, usize), CommandError> {
    let n = model.model.summary().feature_columns.len();
    let (x, y) = (x as usize, y as usize);
    if x >= n || y >= n {
        return Err(CommandError::invalid_selection(
            "La característica elegida no existe en este modelo.",
        ));
    }
    if x == y {
        return Err(CommandError::invalid_selection(
            "Elige dos características distintas.",
        ));
    }
    Ok((x, y))
}

/// Checks the values of the prediction form (one finite number per feature).
pub fn validate_values(values: &[f64], n_features: usize) -> Result<(), CommandError> {
    if values.len() != n_features {
        return Err(CommandError::invalid_input(format!(
            "Se esperaban {n_features} valores y se recibieron {}.",
            values.len()
        )));
    }
    if let Some(position) = values.iter().position(|v| !v.is_finite()) {
        return Err(CommandError::invalid_input(format!(
            "El valor de la característica {} no es un número válido.",
            position + 1
        )));
    }
    Ok(())
}

/// Default file name and save-dialog filter of each export (§5.4).
fn export_target(
    config: ModelConfig,
    format: ExportFormat,
) -> (&'static str, &'static str, &'static str) {
    match (format, config) {
        (ExportFormat::Onnx, ModelConfig::Linear) => {
            ("regresion_lineal.onnx", "Modelo ONNX (*.onnx)", "onnx")
        }
        (ExportFormat::Onnx, ModelConfig::Logistic { .. }) => {
            ("regresion_logistica.onnx", "Modelo ONNX (*.onnx)", "onnx")
        }
        (ExportFormat::PythonPackage, _) => {
            ("modelo_regresion.zip", "Paquete Python (*.zip)", "zip")
        }
    }
}

fn models(state: &State<'_, ModelState>) -> ModelState {
    state.inner().clone()
}

/// Minimum time between two `task-progress` events with the same integer percentage.
const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Rate limit of `task-progress`: L-BFGS reports every iteration (thousands per second),
/// but the UI only needs a new event when the integer percentage or the message changes,
/// or every [`PROGRESS_INTERVAL`] as a heartbeat.
#[derive(Default)]
pub struct ProgressThrottle {
    last: Mutex<Option<(Option<i32>, String, std::time::Instant)>>,
}

impl ProgressThrottle {
    /// Whether an update must be emitted now (and records it if so).
    pub fn admit(&self, message: &str, percent: Option<f32>, now: std::time::Instant) -> bool {
        // A poisoned throttle only means a panicked emitter: its last value is still valid.
        let mut last = self.last.lock().unwrap_or_else(|p| p.into_inner());
        let whole = percent.map(|p| p.floor() as i32);
        let emit = match last.as_ref() {
            None => true,
            Some((previous, text, at)) => {
                *previous != whole
                    || text != message
                    || now.saturating_duration_since(*at) >= PROGRESS_INTERVAL
            }
        };
        if emit {
            *last = Some((whole, message.to_owned(), now));
        }
        emit
    }
}

fn emit_progress(app: &AppHandle, task_id: u32, message: &str, percent: Option<f32>) {
    let payload = TaskProgress {
        task_id,
        message: message.to_owned(),
        percent,
    };
    if let Err(error) = app.emit(PROGRESS_EVENT, payload) {
        log::debug!("no se pudo emitir el progreso: {error}");
    }
}

#[tauri::command]
pub async fn train_model(
    app: AppHandle,
    data: State<'_, DataState>,
    models: State<'_, ModelState>,
    req: TrainRequest,
) -> Result<TrainedModelInfo, CommandError> {
    let data = data.inner().clone();
    let models = self::models(&models);
    let guard = data.tasks.start(req.task_id);
    let task_id = req.task_id;
    blocking(move || {
        let throttle = ProgressThrottle::default();
        let progress = |message: &str, percent: Option<f32>| {
            if throttle.admit(message, percent, std::time::Instant::now()) {
                emit_progress(&app, task_id, message, percent);
            }
        };
        progress("Entrenando modelo...", Some(0.0));
        let result = models.train(&data, &req, &guard.token(), &progress);
        if result.is_ok() {
            progress("Modelo entrenado", Some(100.0));
        }
        result
    })
    .await
}

#[tauri::command]
pub async fn lesson_model(
    models: State<'_, ModelState>,
    lesson: LessonId,
) -> Result<Option<TrainedModelInfo>, CommandError> {
    Ok(models.lesson_model(lesson))
}

/// Runs `view` on the model `model_id` in a blocking worker.
async fn with_model<T, F>(
    name: &'static str,
    models: &State<'_, ModelState>,
    model_id: u32,
    view: F,
) -> Result<T, CommandError>
where
    T: Send + 'static,
    F: FnOnce(&TrainedModel) -> Result<T, CommandError> + Send + 'static,
{
    let models = self::models(models);
    blocking(move || {
        let model = models.model(model_id)?;
        let started = std::time::Instant::now();
        let result = view(&model);
        log::debug!(
            "{name} del modelo {model_id}: {:.1} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
        result
    })
    .await
}

#[tauri::command]
pub async fn evaluate_model(
    models: State<'_, ModelState>,
    model_id: u32,
) -> Result<Evaluation, CommandError> {
    with_model("evaluate_model", &models, model_id, |m| m.model.evaluate()).await
}

#[tauri::command]
pub async fn roc_curves(
    models: State<'_, ModelState>,
    model_id: u32,
) -> Result<RocData, CommandError> {
    with_model("roc_curves", &models, model_id, |m| m.model.roc_curves()).await
}

#[tauri::command]
pub async fn probability_view(
    models: State<'_, ModelState>,
    model_id: u32,
) -> Result<ProbabilityView, CommandError> {
    with_model("probability_view", &models, model_id, |m| {
        m.model.probability_view()
    })
    .await
}

#[tauri::command]
pub async fn residuals_view(
    models: State<'_, ModelState>,
    model_id: u32,
) -> Result<ResidualsView, CommandError> {
    with_model("residuals_view", &models, model_id, |m| {
        m.model.residuals_view()
    })
    .await
}

#[tauri::command]
pub async fn decision_grid(
    app: AppHandle,
    data: State<'_, DataState>,
    models: State<'_, ModelState>,
    req: GridRequest,
) -> Result<DecisionGrid, CommandError> {
    let guard = data.tasks.start(req.task_id);
    let task_id = req.task_id;
    with_model("decision_grid", &models, req.model_id, move |m| {
        let (x, y) = validate_pair(m, req.feature_x, req.feature_y)?;
        let throttle = ProgressThrottle::default();
        let progress = |message: &str, percent: Option<f32>| {
            if throttle.admit(message, percent, std::time::Instant::now()) {
                emit_progress(&app, task_id, message, percent);
            }
        };
        m.model.decision_grid(x, y, &progress, &guard.token())
    })
    .await
}

#[tauri::command]
pub async fn coefficients_view(
    models: State<'_, ModelState>,
    model_id: u32,
    class_index: Option<u32>,
) -> Result<CoefficientsView, CommandError> {
    with_model("coefficients_view", &models, model_id, move |m| {
        let class = match class_index {
            None => None,
            // A row of `coef_`: indexed by `classes_`, not by the dataset's classes.
            Some(index) => {
                let n = m.model.summary().model_classes.map_or(0, |c| c.len());
                if index as usize >= n.max(1) {
                    return Err(CommandError::invalid_selection(
                        "La clase elegida no existe en este modelo.",
                    ));
                }
                Some(index as usize)
            }
        };
        m.model.coefficients(class)
    })
    .await
}

#[tauri::command]
pub async fn prediction_form(
    models: State<'_, ModelState>,
    model_id: u32,
) -> Result<PredictionForm, CommandError> {
    with_model("prediction_form", &models, model_id, |m| {
        m.model.prediction_form()
    })
    .await
}

#[tauri::command]
pub async fn predict_one(
    models: State<'_, ModelState>,
    req: PredictRequest,
) -> Result<PredictionResult, CommandError> {
    with_model("predict_one", &models, req.model_id, move |m| {
        validate_values(&req.values, m.model.summary().feature_columns.len())?;
        m.model.predict_one(&req.values)
    })
    .await
}

#[tauri::command]
pub async fn generate_code(
    models: State<'_, ModelState>,
    model_id: u32,
    template: CodeTemplate,
    extra: Option<CodeExtra>,
) -> Result<String, CommandError> {
    with_model("generate_code", &models, model_id, move |m| {
        let pair = match extra {
            Some(extra) => Some(validate_pair(m, extra.feature_x, extra.feature_y)?),
            None => None,
        };
        m.model.code(template, pair)
    })
    .await
}

/// Builds the ONNX file or the Python package in Rust and saves it with the native
/// dialog: the path never goes through JS. `None` if the user cancels the dialog.
#[tauri::command]
pub async fn export_model(
    app: AppHandle,
    models: State<'_, ModelState>,
    model_id: u32,
    format: ExportFormat,
) -> Result<Option<String>, CommandError> {
    let models = self::models(&models);
    blocking(move || {
        let model = models.model(model_id)?;
        let bytes = model.model.export(format)?;
        let (name, filter, extension) = export_target(model.config, format);
        drop(model);
        let Some(path) = ask_save_path(&app, name, filter, extension)? else {
            return Ok(None);
        };
        std::fs::write(&path, &bytes).map_err(CommandError::file_save)?;
        log::info!(
            "modelo exportado: {} ({} bytes)",
            path.display(),
            bytes.len()
        );
        Ok(Some(
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
        ))
    })
    .await
}

/// Save dialog with the filter of the export; adds the extension if the user removed it
/// and confirms the overwrite of that final name (GTK only asks about the typed name).
fn ask_save_path(
    app: &AppHandle,
    name: &str,
    filter: &str,
    extension: &str,
) -> Result<Option<std::path::PathBuf>, CommandError> {
    let window = app.get_webview_window("main");
    loop {
        let mut dialog = app
            .dialog()
            .file()
            .set_file_name(name)
            .add_filter(filter, &[extension]);
        if let Some(window) = &window {
            dialog = dialog.set_parent(window);
        }
        let Some(picked) = dialog.blocking_save_file() else {
            return Ok(None);
        };
        let mut path = picked.into_path().map_err(CommandError::file_save)?;
        if path.extension().is_some() {
            return Ok(Some(path));
        }
        path.set_extension(extension);
        if !path.exists() {
            return Ok(Some(path));
        }
        let file = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut question = app
            .dialog()
            .message(format!(
                "Ya existe un archivo llamado «{file}». ¿Quieres reemplazarlo?"
            ))
            .title("Guardar archivo")
            .kind(MessageDialogKind::Warning)
            .buttons(MessageDialogButtons::OkCancelCustom(
                "Reemplazar".to_owned(),
                "Cancelar".to_owned(),
            ));
        if let Some(window) = &window {
            question = question.parent(window);
        }
        if question.blocking_show() {
            return Ok(Some(path));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::fixture_facade::FixtureFacade;
    use crate::ipc::{DatasetId, DatasetRef};

    fn iris() -> DatasetRef {
        DatasetRef::Builtin {
            id: DatasetId::Iris,
        }
    }

    fn request(dataset: DatasetRef, config: ModelConfig, seq: u64) -> TrainRequest {
        TrainRequest {
            lesson: LessonId::Regression,
            dataset,
            config,
            test_size: 0.3,
            seq,
            task_id: 1,
        }
    }

    fn setup() -> (DataState, ModelState) {
        (
            DataState::new(Arc::new(FixtureFacade::from_checkout())),
            ModelState::default(),
        )
    }

    fn quiet(_: &str, _: Option<f32>) {}

    #[test]
    fn newer_trainings_cancel_older_ones() {
        let mut models = LessonModels::default();
        let first = CancelToken::default();
        let second = CancelToken::default();
        assert!(models.begin(LessonId::Regression, 1, first.clone()));
        assert!(models.begin(LessonId::Regression, 2, second.clone()));
        assert!(
            first.load(Ordering::SeqCst),
            "the older training is cancelled"
        );
        assert!(!models.begin(LessonId::Regression, 1, CancelToken::default()));
        let make = |lesson| {
            move |id| TrainedModel {
                id,
                lesson,
                dataset: DatasetRef::default(),
                dataset_label: String::new(),
                config: ModelConfig::Linear,
                test_size: 0.3,
                model: dummy(),
            }
        };
        assert_eq!(
            models
                .finish(LessonId::Regression, 1, make(LessonId::Regression))
                .err(),
            Some(StoreError::Superseded)
        );
        let (model, old) = models
            .finish(LessonId::Regression, 2, make(LessonId::Regression))
            .unwrap();
        assert!(old.is_none());
        assert_eq!(models.by_id(model.id).map(|m| m.id), Some(model.id));
        // Retraining replaces the model: the old id becomes stale.
        assert!(models.begin(LessonId::Regression, 3, CancelToken::default()));
        let (newer, old) = models
            .finish(LessonId::Regression, 3, make(LessonId::Regression))
            .unwrap();
        assert_eq!(old.map(|m| m.id), Some(model.id));
        assert!(models.by_id(model.id).is_none());
        assert!(models.by_id(newer.id).is_some());
    }

    fn blank(id: u32) -> TrainedModel {
        TrainedModel {
            id,
            lesson: LessonId::Regression,
            dataset: DatasetRef::default(),
            dataset_label: String::new(),
            config: ModelConfig::Linear,
            test_size: 0.3,
            model: dummy(),
        }
    }

    /// The most adverse order: the newer request starts and finishes before the older
    /// one even registers. The late one must not replace the newer model.
    #[test]
    fn a_late_older_training_never_replaces_a_finished_newer_one() {
        let mut models = LessonModels::default();
        assert!(models.begin(LessonId::Regression, 20, CancelToken::default()));
        let (newer, _) = models.finish(LessonId::Regression, 20, blank).unwrap();
        // Nothing is pending now; the late request 10 (and a repeated 20) are rejected.
        assert!(!models.begin(LessonId::Regression, 10, CancelToken::default()));
        assert!(!models.begin(LessonId::Regression, 20, CancelToken::default()));
        assert_eq!(
            models.finish(LessonId::Regression, 10, blank).err(),
            Some(StoreError::Superseded)
        );
        assert_eq!(
            models.current(LessonId::Regression).map(|m| m.id),
            Some(newer.id)
        );
        // A failed newer request keeps the barrier too.
        assert!(models.begin(LessonId::Regression, 30, CancelToken::default()));
        models.abandon(LessonId::Regression, 30);
        assert!(!models.begin(LessonId::Regression, 25, CancelToken::default()));
        // Other lessons have their own sequence.
        assert!(models.begin(LessonId::Knn, 1, CancelToken::default()));
        assert!(models.begin(LessonId::Regression, 31, CancelToken::default()));
    }

    /// Same order through `ModelState::train`, with real trainings.
    #[test]
    fn late_training_requests_are_cancelled_end_to_end() {
        let (data, models) = setup();
        let diabetes = DatasetRef::Builtin {
            id: DatasetId::Diabetes,
        };
        let newer = models
            .train(
                &data,
                &request(diabetes, ModelConfig::Linear, 5),
                &CancelToken::default(),
                &quiet,
            )
            .unwrap();
        let late = models.train(
            &data,
            &request(iris(), ModelConfig::Logistic { max_iter: 1000 }, 4),
            &CancelToken::default(),
            &quiet,
        );
        assert_eq!(late, Err(CommandError::cancelled()));
        assert_eq!(models.lesson_model(LessonId::Regression), Some(newer));
    }

    fn dummy() -> ModelHandle {
        let (data, _) = setup();
        let (dataset, _) = data.resolve(DatasetRef::default()).unwrap();
        let spec = TrainSpec {
            config: ModelConfig::Logistic { max_iter: 1000 },
            test_size: 0.3,
            dataset_label: String::new(),
        };
        dataset
            .train(&spec, &quiet, &AtomicBool::new(false))
            .unwrap()
    }

    #[test]
    fn trains_and_serves_views_until_retrained() {
        let (data, models) = setup();
        let token = CancelToken::default();
        let info = models
            .train(
                &data,
                &request(iris(), ModelConfig::Logistic { max_iter: 1000 }, 1),
                &token,
                &quiet,
            )
            .unwrap();
        assert_eq!(info.model_task, TaskType::Classification);
        assert_eq!((info.n_train, info.n_test), (105, 45));
        assert_eq!(
            models.lesson_model(LessonId::Regression),
            Some(info.clone())
        );
        assert!(models.model(info.model_id).is_ok());
        let again = models
            .train(
                &data,
                &request(iris(), ModelConfig::Logistic { max_iter: 1000 }, 2),
                &token,
                &quiet,
            )
            .unwrap();
        assert_ne!(again.model_id, info.model_id);
        assert!(matches!(
            models.model(info.model_id).err(),
            Some(CommandError::ModelStale { .. })
        ));
    }

    #[test]
    fn logistic_on_a_regression_target_is_blocked() {
        let (data, models) = setup();
        let diabetes = DatasetRef::Builtin {
            id: DatasetId::Diabetes,
        };
        let error = models
            .train(
                &data,
                &request(diabetes, ModelConfig::Logistic { max_iter: 1000 }, 1),
                &CancelToken::default(),
                &quiet,
            )
            .unwrap_err();
        assert_eq!(error, CommandError::invalid_target(LOGISTIC_NEEDS_CLASSES));
        // The failed request does not block the next one.
        assert!(models
            .train(
                &data,
                &request(diabetes, ModelConfig::Linear, 2),
                &CancelToken::default(),
                &quiet
            )
            .is_ok());
    }

    #[test]
    fn invalid_requests_are_rejected() {
        let mut bad = request(iris(), ModelConfig::Linear, 1);
        bad.test_size = f64::NAN;
        assert!(validate_train(&bad).is_err());
        for test_size in [0.05, 0.09, 0.51, 0.95] {
            bad.test_size = test_size;
            assert!(validate_train(&bad).is_err(), "{test_size}");
        }
        for test_size in [0.1, 0.1 + 0.1 + 0.1, 0.5] {
            bad.test_size = test_size;
            assert!(validate_train(&bad).is_ok(), "{test_size}");
        }
        for max_iter in [0, 1, 99, 2001, 100_000] {
            bad.config = ModelConfig::Logistic { max_iter };
            assert!(validate_train(&bad).is_err(), "{max_iter}");
        }
        for max_iter in [100, 1000, 2000] {
            bad.config = ModelConfig::Logistic { max_iter };
            assert!(validate_train(&bad).is_ok(), "{max_iter}");
        }
        assert!(validate_values(&[1.0, f64::NAN], 2).is_err());
        assert!(validate_values(&[1.0], 2).is_err());
        assert!(validate_values(&[1.0, 2.0], 2).is_ok());
    }

    #[test]
    fn progress_is_throttled_by_whole_percent_or_time() {
        let throttle = ProgressThrottle::default();
        let t0 = std::time::Instant::now();
        let ms = |n: u64| t0 + std::time::Duration::from_millis(n);
        let msg = "Entrenando modelo...";
        assert!(throttle.admit(msg, Some(0.0), t0));
        // Same whole percent within 100 ms: dropped.
        assert!(!throttle.admit(msg, Some(0.4), ms(10)));
        assert!(!throttle.admit(msg, Some(0.9), ms(99)));
        // A new whole percent: emitted at once.
        assert!(throttle.admit(msg, Some(1.0), ms(100)));
        assert!(!throttle.admit(msg, Some(1.5), ms(150)));
        // Heartbeat after 100 ms without changes.
        assert!(throttle.admit(msg, Some(1.6), ms(200)));
        // A thousand iterations in the same millisecond produce at most 100 events.
        let emitted = (0..1000)
            .filter(|&i| throttle.admit(msg, Some(i as f32 / 10.0), ms(201)))
            .count();
        assert!(emitted <= 100, "{emitted}");
        // A different message (or the final 100 %) always goes through.
        assert!(throttle.admit("Modelo entrenado", Some(100.0), ms(201)));
    }

    #[test]
    fn not_converged_message_comes_from_the_content() {
        let message = not_converged_message(300);
        assert!(message.contains("300 iteraciones"), "{message}");
        assert!(!message.contains("{max_iter}"));
        let toml = mltutor_core::content::Messages::regression().unwrap();
        let template = toml.text("entrenamiento", "aviso_no_convergencia").unwrap();
        assert_eq!(message, template.replace("{max_iter}", "300"));
    }

    #[test]
    fn export_names_follow_the_contract() {
        assert_eq!(
            export_target(ModelConfig::Linear, ExportFormat::Onnx).0,
            "regresion_lineal.onnx"
        );
        assert_eq!(
            export_target(
                ModelConfig::Logistic { max_iter: 100 },
                ExportFormat::PythonPackage
            ),
            ("modelo_regresion.zip", "Paquete Python (*.zip)", "zip")
        );
    }
}
