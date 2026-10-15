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
    ExportFormat, FeatureImportanceView, GridRequest, ImportanceMethod, LessonId, ModelConfig,
    PredictRequest, PredictionForm, PredictionResult, ProbabilityView, ResidualsView, RocData,
    RulesRequest, TaskProgress, TaskType, TrainNotice, TrainRequest, TrainedModelInfo, TreeDiagram,
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

/// `content/arboles/mensajes.toml`: texts of 🌲 Árboles de Decisión that the backend
/// sends (D37 error, type mismatch warning). The UI reads the same file.
const TREE_MESSAGES: &str = include_str!("../../content/arboles/mensajes.toml");

/// A text of `content/arboles/mensajes.toml` (`fallback` if the file lacks it, logged).
pub fn tree_text(section: &str, key: &str, fallback: &str) -> String {
    static MESSAGES: std::sync::OnceLock<Option<mltutor_core::content::Messages>> =
        std::sync::OnceLock::new();
    let messages = MESSAGES.get_or_init(|| {
        mltutor_core::content::Messages::parse("content/arboles/mensajes.toml", TREE_MESSAGES)
            .map_err(|error| log::error!("contenido: {error}"))
            .ok()
    });
    match messages.as_ref().map(|m| m.text(section, key)) {
        Some(Ok(text)) => text.to_owned(),
        Some(Err(error)) => {
            log::error!("contenido: {error}");
            fallback.to_owned()
        }
        None => fallback.to_owned(),
    }
}

/// D37: a classification tree on a continuous target.
pub fn tree_needs_classes_message() -> String {
    tree_text(
        "entrenamiento",
        "error_clasificacion_regresion",
        "Un árbol de clasificación necesita una variable objetivo con clases.",
    )
}

/// Largest training set of a tree with `absolute_error`: the exact MAE criterion is
/// O(n² · features) per node and would take minutes (sklearn has the same cost).
pub const MAE_MAX_TRAIN: usize = 20_000;

/// `[entrenamiento] mae_limite`: rejects `absolute_error` with more than
/// [`MAE_MAX_TRAIN`] training rows (training and the 2D refit of the boundary).
pub fn check_mae_limit(config: ModelConfig, n_train: usize) -> Result<(), CommandError> {
    let mae = matches!(
        config,
        ModelConfig::Tree {
            criterion: crate::ipc::TreeCriterion::AbsoluteError,
            ..
        }
    );
    if mae && n_train > MAE_MAX_TRAIN {
        let message = tree_text(
            "entrenamiento",
            "mae_limite",
            "El criterio absolute_error es demasiado lento con {n} muestras de entrenamiento (máximo {max}). Usa squared_error.",
        )
        .replace("{n}", &n_train.to_string())
        .replace("{max}", &MAE_MAX_TRAIN.to_string());
        return Err(CommandError::invalid_config(message));
    }
    Ok(())
}

/// `[entrenamiento] error_nan`: the dataset has missing (or infinite) values in
/// `columns`. sklearn 1.7 trains trees with NaN; the app rejects them in every lesson.
pub fn missing_values_error(columns: &[String]) -> CommandError {
    let message = tree_text(
        "entrenamiento",
        "error_nan",
        "El dataset tiene valores faltantes en: {columnas}. Elimina o rellena esas filas antes de entrenar.",
    )
    .replace("{columnas}", &columns.join(", "));
    CommandError::MissingValues {
        columns: columns.to_vec(),
        message,
    }
}

/// D46: label of a categorical value of the decision path ("0 (Mujer)"), from the select
/// of the prediction form of `feature`. `None` for sliders and checkboxes.
pub fn path_display(
    form: &crate::ipc::PredictionForm,
    feature: usize,
    value: f64,
) -> Option<String> {
    let control = form.controls.iter().find(|c| c.index as usize == feature)?;
    let crate::ipc::FormControlKind::Select { options, .. } = &control.control else {
        return None;
    };
    // `value` is the float32 of the input (or the f64 code itself): compare as float32
    // with a tolerance of one f32 ulp around the code.
    let value = value as f32;
    let option = options.iter().find(|o| {
        let code = o.code as f32;
        (code - value).abs() <= f32::EPSILON * code.abs().max(1.0)
    })?;
    let template = tree_text(
        "predicciones",
        "camino_valor_categorico",
        "{codigo:g} ({etiqueta})",
    );
    Some(
        template
            .replace("{codigo:g}", &python_g(option.code))
            .replace("{etiqueta}", &option.label),
    )
}

/// `format(x, "g")` of Python for the codes of the form (integers in practice).
pub fn python_g(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
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
            notices: with_mismatch(
                summary.notices,
                self.config,
                summary.model_task,
                summary.dataset_task,
            ),
            tree: summary.tree,
        }
    }
}

/// Adds the warning of a model type that does not match the dataset (as in Python, the
/// model is trained anyway: Lineal on Iris, D25).
fn with_mismatch(
    mut notices: Vec<TrainNotice>,
    config: ModelConfig,
    model_task: TaskType,
    dataset_task: TaskType,
) -> Vec<TrainNotice> {
    if model_task != dataset_task
        && !notices
            .iter()
            .any(|n| matches!(n, TrainNotice::ModelTaskMismatch { .. }))
    {
        let message = if config.is_tree() {
            // `aviso_tipo` of 🏋️ Entrenamiento, with the task of the dataset.
            let task = match dataset_task {
                TaskType::Classification => "Clasificación",
                TaskType::Regression => "Regresión",
            };
            tree_text(
                "entrenamiento",
                "aviso_tipo",
                "Este dataset parece ser más adecuado para {tarea}. La selección actual podría no ofrecer resultados óptimos.",
            )
            .replace("{tarea}", task)
        } else {
            match dataset_task {
                TaskType::Classification => "Este dataset parece ser más adecuado para Regresión Logística. La selección actual podría no ofrecer resultados óptimos.",
                TaskType::Regression => "Este dataset parece ser más adecuado para Regresión Lineal. La selección actual podría no ofrecer resultados óptimos.",
            }
            .to_owned()
        };
        notices.push(TrainNotice::ModelTaskMismatch { message });
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

    /// The highest `seq` accepted by any lesson (0 if none): `get_selected_dataset`
    /// reports it so that a reloaded page never reuses a `seq` (clock gone backwards).
    pub fn max_seq(&self) -> u64 {
        self.last_seq.values().copied().max().unwrap_or(0)
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

    /// See [`LessonModels::max_seq`].
    pub fn last_seq(&self) -> u64 {
        self.lock().max_seq()
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
        let dataset_task = dataset.summary().task;
        if matches!(req.config, ModelConfig::Logistic { .. })
            && dataset_task == TaskType::Regression
        {
            return Err(CommandError::invalid_target(LOGISTIC_NEEDS_CLASSES));
        }
        // D37: a classification tree needs classes (Python trained 192 "classes" on
        // Diabetes and failed in English on California).
        if matches!(req.config, ModelConfig::Tree { task, .. } if task.task() == TaskType::Classification)
            && dataset_task == TaskType::Regression
        {
            return Err(CommandError::invalid_target(tree_needs_classes_message()));
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
    match req.config {
        ModelConfig::Logistic { max_iter }
            if !(MIN_MAX_ITER..=MAX_MAX_ITER).contains(&max_iter) =>
        {
            return Err(CommandError::invalid_selection(
                "El máximo de iteraciones debe estar entre 100 y 2000.",
            ));
        }
        ModelConfig::Tree {
            task,
            criterion,
            max_depth,
            min_samples_split,
        } => {
            if !(1..=10).contains(&max_depth) {
                return Err(CommandError::invalid_config(
                    "La profundidad máxima debe estar entre 1 y 10.",
                ));
            }
            if !(2..=10).contains(&min_samples_split) {
                return Err(CommandError::invalid_config(
                    "Las muestras mínimas para división deben estar entre 2 y 10.",
                ));
            }
            if !criterion.fits(task) {
                return Err(CommandError::invalid_config(format!(
                    "El criterio «{}» no sirve para un árbol de {}.",
                    criterion.sklearn_name(),
                    match task {
                        crate::ipc::TreeTask::Classification => "clasificación",
                        crate::ipc::TreeTask::Regression => "regresión",
                    }
                )));
            }
        }
        _ => {}
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

/// Default file name and save-dialog filter of each export (§5.4). Trees read them from
/// `[exportar]` of `content/arboles/mensajes.toml`.
fn export_target(config: ModelConfig, format: ExportFormat) -> (String, String, &'static str) {
    const ONNX: &str = "Modelo ONNX (*.onnx)";
    const ZIP: &str = "Paquete Python (*.zip)";
    let tree = |name: &str, default_name: &str, filter: &str, default_filter: &str| {
        (
            tree_text("exportar", name, default_name),
            tree_text("exportar", filter, default_filter),
        )
    };
    let ((name, filter), extension) = match (format, config) {
        (ExportFormat::Onnx, ModelConfig::Linear) => {
            (("regresion_lineal.onnx".into(), ONNX.into()), "onnx")
        }
        (ExportFormat::Onnx, ModelConfig::Logistic { .. }) => {
            (("regresion_logistica.onnx".into(), ONNX.into()), "onnx")
        }
        (ExportFormat::Onnx, ModelConfig::Tree { .. }) => (
            tree("archivo_onnx", "arbol_decision.onnx", "filtro_onnx", ONNX),
            "onnx",
        ),
        (ExportFormat::PythonPackage, ModelConfig::Tree { .. }) => (
            tree("archivo_paquete", "modelo_arbol.zip", "filtro_zip", ZIP),
            "zip",
        ),
        (ExportFormat::PythonPackage, _) => (("modelo_regresion.zip".into(), ZIP.into()), "zip"),
        (ExportFormat::TreeRules, _) => (
            tree(
                "archivo_reglas",
                "decision_tree_rules.txt",
                "filtro_txt",
                "Texto (*.txt)",
            ),
            "txt",
        ),
        (ExportFormat::TreePython, _) => (
            tree(
                "archivo_codigo",
                "arbol_decision.py",
                "filtro_py",
                "Código Python (*.py)",
            ),
            "py",
        ),
    };
    (name, filter, extension)
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
        if template.needs_tree() && !m.config.is_tree() {
            return Err(CommandError::not_tree());
        }
        let pair = match extra {
            Some(extra) => Some(validate_pair(m, extra.feature_x, extra.feature_y)?),
            None => None,
        };
        m.model.code(template, pair)
    })
    .await
}

/// C13: the boxes of `plot_tree` for the SVG diagram (D41).
#[tauri::command]
pub async fn tree_diagram(
    models: State<'_, ModelState>,
    model_id: u32,
) -> Result<TreeDiagram, CommandError> {
    with_model("tree_diagram", &models, model_id, |m| {
        m.model.tree_diagram()
    })
    .await
}

/// C14: `export_text` of the tree (📝 Texto or the rules of 💾 Exportar).
#[tauri::command]
pub async fn tree_rules(
    models: State<'_, ModelState>,
    req: RulesRequest,
) -> Result<String, CommandError> {
    with_model("tree_rules", &models, req.model_id, move |m| {
        m.model.tree_rules(req.kind, req.class_names)
    })
    .await
}

/// C10: native importance or permutation importance (cancellable, with progress).
#[tauri::command]
pub async fn feature_importance(
    app: AppHandle,
    data: State<'_, DataState>,
    models: State<'_, ModelState>,
    model_id: u32,
    method: ImportanceMethod,
    task_id: u32,
) -> Result<FeatureImportanceView, CommandError> {
    let guard = data.tasks.start(task_id);
    with_model("feature_importance", &models, model_id, move |m| {
        let throttle = ProgressThrottle::default();
        let progress = |message: &str, percent: Option<f32>| {
            if throttle.admit(message, percent, std::time::Instant::now()) {
                emit_progress(&app, task_id, message, percent);
            }
        };
        m.model
            .feature_importance(method, &progress, &guard.token())
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
        let Some(path) = ask_save_path(&app, &name, &filter, extension)? else {
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

    const TREE: ModelConfig = ModelConfig::Tree {
        task: crate::ipc::TreeTask::Classification,
        criterion: crate::ipc::TreeCriterion::Gini,
        max_depth: 3,
        min_samples_split: 2,
    };

    fn tree_request(dataset: DatasetRef, config: ModelConfig, seq: u64) -> TrainRequest {
        TrainRequest {
            lesson: LessonId::Trees,
            ..request(dataset, config, seq)
        }
    }

    /// D37: a classification tree on a continuous target is an error in Spanish (from
    /// `content/arboles/mensajes.toml`); a regression tree on Iris trains with the warning.
    #[test]
    fn classification_trees_need_classes() {
        let (data, models) = setup();
        let diabetes = DatasetRef::Builtin {
            id: DatasetId::Diabetes,
        };
        let error = models
            .train(
                &data,
                &tree_request(diabetes, TREE, 1),
                &CancelToken::default(),
                &quiet,
            )
            .unwrap_err();
        assert_eq!(
            error,
            CommandError::invalid_target(tree_needs_classes_message())
        );
        assert!(tree_needs_classes_message().contains("📈 Regresión"));
        let regression = ModelConfig::Tree {
            task: crate::ipc::TreeTask::Regression,
            criterion: crate::ipc::TreeCriterion::SquaredError,
            max_depth: 3,
            min_samples_split: 2,
        };
        let info = models
            .train(
                &data,
                &tree_request(iris(), regression, 2),
                &CancelToken::default(),
                &quiet,
            )
            .unwrap();
        assert_eq!(info.model_task, TaskType::Regression);
        assert!(info.tree.is_some());
        assert_eq!(
            info.notices,
            vec![TrainNotice::ModelTaskMismatch {
                message: "Este dataset parece ser más adecuado para Clasificación. La selección actual podría no ofrecer resultados óptimos.".into()
            }]
        );
        let iris_tree = models
            .train(
                &data,
                &tree_request(iris(), TREE, 3),
                &CancelToken::default(),
                &quiet,
            )
            .unwrap();
        assert!(iris_tree.notices.is_empty());
        assert_eq!(iris_tree.tree.map(|t| t.node_count), Some(9));
    }

    #[test]
    fn tree_configs_are_validated() {
        let bad = |config| validate_train(&tree_request(iris(), config, 1));
        assert!(bad(TREE).is_ok());
        for (depth, mss) in [(0, 2), (11, 2), (3, 1), (3, 11)] {
            let config = ModelConfig::Tree {
                task: crate::ipc::TreeTask::Classification,
                criterion: crate::ipc::TreeCriterion::Gini,
                max_depth: depth,
                min_samples_split: mss,
            };
            assert!(
                matches!(bad(config), Err(CommandError::InvalidConfig { .. })),
                "{depth} {mss}"
            );
        }
        let mixed = ModelConfig::Tree {
            task: crate::ipc::TreeTask::Classification,
            criterion: crate::ipc::TreeCriterion::SquaredError,
            max_depth: 3,
            min_samples_split: 2,
        };
        assert!(matches!(
            bad(mixed),
            Err(CommandError::InvalidConfig { .. })
        ));
        assert_eq!(
            export_target(TREE, ExportFormat::TreeRules).0,
            "decision_tree_rules.txt"
        );
        assert_eq!(
            export_target(TREE, ExportFormat::Onnx).0,
            "arbol_decision.onnx"
        );
        assert_eq!(
            export_target(TREE, ExportFormat::PythonPackage).0,
            "modelo_arbol.zip"
        );
        assert_eq!(
            export_target(TREE, ExportFormat::TreePython).0,
            "arbol_decision.py"
        );
    }

    /// Pending from F2: after a reload with the clock gone backwards, the UI continues
    /// above every accepted `seq` (selection AND trainings), so no lesson is left with
    /// trainings rejected as old.
    #[test]
    fn selected_dataset_reports_the_highest_seq_of_selection_and_trainings() {
        let (data, models) = setup();
        data.select(iris(), 10).unwrap();
        assert_eq!(
            crate::data::selection_with_models(&data, &models).last_seq,
            10
        );
        models
            .train(
                &data,
                &tree_request(iris(), TREE, 500),
                &CancelToken::default(),
                &quiet,
            )
            .unwrap();
        models
            .train(
                &data,
                &request(iris(), ModelConfig::Logistic { max_iter: 1000 }, 300),
                &CancelToken::default(),
                &quiet,
            )
            .unwrap();
        let selection = crate::data::selection_with_models(&data, &models);
        assert_eq!(selection.last_seq, 500);
        assert_eq!(selection.dataset, iris());
        // A training continued above it is accepted in every lesson.
        let next = selection.last_seq + 1;
        assert!(models
            .train(
                &data,
                &tree_request(iris(), TREE, next),
                &CancelToken::default(),
                &quiet
            )
            .is_ok());
        assert!(models
            .train(
                &data,
                &request(iris(), ModelConfig::Logistic { max_iter: 1000 }, next),
                &CancelToken::default(),
                &quiet
            )
            .is_ok());
        // A failed training also raises it (its `seq` was accepted).
        let diabetes = DatasetRef::Builtin {
            id: DatasetId::Diabetes,
        };
        let _ = models.train(
            &data,
            &tree_request(diabetes, TREE, 900),
            &CancelToken::default(),
            &quiet,
        );
        assert_eq!(
            crate::data::selection_with_models(&data, &models).last_seq,
            900
        );
    }

    #[test]
    fn categorical_path_values_show_the_option() {
        use crate::ipc::{FormControl, FormControlKind, PredictionForm, SelectOption};
        let form = PredictionForm {
            controls: vec![FormControl {
                index: 1,
                column: "sex".into(),
                label: "Sexo".into(),
                description: None,
                control: FormControlKind::Select {
                    options: vec![
                        SelectOption {
                            label: "Mujer".into(),
                            code: 0.0,
                        },
                        SelectOption {
                            label: "Hombre".into(),
                            code: 1.0,
                        },
                    ],
                    default_index: 0,
                },
            }],
        };
        assert_eq!(path_display(&form, 1, 0.0).as_deref(), Some("0 (Mujer)"));
        assert_eq!(path_display(&form, 1, 0.5), None);
        assert_eq!(path_display(&form, 0, 0.0), None);
        // The f32 of a code and a value a rounding away still match; a different one not.
        assert_eq!(
            path_display(&form, 1, f64::from(1.0_f32) + 1e-12).as_deref(),
            Some("1 (Hombre)")
        );
        assert_eq!(path_display(&form, 1, 1.0 + 1e-3), None);
        assert_eq!(python_g(3.0), "3");
        assert_eq!(python_g(0.25), "0.25");
    }

    #[test]
    fn absolute_error_is_limited_to_20000_training_rows() {
        let tree = |criterion| ModelConfig::Tree {
            task: crate::ipc::TreeTask::Regression,
            criterion,
            max_depth: 3,
            min_samples_split: 2,
        };
        let mae = tree(crate::ipc::TreeCriterion::AbsoluteError);
        assert!(check_mae_limit(mae, MAE_MAX_TRAIN).is_ok());
        let Err(CommandError::InvalidConfig { message }) = check_mae_limit(mae, 20_001) else {
            panic!("absolute_error with 20 001 rows must be rejected");
        };
        assert!(message.contains("20001"), "{message}");
        assert!(message.contains("20000"), "{message}");
        assert!(message.contains("squared_error"), "{message}");
        for config in [
            tree(crate::ipc::TreeCriterion::SquaredError),
            tree(crate::ipc::TreeCriterion::FriedmanMse),
            ModelConfig::Linear,
        ] {
            assert!(check_mae_limit(config, 1_000_000).is_ok());
        }
    }

    #[test]
    fn missing_values_list_the_columns() {
        let error = missing_values_error(&["edad".into(), "tarifa".into()]);
        let CommandError::MissingValues { columns, message } = &error else {
            panic!("{error:?}");
        };
        assert_eq!(columns, &["edad", "tarifa"]);
        assert!(message.contains("edad, tarifa"), "{message}");
        assert!(!message.contains('{'), "{message}");
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
            (
                "modelo_regresion.zip".to_owned(),
                "Paquete Python (*.zip)".to_owned(),
                "zip"
            )
        );
    }
}
