//! 🧠 Redes Neuronales, F5d: the commands of 🔮 Predicciones that are new in Rust (C29,
//! D81: 🎲 Exploración and 📊 Predicción por lotes) and of 📋 Metadatos (D75).
//!
//! The CSV of the batch is chosen with the native dialog here (as `pick_and_load_csv`):
//! the path never comes from JS. The CSV with the results stays in [`NnBatches`] and
//! "📥 Descargar Resultados" saves it with the same dialog as `save_file`, so a batch of
//! thousands of rows does not cross the IPC twice.
//!
//! In the automation modes (`--smoke-test`, `--screenshot`) no dialog is opened: the batch
//! reads the CSV of `--scenario <nombre>:<ruta.csv>` (or a CSV written from the dataset
//! itself) and every save goes to [`automation_dir`].

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::cli::Mode;
use crate::core_facade::ModelView;
use crate::data::{blocking, DataState};
use crate::files::SaveKind;
use crate::ipc::{
    BatchPredictRequest, BatchPredictView, BatchTemplate, Cell, CommandError, ExploreRequest,
    ExploreView,
};
use crate::models::{with_model, ModelState};
use crate::state::AppState;

/// Rows of "📋 Formato esperado del archivo CSV" (`df[feature_cols].head(3)`).
pub const TEMPLATE_ROWS: usize = 3;
/// Rows of the CSV the automation modes write when no CSV is given.
pub const AUTOMATION_ROWS: usize = 25;
/// Suggested name of "📥 Descargar Resultados" (Python's `predicciones_neural_network.csv`).
pub const RESULTS_FILE: &str = "predicciones_neural_network.csv";
/// The UTF-8 byte order mark: Excel needs it to read the accents of a CSV.
pub const UTF8_BOM: &[u8] = b"\xEF\xBB\xBF";

/// The CSV of a batch and the model that predicted it (shared, never copied).
type StoredBatch = Option<(u32, Arc<[u8]>)>;

/// The CSV with the predictions of the last batch (one per app: a new batch replaces it,
/// and a new training releases it).
#[derive(Default, Clone)]
pub struct NnBatches {
    last: Arc<Mutex<StoredBatch>>,
}

impl NnBatches {
    fn replace(&self, next: StoredBatch) {
        let old = {
            let mut last = self.last.lock().unwrap_or_else(|p| p.into_inner());
            std::mem::replace(&mut *last, next)
        };
        // Released without the lock (it may be a few MB).
        drop(old);
    }

    fn store(&self, model_id: u32, csv: Option<Vec<u8>>) {
        self.replace(csv.map(|c| (model_id, Arc::from(c))));
    }

    /// Frees the CSV of the last batch (a new model is being trained).
    pub fn clear(&self) {
        self.replace(None);
    }

    fn get(&self, model_id: u32) -> Option<Arc<[u8]>> {
        let last = self.last.lock().unwrap_or_else(|p| p.into_inner());
        last.as_ref()
            .filter(|(id, _)| *id == model_id)
            .map(|(_, csv)| Arc::clone(csv))
    }
}

/// Where the automation modes write the files they would ask for: one private folder
/// per run (0700 and a random name, so another user cannot plant or read its files).
/// It is kept after the run (its path is in the log).
pub fn automation_dir() -> Result<PathBuf, CommandError> {
    static DIR: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    let dir = DIR.get_or_init(|| {
        let mut builder = tempfile::Builder::new();
        builder.prefix("mltutor-automatizacion-");
        // tempfile creates directories with 0777 & umask: private on purpose.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            builder.permissions(std::fs::Permissions::from_mode(0o700));
        }
        builder
            .tempdir()
            .map(tempfile::TempDir::keep)
            .map_err(|error| error.to_string())
    });
    match dir {
        Ok(dir) => Ok(dir.clone()),
        Err(error) => Err(CommandError::file_save(std::io::Error::other(
            error.clone(),
        ))),
    }
}

/// `name` in `dir`, or `stem-2.ext`, `stem-3.ext`… if it exists: a new file opened with
/// `create_new` (never follows or overwrites an existing path).
fn create_unique(dir: &Path, name: &str) -> std::io::Result<(std::fs::File, PathBuf)> {
    let (stem, ext) = match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => (stem, format!(".{ext}")),
        _ => (name, String::new()),
    };
    for i in 1..=1000u32 {
        let candidate = if i == 1 {
            name.to_owned()
        } else {
            format!("{stem}-{i}{ext}")
        };
        let path = dir.join(candidate);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok((file, path)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        format!("demasiados archivos llamados {name}"),
    ))
}

/// Writes `bytes` as a new file named `name` in [`automation_dir`] and returns its path.
fn write_automated(name: &str, bytes: &[u8]) -> Result<PathBuf, CommandError> {
    let dir = automation_dir()?;
    let (mut file, path) = create_unique(&dir, name).map_err(CommandError::file_save)?;
    file.write_all(bytes).map_err(CommandError::file_save)?;
    Ok(path)
}

/// `--smoke-test` and `--screenshot` never open a dialog (nobody would answer it).
pub fn is_automated(mode: &Mode) -> bool {
    !matches!(mode, Mode::Normal)
}

/// Writes `bytes` as `name` in [`automation_dir`] and returns the name shown to the user.
pub fn save_automated(name: &str, bytes: &[u8]) -> Result<String, CommandError> {
    let path = write_automated(name, bytes)?;
    log::info!(
        "automatización: guardado {} ({} bytes)",
        path.display(),
        bytes.len()
    );
    Ok(crate::files::display_name(&path))
}

/// Whether a text field would run as a formula in a spreadsheet (CSV injection): it
/// starts with `=`, `+`, `@`, a tab or a CR, or with `-` followed by something that is
/// not a digit (`-3.5` is a number; `-cmd` is not).
fn looks_like_formula(text: &str) -> bool {
    let mut chars = text.chars();
    match chars.next() {
        Some('=' | '+' | '@' | '\t' | '\r') => true,
        Some('-') => chars.next().is_some_and(|c| !c.is_ascii_digit()),
        _ => false,
    }
}

/// A text field of a CSV written by the app: neutralised if it looks like a formula (a
/// leading `'`, as OWASP recommends) and quoted as RFC 4180 when needed. Numbers written
/// by [`number`] do not go through here.
pub fn csv_field(text: &str) -> String {
    let text = if looks_like_formula(text) {
        std::borrow::Cow::Owned(format!("'{text}"))
    } else {
        std::borrow::Cow::Borrowed(text)
    };
    if text.contains([',', '"', '\n', '\r', '\t']) || text.trim() != text {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.into_owned()
    }
}

fn cell_text(cell: &Cell) -> String {
    match cell {
        Cell::Number { v } => number(*v),
        Cell::Text { v } => v.clone(),
        Cell::Bool { v } => if *v { "1" } else { "0" }.to_owned(),
        Cell::Missing => String::new(),
    }
}

/// A number as pandas' `to_csv` writes it (`repr` of the float; integers without `.0`).
pub fn number(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        shortest(v)
    }
}

fn shortest(v: f64) -> String {
    let text = format!("{v}");
    if text == "NaN" {
        String::new()
    } else {
        text
    }
}

/// The CSV the automation modes predict when the scenario does not name one: the first
/// rows of the dataset with the feature names and one extra column (it must be ignored,
/// with the "⚠️ Características adicionales" warning).
pub fn sample_csv(template: &BatchTemplate) -> Vec<u8> {
    let mut out = String::new();
    let header: Vec<String> = template.columns.iter().map(|c| csv_field(c)).collect();
    out.push_str(&header.join(","));
    out.push_str(",comentario\n");
    for (i, row) in template.sample.rows.iter().enumerate() {
        let fields: Vec<String> = row.iter().map(|c| csv_field(&cell_text(c))).collect();
        out.push_str(&fields.join(","));
        out.push_str(&format!(",fila {}\n", i + 1));
    }
    out.into_bytes()
}

/// C29 🎲 Exploración: the response curve of one feature of one sample.
#[tauri::command]
pub async fn nn_explore(
    models: State<'_, ModelState>,
    req: ExploreRequest,
) -> Result<ExploreView, CommandError> {
    with_model("nn_explore", &models, req.model_id, move |m| {
        let n_features = m.model.summary().feature_columns.len();
        if req.feature as usize >= n_features {
            return Err(CommandError::invalid_selection(
                "La característica elegida no existe en este modelo.",
            ));
        }
        m.model.nn_explore(req.row as usize, req.feature as usize)
    })
    .await
}

/// "📋 Formato esperado del archivo CSV" of 📊 Predicción por lotes.
#[tauri::command]
pub async fn nn_batch_template(
    models: State<'_, ModelState>,
    model_id: u32,
) -> Result<BatchTemplate, CommandError> {
    with_model("nn_batch_template", &models, model_id, |m| {
        m.model.nn_batch_template(TEMPLATE_ROWS)
    })
    .await
}

/// Where the CSV of the batch comes from.
fn batch_source(
    app: &AppHandle,
    mode: &Mode,
    model: &dyn ModelView,
) -> Result<Option<PathBuf>, CommandError> {
    if is_automated(mode) {
        if let Some(path) = crate::data::scenario_csv_path(mode) {
            return Ok(Some(PathBuf::from(path)));
        }
        let template = model.nn_batch_template(AUTOMATION_ROWS)?;
        return write_automated("lote_prueba.csv", &sample_csv(&template)).map(Some);
    }
    let mut dialog = app.dialog().file().add_filter(
        crate::nn::nn_text("exportar_rust", "filtro_csv", "Archivos (*.csv)"),
        &["csv"],
    );
    if let Some(window) = app.get_webview_window("main") {
        dialog = dialog.set_parent(&window);
    }
    let Some(picked) = dialog.blocking_pick_file() else {
        return Ok(None);
    };
    picked.into_path().map(Some).map_err(CommandError::csv_io)
}

/// C29 📊 Predicción por lotes: opens the dialog (in Rust), reads the CSV, checks its
/// columns and predicts every row. `None` if the user closes the dialog.
#[tauri::command]
pub async fn nn_predict_batch(
    app: AppHandle,
    app_state: State<'_, AppState>,
    data: State<'_, DataState>,
    models: State<'_, ModelState>,
    batches: State<'_, NnBatches>,
    req: BatchPredictRequest,
) -> Result<Option<BatchPredictView>, CommandError> {
    let mode = app_state.mode.clone();
    let guard = data.tasks.start(req.task_id);
    let batches = batches.inner().clone();
    let model_id = req.model_id;
    with_model("nn_predict_batch", &models, model_id, move |m| {
        let Some(path) = batch_source(&app, &mode, m.model.as_ref())? else {
            return Ok(None);
        };
        let outcome = m.model.nn_predict_batch(&path, &guard.token())?;
        batches.store(model_id, outcome.csv);
        Ok(Some(outcome.view))
    })
    .await
}

/// "📥 Descargar Resultados": the CSV of the last batch of `model_id`, with the save
/// dialog of `save_file` (or in [`automation_dir`]). `None` if the user cancels.
#[tauri::command]
pub async fn nn_save_batch(
    app: AppHandle,
    app_state: State<'_, AppState>,
    batches: State<'_, NnBatches>,
    model_id: u32,
) -> Result<Option<String>, CommandError> {
    let Some(csv) = batches.get(model_id) else {
        return Err(CommandError::invalid_selection(
            "No hay predicciones por lotes que guardar: vuelve a elegir el CSV.",
        ));
    };
    let automated = is_automated(&app_state.mode);
    let name = crate::nn::nn_text("exportar_rust", "archivo_lotes", RESULTS_FILE);
    blocking(move || {
        if automated {
            return save_automated(&name, &csv).map(Some);
        }
        let Some(path) = crate::files::ask_path(&app, &name, SaveKind::Csv)? else {
            return Ok(None);
        };
        save_to(&path, &csv).map(Some)
    })
    .await
}

fn save_to(path: &Path, bytes: &[u8]) -> Result<String, CommandError> {
    std::fs::write(path, bytes).map_err(CommandError::file_save)?;
    log::info!(
        "archivo guardado: {} ({} bytes)",
        path.display(),
        bytes.len()
    );
    Ok(crate::files::display_name(path))
}

/// 📋 Metadatos (D75): the JSON shown in 💾 Exportar.
#[tauri::command]
pub async fn nn_metadata(
    models: State<'_, ModelState>,
    model_id: u32,
    trained_at: String,
) -> Result<String, CommandError> {
    with_model("nn_metadata", &models, model_id, move |m| {
        m.model.nn_metadata(&trained_at)
    })
    .await
}

/// The local date of the training sent by the UI, if it has the shape `AAAA-MM-DD HH:MM`
/// (it goes into a JSON string and a file): anything else is rejected.
pub fn checked_date(text: &str) -> Option<String> {
    let b = text.as_bytes();
    let digits = |r: std::ops::Range<usize>| -> Option<u32> {
        let part = b.get(r)?;
        part.iter()
            .all(u8::is_ascii_digit)
            .then(|| std::str::from_utf8(part).ok()?.parse().ok())
            .flatten()
    };
    let shape = b.len() == 16 && b[4] == b'-' && b[7] == b'-' && b[10] == b' ' && b[13] == b':';
    let (month, day, hour, minute) = (
        digits(5..7)?,
        digits(8..10)?,
        digits(11..13)?,
        digits(14..16)?,
    );
    digits(0..4)?;
    let valid =
        shape && (1..=12).contains(&month) && (1..=31).contains(&day) && hour < 24 && minute < 60;
    valid.then(|| text.to_owned())
}

/// `ms` since the Unix epoch as `AAAA-MM-DD HH:MM UTC` (when the UI sends no valid
/// local date).
pub fn utc_date(ms: u64) -> String {
    let secs = ms / 1000;
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02} UTC",
        rem / 3600,
        rem % 3600 / 60
    )
}

/// The date of 📋 Metadatos: the one the UI sent if valid, else the UTC date of the
/// training (`NnSummary.trained_at_ms`).
pub fn trained_at(text: &str, trained_at_ms: u64) -> String {
    checked_date(text).unwrap_or_else(|| {
        log::warn!("fecha de entrenamiento no válida desde la interfaz: {text:?}");
        utc_date(trained_at_ms)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::TablePreview;

    #[test]
    fn dates_from_the_ui_are_checked() {
        assert_eq!(
            checked_date("2026-10-15 17:49").as_deref(),
            Some("2026-10-15 17:49")
        );
        for bad in [
            "",
            "a\nb",
            "2026-13-15 17:49",
            "2026-10-15 24:00",
            "2026-10-15T17:49",
            "2026-10-15 17:49\"x",
        ] {
            assert_eq!(checked_date(bad), None, "{bad:?}");
        }
        assert_eq!(utc_date(0), "1970-01-01 00:00 UTC");
        assert_eq!(utc_date(1_792_152_540_000), "2026-10-16 12:09 UTC");
        assert_eq!(trained_at("{\"x\": 1}", 0), "1970-01-01 00:00 UTC");
    }

    #[test]
    fn csv_fields_cannot_be_formulas() {
        assert_eq!(csv_field("=1+1"), "'=1+1");
        assert_eq!(csv_field("+34 600"), "'+34 600");
        assert_eq!(csv_field("@SUM(A1)"), "'@SUM(A1)");
        assert_eq!(csv_field("-cmd"), "'-cmd");
        assert_eq!(csv_field("\tx"), "\"'\tx\"");
        assert_eq!(csv_field("-3.5"), "-3.5");
        assert_eq!(csv_field("-"), "-");
        assert_eq!(csv_field("setosa"), "setosa");
    }

    #[test]
    fn automation_files_are_new_and_private() {
        let dir = automation_dir().unwrap();
        assert_eq!(dir, automation_dir().unwrap(), "one folder per run");
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("mltutor-automatizacion-"), "{name}");
        assert_ne!(
            name,
            format!("mltutor-automatizacion-{}", std::process::id())
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o700);
        }
        let a = write_automated("prueba_unica.csv", b"1").unwrap();
        let b = write_automated("prueba_unica.csv", b"2").unwrap();
        assert_ne!(a, b);
        assert!(b.ends_with("prueba_unica-2.csv"), "{b:?}");
        assert_eq!(std::fs::read(&a).unwrap(), b"1");
    }

    #[test]
    fn csv_fields_are_quoted_when_needed() {
        assert_eq!(csv_field("sepal length (cm)"), "sepal length (cm)");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("di \"x\""), "\"di \"\"x\"\"\"");
        assert_eq!(number(3.0), "3");
        assert_eq!(number(0.1), "0.1");
        assert_eq!(number(f64::NAN), "");
    }

    #[test]
    fn the_automation_csv_has_an_extra_column() {
        let template = BatchTemplate {
            columns: vec!["a".into(), "b, c".into()],
            sample: TablePreview {
                columns: vec!["a".into(), "b, c".into()],
                index: vec!["0".into()],
                rows: vec![vec![Cell::Number { v: 1.5 }, Cell::Number { v: 2.0 }]],
                total_rows: 1,
            },
            train_min: vec![1.0, 2.0],
            train_max: vec![1.0, 2.0],
        };
        let text = String::from_utf8(sample_csv(&template)).unwrap();
        assert_eq!(text, "a,\"b, c\",comentario\n1.5,2,fila 1\n");
    }

    #[test]
    fn batches_are_kept_for_their_model_only() {
        let batches = NnBatches::default();
        batches.store(3, Some(b"x".to_vec()));
        assert_eq!(batches.get(3).as_deref(), Some(&b"x"[..]));
        assert!(batches.get(4).is_none());
        batches.store(4, None);
        assert!(batches.get(3).is_none());
        batches.store(5, Some(b"y".to_vec()));
        batches.clear();
        assert!(batches.get(5).is_none(), "a new training releases it");
    }
}
