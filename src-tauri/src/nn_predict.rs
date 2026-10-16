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

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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

/// The CSV of a batch and the model that predicted it.
type StoredBatch = Option<(u32, Vec<u8>)>;

/// The CSV with the predictions of the last batch (one per app: a new batch replaces it).
#[derive(Default, Clone)]
pub struct NnBatches {
    last: Arc<Mutex<StoredBatch>>,
}

impl NnBatches {
    fn store(&self, model_id: u32, csv: Option<Vec<u8>>) {
        let old = {
            let mut last = self.last.lock().unwrap_or_else(|p| p.into_inner());
            std::mem::replace(&mut *last, csv.map(|c| (model_id, c)))
        };
        // Released without the lock (it may be a few MB).
        drop(old);
    }

    fn get(&self, model_id: u32) -> Option<Vec<u8>> {
        let last = self.last.lock().unwrap_or_else(|p| p.into_inner());
        last.as_ref()
            .filter(|(id, _)| *id == model_id)
            .map(|(_, csv)| csv.clone())
    }
}

/// Where the automation modes write the files they would ask for (one folder per run).
pub fn automation_dir() -> Result<PathBuf, CommandError> {
    let dir = std::env::temp_dir().join(format!("mltutor-automatizacion-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(CommandError::file_save)?;
    Ok(dir)
}

/// `--smoke-test` and `--screenshot` never open a dialog (nobody would answer it).
pub fn is_automated(mode: &Mode) -> bool {
    !matches!(mode, Mode::Normal)
}

/// Writes `bytes` as `name` in [`automation_dir`] and returns the name shown to the user.
pub fn save_automated(name: &str, bytes: &[u8]) -> Result<String, CommandError> {
    let path = automation_dir()?.join(name);
    std::fs::write(&path, bytes).map_err(CommandError::file_save)?;
    log::info!(
        "automatización: guardado {} ({} bytes)",
        path.display(),
        bytes.len()
    );
    Ok(crate::files::display_name(&path))
}

/// A field of a CSV written by the app (RFC 4180 quoting).
pub fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) || text.trim() != text {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_owned()
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
        let path = automation_dir()?.join("lote_prueba.csv");
        std::fs::write(&path, sample_csv(&template)).map_err(CommandError::csv_io)?;
        return Ok(Some(path));
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
    let trained_at = clean_date(&trained_at);
    with_model("nn_metadata", &models, model_id, move |m| {
        m.model.nn_metadata(&trained_at)
    })
    .await
}

/// The local date of the training sent by the UI ("2026-10-15 17:49"): at most 40
/// characters, no control characters (it goes into a JSON string).
pub fn clean_date(text: &str) -> String {
    text.chars().filter(|c| !c.is_control()).take(40).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::TablePreview;

    #[test]
    fn dates_from_the_ui_are_cleaned() {
        assert_eq!(clean_date("2026-10-15 17:49"), "2026-10-15 17:49");
        assert_eq!(clean_date("a\nb"), "ab");
        assert_eq!(clean_date(&"x".repeat(100)).len(), 40);
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
    }
}
