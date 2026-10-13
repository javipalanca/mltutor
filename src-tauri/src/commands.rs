//! IPC commands. Every command is `async` so it runs on the async runtime instead of
//! the main thread: a panic there would cross the native event loop and abort the app.

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};
use ts_rs::TS;

use crate::cli::Mode;
use crate::logging;
use crate::state::{AppState, EXIT_FAILURE, EXIT_OK, SMOKE_ROUTES};

/// How the frontend should behave after mounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[ts(export)]
pub enum LaunchMode {
    Normal,
    /// Visit every route in `routes` and report each one.
    SmokeTest {
        routes: Vec<String>,
    },
    /// Render the initial page and call `frontend_ready`; the backend takes the capture.
    Screenshot,
}

impl From<&Mode> for LaunchMode {
    fn from(mode: &Mode) -> Self {
        match mode {
            Mode::Normal => Self::Normal,
            Mode::SmokeTest => Self::SmokeTest {
                routes: SMOKE_ROUTES
                    .iter()
                    .map(|route| (*route).to_owned())
                    .collect(),
            },
            Mode::Screenshot(_) => Self::Screenshot,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
#[ts(export)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
}

impl From<LogLevel> for log::Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => Self::Error,
            LogLevel::Warn => Self::Warn,
            LogLevel::Info => Self::Info,
            LogLevel::Debug => Self::Debug,
        }
    }
}

#[tauri::command]
pub async fn launch_mode(state: State<'_, AppState>) -> Result<LaunchMode, String> {
    Ok(LaunchMode::from(&state.mode))
}

#[tauri::command]
pub async fn frontend_ready(
    app: AppHandle,
    state: State<'_, AppState>,
    route: String,
) -> Result<(), String> {
    if !state.mark_ready() {
        return Ok(());
    }
    let elapsed = state.started.elapsed();
    log::info!(
        "interfaz lista en {} ms (ruta inicial: {route})",
        elapsed.as_millis()
    );
    if matches!(state.mode, Mode::SmokeTest) {
        println!("Arranque: {} ms", elapsed.as_millis());
    }
    if let Mode::Screenshot(path) = &state.mode {
        crate::screenshot::capture_main_window(&app, path.clone());
    }
    Ok(())
}

fn require_smoke(state: &AppState) -> Result<(), String> {
    match state.mode {
        Mode::SmokeTest => Ok(()),
        _ => Err("Este comando solo está disponible con --smoke-test".to_owned()),
    }
}

#[tauri::command]
pub async fn smoke_route_rendered(
    app: AppHandle,
    state: State<'_, AppState>,
    route: String,
    title: String,
) -> Result<(), String> {
    require_smoke(&state)?;
    match state.record_smoke_route(&route, &title) {
        Ok(done) => {
            log::info!("smoke test: «{route}» renderizada ({title})");
            if done {
                state.finish(
                    &app,
                    EXIT_OK,
                    &format!(
                        "MLTutor: las {} pantallas se abren correctamente ({} ms).",
                        SMOKE_ROUTES.len(),
                        state.started.elapsed().as_millis()
                    ),
                );
            }
            Ok(())
        }
        Err(error) => {
            state.finish(&app, EXIT_FAILURE, &format!("Smoke test fallido: {error}"));
            Err(error)
        }
    }
}

#[tauri::command]
pub async fn smoke_failed(
    app: AppHandle,
    state: State<'_, AppState>,
    message: String,
) -> Result<(), String> {
    require_smoke(&state)?;
    state.finish(
        &app,
        EXIT_FAILURE,
        &format!("Smoke test fallido: {message}"),
    );
    Ok(())
}

/// Panics on a blocking worker and reports whether the app survived and the hook logged it.
#[tauri::command]
pub async fn smoke_panic_probe(state: State<'_, AppState>) -> Result<bool, String> {
    require_smoke(&state)?;
    let before = logging::panic_count();
    let joined = tauri::async_runtime::spawn_blocking(|| {
        panic!("panic intencionado de --smoke-test para comprobar el hook");
    })
    .await;
    let survived = joined.is_err() && logging::panic_count() > before;
    log::info!("smoke test: panic recuperado = {survived}");
    Ok(survived)
}

#[tauri::command]
pub async fn log_message(level: LogLevel, message: String) -> Result<(), String> {
    log::log!(target: "mltutor::ui", level.into(), "{message}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_mode_serializes_with_a_kind_tag() {
        let smoke = LaunchMode::from(&Mode::SmokeTest);
        let json = serde_json::to_value(&smoke).unwrap();
        assert_eq!(json["kind"], "smoke_test");
        assert_eq!(json["routes"].as_array().unwrap().len(), SMOKE_ROUTES.len());
        assert_eq!(
            serde_json::to_value(LaunchMode::from(&Mode::Normal)).unwrap(),
            serde_json::json!({ "kind": "normal" })
        );
    }

    #[test]
    fn log_levels_deserialize_from_lowercase() {
        let level: LogLevel = serde_json::from_str("\"warn\"").unwrap();
        assert_eq!(log::Level::from(level), log::Level::Warn);
    }
}
