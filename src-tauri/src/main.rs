//! MLTutor desktop app (Tauri 2).

// No console window on Windows release builds; logs go to ~/.mltutor/mltutor.log.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod cli;
mod commands;
mod core_facade;
mod data;
#[cfg(any(test, debug_assertions))]
mod fixture_facade;
mod ipc;
mod logging;
mod real_facade;
mod screenshot;
mod session;
mod state;

use std::process::ExitCode;
use std::time::Instant;

use tauri::Manager;

use crate::cli::{Command, Mode};
use crate::state::{AppState, AUTOMATION_TIMEOUT, EXIT_FAILURE, EXIT_TIMEOUT};

/// Exit code for invalid command-line usage (sysexits `EX_USAGE`).
const EXIT_USAGE: u8 = 64;

fn main() -> ExitCode {
    let started = Instant::now();
    let log_path = logging::init();
    logging::install_panic_hook();

    let (command, ignored) = match cli::parse(std::env::args().skip(1)) {
        Ok(parsed) => parsed,
        Err(error) => {
            log::error!("argumentos no válidos: {error}");
            eprintln!("mltutor: {error}\n\n{}", cli::USAGE);
            return ExitCode::from(EXIT_USAGE);
        }
    };
    if !ignored.is_empty() {
        log::debug!("argumentos ignorados: {ignored:?}");
    }
    let mode = match command {
        Command::PrintHelp => {
            println!("{}", cli::USAGE);
            return ExitCode::SUCCESS;
        }
        Command::PrintVersion => {
            println!("mltutor {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        Command::Run(mode) => mode,
    };

    log::info!(
        "MLTutor {} ({} {}) modo {mode:?}; registro en {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        log_path
            .as_deref()
            .map_or_else(|| "<stderr>".into(), |path| path.display().to_string())
    );

    let code = run(mode, started);
    log::info!("MLTutor termina con código {code}");
    log::logger().flush();
    ExitCode::from(u8::try_from(code).unwrap_or(EXIT_FAILURE as u8))
}

fn run(mode: Mode, started: Instant) -> i32 {
    let automated = !matches!(mode, Mode::Normal);
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(AppState::new(mode, started))
        .manage(data::DataState::new(facade()))
        .invoke_handler(tauri::generate_handler![
            commands::launch_mode,
            commands::frontend_ready,
            commands::screenshot_ready,
            commands::smoke_route_rendered,
            commands::smoke_check_passed,
            commands::smoke_failed,
            commands::smoke_panic_probe,
            commands::log_message,
            data::list_datasets,
            data::get_selected_dataset,
            data::select_dataset,
            data::dataset_summary,
            data::dataset_preview,
            data::dataset_describe,
            data::target_distribution,
            data::correlation_matrix,
            data::pairplot_data,
            data::load_csv,
            data::csv_target_info,
            data::set_csv_target,
        ])
        .setup(move |app| {
            if automated {
                spawn_watchdog(app.handle().clone());
            }
            Ok(())
        })
        .build(tauri::generate_context!());

    match app {
        Ok(app) => app.run_return(|_, _| {}),
        Err(error) => {
            log::error!("no se pudo iniciar la ventana: {error}");
            eprintln!("mltutor: no se pudo iniciar la ventana: {error}");
            EXIT_FAILURE
        }
    }
}

/// The computation backend of the data commands: `mltutor-core`. Debug builds can use
/// the golden fixtures instead with `MLTUTOR_FACADE=fixture` (to tell UI bugs from
/// core bugs).
fn facade() -> std::sync::Arc<dyn core_facade::CoreFacade> {
    #[cfg(debug_assertions)]
    if std::env::var("MLTUTOR_FACADE").is_ok_and(|v| v == "fixture") {
        return std::sync::Arc::new(fixture_facade::FixtureFacade::from_checkout());
    }
    std::sync::Arc::new(real_facade::RealFacade)
}

/// Kills an automation run (`--smoke-test`, `--screenshot`) that hangs.
fn spawn_watchdog(app: tauri::AppHandle) {
    let spawned = std::thread::Builder::new()
        .name("mltutor-watchdog".into())
        .spawn(move || {
            std::thread::sleep(AUTOMATION_TIMEOUT);
            let state = app.state::<AppState>();
            if !state.is_finished() {
                log::error!(
                    "tiempo agotado: la interfaz no terminó en {} s",
                    AUTOMATION_TIMEOUT.as_secs()
                );
                eprintln!(
                    "mltutor: tiempo agotado ({} s)",
                    AUTOMATION_TIMEOUT.as_secs()
                );
                log::logger().flush();
                // The event loop may be stuck, so do not rely on `app.exit`.
                std::process::exit(EXIT_TIMEOUT);
            }
        });
    if let Err(error) = spawned {
        log::error!("no se pudo lanzar el watchdog: {error}");
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    fn json_version(relative: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
        let text = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["version"].as_str().unwrap().to_owned()
    }

    /// D4: the version lives in the workspace Cargo.toml and is mirrored elsewhere.
    #[test]
    fn versions_are_in_sync() {
        let version = env!("CARGO_PKG_VERSION");
        assert_eq!(json_version("tauri.conf.json"), version);
        assert_eq!(json_version("../ui/package.json"), version);
    }
}
