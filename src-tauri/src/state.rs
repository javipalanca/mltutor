//! Managed application state and the exit protocol of the automation modes.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::AppHandle;

use crate::cli::Mode;

pub const EXIT_OK: i32 = 0;
pub const EXIT_FAILURE: i32 = 1;
pub const EXIT_TIMEOUT: i32 = 2;

/// `--smoke-test` and `--screenshot` give up after this long.
pub const AUTOMATION_TIMEOUT: Duration = Duration::from_secs(180);

/// Routes the frontend must render in `--smoke-test`, in navigation order.
/// Must match `ui/src/lib/routes.ts` (the frontend fails the test if they differ).
pub const SMOKE_ROUTES: [&str; 6] = ["inicio", "arboles", "regresion", "knn", "redes", "csv"];

pub struct AppState {
    pub mode: Mode,
    pub started: Instant,
    ready: AtomicBool,
    finished: AtomicBool,
    smoke: Mutex<SmokeProgress>,
}

impl AppState {
    pub fn new(mode: Mode, started: Instant) -> Self {
        Self {
            mode,
            started,
            ready: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            smoke: Mutex::new(SmokeProgress::default()),
        }
    }

    /// Returns `true` only the first time the frontend reports it is ready.
    pub fn mark_ready(&self) -> bool {
        !self.ready.swap(true, Ordering::SeqCst)
    }

    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    pub fn record_smoke_route(&self, route: &str, title: &str) -> Result<bool, String> {
        self.smoke
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record(route, title)
    }

    /// Ends an automation run with `code`. Only the first call has an effect.
    pub fn finish(&self, app: &AppHandle, code: i32, message: &str) {
        if self.finished.swap(true, Ordering::SeqCst) {
            return;
        }
        if code == EXIT_OK {
            log::info!("{message}");
            println!("{message}");
        } else {
            log::error!("{message} (código de salida {code})");
            eprintln!("{message}");
        }
        log::logger().flush();
        app.exit(code);
    }
}

/// Routes already reported by the frontend during `--smoke-test`.
#[derive(Debug, Default)]
pub struct SmokeProgress {
    seen: BTreeSet<String>,
}

impl SmokeProgress {
    /// Records a rendered route. Returns `Ok(true)` once every route has been seen.
    pub fn record(&mut self, route: &str, title: &str) -> Result<bool, String> {
        if !SMOKE_ROUTES.contains(&route) {
            return Err(format!("ruta desconocida en el smoke test: {route}"));
        }
        if title.trim().is_empty() {
            return Err(format!("la ruta {route} se ha renderizado sin título"));
        }
        self.seen.insert(route.to_owned());
        Ok(self.seen.len() == SMOKE_ROUTES.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_progress_completes_after_every_route() {
        let mut progress = SmokeProgress::default();
        for (index, route) in SMOKE_ROUTES.iter().enumerate() {
            let done = progress.record(route, "Título").unwrap();
            assert_eq!(done, index + 1 == SMOKE_ROUTES.len());
        }
    }

    #[test]
    fn smoke_progress_ignores_duplicates_and_rejects_bad_reports() {
        let mut progress = SmokeProgress::default();
        assert!(!progress.record("inicio", "Inicio").unwrap());
        assert!(!progress.record("inicio", "Inicio").unwrap());
        assert!(progress.record("otra", "X").is_err());
        assert!(progress.record("knn", "  ").is_err());
    }

    #[test]
    fn smoke_routes_match_the_frontend_routes() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../ui/src/lib/routes.ts");
        let source = std::fs::read_to_string(path).unwrap();
        let ids: Vec<&str> = source
            .lines()
            .filter_map(|line| line.trim().strip_prefix("id: \""))
            .filter_map(|rest| rest.split('"').next())
            .collect();
        assert_eq!(ids, SMOKE_ROUTES);
    }

    #[test]
    fn mark_ready_is_true_only_once() {
        let state = AppState::new(Mode::Normal, Instant::now());
        assert!(state.mark_ready());
        assert!(!state.mark_ready());
    }
}
