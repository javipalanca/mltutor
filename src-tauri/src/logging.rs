//! File logging to `~/.mltutor/mltutor.log` and the panic hook.
//!
//! The log is the first thing to check when the app misbehaves on a student's
//! machine (there is no console on Windows). The Python version writes to the same file.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

use log::{Level, LevelFilter, Log, Metadata, Record};

pub const LOG_FILE_NAME: &str = "mltutor.log";
/// Override of the log directory (CI, tests). Defaults to `~/.mltutor`.
pub const LOG_DIR_ENV: &str = "MLTUTOR_LOG_DIR";
/// Above this size the log is moved to `mltutor.log.1` at startup.
const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

static PANICS: AtomicUsize = AtomicUsize::new(0);

/// Number of panics recorded by the hook since startup.
pub fn panic_count() -> usize {
    PANICS.load(Ordering::SeqCst)
}

pub fn log_dir() -> Option<PathBuf> {
    match std::env::var_os(LOG_DIR_ENV) {
        Some(dir) if !dir.is_empty() => Some(PathBuf::from(dir)),
        _ => dirs::home_dir().map(|home| home.join(".mltutor")),
    }
}

struct FileLogger {
    file: Mutex<Option<File>>,
}

impl Log for FileLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        // Our own crate logs at debug; dependencies (tao, wry…) only from info.
        metadata.level() <= Level::Info
            || (metadata.level() <= Level::Debug && metadata.target().starts_with("mltutor"))
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format_line(
            SystemTime::now(),
            record.level(),
            record.target(),
            record.args(),
        );
        // Logging must never panic: a poisoned lock still holds a usable file.
        let mut guard = self
            .file
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(file) = guard.as_mut() {
            let _ = file.write_all(line.as_bytes());
        }
        if cfg!(debug_assertions) || record.level() <= Level::Warn {
            let _ = io::stderr().write_all(line.as_bytes());
        }
    }

    fn flush(&self) {
        let mut guard = self
            .file
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(file) = guard.as_mut() {
            let _ = file.flush();
        }
    }
}

fn format_line(
    time: SystemTime,
    level: Level,
    target: &str,
    message: &std::fmt::Arguments<'_>,
) -> String {
    format!(
        "{} {level:<5} [{target}] {message}\n",
        humantime::format_rfc3339_millis(time)
    )
}

fn open_log_file(dir: &Path) -> io::Result<File> {
    fs::create_dir_all(dir)?;
    let path = dir.join(LOG_FILE_NAME);
    if fs::metadata(&path).is_ok_and(|meta| meta.len() > MAX_LOG_BYTES) {
        let _ = fs::rename(&path, dir.join(format!("{LOG_FILE_NAME}.1")));
    }
    OpenOptions::new().create(true).append(true).open(path)
}

/// Installs the global logger. If the log file cannot be opened (read-only home,
/// permissions…) the app keeps running and logs only to stderr.
/// Returns the log file path when it could be opened.
pub fn init() -> Option<PathBuf> {
    let dir = log_dir();
    let (file, path, error) = match dir.as_deref().map(|dir| (dir, open_log_file(dir))) {
        Some((dir, Ok(file))) => (Some(file), Some(dir.join(LOG_FILE_NAME)), None),
        Some((dir, Err(error))) => (None, None, Some(format!("{}: {error}", dir.display()))),
        None => (
            None,
            None,
            Some("no se encontró el directorio personal".to_owned()),
        ),
    };
    let logger = FileLogger {
        file: Mutex::new(file),
    };
    if log::set_boxed_logger(Box::new(logger)).is_ok() {
        log::set_max_level(LevelFilter::Debug);
    }
    if let Some(error) = error {
        log::warn!("no se puede escribir el registro: {error}");
    }
    path
}

/// Installs a panic hook that records every panic (message, location and backtrace)
/// in the log before the default hook runs. With `panic = "unwind"` a panic on a
/// worker/async thread only ends that task; the window stays open.
pub fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        PANICS.fetch_add(1, Ordering::SeqCst);
        let thread = std::thread::current();
        let message = panic_message(info.payload());
        let location = info
            .location()
            .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
            .unwrap_or_else(|| "ubicación desconocida".to_owned());
        // Release binaries are stripped, so their frames carry no symbols: only capture
        // there when RUST_BACKTRACE asks for it.
        let backtrace = if cfg!(debug_assertions) {
            std::backtrace::Backtrace::force_capture()
        } else {
            std::backtrace::Backtrace::capture()
        };
        log::error!(
            "panic en el hilo '{}' ({location}): {message}\n{backtrace}",
            thread.name().unwrap_or("<sin nombre>")
        );
        log::logger().flush();
        default_hook(info);
    }));
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(text) = payload.downcast_ref::<&str>() {
        text
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text
    } else {
        "<payload desconocido>"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_lines_with_timestamp_level_and_target() {
        let line = format_line(
            SystemTime::UNIX_EPOCH,
            Level::Info,
            "mltutor::app",
            &format_args!("hola {}", 1),
        );
        assert_eq!(
            line,
            "1970-01-01T00:00:00.000Z INFO  [mltutor::app] hola 1\n"
        );
    }

    #[test]
    fn extracts_panic_messages() {
        assert_eq!(panic_message(&"texto"), "texto");
        assert_eq!(panic_message(&String::from("cadena")), "cadena");
        assert_eq!(panic_message(&42_u8), "<payload desconocido>");
    }

    #[test]
    fn rotates_large_logs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LOG_FILE_NAME);
        fs::write(&path, vec![b'x'; (MAX_LOG_BYTES + 1) as usize]).unwrap();
        let mut file = open_log_file(dir.path()).unwrap();
        file.write_all(b"nuevo\n").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"nuevo\n");
        assert!(dir.path().join("mltutor.log.1").exists());
    }
}
