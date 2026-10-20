//! D3: the system web engine, checked before any window exists.
//!
//! On Windows the app renders in Microsoft Edge WebView2. It ships with Windows 10/11, but a
//! lab image may lack it; without it Tauri cannot create the window, and the portable exe
//! (GUI subsystem, no console) would just vanish. So, before building the app:
//!
//! 1. Plan B for locked-down machines: a Fixed Version runtime unpacked into a `WebView2`
//!    folder next to `mltutor.exe` is used through `WEBVIEW2_BROWSER_EXECUTABLE_FOLDER`.
//! 2. `tauri::webview_version()` (wry → `GetAvailableCoreWebView2BrowserVersionString`, the
//!    same loader and overrides that window creation uses) must report a version.
//! 3. Otherwise, a native message box (the webview cannot show it) explains how to install
//!    the Evergreen runtime without admin rights and offers to open the download page. The
//!    same text goes to the log. Automation modes (`--smoke-test`, `--screenshot`) never
//!    block on a dialog: they log and exit with 1.
//!
//! The texts live in `content/app/arranque.toml` (rule 8). They are compiled in with
//! `include_str!`, so they are available before Tauri starts; a unit test parses them.
//!
//! There is no installer (`bundle.active: false` on Windows), so Tauri's
//! `webviewInstallMode` (which only affects NSIS/MSI) does not apply: this check replaces it.

// The notice and the plan-B lookup only exist on Windows; the tests build them everywhere.

use std::path::Path;
#[cfg(any(windows, test))]
use std::path::PathBuf;

#[cfg(any(windows, test))]
use mltutor_core::content::Messages;

/// The variable that points WebView2's loader at a Fixed Version runtime (plan B).
#[cfg(windows)]
const FIXED_RUNTIME_ENV: &str = "WEBVIEW2_BROWSER_EXECUTABLE_FOLDER";
/// Name prefix of the folder that a Fixed Version archive unpacks to.
#[cfg(any(windows, test))]
const FIXED_RUNTIME_PREFIX: &str = "Microsoft.WebView2.FixedVersionRuntime.";

#[cfg(any(windows, test))]
const MESSAGES_FILE: &str = "content/app/arranque.toml";
#[cfg(any(windows, test))]
const MESSAGES: &str = include_str!("../../content/app/arranque.toml");

/// Used only if the embedded content could not be read (a unit test makes that impossible
/// in a release build); never leave the student without the link.
#[cfg(any(windows, test))]
const FALLBACK_URL: &str = "https://developer.microsoft.com/microsoft-edge/webview2/";

/// A native notice: title, body and the page that the "Yes" button opens.
#[cfg(any(windows, test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notice {
    pub title: String,
    pub body: String,
    pub url: String,
}

/// Which startup problem to explain (a `[section]` of `arranque.toml`).
#[cfg(any(windows, test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Problem {
    /// No WebView2 Runtime was found before creating the window.
    MissingWebView2,
    /// Tauri could not create the window (corrupt runtime, policy…).
    WindowFailed,
}

#[cfg(any(windows, test))]
impl Problem {
    fn section(self) -> &'static str {
        match self {
            Self::MissingWebView2 => "webview2",
            Self::WindowFailed => "ventana",
        }
    }
}

/// Builds the notice for `problem` from `content/app/arranque.toml`.
#[cfg(any(windows, test))]
pub fn notice(problem: Problem, detail: &str, log_path: Option<&Path>) -> Notice {
    let log = log_path.map_or_else(
        || "no disponible".to_owned(),
        |path| path.display().to_string(),
    );
    let from_content = Messages::parse(MESSAGES_FILE, MESSAGES).and_then(|messages| {
        let section = problem.section();
        let url = messages.text("webview2", "url")?.to_owned();
        let title = messages.text(section, "titulo")?.to_owned();
        let body = messages
            .text(section, "mensaje")?
            .replace("{url}", &url)
            .replace("{detalle}", detail)
            .replace("{registro}", &log);
        Ok(Notice { title, body, url })
    });
    from_content.unwrap_or_else(|error| {
        log::error!("no se pudo leer {MESSAGES_FILE}: {error}");
        Notice {
            title: "MLTutor no puede arrancar".to_owned(),
            body: format!(
                "Falta Microsoft Edge WebView2 Runtime. Descárgalo desde {FALLBACK_URL}\n\n\
                 Detalle técnico: {detail}\nRegistro: {log}"
            ),
            url: FALLBACK_URL.to_owned(),
        }
    })
}

/// `None` if a usable runtime version was reported, otherwise why not.
pub fn missing_runtime(version: &Result<String, String>) -> Option<String> {
    match version {
        Ok(version) if !version.trim().is_empty() => None,
        Ok(_) => Some("no se encontró ninguna versión instalada".to_owned()),
        Err(error) => Some(error.clone()),
    }
}

/// Plan B: a Fixed Version runtime in `dir` (the folder itself, or the single
/// `Microsoft.WebView2.FixedVersionRuntime.*` folder that its archive unpacks to).
#[cfg(any(windows, test))]
pub fn find_fixed_runtime(dir: &Path) -> Option<PathBuf> {
    const EXE: &str = "msedgewebview2.exe";
    if dir.join(EXE).is_file() {
        return Some(dir.to_path_buf());
    }
    let mut found = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(FIXED_RUNTIME_PREFIX))
        })
        .map(|entry| entry.path())
        .filter(|path| path.join(EXE).is_file());
    let first = found.next()?;
    // Two candidate folders: ambiguous, let the system runtime decide.
    found.next().is_none().then_some(first)
}

/// Checks the web engine before the app is built. Returns `false` if the app cannot start
/// (the student has already been told how to fix it).
pub fn ensure_available(interactive: bool, log_path: Option<&Path>) -> bool {
    #[cfg(windows)]
    use_bundled_fixed_runtime();

    let version = tauri::webview_version().map_err(|error| error.to_string());
    let Some(detail) = missing_runtime(&version) else {
        log::info!(
            "motor web del sistema: {}",
            version.as_deref().unwrap_or_default()
        );
        return true;
    };
    #[cfg(windows)]
    {
        report(Problem::MissingWebView2, &detail, interactive, log_path);
        false
    }
    #[cfg(not(windows))]
    {
        // WKWebView and WebKitGTK are linked libraries: if the binary runs, they exist.
        // Only log; window creation reports any real failure.
        let _ = (interactive, log_path);
        log::warn!("no se pudo leer la versión del motor web: {detail}");
        true
    }
}

/// Logs the problem and, in an interactive session, shows the native notice.
#[cfg(windows)]
pub fn report(problem: Problem, detail: &str, interactive: bool, log_path: Option<&Path>) {
    let detail = with_runtime_folder(problem, detail, std::env::var_os(FIXED_RUNTIME_ENV));
    let notice = notice(problem, &detail, log_path);
    log::error!("{}\n{}", notice.title, notice.body);
    log::logger().flush();
    if interactive {
        windows_ui::show(&notice);
    }
}

/// If the window failed while a Fixed Version runtime was in use, say which one: a broken
/// or incomplete plan-B folder is then the likely cause.
#[cfg(any(windows, test))]
fn with_runtime_folder(
    problem: Problem,
    detail: &str,
    folder: Option<std::ffi::OsString>,
) -> String {
    match (problem, folder) {
        (Problem::WindowFailed, Some(folder)) => format!(
            "{detail} (WebView2 Fixed Version en {})",
            Path::new(&folder).display()
        ),
        _ => detail.to_owned(),
    }
}

#[cfg(windows)]
fn use_bundled_fixed_runtime() {
    if let Some(folder) = std::env::var_os(FIXED_RUNTIME_ENV) {
        log::info!(
            "WebView2 de {FIXED_RUNTIME_ENV}={}",
            Path::new(&folder).display()
        );
        return;
    }
    let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.join("WebView2")))
    else {
        return;
    };
    if let Some(folder) = find_fixed_runtime(&dir) {
        log::info!(
            "WebView2 Fixed Version junto al ejecutable: {}",
            folder.display()
        );
        // Called from `main` before any other thread exists (the logger spawns none).
        std::env::set_var(FIXED_RUNTIME_ENV, &folder);
    }
}

#[cfg(windows)]
mod windows_ui {
    use windows::core::{w, HSTRING, PCWSTR};
    use windows::Win32::System::Com::{
        CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE,
    };
    use windows::Win32::UI::Shell::ShellExecuteW;
    use windows::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, IDYES, MB_ICONWARNING, MB_SETFOREGROUND, MB_YESNO, SW_SHOWNORMAL,
    };

    use super::Notice;

    /// A blocking `MessageBoxW` (no window or COM needed); "Sí" opens the download page.
    pub fn show(notice: &Notice) {
        let title = HSTRING::from(notice.title.as_str());
        let body = HSTRING::from(notice.body.as_str());
        // SAFETY: valid null-terminated wide strings that outlive the call; no owner window.
        let answer = unsafe {
            MessageBoxW(
                None,
                PCWSTR(body.as_ptr()),
                PCWSTR(title.as_ptr()),
                MB_YESNO | MB_ICONWARNING | MB_SETFOREGROUND,
            )
        };
        if answer == IDYES {
            let url = HSTRING::from(notice.url.as_str());
            // ShellExecuteW may use COM shell extensions: Microsoft asks for an STA first.
            // SAFETY: plain COM initialisation of this (main) thread, paired below.
            let com =
                unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) };
            // SAFETY: as above; the shell opens the URL in the default browser.
            let result = unsafe {
                ShellExecuteW(
                    None,
                    w!("open"),
                    PCWSTR(url.as_ptr()),
                    PCWSTR::null(),
                    PCWSTR::null(),
                    SW_SHOWNORMAL,
                )
            };
            // S_OK and S_FALSE both need a matching CoUninitialize; a failure (another
            // apartment mode) does not.
            if com.is_ok() {
                // SAFETY: balances the successful CoInitializeEx above, on the same thread.
                unsafe { CoUninitialize() };
            }
            // ShellExecuteW reports success with a value greater than 32.
            if result.0 as usize <= 32 {
                log::warn!(
                    "no se pudo abrir {}: código {}",
                    notice.url,
                    result.0 as usize
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notices_come_from_content_with_every_placeholder_filled() {
        for problem in [Problem::MissingWebView2, Problem::WindowFailed] {
            let notice = notice(problem, "0x80070002", Some(Path::new("C:\\x\\mltutor.log")));
            assert_eq!(notice.url, FALLBACK_URL);
            assert!(notice.title.contains("MLTutor"), "{notice:?}");
            assert!(notice.body.contains(FALLBACK_URL), "{notice:?}");
            assert!(notice.body.contains("0x80070002"), "{notice:?}");
            assert!(notice.body.contains("C:\\x\\mltutor.log"), "{notice:?}");
            assert!(!notice.body.contains('{'), "placeholder left: {notice:?}");
            assert!(!notice.body.starts_with('\n'));
        }
        let missing = notice(Problem::MissingWebView2, "x", None);
        assert!(missing.body.contains("WebView2 junto a mltutor.exe"));
        assert!(missing.body.contains("no disponible"));
    }

    #[test]
    fn missing_runtime_needs_a_non_empty_version() {
        assert_eq!(missing_runtime(&Ok("130.0.2849.80".into())), None);
        assert!(missing_runtime(&Ok("  ".into())).is_some());
        assert_eq!(
            missing_runtime(&Err("HRESULT 0x80070002".into())).as_deref(),
            Some("HRESULT 0x80070002")
        );
    }

    #[test]
    fn finds_a_fixed_runtime_next_to_the_exe() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("WebView2");
        assert_eq!(find_fixed_runtime(&dir), None);
        std::fs::create_dir(&dir).unwrap();
        assert_eq!(find_fixed_runtime(&dir), None);

        let nested = dir.join("Microsoft.WebView2.FixedVersionRuntime.130.0.2849.80.x64");
        std::fs::create_dir(&nested).unwrap();
        std::fs::write(nested.join("msedgewebview2.exe"), b"").unwrap();
        assert_eq!(find_fixed_runtime(&dir), Some(nested.clone()));

        let other = dir.join("Microsoft.WebView2.FixedVersionRuntime.131.x64");
        std::fs::create_dir(&other).unwrap();
        std::fs::write(other.join("msedgewebview2.exe"), b"").unwrap();
        assert_eq!(
            find_fixed_runtime(&dir),
            None,
            "two candidates are ambiguous"
        );

        std::fs::write(dir.join("msedgewebview2.exe"), b"").unwrap();
        assert_eq!(find_fixed_runtime(&dir), Some(dir));
    }

    #[test]
    fn ignores_folders_without_the_fixed_runtime_prefix() {
        let root = tempfile::tempdir().unwrap();
        let unrelated = root.path().join("otra-cosa");
        std::fs::create_dir(&unrelated).unwrap();
        std::fs::write(unrelated.join("msedgewebview2.exe"), b"").unwrap();
        assert_eq!(find_fixed_runtime(root.path()), None);
    }

    #[test]
    fn window_failures_name_the_fixed_runtime_in_use() {
        let folder = Some(std::ffi::OsString::from("D:\\MLTutor\\WebView2"));
        assert_eq!(
            with_runtime_folder(Problem::WindowFailed, "0x1", folder.clone()),
            "0x1 (WebView2 Fixed Version en D:\\MLTutor\\WebView2)"
        );
        assert_eq!(
            with_runtime_folder(Problem::WindowFailed, "0x1", None),
            "0x1"
        );
        assert_eq!(
            with_runtime_folder(Problem::MissingWebView2, "0x1", folder),
            "0x1"
        );
    }
}
