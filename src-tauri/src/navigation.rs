//! Navigation guard: the webview may only show the app's own pages.
//!
//! Content shown by the app (Markdown, Plotly labels) is escaped, but as a second line of
//! defence any navigation away from the app origin — a link in a dataset value, a
//! `location = …` — is blocked and logged.

use tauri::plugin::{Builder, TauriPlugin};
use tauri::{Manager, Runtime, Url};

/// Whether the app is served from `http(s)://tauri.localhost`: only on Windows (WebView2
/// cannot register custom schemes); macOS and Linux use `tauri://localhost`.
const SERVED_OVER_HTTP: bool = cfg!(windows);

/// Whether `url` belongs to the app: `tauri://localhost` (macOS, Linux),
/// `http(s)://tauri.localhost` (Windows only) or, in development, the origin of `devUrl`.
pub fn is_allowed(url: &Url, dev_url: Option<&Url>) -> bool {
    allowed_on(url, dev_url, SERVED_OVER_HTTP)
}

fn allowed_on(url: &Url, dev_url: Option<&Url>, served_over_http: bool) -> bool {
    let host = url.host_str();
    match url.scheme() {
        "tauri" => host == Some("localhost"),
        "http" | "https" if served_over_http && host == Some("tauri.localhost") => true,
        "about" => url.path() == "blank",
        _ => dev_url.is_some_and(|dev| dev.origin() == url.origin()),
    }
}

/// Plugin that applies [`is_allowed`] to every navigation of every webview.
pub fn plugin<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("navigation-guard")
        .on_navigation(|webview, url| {
            let dev_url = if cfg!(debug_assertions) {
                webview.app_handle().config().build.dev_url.clone()
            } else {
                None
            };
            let allowed = is_allowed(url, dev_url.as_ref());
            if !allowed {
                log::warn!("navegación bloqueada: {url}");
            }
            allowed
        })
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(text: &str) -> Url {
        Url::parse(text).unwrap()
    }

    #[test]
    fn only_the_app_origin_is_allowed() {
        assert!(is_allowed(&url("tauri://localhost/index.html"), None));
        assert!(is_allowed(&url("about:blank"), None));
        for blocked in [
            "https://example.com/",
            "http://localhost:5173/",
            "tauri://evil/",
            "https://tauri.localhost.evil.com/",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,hola",
            "about:config",
        ] {
            assert!(!is_allowed(&url(blocked), None), "{blocked}");
        }
    }

    /// `http(s)://tauri.localhost` is the app origin only on Windows.
    #[test]
    fn tauri_localhost_over_http_only_on_windows() {
        for app_url in ["http://tauri.localhost/", "https://tauri.localhost/#/csv"] {
            assert!(allowed_on(&url(app_url), None, true), "{app_url}");
            assert!(!allowed_on(&url(app_url), None, false), "{app_url}");
            assert_eq!(is_allowed(&url(app_url), None), cfg!(windows), "{app_url}");
        }
        assert!(!allowed_on(
            &url("https://tauri.localhost.evil.com/"),
            None,
            true
        ));
        assert!(allowed_on(&url("tauri://localhost/"), None, false));
    }

    #[test]
    fn the_dev_server_is_allowed_only_when_given() {
        let dev = url("http://localhost:5173");
        assert!(is_allowed(
            &url("http://localhost:5173/src/main.ts"),
            Some(&dev)
        ));
        assert!(!is_allowed(&url("http://localhost:5174/"), Some(&dev)));
        assert!(!is_allowed(&url("https://localhost:5173/"), Some(&dev)));
    }
}
