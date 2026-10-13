//! `--screenshot <png>`: captures the main webview and exits.
//!
//! With `--full-page` the web view (not the window, which macOS keeps within the screen)
//! grows to the height of the page reported by the frontend, so the PNG holds the whole
//! page; WebKit renders the off-screen part for the snapshot.
//!
//! Implemented natively on macOS (`WKWebView.takeSnapshot`). Windows (WebView2
//! `CapturePreview`) and Linux (`webkit_web_view_get_snapshot`) are not implemented yet:
//! there the command exits with code 1 and a clear message.

use std::path::PathBuf;
use std::time::Duration;

use tauri::{AppHandle, Manager};

use crate::state::{AppState, EXIT_FAILURE};

/// Time for the page (and Plotly's responsive charts) to lay out after a resize.
const RESIZE_SETTLE: Duration = Duration::from_millis(900);
/// Upper bound of a full-page capture, in CSS pixels.
const MAX_FULL_PAGE_HEIGHT: f64 = 16_000.0;

pub async fn capture_main_window(app: &AppHandle, path: PathBuf, full_height: Option<f64>) {
    let Some(window) = app.get_webview_window("main") else {
        fail(app, "no existe la ventana principal");
        return;
    };
    if let Some(height) = full_height {
        let height = height.clamp(200.0, MAX_FULL_PAGE_HEIGHT).ceil();
        if let Err(error) = platform::grow_webview(&window, height) {
            log::warn!("no se pudo ampliar la vista para la captura: {error}");
        }
        let _ = tauri::async_runtime::spawn_blocking(|| std::thread::sleep(RESIZE_SETTLE)).await;
    }
    if let Err(error) = platform::capture(app.clone(), &window, path) {
        fail(app, &error);
    }
}

fn fail(app: &AppHandle, error: &str) {
    app.state::<AppState>().finish(
        app,
        EXIT_FAILURE,
        &format!("No se pudo guardar la captura: {error}"),
    );
}

#[cfg(target_os = "macos")]
fn succeed(app: &AppHandle, path: &std::path::Path) {
    app.state::<AppState>().finish(
        app,
        crate::state::EXIT_OK,
        &format!("Captura guardada en {}", path.display()),
    );
}

#[cfg(target_os = "macos")]
mod platform {
    use std::path::PathBuf;

    use block2::RcBlock;
    use objc2::rc::Retained;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSBitmapImageFileType, NSBitmapImageRep, NSImage};
    use objc2_core_foundation::{CGPoint, CGRect, CGSize};
    use objc2_foundation::{NSDictionary, NSError};
    use objc2_web_kit::{WKSnapshotConfiguration, WKWebView};
    use tauri::{AppHandle, WebviewWindow};

    /// Makes the web view `height` CSS px tall (plus the title-bar inset), beyond the window.
    pub fn grow_webview(window: &WebviewWindow, height: f64) -> Result<(), String> {
        window
            .with_webview(move |webview| {
                let ptr = webview.inner().cast::<WKWebView>();
                // SAFETY: as in `capture`, the WKWebView owned by wry, on the main thread.
                let Some(wk) = (unsafe { ptr.as_ref() }) else {
                    return;
                };
                let mut frame = wk.frame();
                let inset = wk.safeAreaInsets().top;
                let grow = height + inset - frame.size.height;
                if grow > 0.0 {
                    frame.size.height += grow;
                    // AppKit's y axis points up: keep the top edge where it was.
                    frame.origin.y -= grow;
                    wk.setFrame(frame);
                }
            })
            .map_err(|error| error.to_string())
    }

    pub fn capture(app: AppHandle, window: &WebviewWindow, path: PathBuf) -> Result<(), String> {
        window
            .with_webview(move |webview| {
                let ptr = webview.inner().cast::<WKWebView>();
                // SAFETY: on macOS `inner()` is the WKWebView owned by wry; it stays alive
                // while the window exists and this closure runs on the main thread.
                let Some(wk) = (unsafe { ptr.as_ref() }) else {
                    super::fail(&app, "el WKWebView no está disponible");
                    return;
                };
                let Some(mtm) = MainThreadMarker::new() else {
                    super::fail(&app, "la captura debe hacerse en el hilo principal");
                    return;
                };
                // The page viewport excludes the area under the title bar (safe-area
                // inset); without this rect the PNG ends with an empty strip.
                let bounds = wk.bounds();
                let top_inset = wk.safeAreaInsets().top;
                let rect = CGRect::new(
                    CGPoint::new(0.0, 0.0),
                    CGSize::new(bounds.size.width, bounds.size.height - top_inset),
                );
                // SAFETY: plain configuration object created on the main thread.
                let config = unsafe { WKSnapshotConfiguration::new(mtm) };
                // SAFETY: `rect` lies within the web view bounds.
                unsafe { config.setRect(rect) };
                let handler = RcBlock::new(move |image: *mut NSImage, error: *mut NSError| {
                    // SAFETY: WebKit passes either a valid image or a valid error (or null).
                    let result = match unsafe { (image.as_ref(), error.as_ref()) } {
                        (Some(image), _) => png_bytes(image).and_then(|bytes| {
                            std::fs::write(&path, bytes)
                                .map_err(|e| format!("{}: {e}", path.display()))
                        }),
                        (None, Some(error)) => Err(error.localizedDescription().to_string()),
                        (None, None) => Err("WebKit no devolvió ninguna imagen".to_owned()),
                    };
                    match result {
                        Ok(()) => super::succeed(&app, &path),
                        Err(error) => super::fail(&app, &error),
                    }
                });
                // SAFETY: called on the main thread with a valid web view and handler.
                unsafe {
                    wk.takeSnapshotWithConfiguration_completionHandler(Some(&config), &handler)
                };
            })
            .map_err(|error| error.to_string())
    }

    fn png_bytes(image: &NSImage) -> Result<Vec<u8>, String> {
        let tiff = image
            .TIFFRepresentation()
            .ok_or("no se pudo convertir la imagen a TIFF")?;
        let rep: Retained<NSBitmapImageRep> = NSBitmapImageRep::imageRepWithData(&tiff)
            .ok_or("no se pudo leer la imagen capturada")?;
        let properties = NSDictionary::new();
        // SAFETY: an empty properties dictionary is valid for PNG output.
        let png = unsafe {
            rep.representationUsingType_properties(NSBitmapImageFileType::PNG, &properties)
        }
        .ok_or("no se pudo codificar el PNG")?;
        Ok(png.to_vec())
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use std::path::PathBuf;

    use tauri::{AppHandle, WebviewWindow};

    pub fn grow_webview(_window: &WebviewWindow, _height: f64) -> Result<(), String> {
        Err("no disponible en esta plataforma".into())
    }

    pub fn capture(_app: AppHandle, _window: &WebviewWindow, _path: PathBuf) -> Result<(), String> {
        Err(format!(
            "--screenshot todavía no está disponible en {}",
            std::env::consts::OS
        ))
    }
}
