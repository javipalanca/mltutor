//! `--screenshot <png>`: captures the main webview and exits.
//!
//! With `--full-page` the viewport (not the window, which macOS keeps within the screen)
//! grows to the height of the page reported by the frontend, so the PNG holds the whole
//! page; the page lays out again at that height before the capture.
//!
//! | Platform | Capture | `--full-page` |
//! |---|---|---|
//! | macOS | `WKWebView.takeSnapshot` | the WKWebView frame grows beyond the window |
//! | Windows | WebView2 `CapturePreview` (PNG) | DevTools protocol: `Emulation.setDeviceMetricsOverride` + `Page.captureScreenshot` |
//! | Linux | `webkit_web_view_get_snapshot` (visible region) | best effort: the window grows (works under Xvfb, no window manager) |
//!
//! Every PNG is checked (signature, `IHDR`, non-zero size) before it is written.

use std::path::{Path, PathBuf};
use std::time::Duration;

use tauri::{AppHandle, Manager};

use crate::state::{AppState, EXIT_FAILURE, EXIT_OK};

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

/// Validates `bytes` as a PNG, writes it to `path` and ends the run (0 or 1).
#[cfg_attr(
    not(any(target_os = "macos", windows, target_os = "linux")),
    allow(dead_code)
)]
fn finish_with_png(app: &AppHandle, path: &Path, bytes: Result<Vec<u8>, String>) {
    let saved = bytes.and_then(|bytes| {
        let (width, height) = png_dimensions(&bytes)?;
        std::fs::write(path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((width, height, bytes.len()))
    });
    match saved {
        Ok((width, height, size)) => app.state::<AppState>().finish(
            app,
            EXIT_OK,
            &format!(
                "Captura guardada en {} ({width}×{height} px, {size} bytes)",
                path.display()
            ),
        ),
        Err(error) => fail(app, &error),
    }
}

/// Width and height from the `IHDR` chunk; an error if `bytes` is not a non-empty PNG.
fn png_dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    let header = bytes
        .get(..24)
        .filter(|head| head.starts_with(SIGNATURE) && &head[12..16] == b"IHDR")
        .ok_or("la captura no es un PNG válido")?;
    let read = |at: usize| {
        u32::from_be_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
    };
    let (width, height) = (read(16), read(20));
    if width == 0 || height == 0 {
        return Err("la captura está vacía (0 píxeles)".to_owned());
    }
    Ok((width, height))
}

/// The PNG inside a `Page.captureScreenshot` response (`{"data": "<base64>"}`).
#[cfg(any(windows, test))]
fn devtools_png(json: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("respuesta de DevTools no válida: {e}"))?;
    let data = value
        .get("data")
        .and_then(serde_json::Value::as_str)
        .ok_or("la respuesta de DevTools no trae la imagen")?;
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| format!("imagen de DevTools no válida: {e}"))
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
                    let bytes = match unsafe { (image.as_ref(), error.as_ref()) } {
                        (Some(image), _) => png_bytes(image),
                        (None, Some(error)) => Err(error.localizedDescription().to_string()),
                        (None, None) => Err("WebKit no devolvió ninguna imagen".to_owned()),
                    };
                    super::finish_with_png(&app, &path, bytes);
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

#[cfg(windows)]
mod platform {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};

    use tauri::{AppHandle, WebviewWindow};
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        ICoreWebView2, COREWEBVIEW2_CAPTURE_PREVIEW_IMAGE_FORMAT_PNG,
    };
    use webview2_com::{
        CallDevToolsProtocolMethodCompletedHandler, CapturePreviewCompletedHandler,
    };
    use windows::core::{HSTRING, PCWSTR};
    use windows::Win32::System::Com::{IStream, STREAM_SEEK_SET};
    use windows::Win32::UI::Shell::SHCreateMemStream;

    /// The viewport was enlarged with the DevTools protocol: `CapturePreview` would only
    /// see the window, so the capture goes through `Page.captureScreenshot` too.
    static EMULATED: AtomicBool = AtomicBool::new(false);

    /// Emulates a viewport of the window's width and `height` CSS px (what Chromium's own
    /// full-page screenshots do). The window, which Windows keeps within the screen, stays.
    pub fn grow_webview(window: &WebviewWindow, height: f64) -> Result<(), String> {
        let size = window.inner_size().map_err(|e| e.to_string())?;
        let scale = window.scale_factor().map_err(|e| e.to_string())?;
        let width = (f64::from(size.width) / scale).round();
        let params = serde_json::json!({
            "width": width,
            "height": height,
            // 1, not the screen's DPR: at 1.5–2× a 16 000 CSS px page would exceed Chromium's
            // ~16 384 px texture limit (MAX_FULL_PAGE_HEIGHT is in CSS px).
            "deviceScaleFactor": 1,
            "mobile": false,
        })
        .to_string();
        window
            .with_webview(move |webview| {
                let issued = core_webview(&webview).and_then(|core| {
                    let handler = CallDevToolsProtocolMethodCompletedHandler::create(Box::new(
                        |result, _json| {
                            if let Err(error) = result {
                                log::warn!("Emulation.setDeviceMetricsOverride: {error}");
                            }
                            Ok(())
                        },
                    ));
                    call_devtools(
                        &core,
                        "Emulation.setDeviceMetricsOverride",
                        &params,
                        &handler,
                    )
                });
                match issued {
                    Ok(()) => EMULATED.store(true, Ordering::SeqCst),
                    Err(error) => log::warn!("no se pudo ampliar la vista: {error}"),
                }
            })
            .map_err(|error| error.to_string())
    }

    pub fn capture(app: AppHandle, window: &WebviewWindow, path: PathBuf) -> Result<(), String> {
        window
            .with_webview(move |webview| {
                // Read here, on the UI thread: `with_webview` closures run in order there, so
                // the one queued by `grow_webview` has already stored the flag.
                let emulated = EMULATED.load(Ordering::SeqCst);
                let started = core_webview(&webview).and_then(|core| {
                    if emulated {
                        capture_devtools(app.clone(), &core, path.clone())
                    } else {
                        capture_preview(app.clone(), &core, path.clone())
                    }
                });
                if let Err(error) = started {
                    super::fail(&app, &error);
                }
            })
            .map_err(|error| error.to_string())
    }

    fn core_webview(webview: &tauri::webview::PlatformWebview) -> Result<ICoreWebView2, String> {
        // SAFETY: wry's controller, used on the UI thread inside `with_webview`.
        unsafe { webview.controller().CoreWebView2() }
            .map_err(|e| format!("WebView2 no está disponible: {e}"))
    }

    fn call_devtools(
        core: &ICoreWebView2,
        method: &str,
        params: &str,
        handler: &webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2CallDevToolsProtocolMethodCompletedHandler,
    ) -> Result<(), String> {
        let method = HSTRING::from(method);
        let params = HSTRING::from(params);
        // SAFETY: the strings outlive the call (WebView2 copies them); UI thread.
        unsafe {
            core.CallDevToolsProtocolMethod(
                PCWSTR(method.as_ptr()),
                PCWSTR(params.as_ptr()),
                handler,
            )
        }
        .map_err(|e| e.to_string())
    }

    /// The visible viewport as PNG, written by WebView2 into an in-memory stream.
    fn capture_preview(app: AppHandle, core: &ICoreWebView2, path: PathBuf) -> Result<(), String> {
        // SAFETY: creates an empty, growable memory stream.
        let stream: IStream =
            unsafe { SHCreateMemStream(None) }.ok_or("no se pudo crear el flujo en memoria")?;
        let reader = stream.clone();
        let handler = CapturePreviewCompletedHandler::create(Box::new(move |result| {
            let bytes = result
                .map_err(|e| format!("CapturePreview: {e}"))
                .and_then(|()| read_stream(&reader));
            super::finish_with_png(&app, &path, bytes);
            Ok(())
        }));
        // SAFETY: valid stream and handler; UI thread.
        unsafe {
            core.CapturePreview(
                COREWEBVIEW2_CAPTURE_PREVIEW_IMAGE_FORMAT_PNG,
                &stream,
                &handler,
            )
        }
        .map_err(|e| format!("CapturePreview: {e}"))
    }

    /// The emulated (full-page) viewport through `Page.captureScreenshot`.
    fn capture_devtools(app: AppHandle, core: &ICoreWebView2, path: PathBuf) -> Result<(), String> {
        let handler =
            CallDevToolsProtocolMethodCompletedHandler::create(Box::new(move |result, json| {
                let bytes = result
                    .map_err(|e| format!("Page.captureScreenshot: {e}"))
                    .and_then(|()| super::devtools_png(&json));
                super::finish_with_png(&app, &path, bytes);
                Ok(())
            }));
        call_devtools(
            core,
            "Page.captureScreenshot",
            r#"{"format":"png","captureBeyondViewport":true}"#,
            &handler,
        )
    }

    fn read_stream(stream: &IStream) -> Result<Vec<u8>, String> {
        // SAFETY: rewinds the stream that CapturePreview has just filled.
        unsafe { stream.Seek(0, STREAM_SEEK_SET, None) }.map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        let mut chunk = vec![0_u8; 256 * 1024];
        loop {
            let mut read = 0_u32;
            // SAFETY: `chunk` is writable for `chunk.len()` bytes; `read` is a valid out pointer.
            let status = unsafe {
                stream.Read(
                    chunk.as_mut_ptr().cast(),
                    u32::try_from(chunk.len()).unwrap_or(u32::MAX),
                    Some(&mut read),
                )
            };
            status.ok().map_err(|e| e.to_string())?;
            let read = usize::try_from(read).unwrap_or(0).min(chunk.len());
            if read == 0 {
                return Ok(bytes);
            }
            bytes.extend_from_slice(&chunk[..read]);
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::path::PathBuf;

    use tauri::{AppHandle, LogicalSize, WebviewWindow};
    use webkit2gtk::{SnapshotOptions, SnapshotRegion, WebViewExt};

    /// Best effort: resizes the window itself. Without a window manager (Xvfb in CI) X11
    /// accepts windows taller than the screen; a desktop WM may clamp it.
    pub fn grow_webview(window: &WebviewWindow, height: f64) -> Result<(), String> {
        let size = window.inner_size().map_err(|e| e.to_string())?;
        let scale = window.scale_factor().map_err(|e| e.to_string())?;
        let width = (f64::from(size.width) / scale).round();
        window
            .set_size(LogicalSize::new(width, height))
            .map_err(|e| e.to_string())
    }

    pub fn capture(app: AppHandle, window: &WebviewWindow, path: PathBuf) -> Result<(), String> {
        window
            .with_webview(move |webview| {
                let view = webview.inner();
                view.snapshot(
                    SnapshotRegion::Visible,
                    SnapshotOptions::NONE,
                    None::<&webkit2gtk::gio::Cancellable>,
                    move |result| {
                        let bytes = result
                            .map_err(|e| format!("WebKitGTK: {e}"))
                            .and_then(surface_png);
                        super::finish_with_png(&app, &path, bytes);
                    },
                );
            })
            .map_err(|error| error.to_string())
    }

    fn surface_png(surface: cairo::Surface) -> Result<Vec<u8>, String> {
        let image = cairo::ImageSurface::try_from(surface)
            .map_err(|_| "WebKitGTK no devolvió una imagen de mapa de bits".to_owned())?;
        let mut bytes = Vec::new();
        image
            .write_to_png(&mut bytes)
            .map_err(|e| format!("no se pudo codificar el PNG: {e}"))?;
        Ok(bytes)
    }
}

#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
mod platform {
    use std::path::PathBuf;

    use tauri::{AppHandle, WebviewWindow};

    pub fn grow_webview(_window: &WebviewWindow, _height: f64) -> Result<(), String> {
        Err("no disponible en esta plataforma".into())
    }

    pub fn capture(_app: AppHandle, _window: &WebviewWindow, _path: PathBuf) -> Result<(), String> {
        Err(format!(
            "--screenshot no está disponible en {}",
            std::env::consts::OS
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header of a 2×1 RGBA PNG (signature, IHDR, IEND; the CRCs are not checked).
    const TINY_PNG: &[u8] = &[
        0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n', 0, 0, 0, 13, b'I', b'H', b'D', b'R', 0,
        0, 0, 2, 0, 0, 0, 1, 8, 6, 0, 0, 0, 0xf4, 0x22, 0x7f, 0x8a, 0, 0, 0, 0, b'I', b'E', b'N',
        b'D', 0xae, 0x42, 0x60, 0x82,
    ];

    #[test]
    fn reads_png_dimensions_and_rejects_other_data() {
        assert_eq!(png_dimensions(TINY_PNG), Ok((2, 1)));
        assert!(png_dimensions(b"GIF89a").is_err());
        assert!(png_dimensions(&TINY_PNG[..20]).is_err());
        let mut empty = TINY_PNG.to_vec();
        empty[16..20].copy_from_slice(&[0, 0, 0, 0]);
        assert!(png_dimensions(&empty).is_err());
    }

    #[test]
    fn decodes_the_devtools_screenshot_response() {
        use base64::Engine as _;
        let data = base64::engine::general_purpose::STANDARD.encode(TINY_PNG);
        let json = format!(r#"{{"data":"{data}"}}"#);
        assert_eq!(devtools_png(&json).unwrap(), TINY_PNG);
        assert!(devtools_png("{}").is_err());
        assert!(devtools_png(r#"{"data":"%%%"}"#).is_err());
        assert!(devtools_png("no json").is_err());
    }
}
