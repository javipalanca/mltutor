//! Saving files chosen by the user ("Guardar imagen…", "Descargar código").
//!
//! The frontend has no filesystem access: `save_file` opens the native save dialog here
//! and writes the bytes it received as the raw IPC body (no JSON/base64 round trip for
//! PNGs). The suggested name and the kind travel in headers.

use std::path::{Path, PathBuf};

use tauri::ipc::{InvokeBody, Request};
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

use crate::data::blocking;
use crate::ipc::CommandError;

/// Header with the suggested file name, percent-encoded (`encodeURIComponent`).
pub const NAME_HEADER: &str = "x-mltutor-file-name";
/// Header with the kind of file: `png`, `svg`, `python` or `text`.
pub const KIND_HEADER: &str = "x-mltutor-file-kind";
/// Larger bodies are rejected (a 2× PNG of the biggest pairplot is a few MB). The IPC
/// layer has already received the whole body when the command runs, so this limit does not
/// bound memory use: it only refuses to write absurd files.
const MAX_SAVE_BYTES: usize = 64 * 1024 * 1024;

/// What can be saved, with the filter of the save dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveKind {
    Png,
    Svg,
    Python,
    Text,
}

impl SaveKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "png" => Some(Self::Png),
            "svg" => Some(Self::Svg),
            "python" => Some(Self::Python),
            "text" => Some(Self::Text),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Svg => "svg",
            Self::Python => "py",
            Self::Text => "txt",
        }
    }

    pub fn filter_name(self) -> &'static str {
        match self {
            Self::Png => "Imagen PNG",
            Self::Svg => "Imagen SVG",
            Self::Python => "Código Python",
            Self::Text => "Texto",
        }
    }
}

/// Decodes `%XX` escapes (invalid ones are kept literally).
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| char::from(b).to_digit(16);
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push(u8::try_from(hi * 16 + lo).unwrap_or(b'?'));
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The suggested name, without directories or characters that file systems reject,
/// and with the extension of `kind`.
pub fn suggested_name(encoded: &str, kind: SaveKind) -> String {
    let decoded = percent_decode(encoded);
    let base = decoded.rsplit(['/', '\\']).next().unwrap_or_default();
    let mut name: String = base
        .chars()
        .filter(|c| !c.is_control() && !matches!(c, ':' | '*' | '?' | '"' | '<' | '>' | '|'))
        .take(120)
        .collect();
    name = name.trim().trim_start_matches('.').to_owned();
    if name.is_empty() {
        name = "archivo".to_owned();
    }
    let extension = format!(".{}", kind.extension());
    if !name.to_ascii_lowercase().ends_with(&extension) {
        name.push_str(&extension);
    }
    name
}

/// Where to write the file chosen in the save dialog.
#[derive(Debug, PartialEq, Eq)]
enum SaveTarget {
    /// The dialog already confirmed this path (it has an extension, or it is new).
    Ready(PathBuf),
    /// The user removed the extension and `path` (chosen path + extension) already
    /// exists: the dialog only asked about the path without extension (GTK), so the
    /// overwrite must be confirmed again.
    Confirm(PathBuf),
}

/// The chosen path, with the extension of `kind` if the user removed it. `exists` is
/// `Path::exists` outside the tests.
fn save_target(picked: PathBuf, kind: SaveKind, exists: impl Fn(&Path) -> bool) -> SaveTarget {
    if picked.extension().is_some() {
        return SaveTarget::Ready(picked);
    }
    let mut path = picked;
    path.set_extension(kind.extension());
    if exists(&path) {
        SaveTarget::Confirm(path)
    } else {
        SaveTarget::Ready(path)
    }
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn header<'a>(request: &'a Request<'_>, name: &str) -> Option<&'a str> {
    request.headers().get(name).and_then(|v| v.to_str().ok())
}

/// Opens the save dialog (with `name` and the filter of `kind`, so the native dialog
/// checks the overwrite of the final name) until the user picks a path or cancels.
fn ask_path(app: &AppHandle, name: &str, kind: SaveKind) -> Result<Option<PathBuf>, CommandError> {
    let window = app.get_webview_window("main");
    loop {
        let mut dialog = app
            .dialog()
            .file()
            .set_file_name(name)
            .add_filter(kind.filter_name(), &[kind.extension()]);
        if let Some(window) = &window {
            dialog = dialog.set_parent(window);
        }
        let Some(picked) = dialog.blocking_save_file() else {
            return Ok(None);
        };
        let picked = picked.into_path().map_err(CommandError::file_save)?;
        match save_target(picked, kind, Path::exists) {
            SaveTarget::Ready(path) => return Ok(Some(path)),
            SaveTarget::Confirm(path) => {
                let mut question = app
                    .dialog()
                    .message(format!(
                        "Ya existe un archivo llamado «{}». ¿Quieres reemplazarlo?",
                        display_name(&path)
                    ))
                    .title("Guardar archivo")
                    .kind(MessageDialogKind::Warning)
                    .buttons(MessageDialogButtons::OkCancelCustom(
                        "Reemplazar".to_owned(),
                        "Cancelar".to_owned(),
                    ));
                if let Some(window) = &window {
                    question = question.parent(window);
                }
                if question.blocking_show() {
                    return Ok(Some(path));
                }
                // "Cancelar": back to the save dialog to choose another name.
            }
        }
    }
}

/// Opens the save dialog and writes the raw body of the request to the chosen file.
/// Returns the saved file name, or `None` if the user cancelled.
#[tauri::command]
pub async fn save_file(
    app: AppHandle,
    request: Request<'_>,
) -> Result<Option<String>, CommandError> {
    let InvokeBody::Raw(bytes) = request.body() else {
        return Err(CommandError::file_save("el contenido no es binario"));
    };
    if bytes.len() > MAX_SAVE_BYTES {
        return Err(CommandError::file_save("el archivo es demasiado grande"));
    }
    let kind = header(&request, KIND_HEADER)
        .and_then(SaveKind::parse)
        .ok_or_else(|| CommandError::file_save("tipo de archivo no válido"))?;
    let name = suggested_name(header(&request, NAME_HEADER).unwrap_or_default(), kind);
    let Some(path) = blocking(move || ask_path(&app, &name, kind)).await? else {
        return Ok(None);
    };
    // `Request` only borrows the body and `spawn_blocking` needs owned data, so one copy
    // is unavoidable; it is made only once the user has chosen where to save.
    let bytes = bytes.clone();
    blocking(move || {
        std::fs::write(&path, &bytes).map_err(CommandError::file_save)?;
        log::info!(
            "archivo guardado: {} ({} bytes)",
            path.display(),
            bytes.len()
        );
        Ok(Some(display_name(&path)))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_have_their_extension_and_filter() {
        assert_eq!(SaveKind::parse("png"), Some(SaveKind::Png));
        assert_eq!(
            SaveKind::parse("python").map(SaveKind::extension),
            Some("py")
        );
        assert_eq!(SaveKind::parse("exe"), None);
        assert_eq!(SaveKind::parse("svg").map(SaveKind::extension), Some("svg"));
        assert_eq!(
            SaveKind::parse("text").map(SaveKind::extension),
            Some("txt")
        );
        assert_eq!(SaveKind::Python.filter_name(), "Código Python");
    }

    #[test]
    fn suggested_names_are_plain_file_names() {
        assert_eq!(
            suggested_name("matriz_correlacion.png", SaveKind::Png),
            "matriz_correlacion.png"
        );
        assert_eq!(
            suggested_name("an%C3%A1lisis%20exploratorio.py", SaveKind::Python),
            "análisis exploratorio.py"
        );
        assert_eq!(
            suggested_name("..%2F..%2Fetc%2Fpasswd", SaveKind::Png),
            "passwd.png"
        );
        assert_eq!(suggested_name("C:\\x\\a<b>.PNG", SaveKind::Png), "ab.PNG");
        assert_eq!(suggested_name("", SaveKind::Python), "archivo.py");
        assert_eq!(suggested_name("%zz%4", SaveKind::Png), "%zz%4.png");
        assert_eq!(suggested_name(".oculto", SaveKind::Png), "oculto.png");
    }

    #[test]
    fn the_extension_is_added_when_missing() {
        let nothing = |_: &Path| false;
        assert_eq!(
            save_target(PathBuf::from("/tmp/grafico"), SaveKind::Png, nothing),
            SaveTarget::Ready(PathBuf::from("/tmp/grafico.png"))
        );
        assert_eq!(
            save_target(PathBuf::from("/tmp/a.txt"), SaveKind::Python, |_| true),
            SaveTarget::Ready(PathBuf::from("/tmp/a.txt")),
            "the dialog already confirmed a path with an extension"
        );
    }

    /// GTK only confirms the overwrite of the name typed by the user: if the user
    /// removed the extension, an existing `grafico.png` must not be replaced silently.
    #[test]
    fn an_existing_file_with_the_added_extension_needs_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("grafico.png"), b"previo").unwrap();
        assert_eq!(
            save_target(dir.path().join("grafico"), SaveKind::Png, Path::exists),
            SaveTarget::Confirm(dir.path().join("grafico.png"))
        );
        assert_eq!(
            save_target(dir.path().join("otro"), SaveKind::Png, Path::exists),
            SaveTarget::Ready(dir.path().join("otro.png"))
        );
        assert_eq!(display_name(&dir.path().join("grafico.png")), "grafico.png");
    }
}
