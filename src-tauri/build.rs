fn main() {
    // Declaring the app commands generates one `allow-<command>` permission per command,
    // so capabilities can grant them explicitly instead of exposing all of them.
    let manifest = tauri_build::AppManifest::new().commands(&[
        "launch_mode",
        "frontend_ready",
        "smoke_route_rendered",
        "smoke_failed",
        "smoke_panic_probe",
        "log_message",
    ]);
    if let Err(error) =
        tauri_build::try_build(tauri_build::Attributes::new().app_manifest(manifest))
    {
        panic!("tauri-build failed: {error:#}");
    }
}
