fn main() {
    // Declaring the app commands generates one `allow-<command>` permission per command,
    // so capabilities can grant them explicitly instead of exposing all of them.
    let manifest = tauri_build::AppManifest::new().commands(&[
        "launch_mode",
        "frontend_ready",
        "screenshot_ready",
        "smoke_route_rendered",
        "smoke_check_passed",
        "smoke_failed",
        "smoke_panic_probe",
        "log_message",
        "list_datasets",
        "get_selected_dataset",
        "select_dataset",
        "dataset_summary",
        "dataset_preview",
        "dataset_describe",
        "target_distribution",
        "correlation_matrix",
        "pairplot_data",
        "cancel_task",
        "pick_and_load_csv",
        "load_scenario_csv",
        "csv_target_info",
        "set_csv_target",
        "discard_csv",
        "save_file",
        "train_model",
        "lesson_model",
        "evaluate_model",
        "roc_curves",
        "probability_view",
        "residuals_view",
        "decision_grid",
        "coefficients_view",
        "prediction_form",
        "predict_one",
        "export_model",
        "generate_code",
    ]);
    if let Err(error) =
        tauri_build::try_build(tauri_build::Attributes::new().app_manifest(manifest))
    {
        panic!("tauri-build failed: {error:#}");
    }
}
