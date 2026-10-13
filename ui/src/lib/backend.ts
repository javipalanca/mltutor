/**
 * Thin typed wrappers around the Tauri IPC commands defined in `src-tauri/src/commands.rs`.
 * Payload types come from `./bindings`, generated from Rust with ts-rs (`cargo test`).
 */
import { invoke, isTauri } from "@tauri-apps/api/core";
import type { LaunchMode } from "./bindings/LaunchMode";
import type { LogLevel } from "./bindings/LogLevel";

export type { LaunchMode, LogLevel };

export { isTauri };

export function launchMode(): Promise<LaunchMode> {
  return invoke<LaunchMode>("launch_mode");
}

/** Tells the backend that the first page is on screen (startup timing, screenshots). */
export function frontendReady(route: string): Promise<void> {
  return invoke("frontend_ready", { route });
}

export function smokeRouteRendered(route: string, title: string): Promise<void> {
  return invoke("smoke_route_rendered", { route, title });
}

export function smokeFailed(message: string): Promise<void> {
  return invoke("smoke_failed", { message });
}

/** Panics on a backend worker thread; resolves `true` if the app survived it. */
export function smokePanicProbe(): Promise<boolean> {
  return invoke<boolean>("smoke_panic_probe");
}

export function logToBackend(level: LogLevel, message: string): Promise<void> {
  return invoke("log_message", { level, message });
}
