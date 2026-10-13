/// <reference types="vitest/config" />
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// Tauri expects a fixed port in development and loads `dist/` in release builds.
// See https://v2.tauri.app/start/frontend/vite/
const host = process.env["TAURI_DEV_HOST"];

export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    host: host ?? false,
    ...(host ? { hmr: { protocol: "ws", host, port: 5174 } } : {}),
    watch: { ignored: ["**/src-tauri/**"] },
  },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    // WebView2 (Chromium), WKWebView (Safari 15+) and WebKitGTK 2.40+.
    target: ["es2022", "chrome105", "safari15"],
    sourcemap: Boolean(process.env["TAURI_ENV_DEBUG"]),
    // No inline data: URIs, so the CSP can stay strict.
    assetsInlineLimit: 0,
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
