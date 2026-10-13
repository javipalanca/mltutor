/// <reference types="vitest/config" />
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { fileURLToPath } from "node:url";
import { parse as parseToml } from "smol-toml";
import { defineConfig, type Plugin } from "vite";

// Tauri expects a fixed port in development and loads `dist/` in release builds.
// See https://v2.tauri.app/start/frontend/vite/
const host = process.env["TAURI_DEV_HOST"];
const contentDir = fileURLToPath(new URL("../content", import.meta.url));

/** `import data from "…/x.toml"` → the parsed object, at build time (no TOML parser shipped). */
function toml(): Plugin {
  return {
    name: "mltutor-toml",
    transform(code, id) {
      if (!id.endsWith(".toml")) {
        return null;
      }
      return { code: `export default ${JSON.stringify(parseToml(code))};`, map: null };
    },
  };
}

export default defineConfig({
  plugins: [toml(), svelte()],
  clearScreen: false,
  resolve: {
    alias: { "@content": contentDir },
    // Svelte 5 components must resolve to their browser build in vitest (jsdom).
    ...(process.env["VITEST"] ? { conditions: ["browser"] } : {}),
  },
  server: {
    port: 5173,
    strictPort: true,
    host: host ?? false,
    ...(host ? { hmr: { protocol: "ws", host, port: 5174 } } : {}),
    watch: { ignored: ["**/src-tauri/**"] },
    // `content/` lives next to `ui/`.
    fs: { allow: [".", contentDir] },
  },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    // WebView2 (Chromium), WKWebView (Safari 15+) and WebKitGTK 2.40+.
    target: ["es2022", "chrome105", "safari15"],
    sourcemap: Boolean(process.env["TAURI_ENV_DEBUG"]),
    // No inline data: URIs, so the CSP can stay strict.
    assetsInlineLimit: 0,
    // Plotly is a large, lazily loaded chunk; that is expected.
    chunkSizeWarningLimit: 5000,
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts"],
    // jsdom has no canvas/WebGL; charts are tested through their figure builders.
    css: false,
  },
});
