/**
 * Data checks of `--smoke-test` (fase1-shell-datos.md §8.2.5): 📊 Datos of a lesson with
 * Iris, then its correlation heatmap and its pairplot drawn by Plotly (WebGL included).
 */
import * as backend from "./backend";
import { busy } from "./busy.svelte";
import { hasWebGL } from "./charts";
import { waitFor, waitForIdle } from "./dom";
import { router } from "./router.svelte";
import { chooseDataset } from "./selection";
import { session } from "./session.svelte";

const LESSON = "regresion";
const IRIS = { kind: "builtin", id: "iris" } as const;

async function datosVistas(): Promise<void> {
  await chooseDataset(IRIS);
  session.activeTab[LESSON] = 0;
  router.go(LESSON);
  await waitFor(`[data-dataset-tab="${LESSON}"][data-ready="true"]`, 30000, "la pestaña 📊 Datos");
  await waitFor(`[data-dataset-tab="${LESSON}"] table`, 10000, "las tablas");
  await waitFor(`[data-dataset-tab="${LESSON}"] [data-plot-rendered="true"]`, 30000, "el gráfico del objetivo");
}

async function correlacion(): Promise<void> {
  session.datasetTab(LESSON, IRIS).analysis = "correlation";
  await waitFor('[data-analysis="correlation"] [data-plot-rendered="true"]', 30000, "el heatmap de correlación");
  const cells = document.querySelectorAll('[data-analysis="correlation"] .heatmaplayer, [data-analysis="correlation"] image');
  if (cells.length === 0) {
    throw new Error("el heatmap no tiene celdas");
  }
}

async function pairplot(): Promise<void> {
  const state = session.datasetTab(LESSON, IRIS);
  state.analysis = "pairplot";
  state.generated = true;
  await waitFor('[data-analysis="pairplot"] [data-plot-rendered="true"]', 30000, "el pairplot");
  // scattergl draws on a WebGL canvas: it fails if the CSP breaks it. Without WebGL the
  // pairplot falls back to SVG, which is logged.
  if (hasWebGL()) {
    await waitFor('[data-analysis="pairplot"] canvas', 10000, "el canvas WebGL del pairplot");
  } else {
    void backend.logToBackend("warn", "WebGL no disponible: el pairplot usa SVG");
  }
  await waitForIdle(() => busy.active);
}

const CHECKS: Readonly<Record<string, () => Promise<void>>> = {
  datos_vistas: datosVistas,
  datos_correlacion: correlacion,
  datos_pairplot: pairplot,
};

export async function runSmokeCheck(name: string): Promise<void> {
  const check = CHECKS[name];
  if (!check) {
    throw new Error(`la interfaz no conoce esta comprobación: ${name}`);
  }
  await check();
}
