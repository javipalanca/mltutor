import * as backend from "./backend";
import { busy } from "./busy.svelte";
import { nextPaint, pageHeading, waitForIdle } from "./dom";
import { router } from "./router.svelte";
import { runScenario } from "./scenarios";
import { session } from "./session.svelte";
import { status } from "./status.svelte";
import { runSmokeTest } from "./smoke";
import { runSmokeCheck } from "./smokeChecks";

function describe(reason: unknown): string {
  if (reason instanceof Error) {
    const stack = reason.stack?.split("\n").slice(0, 12).join("\n");
    return `${reason.name}: ${reason.message}${stack ? `\n${stack}` : ""}`;
  }
  return String(reason);
}

async function loadDatasets(): Promise<void> {
  const [datasets, selected] = await Promise.all([backend.listDatasets(), backend.getSelectedDataset()]);
  backend.continueSeqAfter(selected.last_seq);
  session.datasets = datasets;
  session.selected = selected.dataset;
}

/** Height of the whole page for `--full-page` captures (CSS px). */
function fullPageHeight(): number {
  const content = document.querySelector<HTMLElement>("[data-content]");
  const status = document.querySelector<HTMLElement>(".status");
  const sidebar = document.querySelector<HTMLElement>(".sidebar");
  const page = content?.scrollHeight ?? window.innerHeight;
  return Math.max(page, sidebar?.scrollHeight ?? 0) + (status?.offsetHeight ?? 0);
}

/**
 * Runs once the shell is mounted: loads the dataset list, reports startup to the backend
 * and, when launched with `--smoke-test` or `--screenshot`, drives the UI. Outside Tauri
 * (plain `vite dev` in a browser) it does nothing.
 */
export async function startup(): Promise<void> {
  if (!backend.isTauri()) {
    return;
  }
  const mode = await backend.launchMode();
  const smoke = mode.kind === "smoke_test";
  if (mode.kind !== "normal") {
    // Hidden or offscreen windows (CI, full-page captures) do not advance CSS transitions:
    // a capture could show a button half-way between two states.
    document.documentElement.dataset["automation"] = "true";
  }

  const onError = (reason: unknown): void => {
    const message = `Error no capturado en la interfaz: ${describe(reason)}`;
    void (smoke ? backend.smokeFailed(message) : backend.logToBackend("error", message));
  };
  window.addEventListener("error", (event) => onError(event.error ?? event.message));
  window.addEventListener("unhandledrejection", (event) => onError(event.reason));
  // A CSP violation means something (Plotly, KaTeX…) does not work in the packaged app.
  document.addEventListener("securitypolicyviolation", (event) => {
    onError(`CSP bloquea ${event.violatedDirective}: ${event.blockedURI || "(en línea)"}`);
  });

  void backend.onTaskProgress((progress) => status.onProgress(progress.message, progress.percent));

  try {
    await loadDatasets();
  } catch (error) {
    onError(error);
  }

  await document.fonts.ready;
  await nextPaint();
  void backend.logToBackend(
    "info",
    `viewport ${window.innerWidth}x${window.innerHeight} @${window.devicePixelRatio}x, ${navigator.userAgent}`,
  );
  await backend.frontendReady(router.current);

  if (mode.kind === "smoke_test") {
    await runSmokeTest(mode.routes, mode.checks, {
      panicProbe: backend.smokePanicProbe,
      navigate: (id) => router.go(id),
      waitForRender: () => nextPaint(),
      renderedTitle: pageHeading,
      report: backend.smokeRouteRendered,
      runCheck: runSmokeCheck,
      reportCheck: backend.smokeCheckPassed,
      fail: backend.smokeFailed,
    });
  } else if (mode.kind === "screenshot") {
    try {
      if (mode.scenario) {
        await runScenario(mode.scenario);
      }
      await waitForIdle(() => busy.active, 60000);
      // Plotly animations and fonts settle.
      await new Promise((resolve) => setTimeout(resolve, 400));
      await waitForIdle(() => busy.active, 60000);
      await backend.screenshotReady(mode.full_page ? fullPageHeight() : null);
    } catch (error) {
      await backend.screenshotReady(null, describe(error));
    }
  }
}
