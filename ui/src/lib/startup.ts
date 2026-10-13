import * as backend from "./backend";
import { nextPaint, pageHeading } from "./dom";
import { router } from "./router.svelte";
import { runSmokeTest } from "./smoke";

function describe(reason: unknown): string {
  if (reason instanceof Error) {
    return `${reason.name}: ${reason.message}${reason.stack ? `\n${reason.stack}` : ""}`;
  }
  return String(reason);
}

/**
 * Runs once the shell is mounted: reports startup to the backend and, when launched
 * with `--smoke-test`, walks through every page. Outside Tauri (plain `vite dev` in a
 * browser) it does nothing.
 */
export async function startup(): Promise<void> {
  if (!backend.isTauri()) {
    return;
  }
  const mode = await backend.launchMode();
  const smoke = mode.kind === "smoke_test";

  const onError = (reason: unknown): void => {
    const message = `Error no capturado en la interfaz: ${describe(reason)}`;
    void (smoke ? backend.smokeFailed(message) : backend.logToBackend("error", message));
  };
  window.addEventListener("error", (event) => onError(event.error ?? event.message));
  window.addEventListener("unhandledrejection", (event) => onError(event.reason));

  await document.fonts.ready;
  await nextPaint();
  void backend.logToBackend(
    "info",
    `viewport ${window.innerWidth}x${window.innerHeight} @${window.devicePixelRatio}x, ${navigator.userAgent}`,
  );
  await backend.frontendReady(router.current);

  if (mode.kind === "smoke_test") {
    await runSmokeTest(mode.routes, {
      panicProbe: backend.smokePanicProbe,
      navigate: (id) => router.go(id),
      waitForRender: () => nextPaint(),
      renderedTitle: pageHeading,
      report: backend.smokeRouteRendered,
      fail: backend.smokeFailed,
    });
  }
}
