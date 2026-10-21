import { tick } from "svelte";

/**
 * Waits for Svelte to flush pending updates and for the browser to paint.
 * `requestAnimationFrame` is paused for occluded windows (e.g. CI), so it is raced
 * against a short timeout.
 */
export async function nextPaint(timeoutMs = 100): Promise<void> {
  await tick();
  await new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, timeoutMs);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        clearTimeout(timer);
        resolve();
      });
    });
  });
}

export function pageHeading(id: string): string | null {
  const heading = document.querySelector(`[data-route-page="${id}"] [data-page-title]`);
  return heading?.textContent?.trim() ?? null;
}

/** Resolves once `selector` matches (polling; works in hidden windows), or rejects after `timeoutMs`. */
export async function waitFor(selector: string, timeoutMs = 20000, what = selector): Promise<Element> {
  const started = performance.now();
  for (;;) {
    const found = document.querySelector(selector);
    if (found) {
      return found;
    }
    if (performance.now() - started > timeoutMs) {
      throw new Error(`Tiempo agotado esperando ${what}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

/** Waits until no chart is still drawing and no backend call is pending. */
export async function waitForIdle(isBusy: () => boolean, timeoutMs = 20000): Promise<void> {
  const started = performance.now();
  for (;;) {
    await nextPaint();
    const drawing = document.querySelector('[data-plot-rendered="false"]');
    if (!drawing && !isBusy()) {
      return;
    }
    if (performance.now() - started > timeoutMs) {
      throw new Error("Tiempo agotado esperando a que terminen los gráficos");
    }
    await new Promise((resolve) => setTimeout(resolve, 60));
  }
}
