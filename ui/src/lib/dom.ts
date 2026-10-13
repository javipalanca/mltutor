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
  const heading = document.querySelector(`[data-route-page="${id}"] h1`);
  return heading?.textContent?.trim() ?? null;
}
