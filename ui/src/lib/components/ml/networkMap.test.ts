import { cleanup, render } from "@testing-library/svelte";
import { tick } from "svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import NetworkMap from "./NetworkMap.svelte";
import { DEFAULT_MAP_OPTIONS } from "./networkLayout";

/**
 * Regression test of the blank 🗺️ map (2026-10-20): Svelte's `width={…}` attribute was applied
 * after `draw` had painted, and any assignment to `canvas.width` clears the bitmap. The mock
 * context records every bitmap reset and every circle, so the canvas is blank when the last
 * reset comes after the last circle.
 */
const events: string[] = [];
let boxWidth = 1100;
let observers: (() => void)[] = [];
const widthDescriptor = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, "width");
const clientWidthDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth");

beforeEach(() => {
  events.length = 0;
  boxWidth = 1100;
  observers = [];
  vi.useFakeTimers();
  Object.defineProperty(HTMLCanvasElement.prototype, "width", {
    configurable: true,
    get(this: HTMLCanvasElement) {
      return widthDescriptor?.get?.call(this) as number;
    },
    set(this: HTMLCanvasElement, v: number) {
      events.push(`reset ${v}`);
      widthDescriptor?.set?.call(this, v);
    },
  });
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get: () => boxWidth,
  });
  const ctx = new Proxy({}, { get: (_t, p) => (p === "arc" ? () => events.push("arc") : () => undefined), set: () => true });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctx as never);
  vi.stubGlobal("devicePixelRatio", 2);
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(private readonly cb: () => void) {}
      observe(): void {
        observers.push(() => this.cb());
      }
      disconnect(): void {}
    },
  );
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  if (widthDescriptor) {
    Object.defineProperty(HTMLCanvasElement.prototype, "width", widthDescriptor);
  }
  if (clientWidthDescriptor) {
    Object.defineProperty(HTMLElement.prototype, "clientWidth", clientWidthDescriptor);
  }
});

function lastResetIndex(): number {
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i]?.startsWith("reset")) {
      return i;
    }
  }
  return -1;
}

function lastReset(): string | undefined {
  return events[lastResetIndex()];
}

/** Circles painted after the last reset of the bitmap (17 neurons when the map is visible). */
function painted(): number {
  return events.slice(lastResetIndex() + 1).filter((e) => e === "arc").length;
}

describe("NetworkMap", () => {
  const props = {
    map: { layers: [4, 10, 3], neurons: [] } as never,
    hidden: [{ dead: [1], constant: [] }] as never,
    options: { ...DEFAULT_MAP_OPTIONS },
  };

  it("paints after mount, at the measured width and HiDPI", async () => {
    const { container } = render(NetworkMap, props);
    await tick();
    expect(painted()).toBe(17);
    expect(lastReset()).toBe("reset 2200");
    expect(container.querySelector("canvas")?.style.width).toBe("1100px");
  });

  it("paints again after the box is resized or becomes visible", async () => {
    boxWidth = 0;
    render(NetworkMap, props);
    await tick();
    boxWidth = 800;
    for (const fire of observers) {
      fire();
    }
    vi.advanceTimersByTime(100);
    await tick();
    expect(lastReset()).toBe("reset 1600");
    expect(painted()).toBe(17);
  });
});
