/**
 * Lazy loader of Plotly's *strict* bundle (`plotly.js-strict-dist-min`): its WebGL
 * traces (scattergl) use precompiled shaders instead of `new Function`, so the CSP does
 * not need `'unsafe-eval'`. Loaded on first use to keep startup fast.
 */
import type * as PlotlyTypes from "plotly.js";

export type PlotlyModule = typeof PlotlyTypes;

let loading: Promise<PlotlyModule> | null = null;

export function loadPlotly(): Promise<PlotlyModule> {
  loading ??= import("plotly.js-strict-dist-min").then((module) => {
    const candidate = (module as unknown as { default?: PlotlyModule }).default;
    return candidate ?? (module as unknown as PlotlyModule);
  });
  return loading;
}
