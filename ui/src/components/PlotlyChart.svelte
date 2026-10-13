<script lang="ts">
  import type { Config, Data, Layout } from "plotly.js";
  import { onDestroy } from "svelte";
  import { isTauri, logToBackend } from "../lib/backend";
  import { dataUrlToBytes, PNG_FILTER, saveBytes } from "../lib/download";
  import { loadPlotly, type PlotlyModule } from "../lib/plotly";
  import { status } from "../lib/status.svelte";
  import Spinner from "./Spinner.svelte";

  interface Props {
    data: Data[];
    layout: Partial<Layout>;
    config?: Partial<Config>;
    /** Height in px (the width follows the container). */
    height?: number;
    /** File name of the "save image" button; no button if omitted. */
    downloadName?: string;
    downloadLabel?: string;
    /** Accessible description of the chart. */
    label?: string;
  }

  let {
    data,
    layout,
    config = {},
    height = 460,
    downloadName,
    downloadLabel = "Guardar imagen…",
    label,
  }: Props = $props();

  let container: HTMLDivElement | undefined = $state();
  let plotly: PlotlyModule | null = null;
  let rendered = $state(false);
  let failed = $state<string | null>(null);
  let observer: ResizeObserver | undefined;
  let frame = 0;

  const fullConfig = $derived<Partial<Config>>({
    responsive: false,
    displaylogo: false,
    displayModeBar: "hover",
    // Saving goes through the native dialog (button below), not a browser download.
    modeBarButtonsToRemove: ["toImage", "sendDataToCloud", "lasso2d", "select2d"],
    scrollZoom: false,
    locale: "es",
    ...config,
  });

  $effect(() => {
    const target = container;
    const figureLayout = { ...layout, height, autosize: true };
    const figureData = data;
    const figureConfig = fullConfig;
    if (!target) {
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        plotly ??= await loadPlotly();
        if (cancelled) {
          return;
        }
        await plotly.react(target, figureData, figureLayout, figureConfig);
        rendered = true;
        failed = null;
        if (!observer) {
          observer = new ResizeObserver(() => {
            cancelAnimationFrame(frame);
            frame = requestAnimationFrame(() => {
              if (plotly && container) {
                void plotly.Plots.resize(container);
              }
            });
          });
          observer.observe(target);
        }
      } catch (error) {
        failed = error instanceof Error ? error.message : String(error);
        if (isTauri()) {
          void logToBackend("error", `Plotly no pudo dibujar «${label ?? "gráfico"}»: ${failed}`);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  });

  onDestroy(() => {
    observer?.disconnect();
    cancelAnimationFrame(frame);
    if (plotly && container) {
      plotly.purge(container);
    }
  });

  async function download(): Promise<void> {
    if (!plotly || !container || !downloadName) {
      return;
    }
    const url = await plotly.toImage(container, {
      format: "png",
      width: container.clientWidth,
      height,
      scale: 2,
    });
    const saved = await saveBytes(downloadName, dataUrlToBytes(url), PNG_FILTER);
    if (saved) {
      status.flash(`Guardado: ${saved}`);
    }
  }
</script>

<figure class="chart" aria-label={label}>
  <div class="wrap" style:height="{height}px">
    <div class="plot" bind:this={container} data-plot-rendered={failed ? "error" : rendered}></div>
    {#if !rendered && !failed}
      <div class="loading"><Spinner label="Dibujando el gráfico…" /></div>
    {/if}
  </div>
  {#if failed}
    <p class="error">No se pudo dibujar el gráfico: {failed}</p>
  {/if}
  {#if downloadName}
    <button type="button" class="btn block save" disabled={!rendered} onclick={download}>{downloadLabel}</button>
  {/if}
</figure>

<style>
  .chart {
    margin: 12px 0 18px;
  }

  .wrap {
    position: relative;
    width: 100%;
  }

  .plot {
    width: 100%;
    height: 100%;
  }

  .loading {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
  }

  .save {
    margin-top: 10px;
  }

  .error {
    color: var(--color-text-muted);
  }
</style>
