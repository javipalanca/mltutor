<script lang="ts">
  /**
   * C8 (`viz/decision_boundary.py`): the decision boundary of a classifier or the
   * prediction surface of a regressor, with a model refitted in Rust on the two chosen
   * features of the train set. Shared by Regresión, Árboles and KNN.
   */
  import type { DecisionGrid } from "../../bindings/DecisionGrid";
  import type { GridNotice } from "../../bindings/GridNotice";
  import Notice from "../../../components/Notice.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import { hasWebGL } from "../../charts";
  import { compRaw, compText, optionalText } from "../../content";
  import { fmt } from "../../fmt";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import { theme } from "../../theme.svelte";
  import FeaturePairSelector from "./FeaturePairSelector.svelte";
  import { boundaryFigure, surfaceFigure } from "./figures";
  import ModelCode from "./ModelCode.svelte";

  interface Props {
    modelId: number;
    features: readonly string[];
    pair: [number, number];
    onpair: (pair: [number, number]) => void;
    /** `boundary` (classification) or `surface` (regression). */
    mode: "boundary" | "surface";
    idPrefix: string;
    onstale: () => void;
    /**
     * 🧠 (D78): the network is not refitted on two features; this caption replaces the
     * warning about the 2D model and the boundary code is not offered.
     */
    note?: string | null;
    /** 🧠: "✅ Superficie generada exitosamente" under the chart, as Python. */
    success?: string | null;
    /** 🧠: surface and points with their own colour scales (titles of the two bars). */
    ownScales?: { surface: string; points: string } | null;
  }

  let { modelId, features, pair, onpair, mode, idPrefix, onstale, note = null, success = null, ownScales = null }: Props = $props();

  const t = compText;
  const grid = new ViewLoader<DecisionGrid>();
  const same = $derived(pair[0] === pair[1]);
  const key = $derived(`${modelId}|${pair[0]}|${pair[1]}`);

  $effect(() => {
    if (same) {
      return;
    }
    const [x, y] = pair;
    const id = modelId;
    // Debounced: changing one select and then the other only computes once.
    return grid.load(key, () => backend.decisionGrid(id, x, y), onstale, 120);
  });

  // While another pair loads, the previous chart stays (dimmed) instead of a spinner.
  const data = $derived(grid.for(key) ?? (grid.loading ? grid.data : null));
  const figure = $derived(
    data ? (mode === "boundary" ? boundaryFigure(data, theme.chart, hasWebGL()) : surfaceFigure(data, theme.chart, hasWebGL(), ownScales)) : null,
  );

  function noticeText(notice: GridNotice): string {
    switch (notice.kind) {
      case "created":
        return fmt(t("frontera", "malla"), { filas: notice.rows, columnas: notice.cols, n: notice.rows * notice.cols });
      case "too_large":
        return fmt(t("frontera", "malla_grande"), { n: notice.points });
      case "resized":
        return fmt(t("frontera", "malla_nueva"), { filas: notice.rows, columnas: notice.cols, n: notice.rows * notice.cols });
    }
  }

  /** The surface shows the three lines in a single notice ("- …\n - …"), like Python. */
  const surfaceNotice = $derived(
    data ? data.notices.map((n, i) => (i === 0 ? `- ${noticeText(n)}` : ` - ${noticeText(n)}`)).join("\n") : "",
  );
</script>

<section class="grid-view" data-grid-view={mode}>
  <FeaturePairSelector {features} {pair} onchange={onpair} {idPrefix} />
  {#if same}
    <Notice severity="info" markdown={optionalText(compRaw, "frontera", "misma_caracteristica", "Elige dos características distintas para dibujar el gráfico.")} />
  {:else if grid.error}
    <Notice severity="error" markdown={fmt(t("frontera", "error_frontera"), { error: grid.error })} />
  {:else if data && figure}
    {#if mode === "boundary"}
      {#each data.notices as notice, i (i)}
        <Notice severity={notice.kind === "too_large" ? "warning" : "info"} markdown={noticeText(notice)} />
      {/each}
    {:else}
      <Notice severity="info" markdown={surfaceNotice} />
    {/if}
    <div class="chart" class:updating={grid.loading}>
      <PlotlyChart
        data={figure.data}
        layout={figure.layout}
        height={640}
        {...mode === "boundary" ? { downloadName: "frontera_decision.png", downloadLabel: t("frontera", "descarga_frontera") } : { downloadName: "superficie_prediccion.png" }}
        label={mode === "boundary" ? "Frontera de decisión" : "Superficie de predicción"}
      />
    </div>
    {#if success && !grid.loading}
      <Notice severity="success" markdown={success} />
    {/if}
    {#if note}
      <p class="note" data-grid-note>{note}</p>
    {:else if data.show_2d_warning}
      <Notice severity="warning" markdown={t("frontera", "aviso_2d")} />
    {/if}
    {#if mode === "boundary" && !note}
      <ModelCode
        {modelId}
        template="decision_boundary"
        title={t("frontera", "codigo_frontera")}
        downloadName="frontera_decision.py"
        id="{idPrefix}-codigo-frontera"
        {pair}
        {onstale}
      />
    {/if}
  {:else}
    <Spinner label={t("frontera", mode === "boundary" ? "spinner_frontera" : "spinner_superficie")} />
  {/if}
</section>

<style>
  .chart {
    max-width: 960px;
    margin: 0 auto;
    transition: opacity 120ms ease;
  }

  .chart.updating {
    opacity: 0.55;
  }

  .note {
    max-width: 960px;
    margin: 4px auto 10px;
    color: var(--color-text-muted);
    font-size: 13px;
  }
</style>
