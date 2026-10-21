<script lang="ts">
  /** C27: the history chart (live in ⚙️ Entrenamiento; 📈 and 📊 Historial reuse it). */
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import type { EpochLog } from "../../bindings/EpochLog";
  import type { NnMetric } from "../../bindings/NnMetric";
  import { nnRaw, optionalText } from "../../content";
  import { theme } from "../../theme.svelte";
  import { historyFigure } from "./nnFigures";

  interface Props {
    logs: readonly EpochLog[];
    metric: NnMetric;
    bestEpoch?: number | null;
    lrDrops?: readonly number[];
    /** Live: the x axis spans every epoch of the training from the start. */
    maxEpochs?: number | null;
    downloadName?: string;
  }

  let { logs, metric, bestEpoch = null, lrDrops = [], maxEpochs = null, downloadName }: Props = $props();
  const figure = $derived(historyFigure(logs, metric, { bestEpoch, lrDrops, maxEpochs }, theme.chart));
  const marks = $derived(
    [
      bestEpoch ? optionalText(nnRaw, "entrenamiento", "grafica_mejor_epoca", "Línea verde discontinua: la mejor época, cuyos pesos restauró Early Stopping ({epoca}).").replace("{epoca}", String(bestEpoch)) : null,
      lrDrops.length > 0
        ? optionalText(nnRaw, "entrenamiento", "grafica_reduccion_lr", "Líneas moradas punteadas: el learning rate se redujo tras las épocas {epocas}.").replace("{epocas}", lrDrops.join(", "))
        : null,
    ].filter((m): m is string => m !== null),
  );
</script>

<div class="history" data-history-chart data-epochs={logs.length}>
  <PlotlyChart data={figure.data} layout={figure.layout} height={400} label="Historial de entrenamiento" {...downloadName ? { downloadName } : {}} />
  {#each marks as mark (mark)}
    <p class="caption">{mark}</p>
  {/each}
</div>

<style>
  .caption {
    margin: -8px 2px 4px;
    color: var(--color-text-muted);
    font-size: 13px;
  }
</style>
