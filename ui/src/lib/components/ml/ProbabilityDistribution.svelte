<script lang="ts">
  /**
   * C5 (+ C6 in binary): distribution of the predicted probabilities
   * (`plot_threshold_analysis`). Binary: histograms of p(clase 1) by real class, the
   * uncertainty metrics and the threshold analysis. Multiclass: one histogram per class.
   * Shared by logística, Árboles and KNN.
   */
  import type { ProbabilityView } from "../../bindings/ProbabilityView";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import { compRaw, compText, md } from "../../content";
  import { toFixedPython } from "../../fmt";
  import { pickLevel } from "../../ml/levels";
  import { theme } from "../../theme.svelte";
  import { binaryProbabilityFigure, classProbabilityFigure } from "./figures";
  import MetricGrid from "./MetricGrid.svelte";
  import ThresholdAnalysis from "./ThresholdAnalysis.svelte";

  interface Props {
    view: ProbabilityView;
    idPrefix: string;
  }

  let { view, idPrefix }: Props = $props();

  const t = compText;
  const binary = $derived(view.kind === "binary" ? view : null);
  const multiclass = $derived(view.kind === "multiclass" ? view : null);
  const figure = $derived(binary ? binaryProbabilityFigure(binary.per_real_class, theme.chart) : null);
  const bothClasses = $derived(binary ? binary.per_real_class.every((g) => g.histogram.counts.length > 0) : false);
  const fraction = $derived(binary && binary.n > 0 ? binary.uncertain / binary.n : 0);
  const separation = $derived(pickLevel(compRaw("probabilidades", "separacion"), fraction, {}, { below: true }));
  const metrics = $derived(
    binary
      ? [
          { label: t("probabilidades", "metric_incierta"), value: `${binary.uncertain}/${binary.n}` },
          // `{:.1%}`
          { label: t("probabilidades", "metric_pct"), value: `${toFixedPython(fraction * 100, 1)}%` },
          { label: t("probabilidades", "metric_confiables"), value: `${binary.confident}/${binary.n}` },
        ]
      : [],
  );
</script>

<section class="probabilities" data-probabilities={view.kind}>
  {#if binary && figure}
    <Expander id="{idPrefix}-probs-binaria" label={t("probabilidades", "expander_binaria")}>
      <Markdown source={md("componentes/probabilidades-binaria.md")} />
    </Expander>
    <div class="chart">
      <PlotlyChart
        data={figure.data}
        layout={figure.layout}
        height={460}
        downloadName="distribucion_probabilidades.png"
        label={t("probabilidades", "title_binaria")}
      />
    </div>
    {#if bothClasses}
      <MetricGrid items={metrics} />
      {#if separation}
        <Notice severity={separation.severity} markdown={separation.markdown} />
      {/if}
    {/if}
    <ThresholdAnalysis
      thresholds={binary.thresholds}
      best={binary.best_f1_threshold}
      shown={binary.shown_thresholds}
      {idPrefix}
    />
  {:else if multiclass}
    <Notice severity="info" markdown={t("probabilidades", "nota_multiclase")} />
    <Expander id="{idPrefix}-probs-multiclase" label={t("probabilidades", "expander_multiclase")}>
      <Markdown source={md("componentes/probabilidades-multiclase.md")} />
    </Expander>
    <div class="grid" style:--columns={Math.min(3, multiclass.per_class.length)}>
      {#each multiclass.per_class as group (group.class_index)}
        {@const f = classProbabilityFigure(group, theme.chart)}
        <PlotlyChart data={f.data} layout={f.layout} height={320} label={group.name} />
      {/each}
    </div>
  {/if}
</section>

<style>
  .chart {
    max-width: 920px;
    margin: 0 auto;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(var(--columns), minmax(0, 1fr));
    gap: 14px;
  }

  .grid :global(.chart) {
    margin: 0;
  }
</style>
