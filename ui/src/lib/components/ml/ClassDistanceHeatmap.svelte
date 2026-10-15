<script lang="ts">
  /**
   * C19 (📏 Análisis de Distancias): heatmap viridis of the mean distances between the
   * classes of the train set (D56), the within-class means on the diagonal (D67), the
   * three metrics of "Análisis de Separabilidad" and the verdict (D67 with two classes).
   */
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice, { type Severity } from "../../../components/Notice.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import type { ClassDistancesView } from "../../bindings/ClassDistancesView";
  import type { SeparationVerdict } from "../../bindings/SeparationVerdict";
  import { knnText, md } from "../../content";
  import { fmt } from "../../fmt";
  import { fixed } from "../../format";
  import { theme } from "../../theme.svelte";
  import { distanceHeatmapFigure, distanceHeatmapHeight } from "./knnFigures";
  import MetricGrid from "./MetricGrid.svelte";

  interface Props {
    view: ClassDistancesView;
    idPrefix: string;
  }

  let { view, idPrefix }: Props = $props();
  const t = (key: string): string => knnText("distancias", key);

  const figure = $derived(distanceHeatmapFigure(view, theme.chart));
  const verdicts: Readonly<Record<Exclude<SeparationVerdict, "too_few_classes">, [Severity, () => string]>> = {
    separated: ["success", () => t("bien_separadas")],
    moderate: ["warning", () => t("separacion_moderada")],
    close: ["error", () => t("muy_proximas")],
    binary: ["info", () => knnText("distancias_nuevo", "binaria")],
  };
  const verdict = $derived(view.verdict === "too_few_classes" ? null : verdicts[view.verdict]);
  const hasWithin = $derived(view.within.some((w) => w !== null));
</script>

<section class="distances" data-class-distances={view.verdict}>
  <Markdown source={t("titulo")} />
  <Expander id="{idPrefix}-distancias-interpretar" label={t("expander")}>
    <Markdown source={fmt(md("knn/distancias-interpretar.md"), { metrica: view.metric })} />
  </Expander>
  {#if view.verdict === "too_few_classes"}
    <Notice severity="info" markdown={t("pocas_clases")} />
  {:else}
    <div class="chart">
      <PlotlyChart
        data={figure.data}
        layout={figure.layout}
        height={distanceHeatmapHeight(view.classes.length)}
        downloadName="distancias_clases.png"
        label="Distancias promedio entre clases"
      />
    </div>
    <p class="caption">
      {fmt(t("nota_datos"), { n: view.n_samples })}
      {#if hasWithin}{knnText("distancias_nuevo", "diagonal_nota")}{/if}
    </p>
    <Markdown source={t("separabilidad_titulo")} />
    <MetricGrid
      items={[
        { label: t("metrica_min"), value: view.min === null ? "—" : fixed(view.min, 3) },
        { label: t("metrica_max"), value: view.max === null ? "—" : fixed(view.max, 3) },
        { label: t("metrica_ratio"), value: fixed(view.ratio, 2) },
      ]}
    />
    {#if verdict}
      <Notice severity={verdict[0]} markdown={verdict[1]()} />
    {/if}
  {/if}
</section>

<style>
  .chart {
    max-width: 860px;
    margin: 0 auto;
  }

  .caption {
    margin: -6px 2px 8px;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
  }
</style>
