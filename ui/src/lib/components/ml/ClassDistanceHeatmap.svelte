<script lang="ts">
  /**
   * C19 (📏 Análisis de Distancias): heatmap viridis of the mean distances between the
   * classes of the train set (D56), the within-class means on the diagonal (D67), the
   * three metrics of Python's "Análisis de Separabilidad" (only informative) and the J2
   * verdict on the separation `min M[i,j] / max(W_i, W_j)`, for two classes too. Without
   * a separation (a class with one sample) the verdict falls back to Python's ratio.
   */
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice, { type Severity } from "../../../components/Notice.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import type { ClassDistancesView } from "../../bindings/ClassDistancesView";
  import type { SeparationVerdict } from "../../bindings/SeparationVerdict";
  import { knnText, md } from "../../content";
  import { fmt, fmtText } from "../../fmt";
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
  type Shown = Exclude<SeparationVerdict, "too_few_classes">;
  const severity: Readonly<Record<Shown, Severity>> = { separated: "success", moderate: "warning", close: "error" };
  /** J2 texts (with the separation) and Python's, on the ratio, for the fallback. */
  const bySeparation: Readonly<Record<Shown, string>> = { separated: "veredicto_separadas", moderate: "veredicto_moderadas", close: "veredicto_proximas" };
  const byRatio: Readonly<Record<Shown, string>> = { separated: "bien_separadas", moderate: "separacion_moderada", close: "muy_proximas" };
  const verdict = $derived.by((): [Severity, string] | null => {
    const v = view.verdict;
    if (v === "too_few_classes") {
      return null;
    }
    // +∞ (a class with zero spread) travels as `separation: null` + `separation_unbounded`.
    const text = view.separation_unbounded
      ? fmt(t(bySeparation[v]), { separacion: "∞" })
      : view.separation === null
        ? t(byRatio[v])
        : fmt(t(bySeparation[v]), { separacion: view.separation });
    return [severity[v], text];
  });
  const hasWithin = $derived(view.within.some((w) => w !== null));
  // With two classes the ratio is always 1: `binaria` explains it instead of `ratio_informativo`.
  const ratioNote = $derived(view.classes.length === 2 ? knnText("distancias_nuevo", "binaria") : t("ratio_informativo"));
</script>

<section class="distances" data-class-distances={view.verdict} data-n-classes={view.classes.length} data-separation={view.separation === null ? "" : view.separation.toFixed(4)}>
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
    {#if view.sampled}
      <p class="caption" data-distances-sampled>{fmtText(t("aviso_muestreo"), { max: view.sample_limit })}</p>
    {/if}
    <Markdown source={t("separabilidad_titulo")} />
    <MetricGrid
      items={[
        { label: t("metrica_min"), value: view.min === null ? "—" : fixed(view.min, 3) },
        { label: t("metrica_max"), value: view.max === null ? "—" : fixed(view.max, 3) },
        { label: t("metrica_ratio"), value: fixed(view.ratio, 2) },
      ]}
    />
    <p class="caption ratio-note">{ratioNote}</p>
    {#if verdict}
      <Markdown source={t("separacion_titulo")} />
      <Notice severity={verdict[0]} markdown={verdict[1]} />
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
