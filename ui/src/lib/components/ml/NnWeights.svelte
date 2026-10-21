<script lang="ts">
  /**
   * C25: 🧠 Análisis de Pesos (`viz/nn.py::show_weights_analysis_tab`): one caption per
   * dense layer, "📋 Resumen por Capa", the global metrics and their verdict, two histograms
   * per layer (weights `#1f77b4`, biases `#ff7f0e`, grey line at 0, green dashed mean) with
   * their captions and "🩺 Evaluación General". D77: the layers are named after the
   * architecture everywhere. K2: the bins come from Rust; changing them asks again.
   */
  import Checkbox from "../../../components/Checkbox.svelte";
  import DataTable from "../../../components/DataTable.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import Slider from "../../../components/Slider.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { Cell } from "../../bindings/Cell";
  import type { WeightsView } from "../../bindings/WeightsView";
  import { md, nnList, nnText } from "../../content";
  import { escapeMarkdown, escapePlotly } from "../../escape";
  import { fixed } from "../../format";
  import { fmtText } from "../../fmt";
  import { BIAS_BINS, nnViz, WEIGHT_BINS } from "../../lessons/redes/visualization.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import { theme } from "../../theme.svelte";
  import MetricGrid from "./MetricGrid.svelte";
  import { binnedHistogramFigure } from "./nnFigures";

  interface Props {
    modelId: number;
    onstale: () => void;
  }

  let { modelId, onstale }: Props = $props();
  const t = (key: string): string => nnText("pesos", key);
  const loader = new ViewLoader<WeightsView>();
  const key = $derived(`${modelId}|${nnViz.binsWeights}|${nnViz.binsBiases}`);

  $effect(() => {
    const req = { model_id: modelId, bins_weights: nnViz.binsWeights, bins_biases: nnViz.binsBiases };
    // Debounced: dragging a slider asks once it stops.
    return loader.load(key, () => backend.nnWeights(req), onstale, 80);
  });

  // While new bins load, the previous histograms stay (dimmed).
  const view = $derived(loader.for(key) ?? (loader.loading ? loader.data : null));

  const shape = (a: number, b: number): string => `(${a}, ${b})`;
  const num = (v: number): Cell => ({ kind: "number", v });
  const text = (v: string): Cell => ({ kind: "text", v });

  const rows = $derived(
    (view?.layers ?? []).map((l) => [
      text(l.name),
      text(shape(l.inputs, l.units)),
      num(l.mean_w),
      num(l.std_w),
      num(l.near_zero_pct),
      num(l.mean_b),
      num(l.std_b),
      num(l.n_params),
    ]),
  );

  const metrics = $derived(
    view
      ? [
          { label: t("media_global"), value: fixed(view.global_mean, 4) },
          { label: t("std_global"), value: fixed(view.global_std, 4) },
          { label: t("casi_cero_global"), value: `${fixed(view.global_near_zero_pct, 1)}%` },
          { label: t("capas_analizadas"), value: view.layers.length },
        ]
      : [],
  );

  const verdict = $derived.by(() => {
    switch (view?.verdict) {
      case "too_small":
        return { severity: "warning" as const, text: t("pesos_pequenos") };
      case "too_large":
        return { severity: "warning" as const, text: t("pesos_varianza_alta") };
      case "healthy":
        return { severity: "success" as const, text: t("pesos_sanos") };
      default:
        return null;
    }
  });

  const INSIGHTS = { many_near_zero: "insight_casi_cero", low_variance: "insight_varianza_baja", high_variance: "insight_varianza_alta", balanced: "insight_ok" } as const;

  const charts = $derived(
    (view?.layers ?? []).map((l) => ({
      layer: l,
      weights: binnedHistogramFigure(
        l.weights_hist,
        {
          title: escapePlotly(t("hist_pesos_titulo").replace("{capa}", l.name)),
          xLabel: t("eje_peso"),
          yLabel: t("eje_frecuencia"),
          color: "#1f77b4",
          mean: l.mean_w,
          meanLabel: t("anotacion_media"),
          log: nnViz.logScale,
        },
        theme.chart,
      ),
      biases: binnedHistogramFigure(
        l.biases_hist,
        {
          title: escapePlotly(t("hist_sesgos_titulo").replace("{capa}", l.name)),
          xLabel: t("eje_sesgo"),
          yLabel: t("eje_frecuencia"),
          color: "#ff7f0e",
          mean: l.mean_b,
          meanLabel: t("anotacion_media"),
          log: nnViz.logScale,
        },
        theme.chart,
      ),
      weightsCaption: fmtText(t("pie_pesos"), { media: l.mean_w, std: l.std_w, casi_cero: l.near_zero_pct_raw, min: l.min_w, max: l.max_w }),
      biasesCaption: fmtText(t("pie_sesgos"), { media: l.mean_b, std: l.std_b, min: l.min_b, max: l.max_b }),
    })),
  );
</script>

<section class="weights" data-nn-weights={view?.layers.length ?? 0}>
  <h3>{t("titulo")}</h3>
  <Expander id="nn-pesos-interpretar" label={t("interpretar_expander")}>
    <Markdown source={md("redes/pesos-interpretar.md")} />
  </Expander>
  <Expander id="nn-pesos-opciones" label={t("opciones_expander")}>
    <div class="options">
      <Slider id="nn-bins-pesos" label={t("bins_pesos")} min={WEIGHT_BINS.min} max={WEIGHT_BINS.max} value={nnViz.binsWeights} onchange={(v) => (nnViz.binsWeights = Math.round(v))} />
      <Slider id="nn-bins-sesgos" label={t("bins_sesgos")} min={BIAS_BINS.min} max={BIAS_BINS.max} value={nnViz.binsBiases} onchange={(v) => (nnViz.binsBiases = Math.round(v))} />
      <Checkbox id="nn-pesos-log" label={t("escala_log")} help={t("escala_log_help")} checked={nnViz.logScale} onchange={(v) => (nnViz.logScale = v)} />
    </div>
  </Expander>

  {#if loader.error}
    <Notice severity="error" markdown={escapeMarkdown(loader.error)} />
  {:else if !view}
    <Spinner label="Analizando los pesos…" />
  {:else}
    <div class="loaded" class:updating={loader.loading}>
      {#each view.layers as l (l.name)}
        <p class="caption">{fmtText(t("capa_cargada"), { capa: l.name, forma_pesos: shape(l.inputs, l.units), forma_sesgos: `(${l.units},)` })}</p>
      {/each}
      <Markdown source={t("resumen_titulo")} />
      <DataTable columns={nnList("pesos", "resumen_columnas")} {rows} hideIndex numericColumns={[2, 3, 4, 5, 6, 7]} label={t("resumen_titulo").replace(/^#+\s*/, "")} />
      <MetricGrid items={metrics} />
      {#if verdict}
        <Notice severity={verdict.severity} markdown={verdict.text} />
      {/if}

      <Markdown source={t("distribucion_titulo")} />
      {#each charts as c (c.layer.name)}
        <div class="layer" data-weights-layer={c.layer.keras_name}>
          <Markdown source={fmtText(t("capa_titulo"), { capa: c.layer.name })} />
          <div class="two">
            <div>
              <PlotlyChart data={c.weights.data} layout={c.weights.layout} height={320} label={t("hist_pesos_titulo").replace("{capa}", c.layer.name)} />
              <p class="caption">{c.weightsCaption}</p>
            </div>
            <div>
              <PlotlyChart data={c.biases.data} layout={c.biases.layout} height={320} label={t("hist_sesgos_titulo").replace("{capa}", c.layer.name)} />
              <p class="caption">{c.biasesCaption}</p>
            </div>
          </div>
        </div>
      {/each}

      <Markdown source={t("evaluacion_titulo")} />
      {#each view.insights as insight (insight)}
        <p class="insight">• {t(INSIGHTS[insight])}</p>
      {/each}
    </div>
  {/if}
</section>

<style>
  h3 {
    margin: 18px 0 10px;
    color: var(--color-heading);
    font-size: 20px;
  }

  .options {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px 20px;
    align-items: end;
  }

  .caption {
    margin: 2px 0;
    color: var(--color-text-muted);
    font-size: 13px;
  }

  .two {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
  }

  .loaded {
    transition: opacity 120ms ease;
  }

  .loaded.updating {
    opacity: 0.55;
  }

  .insight {
    margin: 6px 0;
  }

  @container page (max-width: 520px) {
    .two,
    .options {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
