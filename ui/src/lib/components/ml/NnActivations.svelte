<script lang="ts">
  /**
   * C26: 📉 Análisis de Activaciones (`viz/nn.py::show_layer_activations_tab`): the 10
   * controls of "⚙️ Parámetros de Análisis", "📋 Resumen por Capa Oculta" with the global
   * verdict on dead neurons, one expander per hidden layer (4 metrics, the histogram of its
   * activations and the means per neuron) and the map of dead (💀) and constant (😴) neurons.
   * The statistics are computed in Rust on the first test rows (K3); the drawing options only
   * redraw the map.
   */
  import Checkbox from "../../../components/Checkbox.svelte";
  import DataTable from "../../../components/DataTable.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Metric from "../../../components/Metric.svelte";
  import Notice from "../../../components/Notice.svelte";
  import NumberInput from "../../../components/NumberInput.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import Slider from "../../../components/Slider.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { ActivationsView } from "../../bindings/ActivationsView";
  import type { Cell } from "../../bindings/Cell";
  import { md, nnList, nnRaw, nnText, optionalText } from "../../content";
  import { escapePlotly } from "../../escape";
  import { fmt, fmtText } from "../../fmt";
  import { fixed } from "../../format";
  import { nnViz, SAMPLES } from "../../lessons/redes/visualization.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import { theme } from "../../theme.svelte";
  import NetworkMap from "./NetworkMap.svelte";
  import { binnedHistogramFigure, neuronMeansFigure } from "./nnFigures";

  interface Props {
    modelId: number;
    onstale: () => void;
  }

  let { modelId, onstale }: Props = $props();
  const t = (key: string): string => nnText("activaciones", key);
  const loader = new ViewLoader<ActivationsView>();
  const key = $derived(`${modelId}|${nnViz.sampleSize}|${nnViz.deadThreshold}|${nnViz.constantStd}`);

  $effect(() => {
    const req = { model_id: modelId, sample_size: nnViz.sampleSize, dead_threshold: nnViz.deadThreshold, constant_std: nnViz.constantStd };
    // Debounced and cancellable: a newer request cancels the one in flight.
    return loader.load(key, () => backend.nnActivations(req), onstale, 100);
  });

  const view = $derived(loader.for(key) ?? (loader.loading ? loader.data : null));
  const num = (v: number): Cell => ({ kind: "number", v });
  const pct = (part: number, whole: number): number => (whole > 0 ? (part / whole) * 100 : 0);

  const rows = $derived(
    (view?.layers ?? []).map((l) => [
      { kind: "text", v: l.name } as Cell,
      num(l.units),
      num(l.dead.length),
      num(Number(fixed(pct(l.dead.length, l.units), 1))),
      num(l.constant.length),
      num(Number(fixed(pct(l.constant.length, l.units), 1))),
      num(Number(fixed(l.saturation_pct, 1))),
      num(l.mean_act),
      num(l.std_act),
    ]),
  );

  const verdict = $derived.by(() => {
    if (!view) {
      return null;
    }
    const p = view.dead_pct;
    const [severity, keyName] = p > 50 ? (["error", "muertas_alta"] as const) : p > 20 ? (["warning", "muertas_moderada"] as const) : (["success", "muertas_ok"] as const);
    return { severity, text: fmt(t(keyName), { pct: p }) };
  });

  const rowsNote = $derived(
    view && view.rows_used < nnViz.sampleSize
      ? fmtText(
          optionalText(nnRaw, "activaciones", "muestras_usadas", "Se usan las {n} muestras de prueba: el conjunto de prueba no tiene más filas."),
          { n: view.rows_used },
        )
      : null,
  );

  function histFigure(i: number) {
    const l = view?.layers[i];
    if (!l) {
      return null;
    }
    return binnedHistogramFigure(l.hist, { title: escapePlotly(t("hist_titulo").replace("{capa}", l.name)), xLabel: "", yLabel: "", color: "#636efa" }, theme.chart);
  }

  function meansFigure(i: number) {
    const l = view?.layers[i];
    if (!l) {
      return null;
    }
    return neuronMeansFigure(l.neuron_means, escapePlotly(t("medias_titulo").replace("{capa}", l.name)), t("eje_neurona"), t("eje_media"), theme.chart);
  }
</script>

<section class="activations" data-nn-activations={view?.rows_used ?? 0}>
  <h3>{t("titulo")}</h3>
  <Expander id="nn-act-que-veras" label={t("que_veras_expander")}>
    <Markdown source={md("redes/activaciones-que-veras.md")} />
  </Expander>
  <Expander id="nn-act-parametros" label={t("parametros_expander")}>
    <div class="grid four">
      <Slider id="nn-act-muestras" label={t("muestras")} help={t("muestras_help")} min={SAMPLES.min} max={SAMPLES.max} value={nnViz.sampleSize} onchange={(v) => (nnViz.sampleSize = Math.round(v))} />
      <NumberInput
        id="nn-act-umbral"
        label={t("umbral_muerta")}
        help={t("umbral_muerta_help")}
        min={0}
        max={1}
        step={0.01}
        value={nnViz.deadThreshold}
        onchange={(v) => (nnViz.deadThreshold = Math.min(1, Math.max(0, v)))}
      />
      <NumberInput
        id="nn-act-std"
        label={t("std_constante")}
        help={t("std_constante_help")}
        min={0.000001}
        max={1}
        step={0.0001}
        value={nnViz.constantStd}
        onchange={(v) => (nnViz.constantStd = Math.min(1, Math.max(1e-6, v)))}
      />
      <Checkbox id="nn-act-mapa" label={t("mostrar_mapa")} checked={nnViz.showMap} onchange={(v) => (nnViz.showMap = v)} />
    </div>
    <div class="grid three">
      <Slider id="nn-act-dispersion" label={t("dispersion")} help={t("dispersion_help")} min={0} max={50} value={nnViz.map.jitter} onchange={(v) => (nnViz.map.jitter = Math.round(v))} />
      <Slider id="nn-act-conexiones" label={t("conexiones")} help={t("conexiones_help")} min={10} max={100} value={nnViz.map.connectionPct} onchange={(v) => (nnViz.map.connectionPct = Math.round(v))} />
      <Checkbox id="nn-act-radio" label={t("radio_adaptativo")} help={t("radio_adaptativo_help")} checked={nnViz.map.adaptiveRadius} onchange={(v) => (nnViz.map.adaptiveRadius = v)} />
    </div>
    <div class="grid three">
      <Checkbox id="nn-act-agrupar" label={t("agrupar")} help={t("agrupar_help")} checked={nnViz.map.group} onchange={(v) => (nnViz.map.group = v)} />
      <Slider id="nn-act-espaciado" label={t("espaciado")} help={t("espaciado_help")} min={12} max={50} value={nnViz.map.minSpacing} onchange={(v) => (nnViz.map.minSpacing = Math.round(v))} />
      <Slider id="nn-act-subcolumnas" label={t("subcolumnas")} help={t("subcolumnas_help")} min={1} max={6} value={nnViz.map.maxColumns} onchange={(v) => (nnViz.map.maxColumns = Math.round(v))} />
    </div>
  </Expander>

  {#if loader.error}
    <Notice severity="error" markdown={fmt(t("error"), { error: loader.error })} />
    <Notice severity="info" markdown={t("error_info")} />
  {:else if !view}
    <Spinner label="Calculando las activaciones…" />
  {:else}
    <div class="loaded" class:updating={loader.loading}>
      {#if rowsNote}
        <p class="caption" data-rows-used={view.rows_used}>{rowsNote}</p>
      {/if}
      <Markdown source={t("resumen_titulo")} />
      <DataTable columns={nnList("activaciones", "resumen_columnas")} {rows} hideIndex label={t("resumen_titulo").replace(/^#+\s*/, "")} />
      {#if verdict}
        <div data-pedagogical data-dead-pct={view.dead_pct}>
          <Notice severity={verdict.severity} markdown={verdict.text} />
        </div>
      {/if}

      <Markdown source={t("detalle_titulo")} />
      {#each view.layers as l, i (l.keras_name)}
        <Expander id="nn-act-capa-{i}" label={fmtText(t("capa_expander"), { capa: l.name, nombre: l.keras_name })}>
          <div class="grid four metrics">
            <Metric label={t("metrica_neuronas")} value={l.units} />
            <Metric label={t("metrica_muertas")} value={String(l.dead.length)} />
            <Metric label={t("metrica_constantes")} value={String(l.constant.length)} />
            <Metric label={t("metrica_saturacion")} value={fixed(l.saturation_pct, 1)} />
          </div>
          <Checkbox
            id="nn-act-hist-{i}"
            label={fmtText(t("mostrar_histograma"), { capa: l.name })}
            checked={nnViz.layerHist[i] ?? false}
            onchange={(v) => (nnViz.layerHist = { ...nnViz.layerHist, [i]: v })}
          />
          {#if nnViz.layerHist[i]}
            {@const fig = histFigure(i)}
            {#if fig}
              <PlotlyChart data={fig.data} layout={fig.layout} height={280} label={t("hist_titulo").replace("{capa}", l.name)} />
            {/if}
          {/if}
          <Checkbox
            id="nn-act-medias-{i}"
            label={fmtText(t("mostrar_medias"), { capa: l.name })}
            checked={nnViz.layerMeans[i] ?? false}
            onchange={(v) => (nnViz.layerMeans = { ...nnViz.layerMeans, [i]: v })}
          />
          {#if nnViz.layerMeans[i]}
            {@const fig = meansFigure(i)}
            {#if fig}
              <PlotlyChart data={fig.data} layout={fig.layout} height={260} label={t("medias_titulo").replace("{capa}", l.name)} />
            {/if}
          {/if}
        </Expander>
      {/each}

      {#if nnViz.showMap}
        <Markdown source={t("mapa_titulo")} />
        <NetworkMap map={view.map} hidden={view.layers} options={nnViz.map} />
      {/if}
      <Expander id="nn-act-muertas" label={t("muertas_expander")}>
        <Markdown source={md("redes/activaciones-muertas.md")} />
      </Expander>
    </div>
  {/if}
</section>

<style>
  h3 {
    margin: 18px 0 10px;
    color: var(--color-heading);
    font-size: 20px;
  }

  .grid {
    display: grid;
    gap: 8px 20px;
    align-items: end;
    margin-bottom: 6px;
  }

  .grid.four {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .grid.three {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .grid.metrics {
    gap: 12px;
    margin-bottom: 8px;
  }

  .caption {
    margin: 2px 0 6px;
    color: var(--color-text-muted);
    font-size: 13px;
  }

  .loaded {
    transition: opacity 120ms ease;
  }

  .loaded.updating {
    opacity: 0.55;
  }

  @media (max-width: 900px) {
    .grid.four,
    .grid.three {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
