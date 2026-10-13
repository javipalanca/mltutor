<script lang="ts">
  /**
   * 📊 Datos tab shared by the four lessons (fase1-shell-datos.md §2):
   * selector, information, preview, describe, target distribution, correlation,
   * pairplot and the exploratory-analysis code.
   */
  import { untrack } from "svelte";
  import ButtonPanel from "../components/ButtonPanel.svelte";
  import CodeBlock from "../components/CodeBlock.svelte";
  import DataTable from "../components/DataTable.svelte";
  import Expander from "../components/Expander.svelte";
  import Markdown from "../components/Markdown.svelte";
  import Metric from "../components/Metric.svelte";
  import MultiSelect from "../components/MultiSelect.svelte";
  import Notice from "../components/Notice.svelte";
  import PlotlyChart from "../components/PlotlyChart.svelte";
  import Radio from "../components/Radio.svelte";
  import Select from "../components/Select.svelte";
  import Slider from "../components/Slider.svelte";
  import Spinner from "../components/Spinner.svelte";
  import * as backend from "../lib/backend";
  import type { CorrelationMatrix } from "../lib/bindings/CorrelationMatrix";
  import type { DatasetRef } from "../lib/bindings/DatasetRef";
  import type { DatasetSummary } from "../lib/bindings/DatasetSummary";
  import type { DescribeTable } from "../lib/bindings/DescribeTable";
  import type { DiagKind } from "../lib/bindings/DiagKind";
  import type { PairplotData } from "../lib/bindings/PairplotData";
  import type { TablePreview } from "../lib/bindings/TablePreview";
  import type { TargetDistribution } from "../lib/bindings/TargetDistribution";
  import { correlationFigure, correlationHeight, hasWebGL, pairplotFigure, pairplotHeight, targetDistributionFigure } from "../lib/charts";
  import { codeTemplate, datosList, datosText, headingText, md, type LessonId } from "../lib/content";
  import { fmt } from "../lib/fmt";
  import { fixed } from "../lib/format";
  import type { DataBlock } from "../lib/lessons";
  import { datasetKey, session, type Analysis } from "../lib/session.svelte";
  import { matrixCells } from "../lib/table";
  import { theme } from "../lib/theme.svelte";
  import { pairplotDefaults, pairplotNotices } from "../lib/pairplot";

  interface Props {
    lesson: LessonId;
    block: DataBlock;
  }

  let { lesson, block }: Props = $props();

  const t = datosText;
  const selected = $derived(session.selected);
  const entry = $derived(session.entry(selected));
  const key = $derived(datasetKey(selected));

  let summary = $state<DatasetSummary | null>(null);
  let preview = $state<TablePreview | null>(null);
  let describe = $state<DescribeTable | null>(null);
  let distribution = $state<TargetDistribution | null>(null);
  let loadError = $state<string | null>(null);

  // Load the four views in parallel whenever the dataset changes (§5.4).
  $effect(() => {
    const dataset = selected;
    const full = block === "full";
    if (block === "current") {
      return;
    }
    let stale = false;
    loadError = null;
    void (async () => {
      try {
        const [s, p, d, dist] = await Promise.all([
          backend.datasetSummary(dataset),
          full ? backend.datasetPreview(dataset, 10) : Promise.resolve(null),
          full ? backend.datasetDescribe(dataset) : Promise.resolve(null),
          full ? backend.targetDistribution(dataset) : Promise.resolve(null),
        ]);
        if (!stale) {
          summary = s;
          preview = p;
          describe = d;
          distribution = dist;
        }
      } catch (error) {
        if (!stale) {
          loadError = backend.errorMessage(error);
          summary = null;
        }
      }
    })();
    return () => {
      stale = true;
    };
  });

  const ready = $derived(summary !== null && datasetKey(summary.dataset) === key);
  const tabState = $derived(session.datasetTab(lesson, selected));

  async function choose(dataset: DatasetRef): Promise<void> {
    if (datasetKey(dataset) === key) {
      return;
    }
    session.selected = dataset;
    try {
      await backend.selectDataset(dataset);
    } catch (error) {
      loadError = backend.errorMessage(error);
    }
  }

  // --- Advanced analysis ---

  function setAnalysis(value: Exclude<Analysis, null>): void {
    tabState.analysis = value;
  }

  let correlation = $state<{ key: string; data: CorrelationMatrix } | null>(null);
  let correlationError = $state<string | null>(null);

  $effect(() => {
    if (block !== "full" || tabState.analysis !== "correlation" || untrack(() => correlation?.key) === key) {
      return;
    }
    const dataset = selected;
    const requested = key;
    let stale = false;
    correlationError = null;
    void backend
      .correlationMatrix(dataset)
      .then((data) => {
        if (!stale) {
          correlation = { key: requested, data };
        }
      })
      .catch((error: unknown) => {
        if (!stale) {
          correlationError = backend.errorMessage(error);
        }
      });
    return () => {
      stale = true;
    };
  });

  const defaults = $derived(summary && ready ? pairplotDefaults(summary.feature_display) : null);
  const maxFeatures = $derived(defaults ? (tabState.maxFeatures ?? defaults.defaultMax) : 0);
  const chosen = $derived.by(() => {
    if (!defaults || !summary) {
      return [] as string[];
    }
    const picked = (tabState.features ?? summary.feature_display.slice(0, maxFeatures)).filter((f) =>
      summary?.feature_display.includes(f),
    );
    return picked.slice(0, maxFeatures);
  });
  const emptySelection = $derived(tabState.features !== null && tabState.features.length === 0);
  const plotFeatures = $derived(
    emptySelection && summary ? summary.feature_display.slice(0, maxFeatures) : chosen,
  );
  const notices = $derived(summary ? pairplotNotices(plotFeatures.length, summary.n_rows) : []);

  let pairplot = $state<{ request: string; data: PairplotData; ms: number } | null>(null);
  let pairplotError = $state<string | null>(null);
  let pairplotLoading = $state(false);
  const pairplotRequest = $derived(JSON.stringify([key, plotFeatures, tabState.diag]));

  // Once generated, option changes regenerate the pairplot (debounced); old answers are ignored.
  $effect(() => {
    const request = pairplotRequest;
    if (block !== "full" || tabState.analysis !== "pairplot" || !tabState.generated || !ready) {
      return;
    }
    if (untrack(() => pairplot?.request) === request || plotFeatures.length === 0) {
      return;
    }
    const features = [...plotFeatures];
    const diag = tabState.diag;
    const dataset = selected;
    let stale = false;
    pairplotLoading = true;
    pairplotError = null;
    const timer = setTimeout(() => {
      const started = performance.now();
      backend
        .pairplotData({ dataset, features, diag })
        .then((data) => {
          if (!stale) {
            pairplotLoading = false;
            pairplot = { request, data, ms: performance.now() - started };
          }
        })
        .catch((error: unknown) => {
          if (!stale) {
            pairplotLoading = false;
            pairplotError = backend.errorMessage(error);
          }
        });
    }, 150);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
  });

  function generate(): void {
    tabState.generated = true;
  }

  function reset(): void {
    tabState.generated = false;
    pairplot = null;
    pairplotError = null;
    pairplotLoading = false;
  }

  const diagOptions = datosList("pairplot", "diag_opciones");
  const DIAG_VALUES: DiagKind[] = ["hist", "kde"];

  const datasetOptions = $derived(session.datasets.map((e) => ({ label: e.label, value: e.dataset })));
  const selectedOption = $derived(datasetOptions.find((o) => datasetKey(o.value) === key)?.value ?? selected);
  const label = $derived(entry?.label ?? summary?.label ?? "");
  const taskIcon = $derived(summary?.task === "classification" ? "🏷️" : "📈");
  const taskLabel = $derived(summary?.task === "classification" ? "Clasificación" : "Regresión");
  const code = codeTemplate("code/analisis_exploratorio.py.j2");
</script>

{#if block === "current"}
  <Notice severity="info" markdown={fmt(t("seleccion", "dataset_actual"), { dataset: label })} />
  <hr />
{:else}
  <section class="dataset-tab" data-dataset-tab={lesson} data-ready={ready}>
    <h2>{t("seleccion", "header")}</h2>
    {#if lesson === "redes" && block === "full"}
      <Notice severity="info" markdown={md("datos/tips-redes.md")} />
    {/if}
    <h3>{headingText(t("seleccion", "subheader"))}</h3>
    <Select
      id="dataset-{lesson}"
      label={t("seleccion", "select_label")}
      help={t("seleccion", "select_help")}
      options={datasetOptions}
      value={selectedOption}
      onchange={(dataset) => void choose(dataset)}
    />
    {#if entry?.blurb}
      <div class="blurb"><Markdown source={entry.blurb} /></div>
    {/if}
    {#if entry?.synthetic}
      <Notice severity="info" markdown={t("seleccion", "moons_info")} />
    {/if}
    <hr />

    {#if block === "full"}
      {#if loadError}
        <Notice severity="error" markdown={fmt(t("exploracion", "error"), { error: loadError })} />
        <Notice severity="info" markdown={t("exploracion", "error_info")} />
      {:else if !ready || !summary}
        <Spinner label="Cargando el dataset…" />
      {:else}
        <Notice severity="success" markdown={fmt(t("info", "seleccionado"), { dataset: label })} />
        <h3>{headingText(t("info", "titulo"))}</h3>
        <div class="metrics four">
          <Metric label={t("info", "metric_filas")} value={summary.n_rows} help={t("info", "metric_filas_help")} />
          <Metric label={t("info", "metric_columnas")} value={summary.n_features + 1} help={t("info", "metric_columnas_help")} />
          <Metric label={t("info", "metric_objetivo")} value={summary.target_column} help={t("info", "metric_objetivo_help")} />
          <Metric label={fmt(t("info", "metric_tarea"), { icon: taskIcon })} value={taskLabel} help={t("info", "metric_tarea_help")} />
        </div>

        <h2 class="exploration">{t("exploracion", "header")}</h2>
        <h3>{headingText(t("exploracion", "vista_previa"))}</h3>
        {#if preview}
          <DataTable label="Vista previa de datos" columns={preview.columns} rows={preview.rows} index={preview.index} sortable />
        {/if}
        <Expander id="datos-que-significan-{lesson}" label={t("exploracion", "expander_que_significan")}>
          <Markdown source={fmt(md("datos/que-significan.md"), { target_col: summary.target_column })} />
        </Expander>

        <h3>{headingText(t("exploracion", "estadisticas"))}</h3>
        {#if describe}
          <DataTable
            label="Estadísticas descriptivas"
            columns={describe.columns}
            rows={matrixCells(describe.values.values, describe.values.rows, describe.values.cols)}
            index={describe.stats}
          />
        {/if}

        <h3>{headingText(t("exploracion", "distribucion"))}</h3>
        {#if distribution}
          {#if distribution.kind === "bar"}
            {#if distribution.balanced}
              <Notice severity="success" markdown={t("distribucion", "balanceado")} />
            {:else}
              <Notice severity="warning" markdown={fmt(t("distribucion", "desbalanceado"), { ratio: distribution.balance_ratio })} />
              <Notice severity="info" markdown={t("distribucion", "desbalanceado_info")} />
            {/if}
          {:else}
            <Expander id="datos-estadisticas-objetivo-{lesson}" label={t("distribucion", "expander_estadisticas")}>
              <div class="metrics four">
                <Metric label={t("distribucion", "metric_media")} value={fixed(distribution.mean, 2)} />
                <Metric label={t("distribucion", "metric_std")} value={fixed(distribution.std, 2)} />
                <Metric label={t("distribucion", "metric_min")} value={fixed(distribution.min, 2)} />
                <Metric label={t("distribucion", "metric_max")} value={fixed(distribution.max, 2)} />
              </div>
            </Expander>
          {/if}
          {@const figure = targetDistributionFigure(distribution, theme.chart)}
          <div class="centered">
            <PlotlyChart
              data={figure.data}
              layout={figure.layout}
              height={420}
              downloadName={distribution.kind === "bar" ? "distribucion_clases.png" : "distribucion_objetivo.png"}
              label="Distribución del objetivo"
            />
          </div>
        {/if}

        <h3>{headingText(t("analisis", "titulo"))}</h3>
        <h4>{headingText(t("analisis", "elige"))}</h4>
        <ButtonPanel
          label="Tipo de análisis"
          value={tabState.analysis}
          onchange={setAnalysis}
          options={[
            { label: t("analisis", "boton_correlacion"), value: "correlation", help: t("analisis", "boton_correlacion_help") },
            { label: t("analisis", "boton_pairplot"), value: "pairplot", help: t("analisis", "boton_pairplot_help") },
          ]}
        />

        {#if tabState.analysis === "correlation"}
          <section class="analysis" data-analysis="correlation">
            <h4>{headingText(t("correlacion", "titulo"))}</h4>
            <Notice severity="info" markdown={md("datos/correlacion.md")} />
            {#if correlationError}
              <Notice severity="error" markdown={correlationError} />
            {:else if correlation?.key === key}
              {@const figure = correlationFigure(correlation.data, theme.chart)}
              {@const n = correlation.data.labels.length}
              <PlotlyChart
                data={figure.data}
                layout={figure.layout}
                height={correlationHeight(n)}
                downloadName="matriz_correlacion.png"
                downloadLabel={t("correlacion", "descargar")}
                label="Matriz de correlación"
              />
            {:else}
              <Spinner label="Calculando la correlación…" />
            {/if}
          </section>
        {:else if tabState.analysis === "pairplot" && defaults}
          <section class="analysis" data-analysis="pairplot">
            <h4>{headingText(t("pairplot", "titulo"))}</h4>
            <Notice severity="info" markdown={md("datos/pairplot.md")} />
            <h5>{headingText(t("pairplot", "opciones"))}</h5>
            <div class="columns">
              <Radio
                name="diag-{lesson}"
                horizontal
                label={t("pairplot", "diag_label")}
                options={DIAG_VALUES.map((value, i) => ({ value, label: diagOptions[i] ?? value }))}
                value={tabState.diag}
                onchange={(value) => (tabState.diag = value)}
              />
              {#if defaults.slider}
                <Slider
                  id="pairplot-max-{lesson}"
                  label={t("pairplot", "slider_label")}
                  help={t("pairplot", "slider_help")}
                  min={defaults.slider.min}
                  max={defaults.slider.max}
                  value={maxFeatures}
                  onchange={(value) => {
                    tabState.maxFeatures = value;
                    if (tabState.features && tabState.features.length > value) {
                      tabState.features = tabState.features.slice(0, value);
                    }
                  }}
                />
              {:else}
                <Notice severity="info" markdown={fmt(t("pairplot", "pocas_caracteristicas"), { n: defaults.n })} />
              {/if}
            </div>
            <h5>{headingText(t("pairplot", "selecciona"))}</h5>
            <MultiSelect
              label={t("pairplot", "multiselect_label")}
              help={fmt(t("pairplot", "multiselect_help"), { max: maxFeatures })}
              options={summary.feature_display}
              value={chosen}
              max={maxFeatures}
              onchange={(value) => (tabState.features = value)}
            />
            {#if emptySelection}
              <Notice severity="info" markdown={fmt(t("pairplot", "ninguna"), { max: maxFeatures })} />
            {/if}
            {#each notices as notice (notice.key)}
              <Notice severity={notice.severity} markdown={fmt(t("pairplot", notice.key), notice.values)} />
            {/each}
            <div class="generate">
              <button type="button" class="btn block primary" onclick={generate}>{t("pairplot", "boton_generar")}</button>
              <button type="button" class="btn block" title={t("pairplot", "boton_reiniciar_help")} onclick={reset}>
                {t("pairplot", "boton_reiniciar")}
              </button>
            </div>
            {#if !tabState.generated}
              <Notice severity="info" markdown={t("pairplot", "haz_clic")} />
              <Expander id="pairplot-consejos-{lesson}" label={t("pairplot", "expander_consejos")}>
                <Markdown source={md("datos/pairplot-consejos.md")} />
              </Expander>
            {:else if pairplotError}
              <Notice severity="error" markdown={fmt(t("pairplot", "error"), { error: pairplotError })} />
              <Notice severity="info" markdown={t("pairplot", "error_info")} />
            {:else if pairplot}
              {#if pairplot.data.sampled}
                <Notice
                  severity="info"
                  markdown={fmt(t("pairplot", "usando_puntos"), { n: pairplot.data.n_points, total: pairplot.data.n_total })}
                />
              {/if}
              {@const figure = pairplotFigure(pairplot.data, theme.chart, hasWebGL())}
              <div class="pairplot" class:updating={pairplotLoading}>
                <PlotlyChart
                  data={figure.data}
                  layout={figure.layout}
                  height={pairplotHeight(pairplot.data.features.length)}
                  downloadName="matriz_dispersion.png"
                  downloadLabel={t("pairplot", "descargar")}
                  label="Matriz de dispersión"
                />
              </div>
              <Notice severity="success" markdown={fmt(t("pairplot", "exito"), { t: pairplot.ms / 1000 })} />
            {:else}
              <Spinner label={headingText(t("pairplot", "estado_inicio"))} />
            {/if}
          </section>
        {/if}

        <CodeBlock
          id="datos-codigo-{lesson}"
          title={headingText(t("codigo", "titulo"))}
          {code}
          downloadName="analisis_exploratorio.py"
        />
      {/if}
    {:else if loadError}
      <Notice severity="error" markdown={fmt(t("info", "error"), { error: loadError })} />
    {/if}
  </section>
{/if}

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  h2.exploration {
    margin-top: 34px;
  }

  h3 {
    margin: 24px 0 12px;
    color: var(--color-heading);
    font-size: 19px;
  }

  h4 {
    margin: 20px 0 10px;
    color: var(--color-heading);
    font-size: 16px;
  }

  h5 {
    margin: 18px 0 8px;
    color: var(--color-heading);
    font-size: 14px;
  }

  .blurb {
    margin: 4px 4px 0;
  }

  .metrics {
    display: grid;
    gap: 16px;
    margin: 10px 0 14px;
  }

  .metrics.four {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .centered {
    max-width: 820px;
    margin: 0 auto;
  }

  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
    align-items: start;
  }

  .generate {
    display: grid;
    grid-template-columns: 3fr 1fr;
    gap: 16px;
    margin: 14px 0;
  }

  .pairplot {
    transition: opacity 120ms ease;
  }

  .pairplot.updating {
    opacity: 0.55;
  }
</style>
