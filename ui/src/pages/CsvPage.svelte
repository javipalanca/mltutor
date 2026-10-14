<script lang="ts">
  import DataTable from "../components/DataTable.svelte";
  import Expander from "../components/Expander.svelte";
  import FilePicker from "../components/FilePicker.svelte";
  import Markdown from "../components/Markdown.svelte";
  import Metric from "../components/Metric.svelte";
  import Notice from "../components/Notice.svelte";
  import PlotlyChart from "../components/PlotlyChart.svelte";
  import Select from "../components/Select.svelte";
  import type { Cell } from "../lib/bindings/Cell";
  import type { TaskChoice } from "../lib/bindings/TaskChoice";
  import { csvTargetFigure } from "../lib/charts";
  import { CSV_EXAMPLES, csvList, csvText, headingText, md } from "../lib/content";
  import { clearCsv, pickCsv, retryRegistration, setTarget, setTask } from "../lib/csvFlow";
  import { escapeMarkdown } from "../lib/escape";
  import { fmt, fmtText } from "../lib/fmt";
  import { isTauri } from "../lib/backend";
  import { session } from "../lib/session.svelte";
  import { matrixCells } from "../lib/table";
  import { theme } from "../lib/theme.svelte";

  const t = csvText;
  const csv = $derived(session.csv);
  const preview = $derived(csv.preview);
  const loaded = $derived(csv.loaded);
  const TASKS: TaskChoice[] = ["auto", "classification", "regression"];
  const taskLabels = csvList("selector", "tarea_opciones");
  const taskOptions = TASKS.map((value, i) => ({ value, label: taskLabels[i] ?? value }));
  let picking = $state(false);

  async function pick(): Promise<void> {
    if (!isTauri()) {
      return;
    }
    picking = true;
    try {
      await pickCsv(t("selector", "dialogo_filtro").replace(/\s*\(.*\)$/, ""));
    } finally {
      picking = false;
    }
  }

  function taskName(choice: TaskChoice): string {
    return taskOptions.find((o) => o.value === choice)?.label ?? choice;
  }

  const total = $derived(preview?.n_rows ?? 0);
  const countRows = (counts: readonly [string, number][]): Cell[][] =>
    counts.map(([, count]) => [{ kind: "number", v: count }]);
</script>

<article class="csv-page" data-route-page="csv">
  <h1 data-page-title>{t("pagina", "header")}</h1>
  <p class="subtitle">{t("pagina", "subtitulo")}</p>
  <Expander id="csv-ayuda" open label={t("pagina", "expander_ayuda")}>
    <Markdown source={md("csv/ayuda.md")} />
  </Expander>
  <hr />

  <h2>{t("selector", "subheader")}</h2>
  <Notice severity="info" markdown={t("selector", "info")} />
  <h3>{headingText(t("selector", "titulo"))}</h3>
  <FilePicker
    label={t("selector", "file_label")}
    help={t("selector", "file_help")}
    fileName={preview?.file_name ?? null}
    buttonLabel={t("selector", "file_boton")}
    clearLabel={t("selector", "file_quitar")}
    disabled={picking}
    onpick={() => void pick()}
    onclear={clearCsv}
  />

  {#if csv.error}
    <Notice severity="error" markdown={escapeMarkdown(csv.error)} />
  {/if}

  {#if !preview}
    <Notice severity="info" markdown={t("selector", "sin_archivo")} />
    <h3>{headingText(CSV_EXAMPLES.heading)}</h3>
    <div class="examples">
      {#each CSV_EXAMPLES.examples as example (example.name)}
        <div class="example">
          <span class="icon" aria-hidden="true">{example.icon}</span>
          <div>
            <Markdown source={`**${example.name}** (${example.task})\n\n_${example.description}_`} />
          </div>
        </div>
      {/each}
    </div>
  {:else}
    {#each preview.notices as notice (notice)}
      <Notice severity="info" markdown={`ℹ️ ${escapeMarkdown(notice)}`} />
    {/each}
    <Notice severity="success" markdown={fmt(t("selector", "cargado"), { filas: preview.n_rows, columnas: preview.n_cols })} />
    <Expander id="csv-vista-previa" open label={t("selector", "expander_vista_previa")}>
      <DataTable label="Vista previa del CSV" columns={preview.head.columns} rows={preview.head.rows} index={preview.head.index} />
      <div class="metrics three">
        <Metric label={t("selector", "metric_filas")} value={preview.n_rows} />
        <Metric label={t("selector", "metric_columnas")} value={preview.n_cols} />
        <Metric label={t("selector", "metric_faltantes")} value={preview.missing_values} />
      </div>
    </Expander>

    <h3>{headingText(t("selector", "configuracion"))}</h3>
    <div class="columns">
      <Select
        id="csv-objetivo"
        label={t("selector", "objetivo_label")}
        help={t("selector", "objetivo_help")}
        options={preview.columns.map((c) => ({ label: c.name, value: c.name }))}
        value={csv.target ?? preview.default_target}
        onchange={(value) => void setTarget(value)}
      />
      <Select
        id="csv-tarea"
        label={t("selector", "tarea_label")}
        help={t("selector", "tarea_help")}
        options={taskOptions}
        value={csv.task ?? "auto"}
        onchange={(value) => void setTask(value)}
      />
    </div>

    <h3>{headingText(t("selector", "info_objetivo"))}</h3>
    <div class="columns target-info">
      <div>
        <Markdown source={fmt(t("selector", "columna"), { col: preview.target.column })} />
        <Markdown source={fmt(t("selector", "valores_unicos"), { n: preview.target.nunique })} />
        <Markdown source={fmt(t("selector", "tipo_datos"), { dtype: preview.target.dtype })} />
      </div>
      {#if preview.target.value_counts}
        <div>
          <Markdown source={t("selector", "distribucion_valores")} />
          {#each preview.target.value_counts as [value, count] (value)}
            <p class="count-line">{fmtText(t("selector", "valor_linea"), { valor: value, n: count, pct: (100 * count) / Math.max(1, total) })}</p>
          {/each}
        </div>
      {/if}
    </div>

    {#if csv.registerCancelled}
      <div class="cancelled" data-csv-register-cancelled>
        <Notice severity="info" markdown="Registro del CSV cancelado: el archivo está leído, pero no se ha añadido con esta configuración." />
        <button type="button" class="btn" onclick={() => void retryRegistration()}>Registrar de nuevo</button>
      </div>
    {/if}

    {#if loaded}
      <Notice severity="success" markdown={fmt(t("selector", "anadido"), { nombre: preview.file_name })} />
      <hr />
      <section class="result" data-csv-loaded>
        <Notice severity="success" markdown={t("resultado", "exito")} />
        <h3>{headingText(t("resultado", "titulo"))}</h3>
        <div class="metrics four">
          <Metric label={t("resultado", "metric_filas")} value={loaded.n_rows} />
          <Metric label={t("resultado", "metric_columnas")} value={loaded.n_cols} />
          <Metric label={t("resultado", "metric_objetivo")} value={loaded.target} />
          <Metric
            label={fmtText(t("resultado", "metric_tarea"), { icon: loaded.task_choice === "classification" ? "🏷️" : "📈" })}
            value={taskName(loaded.task_choice)}
          />
        </div>
        <Expander id="csv-estadisticas" label={t("resultado", "expander_estadisticas")}>
          <DataTable
            label="Estadísticas descriptivas del CSV"
            columns={loaded.describe.columns}
            rows={matrixCells(loaded.describe.values.values, loaded.describe.values.rows, loaded.describe.values.cols)}
            index={loaded.describe.stats}
          />
        </Expander>
        <Expander id="csv-distribucion" label={t("resultado", "expander_distribucion")}>
          {@const figure = csvTargetFigure(loaded.target, loaded.target_stats, theme.chart)}
          {#if loaded.target_stats.kind === "classes"}
            <div class="columns">
              <DataTable
                label="Recuento por clase"
                columns={[t("resultado", "tabla_cantidad")]}
                rows={countRows(loaded.target_stats.value_counts)}
                index={loaded.target_stats.value_counts.map(([label]) => label)}
              />
              <PlotlyChart data={figure.data} layout={figure.layout} height={420} downloadName="distribucion_objetivo.png" />
            </div>
          {:else}
            {@const s = loaded.target_stats}
            <div class="columns">
              <div>
                <Markdown source={t("resultado", "estadisticas")} />
                <p class="count-line">{fmtText(t("resultado", "minimo"), { v: s.min })}</p>
                <p class="count-line">{fmtText(t("resultado", "maximo"), { v: s.max })}</p>
                <p class="count-line">{fmtText(t("resultado", "media"), { v: s.mean })}</p>
                <p class="count-line">{fmtText(t("resultado", "mediana"), { v: s.median })}</p>
                <p class="count-line">{fmtText(t("resultado", "std"), { v: s.std })}</p>
              </div>
              <PlotlyChart data={figure.data} layout={figure.layout} height={420} downloadName="distribucion_objetivo.png" />
            </div>
          {/if}
        </Expander>
      </section>
    {/if}
  {/if}
</article>

<style>
  h1 {
    margin-top: 8px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .subtitle {
    margin: 18px 4px 22px;
  }

  h2 {
    margin: 10px 0 12px;
    color: var(--color-heading);
    font-size: 19px;
  }

  h3 {
    margin: 22px 0 12px;
    color: var(--color-heading);
    font-size: 17px;
  }

  .examples {
    display: grid;
    gap: 0;
  }

  .example {
    display: grid;
    grid-template-columns: 56px 1fr;
    align-items: center;
    padding: 12px 4px;
    border-bottom: 1px solid var(--color-divider);
  }

  .icon {
    font-size: 30px;
  }

  .metrics {
    display: grid;
    gap: 16px;
    margin: 12px 0;
  }

  .metrics.three {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .metrics.four {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
    align-items: start;
  }

  .target-info :global(.md p) {
    margin-bottom: 12px;
  }

  .cancelled {
    margin: 12px 0;
  }

  .count-line {
    margin: 0 0 8px;
    user-select: text;
  }
</style>
