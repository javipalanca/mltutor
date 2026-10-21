<script lang="ts">
  /**
   * C1: evaluation of a classifier (`show_detailed_evaluation(..., "Clasificación")`):
   * metrics with badges, confusion matrix, per-class report, correctly and incorrectly
   * classified samples and precision by class, with their PNG and code downloads.
   * Shared by Regresión logística, Árboles, KNN and Redes.
   */
  import type { Cell } from "../../bindings/Cell";
  import type { ClassificationEvaluation } from "../../bindings/ClassificationEvaluation";
  import type { SampleRow } from "../../bindings/SampleRow";
  import DataTable from "../../../components/DataTable.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import { compList, compRaw, compText, md } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fixed } from "../../format";
  import { pickLevel } from "../../ml/levels";
  import { theme } from "../../theme.svelte";
  import { confusionFigure, precisionByClassFigure } from "./figures";
  import Interpretation from "./Interpretation.svelte";
  import MetricGrid from "./MetricGrid.svelte";
  import ModelCode from "./ModelCode.svelte";

  interface Props {
    evaluation: ClassificationEvaluation;
    modelId: number;
    /** Prefix of the expander ids (one per lesson). */
    idPrefix: string;
    onstale?: () => void;
  }

  let { evaluation, modelId, idPrefix, onstale }: Props = $props();

  const t = compText;
  const metrics = $derived([
    {
      label: t("clasificacion", "accuracy"),
      value: fixed(evaluation.accuracy, 4),
      help: t("clasificacion", "accuracy_help"),
      badge: pickLevel(compRaw("clasificacion", "accuracy_niveles"), evaluation.accuracy),
    },
    { label: t("clasificacion", "precision"), value: fixed(evaluation.weighted_avg.precision, 4), help: t("clasificacion", "precision_help") },
    { label: t("clasificacion", "recall"), value: fixed(evaluation.weighted_avg.recall, 4), help: t("clasificacion", "recall_help") },
    {
      label: t("clasificacion", "f1"),
      value: fixed(evaluation.weighted_avg.f1, 4),
      help: t("clasificacion", "f1_help"),
      badge: pickLevel(compRaw("clasificacion", "f1_niveles"), evaluation.weighted_avg.f1),
    },
  ]);

  const names = $derived(evaluation.per_class.map((c) => c.name));
  const confusion = $derived(confusionFigure(evaluation.confusion, names, theme.chart));
  const precision = $derived(precisionByClassFigure(evaluation.per_class, theme.chart));
  const confusionHeight = $derived(Math.max(380, Math.min(720, 200 + names.length * 70)));

  const text = (v: string): Cell => ({ kind: "text", v });
  // D33: the format Python meant ({:.4f}, support {:.0f}).
  const report = $derived(
    evaluation.per_class.map((c) => [text(fixed(c.precision, 4)), text(fixed(c.recall, 4)), text(fixed(c.f1, 4)), text(String(c.support))]),
  );
  const sampleRows = (rows: readonly SampleRow[]): Cell[][] =>
    rows.map((r) => [text(r.real), text(r.predicted), { kind: "bool", v: r.correct }]);
</script>

<section class="classification" data-evaluation="classification">
  <MetricGrid items={metrics} />

  {#each evaluation.warnings as warning, i (i)}
    <Notice severity="warning" markdown={escapeMarkdown(warning)} />
  {/each}

  <Markdown source={t("clasificacion", "matriz_titulo")} />
  <div class="columns">
    <div>
      <PlotlyChart
        data={confusion.data}
        layout={confusion.layout}
        height={confusionHeight}
        downloadName="matriz_confusion.png"
        downloadLabel={t("clasificacion", "matriz_descarga")}
        label={t("clasificacion", "matriz_title")}
      />
      <ModelCode
        {modelId}
        template="confusion_matrix"
        title={t("clasificacion", "matriz_codigo")}
        downloadName="matriz_confusion.py"
        id="{idPrefix}-codigo-matriz"
        {...onstale ? { onstale } : {}}
      />
    </div>
    <div>
      <DataTable
        label="Informe por clase"
        columns={compList("clasificacion", "informe_columnas")}
        rows={report}
        index={names}
        numericColumns={[0, 1, 2, 3]}
      />
      <Expander id="{idPrefix}-explicacion-metricas" label={t("clasificacion", "metricas_expander")}>
        <Markdown source={md("componentes/metricas-clasificacion.md")} />
      </Expander>
    </div>
  </div>

  <Markdown source={t("clasificacion", "predicciones_titulo")} />
  <div class="columns">
    <div>
      <Markdown source={t("clasificacion", "correctas_titulo")} />
      {#if evaluation.correct_head.length > 0}
        <DataTable
          label={t("clasificacion", "correctas_titulo")}
          columns={compList("clasificacion", "columnas_muestras")}
          rows={sampleRows(evaluation.correct_head)}
          index={evaluation.correct_head.map((r) => String(r.position))}
        />
      {/if}
    </div>
    <div>
      <Markdown source={t("clasificacion", "incorrectas_titulo")} />
      {#if evaluation.incorrect_head.length > 0}
        <DataTable
          label={t("clasificacion", "incorrectas_titulo")}
          columns={compList("clasificacion", "columnas_muestras")}
          rows={sampleRows(evaluation.incorrect_head)}
          index={evaluation.incorrect_head.map((r) => String(r.position))}
        />
      {:else}
        <Notice severity="info" markdown={t("clasificacion", "todas_correctas")} />
      {/if}
    </div>
  </div>

  <div class="centered">
    <PlotlyChart
      data={precision.data}
      layout={precision.layout}
      height={380}
      downloadName="precision_por_clase.png"
      downloadLabel={t("clasificacion", "precision_clase_descarga")}
      label={t("clasificacion", "precision_clase_title")}
    />
  </div>
  <ModelCode
    {modelId}
    template="precision_by_class"
    title={t("clasificacion", "precision_clase_codigo")}
    downloadName="precision_por_clase.py"
    id="{idPrefix}-codigo-precision"
    {...onstale ? { onstale } : {}}
  />

  <Interpretation markdown={evaluation.interpretation_md} />
</section>

<style>
  .columns {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 28px;
    align-items: start;
  }

  .centered {
    max-width: 820px;
    margin: 18px auto 0;
  }

  .classification :global(.md h3) {
    margin-top: 26px;
  }
</style>
