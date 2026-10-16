<script lang="ts">
  /**
   * 📊 Predicción por lotes (C29, D81; unreachable in Python): Rust opens the native dialog,
   * reads the CSV, checks its columns by NAME (sin_filas → duplicadas → faltan →
   * no_numericas → vacias) and predicts every row with the scaler of the training. The
   * table shows the first rows; "📥 Descargar Resultados" saves the whole CSV from Rust.
   */
  import CodeBlock from "../../../components/CodeBlock.svelte";
  import DataTable from "../../../components/DataTable.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import * as backend from "../../backend";
  import type { BatchTemplate } from "../../bindings/BatchTemplate";
  import type { TrainedModelInfo } from "../../bindings/TrainedModelInfo";
  import { nnRaw, nnText, optionalText, regText } from "../../content";
  import { fmt, fmtText } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { status } from "../../status.svelte";
  import { nnPredictions } from "./predictions.svelte";
  import { LESSON } from "./state.svelte";

  interface Props {
    info: TrainedModelInfo;
    template: BatchTemplate | null;
  }

  let { info, template }: Props = $props();
  const t = (key: string): string => nnText("lotes", key);
  const stale = (): void => void models.refresh(LESSON);
  let reading = $state(false);
  let error = $state<string | null>(null);
  let saved = $state<{ severity: "success" | "error"; markdown: string } | null>(null);
  const batch = $derived(nnPredictions.batch[info.model_id] ?? null);

  async function pick(): Promise<void> {
    const id = info.model_id;
    reading = true;
    error = null;
    saved = null;
    try {
      const view = await backend.nnPredictBatch(id).promise;
      if (view) {
        nnPredictions.batch = { ...nnPredictions.batch, [id]: view };
      }
    } catch (reason) {
      if (backend.isStale(reason)) {
        stale();
      } else if (!backend.isCancelled(reason)) {
        error = backend.errorMessage(reason);
      }
    } finally {
      reading = false;
    }
  }

  async function download(): Promise<void> {
    saved = null;
    try {
      const name = await backend.nnSaveBatch(info.model_id);
      if (name) {
        saved = { severity: "success", markdown: fmt(regText("exportar", "exito_guardado"), { archivo: name }) };
        status.flash(`Guardado: ${name}`);
      }
    } catch (reason) {
      saved = { severity: "error", markdown: fmt(regText("exportar", "error_guardado"), { error: backend.errorMessage(reason) }) };
    }
  }

  /** Captions of the result columns (Probabilidad, Confianza, Fuera de rango). */
  const helps = $derived.by(() => {
    if (!batch?.results) {
      return [];
    }
    const out: string[] = [];
    if (batch.kind === "binary") {
      out.push(`**${t("columna_probabilidad")}:** ${t("probabilidad_help")}`);
    }
    if (batch.kind !== "regression") {
      out.push(`**${t("columna_confianza")}:** ${t("confianza_help")}`);
    }
    out.push(`**${t("columna_fuera_rango")}:** ${t("columna_fuera_rango_help")}`);
    return out;
  });
</script>

<section class="batch" data-nn-batch>
  <h3>{t("titulo")}</h3>
  <Markdown source={t("texto")} />

  {#if !batch && template}
    <Notice severity="info" markdown={t("formato")} />
    <DataTable label="Formato esperado" columns={template.sample.columns} rows={template.sample.rows} index={template.sample.index} compact />
    <Markdown source={t("columnas")} />
    <CodeBlock code={template.columns.join(", ")} language="text" />
    <p class="caption">{t("nota_unidades")}</p>
  {/if}

  <button type="button" class="btn block" class:primary={!batch} disabled={reading} onclick={() => void pick()} data-nn-batch-pick>
    {t("btn_elegir")}
  </button>

  {#if error}
    <Notice severity="error" markdown={fmt(t("error"), { error })} />
  {:else if batch}
    <div data-nn-batch-file={batch.file_name}>
      <Notice severity="success" markdown={fmt(t("cargado"), { filas: batch.rows, columnas: batch.cols })} />
      {#each batch.errors as message (message)}
        <Notice severity="error" markdown={message} />
      {/each}
      {#if batch.missing.length > 0}
        <p class="caption">{t("faltan_ayuda")}</p>
      {/if}
      {#if batch.errors.length === 0 && batch.extra.length > 0}
        <Notice severity="warning" markdown={fmt(t("sobran"), { columnas: batch.extra.join(", ") })} />
      {/if}
      {#if batch.errors.length === 0 && batch.reordered}
        <Notice severity="info" markdown={t("reordenadas")} />
      {/if}
      {#if batch.results}
        <Notice severity="success" markdown={fmt(t("generadas"), { n: batch.predicted })} />
        {#if batch.out_of_range}
          <div data-nn-out-of-range>
            <Notice severity="warning" markdown={batch.out_of_range} />
          </div>
        {/if}
        <div data-nn-batch-results={batch.predicted}>
          <DataTable
            label="Predicciones"
            columns={batch.results.columns}
            rows={batch.results.rows}
            index={batch.results.index}
            sortable
          />
        </div>
        {#if batch.results.rows.length < batch.predicted}
          <p class="caption">
            {fmtText(optionalText(nnRaw, "lotes", "primeras_filas", "Se muestran las primeras {n} de {total} filas; el CSV descargado las contiene todas."), {
              n: batch.results.rows.length,
              total: batch.predicted,
            })}
          </p>
        {/if}
        {#if !batch.output_is_probability && batch.kind !== "regression"}
          <p class="caption" data-nn-not-probability>{nnText("predicciones", "nota_tanh_multiclase")}</p>
        {/if}
        {#each helps as help (help)}
          <p class="caption"><Markdown inline source={help} /></p>
        {/each}
        <button type="button" class="btn block primary" onclick={() => void download()} data-nn-batch-download>
          {t("btn_descargar")}
        </button>
        {#if saved}
          <div data-nn-batch-saved={saved.severity}>
            <Notice severity={saved.severity} markdown={saved.markdown} />
          </div>
        {/if}
      {/if}
    </div>
  {/if}
</section>

<style>
  h3 {
    margin: 18px 0 10px;
    color: var(--color-heading);
    font-size: 20px;
  }

  .caption {
    margin: 4px 2px 10px;
    color: var(--color-text-muted);
    font-size: 13px;
  }

  .btn {
    margin: 10px 0;
  }
</style>
