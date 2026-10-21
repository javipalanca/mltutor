<script lang="ts">
  /**
   * 🔮 Predicciones (§2.6): C9 with the ranges of the whole dataset (D63), the result card,
   * "Información del modelo KNN" and, new (D58), the probabilities and the K neighbours (C20).
   */
  import DataTable from "../../../components/DataTable.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { Cell } from "../../bindings/Cell";
  import type { FormControl } from "../../bindings/FormControl";
  import type { PredictionForm as Form } from "../../bindings/PredictionForm";
  import KnnNeighborsTable from "../../components/ml/KnnNeighborsTable.svelte";
  import PredictionForm from "../../components/ml/PredictionForm.svelte";
  import PredictionResultCard from "../../components/ml/PredictionResultCard.svelte";
  import { compList, compText, knnText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fmt } from "../../fmt";
  import { fixed } from "../../format";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import KnnBanner from "./KnnBanner.svelte";
  import { LESSON } from "./state.svelte";

  const t = (key: string): string => knnText("predicciones", key);
  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const form = new ViewLoader<Form>();
  let predicting = $state(false);
  let error = $state<string | null>(null);

  const info = $derived(lessonState.info);
  const key = $derived(String(info?.model_id ?? ""));

  $effect(() => {
    const current = info;
    if (!current) {
      return;
    }
    return form.load(String(current.model_id), () => backend.predictionForm(current.model_id), stale);
  });

  const controls = $derived(form.for(key)?.controls ?? null);

  function defaults(list: readonly FormControl[]): number[] {
    return list.map((c) => {
      switch (c.control.kind) {
        case "checkbox":
          return c.control.default ? 1 : 0;
        case "select":
          return c.control.options[c.control.default_index]?.code ?? 0;
        case "slider":
          return c.control.default;
      }
    });
  }

  const values = $derived(info && controls ? (lessonState.form[info.model_id] ?? defaults(controls)) : []);
  const result = $derived(info ? (lessonState.prediction[info.model_id] ?? null) : null);
  const config = $derived(info?.config.kind === "knn" ? info.config : null);
  const text = (v: string): Cell => ({ kind: "text", v });

  async function predict(): Promise<void> {
    if (!info) {
      return;
    }
    const id = info.model_id;
    predicting = true;
    error = null;
    try {
      lessonState.prediction[id] = await backend.predictOne(id, [...values]);
    } catch (reason) {
      if (backend.isStale(reason)) {
        stale();
      } else {
        error = backend.errorMessage(reason);
      }
    } finally {
      predicting = false;
    }
  }
</script>

<h2>{t("header")}</h2>
{#if !info}
  <Notice severity="info" markdown={knnText("comun", "sin_entrenar")} />
{:else}
  <KnnBanner {info} />
  {#if form.error}
    <Notice severity="error" markdown={escapeMarkdown(form.error)} />
  {:else if !controls}
    <Spinner label="Preparando el formulario…" />
  {:else}
    <PredictionForm {controls} {values} onchange={(next) => (lessonState.form[info.model_id] = next)} idPrefix="knn-pred" />
    <button type="button" class="btn block primary" disabled={predicting} onclick={() => void predict()} data-predict-button>
      {compText("prediccion", "boton")}
    </button>
    {#if error}
      <Notice severity="error" markdown={fmt(compText("prediccion", "error"), { error })} />
    {:else if result}
      <PredictionResultCard result={result.kind === "class" ? { ...result, proba: null } : result} />
      {#if config}
        <Markdown source={t("info_modelo_titulo")} />
        <!-- Weights and metric are sklearn identifiers: substituted before `fmt`. -->
        <Notice severity="info" markdown={t("info_modelo").replace("{weights}", config.weights).replace("{metric}", config.metric).replace("{k}", String(config.n_neighbors))} />
      {/if}
      {#if result.kind === "class" && result.proba && config}
        <Markdown source={t("probabilidades_titulo")} />
        <p class="caption">
          <Markdown
            inline
            source={fmt(t(config.weights === "uniform" ? "probabilidades_nota_uniform" : "probabilidades_nota_distance"), { k: config.n_neighbors })}
          />
        </p>
        <DataTable
          label="Probabilidades"
          columns={compList("prediccion", "prob_columnas")}
          rows={result.proba.map((p) => [text(p.name), text(fixed(p.probability, 4))])}
          index={result.proba.map((_, i) => String(i))}
          numericColumns={[1]}
          compact
        />
      {/if}
      {#if result.neighbors && config}
        <KnnNeighborsTable
          neighbors={result.neighbors.rows}
          tie={result.neighbors.tie_at_kth}
          featureNames={info.feature_names}
          classNames={result.kind === "class" ? (info.class_names ?? []) : null}
          weights={config.weights}
        />
      {/if}
    {/if}
  {/if}
{/if}

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .caption {
    margin: -4px 2px 8px;
    color: var(--color-text-muted);
    font-size: 13px;
  }

  .btn.primary {
    margin-top: 12px;
  }
</style>
