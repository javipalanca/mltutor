<script lang="ts">
  /**
   * 🔮 Predicciones (§2.6): C9 (form and result card, with the probabilities of the leaf,
   * D46) and C16 (the decision path), with the metadata of the training dataset (D32).
   */
  import DataTable from "../../../components/DataTable.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { FormControl } from "../../bindings/FormControl";
  import type { PredictionForm as Form } from "../../bindings/PredictionForm";
  import DecisionPathList from "../../components/ml/DecisionPathList.svelte";
  import ModelCode from "../../components/ml/ModelCode.svelte";
  import PredictionForm from "../../components/ml/PredictionForm.svelte";
  import PredictionResultCard from "../../components/ml/PredictionResultCard.svelte";
  import type { Cell } from "../../bindings/Cell";
  import { arbText, compList, compText, md } from "../../content";
  import { fixed } from "../../format";
  import { escapeMarkdown } from "../../escape";
  import { fmt } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import TreeBanner from "./TreeBanner.svelte";
  import { LESSON } from "./state.svelte";

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
  const text = (v: string): Cell => ({ kind: "text", v });
  const median = $derived(info?.config.kind === "tree" && info.config.criterion === "absolute_error");

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

<h2>{arbText("predicciones", "header")}</h2>
{#if !info}
  <Notice severity="warning" markdown={arbText("comun", "sin_entrenar")} />
{:else}
  <TreeBanner {info} />
  {#if form.error}
    <Notice severity="error" markdown={escapeMarkdown(form.error)} />
  {:else if !controls}
    <Spinner label="Preparando el formulario…" />
  {:else}
    <PredictionForm {controls} {values} onchange={(next) => (lessonState.form[info.model_id] = next)} idPrefix="arboles-pred" />
    <button type="button" class="btn block primary" disabled={predicting} onclick={() => void predict()} data-predict-button>
      {compText("prediccion", "boton")}
    </button>
    {#if error}
      <Notice severity="error" markdown={fmt(compText("prediccion", "error"), { error })} />
    {:else if result}
      <!-- D46: the probabilities of the leaf go below, with their own title. -->
      <PredictionResultCard result={result.kind === "class" ? { ...result, proba: null } : result} />
      {#if result.kind === "class" && result.proba && result.path}
        <Markdown source={arbText("predicciones", "probabilidades_titulo")} />
        <p class="caption">
          <Markdown inline source={fmt(arbText("predicciones", "probabilidades_nota"), { muestras: result.path.leaf_samples, nodo: result.path.leaf })} />
        </p>
        <DataTable
          label="Probabilidades de la hoja"
          columns={compList("prediccion", "prob_columnas")}
          rows={result.proba.map((p) => [text(p.name), text(fixed(p.probability, 4))])}
          index={result.proba.map((_, i) => String(i))}
          numericColumns={[1]}
          compact
        />
      {/if}
      {#if result.path}
        <Markdown source={compText("prediccion", "camino_titulo")} />
        <DecisionPathList path={result.path} classNames={info.class_names} {median} />
        <Expander id="arboles-prediccion-hoja" label={arbText("predicciones", "hoja_expander")}>
          <Markdown source={md("arboles/prediccion-hoja.md")} />
        </Expander>
        <ModelCode
          modelId={info.model_id}
          template="decision_path"
          title={arbText("predicciones", "codigo_camino")}
          downloadName="camino_decision.py"
          id="arboles-codigo-camino"
          onstale={stale}
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
