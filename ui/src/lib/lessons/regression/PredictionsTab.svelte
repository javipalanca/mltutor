<script lang="ts">
  /** 🔮 Predicciones (§2.6): C9 with the metadata of the training dataset (D32). */
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { FormControl } from "../../bindings/FormControl";
  import type { PredictionForm as Form } from "../../bindings/PredictionForm";
  import PredictionForm from "../../components/ml/PredictionForm.svelte";
  import PredictionResultCard from "../../components/ml/PredictionResultCard.svelte";
  import { compText, regText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fmt } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import ModelBanner from "./ModelBanner.svelte";
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

<h2>{regText("predicciones", "header")}</h2>
{#if !info}
  <Notice severity="info" markdown={regText("comun", "sin_entrenar_predicciones")} />
{:else}
  <ModelBanner {info} />
  {#if form.error}
    <Notice severity="error" markdown={escapeMarkdown(form.error)} />
  {:else if !controls}
    <Spinner label="Preparando el formulario…" />
  {:else}
    <PredictionForm
      {controls}
      {values}
      onchange={(next) => (lessonState.form[info.model_id] = next)}
      idPrefix="regresion-pred"
    />
    <button type="button" class="btn block primary" disabled={predicting} onclick={() => void predict()} data-predict-button>
      {compText("prediccion", "boton")}
    </button>
    {#if error}
      <Notice severity="error" markdown={fmt(compText("prediccion", "error"), { error })} />
    {:else if result}
      <PredictionResultCard {result} />
      {#if result.kind === "value" && info.dataset_task === "classification"}
        <Notice severity="info" markdown={fmt(regText("predicciones", "aviso_lineal_valor"), { valor: result.value })} />
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

  .btn.primary {
    margin-top: 12px;
  }
</style>
