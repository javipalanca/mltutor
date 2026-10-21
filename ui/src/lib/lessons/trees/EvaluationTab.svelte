<script lang="ts">
  /** 📈 Evaluación (§2.3): C1/C2/C3 by the task of the MODEL (D45). */
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { Evaluation } from "../../bindings/Evaluation";
  import ClassificationEvaluation from "../../components/ml/ClassificationEvaluation.svelte";
  import RegressionEvaluation from "../../components/ml/RegressionEvaluation.svelte";
  import { arbText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import TreeBanner from "./TreeBanner.svelte";
  import { LESSON } from "./state.svelte";

  const lessonState = models.lesson(LESSON);
  const view = new ViewLoader<Evaluation>();
  const stale = (): void => void models.refresh(LESSON);

  $effect(() => {
    const info = lessonState.info;
    if (!info) {
      return;
    }
    return view.load(String(info.model_id), () => backend.evaluateModel(info.model_id), stale);
  });

  const evaluation = $derived(lessonState.info ? view.for(String(lessonState.info.model_id)) : null);
</script>

<h2>{arbText("evaluacion", "header")}</h2>
{#if !lessonState.info}
  <Notice severity="warning" markdown={arbText("comun", "sin_entrenar")} />
{:else}
  <TreeBanner info={lessonState.info} />
  {#if view.error}
    <Notice severity="error" markdown={escapeMarkdown(view.error)} />
  {:else if !evaluation}
    <Spinner label="Evaluando el modelo…" />
  {:else if evaluation.kind === "classification"}
    <ClassificationEvaluation evaluation={evaluation} modelId={lessonState.info.model_id} idPrefix="arboles" onstale={stale} />
  {:else}
    <RegressionEvaluation evaluation={evaluation} modelId={lessonState.info.model_id} idPrefix="arboles" onstale={stale} />
  {/if}
{/if}

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }
</style>
