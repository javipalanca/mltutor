<script lang="ts">
  /** 🔍 Características (§2.5): C10 by permutation only (KNN has no native importance). */
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { FeatureImportanceView } from "../../bindings/FeatureImportanceView";
  import FeatureImportanceChart from "../../components/ml/FeatureImportanceChart.svelte";
  import { compText, knnText } from "../../content";
  import { fmt } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import KnnBanner from "./KnnBanner.svelte";
  import { LESSON } from "./state.svelte";

  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const view = new ViewLoader<FeatureImportanceView>();
  const key = $derived(String(lessonState.info?.model_id ?? ""));

  $effect(() => {
    const info = lessonState.info;
    if (!info) {
      return;
    }
    return view.load(String(info.model_id), () => backend.featureImportance(info.model_id, "permutation"), stale);
  });

  const data = $derived(view.for(key));
</script>

<h2>{knnText("caracteristicas", "header")}</h2>
{#if !lessonState.info}
  <Notice severity="info" markdown={knnText("comun", "sin_entrenar_caracteristicas")} />
{:else}
  <KnnBanner info={lessonState.info} />
  {#if view.error}
    <Notice severity="error" markdown={fmt(compText("importancia", "error_perm"), { error: view.error })} />
    <Notice severity="info" markdown={compText("importancia", "error_perm_info")} />
  {:else if data}
    <p class="caption">{knnText("caracteristicas", "nota_escala")}</p>
    <FeatureImportanceChart view={data} idPrefix="knn" permutationText={knnText("caracteristicas", "permutacion")} />
  {:else}
    <Notice severity="info" markdown={compText("importancia", "calculando")} />
    <Spinner label={compText("importancia", "spinner")} />
  {/if}
{/if}

<style>
  .caption {
    margin: -2px 2px 10px;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
  }

  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }
</style>
