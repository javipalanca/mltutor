<script lang="ts">
  /**
   * 📉 Visualización (§2.4): 📊 Probabilidades, 📉 ROC, 🌈 Frontera, 🎮 Interactiva and
   * 📏 Distancias (classification) or 📊 Predicciones and 🌐 Superficie (regression).
   */
  import { untrack } from "svelte";
  import ButtonPanel from "../../../components/ButtonPanel.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { ClassDistancesView } from "../../bindings/ClassDistancesView";
  import type { ProbabilityView } from "../../bindings/ProbabilityView";
  import type { ResidualsView } from "../../bindings/ResidualsView";
  import type { RocData } from "../../bindings/RocData";
  import ClassDistanceHeatmap from "../../components/ml/ClassDistanceHeatmap.svelte";
  import DecisionGridView from "../../components/ml/DecisionGridView.svelte";
  import KnnPlayground from "../../components/ml/KnnPlayground.svelte";
  import ProbabilityDistribution from "../../components/ml/ProbabilityDistribution.svelte";
  import ResidualsAnalysis from "../../components/ml/ResidualsAnalysis.svelte";
  import RocPrCharts from "../../components/ml/RocPrCharts.svelte";
  import { knnRaw, knnText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { classOfRow } from "../../ml/modelClasses";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import KnnBanner from "./KnnBanner.svelte";
  import { K_MAX, knn, LESSON } from "./state.svelte";

  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const t = (key: string): string => knnText("visualizacion", key);

  type Panel = readonly (readonly [string, string, string])[];
  const panel = (key: string): Panel => (knnRaw("visualizacion", key) as Panel | undefined) ?? [];

  const classifier = $derived(lessonState.info?.model_task === "classification");
  const options = $derived(panel(classifier ? "clasificacion" : "regresion").map(([label, value]) => ({ label, value })));
  // `create_button_panel`: the first option if the stored one is not in this panel.
  const viz = $derived(options.some((o) => o.value === lessonState.viz) ? (lessonState.viz ?? "") : (options[0]?.value ?? ""));

  const roc = new ViewLoader<RocData>();
  const probs = new ViewLoader<ProbabilityView>();
  const residuals = new ViewLoader<ResidualsView>();
  const distances = new ViewLoader<ClassDistancesView>();

  $effect(() => {
    const info = lessonState.info;
    if (!info) {
      return;
    }
    const id = info.model_id;
    const key = String(id);
    switch (viz) {
      case "Probabilidades":
        return probs.load(key, () => backend.probabilityView(id), stale);
      case "ROC":
        return roc.load(key, () => backend.rocCurves(id), stale);
      case "Distancias":
        return distances.load(key, () => backend.classDistances(id), stale);
      case "Predicciones":
        return residuals.load(key, () => backend.residualsView(id), stale);
      default:
        return undefined;
    }
  });

  // The canvas starts with the model's K, weights and metric; retraining empties it.
  $effect(() => {
    const info = lessonState.info;
    if (info) {
      untrack(() => knn.syncPlayground(info));
    }
  });

  const key = $derived(String(lessonState.info?.model_id ?? ""));
</script>

<h2>{t("header")}</h2>
{#if !lessonState.info}
  <Notice severity="info" markdown={knnText("comun", "sin_entrenar")} />
{:else}
  {@const info = lessonState.info}
  <KnnBanner {info} />
  <Markdown source={t("opciones_titulo")} />
  <ButtonPanel label="Tipo de visualización" value={viz} {options} onchange={(value) => (lessonState.viz = value)} />
  {#if viz === "Probabilidades"}
    {@const data = probs.for(key)}
    <Markdown source={t("probabilidades_titulo")} />
    {#if probs.error}
      <Notice severity="error" markdown={escapeMarkdown(probs.error)} />
    {:else if data}
      <ProbabilityDistribution view={data} idPrefix="knn" />
    {:else}
      <Spinner label="Calculando las probabilidades…" />
    {/if}
  {:else if viz === "ROC"}
    {@const data = roc.for(key)}
    {#if roc.error}
      <Notice severity="error" markdown={escapeMarkdown(roc.error)} />
    {:else if data}
      <RocPrCharts
        roc={data}
        showPerClass={lessonState.rocPerClass}
        averages={lessonState.rocAverages}
        onoptions={(perClass, averages) => {
          lessonState.rocPerClass = perClass;
          lessonState.rocAverages = averages;
        }}
        idPrefix="knn"
        negativeClass={classOfRow(info, 0)}
      />
    {:else}
      <Spinner label="Calculando las curvas…" />
    {/if}
  {:else if viz === "Frontera" || viz === "Superficie"}
    <DecisionGridView
      modelId={info.model_id}
      features={info.feature_names}
      pair={lessonState.pair}
      onpair={(pair) => (lessonState.pair = pair)}
      mode={viz === "Frontera" ? "boundary" : "surface"}
      idPrefix="knn"
      onstale={stale}
    />
  {:else if viz === "Interactiva"}
    {#if knn.playground.modelId === info.model_id}
      <KnnPlayground modelId={info.model_id} features={info.feature_names} controller={knn} kMax={K_MAX} onstale={stale} />
    {/if}
  {:else if viz === "Distancias"}
    {@const data = distances.for(key)}
    {#if distances.error}
      <Notice severity="error" markdown={escapeMarkdown(distances.error)} />
    {:else if data}
      <ClassDistanceHeatmap view={data} idPrefix="knn" />
    {:else}
      <Spinner label="Calculando las distancias…" />
    {/if}
  {:else if viz === "Predicciones"}
    {@const data = residuals.for(key)}
    {#if residuals.error}
      <Notice severity="error" markdown={escapeMarkdown(residuals.error)} />
    {:else if data}
      <ResidualsAnalysis view={data} idPrefix="knn" predictionsTitle={t("predicciones_titulo")} residualsTitle={t("residuos_titulo")} />
    {:else}
      <Spinner label="Calculando los residuos…" />
    {/if}
  {/if}
{/if}

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }
</style>
