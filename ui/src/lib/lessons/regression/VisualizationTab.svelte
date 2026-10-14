<script lang="ts">
  /** 📉 Visualización (§2.4): residuals/surface (linear) or ROC/boundary/probabilities (logistic). */
  import ButtonPanel from "../../../components/ButtonPanel.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { ProbabilityView } from "../../bindings/ProbabilityView";
  import type { ResidualsView } from "../../bindings/ResidualsView";
  import type { RocData } from "../../bindings/RocData";
  import DecisionGridView from "../../components/ml/DecisionGridView.svelte";
  import ProbabilityDistribution from "../../components/ml/ProbabilityDistribution.svelte";
  import ResidualsAnalysis from "../../components/ml/ResidualsAnalysis.svelte";
  import RocPrCharts from "../../components/ml/RocPrCharts.svelte";
  import { regRaw, regText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import ModelBanner from "./ModelBanner.svelte";
  import { LESSON } from "./state.svelte";

  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const t = (key: string): string => regText("visualizacion", key);

  type Panel = readonly (readonly [string, string, string])[];
  const panel = (key: string): Panel => (regRaw("visualizacion", key) as Panel | undefined) ?? [];

  const linear = $derived(lessonState.info?.model_task === "regression");
  const options = $derived(panel(linear ? "lineal" : "logistica").map(([label, value]) => ({ label, value })));
  // `create_button_panel`: the first option if the stored one is not in this panel.
  const viz = $derived(options.some((o) => o.value === lessonState.viz) ? (lessonState.viz ?? "") : (options[0]?.value ?? ""));

  const roc = new ViewLoader<RocData>();
  const probs = new ViewLoader<ProbabilityView>();
  const residuals = new ViewLoader<ResidualsView>();

  $effect(() => {
    const info = lessonState.info;
    if (!info) {
      return;
    }
    const id = info.model_id;
    const key = String(id);
    if (viz === "ROC") {
      return roc.load(key, () => backend.rocCurves(id), stale);
    }
    if (viz === "Probs") {
      return probs.load(key, () => backend.probabilityView(id), stale);
    }
    if (viz === "Residuos") {
      return residuals.load(key, () => backend.residualsView(id), stale);
    }
    return undefined;
  });

  const key = $derived(String(lessonState.info?.model_id ?? ""));
</script>

<h2>{t("header")}</h2>
{#if !lessonState.info}
  <Notice severity="info" markdown={regText("comun", "sin_entrenar_visualizacion")} />
{:else}
  {@const info = lessonState.info}
  <ModelBanner {info} />
  <ButtonPanel label="Visualización" value={viz} {options} onchange={(value) => (lessonState.viz = value)} />
  {#if viz === "Residuos"}
    {@const data = residuals.for(key)}
    {#if residuals.error}
      <Notice severity="error" markdown={escapeMarkdown(residuals.error)} />
    {:else if data}
      <ResidualsAnalysis view={data} idPrefix="regresion" predictionsTitle={t("residuos_titulo_pred")} residualsTitle={t("residuos_titulo")} />
    {:else}
      <Spinner label="Calculando los residuos…" />
    {/if}
  {:else if viz === "Superficie" || viz === "Frontera"}
    <DecisionGridView
      modelId={info.model_id}
      features={info.feature_names}
      pair={lessonState.pair}
      onpair={(pair) => (lessonState.pair = pair)}
      mode={viz === "Frontera" ? "boundary" : "surface"}
      idPrefix="regresion"
      onstale={stale}
    />
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
        idPrefix="regresion"
        negativeClass={info.class_names?.[0] ?? "0"}
      />
    {:else}
      <Spinner label="Calculando las curvas…" />
    {/if}
  {:else if viz === "Probs"}
    {@const data = probs.for(key)}
    <Markdown source={t("probs_titulo")} />
    {#if probs.error}
      <Notice severity="error" markdown={escapeMarkdown(probs.error)} />
    {:else if data}
      <ProbabilityDistribution view={data} idPrefix="regresion" />
    {:else}
      <Spinner label="Calculando las probabilidades…" />
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
