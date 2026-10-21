<script lang="ts">
  /**
   * 🎯 Visualización of 🧠 Redes Neuronales (fase5-redes.md §2.5): 📊 Historial (C27 + the 4
   * cards), 🧠 Análisis de Pesos (C25), 🎯 Superficie de Decisión (C8 in D78 mode: the whole
   * network with the other features at their test mean, axes in original units) and 📉
   * Análisis de Activaciones (C26). A network whose weights are
   * not finite (K1) only shows its history; a recovered divergence only warns.
   */
  import ButtonPanel from "../../../components/ButtonPanel.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { HistoryView } from "../../bindings/HistoryView";
  import DecisionGridView from "../../components/ml/DecisionGridView.svelte";
  import NnActivations from "../../components/ml/NnActivations.svelte";
  import NnViewTree from "../../components/ml/NnViewTree.svelte";
  import NnWeights from "../../components/ml/NnWeights.svelte";
  import TrainingHistoryChart from "../../components/ml/TrainingHistoryChart.svelte";
  import { md, nnRaw, nnText, optionalText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import NnBanner from "./NnBanner.svelte";
  import { divergenceText, recoveredDivergenceText } from "./nnTexts";
  import { LESSON } from "./state.svelte";

  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const t = (key: string): string => nnText("visualizacion", key);

  type Panel = readonly (readonly [string, string, string])[];
  const options = ((nnRaw("visualizacion", "botones") as Panel | undefined) ?? []).map(([label, value]) => ({ label, value }));
  // `create_button_panel`: the first option if the stored one is not in this panel (D13).
  const viz = $derived(options.some((o) => o.value === lessonState.viz) ? (lessonState.viz ?? "") : (options[0]?.value ?? ""));

  const history = new ViewLoader<HistoryView>();
  $effect(() => {
    const info = lessonState.info;
    if (!info) {
      return;
    }
    const id = info.model_id;
    return history.load(String(id), () => backend.nnHistory(id), stale);
  });
  const hist = $derived(history.for(String(lessonState.info?.model_id ?? "")));
  const diverged = $derived(hist?.diverged_epoch ?? null);
  // K1: only non-finite weights block 🧠/🎯/📉; a recovered divergence only warns.
  const blocked = $derived(diverged !== null && hist?.network_valid === false);
</script>

<section class="nn-visualization" data-nn-visualization={viz}>
  <h2>{t("header")}</h2>
  {#if !lessonState.info}
    <Notice severity="warning" markdown={nnText("comun", "sin_entrenar")} />
  {:else}
    {@const info = lessonState.info}
    <NnBanner {info} />
    <Notice severity="info" markdown={md("redes/visualizacion-intro.md")} />
    <ButtonPanel label="Visualización" value={viz} {options} onchange={(value) => (lessonState.viz = value)} />
    {#if diverged !== null && !blocked && viz !== "Historial" && hist}
      <div data-nn-diverged-recovered={diverged} data-pedagogical>
        <Notice severity="warning" markdown={recoveredDivergenceText(diverged, hist.best_epoch)} />
      </div>
    {/if}
    {#if blocked && diverged !== null && viz !== "Historial"}
      <div data-nn-diverged={diverged} data-pedagogical>
        <Notice severity="error" markdown={divergenceText(diverged)} />
      </div>
    {:else if viz === "Historial"}
      <h3>{nnText("historial", "titulo")}</h3>
      <Expander id="nn-historial-interpretar" label={nnText("historial", "interpretar_expander")}>
        <Markdown source={md("redes/historial-interpretar.md")} />
      </Expander>
      {#if history.error}
        <Notice severity="error" markdown={escapeMarkdown(history.error)} />
      {:else if hist}
        <TrainingHistoryChart logs={hist.epochs} metric={hist.metric} bestEpoch={hist.best_epoch} lrDrops={hist.lr_drops} downloadName="historial_entrenamiento.png" />
        {#if diverged !== null && blocked}
          <div data-nn-diverged={diverged} data-pedagogical>
            <Notice severity="error" markdown={divergenceText(diverged)} />
          </div>
        {:else if diverged !== null}
          <div data-nn-diverged-recovered={diverged} data-pedagogical>
            <Notice severity="warning" markdown={recoveredDivergenceText(diverged, hist.best_epoch)} />
          </div>
        {:else if hist.cards}
          <div data-nn-history-cards>
            <NnViewTree nodes={[hist.cards]} idPrefix="nn-historial" />
          </div>
        {/if}
      {:else}
        <Spinner label="Cargando el historial…" />
      {/if}
    {:else if viz === "Pesos"}
      <NnWeights modelId={info.model_id} onstale={stale} />
    {:else if viz === "Superficie"}
      <h3>{nnText("superficie", "titulo")}</h3>
      {#if info.feature_names.length < 2}
        <Notice severity="info" markdown="Se necesitan al menos 2 características para dibujar la superficie de decisión." />
      {:else}
        <DecisionGridView
          modelId={info.model_id}
          features={info.feature_names}
          pair={lessonState.pair}
          onpair={(pair) => (lessonState.pair = pair)}
          mode={info.model_task === "classification" ? "boundary" : "surface"}
          idPrefix="redes"
          onstale={stale}
          note={nnText("superficie", "nota_medias")}
          ownScales={{ surface: nnText("superficie", "colorbar_regresion"), points: optionalText(nnRaw, "superficie", "colorbar_puntos", "Valor real") }}
          success={nnText("superficie", info.model_task === "classification" ? "exito_clasificacion" : "exito_regresion")}
        />
      {/if}
    {:else if viz === "Activaciones"}
      <NnActivations modelId={info.model_id} hiddenActivation={info.config.kind === "neural_net" ? info.config.arch.activation : "relu"} onstale={stale} />
    {/if}
  {/if}
</section>

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  h3 {
    margin: 18px 0 10px;
    color: var(--color-heading);
    font-size: 20px;
  }
</style>
