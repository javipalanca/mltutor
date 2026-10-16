<script lang="ts">
  /** Tabs 1–6 of 🧠 Redes Neuronales (tab 0 and "📊 Dataset actual" are `DatasetTab`). */
  import { onDestroy, onMount } from "svelte";
  import Notice from "../../../components/Notice.svelte";
  import * as backend from "../../backend";
  import { nnText } from "../../content";
  import { lesson } from "../../lessons";
  import { models } from "../../ml/modelSession.svelte";
  import ArchitectureTab from "./ArchitectureTab.svelte";
  import EvaluationTab from "./EvaluationTab.svelte";
  import ExportTab from "./ExportTab.svelte";
  import PredictionsTab from "./PredictionsTab.svelte";
  import { LESSON, nn } from "./state.svelte";
  import TrainingTab from "./TrainingTab.svelte";
  import VisualizationTab from "./VisualizationTab.svelte";

  interface Props {
    active: number;
  }

  let { active }: Props = $props();
  const tabs = lesson(LESSON).tabs;
  let unlisten: (() => void) | null = null;
  let destroyed = false;

  onMount(() => {
    void models.restore(LESSON);
    void backend.onNnEpoch((event) => nn.onEpochEvent(event)).then((stop) => {
      if (destroyed) {
        stop();
      } else {
        unlisten = stop;
      }
    });
  });

  onDestroy(() => {
    destroyed = true;
    unlisten?.();
  });
</script>

<div class="nn-tab ml-view" data-nn-tab={active}>
  {#if active === 1}
    <ArchitectureTab />
  {:else if active === 2}
    <TrainingTab />
  {:else if active === 3}
    <EvaluationTab />
  {:else if active === 4}
    <VisualizationTab />
  {:else if active === 5}
    <PredictionsTab />
  {:else if active === 6}
    <ExportTab />
  {:else}
    <h2>{tabs[active]}</h2>
    <Notice severity="warning" markdown={nnText("comun", "sin_entrenar")} />
  {/if}
</div>

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }
</style>
