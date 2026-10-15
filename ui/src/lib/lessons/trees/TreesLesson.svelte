<script lang="ts">
  /** Tabs 1–6 of 🌲 Árboles de Decisión (tab 0 and the selector are `DatasetTab`). */
  import { onMount } from "svelte";
  import { models } from "../../ml/modelSession.svelte";
  import EvaluationTab from "./EvaluationTab.svelte";
  import ExportTab from "./ExportTab.svelte";
  import FeaturesTab from "./FeaturesTab.svelte";
  import PredictionsTab from "./PredictionsTab.svelte";
  import { LESSON } from "./state.svelte";
  import TrainTab from "./TrainTab.svelte";
  import VisualizationTab from "./VisualizationTab.svelte";

  interface Props {
    active: number;
  }

  let { active }: Props = $props();

  onMount(() => {
    void models.restore(LESSON);
  });
</script>

<div class="trees-tab ml-view" data-trees-tab={active}>
  {#if active === 1}
    <TrainTab />
  {:else if active === 2}
    <EvaluationTab />
  {:else if active === 3}
    <VisualizationTab />
  {:else if active === 4}
    <FeaturesTab />
  {:else if active === 5}
    <PredictionsTab />
  {:else if active === 6}
    <ExportTab />
  {/if}
</div>
