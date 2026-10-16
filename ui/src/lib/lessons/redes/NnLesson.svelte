<script lang="ts">
  /** Tabs 1–6 of 🧠 Redes Neuronales (tab 0 and "📊 Dataset actual" are `DatasetTab`). */
  import { onDestroy, onMount } from "svelte";
  import Notice from "../../../components/Notice.svelte";
  import * as backend from "../../backend";
  import { nnText } from "../../content";
  import { lesson, tabName } from "../../lessons";
  import { models } from "../../ml/modelSession.svelte";
  import ArchitectureTab from "./ArchitectureTab.svelte";
  import EvaluationTab from "./EvaluationTab.svelte";
  import { LESSON, nn } from "./state.svelte";
  import TrainingTab from "./TrainingTab.svelte";
  import VisualizationTab from "./VisualizationTab.svelte";

  interface Props {
    active: number;
  }

  let { active }: Props = $props();
  const lessonState = models.lesson(LESSON);
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
  {:else if !lessonState.info}
    <h2>{tabs[active]}</h2>
    <Notice severity="warning" markdown={nnText("comun", "sin_entrenar")} />
  {:else}
    <h2>{tabs[active]}</h2>
    <Notice
      severity="info"
      markdown={`🚧 La pestaña **${tabName(tabs[active] ?? "")}** de esta lección se está portando a la nueva versión de MLTutor (Fase 5 del plan de migración). La red ya está entrenada y se mostrará aquí.`}
    />
  {/if}
</div>

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }
</style>
