<script lang="ts">
  /**
   * D32 + D45: the dataset and the configuration of the trained KNN (tabs 2–6 always show
   * that model), and a warning if the controls of 🏋️ Entrenamiento no longer match it.
   */
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import type { TrainedModelInfo } from "../../bindings/TrainedModelInfo";
  import { knnText } from "../../content";
  import { fmt } from "../../fmt";
  import { sameConfig } from "../../ml/modelSession.svelte";
  import ModelBanner from "../regression/ModelBanner.svelte";
  import { knn } from "./state.svelte";

  interface Props {
    info: TrainedModelInfo;
  }

  let { info }: Props = $props();

  const config = $derived.by(() => {
    const c = info.config;
    if (c.kind !== "knn") {
      return null;
    }
    // Weights and metric are scikit-learn identifiers inside code spans: not escaped.
    const template = knnText("comun", "config_modelo").replace("{weights}", c.weights).replace("{metric}", c.metric);
    return fmt(template, {
      tarea: info.model_task === "classification" ? "clasificación" : "regresión",
      k: c.n_neighbors,
      prueba_pct: Math.round(info.test_size * 100),
      n_train: info.n_train,
    });
  });
  const changed = $derived(!sameConfig(info.config, knn.config()) || Math.abs(info.test_size - knn.testSize) > 1e-9);
</script>

<ModelBanner {info} />
{#if config}
  <p class="config" data-knn-config><Markdown inline source={config} /></p>
{/if}
{#if changed}
  <Notice severity="warning" markdown={knnText("comun", "aviso_config_cambiada")} />
{/if}

<style>
  .config {
    margin: -4px 2px 12px;
    color: var(--color-text-muted);
    font-size: 13.5px;
  }
</style>
