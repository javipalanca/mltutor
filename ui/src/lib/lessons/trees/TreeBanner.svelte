<script lang="ts">
  /**
   * D32 + D45: the dataset and the configuration of the trained tree (tabs 2–6 always show
   * that model), and a warning if the controls of 🏋️ Entrenamiento no longer match it.
   */
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import type { TrainedModelInfo } from "../../bindings/TrainedModelInfo";
  import { arbText } from "../../content";
  import { fmt } from "../../fmt";
  import { sameConfig } from "../../ml/modelSession.svelte";
  import ModelBanner from "../regression/ModelBanner.svelte";
  import { trees } from "./state.svelte";

  interface Props {
    info: TrainedModelInfo;
  }

  let { info }: Props = $props();

  const config = $derived.by(() => {
    const c = info.config;
    if (c.kind !== "tree") {
      return null;
    }
    // The criterion is an identifier of scikit-learn inside a code span: not escaped.
    return fmt(arbText("comun", "config_modelo").replace("{criterio}", c.criterion), {
      tarea: c.task === "classification" ? "clasificación" : "regresión",
      max_depth: c.max_depth,
      min_samples_split: c.min_samples_split,
      prueba_pct: Math.round(info.test_size * 100),
      nodos: info.tree?.node_count ?? 0,
      hojas: info.tree?.n_leaves ?? 0,
      profundidad: info.tree?.max_depth ?? 0,
    });
  });
  const changed = $derived(!sameConfig(info.config, trees.config()) || Math.abs(info.test_size - trees.testSize) > 1e-9);
</script>

<ModelBanner {info} />
{#if config}
  <p class="config" data-tree-config><Markdown inline source={config} /></p>
{/if}
{#if changed}
  <Notice severity="warning" markdown={arbText("comun", "aviso_config_cambiada")} />
{/if}

<style>
  .config {
    margin: -4px 2px 12px;
    color: var(--color-text-muted);
    font-size: 13.5px;
  }
</style>
