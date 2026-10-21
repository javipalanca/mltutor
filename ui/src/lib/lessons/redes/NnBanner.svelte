<script lang="ts">
  /**
   * D32/D91 in 📈 and 🎯: which dataset the network was trained on and a warning if the
   * dataset or the controls of 🏗️/⚙️ changed since then (Python kept showing the old network
   * without saying so).
   */
  import Notice from "../../../components/Notice.svelte";
  import type { TrainedModelInfo } from "../../bindings/TrainedModelInfo";
  import { nnText, regText } from "../../content";
  import { fmt } from "../../fmt";
  import { sameConfig } from "../../ml/modelSession.svelte";
  import { datasetKey, session } from "../../session.svelte";
  import { architecture } from "./architecture.svelte";
  import { nn } from "./state.svelte";

  interface Props {
    info: TrainedModelInfo;
  }

  let { info }: Props = $props();
  const key = $derived(datasetKey(session.selected));
  const view = $derived(architecture.matches(session.selected) ? architecture.view : null);

  $effect(() => {
    architecture.request(session.selected, nn.arch(architecture.view), nn.validationPct);
  });

  const datasetChanged = $derived(datasetKey(info.dataset) !== key);
  // Only with the view of the current dataset: before it, the default neurons are unknown.
  const configChanged = $derived(!datasetChanged && view !== null && !sameConfig(info.config, nn.config(key, view)));
</script>

<div data-nn-banner data-stale={datasetChanged || configChanged}>
  <Notice severity="info" markdown={fmt(regText("comun", "modelo_entrenado_con"), { dataset: info.dataset_label })} />
  {#if datasetChanged}
    <Notice
      severity="warning"
      markdown={fmt(regText("comun", "aviso_dataset_cambiado"), {
        dataset_modelo: info.dataset_label,
        dataset_actual: session.entry(session.selected)?.label ?? "",
      })}
    />
  {:else if configChanged}
    <Notice severity="warning" markdown={nnText("entrenamiento", "aviso_config_cambiada")} />
  {/if}
</div>
