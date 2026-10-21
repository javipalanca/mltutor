<script lang="ts">
  /** D32: which dataset the model of tabs 2–6 was trained on, and a warning if it changed. */
  import Notice from "../../../components/Notice.svelte";
  import type { TrainedModelInfo } from "../../bindings/TrainedModelInfo";
  import { regText } from "../../content";
  import { fmt } from "../../fmt";
  import { datasetKey, session } from "../../session.svelte";

  interface Props {
    info: TrainedModelInfo;
  }

  let { info }: Props = $props();
  const changed = $derived(datasetKey(info.dataset) !== datasetKey(session.selected));
  const current = $derived(session.entry(session.selected)?.label ?? "");
</script>

<Notice severity="info" markdown={fmt(regText("comun", "modelo_entrenado_con"), { dataset: info.dataset_label })} />
{#if changed}
  <Notice
    severity="warning"
    markdown={fmt(regText("comun", "aviso_dataset_cambiado"), { dataset_modelo: info.dataset_label, dataset_actual: current })}
  />
{/if}
