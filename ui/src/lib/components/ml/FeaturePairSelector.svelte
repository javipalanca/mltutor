<script lang="ts">
  /** C8: "### Selección de Características" with the two selects (`viz_feature1/2`). */
  import Markdown from "../../../components/Markdown.svelte";
  import Select from "../../../components/Select.svelte";
  import { compText } from "../../content";

  interface Props {
    features: readonly string[];
    pair: [number, number];
    onchange: (pair: [number, number]) => void;
    idPrefix: string;
  }

  let { features, pair, onchange, idPrefix }: Props = $props();
  const options = $derived(features.map((label, value) => ({ label, value })));
</script>

<Markdown source={compText("frontera", "titulo")} />
<p class="text">{compText("frontera", "texto")}</p>
<div class="columns">
  <Select id="{idPrefix}-feature-x" label={compText("frontera", "primera")} {options} value={pair[0]} onchange={(x) => onchange([x, pair[1]])} />
  <Select id="{idPrefix}-feature-y" label={compText("frontera", "segunda")} {options} value={pair[1]} onchange={(y) => onchange([pair[0], y])} />
</div>

<style>
  .text {
    margin: 6px 4px 4px;
  }

  .columns {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 28px;
  }
</style>
