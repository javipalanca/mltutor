<script lang="ts">
  /** C3: "### 🔍 Interpretación de Resultados" + the summary built in Rust from the TOML. */
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import { compText } from "../../content";

  interface Props {
    /** Markdown with numbers only (no data strings), from `interpretation_md`. */
    markdown: string;
  }

  let { markdown }: Props = $props();
</script>

{#if markdown.trim()}
  <section class="interpretation" data-interpretation>
    <Markdown source={compText("interpretacion", "titulo")} />
    <Notice severity="info" {markdown} />
  </section>
{/if}
