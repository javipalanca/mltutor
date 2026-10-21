<script lang="ts">
  import { renderInline, renderMarkdown } from "../lib/markdown";

  interface Props {
    /** Markdown source (content/*.md or a short message). */
    source: string;
    /** Render without a wrapping paragraph. */
    inline?: boolean;
  }

  let { source, inline = false }: Props = $props();
  const html = $derived(inline ? renderInline(source) : renderMarkdown(source));
</script>

{#if inline}
  <span class="md">{@html html}</span>
{:else}
  <div class="md">{@html html}</div>
{/if}
