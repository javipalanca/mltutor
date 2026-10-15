<script lang="ts">
  /**
   * C14: the rules of the tree (`export_text`), as `st.text` in Python: a monospaced block
   * that keeps the indentation, with its download (`arbol_texto.txt`).
   */
  import { errorMessage } from "../../backend";
  import { saveText } from "../../download";
  import { status } from "../../status.svelte";

  interface Props {
    text: string;
    downloadName: string;
    downloadLabel: string;
  }

  let { text, downloadName, downloadLabel }: Props = $props();

  async function download(): Promise<void> {
    try {
      const saved = await saveText(downloadName, text, "text");
      if (saved) {
        status.flash(`Guardado: ${saved}`);
      }
    } catch (error) {
      status.flash(errorMessage(error), 8000);
    }
  }
</script>

<pre class="rules" data-tree-rules>{text}</pre>
<button type="button" class="btn" onclick={() => void download()} data-tree-rules-download>{downloadLabel}</button>

<style>
  .rules {
    margin: 12px 0;
    padding: 14px 16px;
    max-height: 640px;
    overflow: auto;
    background: var(--color-code-inline-bg);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    font-family: "SF Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace;
    font-size: 13px;
    line-height: 1.5;
    tab-size: 4;
    user-select: text;
  }
</style>
