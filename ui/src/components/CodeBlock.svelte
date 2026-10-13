<script lang="ts">
  import { datosText } from "../lib/content";
  import { PYTHON_FILTER, saveText } from "../lib/download";
  import { highlightPython } from "../lib/markdown";
  import { status } from "../lib/status.svelte";
  import Expander from "./Expander.svelte";
  import Notice from "./Notice.svelte";

  interface Props {
    code: string;
    language?: string;
    /** With a title: `show_code_with_download` (heading, "Ver código", download + tip). */
    title?: string;
    downloadName?: string;
    /** Expander id (remembers whether it is open). */
    id?: string;
  }

  let { code, language = "python", title, downloadName, id }: Props = $props();
  let copied = $state(false);
  const html = $derived(language === "python" ? highlightPython(code) : null);

  async function copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(code);
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch {
      copied = false;
    }
  }

  async function download(): Promise<void> {
    if (!downloadName) {
      return;
    }
    const saved = await saveText(downloadName, code, PYTHON_FILTER);
    if (saved) {
      status.flash(`Guardado: ${saved}`);
    }
  }
</script>

{#snippet panel()}
  <div class="code-panel">
    <div class="toolbar">
      <span class="language">{language.toUpperCase()}</span>
      <button type="button" class="copy" onclick={copy}>{copied ? "✓ Copiado" : "Copiar código"}</button>
    </div>
    <pre><code class="hljs">{#if html !== null}{@html html}{:else}{code}{/if}</code></pre>
  </div>
{/snippet}

{#if title}
  <h4 class="title">{title}</h4>
  <Expander label={datosText("codigo", "expander")} {...id ? { id } : {}}>
    {@render panel()}
  </Expander>
  {#if downloadName}
    <div class="actions">
      <button type="button" class="btn block" onclick={download}>{datosText("codigo", "descargar")}</button>
      <Notice severity="info" markdown={datosText("codigo", "nota_copiar")} />
    </div>
  {/if}
{:else}
  {@render panel()}
{/if}

<style>
  .title {
    margin: 22px 0 8px;
    color: var(--color-heading);
    font-size: 16px;
  }

  .code-panel {
    background: var(--color-code-bg);
    border: 1px solid var(--color-code-border);
    border-radius: var(--radius-box);
    overflow: hidden;
  }

  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px 0;
  }

  .language {
    color: var(--color-code-label);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
  }

  .copy {
    padding: 4px 12px;
    font-size: 12px;
    color: var(--color-code-text);
    background: var(--color-code-button-bg);
    border: 1px solid var(--color-code-button-border);
    border-radius: 6px;
    cursor: pointer;
  }

  .copy:hover {
    background: var(--color-code-button-hover);
  }

  pre {
    margin: 0;
    padding: 12px 18px 18px;
    overflow-x: auto;
    color: var(--color-code-text);
    font-family: var(--font-mono);
    font-size: 14px;
    line-height: 1.5;
    user-select: text;
    tab-size: 4;
  }

  pre :global(.hljs-keyword),
  pre :global(.hljs-literal) {
    color: #c4a7ff;
  }

  pre :global(.hljs-built_in) {
    color: #7dd3fc;
  }

  pre :global(.hljs-title),
  pre :global(.hljs-params),
  pre :global(.hljs-attr) {
    color: #82e2bf;
  }

  pre :global(.hljs-string) {
    color: #a8db8f;
  }

  pre :global(.hljs-comment) {
    color: #98a8c2;
    font-style: italic;
  }

  pre :global(.hljs-number) {
    color: #f6bd79;
  }

  pre :global(.hljs-operator),
  pre :global(.hljs-punctuation) {
    color: #89c9ef;
  }

  .actions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    align-items: start;
  }

  .actions :global(.notice) {
    margin: 0;
  }
</style>
