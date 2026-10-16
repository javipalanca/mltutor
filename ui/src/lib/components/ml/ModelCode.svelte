<script lang="ts">
  /**
   * C12: Python code generated in Rust for the current model (`generate_code`), shown with
   * `show_code_with_download` (heading, "Ver código", download with `save_file`).
   */
  import CodeBlock from "../../../components/CodeBlock.svelte";
  import Notice from "../../../components/Notice.svelte";
  import * as backend from "../../backend";
  import type { CodeTemplate } from "../../bindings/CodeTemplate";
  import { escapeMarkdown } from "../../escape";

  interface Props {
    modelId: number;
    template: CodeTemplate;
    title: string;
    /** Without it, no download button (the file is saved another way). */
    downloadName?: string;
    /** Expander id (remembers whether it is open). */
    id: string;
    /** D30: features of the boundary. */
    pair?: [number, number] | null;
    onstale?: () => void;
  }

  let { modelId, template, title, downloadName, id, pair = null, onstale }: Props = $props();

  let code = $state<{ key: string; text: string } | null>(null);
  let error = $state<string | null>(null);
  const key = $derived(`${modelId}|${template}|${pair?.join(",") ?? ""}`);

  $effect(() => {
    const requested = key;
    const extra = pair ? { feature_x: pair[0], feature_y: pair[1] } : null;
    let stale = false;
    error = null;
    backend
      .generateCode(modelId, template, extra)
      .then((text) => {
        if (!stale) {
          code = { key: requested, text };
        }
      })
      .catch((reason: unknown) => {
        if (stale) {
          return;
        }
        if (backend.isStale(reason)) {
          onstale?.();
        } else {
          error = backend.errorMessage(reason);
        }
      });
    return () => {
      stale = true;
    };
  });
</script>

{#if error}
  <Notice severity="error" markdown={escapeMarkdown(error)} />
{:else if code}
  <CodeBlock {id} {title} code={code.text} {...downloadName ? { downloadName } : {}} />
{/if}
