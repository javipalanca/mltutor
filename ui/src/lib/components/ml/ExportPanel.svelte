<script lang="ts">
  /**
   * C11 + C12 (D1/D28): "📄 Generar Código", "💾 Exportar a ONNX" and "📦 Paquete Python
   * (.zip)". ONNX and the package are built in Rust and saved with the native dialog
   * (`export_model`): the path never goes through JS. Shared by the four lessons.
   */
  import type { CodeTemplate } from "../../bindings/CodeTemplate";
  import type { ExportFormat } from "../../bindings/ExportFormat";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import * as backend from "../../backend";
  import { fmt } from "../../fmt";
  import { session } from "../../session.svelte";
  import { status } from "../../status.svelte";
  import ModelCode from "./ModelCode.svelte";

  export interface ExportButton {
    format: ExportFormat;
    label: string;
    help?: string | undefined;
  }

  export interface ExportTexts {
    intro?: string;
    /** "📄 Generar Código" (empty: no code button, e.g. 🌲 Árboles). */
    code: string;
    codeHelp?: string;
    onnx: string;
    onnxHelp?: string;
    package: string;
    packageHelp?: string;
    codeTitle: string;
    codeFile: string;
    saved: string;
    error: string;
    helpExpander?: string;
    helpMarkdown?: string;
  }

  interface Props {
    modelId: number;
    texts: ExportTexts;
    idPrefix: string;
    onstale: () => void;
    /** Buttons of the formats; by default ONNX and the Python package (F2). */
    formats?: readonly ExportButton[];
    /** Script shown by the code button. */
    codeTemplate?: CodeTemplate;
  }

  let { modelId, texts, idPrefix, onstale, formats, codeTemplate = "regression_code" }: Props = $props();
  const buttons = $derived<readonly ExportButton[]>(
    formats ?? [
      { format: "onnx", label: texts.onnx, help: texts.onnxHelp },
      { format: "python_package", label: texts.package, help: texts.packageHelp },
    ],
  );
  const columns = $derived(buttons.length + (texts.code ? 1 : 0));

  let showCode = $state(false);
  let saving = $state<ExportFormat | null>(null);
  let result = $state<{ severity: "success" | "error"; markdown: string } | null>(null);

  /** Python shows the code at once (`st.code`): the "Ver código" expander opens with it. */
  function toggleCode(): void {
    showCode = !showCode;
    if (showCode) {
      session.setExpander(`${idPrefix}-codigo-export`, true);
    }
  }

  async function save(format: ExportFormat): Promise<void> {
    saving = format;
    result = null;
    try {
      const saved = await backend.exportModel(modelId, format);
      if (saved) {
        result = { severity: "success", markdown: fmt(texts.saved, { archivo: saved }) };
        status.flash(`Guardado: ${saved}`);
      }
    } catch (error) {
      if (backend.isStale(error)) {
        onstale();
      } else {
        result = { severity: "error", markdown: fmt(texts.error, { error: backend.errorMessage(error) }) };
      }
    } finally {
      saving = null;
    }
  }
</script>

<section class="export" data-export-panel>
  {#if texts.intro}<Markdown source={texts.intro} />{/if}
  <div class="buttons" style:grid-template-columns="repeat({columns}, minmax(0, 1fr))">
    {#if texts.code}
      <button type="button" class="btn block" class:primary={showCode} title={texts.codeHelp} onclick={toggleCode}>
        {texts.code}
      </button>
    {/if}
    {#each buttons as button (button.format)}
      <button
        type="button"
        class="btn block"
        title={button.help}
        disabled={saving !== null}
        onclick={() => void save(button.format)}
        data-export-format={button.format}
      >
        {button.label}
      </button>
    {/each}
  </div>
  {#if result}
    <Notice severity={result.severity} markdown={result.markdown} />
  {/if}
  {#if showCode && texts.code}
    <ModelCode {modelId} template={codeTemplate} title={texts.codeTitle} downloadName={texts.codeFile} id="{idPrefix}-codigo-export" {onstale} />
  {/if}
  {#if texts.helpExpander && texts.helpMarkdown}
    <Expander id="{idPrefix}-export-ayuda" label={texts.helpExpander}>
      <Markdown source={texts.helpMarkdown} />
    </Expander>
  {/if}
</section>

<style>
  .buttons {
    display: grid;
    gap: 16px;
    margin: 14px 0;
  }
</style>
