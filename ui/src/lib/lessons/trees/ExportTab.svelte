<script lang="ts">
  /**
   * 💾 Exportar (§2.7, D43/D44): ONNX, Python package (rebuilds `decision_tree_model.pkl`),
   * the rules as text and the tree as a Python function (corrected, D44), all saved with
   * the native dialog; the preview of that function and the "Código para Usar el Modelo".
   */
  import CodeBlock from "../../../components/CodeBlock.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import ExportPanel, { type ExportButton } from "../../components/ml/ExportPanel.svelte";
  import ModelCode from "../../components/ml/ModelCode.svelte";
  import { arbText, md, regText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fmt } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import TreeBanner from "./TreeBanner.svelte";
  import { LESSON } from "./state.svelte";

  const t = (key: string): string => arbText("exportar", key);
  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const info = $derived(lessonState.info);

  const formats: readonly ExportButton[] = [
    { format: "onnx", label: t("btn_onnx"), help: t("btn_onnx_help") },
    { format: "python_package", label: t("btn_paquete"), help: t("btn_paquete_help") },
    { format: "tree_rules", label: t("btn_reglas"), help: t("btn_reglas_help") },
    { format: "tree_python", label: t("btn_codigo"), help: t("btn_codigo_help") },
  ];
  const texts = {
    code: t("btn_entrenamiento"),
    codeHelp: t("btn_entrenamiento_help"),
    onnx: t("btn_onnx"),
    package: t("btn_paquete"),
    codeTitle: t("entrenamiento_titulo"),
    codeFile: t("archivo_entrenamiento"),
    saved: regText("exportar", "exito_guardado"),
    error: t("error_guardado"),
    helpExpander: t("ayuda_expander"),
    helpMarkdown: md("arboles/exportar.md"),
  };

  const preview = new ViewLoader<string>();
  $effect(() => {
    const current = info;
    if (!current) {
      return;
    }
    return preview.load(String(current.model_id), () => backend.generateCode(current.model_id, "tree_python"), stale);
  });
  const code = $derived(info ? preview.for(String(info.model_id)) : null);

  const summary = $derived.by(() => {
    if (!info?.tree) {
      return null;
    }
    const names = info.feature_names;
    const first = names.slice(0, 3).join(", ") + (names.length > 3 ? "..." : "");
    return fmt(t("info_codigo"), {
      nodos: info.tree.node_count,
      hojas: info.tree.n_leaves,
      profundidad: info.tree.max_depth,
      n: info.tree.n_features,
      primeras: first,
      tarea: info.model_task === "classification" ? "Clasificación" : "Regresión",
    });
  });
</script>

<h2>{t("header")}</h2>
{#if !info}
  <Notice severity="warning" markdown={arbText("comun", "sin_entrenar")} />
{:else}
  <TreeBanner {info} />
  <Markdown source={t("formatos_titulo")} />
  <ExportPanel modelId={info.model_id} {texts} {formats} codeTemplate="tree_training" idPrefix="arboles" onstale={stale} />
  <p class="caption"><Markdown inline source={t("paquete_ayuda")} /></p>

  <Markdown source={t("vista_previa_titulo")} />
  <Expander id="arboles-vista-previa" label={t("vista_previa_expander")}>
    {#if preview.error}
      <Notice severity="error" markdown={escapeMarkdown(preview.error)} />
    {:else if code !== null}
      <CodeBlock {code} />
      {#if summary}<Notice severity="info" markdown={summary} />{/if}
      <Expander id="arboles-umbrales-float32" label={t("umbrales_expander")}>
        <Markdown source={md("arboles/umbrales-float32.md")} />
      </Expander>
    {:else}
      <Spinner label="Generando el código…" />
    {/if}
  </Expander>

  <ModelCode modelId={info.model_id} template="tree_usage" title={t("codigo_uso")} downloadName="usar_modelo.py" id="arboles-codigo-uso" onstale={stale} />
{/if}

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .caption {
    margin: -4px 2px 16px;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
  }
</style>
