<script lang="ts">
  /**
   * 💾 Exportar (§2.7, D57): "📄 Código de Entrenamiento" (`knn_entrenamiento.py.j2`),
   * "💾 Exportar a ONNX" and "📦 Paquete Python (.zip)", saved with the native dialog.
   * Python only had the pickle and the snippet of Regresión.
   */
  import Notice from "../../../components/Notice.svelte";
  import ExportPanel, { type ExportButton } from "../../components/ml/ExportPanel.svelte";
  import { hasMd, knnText, md, regText } from "../../content";
  import { models } from "../../ml/modelSession.svelte";
  import KnnBanner from "./KnnBanner.svelte";
  import { LESSON } from "./state.svelte";

  const t = (key: string): string => knnText("exportar", key);
  const lessonState = models.lesson(LESSON);
  const info = $derived(lessonState.info);

  const formats: readonly ExportButton[] = [
    { format: "onnx", label: t("btn_onnx"), help: t("btn_onnx_help") },
    { format: "python_package", label: t("btn_paquete"), help: t("btn_paquete_help") },
  ];
  const help = hasMd("knn/exportar.md") ? { helpExpander: t("ayuda_expander"), helpMarkdown: md("knn/exportar.md") } : {};
  const texts = {
    intro: t("intro"),
    code: t("btn_entrenamiento"),
    codeHelp: t("btn_entrenamiento_help"),
    onnx: t("btn_onnx"),
    package: t("btn_paquete"),
    codeTitle: t("entrenamiento_titulo"),
    codeFile: t("archivo_entrenamiento"),
    saved: regText("exportar", "exito_guardado"),
    error: regText("exportar", "error_guardado"),
    ...help,
  };
</script>

<h2>{t("header")}</h2>
{#if !info}
  <Notice severity="info" markdown={knnText("comun", "sin_entrenar")} />
{:else}
  <KnnBanner {info} />
  <ExportPanel modelId={info.model_id} {texts} {formats} codeTemplate="knn_training" idPrefix="knn" onstale={() => void models.refresh(LESSON)} />
{/if}

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }
</style>
