<script lang="ts">
  /** 💾 Exportar (§2.7, D28/D29): code, ONNX and Python package. */
  import Notice from "../../../components/Notice.svelte";
  import ExportPanel from "../../components/ml/ExportPanel.svelte";
  import { md, regText } from "../../content";
  import { models } from "../../ml/modelSession.svelte";
  import ModelBanner from "./ModelBanner.svelte";
  import { LESSON } from "./state.svelte";

  const t = (key: string): string => regText("exportar", key);
  const lessonState = models.lesson(LESSON);
  const info = $derived(lessonState.info);
  const texts = $derived({
    intro: t("intro"),
    code: t("btn_codigo"),
    codeHelp: t("btn_codigo_help"),
    onnx: t("btn_onnx"),
    onnxHelp: t("btn_onnx_help"),
    package: t("btn_paquete"),
    packageHelp: t("btn_paquete_help"),
    codeTitle: t("codigo_titulo"),
    codeFile: t(info?.config.kind === "logistic" ? "codigo_archivo_logistica" : "codigo_archivo_lineal"),
    saved: t("exito_guardado"),
    error: t("error_guardado"),
    helpExpander: t("ayuda_expander"),
    helpMarkdown: md("regresion/exportar.md"),
  });
</script>

<h2>{t("header")}</h2>
{#if !info}
  <Notice severity="info" markdown={regText("comun", "sin_entrenar_exportar")} />
{:else}
  <ModelBanner {info} />
  <ExportPanel modelId={info.model_id} {texts} idPrefix="regresion" onstale={() => void models.refresh(LESSON)} />
{/if}

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }
</style>
