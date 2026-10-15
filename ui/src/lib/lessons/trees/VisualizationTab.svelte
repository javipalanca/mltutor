<script lang="ts">
  /**
   * 📉 Visualización (§2.4): 🌲 Árbol and 📝 Texto, plus 🌈 Frontera / 📉 Curva ROC
   * (classification) or 📈 Residuos / 🌐 Superficie (regression), by the task of the model.
   */
  import ButtonPanel from "../../../components/ButtonPanel.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Slider from "../../../components/Slider.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { ResidualsView } from "../../bindings/ResidualsView";
  import type { RocData } from "../../bindings/RocData";
  import type { TreeDiagram as Diagram } from "../../bindings/TreeDiagram";
  import DecisionGridView from "../../components/ml/DecisionGridView.svelte";
  import ModelCode from "../../components/ml/ModelCode.svelte";
  import ResidualsAnalysis from "../../components/ml/ResidualsAnalysis.svelte";
  import RocPrCharts from "../../components/ml/RocPrCharts.svelte";
  import TreeDiagram from "../../components/ml/TreeDiagram.svelte";
  import TreeRulesText from "../../components/ml/TreeRulesText.svelte";
  import Expander from "../../../components/Expander.svelte";
  import { arbRaw, arbText, md } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fmt } from "../../fmt";
  import { classOfRow } from "../../ml/modelClasses";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import TreeBanner from "./TreeBanner.svelte";
  import { LESSON, trees } from "./state.svelte";

  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const t = (key: string): string => arbText("visualizacion", key);

  type Panel = readonly (readonly [string, string, string])[];
  const panel = (key: string): Panel => (arbRaw("visualizacion", key) as Panel | undefined) ?? [];

  const classifier = $derived(lessonState.info?.model_task === "classification");
  const options = $derived(
    [...panel("comunes"), ...panel(classifier ? "clasificacion" : "regresion")].map(([label, value]) => ({ label, value })),
  );
  // `create_button_panel`: the first option if the stored one is not in this panel.
  const viz = $derived(options.some((o) => o.value === lessonState.viz) ? (lessonState.viz ?? "") : (options[0]?.value ?? ""));

  const diagram = new ViewLoader<Diagram>();
  const rules = new ViewLoader<string>();
  const roc = new ViewLoader<RocData>();
  const residuals = new ViewLoader<ResidualsView>();

  $effect(() => {
    const info = lessonState.info;
    if (!info) {
      return;
    }
    const id = info.model_id;
    const key = String(id);
    switch (viz) {
      case "Árbol":
        return diagram.load(key, () => backend.treeDiagram(id), stale);
      case "Texto":
        return rules.load(key, () => backend.treeRules(id, "view", true), stale);
      case "ROC":
        return roc.load(key, () => backend.rocCurves(id), stale);
      case "Residuos":
        return residuals.load(key, () => backend.residualsView(id), stale);
      default:
        return undefined;
    }
  });

  const key = $derived(String(lessonState.info?.model_id ?? ""));
</script>

<h2>{t("header")}</h2>
{#if !lessonState.info}
  <Notice severity="warning" markdown={arbText("comun", "sin_entrenar")} />
{:else}
  {@const info = lessonState.info}
  <TreeBanner {info} />
  <Markdown source={t("tipo_titulo")} />
  <ButtonPanel label="Tipo de visualización" value={viz} {options} onchange={(value) => (lessonState.viz = value)} />
  {#if viz === "Árbol"}
    {@const data = diagram.for(key)}
    <div class="columns">
      <Slider id="arboles-ancho" label={t("ancho_label")} min={8} max={20} step={1} value={trees.figWidth} onchange={(v) => (trees.figWidth = Math.round(v))} />
      <Slider id="arboles-alto" label={t("alto_label")} min={6} max={15} step={1} value={trees.figHeight} onchange={(v) => (trees.figHeight = Math.round(v))} />
    </div>
    {#if diagram.error}
      <Notice severity="error" markdown={escapeMarkdown(diagram.error)} />
    {:else if data}
      <TreeDiagram
        diagram={data}
        widthIn={trees.figWidth}
        heightIn={trees.figHeight}
        pngLabel={t("descarga_arbol")}
        svgLabel={t("descarga_arbol_svg")}
        help={t("arbol_ayuda")}
      />
      <Expander id="arboles-leer-arbol" label={t("leer_arbol_expander")}>
        <Markdown source={md("arboles/leer-arbol.md")} />
      </Expander>
      <ModelCode modelId={info.model_id} template="tree_viz" title={t("codigo_arbol")} downloadName="visualizar_arbol.py" id="arboles-codigo-arbol" onstale={stale} />
    {:else}
      <Spinner label="Dibujando el árbol…" />
    {/if}
  {:else if viz === "Texto"}
    {@const data = rules.for(key)}
    {#if rules.error}
      <Notice severity="error" markdown={escapeMarkdown(rules.error)} />
    {:else if data !== null}
      <p class="caption"><Markdown inline source={t("texto_nota")} /></p>
      <TreeRulesText text={data} downloadName="arbol_texto.txt" downloadLabel={t("descarga_texto")} />
      <ModelCode modelId={info.model_id} template="tree_text" title={t("codigo_texto")} downloadName="texto_arbol.py" id="arboles-codigo-texto" onstale={stale} />
    {:else}
      <Spinner label="Generando el texto del árbol…" />
    {/if}
  {:else if viz === "Frontera" || viz === "Superficie"}
    <DecisionGridView
      modelId={info.model_id}
      features={info.feature_names}
      pair={lessonState.pair}
      onpair={(pair) => (lessonState.pair = pair)}
      mode={viz === "Frontera" ? "boundary" : "surface"}
      idPrefix="arboles"
      onstale={stale}
    />
  {:else if viz === "ROC"}
    {@const data = roc.for(key)}
    {#if roc.error}
      <Notice severity="error" markdown={fmt(t("roc_error"), { error: roc.error })} />
      <Notice severity="info" markdown={t("roc_error_info")} />
    {:else if data}
      <RocPrCharts
        roc={data}
        showPerClass={lessonState.rocPerClass}
        averages={lessonState.rocAverages}
        onoptions={(perClass, averages) => {
          lessonState.rocPerClass = perClass;
          lessonState.rocAverages = averages;
        }}
        idPrefix="arboles"
        negativeClass={classOfRow(info, 0)}
      />
    {:else}
      <Spinner label="Calculando las curvas…" />
    {/if}
  {:else if viz === "Residuos"}
    {@const data = residuals.for(key)}
    {#if residuals.error}
      <Notice severity="error" markdown={escapeMarkdown(residuals.error)} />
    {:else if data}
      <ResidualsAnalysis view={data} idPrefix="arboles" predictionsTitle={t("residuos_titulo_pred")} residualsTitle={t("residuos_titulo")} />
    {:else}
      <Spinner label="Calculando los residuos…" />
    {/if}
  {/if}
{/if}

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .caption {
    margin: 4px 2px 0;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
  }

  .columns {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 28px;
    align-items: start;
  }
</style>
