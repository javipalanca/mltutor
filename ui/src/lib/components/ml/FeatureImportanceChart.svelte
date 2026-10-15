<script lang="ts">
  /**
   * C10: importance of the features (`display_feature_importance`), native or by
   * permutation: bars (viridis by position), "### Valores Detallados" and the expander
   * "📖 Cómo interpretar estos resultados". Generic: 🏘️ KNN (F4) uses the permutation.
   */
  import DataTable from "../../../components/DataTable.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import type { Cell } from "../../bindings/Cell";
  import type { FeatureImportanceView } from "../../bindings/FeatureImportanceView";
  import { compList, compText } from "../../content";
  import { fmt } from "../../fmt";
  import { theme } from "../../theme.svelte";
  import { importanceFigure, importanceHeight, importanceTable, methodName } from "./importance";

  interface Props {
    view: FeatureImportanceView;
    idPrefix: string;
    /** Text of the method in the expander, if the lesson has its own (🔍 KNN). */
    permutationText?: string;
  }

  let { view, idPrefix, permutationText }: Props = $props();

  const t = (key: string): string => compText("importancia", key);
  const figure = $derived(importanceFigure(view, theme.chart));
  const table = $derived(importanceTable(view));
  const permutation = $derived(view.method === "permutation");
  const columns = $derived(permutation ? [...compList("importancia", "columnas"), "Desv. Estándar"] : compList("importancia", "columnas"));
  const text = (v: string): Cell => ({ kind: "text", v });
  const explanation = $derived(
    permutation ? (permutationText ?? t("permutacion")) : t(view.task === "classification" ? "nativa_clasificacion" : "nativa_regresion"),
  );
</script>

<section class="importance" data-importance={view.method}>
  <Markdown source={t("titulo")} />
  {#if permutation}
    <Notice severity="info" markdown={fmt(t("perm_explicacion"), { metodo: methodName(view), descripcion: t("metodo_perm_desc") })} />
  {/if}
  <PlotlyChart
    data={figure.data}
    layout={figure.layout}
    height={importanceHeight(view.rows.length)}
    downloadName="importancia_caracteristicas.png"
    label="Importancia de las características"
  />
  <Markdown source={t("valores_titulo")} />
  <DataTable
    label="Importancia de las características"
    {columns}
    rows={table.rows.map((r) => r.filter((v): v is string => v !== undefined).map(text))}
    index={table.index}
    numericColumns={permutation ? [1, 2] : [1]}
  />
  <Expander id="{idPrefix}-importancia-{view.method}" label={t("expander")}>
    <Markdown source={explanation} />
    <Markdown source={t("consejos")} />
  </Expander>
</section>
