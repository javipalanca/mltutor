<script lang="ts">
  /**
   * C4: ROC and precision-recall curves (`viz/roc.py::plot_roc_curve`). Binary: both
   * curves and the interpretation by AUC/AP. Multiclass: options (curves per class,
   * macro/micro averages; filtered here, without asking the backend again), curves,
   * interpretation per average and the per-class table. Shared by logística, Árboles, KNN.
   */
  import type { Cell } from "../../bindings/Cell";
  import type { RocData } from "../../bindings/RocData";
  import Checkbox from "../../../components/Checkbox.svelte";
  import DataTable from "../../../components/DataTable.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import MultiSelect from "../../../components/MultiSelect.svelte";
  import Notice from "../../../components/Notice.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import { compList, compRaw, compText, md, optionalText } from "../../content";
  import { fmt } from "../../fmt";
  import { fixed } from "../../format";
  import { pickLevel } from "../../ml/levels";
  import { theme } from "../../theme.svelte";
  import { rocPrFigure, type Average } from "./figures";

  interface Props {
    roc: RocData;
    showPerClass: boolean;
    averages: readonly Average[];
    onoptions: (showPerClass: boolean, averages: Average[]) => void;
    idPrefix: string;
    /** Name of class 0 (binary): caption "La clase positiva es …". */
    negativeClass?: string;
  }

  let { roc, showPerClass, averages, onoptions, idPrefix, negativeClass }: Props = $props();

  const t = compText;
  const AVERAGES: readonly Average[] = ["macro", "micro"];
  const effective = $derived<Average[]>(averages.length > 0 ? [...averages] : ["macro"]);
  const figure = $derived(rocPrFigure(roc, { showPerClass, averages: effective }, theme.chart));
  const binaryAuc = $derived(roc.kind === "binary" ? pickLevel(compRaw("roc", "binaria_auc"), roc.auc, { auc: roc.auc }) : null);
  const binaryAp = $derived(
    roc.kind === "binary" ? pickLevel(compRaw("roc", "binaria_ap"), roc.ap, { ap: roc.ap }, { baseline: roc.baseline }) : null,
  );
  const text = (v: string): Cell => ({ kind: "text", v });
  const title = (method: Average): string => fmt(t("roc", "promedio_titulo"), { Metodo: method === "macro" ? "Macro" : "Micro" });
</script>

<section class="roc" data-roc={roc.kind}>
  {#if roc.kind === "binary"}
    <Markdown source={t("roc", "titulo_binaria")} />
    {#if negativeClass !== undefined}
      <p class="caption"><Markdown inline source={fmt(optionalText(compRaw, "roc", "clase_positiva", ""), { clase: roc.positive_class, clase_negativa: negativeClass })} /></p>
    {/if}
    <div class="chart">
      <PlotlyChart data={figure.data} layout={figure.layout} height={470} downloadName="curvas_roc_pr.png" label="Curvas ROC y Precision-Recall" />
    </div>
    <Markdown source={t("roc", "interp_titulo_binaria")} />
    <div class="columns">
      <div>
        <Markdown source={t("roc", "interp_roc")} />
        {#if binaryAuc}<Notice severity={binaryAuc.severity} markdown={binaryAuc.markdown} />{/if}
      </div>
      <div>
        <Markdown source={t("roc", "interp_pr")} />
        {#if binaryAp}<Notice severity={binaryAp.severity} markdown={binaryAp.markdown} />{/if}
      </div>
    </div>
  {:else}
    <Markdown source={t("roc", "opciones_titulo")} />
    <div class="columns options">
      <Checkbox
        id="{idPrefix}-roc-por-clase"
        label={t("roc", "por_clase")}
        checked={showPerClass}
        onchange={(checked) => onoptions(checked, [...averages])}
      />
      <MultiSelect
        label={t("roc", "promediado")}
        options={AVERAGES}
        value={averages}
        onchange={(value) => onoptions(showPerClass, value.filter((v): v is Average => v === "macro" || v === "micro"))}
      />
    </div>
    {#if roc.absent_classes.length > 0}
      <Notice
        severity="info"
        markdown={fmt(
          optionalText(compRaw, "roc", "clases_ausentes", "**Nota:** Estas clases no tienen muestras en el conjunto de prueba y no tienen curva: {clases}"),
          { clases: roc.absent_classes.join(", ") },
        )}
      />
    {/if}
    <Markdown source={t("roc", "titulo_multiclase")} />
    <div class="chart">
      <PlotlyChart data={figure.data} layout={figure.layout} height={540} downloadName="curvas_roc_pr.png" label="Curvas ROC y Precision-Recall multiclase" />
    </div>
    <Markdown source={t("roc", "interp_titulo_multiclase")} />
    {#each effective as method (method)}
      {@const auc = method === "macro" ? roc.macro_auc : roc.micro.auc}
      {@const ap = method === "macro" ? roc.macro_ap : roc.micro.ap}
      {@const aucLevel = pickLevel(compRaw("roc", "multiclase_auc"), auc, { auc })}
      {@const apLevel = pickLevel(compRaw("roc", "multiclase_ap"), ap, { ap })}
      <Markdown source={title(method)} />
      <div class="columns">
        <div>
          <Markdown source={t("roc", "interp_roc_auc")} />
          {#if aucLevel}<Notice severity={aucLevel.severity} markdown={aucLevel.markdown} />{/if}
        </div>
        <div>
          <Markdown source={t("roc", "interp_ap")} />
          {#if apLevel}<Notice severity={apLevel.severity} markdown={apLevel.markdown} />{/if}
        </div>
      </div>
    {/each}
    {#if showPerClass}
      <Markdown source={t("roc", "por_clase_titulo")} />
      <DataTable
        label={t("roc", "por_clase_titulo")}
        columns={compList("roc", "por_clase_columnas")}
        rows={roc.per_class.map((c) => [text(c.name), text(fixed(c.auc, 3)), text(fixed(c.ap, 3))])}
        index={roc.per_class.map((_, i) => String(i))}
        numericColumns={[1, 2]}
      />
    {/if}
    <Expander id="{idPrefix}-roc-macro-micro" label={t("roc", "macro_micro_expander")}>
      <Markdown source={md("componentes/roc-macro-micro.md")} />
    </Expander>
  {/if}
  <Expander id="{idPrefix}-roc-interpretar" label={t("roc", "interpretar_expander")}>
    <Markdown source={md("componentes/roc-interpretar.md")} />
  </Expander>
</section>

<style>
  .columns {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 28px;
    align-items: start;
  }

  .options {
    align-items: center;
  }

  .caption {
    margin: 0 4px 6px;
    color: var(--color-text-muted);
    font-size: 13px;
  }

  .chart {
    max-width: 1080px;
    margin: 0 auto;
  }
</style>
