<script lang="ts">
  /**
   * C7 (`viz/residual.py`): predictions against real values with the R² of the test set
   * (D22) and the residual analysis (residuals vs predictions, histogram, three checks).
   * Shared by Regresión lineal, KNN and Árboles (regression).
   */
  import type { ResidualsView } from "../../bindings/ResidualsView";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import { compRaw, compText, md, optionalText } from "../../content";
  import { fmtText } from "../../fmt";
  import { theme } from "../../theme.svelte";
  import { predVsRealFigure, residualsFigure } from "./figures";

  interface Props {
    view: ResidualsView;
    idPrefix: string;
    /** Headings of the lesson ("### 📊 Gráfico de Predicciones vs Valores Reales"…). */
    predictionsTitle?: string;
    residualsTitle?: string;
  }

  let { view, idPrefix, predictionsTitle, residualsTitle }: Props = $props();

  const t = compText;
  const scatter = $derived(predVsRealFigure(view.y_true, view.y_pred, { r2: view.r2 }, theme.chart));
  const residuals = $derived(residualsFigure(view, theme.chart));
  const checks = $derived.by(() => {
    const items: string[] = [t("residuos", view.checks.mean_ok ? "media_ok" : "media_mal")];
    if (view.checks.corr_ok === null) {
      items.push(optionalText(compRaw, "residuos", "corr_pocas_muestras", ""));
    } else {
      items.push(t("residuos", view.checks.corr_ok ? "corr_ok" : "corr_mal"));
    }
    items.push(t("residuos", view.checks.normal_ok ? "normal_ok" : "normal_mal"));
    return items.filter(Boolean);
  });
  const failed = $derived(!view.checks.mean_ok || view.checks.corr_ok === false || !view.checks.normal_ok);
  // The bullets are our own Markdown (no data values), so `fmtText` does not escape them.
  const bullets = $derived(checks.map((texto) => fmtText(t("residuos", "item"), { texto })).join("\n"));
</script>

<section class="residuals" data-residuals>
  {#if predictionsTitle}<Markdown source={predictionsTitle} />{/if}
  <Expander id="{idPrefix}-residuos-pred" label={t("residuos", "expander_pred")}>
    <Markdown source={md("componentes/predicciones-vs-reales.md")} />
  </Expander>
  <div class="chart">
    <PlotlyChart data={scatter.data} layout={scatter.layout} height={540} downloadName="predicciones_vs_reales.png" label={t("residuos", "pred_title")} />
  </div>

  {#if residualsTitle}<Markdown source={residualsTitle} />{/if}
  <Expander id="{idPrefix}-residuos-analisis" label={t("residuos", "expander_residuos")}>
    <Markdown source={md("componentes/residuos-interpretar.md")} />
  </Expander>
  <PlotlyChart data={residuals.data} layout={residuals.layout} height={460} downloadName="analisis_residuos.png" label={t("residuos", "res_title")} />

  <Markdown source={t("residuos", "interp_titulo")} />
  <div class="checks"><Markdown source={bullets} /></div>
  {#if failed}
    <Notice severity="info" markdown={t("residuos", "sugerencias")} />
  {/if}
</section>

<style>
  .chart {
    max-width: 900px;
    margin: 0 auto;
  }

  .checks {
    margin: 4px 8px 12px;
  }
</style>
