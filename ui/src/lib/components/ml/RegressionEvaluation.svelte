<script lang="ts">
  /**
   * C2: evaluation of a regressor: MSE, RMSE, MAE and R², predictions against real values
   * coloured by the absolute error and the distribution of the errors with its KDE.
   * Shared by Regresión lineal, Árboles, KNN and Redes (regression).
   */
  import type { RegressionEvaluation } from "../../bindings/RegressionEvaluation";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import { compText, md } from "../../content";
  import { fixed } from "../../format";
  import { theme } from "../../theme.svelte";
  import { errorHistogramFigure, predVsRealFigure } from "./figures";
  import Interpretation from "./Interpretation.svelte";
  import MetricGrid from "./MetricGrid.svelte";
  import ModelCode from "./ModelCode.svelte";

  interface Props {
    evaluation: RegressionEvaluation;
    modelId: number;
    idPrefix: string;
    onstale?: () => void;
  }

  let { evaluation, modelId, idPrefix, onstale }: Props = $props();

  const t = compText;
  const metrics = $derived([
    { label: t("regresion", "mse"), value: fixed(evaluation.mse, 4), help: t("regresion", "mse_help") },
    { label: t("regresion", "rmse"), value: fixed(evaluation.rmse, 4), help: t("regresion", "rmse_help") },
    { label: t("regresion", "mae"), value: fixed(evaluation.mae, 4), help: t("regresion", "mae_help") },
    { label: t("regresion", "r2"), value: fixed(evaluation.r2, 4), help: t("regresion", "r2_help") },
  ]);
  const scatter = $derived(predVsRealFigure(evaluation.y_true, evaluation.y_pred, { colorByError: true }, theme.chart));
  const errors = $derived(errorHistogramFigure(evaluation.errors_hist, evaluation.errors_kde, theme.chart));
</script>

<section class="regression" data-evaluation="regression">
  <MetricGrid items={metrics} />

  <div class="columns">
    <div>
      <PlotlyChart
        data={scatter.data}
        layout={scatter.layout}
        height={460}
        downloadName="predicciones_vs_reales.png"
        downloadLabel={t("regresion", "descarga")}
        label={t("regresion", "pred_title")}
      />
      <ModelCode
        {modelId}
        template="pred_vs_real"
        title={t("regresion", "codigo")}
        downloadName="predicciones_vs_reales.py"
        id="{idPrefix}-codigo-pred-real"
        {...onstale ? { onstale } : {}}
      />
    </div>
    <div>
      <PlotlyChart
        data={errors.data}
        layout={errors.layout}
        height={460}
        downloadName="distribucion_errores.png"
        downloadLabel={t("regresion", "descarga")}
        label={t("regresion", "errores_title")}
      />
      <ModelCode
        {modelId}
        template="error_distribution"
        title={t("regresion", "codigo")}
        downloadName="distribucion_errores.py"
        id="{idPrefix}-codigo-errores"
        {...onstale ? { onstale } : {}}
      />
    </div>
  </div>

  <Expander id="{idPrefix}-explicacion-regresion" label={t("regresion", "metricas_expander")}>
    <Markdown source={md("componentes/metricas-regresion.md")} />
  </Expander>

  <Interpretation markdown={evaluation.interpretation_md} />
</section>

<style>
  .columns {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
    gap: 28px;
    align-items: start;
  }
</style>
