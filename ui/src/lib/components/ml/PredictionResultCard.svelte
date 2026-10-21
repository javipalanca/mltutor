<script lang="ts">
  /**
   * C9: "Resultado de la predicción" card (#E8F5E9, value in #2E7D32) and its extras: the
   * probability table of a classifier or the coefficient table of a linear model.
   */
  import type { Cell } from "../../bindings/Cell";
  import type { PredictionResult } from "../../bindings/PredictionResult";
  import DataTable from "../../../components/DataTable.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import { compList, compText } from "../../content";
  import { fmtText } from "../../fmt";
  import { fixed } from "../../format";

  interface Props {
    result: PredictionResult;
    /** A second line inside the card (🧠 binary: "Probabilidad clase positiva: 0.9584"). */
    detail?: string | null;
  }

  let { result, detail = null }: Props = $props();
  const text = (v: string): Cell => ({ kind: "text", v });
</script>

<div class="card" data-prediction-result={result.kind}>
  <h3>{compText("prediccion", "resultado_titulo")}</h3>
  <p class="value">
    {#if result.kind === "class"}
      {fmtText(compText("prediccion", "resultado_clase"), { clase: result.name })}
    {:else}
      {fmtText(compText("prediccion", "resultado_valor"), { valor: result.value })}
    {/if}
  </p>
  {#if detail}
    <p class="detail" data-prediction-detail>{detail}</p>
  {/if}
</div>
{#if result.kind === "class" && result.proba}
  <DataTable
    label="Probabilidades"
    columns={compList("prediccion", "prob_columnas")}
    rows={result.proba.map((p) => [text(p.name), text(fixed(p.probability, 4))])}
    index={result.proba.map((_, i) => String(i))}
    numericColumns={[1]}
  />
{:else if result.kind === "value" && result.coefficients}
  <Markdown source={compText("prediccion", "coef_titulo")} />
  <DataTable
    label="Coeficientes"
    columns={["Característica", "Coeficiente"]}
    rows={result.coefficients.map((c) => [text(c.feature), text(fixed(c.coef, 4))])}
    hideIndex
    numericColumns={[1]}
  />
{/if}

<style>
  .card {
    margin: 14px 0;
    padding: 18px 20px;
    background: #e8f5e9;
    border-radius: 10px;
    text-align: center;
  }

  h3 {
    margin: 0 0 8px;
    color: #0d47a1;
    font-size: 17px;
  }

  .value {
    margin: 0;
    color: #2e7d32;
    font-size: 24px;
    font-weight: 700;
    user-select: text;
  }

  .detail {
    margin: 10px 0 0;
    color: var(--color-text);
    font-size: 14px;
    user-select: text;
  }

  :global(:root[data-theme="dark"]) .card {
    background: #13291b;
  }

  :global(:root[data-theme="dark"]) h3 {
    color: #8cc0ff;
  }

  :global(:root[data-theme="dark"]) .value {
    color: #7fd08a;
  }
</style>
