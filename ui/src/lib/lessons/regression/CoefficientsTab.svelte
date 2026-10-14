<script lang="ts">
  /**
   * 🔍 Coeficientes (§2.5): table, intercept, chart, detailed analysis and recommendations,
   * with the class selector in multiclass logistic. D27 applied (intercept of the chosen
   * class, numbering 1-2-3, symmetric strength).
   */
  import DataTable from "../../../components/DataTable.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Metric from "../../../components/Metric.svelte";
  import Notice from "../../../components/Notice.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import Select from "../../../components/Select.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { Cell } from "../../bindings/Cell";
  import type { CoefficientRow } from "../../bindings/CoefficientRow";
  import type { CoefficientsView } from "../../bindings/CoefficientsView";
  import type { Level } from "../../bindings/Level";
  import { coefficientFigure, coefficientHeight } from "../../components/ml/figures";
  import { md, regList, regText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fmt, toFixedPython, type FmtValue } from "../../fmt";
  import { fixed } from "../../format";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import { theme } from "../../theme.svelte";
  import ModelBanner from "./ModelBanner.svelte";
  import { LESSON } from "./state.svelte";

  const t = (key: string): string => regText("coeficientes", key);
  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const view = new ViewLoader<CoefficientsView>();

  const info = $derived(lessonState.info);
  const logistic = $derived(info?.config.kind === "logistic");
  const classes = $derived(info?.class_names ?? []);
  const multiclass = $derived(logistic && classes.length > 2);
  const classIndex = $derived(multiclass ? Math.min(lessonState.coefClass, classes.length - 1) : null);
  const key = $derived(info ? `${info.model_id}|${classIndex ?? ""}` : "");

  $effect(() => {
    const current = info;
    if (!current) {
      return;
    }
    const cls = classIndex;
    return view.load(key, () => backend.coefficientsView(current.model_id, cls), stale);
  });

  const data = $derived(view.for(key) ?? (view.loading ? view.data : null));
  /** The class the odds refer to: the chosen one (multiclass) or the positive one (binary). */
  const targetClass = $derived(multiclass ? (classes[classIndex ?? 0] ?? "") : (classes[1] ?? "1"));

  const text = (v: string): Cell => ({ kind: "text", v });
  const effectText = (r: CoefficientRow): string => t(r.effect === "positive" ? "efecto_positivo" : "efecto_negativo");
  const IMPORTANCE: Record<Level, string> = { very_high: "importancia_alta", high: "importancia_alta", medium: "importancia_media", low: "importancia_baja" };
  const STRENGTH: Record<Level, string> = { very_high: "fuerza_muy_fuerte", high: "fuerza_fuerte", medium: "fuerza_moderado", low: "fuerza_debil" };
  const INTENSITY: Record<Level, string> = { very_high: "intensidad_fuerte", high: "intensidad_fuerte", medium: "intensidad_moderado", low: "intensidad_debil" };

  const table = $derived.by(() => {
    if (!data) {
      return { columns: [] as readonly string[], rows: [] as Cell[][] };
    }
    if (logistic) {
      return {
        columns: regList("coeficientes", "columnas_logistica"),
        rows: data.rows.map((r) => [
          text(r.feature),
          text(fixed(r.coef, 4)),
          text(fixed(r.odds_ratio ?? Number.NaN, 3)),
          text(`${(r.change_pct ?? 0) >= 0 ? "+" : ""}${toFixedPython(r.change_pct ?? 0, 1)}%`),
          text(effectText(r)),
          text(t(STRENGTH[r.strength ?? "low"])),
        ]),
      };
    }
    return {
      columns: regList("coeficientes", "columnas_lineal"),
      rows: data.rows.map((r) => [text(r.feature), text(fixed(r.coef, 4)), text(effectText(r)), text(t(IMPORTANCE[r.importance ?? "low"]))]),
    };
  });

  const figure = $derived(data ? coefficientFigure(data.rows, theme.chart) : null);
  const odds = $derived(data ? Math.exp(data.intercept) : 0);

  const interceptInfo = $derived.by(() => {
    if (!data) {
      return "";
    }
    if (!logistic) {
      return fmt(t("intercepto_info_lineal"), { intercepto: data.intercept });
    }
    if (multiclass) {
      return fmt(t("intercepto_info_multiclase"), { clase: targetClass, intercepto: data.intercept });
    }
    return fmt(t("intercepto_info_logistica"), { clase: targetClass, intercepto: data.intercept, odds, prob: odds / (1 + odds) });
  });

  /** `{caracteristica}` in backticks: a code span would show the escapes, so use `<code>`. */
  function render(template: string, values: Readonly<Record<string, FmtValue>>): string {
    return fmt(template.replace(/`\{caracteristica\}`/g, "<code>{caracteristica}</code>"), values);
  }

  function item(r: CoefficientRow, n: number): string {
    const values = {
      n,
      caracteristica: r.feature,
      // Our own Markdown ("🔥 **FUERTE**"): inserted after `fmt`, which escapes values.
      intensidad: INTENSITY_SLOT,
      coef: r.coef,
      abs_coef: r.abs,
      odds: r.odds_ratio ?? 0,
      cambio: r.change_pct ?? 0,
      reduccion: (1 - (r.odds_ratio ?? 0)) * 100,
    };
    const kind = logistic ? "log" : "lin";
    return render(t(`${kind}_item_${r.coef > 0 ? "pos" : "neg"}`), values).replace(INTENSITY_SLOT, t(INTENSITY[r.intensity]));
  }

  const INTENSITY_SLOT = "\u0002";

  const positives = $derived(data ? data.rows.filter((r) => r.coef > 0).slice(0, 3) : []);
  const negatives = $derived(data ? data.rows.filter((r) => r.coef <= 0).slice(0, 3) : []);
  const top = $derived(data?.rows[0] ?? null);
  const summary = $derived.by(() => {
    if (!top) {
      return "";
    }
    const kind = logistic ? "log" : "lin";
    const impact = fmt(t(`${kind}_impacto_${top.coef > 0 ? "pos" : "neg"}`), {
      odds: top.odds_ratio ?? 0,
      reduccion: (1 - (top.odds_ratio ?? 0)) * 100,
      coef: top.coef,
      abs_coef: top.abs,
    });
    return render(t(`${kind}_resumen`), { caracteristica: top.feature }).replace("{impacto}", impact);
  });
  const balance = $derived.by(() => {
    if (!data || !logistic) {
      return null;
    }
    const nPos = data.rows.filter((r) => r.coef > 0).length;
    const nNeg = data.rows.length - nPos;
    if (nPos === 0) {
      return { severity: "warning" as const, markdown: t("log_todas_negativas") };
    }
    if (nNeg === 0) {
      return { severity: "warning" as const, markdown: t("log_todas_positivas") };
    }
    return { severity: "info" as const, markdown: fmt(t("log_balance"), { n_pos: nPos, n_neg: nNeg }) };
  });

  const OBJECTIVE = "\u0001";
  const recommendations = $derived.by(() => {
    if (!data) {
      return "";
    }
    const objective = logistic ? fmt(t("objetivo_logistica"), { clase: targetClass }) : t("objetivo_lineal");
    return data.rows
      .slice(0, 3)
      .map((r, i) => {
        const line = fmt(t("recomendacion"), {
          caracteristica: r.feature,
          accion: r.coef > 0 ? "Aumentar" : "Reducir",
          verbo: r.coef > 0 ? "incrementar" : "decrementar",
          objetivo: OBJECTIVE,
        }).replace(OBJECTIVE, objective);
        return `${i + 1}. ${line}`;
      })
      .join("\n");
  });
</script>

<h2>{t("header")}</h2>
{#if !info}
  <Notice severity="info" markdown={regText("comun", "sin_entrenar_coeficientes")} />
{:else}
  <ModelBanner {info} />
  <Expander id="regresion-coef-interpretar" label={t("expander")}>
    <Markdown source={md(logistic ? "regresion/coeficientes-logistica.md" : "regresion/coeficientes-lineal.md")} />
  </Expander>
  {#if multiclass}
    <Notice severity="info" markdown={fmt(t("nota_multiclase"), { n_clases: classes.length, clase: targetClass })} />
    <Select
      id="regresion-coef-clase"
      label={t("selector_clase")}
      options={classes.map((label, value) => ({ label, value }))}
      value={classIndex ?? 0}
      onchange={(value) => (lessonState.coefClass = value)}
    />
  {/if}
  {#if view.error}
    <Notice severity="error" markdown={escapeMarkdown(view.error)} />
  {:else if !data || !figure}
    <Spinner label="Calculando los coeficientes…" />
  {:else}
    <div class:updating={view.loading} class="body">
      <Markdown source={t("tabla_titulo")} />
      <DataTable label="Tabla de coeficientes" columns={table.columns} rows={table.rows} hideIndex numericColumns={logistic ? [1, 2, 3] : [1]} />
      <Expander id="regresion-coef-tabla" label={t("tabla_expander")}>
        <Markdown source={md(logistic ? "regresion/tabla-logistica.md" : "regresion/tabla-lineal.md")} />
      </Expander>

      <Markdown source={t("intercepto_titulo")} />
      <div class="metrics" class:two={logistic && !multiclass}>
        <Metric label={t("intercepto_metric")} value={fixed(data.intercept, 4)} />
        {#if logistic && !multiclass}
          <Metric label={t("odds_base_metric")} value={fixed(odds, 3)} />
        {/if}
      </div>
      <Notice severity="info" markdown={interceptInfo} />

      <Markdown source={t("grafico_titulo")} />
      <div class="chart">
        <PlotlyChart
          data={figure.data}
          layout={figure.layout}
          height={coefficientHeight(data.rows.length)}
          downloadName="coeficientes.png"
          label={t("grafico_title")}
        />
      </div>

      <Markdown source={t("analisis_titulo")} />
      {#if multiclass}
        <Notice severity="info" markdown={fmt(t("nota_odds_multiclase"), { clase: targetClass })} />
      {/if}
      <Markdown source={t(logistic ? "log_aumentan" : "lin_aumentan")} />
      {#if positives.length > 0}
        {#each positives as row, i (row.feature_position)}
          <div class="item"><Markdown source={item(row, i + 1)} /></div>
        {/each}
      {:else}
        <Notice severity="info" markdown={t(logistic ? "log_sin_pos" : "lin_sin_pos")} />
      {/if}
      <Markdown source={t(logistic ? "log_disminuyen" : "lin_disminuyen")} />
      {#if negatives.length > 0}
        {#each negatives as row, i (row.feature_position)}
          <div class="item"><Markdown source={item(row, i + 1)} /></div>
        {/each}
      {:else}
        <Notice severity="info" markdown={t(logistic ? "log_sin_neg" : "lin_sin_neg")} />
      {/if}

      <Markdown source={t("resumen_titulo")} />
      <Notice severity="success" markdown={summary} />
      {#if balance}
        <Notice severity={balance.severity} markdown={balance.markdown} />
      {/if}

      <Markdown source={t("recomendaciones_titulo")} />
      <div class="item"><Markdown source={recommendations} /></div>
      <Notice severity="info" markdown={t("recomendaciones_nota")} />

      {#if data.extreme.length > 0}
        <Notice severity="warning" markdown={fmt(t("extremos"), { lista: data.extreme.join(", ") })} />
      {/if}
    </div>
  {/if}
{/if}

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .metrics {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 16px;
    max-width: 420px;
    margin: 10px 0;
  }

  .metrics.two {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    max-width: none;
  }

  .chart {
    max-width: 900px;
    margin: 0 auto;
  }

  .item {
    margin: 6px 8px 14px;
  }

  .body {
    transition: opacity 120ms ease;
  }

  .body.updating {
    opacity: 0.55;
  }
</style>
