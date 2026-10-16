<script lang="ts">
  /**
   * 🎲 Exploración interactiva (C29, D81; unreachable in Python): a base sample of the dataset
   * and a feature; the network's response to 50 values of that feature from the minimum to
   * the maximum of the whole dataset, with the rest fixed (an ICE curve), and Python's
   * analysis (range and volatility of the probability, or range and relative impact). NEW in
   * multiclass: the analysis follows the probability of the base sample's class.
   */
  import DataTable from "../../../components/DataTable.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import NumberInput from "../../../components/NumberInput.svelte";
  import PlotlyChart from "../../../components/PlotlyChart.svelte";
  import Select from "../../../components/Select.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { Cell } from "../../bindings/Cell";
  import type { ExploreView } from "../../bindings/ExploreView";
  import type { TrainedModelInfo } from "../../bindings/TrainedModelInfo";
  import MetricGrid from "../../components/ml/MetricGrid.svelte";
  import { exploreFigure } from "../../components/ml/nnFigures";
  import { md, nnText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fmt, fmtText } from "../../fmt";
  import { fixed } from "../../format";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import { theme } from "../../theme.svelte";
  import { nnPredictions } from "./predictions.svelte";
  import { LESSON } from "./state.svelte";

  interface Props {
    info: TrainedModelInfo;
  }

  let { info }: Props = $props();
  const t = (key: string): string => nnText("exploracion", key);
  const stale = (): void => void models.refresh(LESSON);
  const loader = new ViewLoader<ExploreView>();

  const id = $derived(info.model_id);
  const row = $derived(nnPredictions.exploreRow[id] ?? 0);
  const feature = $derived(Math.min(nnPredictions.exploreFeature[id] ?? 0, info.feature_names.length - 1));
  const key = $derived(`${id}|${row}|${feature}`);

  $effect(() => {
    const req = { model_id: id, row, feature };
    // Debounce: typing an index or moving through the select only asks for the last one.
    return loader.load(key, () => backend.nnExplore(req), stale, 60);
  });

  const view = $derived(loader.for(key) ?? (loader.data && loader.data.feature === feature ? loader.data : null));
  const nRows = $derived(view?.n_rows ?? info.n_train + info.n_test);
  const classNames = $derived(info.class_names ?? []);
  const featureName = $derived(info.feature_names[feature] ?? "");
  const figure = $derived(
    view
      ? exploreFigure(
          { kind: view.kind, xs: view.xs, ys: view.ys, classes: view.classes, confidence: view.confidence, baseValue: view.base_value },
          featureName,
          classNames,
          theme.chart,
        )
      : null,
  );

  const baseText = $derived.by(() => {
    const v = view;
    if (!v) {
      return "";
    }
    const p = v.base_prediction;
    if (p.kind === "value") {
      return fmt(t("base_regresion"), { valor: p.value });
    }
    if (v.kind === "binary") {
      return fmt(t("base_binaria"), { clase: p.name, p: p.proba?.[1]?.probability ?? Number.NaN });
    }
    const confidence = Math.max(...(p.proba ?? []).map((c) => c.probability));
    return fmt(t("base_multiclase"), { clase: p.name, confianza: confidence });
  });

  const metrics = $derived.by(() => {
    const v = view;
    if (!v) {
      return [];
    }
    if (v.kind === "regression") {
      return [
        { label: t("rango_predicciones"), value: fixed(v.range, 6), help: t("rango_predicciones_help") },
        { label: t("impacto_relativo"), value: `${fixed(v.relative_impact ?? 0, 1)}%`, help: t("impacto_relativo_help") },
      ];
    }
    return [
      { label: t("rango_probabilidades"), value: fixed(v.range, 3), help: t("rango_probabilidades_help") },
      { label: t("volatilidad"), value: fixed(v.volatility, 3), help: t("volatilidad_help") },
    ];
  });

  const verdict = $derived.by(() => {
    const v = view;
    if (!v) {
      return null;
    }
    const suffix = v.kind === "regression" ? "_regresion" : "";
    const [key, severity] =
      v.influence === "high"
        ? (["muy_influyente", "success"] as const)
        : v.influence === "moderate"
          ? (["moderada", "warning"] as const)
          : (["poco_influyente", "info"] as const);
    return { severity, markdown: fmt(t(`${key}${suffix}`), { caracteristica: featureName }) };
  });

  const baseRow = $derived<Cell[][]>(view ? [view.base.map((v) => ({ kind: "number", v }) as Cell)] : []);
  const featureOptions = $derived(info.feature_names.map((name, i) => ({ label: name, value: i })));
  const baseClassName = $derived(view?.base_class !== null && view?.base_class !== undefined ? (classNames[view.base_class] ?? `Clase ${view.base_class}`) : "");
</script>

<section class="explore" data-nn-explore>
  <h3>{t("titulo")}</h3>
  <Expander id="nn-exploracion-que-es" label={t("que_es_expander")}>
    <Markdown source={md("redes/exploracion-que-es.md")} />
  </Expander>
  <Markdown source={t("intro")} />

  <Markdown source={t("paso_1")} />
  <Notice severity="info" markdown={t("tip_base")} />
  <div class="row">
    <NumberInput
      id="nn-explore-row"
      label={t("muestra_label")}
      min={0}
      max={Math.max(nRows - 1, 0)}
      value={row}
      onchange={(value) => (nnPredictions.exploreRow = { ...nnPredictions.exploreRow, [id]: Math.round(value) })}
    />
  </div>

  {#if loader.error}
    <Notice severity="error" markdown={fmt(t("error"), { error: escapeMarkdown(loader.error) })} />
  {:else if !view}
    <Spinner label="Explorando la red…" />
  {:else}
    <Markdown source={t("paso_2")} />
    <p class="caption">{t("paso_2_caption")}</p>
    <DataTable label="Valores base" columns={info.feature_names} rows={baseRow} index={[String(view.row)]} compact />
    <div data-nn-explore-base>
      <Notice severity="info" markdown={baseText} />
    </div>
    {#if !view.output_is_probability && view.kind !== "regression"}
      <p class="caption" data-nn-not-probability>{nnText("predicciones", "nota_tanh_multiclase")}</p>
    {/if}
  {/if}

  <Markdown source={t("paso_3")} />
  <Notice severity="info" markdown={t("objetivo")} />
  <Select
    id="nn-explore-feature"
    label={t("caracteristica_label")}
    help={t("caracteristica_help")}
    options={featureOptions}
    value={feature}
    onchange={(value) => (nnPredictions.exploreFeature = { ...nnPredictions.exploreFeature, [id]: value })}
  />

  {#if view && figure}
    <div data-nn-explore-chart data-feature={view.feature}>
      <PlotlyChart data={figure.data} layout={figure.layout} height={500} downloadName="exploracion_{featureName}.png" label={featureName} />
    </div>
    <Markdown source={t("analisis")} />
    <MetricGrid items={metrics} />
    {#if verdict}
      <div data-nn-explore-verdict={view.influence} data-pedagogical>
        <Notice severity={verdict.severity} markdown={verdict.markdown} />
      </div>
    {/if}
    {#if view.kind === "multiclass"}
      <p class="caption">{fmtText(t("multiclase_nota"), { clase: baseClassName })}</p>
    {/if}
    <p class="caption">{t("nota_una_muestra")}</p>
    <Expander id="nn-exploracion-consejos" label={t("consejos_expander")}>
      <Markdown source={fmt(md("redes/exploracion-consejos.md"), { caracteristica: featureName })} />
    </Expander>
  {/if}
</section>

<style>
  h3 {
    margin: 18px 0 10px;
    color: var(--color-heading);
    font-size: 20px;
  }

  .row {
    max-width: 360px;
  }

  .caption {
    margin: 4px 2px 10px;
    color: var(--color-text-muted);
    font-size: 13px;
  }
</style>
