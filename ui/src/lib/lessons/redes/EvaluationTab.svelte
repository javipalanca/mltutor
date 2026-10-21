<script lang="ts">
  /**
   * 📈 Evaluación of 🧠 Redes Neuronales (fase5-redes.md §2.4): "📊 Resultados del
   * Entrenamiento" (loss and Exactitud of `model.evaluate`, D86), the history (C27), C1/C2 on
   * the test set and "🔍 Diagnóstico del Entrenamiento" (C23) with the R² warning (D84). A
   * diverged training (K1) shows why instead of NaN metrics; a tanh output warns that its
   * outputs are not probabilities (D87).
   */
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { DiagnosticsView } from "../../bindings/DiagnosticsView";
  import type { HistoryView } from "../../bindings/HistoryView";
  import type { NnEvaluationView } from "../../bindings/NnEvaluationView";
  import ClassificationEvaluation from "../../components/ml/ClassificationEvaluation.svelte";
  import MetricGrid from "../../components/ml/MetricGrid.svelte";
  import NnViewTree from "../../components/ml/NnViewTree.svelte";
  import RegressionEvaluation from "../../components/ml/RegressionEvaluation.svelte";
  import TrainingHistoryChart from "../../components/ml/TrainingHistoryChart.svelte";
  import { md, nnText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fixed } from "../../format";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import NnBanner from "./NnBanner.svelte";
  import { divergenceText, recoveredDivergenceText, tanhOutputText } from "./nnTexts";
  import { LESSON } from "./state.svelte";

  const t = (key: string): string => nnText("evaluacion", key);
  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const evaluation = new ViewLoader<NnEvaluationView>();
  const history = new ViewLoader<HistoryView>();
  const diagnostics = new ViewLoader<DiagnosticsView>();

  $effect(() => {
    const info = lessonState.info;
    if (!info) {
      return;
    }
    const id = info.model_id;
    const key = String(id);
    const stops = [
      evaluation.load(key, () => backend.nnEvaluation(id), stale),
      history.load(key, () => backend.nnHistory(id), stale),
      diagnostics.load(key, () => backend.nnDiagnostics(id), stale),
    ];
    return () => stops.forEach((stop) => stop());
  });

  const key = $derived(String(lessonState.info?.model_id ?? ""));
  const evalView = $derived(evaluation.for(key));
  const hist = $derived(history.for(key));
  const diag = $derived(diagnostics.for(key));
  const regression = $derived(evalView?.metric === "mae");

  const metrics = $derived.by(() => {
    const v = evalView;
    if (!v) {
      return [];
    }
    const value = (x: number | null): string => (x === null ? "nan" : fixed(x, 3));
    return regression
      ? [{ label: t("metrica_error_regresion"), value: value(v.test_loss), help: t("metrica_error_regresion_help") }]
      : [
          { label: t("metrica_accuracy"), value: value(v.test_metric), help: t("metrica_accuracy_help") },
          { label: t("metrica_perdida"), value: value(v.test_loss), help: t("metrica_perdida_help") },
        ];
  });
</script>

<section class="nn-evaluation" data-nn-evaluation>
  <h2>{t("header")}</h2>
  {#if !lessonState.info}
    <Notice severity="warning" markdown={nnText("comun", "sin_entrenar")} />
  {:else}
    <NnBanner info={lessonState.info} />
    {#if evaluation.error}
      <Notice severity="error" markdown={escapeMarkdown(evaluation.error)} />
    {:else if !evalView}
      <Spinner label="Evaluando la red…" />
    {:else}
      <Markdown source={t("resultados_titulo")} />
      {#if evalView.diverged && !evalView.diverged.network_valid}
        <div data-nn-diverged={evalView.diverged.epoch} data-pedagogical>
          <Notice severity="error" markdown={divergenceText(evalView.diverged.epoch)} />
        </div>
      {:else}
        {#if evalView.diverged}
          <!-- K1: EarlyStopping restored finite weights; everything below is computed. -->
          <div data-nn-diverged-recovered={evalView.diverged.epoch} data-pedagogical>
            <Notice severity="warning" markdown={recoveredDivergenceText(evalView.diverged.epoch, evalView.diverged.best_epoch)} />
          </div>
        {/if}
        <div class="results" class:single={regression}>
          <MetricGrid items={metrics} />
        </div>
      {/if}
      {#if evalView.near_divergence}
        <div data-nn-near-divergence data-pedagogical>
          <Notice severity="warning" markdown={evalView.near_divergence} />
        </div>
      {/if}
      {#if !evalView.output_is_probability && !regression}
        <div data-nn-not-probability>
          <Notice severity="warning" markdown={tanhOutputText()} />
        </div>
      {/if}

      <Markdown source={t("progreso_titulo")} />
      {#if hist}
        <TrainingHistoryChart logs={hist.epochs} metric={hist.metric} bestEpoch={hist.best_epoch} lrDrops={hist.lr_drops} downloadName="historial_entrenamiento.png" />
      {:else if history.error}
        <Notice severity="error" markdown={escapeMarkdown(history.error)} />
      {:else}
        <Spinner label="Cargando el historial…" />
      {/if}

      <!-- The quality badges of C1/C2 and the diagnostics judge the network: pedagogical. -->
      <div data-pedagogical>
        {#if evalView.evaluation?.kind === "classification"}
          <ClassificationEvaluation evaluation={evalView.evaluation} modelId={lessonState.info.model_id} idPrefix="redes" onstale={stale} />
        {:else if evalView.evaluation?.kind === "regression"}
          <RegressionEvaluation evaluation={evalView.evaluation} modelId={lessonState.info.model_id} idPrefix="redes" onstale={stale} />
        {/if}
      </div>

      {#if evalView.r2_negative}
        <div data-nn-r2-negative>
          <Notice severity="warning" markdown={evalView.r2_negative} />
        </div>
        <Expander id="nn-eval-defecto" label={nnText("entrenamiento", "defecto_expander")}>
          <Markdown source={md("redes/valores-defecto.md")} />
        </Expander>
      {/if}

      {#if diagnostics.error}
        <Notice severity="error" markdown={escapeMarkdown(diagnostics.error)} />
      {:else if diag && diag.blocks.length > 0}
        <div class="diagnostics" data-nn-diagnostics data-pedagogical>
          <NnViewTree nodes={diag.blocks} idPrefix="nn-diag" />
        </div>
      {:else if !diag}
        <Spinner label="Analizando el entrenamiento…" />
      {/if}
    {/if}
  {/if}
</section>

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .results.single {
    max-width: 50%;
  }

  @container page (max-width: 420px) {
    .results.single {
      max-width: none;
    }
  }
</style>
