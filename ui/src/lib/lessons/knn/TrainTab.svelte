<script lang="ts">
  /** 🏋️ Entrenamiento of 🔍 KNN (fase4-knn.md §2.2), below the dataset selector. */
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import NumberInput from "../../../components/NumberInput.svelte";
  import Select from "../../../components/Select.svelte";
  import Slider from "../../../components/Slider.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { DatasetSummary } from "../../bindings/DatasetSummary";
  import type { KnnMetric } from "../../bindings/KnnMetric";
  import type { KnnWeights } from "../../bindings/KnnWeights";
  import { knnList, knnText, md } from "../../content";
  import { fmt } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { datasetKey, session } from "../../session.svelte";
  import { K_MAX, K_MIN, knn, LESSON } from "./state.svelte";

  const t = (key: string): string => knnText("entrenamiento", key);
  const lessonState = models.lesson(LESSON);

  let summary = $state<DatasetSummary | null>(null);
  let summaryError = $state<string | null>(null);

  $effect(() => {
    const dataset = session.selected;
    let stale = false;
    summaryError = null;
    backend
      .datasetSummary(dataset)
      .then((s) => {
        if (!stale) {
          summary = s;
        }
      })
      .catch((error: unknown) => {
        if (!stale) {
          summary = null;
          summaryError = backend.errorMessage(error);
        }
      });
    return () => {
      stale = true;
    };
  });

  const ready = $derived(summary !== null && datasetKey(summary.dataset) === datasetKey(session.selected));
  const weights = knnList("entrenamiento", "pesos_opciones").map((w) => ({ label: w, value: w as KnnWeights }));
  const metrics = knnList("entrenamiento", "metrica_opciones").map((m) => ({ label: m, value: m as KnnMetric }));

  function train(): void {
    void models.train(LESSON, session.selected, knn.config(), knn.testSize);
  }

  /** These errors already carry the message of the lesson (D59, NaN). */
  const OWN_MESSAGE = new Set(["too_few_samples", "missing_values", "too_many_classes"]);
</script>

<section class="train" data-train-tab>
  <h2>{t("header")}</h2>
  {#if summaryError}
    <Notice severity="error" markdown={fmt(t("error_carga"), { error: summaryError })} />
  {:else if !ready || !summary}
    <Spinner label="Cargando el dataset…" />
  {:else}
    <!-- D60: always (Python only showed it on the first render after loading). -->
    <Markdown source={t("info_titulo")} />
    <div class="info-box">{summary.description}</div>

    <Markdown source={t("parametros_titulo")} />
    <p>{t("parametros_texto")}</p>
    <p class="caption">{t("nota_sin_escalar")}</p>
    <div class="columns">
      <NumberInput id="knn-k" label={t("k_label")} min={K_MIN} max={K_MAX} step={1} value={knn.k} onchange={(v) => (knn.k = Math.round(v))} />
      <Select id="knn-pesos" label={t("pesos_label")} options={weights} value={knn.weights} onchange={(v) => (knn.weights = v)} />
      <Select id="knn-metrica" label={t("metrica_label")} options={metrics} value={knn.metric} onchange={(v) => (knn.metric = v)} />
      <Slider
        id="knn-test-size"
        label={t("test_size_label")}
        help={t("test_size_help")}
        min={0.1}
        max={0.5}
        step={0.05}
        value={knn.testSize}
        onchange={(v) => (knn.testSize = Math.round(v * 100) / 100)}
      />
    </div>
    <Expander id="knn-parametros" label={t("parametros_expander")}>
      <Markdown source={md("knn/parametros.md")} />
    </Expander>
    <Expander id="knn-escala" label={t("escala_expander")}>
      <Markdown source={md("knn/escala.md")} />
    </Expander>
    <Expander id="knn-empates" label={t("empates_expander")}>
      <Markdown source={md("knn/empates.md")} />
    </Expander>

    <button type="button" class="btn block primary train-button" disabled={lessonState.training} onclick={train} data-train-button>
      {t("btn_entrenar")}
    </button>

    {#if lessonState.training}
      <div class="training">
        <Spinner label={t("spinner")} />
        <button type="button" class="btn" onclick={() => models.cancelTraining(LESSON)}>Cancelar</button>
      </div>
    {:else if lessonState.trainError}
      {#if lessonState.trainErrorKind && OWN_MESSAGE.has(lessonState.trainErrorKind)}
        <Notice severity="error" markdown={fmt("{error}", { error: lessonState.trainError })} />
      {:else}
        <Notice severity="error" markdown={fmt(t("error_entrenar"), { error: lessonState.trainError })} />
      {/if}
    {:else if lessonState.justTrained && lessonState.info}
      <Notice severity="success" markdown={t("exito")} />
      <Notice severity="info" markdown={t("ir_a_evaluacion")} />
    {/if}
  {/if}
</section>

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .info-box {
    margin: 8px 0 18px;
    padding: 12px 16px;
    background: var(--color-info-bg);
    border-left: 4px solid var(--color-primary);
    border-radius: var(--radius-notice);
    user-select: text;
  }

  .caption {
    margin: -8px 2px 6px;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
  }

  .columns {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 24px;
    align-items: start;
  }

  .train-button {
    margin-top: 14px;
  }

  .training {
    display: flex;
    align-items: center;
    gap: 16px;
  }
</style>
