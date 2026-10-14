<script lang="ts">
  /** 🏋️ Entrenamiento of 📊 Regresión (fase2-regresion.md §2.2), below the dataset selector. */
  import ButtonPanel from "../../../components/ButtonPanel.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Slider from "../../../components/Slider.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { DatasetSummary } from "../../bindings/DatasetSummary";
  import { md, optionalText, regRaw, regText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fmt } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { datasetKey, session } from "../../session.svelte";
  import { LESSON, regression, type RegressionModelType } from "./state.svelte";

  const t = (key: string): string => regText("entrenamiento", key);
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
  const mismatch = $derived.by(() => {
    if (!summary || !ready) {
      return null;
    }
    if (summary.task === "classification" && regression.modelType === "linear") {
      return t("aviso_lineal_en_clasificacion");
    }
    if (summary.task === "regression" && regression.modelType === "logistic") {
      return t("aviso_logistica_en_regresion");
    }
    return null;
  });
  const notConverged = $derived(
    lessonState.justTrained ? (lessonState.info?.notices.find((n) => n.kind === "not_converged") ?? null) : null,
  );
  /** Caption under "Máximo de Iteraciones": the solver usually reaches the optimum first. */
  const maxIterNote = optionalText(regRaw, "entrenamiento", "max_iter_nota", "");
  const label = $derived(session.entry(session.selected)?.label ?? summary?.label ?? "");

  function train(): void {
    void models.train(LESSON, session.selected, regression.config(), regression.testSize);
  }

  function setType(value: RegressionModelType): void {
    regression.modelType = value;
  }
</script>

<section class="train" data-train-tab>
  <h2>{t("header")}</h2>
  {#if summaryError}
    <Notice severity="error" markdown={fmt(t("error_carga"), { error: summaryError })} />
  {:else if !ready || !summary}
    <Spinner label="Cargando el dataset…" />
  {:else}
    <Markdown source={t("info_titulo")} />
    <div class="info-box">{summary.description}</div>

    <Markdown source={t("tipo_titulo")} />
    <ButtonPanel
      label="Tipo de modelo"
      value={regression.modelType}
      onchange={setType}
      options={[
        { label: t("btn_lineal"), value: "linear", help: t("btn_lineal_help") },
        { label: t("btn_logistica"), value: "logistic", help: t("btn_logistica_help") },
      ]}
    />
    {#if mismatch}
      <Notice severity="warning" markdown={mismatch} />
    {/if}

    <Markdown source={t("parametros_titulo")} />
    <div class="columns">
      <div>
        {#if regression.modelType === "logistic"}
          <Slider
            id="regresion-max-iter"
            label={t("max_iter_label")}
            help={t("max_iter_help")}
            min={100}
            max={2000}
            step={100}
            value={regression.maxIter}
            onchange={(v) => (regression.maxIter = Math.round(v))}
          />
          {#if maxIterNote}
            <p class="caption" data-max-iter-note><Markdown inline source={maxIterNote} /></p>
          {/if}
        {:else}
          <Notice severity="info" markdown={t("lineal_sin_iteraciones")} />
        {/if}
      </div>
      <Slider
        id="regresion-test-size"
        label={t("test_size_label")}
        help={t("test_size_help")}
        min={0.1}
        max={0.5}
        step={0.05}
        value={regression.testSize}
        onchange={(v) => (regression.testSize = Math.round(v * 100) / 100)}
      />
    </div>

    <button type="button" class="btn block primary train-button" disabled={lessonState.training} onclick={train} data-train-button>
      {t("btn_entrenar")}
    </button>

    {#if lessonState.training}
      <div class="training">
        <Spinner label={t("spinner")} />
        <button type="button" class="btn" onclick={() => models.cancelTraining(LESSON)}>Cancelar</button>
      </div>
    {:else if lessonState.trainError}
      {#if lessonState.trainErrorKind === "invalid_target"}
        <Notice severity="error" markdown={t("error_logistica_regresion")} />
        <Notice severity="info" markdown={fmt(t("error_logistica_regresion_detalle"), { dataset: label })} />
      {:else}
        <Notice severity="error" markdown={escapeMarkdown(lessonState.trainError)} />
      {/if}
    {:else if lessonState.justTrained && lessonState.info}
      <Notice severity="success" markdown={t("exito")} />
      {#if notConverged}
        <Notice severity="warning" markdown={fmt(t("aviso_no_convergencia"), { max_iter: notConverged.max_iter })} />
        <Expander id="regresion-no-convergencia" label={t("no_convergencia_expander")}>
          <Markdown source={md("regresion/no-convergencia.md")} />
        </Expander>
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

  .info-box {
    margin: 8px 0 18px;
    padding: 12px 16px;
    background: var(--color-info-bg);
    border-left: 4px solid var(--color-primary);
    border-radius: var(--radius-notice);
    user-select: text;
  }

  .columns {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 28px;
    align-items: start;
  }

  .caption {
    margin: -4px 2px 8px;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
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
