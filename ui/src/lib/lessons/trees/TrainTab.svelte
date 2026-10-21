<script lang="ts">
  /** 🏋️ Entrenamiento of 🌲 Árboles (fase3-arboles.md §2.2), below the dataset selector. */
  import ButtonPanel from "../../../components/ButtonPanel.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Select from "../../../components/Select.svelte";
  import Slider from "../../../components/Slider.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { DatasetSummary } from "../../bindings/DatasetSummary";
  import type { TreeTask } from "../../bindings/TreeTask";
  import { arbText, md } from "../../content";
  import { fmt } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { datasetKey, session } from "../../session.svelte";
  import { criterionOptions, LESSON, trees } from "./state.svelte";

  const t = (key: string): string => arbText("entrenamiento", key);
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
        if (stale) {
          return;
        }
        summary = s;
        // D37: preselect the task of the dataset (Python always started in Clasificación).
        trees.preset(datasetKey(s.dataset), s.task);
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
    if (!summary || !ready || summary.task === trees.task) {
      return null;
    }
    return fmt(t("aviso_tipo"), { tarea: summary.task === "classification" ? "Clasificación" : "Regresión" });
  });
  const label = $derived(session.entry(session.selected)?.label ?? summary?.label ?? "");
  const criteria = $derived(criterionOptions(trees.task).map((c) => ({ label: c, value: c })));

  function train(): void {
    void models.train(LESSON, session.selected, trees.config(), trees.testSize);
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
      label="Tipo de árbol"
      value={trees.task}
      onchange={(value: TreeTask) => trees.setTask(value)}
      options={[
        { label: t("btn_clasificacion"), value: "classification", help: t("btn_clasificacion_help") },
        { label: t("btn_regresion"), value: "regression", help: t("btn_regresion_help") },
      ]}
    />
    {#if mismatch}
      <Notice severity="warning" markdown={mismatch} />
      {#if trees.task === "regression"}
        <Notice severity="info" markdown={t("aviso_regresion_en_clasificacion")} />
      {/if}
    {/if}

    <Markdown source={t("parametros_titulo")} />
    <div class="columns">
      <div>
        <Select
          id="arboles-criterio"
          label={t("criterio_label")}
          help={t("criterio_help")}
          options={criteria}
          value={trees.criterion}
          onchange={(v) => (trees.criterion = v)}
        />
        <Slider
          id="arboles-profundidad"
          label={t("profundidad_label")}
          help={t("profundidad_help")}
          min={1}
          max={10}
          step={1}
          value={trees.maxDepth}
          onchange={(v) => (trees.maxDepth = Math.round(v))}
        />
        <Expander id="arboles-criterios" label={t("criterios_expander")}>
          <Markdown source={md("arboles/criterios.md")} />
        </Expander>
      </div>
      <div>
        <Slider
          id="arboles-min-muestras"
          label={t("min_muestras_label")}
          help={t("min_muestras_help")}
          min={2}
          max={10}
          step={1}
          value={trees.minSamplesSplit}
          onchange={(v) => (trees.minSamplesSplit = Math.round(v))}
        />
        <Slider
          id="arboles-test-size"
          label={t("test_size_label")}
          help={t("test_size_help")}
          min={0.1}
          max={0.5}
          step={0.05}
          value={trees.testSize}
          onchange={(v) => (trees.testSize = Math.round(v * 100) / 100)}
        />
        <Expander id="arboles-parametros" label={t("parametros_expander")}>
          <Markdown source={md("arboles/parametros.md")} />
        </Expander>
      </div>
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
        <Notice severity="error" markdown={t("error_clasificacion_regresion")} />
        <Notice severity="info" markdown={fmt(t("error_clasificacion_regresion_detalle"), { dataset: label })} />
      {:else}
        <Notice severity="error" markdown={fmt(t("error_entrenar"), { error: lessonState.trainError })} />
      {/if}
    {:else if lessonState.justTrained && lessonState.info}
      <Notice severity="success" markdown={t("exito")} />
      <Notice severity="info" markdown={t("ir_a_visualizacion")} />
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

  .train-button {
    margin-top: 14px;
  }

  .training {
    display: flex;
    align-items: center;
    gap: 16px;
  }
</style>
