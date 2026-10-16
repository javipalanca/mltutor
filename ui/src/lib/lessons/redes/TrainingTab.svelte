<script lang="ts">
  /**
   * ⚙️ Entrenamiento of 🧠 Redes Neuronales (fase5-redes.md §2.3): learning rate, epochs,
   * validation, the advanced callbacks and the seed (D70); the real split (D73); the steps
   * 1–4 and the history chart live while it trains (D82, C27), with "Cancelar" (D83).
   */
  import Checkbox from "../../../components/Checkbox.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import NumberInput from "../../../components/NumberInput.svelte";
  import Select from "../../../components/Select.svelte";
  import Slider from "../../../components/Slider.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import TrainingHistoryChart from "../../components/ml/TrainingHistoryChart.svelte";
  import { md, nnText, regText } from "../../content";
  import { fmt } from "../../fmt";
  import { models, sameConfig } from "../../ml/modelSession.svelte";
  import { datasetKey, session } from "../../session.svelte";
  import { architecture } from "./architecture.svelte";
  import { LEARNING_RATES, LESSON, nn } from "./state.svelte";
  import { costText } from "./nnTexts";
  import { training } from "./training.svelte";

  const t = (key: string): string => nnText("entrenamiento", key);
  const lessonState = models.lesson(LESSON);

  const view = $derived(architecture.matches(session.selected) ? architecture.view : null);
  const key = $derived(datasetKey(session.selected));

  $effect(() => {
    // K6: with the epochs, the view also estimates how long the training will take.
    architecture.request(session.selected, nn.arch(architecture.view), nn.validationPct, false, nn.epochsOf(datasetKey(session.selected), architecture.view));
  });

  const epochs = $derived(nn.epochsOf(key, view));
  const untrainable = $derived(view?.output_check.untrainable ?? null);

  const lrOptions = LEARNING_RATES.map((lr) => ({ label: String(lr), value: lr as number }));
  const lrCaption = $derived(
    nn.learningRate === 0.001 ? t("lr_0001") : nn.learningRate === 0.01 ? t("lr_001") : nn.learningRate === 0.1 ? t("lr_01") : t("lr_03"),
  );
  const epochsCaption = $derived(epochs < 50 ? t("epocas_pocas") : epochs <= 150 ? t("epocas_adecuadas") : t("epocas_muchas"));
  const split = $derived(view?.split ?? null);
  const patienceCaption = $derived(nn.patience <= 5 ? t("paciencia_baja") : nn.patience <= 15 ? t("paciencia_media") : t("paciencia_alta"));
  const factorCaption = $derived(nn.lrFactor <= 0.3 + 1e-9 ? t("factor_agresivo") : nn.lrFactor <= 0.7 + 1e-9 ? t("factor_moderado") : t("factor_suave"));

  /** D91: the trained network is not the one of the current controls or dataset. */
  const stale = $derived.by(() => {
    const info = lessonState.info;
    if (!info || lessonState.training) {
      return false;
    }
    return datasetKey(info.dataset) !== key || !sameConfig(info.config, nn.config(key, view));
  });

  /** These errors already carry the message of the lesson (D71, one sample per class, NaN). */
  const OWN_MESSAGE = new Set(["untrainable_output", "too_few_per_class", "missing_values", "invalid_config"]);

  const phaseText = $derived.by(() => {
    switch (nn.live.phase) {
      case "building":
        return t("paso_2");
      case "compiling":
        return t("paso_3");
      case "training":
        return fmt(t("paso_4"), { epocas: nn.live.epochs || epochs });
      default:
        return t("paso_1");
    }
  });
  const lastLog = $derived(nn.live.logs.at(-1) ?? null);
  const percent = $derived(nn.live.epochs > 0 && lastLog ? Math.min(100, (lastLog.epoch / nn.live.epochs) * 100) : 0);
  const epochLine = $derived(
    lastLog
      ? lastLog.val_loss !== null
        ? fmt(t("progreso_epoca_val"), { epoca: lastLog.epoch, epocas: nn.live.epochs, perdida: lastLog.loss ?? Number.NaN, val_perdida: lastLog.val_loss })
        : fmt(t("progreso_epoca"), { epoca: lastLog.epoch, epocas: nn.live.epochs, perdida: lastLog.loss ?? Number.NaN })
      : null,
  );
  /** D88: weight updates per epoch and in total. */
  const steps = $derived(split ? Math.ceil(split.train / nn.batchSize) : 0);
  /** ReduceLROnPlateau patience of the app: `max(1, paciencia // 2)`, or 5 without early stopping. */
  const rlrPatience = $derived(nn.earlyStopping ? Math.max(1, Math.floor(nn.patience / 2)) : 5);
  /** D82: "📉 Época n: … el learning rate baja a x", live, from the logged learning rates. */
  const lrEvents = $derived.by(() => {
    const out: string[] = [];
    const logs = nn.live.logs;
    for (let i = 1; i < logs.length; i++) {
      const before = logs[i - 1]?.learning_rate ?? null;
      const now = logs[i]?.learning_rate ?? null;
      if (before !== null && now !== null && now < before * (1 - 1e-6)) {
        out.push(fmt(t("progreso_lr"), { epoca: logs[i - 1]?.epoch ?? i, paciencia: rlrPatience, lr: Number(now.toPrecision(3)) }));
      }
    }
    return out;
  });
  const nnInfo = $derived(lessonState.info?.nn ?? null);
  const bestLine = $derived.by(() => {
    if (!nnInfo || nnInfo.best_epoch === null) {
      return null;
    }
    const best = nn.live.logs.find((l) => l.epoch === nnInfo.best_epoch);
    return fmt(t("completado_mejor_epoca"), { epocas: nnInfo.epochs_trained, mejor: nnInfo.best_epoch, val_perdida: best?.val_loss ?? Number.NaN });
  });
  const stopLine = $derived(
    nnInfo && nnInfo.stopped_early && nnInfo.best_epoch !== null
      ? fmt(t("progreso_parada"), { epoca: nnInfo.epochs_trained, mejor: nnInfo.best_epoch })
      : null,
  );
  const trainedWith = $derived.by(() => {
    const info = lessonState.info;
    if (!info || info.config.kind !== "neural_net") {
      return null;
    }
    const c = info.config;
    return fmt(t("config_modelo"), {
      tarea: info.model_task === "regression" ? "regresión" : "clasificación",
      arquitectura: `[${(info.nn?.layers ?? []).join(", ")}]`,
      activacion: c.arch.activation,
      salida: c.arch.output_activation,
      optimizador: c.arch.optimizer,
      lr: Number(c.training.learning_rate.toPrecision(3)),
      epocas: c.training.epochs,
      semilla: c.training.seed,
    });
  });
  const metric = $derived(lessonState.info?.nn?.metric ?? (view?.task === "regression" ? "mae" : "accuracy"));

  const history = $derived(training.history);

  function train(): void {
    if (view && !untrainable) {
      void training.train(session.selected, view);
    }
  }

  function cancel(): void {
    training.cancel();
  }

  const showChart = $derived(nn.live.logs.length > 0 && (lessonState.training || lessonState.justTrained || nn.live.cancelledAt !== null));
</script>

<section class="training" data-nn-training>
  <h2>{t("header")}</h2>
  {#if architecture.error && !view}
    <Notice severity="error" markdown={fmt(nnText("arquitectura", "error_carga"), { error: architecture.error })} />
  {:else if !view}
    <Spinner label="Cargando el dataset…" />
  {:else}
    <Notice severity="info" markdown={md("redes/entrenamiento-conceptos.md")} />
    <Markdown source={t("parametros_titulo")} />
    <div class="three">
      <div class="col">
        <Markdown source={t("lr_titulo")} />
        <Select id="nn-lr" label={t("lr_label")} help={t("lr_help")} options={lrOptions} value={nn.learningRate} onchange={(v: number) => (nn.learningRate = v)} />
        <p class="caption"><Markdown inline source={lrCaption} /></p>
      </div>
      <div class="col">
        <Markdown source={t("epocas_titulo")} />
        <Slider
          id="nn-epocas"
          label={t("epocas_label")}
          help={fmt(t("epocas_help"), { sugerencia: view.suggested_epochs })}
          min={10}
          max={500}
          step={10}
          value={epochs}
          onchange={(v) => nn.setEpochs(key, Math.round(v))}
        />
        <p class="caption"><Markdown inline source={epochsCaption} /></p>
      </div>
      <div class="col">
        <Markdown source={t("validacion_titulo")} />
        <Slider
          id="nn-validacion"
          label={t("validacion_label")}
          help={t("validacion_help")}
          min={10}
          max={40}
          value={nn.validationPct}
          onchange={(v) => (nn.validationPct = Math.round(v))}
        />
        {#if split}
          <p class="caption" data-nn-split={`${split.train},${split.val},${split.test}`}>
            <span title={t("distribucion_help")}><Markdown inline source={fmt(t("distribucion"), { train: split.train, val: split.val, test: split.test })} /></span>
          </p>
          <p class="caption"><Markdown inline source={fmt(t("pasos_por_epoca"), { pasos: steps, train: split.train, batch: nn.batchSize, total: steps * epochs, epocas: epochs })} /></p>
        {/if}
      </div>
    </div>

    <p class="caption scaling"><Markdown inline source={t("escalado_nota")} /></p>
    <Expander id="nn-escalado" label={t("escalado_expander")}>
      <Markdown source={md("redes/escalado.md")} />
    </Expander>
    <Expander id="nn-avanzada" label={t("avanzada_expander")}>
      <Markdown source={t("avanzada_titulo")} />
      <div class="two">
        <div class="col">
          <Markdown source={t("es_titulo")} />
          <Checkbox id="nn-es" label={t("es_label")} help={t("es_help")} checked={nn.earlyStopping} onchange={(v) => (nn.earlyStopping = v)} />
          {#if nn.earlyStopping}
            <Notice severity="success" markdown={t("es_activado")} />
            <Slider
              id="nn-paciencia"
              label={t("paciencia_label")}
              help={t("paciencia_help")}
              min={5}
              max={50}
              value={nn.patience}
              onchange={(v) => (nn.patience = Math.round(v))}
            />
            <p class="caption"><Markdown inline source={patienceCaption} /></p>
          {:else}
            <Notice severity="warning" markdown={t("es_desactivado")} />
          {/if}
        </div>
        <div class="col">
          <Markdown source={t("rlr_titulo")} />
          <Checkbox id="nn-rlr" label={t("rlr_label")} help={t("rlr_help")} checked={nn.reduceLr} onchange={(v) => (nn.reduceLr = v)} />
          {#if nn.reduceLr}
            <Notice severity="success" markdown={t("rlr_activado")} />
            <Slider
              id="nn-factor"
              label={t("factor_label")}
              help={t("factor_help")}
              min={0.1}
              max={0.9}
              step={0.01}
              value={nn.lrFactor}
              onchange={(v) => (nn.lrFactor = Math.round(v * 100) / 100)}
            />
            <p class="caption"><Markdown inline source={factorCaption} /></p>
          {:else}
            <Notice severity="info" markdown={t("rlr_desactivado")} />
          {/if}
        </div>
      </div>
      <Markdown source={t("semilla_titulo")} />
      <div class="seed">
        <NumberInput id="nn-semilla" label={t("semilla_label")} help={t("semilla_help")} min={1} max={9999} step={1} value={nn.seed} onchange={(v) => (nn.seed = Math.max(1, Math.min(9999, Math.round(v))))} />
      </div>
      <Expander id="nn-semilla-por-que" label={t("semilla_expander")}>
        <Markdown source={md("redes/semilla.md")} />
      </Expander>
      <hr />
      <Markdown source={t("tecnicas_titulo")} />
      <Markdown source={md("redes/entrenamiento-tecnicas.md")} />
    </Expander>

    <Markdown source={t("iniciar_titulo")} />
    <Markdown source={t("iniciar_texto")} />
    {#if stale && lessonState.info}
      <div data-nn-stale>
        <Notice severity="info" markdown={fmt(regText("comun", "modelo_entrenado_con"), { dataset: lessonState.info.dataset_label })} />
        {#if trainedWith}
          <p class="caption"><Markdown inline source={trainedWith} /></p>
        {/if}
        {#if datasetKey(lessonState.info.dataset) !== key}
          <Notice
            severity="warning"
            markdown={fmt(regText("comun", "aviso_dataset_cambiado"), { dataset_modelo: lessonState.info.dataset_label, dataset_actual: session.entry(session.selected)?.label ?? "" })}
          />
        {:else}
          <Notice severity="warning" markdown={t("aviso_config_cambiada")} />
        {/if}
      </div>
    {/if}
    {#if view.cost && view.cost.level !== "low" && !untrainable}
      <div data-nn-cost={view.cost.level}>
        <Notice severity={view.cost.level === "very_high" ? "warning" : "info"} markdown={costText(view.cost.level, view.cost.estimated_seconds)} />
      </div>
    {/if}
    {#if untrainable}
      <div data-untrainable>
        <Notice severity="error" markdown={fmt("{m}", { m: untrainable })} />
      </div>
    {/if}
    <!-- A disabled button shows no `title` in WebKit: the help goes on a wrapper. -->
    <span class="train-wrap" title={untrainable ? nnText("arquitectura", "no_entrenable_boton") : undefined}>
      <button
        type="button"
        class="btn block primary train-button"
        disabled={lessonState.training || untrainable !== null}
        onclick={train}
        data-train-button
      >
        {t("btn_entrenar")}
      </button>
    </span>
    <Expander id="nn-defecto" label={t("defecto_expander")}>
      <Markdown source={md("redes/valores-defecto.md")} />
    </Expander>

    {#if lessonState.training}
      <div class="progress" data-nn-progress={nn.live.phase ?? "preparing"}>
        <Notice severity="info" markdown={phaseText} />
        <div class="bar-row">
          <div class="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(percent)} aria-label={t("spinner")}>
            <span style:width="{percent}%"></span>
          </div>
          <button type="button" class="btn" onclick={cancel} disabled={nn.live.cancelRequested} data-nn-cancel>{t("btn_cancelar")}</button>
        </div>
        <p class="epoch-line" data-nn-epoch={lastLog?.epoch ?? 0}>{epochLine ?? t("spinner")}</p>
        {#if nn.live.cancelRequested}
          <Notice severity="warning" markdown={t("cancelando")} />
        {/if}
        <p class="caption">{t("progreso_grafica")}</p>
      </div>
    {:else if lessonState.trainError}
      {#if lessonState.trainErrorKind && OWN_MESSAGE.has(lessonState.trainErrorKind)}
        <Notice severity="error" markdown={fmt("{error}", { error: lessonState.trainError })} />
      {:else}
        <Notice severity="error" markdown={t("error_contenedor")} />
        <Notice severity="error" markdown={fmt(t("error_detalle"), { error: lessonState.trainError })} />
        <Notice severity="info" markdown={t("error_consejo")} />
      {/if}
    {:else if nn.live.cancelledAt !== null}
      <div data-nn-cancelled>
        <Notice
          severity="warning"
          markdown={fmt(lessonState.info ? t("cancelado") : t("cancelado_sin_modelo"), { epoca: nn.live.cancelledAt, epocas: nn.live.epochs })}
        />
      </div>
    {:else if lessonState.justTrained && lessonState.info}
      <div data-nn-trained>
        <Notice severity="success" markdown={t("completado")} />
        {#if stopLine}
          <Notice severity="info" markdown={stopLine} />
        {/if}
        {#if bestLine}
          <p class="caption"><Markdown inline source={bestLine} /></p>
        {/if}
        <Notice severity="success" markdown={t("exito")} />
      </div>
    {/if}

    {#each lrEvents as line (line)}
      {#if showChart}
        <p class="caption lr-event"><Markdown inline source={line} /></p>
      {/if}
    {/each}
    {#if showChart}
      <TrainingHistoryChart
        logs={nn.live.logs}
        {metric}
        maxEpochs={lessonState.training ? nn.live.epochs : null}
        bestEpoch={history?.bestEpoch ?? null}
        lrDrops={history?.lrDrops ?? []}
      />
    {/if}
  {/if}
</section>

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .two,
  .three {
    display: grid;
    gap: 24px;
    align-items: start;
  }

  .two {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .three {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .caption {
    margin: -6px 2px 8px;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
  }

  .seed {
    max-width: 260px;
    margin-top: 6px;
  }

  hr {
    margin: 18px 0;
    border: 0;
    border-top: 1px solid var(--color-rule);
  }

  .train-wrap {
    display: block;
  }

  .train-button {
    margin-top: 10px;
  }

  .progress {
    margin-top: 12px;
  }

  .bar-row {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .bar {
    flex: 1;
    height: 10px;
    overflow: hidden;
    background: var(--color-slider-groove);
    border-radius: 5px;
  }

  .bar span {
    display: block;
    height: 100%;
    background: var(--color-primary);
    border-radius: 5px;
    transition: width 0.15s linear;
  }

  .epoch-line {
    margin: 6px 2px 0;
    color: var(--color-text-muted);
    font-size: 13px;
    font-variant-numeric: tabular-nums;
  }
</style>
