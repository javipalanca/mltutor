<script lang="ts">
  /** 🏗️ Arquitectura of 🧠 Redes Neuronales (fase5-redes.md §2.2), below "📊 Dataset actual". */
  import CodeBlock from "../../../components/CodeBlock.svelte";
  import DataTable from "../../../components/DataTable.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Metric from "../../../components/Metric.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Select from "../../../components/Select.svelte";
  import Slider from "../../../components/Slider.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { Cell } from "../../bindings/Cell";
  import type { NnActivation } from "../../bindings/NnActivation";
  import type { NnOptimizer } from "../../bindings/NnOptimizer";
  import NetworkDiagram from "../../components/ml/NetworkDiagram.svelte";
  import { md, nnList, nnText } from "../../content";
  import { fmt } from "../../fmt";
  import { datasetKey, session } from "../../session.svelte";
  import { architecture } from "./architecture.svelte";
  import { BATCH_SIZES, HIDDEN_ACTIVATIONS, MAX_LAYERS, MAX_NEURONS, nn, OPTIMIZERS, thousands } from "./state.svelte";

  const t = (key: string): string => nnText("arquitectura", key);

  const view = $derived(architecture.matches(session.selected) ? architecture.view : null);
  const key = $derived(datasetKey(session.selected));

  $effect(() => {
    architecture.request(session.selected, nn.arch(architecture.view), nn.validationPct);
  });

  const output = $derived(nn.outputOf(view));
  const taskName = $derived(view?.task === "regression" ? "regresión" : "clasificación");

  const ioInfo = $derived.by(() => {
    if (!view) {
      return "";
    }
    const template = view.task === "regression" ? t("info_regresion") : view.output === 1 ? t("info_binaria") : t("info_multiclase");
    return fmt(template, { entrada: view.input, salida: view.output });
  });

  const layersCaption = $derived(nn.layers === 1 ? t("capas_1") : nn.layers === 2 ? t("capas_2") : t("capas_3_mas"));
  const activationNotice = $derived.by(() => {
    switch (nn.activation) {
      case "tanh":
        return { severity: "info" as const, text: t("activacion_tanh") };
      case "sigmoid":
        return { severity: "warning" as const, text: t("activacion_sigmoid") };
      default:
        return { severity: "success" as const, text: t("activacion_relu") };
    }
  });
  const dropoutCaption = $derived(
    nn.dropout === 0 ? t("dropout_0") : nn.dropout <= 0.2 + 1e-9 ? t("dropout_ligero") : nn.dropout <= 0.5 + 1e-9 ? t("dropout_moderado") : t("dropout_alto"),
  );
  const batchCaption = $derived(
    view?.batch_hint === "large" ? t("batch_grande") : view?.batch_hint === "optimal" ? t("batch_optimo") : t("batch_pequeno"),
  );
  const optimizerCaption = $derived(t(`optimizador_${nn.optimizer}`));

  const outputOptions = $derived((view?.output_options ?? []).map((o) => ({ label: o, value: o })));
  const hiddenOptions = HIDDEN_ACTIVATIONS.map((a) => ({ label: a, value: a }));
  const batchOptions = BATCH_SIZES.map((b) => ({ label: String(b), value: b as number }));
  const optimizerOptions = OPTIMIZERS.map((o) => ({ label: o, value: o }));

  const complexityLevels = nnList("arquitectura", "complejidad_niveles");
  const complexityIndex = $derived(view?.complexity === "low" ? 0 : view?.complexity === "medium" ? 1 : 2);
  const complexityNotice = $derived.by(() => {
    if (!view) {
      return null;
    }
    const values = { total: view.total_params, n: view.n_rows, ratio: view.complexity_ratio };
    switch (view.complexity) {
      case "low":
        return { severity: "success" as const, text: fmt(t("complejidad_baja"), values) };
      case "medium":
        return { severity: "warning" as const, text: fmt(t("complejidad_media"), values) };
      default:
        return { severity: "error" as const, text: fmt(t("complejidad_alta"), values) };
    }
  });

  const tableColumns = nnList("arquitectura", "detalles_columnas");
  const tableRows = $derived<Cell[][]>(
    (view?.layers ?? []).map((row) => [
      { kind: "text", v: row.name },
      { kind: "number", v: row.neurons },
      { kind: "number", v: row.params },
      { kind: "text", v: row.activation ?? t("detalles_entrada") },
    ]),
  );

  let code = $state<string | null>(null);
  let codeError = $state<string | null>(null);
  let codeFor = $state("");
  const codeKey = $derived(JSON.stringify([key, nn.config(key, view)]));

  async function generate(): Promise<void> {
    if (!view) {
      return;
    }
    codeError = null;
    const wanted = codeKey;
    try {
      const script = await backend.nnTrainingCode({
        dataset: session.selected,
        arch: nn.arch(view),
        training: nn.training(key, view),
      });
      code = script;
      codeFor = wanted;
    } catch (error) {
      codeError = backend.errorMessage(error);
    }
  }

  function setLayers(value: number): void {
    nn.layers = Math.max(1, Math.min(MAX_LAYERS, Math.round(value)));
  }

  function setNeurons(i: number, value: number): void {
    const next = [...nn.neurons];
    next[i] = Math.max(1, Math.min(MAX_NEURONS, Math.round(value)));
    nn.neurons = next;
  }
</script>

<section class="architecture" data-nn-architecture>
  <h2>{t("header")}</h2>
  <Markdown source={t("titulo")} />
  {#if architecture.error && !view}
    <Notice severity="error" markdown={fmt(t("error_carga"), { error: architecture.error })} />
  {:else if !view}
    <Spinner label="Cargando el dataset…" />
  {:else}
    <Notice severity="info" markdown={ioInfo} />
    <Expander id="nn-tamano" label={t("tamano_expander")}>
      <Markdown
        source={fmt(md("redes/arquitectura-tamano.md"), {
          n_muestras: view.n_rows,
          n_caracteristicas: view.input,
          mitad: view.size_tips.half,
          doble: view.size_tips.double,
          cuadruple: view.size_tips.quadruple,
          max_parametros: view.size_tips.max_params,
        })}
      />
    </Expander>

    <div class="two">
      <div class="col">
        <Markdown source={t("capas_titulo")} />
        <Slider id="nn-capas" label={t("capas_label")} help={t("capas_help")} min={1} max={MAX_LAYERS} value={nn.layers} onchange={setLayers} />
        <p class="caption"><Markdown inline source={layersCaption} /></p>
        {#each { length: nn.layers }, i (i)}
          <Slider
            id="nn-neuronas-{i}"
            label={fmt(t("neuronas_label"), { n: i + 1 })}
            help={fmt(t("neuronas_help"), { n: i + 1, sugerencia: view.suggested_neurons[i] ?? 10 })}
            min={1}
            max={MAX_NEURONS}
            value={nn.neuronsOf(i, view)}
            onchange={(v) => setNeurons(i, v)}
          />
        {/each}

        <Markdown source={t("activacion_titulo")} />
        <Select
          id="nn-activacion"
          label={t("activacion_label")}
          help={t("activacion_help")}
          options={hiddenOptions}
          value={nn.activation}
          onchange={(v: NnActivation) => (nn.activation = v)}
        />
        <Notice severity={activationNotice.severity} markdown={activationNotice.text} />

        <Markdown source={t("salida_titulo")} />
        <Select
          id="nn-salida"
          label={t("salida_label")}
          help={fmt(t("salida_help"), { tarea: taskName, recomendada: view.recommended_output })}
          options={outputOptions}
          value={output}
          onchange={(v: NnActivation) => view && nn.setOutput(view, v)}
        />
        <div data-output-check={view.output_check.key} data-pedagogical>
          <Notice severity={view.output_check.level} markdown={t(view.output_check.key)} />
          {#if view.output_check.untrainable}
            <div data-untrainable>
              <Notice severity="error" markdown={fmt("{m}", { m: view.output_check.untrainable })} />
            </div>
          {/if}
        </div>
        <Expander id="nn-salidas" label={t("salidas_expander")}>
          <Markdown source={md("redes/activaciones-perdidas.md")} />
        </Expander>
      </div>
      <div class="col">
        <Markdown source={t("diagrama_titulo")} />
        <NetworkDiagram rows={view.layers} activation={nn.activation} outputActivation={output} task={taskName} totalParams={view.total_params} />
      </div>
    </div>

    <Markdown source={t("adicional_titulo")} />
    <Markdown source={t("adicional_texto")} />
    <div class="three">
      <div class="col">
        <Markdown source={t("regularizacion_titulo")} />
        <Slider
          id="nn-dropout"
          label={t("dropout_label")}
          help={t("dropout_help")}
          min={0}
          max={0.8}
          step={0.1}
          value={nn.dropout}
          onchange={(v) => (nn.dropout = Math.round(v * 10) / 10)}
        />
        <p class="caption"><Markdown inline source={dropoutCaption} /></p>
      </div>
      <div class="col">
        <Markdown source={t("procesamiento_titulo")} />
        <Select id="nn-batch" label={t("batch_label")} help={t("batch_help")} options={batchOptions} value={nn.batchSize} onchange={(v: number) => (nn.batchSize = v)} />
        <p class="caption"><Markdown inline source={batchCaption} /></p>
      </div>
      <div class="col">
        <Markdown source={t("optimizacion_titulo")} />
        <Select
          id="nn-optimizador"
          label={t("optimizador_label")}
          help={t("optimizador_help")}
          options={optimizerOptions}
          value={nn.optimizer}
          onchange={(v: NnOptimizer) => (nn.optimizer = v)}
        />
        <p class="caption"><Markdown inline source={optimizerCaption} /></p>
      </div>
    </div>
    <Expander id="nn-consejos" label={t("consejos_expander")}>
      <Markdown source={fmt(md("redes/arquitectura-consejos.md"), { n_muestras: view.n_rows, batch_recomendado: view.size_tips.batch_recommended })} />
    </Expander>

    <Markdown source={t("resumen_titulo")} />
    <div class="three metrics" data-nn-summary>
      <Metric label={t("metrica_parametros")} value={thousands(view.total_params)} help={t("metrica_parametros_help")} />
      <Metric label={t("metrica_capas")} value={view.layers.length} help={t("metrica_capas_help")} />
      <div class="stack">
        <Metric
          label={t("metrica_complejidad")}
          value={complexityLevels[complexityIndex] ?? ""}
          help={fmt(t("metrica_complejidad_help"), { ratio: view.complexity_ratio })}
        />
        <Metric label={t("metrica_tipo")} value={t("metrica_tipo_valor")} />
      </div>
    </div>

    <Markdown source={t("detalles_titulo")} />
    <div data-nn-layers-table>
      <DataTable columns={tableColumns} rows={tableRows} index={tableRows.map((_, i) => String(i))} label={t("detalles_titulo")} />
    </div>

    <Markdown source={t("complejidad_titulo")} />
    {#if complexityNotice}
      <div data-pedagogical>
        <Notice severity={complexityNotice.severity} markdown={complexityNotice.text} />
      </div>
    {/if}

    <Markdown source={t("codigo_titulo")} />
    <button type="button" class="btn block" onclick={generate} disabled={view.output_check.untrainable !== null} data-nn-code-button>
      {t("btn_codigo")}
    </button>
    {#if codeError}
      <Notice severity="error" markdown={fmt("{e}", { e: codeError })} />
    {:else if code}
      {#if codeFor !== codeKey}
        <Notice severity="warning" markdown="⚠️ La configuración ha cambiado: vuelve a generar el código para actualizarlo." />
      {/if}
      <Markdown source={t("codigo_generado")} />
      <CodeBlock {code} id="nn-codigo" downloadName={`red_neuronal_arquitectura_${taskName === "regresión" ? "regresión" : "clasificación"}.py`} />
      <p class="caption">{t("codigo_nota")}</p>
    {/if}
  {/if}
</section>

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .two {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 28px;
    align-items: start;
    margin-top: 8px;
  }

  .three {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 24px;
    align-items: start;
  }

  .metrics {
    margin: 8px 0 14px;
  }

  .stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .caption {
    margin: -6px 2px 8px;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
  }

  @media (max-width: 860px) {
    .two {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
