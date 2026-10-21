<script lang="ts">
  /**
   * 💾 Exportar of 🧠 Redes Neuronales (§2.7, D75): "ℹ️ Información del Modelo" (with the date
   * of the training, not of now), "📁 Opciones de Exportación" with the four panels of
   * `[exportar_rust]` (📦 Paquete Keras, 💾 ONNX, 📝 Código Python, 📋 Metadatos), the
   * "ℹ️ ¿Qué formato elijo?" expander and "💡 Información Adicional". Every file is built in
   * Rust and saved with the native dialog. A network with non-finite weights (K1) is not exported.
   */
  import ButtonPanel from "../../../components/ButtonPanel.svelte";
  import CodeBlock from "../../../components/CodeBlock.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { ExportFormat } from "../../bindings/ExportFormat";
  import type { HistoryView } from "../../bindings/HistoryView";
  import ModelCode from "../../components/ml/ModelCode.svelte";
  import { hasMd, md, nnRaw, nnText, optionalText, regText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fmt, fmtText } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import { status } from "../../status.svelte";
  import NnBanner from "./NnBanner.svelte";
  import { invalidNetworkText, networkBlocked } from "./nnTexts";
  import { nnPredictions, type ExportSubview } from "./predictions.svelte";
  import { LESSON, thousands } from "./state.svelte";

  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const t = (key: string): string => nnText("exportar", key);
  const r = (key: string): string => nnText("exportar_rust", key);

  type Panel = readonly (readonly [string, ExportSubview, string])[];
  const options = ((nnRaw("exportar_rust", "botones") as Panel | undefined) ?? []).map(([label, value]) => ({ label, value }));

  const history = new ViewLoader<HistoryView>();
  const metadata = new ViewLoader<string>();
  $effect(() => {
    const info = lessonState.info;
    if (!info) {
      return;
    }
    const id = info.model_id;
    return history.load(String(id), () => backend.nnHistory(id), stale);
  });
  $effect(() => {
    const info = lessonState.info;
    if (!info || nnPredictions.exportView !== "Metadatos") {
      return;
    }
    const id = info.model_id;
    const at = trainedAt(info.nn?.trained_at_ms ?? 0);
    return metadata.load(String(id), () => backend.nnMetadata(id, at), stale);
  });
  const key = $derived(String(lessonState.info?.model_id ?? ""));
  const hist = $derived(history.for(key));
  const json = $derived(metadata.for(key));

  const pad = (n: number): string => String(n).padStart(2, "0");
  /** "2026-10-15 17:49": the local date of the training (D75; Python showed the current time). */
  function trainedAt(ms: number): string {
    const d = new Date(ms);
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  const infoCards = $derived.by(() => {
    const info = lessonState.info;
    if (!info || info.config.kind !== "neural_net" || !info.nn) {
      return null;
    }
    const { arch } = info.config;
    const layers = info.nn.layers;
    const total = layers.slice(1).reduce((sum, n, i) => sum + n * (layers[i] ?? 0) + n, 0);
    return {
      arch: fmt(t("info_arquitectura"), {
        tarea: info.model_task === "classification" ? "Clasificación" : "Regresión",
        n_capas: layers.length,
        arquitectura: `[${layers.join(", ")}]`,
        activacion: arch.activation,
        optimizador: arch.optimizer,
      }),
      params: fmt(t("info_parametros").replace("{total:,}", thousands(total)), {
        dropout: String(Math.round(arch.dropout * 10) / 10),
        batch: arch.batch_size,
        fecha: trainedAt(info.nn.trained_at_ms),
      }),
    };
  });

  let saving = $state<ExportFormat | null>(null);
  let result = $state<{ format: ExportFormat; severity: "success" | "error"; markdown: string } | null>(null);

  async function save(format: ExportFormat, okKey: string): Promise<void> {
    const info = lessonState.info;
    if (!info) {
      return;
    }
    saving = format;
    result = null;
    try {
      const saved = await backend.exportModel(info.model_id, format, trainedAt(info.nn?.trained_at_ms ?? 0));
      if (saved) {
        const markdown = okKey ? fmt(r(okKey), { ruta: saved }) : fmt(regText("exportar", "exito_guardado"), { archivo: saved });
        result = { format, severity: "success", markdown };
        status.flash(`Guardado: ${saved}`);
      }
    } catch (error) {
      if (backend.isStale(error)) {
        stale();
      } else {
        const message = backend.errorMessage(error);
        result = {
          format,
          severity: "error",
          markdown: message.startsWith("❌") || message.startsWith("🚫") ? message : fmt(regText("exportar", "error_guardado"), { error: message }),
        };
      }
    } finally {
      saving = null;
    }
  }

  const view = $derived(options.some((o) => o.value === nnPredictions.exportView) ? nnPredictions.exportView : "Keras");
</script>

<section class="nn-export" data-nn-export={view}>
  <h2>{t("header")}</h2>
  {#if !lessonState.info}
    <Notice severity="warning" markdown={nnText("comun", "sin_entrenar")} />
  {:else}
    {@const info = lessonState.info}
    <NnBanner {info} />
    <h3>{t("info_titulo")}</h3>
    {#if infoCards}
      <div class="cards" data-nn-export-info>
        <Notice severity="info" markdown={infoCards.arch} />
        <Notice severity="info" markdown={infoCards.params} />
      </div>
    {/if}
    {#if !hist}
      <Spinner label="Cargando la red…" />
    {:else if networkBlocked(hist)}
      <div data-nn-diverged={hist.diverged_epoch} data-pedagogical>
        <Notice severity="error" markdown={invalidNetworkText()} />
      </div>
    {:else}
      <h3>{t("opciones_titulo")}</h3>
      <ButtonPanel label="Exportar" value={view} {options} onchange={(value) => ((nnPredictions.exportView = value), (result = null))} />
      <div class="panel" data-export-panel>
        {#if view === "Keras"}
          <Markdown source={r("keras_titulo")} />
          <Markdown source={r("keras_texto")} />
          <Notice severity="info" markdown={r("keras_cuando")} />
          <button type="button" class="btn block primary" disabled={saving !== null} onclick={() => void save("keras_package", "keras_ok")} data-export-format="keras_package">
            {r("btn_keras")}
          </button>
        {:else if view === "Onnx"}
          <Markdown source={r("onnx_titulo")} />
          <Markdown source={r("onnx_texto")} />
          <Markdown source={r(info.model_task === "classification" ? "onnx_salidas_clasificacion" : "onnx_salidas_regresion")} />
          <Notice severity="info" markdown={r("onnx_cuando")} />
          <button type="button" class="btn block primary" disabled={saving !== null} onclick={() => void save("onnx", "onnx_ok")} data-export-format="onnx">
            {r("btn_onnx")}
          </button>
        {:else if view === "Python"}
          <Markdown source={t("python_titulo")} />
          <Markdown source={r("python_texto")} />
          <Notice severity="info" markdown={r("python_cuando")} />
          <!-- Decision of F5d: the script alone cannot run, so it is saved with its modelo.json. -->
          <ModelCode modelId={info.model_id} template="nn_numpy_predictor" title="predecir_numpy.py" id="nn-export-numpy" onstale={stale} />
          <button
            type="button"
            class="btn block primary"
            disabled={saving !== null}
            onclick={() => void save("nn_numpy_predictor", "")}
            data-export-format="nn_numpy_predictor"
          >
            {fmtText(optionalText(nnRaw, "exportar_rust", "btn_numpy", "💾 Guardar predictor NumPy ({archivo})…"), { archivo: r("archivo_numpy") })}
          </button>
          <Markdown source={r("codigo_entrenamiento")} />
          <div data-export-format="nn_training">
            <ModelCode modelId={info.model_id} template="nn_training" title={r("archivo_entrenamiento")} downloadName={r("archivo_entrenamiento")} id="nn-export-entrenamiento" onstale={stale} />
          </div>
        {:else}
          <Markdown source={t("metadatos_titulo")} />
          <Markdown source={r("metadatos_texto")} />
          {#if metadata.error}
            <Notice severity="error" markdown={escapeMarkdown(metadata.error)} />
          {:else if json}
            <div data-nn-metadata>
              <CodeBlock code={json} language="json" />
            </div>
            <p class="caption">{r("metadatos_fecha_nota")}</p>
          {:else}
            <Spinner label="Preparando los metadatos…" />
          {/if}
          <Notice severity="info" markdown={r("metadatos_cuando")} />
          <button type="button" class="btn block" disabled={saving !== null} onclick={() => void save("nn_metadata", "")} data-export-format="nn_metadata">
            {t("metadatos_descargar")}
          </button>
        {/if}
        {#if result}
          <div data-export-result={result.severity}>
            <Notice severity={result.severity} markdown={result.markdown} />
          </div>
        {/if}
      </div>
      {#if hasMd("redes/exportar.md")}
        <Expander id="nn-export-ayuda" label={r("ayuda_expander")}>
          <Markdown source={md("redes/exportar.md")} />
        </Expander>
      {/if}
    {/if}
    <h3>{t("adicional_titulo")}</h3>
    <Notice severity="info" markdown={md("redes/exportar-recomendaciones.md")} />
  {/if}
</section>

<style>
  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }

  h3 {
    margin: 18px 0 10px;
    color: var(--color-heading);
    font-size: 20px;
  }

  .cards {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
  }

  .cards :global(.notice) {
    margin: 0;
  }

  @container page (max-width: 420px) {
    .cards {
      grid-template-columns: 1fr;
    }
  }

  .caption {
    margin: 4px 2px 10px;
    color: var(--color-text-muted);
    font-size: 13px;
  }

  .btn {
    margin: 12px 0;
  }
</style>
