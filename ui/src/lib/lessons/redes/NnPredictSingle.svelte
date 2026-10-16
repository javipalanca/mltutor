<script lang="ts">
  /**
   * 🔮 Predicción individual (§2.6): C9 with the unscaled data of the whole dataset; Rust
   * scales the input with the scaler of the training (D72). Binary: "Probabilidad clase
   * positiva" inside the card and the threshold note (> 0,5, D87); multiclass: the table;
   * regression: the value. A tanh output is not a probability. NEW: a warning when a value
   * leaves the range of the training rows (the network extrapolates).
   */
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { BatchTemplate } from "../../bindings/BatchTemplate";
  import type { FormControl } from "../../bindings/FormControl";
  import type { PredictionForm as Form } from "../../bindings/PredictionForm";
  import type { TrainedModelInfo } from "../../bindings/TrainedModelInfo";
  import PredictionForm from "../../components/ml/PredictionForm.svelte";
  import PredictionResultCard from "../../components/ml/PredictionResultCard.svelte";
  import { compText, nnText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fmt, fmtText } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import { LESSON } from "./state.svelte";

  interface Props {
    info: TrainedModelInfo;
    template: BatchTemplate | null;
  }

  let { info, template }: Props = $props();
  const t = (key: string): string => nnText("predicciones", key);
  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const form = new ViewLoader<Form>();
  let predicting = $state(false);
  let error = $state<string | null>(null);

  $effect(() => {
    const id = info.model_id;
    return form.load(String(id), () => backend.predictionForm(id), stale);
  });

  const controls = $derived(form.for(String(info.model_id))?.controls ?? null);

  function defaults(list: readonly FormControl[]): number[] {
    return list.map((c) => {
      switch (c.control.kind) {
        case "checkbox":
          return c.control.default ? 1 : 0;
        case "select":
          return c.control.options[c.control.default_index]?.code ?? 0;
        case "slider":
          return c.control.default;
      }
    });
  }

  const values = $derived(controls ? (lessonState.form[info.model_id] ?? defaults(controls)) : []);
  const result = $derived(lessonState.prediction[info.model_id] ?? null);
  const output = $derived(info.config.kind === "neural_net" ? info.config.arch.output_activation : "linear");
  const tanh = $derived(output === "tanh");
  // ReLU hidden layers extrapolate linearly without limit: their own warning.
  const reluHidden = $derived(info.config.kind === "neural_net" && info.config.arch.activation === "relu");
  const binary = $derived(result?.kind === "class" && (result.proba?.length ?? 0) === 2 && info.nn?.layers.at(-1) === 1);
  const positive = $derived(result?.kind === "class" ? (result.proba?.[1] ?? null) : null);

  /** Probability of the positive class (or the tanh output) inside the card. */
  const detail = $derived.by(() => {
    if (!binary || !positive) {
      return null;
    }
    return fmtText(t(tanh ? "salida_tanh_binaria" : "probabilidad_positiva"), { p: positive.probability });
  });

  const short = (v: number): string => String(Number(v.toPrecision(6)));

  /** Values of the last prediction outside the range of the training rows. */
  let predicted = $state<number[] | null>(null);
  const outside = $derived.by(() => {
    if (!template || !predicted || !controls) {
      return [];
    }
    const items: string[] = [];
    for (const c of controls) {
      const i = c.index;
      const v = predicted[controls.indexOf(c)];
      const min = template.train_min[i];
      const max = template.train_max[i];
      if (v === undefined || min === undefined || max === undefined || c.control.kind !== "slider") {
        continue;
      }
      if (v < min || v > max) {
        items.push(fmtText(t("fuera_rango_item"), { caracteristica: c.label, valor: short(v), min: short(min), max: short(max) }));
      }
    }
    return items;
  });

  async function predict(): Promise<void> {
    const id = info.model_id;
    const sent = [...values];
    predicting = true;
    error = null;
    try {
      lessonState.prediction[id] = await backend.predictOne(id, sent);
      predicted = sent;
    } catch (reason) {
      if (backend.isStale(reason)) {
        stale();
      } else {
        error = backend.errorMessage(reason);
      }
    } finally {
      predicting = false;
    }
  }
</script>

<section data-nn-predict-single>
  {#if form.error}
    <Notice severity="error" markdown={escapeMarkdown(form.error)} />
  {:else if !controls}
    <Spinner label="Preparando el formulario…" />
  {:else}
    <PredictionForm {controls} {values} onchange={(next) => (lessonState.form[info.model_id] = next)} idPrefix="nn-pred" />
    <p class="caption">{t("escalado_nota")}</p>
    <button type="button" class="btn block primary" disabled={predicting} onclick={() => void predict()} data-predict-button>
      {compText("prediccion", "boton")}
    </button>
    {#if error}
      <Notice severity="error" markdown={error.startsWith("❌") ? error : fmt(t("error_prediccion"), { error })} />
    {:else if result}
      <PredictionResultCard result={binary && result.kind === "class" ? { ...result, proba: null } : result} {detail} />
      {#if binary && positive && !tanh}
        <p class="caption">{fmtText(t("umbral_nota"), { positiva: positive.name })}</p>
      {/if}
      {#if tanh && !binary && result.kind === "class"}
        <p class="caption" data-nn-not-probability>{t("nota_tanh_multiclase")}</p>
      {/if}
      {#if outside.length > 0}
        <div data-nn-out-of-range>
          <Notice severity="warning" markdown={fmt(t(reluHidden ? "fuera_rango_relu" : "fuera_rango"), { detalles: outside.join("; ") })} />
        </div>
      {/if}
    {/if}
  {/if}
</section>

<style>
  .caption {
    margin: 4px 2px 10px;
    color: var(--color-text-muted);
    font-size: 13px;
  }

  .btn.primary {
    margin-top: 8px;
  }
</style>
