<script lang="ts">
  /**
   * 🔮 Predicciones of 🧠 Redes Neuronales (§2.6 and C29): "🔮 Predicción individual" (C9, as
   * Python) and, NEW (D81), "📊 Predicción por lotes" and "🎲 Exploración interactiva". A
   * diverged network (K1) cannot predict: `[comun] red_no_valida` instead.
   */
  import ButtonPanel from "../../../components/ButtonPanel.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { BatchTemplate } from "../../bindings/BatchTemplate";
  import type { HistoryView } from "../../bindings/HistoryView";
  import { nnRaw, nnText } from "../../content";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import NnBanner from "./NnBanner.svelte";
  import NnBatchPredict from "./NnBatchPredict.svelte";
  import NnExplore from "./NnExplore.svelte";
  import NnPredictSingle from "./NnPredictSingle.svelte";
  import { invalidNetworkText } from "./nnTexts";
  import { nnPredictions, type PredictionSubview } from "./predictions.svelte";
  import { LESSON } from "./state.svelte";

  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);
  const t = (key: string): string => nnText("predicciones", key);

  type Panel = readonly (readonly [string, PredictionSubview, string])[];
  const options = ((nnRaw("predicciones", "subvistas") as Panel | undefined) ?? []).map(([label, value]) => ({ label, value }));

  const history = new ViewLoader<HistoryView>();
  const template = new ViewLoader<BatchTemplate>();
  $effect(() => {
    const info = lessonState.info;
    if (!info) {
      return;
    }
    const id = info.model_id;
    const stops = [
      history.load(String(id), () => backend.nnHistory(id), stale),
      template.load(String(id), () => backend.nnBatchTemplate(id), stale),
    ];
    return () => stops.forEach((stop) => stop());
  });
  const key = $derived(String(lessonState.info?.model_id ?? ""));
  const hist = $derived(history.for(key));
  const tmpl = $derived(template.for(key));
</script>

<section class="nn-predictions" data-nn-predictions={nnPredictions.subview}>
  <h2>{t("header")}</h2>
  {#if !lessonState.info}
    <Notice severity="warning" markdown={nnText("comun", "sin_entrenar")} />
  {:else}
    {@const info = lessonState.info}
    <NnBanner {info} />
    {#if !hist}
      <Spinner label="Cargando la red…" />
    {:else if hist.diverged_epoch !== null}
      <div data-nn-diverged={hist.diverged_epoch} data-pedagogical>
        <Notice severity="error" markdown={invalidNetworkText()} />
      </div>
    {:else}
      <ButtonPanel label="Predicciones" value={nnPredictions.subview} {options} onchange={(value) => (nnPredictions.subview = value)} />
      {#if nnPredictions.subview === "Lotes"}
        <NnBatchPredict {info} template={tmpl} />
      {:else if nnPredictions.subview === "Exploracion"}
        <NnExplore {info} />
      {:else}
        <NnPredictSingle {info} template={tmpl} />
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
</style>
