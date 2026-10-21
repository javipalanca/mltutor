<script lang="ts">
  /** 🔍 Características (§2.5): C10 with the panel of D38 (native by default, as Python). */
  import ButtonPanel from "../../../components/ButtonPanel.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { FeatureImportanceView } from "../../bindings/FeatureImportanceView";
  import type { ImportanceMethod } from "../../bindings/ImportanceMethod";
  import FeatureImportanceChart from "../../components/ml/FeatureImportanceChart.svelte";
  import { arbRaw, arbText, compText, md } from "../../content";
  import { fmt } from "../../fmt";
  import { models } from "../../ml/modelSession.svelte";
  import { ViewLoader } from "../../ml/viewLoader.svelte";
  import TreeBanner from "./TreeBanner.svelte";
  import { LESSON, trees } from "./state.svelte";

  const lessonState = models.lesson(LESSON);
  const stale = (): void => void models.refresh(LESSON);

  type Panel = readonly (readonly [string, string, string])[];
  const METHODS: Readonly<Record<string, ImportanceMethod>> = { Nativa: "native", Permutacion: "permutation" };
  const options = ((arbRaw("caracteristicas", "panel") as Panel | undefined) ?? []).map(([label, value]) => ({
    label,
    value: METHODS[value] ?? "native",
  }));

  const view = new ViewLoader<FeatureImportanceView>();
  const key = $derived(`${lessonState.info?.model_id ?? ""}|${trees.importance}`);

  $effect(() => {
    const info = lessonState.info;
    const method = trees.importance;
    if (!info) {
      return;
    }
    return view.load(`${info.model_id}|${method}`, () => backend.featureImportance(info.model_id, method), stale);
  });

  const data = $derived(view.for(key));
</script>

<h2>{arbText("caracteristicas", "header")}</h2>
{#if !lessonState.info}
  <Notice severity="warning" markdown={arbText("comun", "sin_entrenar")} />
{:else}
  <TreeBanner info={lessonState.info} />
  <ButtonPanel label="Método de importancia" value={trees.importance} {options} onchange={(value) => (trees.importance = value)} />
  <p class="caption">
    {arbText("caracteristicas", trees.importance === "native" ? "panel_nativa_nota" : "panel_permutacion_nota")}
  </p>
  {#if view.error}
    {#if trees.importance === "permutation"}
      <Notice severity="error" markdown={fmt(compText("importancia", "error_perm"), { error: view.error })} />
      <Notice severity="info" markdown={compText("importancia", "error_perm_info")} />
    {:else}
      <Notice severity="error" markdown={fmt("{error}", { error: view.error })} />
    {/if}
  {:else if data}
    <FeatureImportanceChart view={data} idPrefix="arboles" />
  {:else if trees.importance === "permutation"}
    <Notice severity="info" markdown={compText("importancia", "calculando")} />
    <Spinner label={compText("importancia", "spinner")} />
  {:else}
    <Spinner label="Calculando la importancia…" />
  {/if}
  <Expander id="arboles-comparar-importancias" label={arbText("caracteristicas", "comparar_expander")}>
    <Markdown source={md("arboles/importancia.md")} />
  </Expander>
{/if}

<style>
  .caption {
    margin: -2px 2px 10px;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
  }

  h2 {
    margin: 26px 0 14px;
    color: var(--color-heading);
    font-size: 23px;
  }
</style>
