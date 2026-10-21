<script lang="ts">
  /**
   * C20 (D58): "#### 👥 Los {k} vecinos más cercanos" of a prediction (Vecino, Distancia,
   * Clase/Valor, Peso and the features), in scikit-learn's order, with the note of
   * distance 0 when it decides the weights.
   */
  import DataTable from "../../../components/DataTable.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import type { Cell } from "../../bindings/Cell";
  import type { KnnWeights } from "../../bindings/KnnWeights";
  import type { NeighborRow } from "../../bindings/NeighborRow";
  import type { TieAtKth } from "../../bindings/TieAtKth";
  import Expander from "../../../components/Expander.svelte";
  import { knnList, knnText, md } from "../../content";
  import { fmt, fmtText } from "../../fmt";
  import { fixed, formatG } from "../../format";

  interface Props {
    neighbors: readonly NeighborRow[];
    featureNames: readonly string[];
    /** By class code; `null` in regression. */
    classNames: readonly string[] | null;
    weights: KnnWeights;
    /** D52: samples at the K-th distance that do not all fit in the table. */
    tie?: TieAtKth | null;
  }

  let { neighbors, featureNames, classNames, weights, tie = null }: Props = $props();
  const t = (key: string): string => knnText("predicciones", key);
  const classifier = $derived(classNames !== null);
  const columns = $derived([
    ...knnList("predicciones", classifier ? "vecinos_columnas_clasificacion" : "vecinos_columnas_regresion"),
    ...featureNames,
  ]);
  const text = (v: string): Cell => ({ kind: "text", v });
  const rows = $derived(
    neighbors.map((n, i) => [
      text(String(i + 1)),
      text(fixed(n.distance, 4)),
      text(
        classifier
          ? (classNames?.[n.class_index ?? 0] ?? `Clase ${n.class_index ?? 0}`)
          : n.value === null
            ? "—"
            : fixed(n.value, 4),
      ),
      text(weights === "uniform" ? "1" : formatG(n.weight, 4)),
      ...n.features.map((v) => text(formatG(v, 6))),
    ]),
  );
  const zeros = $derived(neighbors.filter((n) => n.distance === 0).length);
</script>

<section class="neighbors" data-knn-neighbors={neighbors.length}>
  <Markdown source={fmt(t("vecinos_titulo"), { k: neighbors.length })} />
  <DataTable
    label="Vecinos más cercanos"
    {columns}
    {rows}
    hideIndex
    numericColumns={[1, ...(classifier ? [] : [2]), 3, ...featureNames.map((_, i) => i + 4)]}
  />
  {#if tie}
    <p class="caption">{fmtText(t("vecinos_nota_empate"), { n: tie.n_tied, huecos: tie.slots, distancia: tie.distance })}</p>
  {/if}
  {#if zeros > 0 && weights === "distance"}
    <Notice severity="info" markdown={fmt(t("vecinos_nota_cero"), { n: zeros })} />
  {/if}
  <Expander id="knn-prediccion-vecinos" label={t("vecinos_expander")}>
    <Markdown source={md("knn/prediccion-vecinos.md")} />
  </Expander>
</section>

<style>
  .caption {
    margin: -4px 2px 8px;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
  }
</style>
