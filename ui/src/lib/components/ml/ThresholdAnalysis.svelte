<script lang="ts">
  /** C6: decision thresholds of a binary classifier (current, best by F1 and comparison, D21). */
  import type { Cell } from "../../bindings/Cell";
  import type { ThresholdRow } from "../../bindings/ThresholdRow";
  import DataTable from "../../../components/DataTable.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Metric from "../../../components/Metric.svelte";
  import Notice from "../../../components/Notice.svelte";
  import { compList, compRaw, compText, md, optionalText } from "../../content";
  import { fmt } from "../../fmt";
  import { fixed } from "../../format";

  interface Props {
    thresholds: readonly ThresholdRow[];
    best: number | null;
    /** Thresholds of the comparison table (0.3, 0.5, 0.7). */
    shown: readonly number[];
    idPrefix: string;
  }

  let { thresholds, best, shown, idPrefix }: Props = $props();

  const t = compText;
  const rows = $derived(thresholds.filter((r) => shown.some((s) => Math.abs(s - r.threshold) < 1e-9)));
  const text = (v: string): Cell => ({ kind: "text", v });
  const cells = $derived(
    rows.map((r) => [text(fixed(r.threshold, 1)), text(fixed(r.accuracy, 3)), text(fixed(r.precision, 3)), text(fixed(r.recall, 3)), text(fixed(r.f1, 3))]),
  );
</script>

{#if thresholds.length > 0}
  <section class="thresholds" data-thresholds>
    <Markdown source={t("umbrales", "titulo")} />
    <Expander id="{idPrefix}-umbrales" label={t("umbrales", "expander")}>
      <Markdown source={md("componentes/umbrales.md")} />
    </Expander>
    <div class="columns">
      <div class="metrics">
        <Metric label={t("umbrales", "actual")} value="0.50" />
        {#if best !== null}
          <Metric label={t("umbrales", "optimo")} value={fixed(best, 2)} />
          {#if Math.abs(best - 0.5) > 0.1}
            <Notice severity="info" markdown={fmt(t("umbrales", "consejo"), { umbral: best })} />
          {/if}
        {/if}
      </div>
      <div>
        <Markdown source={t("umbrales", "tabla_titulo")} />
        <DataTable
          label="Comparación de umbrales"
          columns={compList("umbrales", "tabla_columnas")}
          rows={cells}
          hideIndex
          numericColumns={[0, 1, 2, 3, 4]}
          compact
        />
        <p class="caption">{optionalText(compRaw, "umbrales", "tabla_nota", "")}</p>
      </div>
    </div>
  </section>
{/if}

<style>
  .columns {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 28px;
    align-items: start;
  }

  .caption {
    margin: 0 2px;
    color: var(--color-text-muted);
    font-size: 13px;
  }

  .metrics {
    display: grid;
    gap: 14px;
  }
</style>
