<script lang="ts">
  /** Row of `st.metric` cards with an optional quality badge under each (C1/C2/C5). */
  import Metric from "../../../components/Metric.svelte";
  import Notice from "../../../components/Notice.svelte";
  import type { LevelNotice } from "../../ml/levels";

  export interface MetricItem {
    label: string;
    value: string | number;
    help?: string | undefined;
    badge?: LevelNotice | null | undefined;
  }

  interface Props {
    items: readonly MetricItem[];
  }

  let { items }: Props = $props();
</script>

<div class="grid" style:--columns={items.length} data-metric-grid>
  {#each items as item, i (i)}
    <div class="cell">
      <Metric label={item.label} value={item.value} {...item.help ? { help: item.help } : {}} />
      {#if item.badge}
        <Notice severity={item.badge.severity} markdown={item.badge.markdown} />
      {/if}
    </div>
  {/each}
</div>

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(var(--columns), minmax(0, 1fr));
    gap: 16px;
    margin: 10px 0 14px;
  }

  .cell :global(.notice) {
    margin: 8px 0 0;
    padding: 10px 12px;
  }
</style>
