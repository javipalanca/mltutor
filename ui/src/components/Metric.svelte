<script lang="ts">
  interface Props {
    label: string;
    value: string | number;
    delta?: string;
    help?: string;
  }

  let { label, value, delta, help }: Props = $props();
</script>

<div class="metric" title={help}>
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  {#if delta}
    <div class="delta">{delta}</div>
  {/if}
</div>

<style>
  .metric {
    min-width: 0;
    padding: 14px 18px;
    background: var(--color-metric-bg);
    border: 1px solid var(--color-metric-border);
    border-radius: var(--radius-box);
  }

  .label {
    color: var(--color-text);
    font-size: 14px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .value {
    margin-top: 4px;
    color: var(--color-metric-value);
    font-size: 26px;
    font-weight: 700;
    line-height: 1.2;
    overflow-wrap: anywhere;
    user-select: text;
  }

  .delta {
    margin-top: 2px;
    color: var(--color-text-muted);
    font-size: 13px;
  }
</style>
