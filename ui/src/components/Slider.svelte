<script lang="ts">
  interface Props {
    label: string;
    min: number;
    max: number;
    step?: number;
    value: number;
    /** Called on release / commit, like the Qt slider. */
    onchange: (value: number) => void;
    help?: string;
    id: string;
    /** Second line under the label (italic description of the prediction form). */
    hint?: string | undefined;
    /** Bold label (`**{nombre}**`). */
    strong?: boolean;
  }

  let { label, min, max, step = 1, value, onchange, help, id, hint, strong = false }: Props = $props();
  let dragging = $state<number | null>(null);
  const shown = $derived(dragging ?? value);
  const percent = $derived(max > min ? ((shown - min) / (max - min)) * 100 : 0);

  function clamp(v: number): number {
    return Math.min(max, Math.max(min, Math.round(v / step) * step));
  }

  function commit(v: number): void {
    dragging = null;
    const next = clamp(v);
    if (next !== value) {
      onchange(next);
    }
  }
</script>

<div class="field" title={help}>
  <label for={id} class:strong>{label}</label>
  {#if hint}<p class="hint">{hint}</p>{/if}
  <div class="row">
    <input
      {id}
      class="range"
      type="range"
      {min}
      {max}
      {step}
      value={shown}
      style:--fill="{percent}%"
      oninput={(event) => (dragging = Number(event.currentTarget.value))}
      onchange={(event) => commit(Number(event.currentTarget.value))}
    />
    <input
      class="number"
      type="number"
      aria-label={label}
      {min}
      {max}
      {step}
      value={shown}
      onchange={(event) => commit(Number(event.currentTarget.value))}
    />
  </div>
</div>

<style>
  .field {
    margin: 12px 0;
  }

  label {
    display: block;
    margin-bottom: 10px;
  }

  label.strong {
    font-weight: 700;
  }

  .hint {
    margin: -6px 0 10px;
    color: var(--color-text-muted);
    font-size: 13px;
    font-style: italic;
  }

  .row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 96px;
    gap: 14px;
    align-items: center;
  }

  .range {
    width: 100%;
    height: 6px;
    margin: 0;
    appearance: none;
    border-radius: 3px;
    background: linear-gradient(
      to right,
      var(--color-slider-fill) 0 var(--fill),
      var(--color-slider-groove) var(--fill) 100%
    );
    cursor: pointer;
  }

  .range::-webkit-slider-thumb {
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--color-bg);
    border: 2px solid var(--color-slider-handle);
  }

  .range:focus-visible {
    outline: 2px solid var(--color-focus);
    outline-offset: 4px;
  }

  .number {
    width: 100%;
    min-height: 42px;
    padding: 8px 10px;
    background: var(--color-input-bg);
    border: 1px solid var(--color-input-border);
    border-radius: var(--radius-box);
  }

  .number:focus-visible {
    outline: none;
    border-color: var(--color-input-focus-border);
  }
</style>
