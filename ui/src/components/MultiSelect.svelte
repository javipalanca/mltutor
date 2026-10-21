<script lang="ts">
  interface Props {
    label: string;
    options: readonly string[];
    value: readonly string[];
    /** `max_selections`. */
    max?: number;
    onchange: (value: string[]) => void;
    help?: string;
  }

  let { label, options, value, max, onchange, help }: Props = $props();
  // A generated id: the label may contain spaces, and `aria-labelledby` splits on them.
  const uid = $props.id();
  const limitReached = $derived(max !== undefined && value.length >= max);
  const summary = $derived(
    `${value.length} de ${options.length} seleccionadas${max === undefined ? "" : ` · Máximo ${max}`}`,
  );

  function toggle(option: string): void {
    if (value.includes(option)) {
      onchange(value.filter((v) => v !== option));
    } else if (!limitReached) {
      // Keep the order of the options, like Qt's chips.
      onchange(options.filter((o) => o === option || value.includes(o)));
    }
  }
</script>

<div class="field" title={help}>
  <div class="label" id="{uid}-label">{label}</div>
  <div class="box" role="group" aria-labelledby="{uid}-label" aria-describedby="{uid}-summary">
    <div class="summary" id="{uid}-summary" aria-live="polite">{summary}</div>
    <div class="chips">
      {#each options as option (option)}
        {@const on = value.includes(option)}
        <button
          type="button"
          class="chip"
          class:on
          aria-pressed={on}
          disabled={!on && limitReached}
          onclick={() => toggle(option)}
        >
          <span class="tick" aria-hidden="true">{on ? "✓" : ""}</span>{option}
        </button>
      {/each}
    </div>
  </div>
</div>

<style>
  .field {
    margin: 12px 0;
  }

  .label {
    margin-bottom: 8px;
  }

  .box {
    padding: 10px 12px 12px;
    background: var(--color-input-bg);
    border: 1px solid var(--color-input-border);
    border-radius: 12px;
  }

  .summary {
    margin-bottom: 8px;
    color: var(--color-text-muted);
    font-size: 12px;
  }

  .chips {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 8px;
  }

  .chip {
    display: flex;
    align-items: center;
    gap: 6px;
    min-height: 42px;
    padding: 10px 12px;
    text-align: left;
    color: var(--color-chip-text);
    background: var(--color-chip-bg);
    border: 1px solid var(--color-chip-border);
    border-radius: 8px;
    cursor: pointer;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }

  .chip:hover:not(:disabled) {
    border-color: #93baf0;
    background: var(--color-hover-bg);
  }

  .chip.on {
    color: var(--color-chip-on-text);
    background: var(--color-chip-on-bg);
    border-color: var(--color-chip-on-border);
    font-weight: 700;
  }

  .chip:disabled {
    color: var(--color-disabled-text);
    background: var(--color-disabled-bg);
    cursor: default;
  }

  .chip:focus-visible {
    outline: 2px solid var(--color-focus);
    outline-offset: 2px;
  }

  .tick {
    width: 12px;
    flex: none;
  }
</style>
