<script lang="ts">
  /** `st.number_input` (integer): field with − / + steppers, clamped to `min..max`. */
  interface Props {
    label: string;
    min: number;
    max: number;
    step?: number;
    value: number;
    onchange: (value: number) => void;
    help?: string;
    id: string;
  }

  let { label, min, max, step = 1, value, onchange, help, id }: Props = $props();

  function clamp(v: number): number {
    if (!Number.isFinite(v)) {
      return value;
    }
    // `toFixed(10)` drops the binary noise of decimal steps (0.1 · 3 = 0.30000000000000004).
    return Math.min(max, Math.max(min, Number((Math.round(v / step) * step).toFixed(10))));
  }

  function commit(v: number, input?: HTMLInputElement): void {
    const next = clamp(v);
    if (input) {
      input.value = String(next);
    }
    if (next !== value) {
      onchange(next);
    }
  }
</script>

<div class="field" title={help}>
  <label for={id}>{label}</label>
  <div class="control">
    <input
      {id}
      type="number"
      inputmode="numeric"
      {min}
      {max}
      {step}
      {value}
      onchange={(event) => commit(Number(event.currentTarget.value), event.currentTarget)}
    />
    <div class="steppers">
      <button type="button" aria-label="Disminuir {label}" disabled={value <= min} onclick={() => commit(value - step)}>−</button>
      <button type="button" aria-label="Aumentar {label}" disabled={value >= max} onclick={() => commit(value + step)}>+</button>
    </div>
  </div>
</div>

<style>
  .field {
    margin: 12px 0;
  }

  label {
    display: block;
    margin-bottom: 8px;
  }

  .control {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    min-height: 46px;
    background: var(--color-input-bg);
    border: 1px solid var(--color-input-border);
    border-radius: var(--radius-box);
    overflow: hidden;
  }

  .control:focus-within {
    border-color: var(--color-input-focus-border);
    outline: 2px solid var(--color-focus);
    outline-offset: 2px;
  }

  .control:has(.steppers button:focus-visible) {
    outline: none;
  }

  .steppers button:focus-visible {
    outline: 2px solid var(--color-focus);
    outline-offset: -2px;
  }

  input {
    width: 100%;
    min-width: 0;
    padding: 10px 14px;
    background: transparent;
    border: 0;
    appearance: textfield;
  }

  input:focus-visible {
    outline: none;
  }

  input::-webkit-inner-spin-button,
  input::-webkit-outer-spin-button {
    appearance: none;
    margin: 0;
  }

  .steppers {
    display: flex;
  }

  .steppers button {
    width: 40px;
    border: 0;
    border-left: 1px solid var(--color-input-border);
    background: transparent;
    color: var(--color-text);
    font-size: 18px;
    cursor: pointer;
  }

  .steppers button:hover:not(:disabled) {
    background: var(--color-input-hover-bg);
  }

  .steppers button:disabled {
    color: var(--color-disabled-text);
    cursor: default;
  }
</style>
