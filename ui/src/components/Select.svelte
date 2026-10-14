<script lang="ts" generics="T">
  interface Option {
    label: string;
    value: T;
  }

  interface Props {
    label: string;
    options: readonly Option[];
    value: T;
    onchange: (value: T) => void;
    help?: string;
    disabled?: boolean;
    id: string;
    /** Second line under the label (italic description of the prediction form). */
    hint?: string | undefined;
    /** Bold label (`**{nombre}**`). */
    strong?: boolean;
  }

  let { label, options, value, onchange, help, disabled = false, id, hint, strong = false }: Props = $props();
  const selectedIndex = $derived(options.findIndex((option) => option.value === value));

  function change(event: Event): void {
    const index = Number((event.currentTarget as HTMLSelectElement).value);
    const option = options[index];
    if (option) {
      onchange(option.value);
    }
  }
</script>

<div class="field">
  <label for={id} title={help} class:strong>{label}</label>
  {#if hint}<p class="hint">{hint}</p>{/if}
  <div class="control">
    <select {id} {disabled} value={String(selectedIndex)} onchange={change} title={help}>
      {#each options as option, index (index)}
        <option value={String(index)}>{option.label}</option>
      {/each}
    </select>
    <span class="chevron" aria-hidden="true">⌄</span>
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

  label.strong {
    font-weight: 700;
  }

  .hint {
    margin: -4px 0 8px;
    color: var(--color-text-muted);
    font-size: 13px;
    font-style: italic;
  }

  .control {
    position: relative;
  }

  select {
    width: 100%;
    min-height: 46px;
    padding: 10px 42px 10px 14px;
    appearance: none;
    background: var(--color-input-bg);
    border: 1px solid var(--color-input-border);
    border-radius: var(--radius-box);
    cursor: pointer;
  }

  select:hover:not(:disabled) {
    background: var(--color-input-hover-bg);
    border-color: var(--color-input-hover-border);
  }

  select:focus-visible {
    outline: none;
    background: var(--color-bg);
    border-color: var(--color-input-focus-border);
  }

  select:disabled {
    color: var(--color-disabled-text);
    cursor: default;
  }

  .chevron {
    position: absolute;
    right: 16px;
    top: 50%;
    transform: translateY(-62%);
    color: var(--color-text-muted);
    font-size: 18px;
    pointer-events: none;
  }
</style>
