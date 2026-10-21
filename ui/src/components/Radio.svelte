<script lang="ts" generics="T extends string">
  interface Option {
    label: string;
    value: T;
  }

  interface Props {
    label: string;
    options: readonly Option[];
    value: T;
    onchange: (value: T) => void;
    horizontal?: boolean;
    name: string;
  }

  let { label, options, value, onchange, horizontal = false, name }: Props = $props();
</script>

<fieldset class="radio" class:horizontal>
  <legend>{label}</legend>
  <div class="options">
    {#each options as option (option.value)}
      <label class="option">
        <input type="radio" {name} value={option.value} checked={option.value === value} onchange={() => onchange(option.value)} />
        <span>{option.label}</span>
      </label>
    {/each}
  </div>
</fieldset>

<style>
  .radio {
    margin: 12px 0;
    padding: 0;
    border: none;
  }

  legend {
    padding: 0;
    margin-bottom: 10px;
  }

  .options {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .horizontal .options {
    flex-direction: row;
    flex-wrap: wrap;
    gap: 22px;
  }

  .option {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
  }

  input {
    width: 16px;
    height: 16px;
    margin: 0;
    accent-color: var(--color-primary);
  }
</style>
