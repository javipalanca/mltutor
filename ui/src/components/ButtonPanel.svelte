<script lang="ts" generics="T extends string">
  interface Option {
    label: string;
    value: T;
    help?: string;
  }

  interface Props {
    options: readonly Option[];
    /** Active option (`primary`), or `null` for none. */
    value: T | null;
    onchange: (value: T) => void;
    /** Accessible name of the group. */
    label?: string;
  }

  let { options, value, onchange, label }: Props = $props();
</script>

<div class="panel" role="group" aria-label={label} style:--columns={options.length}>
  {#each options as option (option.value)}
    <button
      type="button"
      class="btn block"
      class:primary={option.value === value}
      aria-pressed={option.value === value}
      title={option.help}
      onclick={() => onchange(option.value)}
    >
      {option.label}
    </button>
  {/each}
</div>

<style>
  .panel {
    display: grid;
    grid-template-columns: repeat(var(--columns), minmax(0, 1fr));
    gap: 16px;
    margin: 12px 0;
  }
</style>
