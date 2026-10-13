<script lang="ts">
  interface Props {
    label: string;
    help?: string;
    /** Name of the chosen file, or `null`. */
    fileName: string | null;
    buttonLabel: string;
    clearLabel: string;
    onpick: () => void;
    onclear: () => void;
    disabled?: boolean;
  }

  let { label, help, fileName, buttonLabel, clearLabel, onpick, onclear, disabled = false }: Props = $props();
</script>

<div class="field">
  <div class="label" title={help}>{label}</div>
  <button type="button" class="btn block" {disabled} title={help} onclick={onpick}>
    {fileName ? `📄 ${fileName}` : buttonLabel}
  </button>
  {#if fileName}
    <button type="button" class="btn block clear" {disabled} onclick={onclear}>{clearLabel}</button>
  {/if}
</div>

<style>
  .field {
    margin: 12px 0;
  }

  .label {
    margin-bottom: 8px;
  }

  .clear {
    margin-top: 10px;
  }
</style>
