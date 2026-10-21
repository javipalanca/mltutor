<script lang="ts">
  /** `st.checkbox`. */
  interface Props {
    label: string;
    checked: boolean;
    onchange: (checked: boolean) => void;
    help?: string;
    id: string;
  }

  let { label, checked, onchange, help, id }: Props = $props();
</script>

<label class="checkbox" for={id} title={help}>
  <input {id} type="checkbox" {checked} onchange={(event) => onchange(event.currentTarget.checked)} />
  <span>{label}</span>
</label>

<style>
  .checkbox {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin: 12px 0;
    cursor: pointer;
  }

  input {
    width: 18px;
    height: 18px;
    margin: 0;
    accent-color: var(--color-primary);
    cursor: pointer;
  }

  input:focus-visible {
    outline: 2px solid var(--color-focus);
    outline-offset: 2px;
  }
</style>
