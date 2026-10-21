<script lang="ts">
  interface Props {
    label?: string;
  }

  let { label = "Cargando…" }: Props = $props();
</script>

<div class="spinner" role="status" aria-live="polite">
  <span class="ring" aria-hidden="true"></span>
  <span class="label">{label}</span>
</div>

<style>
  .spinner {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 12px 0;
    color: var(--color-text-muted);
  }

  .ring {
    width: 18px;
    height: 18px;
    border: 2px solid var(--color-divider);
    border-top-color: var(--color-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
