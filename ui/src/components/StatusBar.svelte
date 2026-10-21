<script lang="ts">
  import { cancelAll } from "../lib/backend";
  import { busy } from "../lib/busy.svelte";
  import { status } from "../lib/status.svelte";
</script>

<footer class="statusbar">
  <span class="text" role="status" aria-live="polite">{status.text}</span>
  {#if busy.active}
    <span class="activity">
      {#if busy.canCancel}
        <button type="button" class="cancel" title="Detiene la lectura o el cálculo en curso" onclick={cancelAll}>
          Cancelar
        </button>
      {/if}
      <span class="progress" role="progressbar" aria-label="Calculando…" aria-valuetext="En curso"><span class="bar"></span></span>
    </span>
  {/if}
</footer>

<style>
  .statusbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    height: 30px;
    padding: 0 12px;
    background: var(--color-statusbar-bg);
    color: var(--color-statusbar-text);
    border-top: 1px solid var(--color-divider);
    font-size: 13px;
  }

  .activity {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .cancel {
    height: 22px;
    padding: 0 10px;
    border: 1px solid var(--color-divider);
    border-radius: 4px;
    background: var(--color-bg);
    color: inherit;
    font: inherit;
    font-size: 12px;
    cursor: pointer;
  }

  .cancel:hover {
    border-color: var(--color-primary);
    color: var(--color-primary);
  }

  .cancel:focus-visible {
    outline: 2px solid var(--color-primary);
    outline-offset: 1px;
  }

  .progress {
    position: relative;
    width: 170px;
    height: 10px;
    overflow: hidden;
    border: 1px solid var(--color-input-border);
    border-radius: 4px;
  }

  .bar {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 40%;
    background: var(--color-primary);
    animation: slide 1.1s ease-in-out infinite;
  }

  /* No sliding bar with reduced motion (theme.css stops animations): a static, full bar. */
  @media (prefers-reduced-motion: reduce) {
    .bar {
      width: 100%;
      opacity: 0.55;
    }
  }

  @keyframes slide {
    from {
      left: -40%;
    }
    to {
      left: 100%;
    }
  }
</style>
