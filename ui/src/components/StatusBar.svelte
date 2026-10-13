<script lang="ts">
  import { busy } from "../lib/busy.svelte";
  import { status } from "../lib/status.svelte";
</script>

<footer class="statusbar" aria-live="polite">
  <span class="text">{status.text}</span>
  {#if busy.active}
    <span class="progress" role="progressbar" aria-label="Actualizando"><span class="bar"></span></span>
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

  .progress {
    position: relative;
    width: 170px;
    height: 10px;
    overflow: hidden;
    border: 1px solid #dce2ea;
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

  @keyframes slide {
    from {
      left: -40%;
    }
    to {
      left: 100%;
    }
  }
</style>
