<script lang="ts">
  import Markdown from "./Markdown.svelte";

  interface Props {
    prev?: string | undefined;
    next?: string | undefined;
    prevNote?: string | undefined;
    nextNote?: string | undefined;
    disabledPrev?: boolean;
    onprev?: () => void;
    onnext?: () => void;
  }

  let { prev, next, prevNote, nextNote, disabledPrev = false, onprev, onnext }: Props = $props();
</script>

<hr />
<nav class="navbar" aria-label="Navegación entre pestañas">
  <h3>🧭 Navegación</h3>
  <div class="columns">
    <div>
      {#if prev}
        <button type="button" class="btn block" disabled={disabledPrev} onclick={() => onprev?.()}>
          🔙 {prev}
        </button>
        {#if prevNote}<Markdown source={prevNote} />{/if}
      {/if}
    </div>
    <div>
      {#if next}
        <button type="button" class="btn block primary" onclick={() => onnext?.()}>🔜 {next}</button>
        {#if nextNote}<div class="note"><Markdown source={nextNote} /></div>{/if}
      {/if}
    </div>
  </div>
</nav>

<style>
  h3 {
    margin: 0 0 14px;
    color: var(--color-heading);
    font-size: 19px;
  }

  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }

  .note {
    margin-top: 10px;
  }
</style>
