<script lang="ts">
  import { footerLines } from "../lib/content";

  const lines = footerLines();
</script>

<footer class="footer">
  {#each lines as line, index (line)}
    {#if index > 0}<span class="sep" aria-hidden="true"> • </span>{/if}<span>{line}</span>
  {/each}
</footer>

<style>
  .footer {
    margin-top: 64px;
    padding: 18px 0 6px;
    border-top: 1px solid var(--color-divider);
    color: var(--color-footer);
    font-size: 11px;
    text-align: center;
  }

  .sep {
    color: var(--color-border);
  }
</style>
