<script lang="ts">
  import type { Snippet } from "svelte";
  import { session } from "../lib/session.svelte";

  interface Props {
    label: string;
    /** Initial state (`st.expander(expanded=…)`). */
    open?: boolean;
    /** Remembers the state for the whole session (`view_state` in Qt). */
    id?: string;
    children: Snippet;
  }

  let { label, open = false, id, children }: Props = $props();
  let local = $state<boolean | null>(null);
  const isOpen = $derived(id ? session.expander(id, open) : (local ?? open));

  function toggle(): void {
    if (id) {
      session.setExpander(id, !isOpen);
    } else {
      local = !isOpen;
    }
  }
</script>

<section class="expander" class:open={isOpen}>
  <button type="button" class="btn block header" aria-expanded={isOpen} onclick={toggle}>
    <span class="chevron" aria-hidden="true">{isOpen ? "▾" : "▸"}</span>
    <span>{label}</span>
  </button>
  {#if isOpen}
    <div class="body">
      {@render children()}
    </div>
  {/if}
</section>

<style>
  .expander {
    margin: 12px 0;
  }

  .header {
    position: relative;
    font-weight: 500;
  }

  .chevron {
    position: absolute;
    left: 16px;
    color: var(--color-text-muted);
  }

  .body {
    padding: 14px 10px 4px;
  }
</style>
