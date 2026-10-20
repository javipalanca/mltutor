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
  const uid = $props.id();
  let local = $state<boolean | null>(null);
  let header: HTMLButtonElement | undefined = $state();
  const isOpen = $derived(id ? session.expander(id, open) : (local ?? open));

  function toggle(): void {
    if (id) {
      session.setExpander(id, !isOpen);
    } else {
      local = !isOpen;
    }
  }

  /** Escape closes the expander from anywhere inside it and returns focus to its header. */
  function onkeydown(event: KeyboardEvent): void {
    if (event.key !== "Escape" || !isOpen || event.defaultPrevented) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    toggle();
    header?.focus();
  }
</script>

<!-- Escape bubbles up from any control inside the expander. -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<section class="expander" class:open={isOpen} {onkeydown}>
  <button
    type="button"
    class="btn block header"
    id="{uid}-header"
    aria-expanded={isOpen}
    aria-controls="{uid}-body"
    bind:this={header}
    onclick={toggle}
  >
    <span class="chevron" aria-hidden="true">{isOpen ? "▾" : "▸"}</span>
    <span>{label}</span>
  </button>
  {#if isOpen}
    <div class="body" id="{uid}-body">
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
