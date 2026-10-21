<script lang="ts">
  import { tick } from "svelte";

  interface Props {
    tabs: readonly string[];
    active: number;
    onchange: (index: number) => void;
    /** Prefix of the tab ids (`{idPrefix}-tab-{i}`); the panel is `{idPrefix}-panel`. */
    idPrefix: string;
    /** Accessible name of the tab list. */
    label?: string;
  }

  let { tabs, active, onchange, idPrefix, label = "Pestañas de la lección" }: Props = $props();
  let list: HTMLDivElement | undefined = $state();

  /** WAI-ARIA tabs with automatic activation: arrows, Home and End move and select. */
  async function onkeydown(event: KeyboardEvent): Promise<void> {
    const last = tabs.length - 1;
    let next: number | null = null;
    if (event.key === "ArrowRight") {
      next = active === last ? 0 : active + 1;
    } else if (event.key === "ArrowLeft") {
      next = active === 0 ? last : active - 1;
    } else if (event.key === "Home") {
      next = 0;
    } else if (event.key === "End") {
      next = last;
    }
    if (next === null) {
      return;
    }
    event.preventDefault();
    onchange(next);
    await tick();
    list?.querySelector<HTMLButtonElement>(`[data-tab-index="${next}"]`)?.focus();
  }
</script>

<div class="tabs" role="tablist" tabindex="-1" aria-label={label} bind:this={list} {onkeydown}>
  {#each tabs as tab, index (tab)}
    <button
      type="button"
      role="tab"
      id="{idPrefix}-tab-{index}"
      class="btn"
      class:primary={index === active}
      aria-selected={index === active}
      aria-controls="{idPrefix}-panel"
      tabindex={index === active ? 0 : -1}
      title={tab}
      data-tab-index={index}
      onclick={() => onchange(index)}
    >
      {tab}
    </button>
  {/each}
</div>
<hr />

<style>
  .tabs {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 10px;
    margin: 20px 0 0;
  }

  .tabs .btn {
    min-width: 0;
    padding-left: 6px;
    padding-right: 6px;
    white-space: nowrap;
    overflow: hidden;
  }

  /* Narrow pages (1024 px windows): the label wraps under its emoji instead of being cut. */
  @container page (max-width: 900px) {
    .tabs {
      gap: 6px;
    }

    .tabs .btn {
      padding: 6px 2px;
      white-space: normal;
      hyphens: auto;
      font-size: 12px;
      line-height: 1.2;
    }
  }
</style>
