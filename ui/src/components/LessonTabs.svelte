<script lang="ts">
  interface Props {
    tabs: readonly string[];
    active: number;
    onchange: (index: number) => void;
  }

  let { tabs, active, onchange }: Props = $props();

  function onkeydown(event: KeyboardEvent): void {
    const delta = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (delta !== 0) {
      event.preventDefault();
      onchange((active + delta + tabs.length) % tabs.length);
    }
  }
</script>

<div class="tabs" role="tablist" tabindex="-1" {onkeydown}>
  {#each tabs as tab, index (tab)}
    <button
      type="button"
      role="tab"
      class="btn"
      class:primary={index === active}
      aria-selected={index === active}
      tabindex={index === active ? 0 : -1}
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
    text-overflow: ellipsis;
  }

  .tabs .btn:focus-visible {
    padding-left: 5px;
    padding-right: 5px;
  }
</style>
