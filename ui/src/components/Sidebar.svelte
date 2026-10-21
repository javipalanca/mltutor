<script lang="ts">
  import { NAV_SECTIONS, type RouteId } from "../lib/routes";

  interface Props {
    current: RouteId;
    onnavigate: (id: RouteId) => void;
  }

  let { current, onnavigate }: Props = $props();
</script>

<nav class="sidebar" aria-label="Navegación principal">
  {#each NAV_SECTIONS as section, index (section.heading)}
    {#if index > 0}
      <hr />
    {/if}
    <h3>{section.heading}</h3>
    <ul>
      {#each section.routes as route (route.id)}
        <li>
          <button
            type="button"
            class:active={route.id === current}
            aria-current={route.id === current ? "page" : undefined}
            onclick={() => onnavigate(route.id)}
          >
            {route.label}
          </button>
        </li>
      {/each}
    </ul>
  {/each}
</nav>

<style>
  .sidebar {
    height: 100%;
    overflow-y: auto;
    padding: 20px 12px;
    background: var(--color-sidebar-bg);
    border-right: 1px solid var(--color-divider);
  }

  h3 {
    margin: 0 4px 14px;
    color: var(--color-heading);
    font-size: 17px;
    font-weight: 700;
  }

  ul {
    list-style: none;
    margin: 0 0 8px;
    padding: 0;
  }

  li + li {
    margin-top: 12px;
  }

  button {
    display: block;
    width: 100%;
    min-height: 42px;
    padding: 10px 14px;
    text-align: center;
    /* Wraps (instead of "Cargar CSV Person…") when the sidebar narrows in small windows. */
    white-space: normal;
    line-height: 1.3;
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-button);
    cursor: pointer;
    transition:
      background-color 120ms ease,
      border-color 120ms ease;
  }

  button:hover {
    background: var(--color-hover-bg);
    border-color: var(--color-hover-border);
  }

  button:active {
    background: var(--color-pressed-bg);
  }

  button:focus-visible {
    outline: 2px solid var(--color-focus);
    outline-offset: 2px;
  }

  button.active {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: var(--color-on-primary);
    font-weight: 600;
  }

  button.active:hover {
    background: var(--color-primary-hover);
    border-color: var(--color-primary-hover);
  }

  hr {
    margin: 28px 10px 32px;
    border: none;
    border-top: 1px solid var(--color-text);
  }
</style>
