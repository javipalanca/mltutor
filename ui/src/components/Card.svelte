<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    background: string;
    border?: string;
    titleColor?: string;
    title?: string;
    /** Fixed height (the home cards are 200 px tall). */
    height?: number;
    children: Snippet;
  }

  let { background, border, titleColor, title, height, children }: Props = $props();
</script>

<div
  class="card"
  class:bordered={border !== undefined}
  style:--card-bg={background}
  style:--card-border={border}
  style:--card-title={titleColor}
  style:min-height={height ? `${height}px` : undefined}
>
  {#if title}<h3>{title}</h3>{/if}
  {@render children()}
</div>

<style>
  .card {
    padding: 18px 20px;
    background: var(--card-bg);
    border-radius: var(--radius-box);
    color: #243247;
  }

  .card.bordered {
    border-left: 5px solid var(--card-border);
  }

  h3 {
    margin: 0 0 10px;
    color: var(--card-title, var(--color-heading));
    font-size: 17px;
  }

  @media (prefers-color-scheme: dark) {
    :global(:root:not([data-theme="light"])) .card {
      filter: brightness(0.92);
    }
  }
</style>
