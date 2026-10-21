<script lang="ts">
  /**
   * Renders a view tree built in Rust with its texts already resolved (C23 diagnostics, the
   * cards of 📊 Historial): the shape `tools/golden_render.py` records from Python. Text is
   * Markdown; nothing is rendered as raw HTML.
   */
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Metric from "../../../components/Metric.svelte";
  import Notice from "../../../components/Notice.svelte";
  import type { NnViewNode } from "../../bindings/NnViewNode";
  import NnViewTree from "./NnViewTree.svelte";

  interface Props {
    nodes: readonly NnViewNode[];
    /** Prefix of the expander ids (they remember whether they are open). */
    idPrefix: string;
  }

  let { nodes, idPrefix }: Props = $props();

  /** Stable id of an expander from its label. */
  function slug(label: string, index: number): string {
    const base = label
      .normalize("NFKD")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
    return `${idPrefix}-${base || index}`;
  }
</script>

{#each nodes as node, i (i)}
  {#if node.kind === "page" || node.kind === "column"}
    <NnViewTree nodes={node.children} idPrefix="{idPrefix}-{i}" />
  {:else if node.kind === "text"}
    <Markdown source={node.text} />
  {:else if node.kind === "heading"}
    {#if node.level <= 1}
      <h2 class="heading">{node.text}</h2>
    {:else}
      <h3 class="heading">{node.text}</h3>
    {/if}
  {:else if node.kind === "notice"}
    <Notice severity={node.severity} markdown={node.text} />
  {:else if node.kind === "metric"}
    <Metric label={node.label} value={node.value} {...node.delta ? { delta: node.delta } : {}} {...node.help ? { help: node.help } : {}} />
  {:else if node.kind === "row"}
    <div class="row" style:grid-template-columns={node.weights.map((w) => `minmax(0, ${w}fr)`).join(" ")} data-view-row>
      {#each node.children as column, c (c)}
        <div class="column">
          <NnViewTree nodes={column.kind === "column" ? column.children : [column]} idPrefix="{idPrefix}-{i}-{c}" />
        </div>
      {/each}
    </div>
  {:else if node.kind === "expander"}
    <Expander id={slug(node.label, i)} label={node.label} open={node.expanded}>
      <NnViewTree nodes={node.children} idPrefix="{idPrefix}-{i}" />
    </Expander>
  {/if}
{/each}

<style>
  .row {
    display: grid;
    gap: 16px;
    margin: 6px 0 10px;
  }

  .column {
    min-width: 0;
  }

  .column :global(.notice) {
    margin: 8px 0;
  }

  .heading {
    margin: 22px 0 10px;
    color: var(--color-heading);
  }

  h2.heading {
    font-size: 23px;
  }

  h3.heading {
    font-size: 19px;
  }
</style>
