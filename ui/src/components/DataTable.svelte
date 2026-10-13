<script lang="ts">
  import type { Cell } from "../lib/bindings/Cell";
  import { cellText, compareCells, toTsv } from "../lib/table";

  interface Props {
    columns: readonly string[];
    rows: readonly (readonly Cell[])[];
    /** Row labels (pandas index); hidden with `hideIndex`. */
    index?: readonly string[] | undefined;
    hideIndex?: boolean;
    /** Click a header to sort (an improvement over Qt). */
    sortable?: boolean;
    /** Accessible name. */
    label?: string;
  }

  let { columns, rows, index, hideIndex = false, sortable = false, label }: Props = $props();

  let sortColumn = $state<number | null>(null);
  let descending = $state(false);
  let selected = $state<Set<number>>(new Set());
  let anchor = $state<number | null>(null);
  let copied = $state(false);

  const order = $derived.by(() => {
    const positions = rows.map((_, i) => i);
    const column = sortColumn;
    if (column === null) {
      return positions;
    }
    return positions.sort((a, b) => {
      const result = compareCells(rows[a]?.[column], rows[b]?.[column]);
      return descending ? -result : result;
    });
  });
  const showIndex = $derived(!hideIndex && index !== undefined);

  function sortBy(column: number): void {
    if (!sortable) {
      return;
    }
    if (sortColumn === column) {
      if (descending) {
        sortColumn = null;
        descending = false;
      } else {
        descending = true;
      }
    } else {
      sortColumn = column;
      descending = false;
    }
  }

  function select(row: number, event: MouseEvent): void {
    const next = new Set(event.metaKey || event.ctrlKey ? selected : []);
    if (event.shiftKey && anchor !== null) {
      const from = order.indexOf(anchor);
      const to = order.indexOf(row);
      for (let i = Math.min(from, to); i <= Math.max(from, to); i += 1) {
        next.add(order[i] ?? row);
      }
    } else if (next.has(row) && (event.metaKey || event.ctrlKey)) {
      next.delete(row);
    } else {
      next.add(row);
      anchor = row;
    }
    selected = next;
  }

  async function onkeydown(event: KeyboardEvent): Promise<void> {
    const modifier = event.metaKey || event.ctrlKey;
    if (modifier && event.key.toLowerCase() === "a") {
      event.preventDefault();
      selected = new Set(rows.map((_, i) => i));
    } else if (modifier && event.key.toLowerCase() === "c" && selected.size > 0) {
      event.preventDefault();
      const picked = order.filter((row) => selected.has(row));
      const text = toTsv(
        columns,
        picked.map((row) => rows[row] ?? []),
        showIndex ? picked.map((row) => index?.[row] ?? "") : undefined,
      );
      try {
        await navigator.clipboard.writeText(text);
        copied = true;
        setTimeout(() => (copied = false), 1500);
      } catch {
        copied = false;
      }
    } else if (event.key === "Escape") {
      selected = new Set();
    }
  }
</script>

<div class="table-block">
  <!-- Focusable region so Ctrl/⌘+A and Ctrl/⌘+C work on the table. -->
  <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
  <div class="scroll" tabindex="0" role="region" aria-label={label ?? "Tabla"} {onkeydown}>
    <table>
      <thead>
        <tr>
          {#if showIndex}<th class="corner" scope="col"></th>{/if}
          {#each columns as column, c (c)}
            <th
              scope="col"
              class:sortable
              aria-sort={sortColumn === c ? (descending ? "descending" : "ascending") : undefined}
              onclick={() => sortBy(c)}
            >
              {column}{#if sortColumn === c}<span class="arrow">{descending ? " ▼" : " ▲"}</span>{/if}
            </th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each order as row (row)}
          <tr class:selected={selected.has(row)} onclick={(event) => select(row, event)}>
            {#if showIndex}<th scope="row" class="index">{index?.[row] ?? ""}</th>{/if}
            {#each rows[row] ?? [] as cell, c (c)}
              {@const text = cellText(cell)}
              <td class:number={cell.kind === "number"} class:missing={cell.kind === "missing"} title={cell.kind === "number" ? String(cell.v) : undefined}>
                {text}
              </td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <div class="caption">
    {rows.length.toLocaleString("en-US")} filas · {columns.length} columnas &nbsp;·&nbsp; Selecciona y copia con Ctrl+C / ⌘C
    {#if copied}<span class="copied"> · ✓ Copiado</span>{/if}
  </div>
</div>

<style>
  .table-block {
    margin: 10px 0 16px;
  }

  .scroll {
    overflow-x: auto;
    border: 1px solid var(--color-table-border);
    border-radius: 8px;
    background: var(--color-bg);
  }

  .scroll:focus-visible {
    outline: 2px solid var(--color-focus);
    outline-offset: 1px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-variant-numeric: tabular-nums;
  }

  th,
  td {
    padding: 6px 12px;
    min-width: 100px;
    max-width: 320px;
    height: 36px;
    border-bottom: 1px solid var(--color-table-row-border);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: left;
  }

  thead th {
    height: 44px;
    padding: 10px 12px;
    background: var(--color-table-head-bg);
    color: var(--color-table-head-text);
    font-weight: 700;
    border-right: 1px solid var(--color-table-border);
    border-bottom: 1px solid var(--color-table-border);
  }

  thead th.sortable {
    cursor: pointer;
  }

  thead th.sortable:hover {
    background: var(--color-hover-bg);
  }

  .arrow {
    font-size: 11px;
  }

  th.corner,
  th.index {
    min-width: 44px;
    width: 1%;
    background: var(--color-table-index-bg);
    color: var(--color-text-muted);
    font-weight: 400;
  }

  tbody tr:nth-child(even) td {
    background: var(--color-table-alt);
  }

  tbody tr.selected td,
  tbody tr.selected th {
    background: var(--color-table-selected);
    color: var(--color-table-selected-text);
  }

  td.number {
    text-align: right;
  }

  td.missing {
    color: var(--color-text-muted);
  }

  .caption {
    margin-top: 8px;
    color: var(--color-text-muted);
    font-size: 12px;
  }

  .copied {
    color: var(--color-primary);
  }
</style>
