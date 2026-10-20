<script lang="ts">
  /**
   * C24 (D80), architecture mode: the network of 🏗️ Arquitectura as a native SVG with the
   * colours, labels and legend of Python's canvas, a tooltip per layer with its neurons and
   * parameters, the total under the drawing and big layers summarised ("⋮").
   */
  import { untrack } from "svelte";
  import type { LayerRow } from "../../bindings/LayerRow";
  import type { NnActivation } from "../../bindings/NnActivation";
  import { nnText } from "../../content";
  import { thousands } from "../../lessons/redes/state.svelte";
  import { COLORS, diagramHeight, layoutNetwork, type LayerLayout } from "./networkLayout";

  interface Props {
    rows: readonly LayerRow[];
    activation: NnActivation;
    outputActivation: NnActivation;
    /** "clasificación" | "regresión". */
    task: string;
    totalParams: number;
  }

  let { rows, activation, outputActivation, task, totalParams }: Props = $props();
  const t = (key: string): string => nnText("arquitectura", key);

  let width = $state(520);
  let canvas: HTMLDivElement | undefined = $state();
  const sizes = $derived(rows.map((r) => r.neurons));
  const height = $derived(diagramHeight(width, sizes));

  // The width follows the column outside the ResizeObserver callback (a synchronous
  // update would resize the observed box again: "ResizeObserver loop" in WebKit).
  $effect(() => {
    const target = canvas;
    if (!target) {
      return;
    }
    let pending: ReturnType<typeof setTimeout> | null = null;
    const apply = (): void => {
      pending = null;
      const next = Math.round(target.clientWidth);
      if (next > 0 && Math.abs(next - width) >= 1) {
        width = next;
      }
    };
    untrack(apply);
    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(() => {
      if (pending === null) {
        pending = setTimeout(apply, 16);
      }
    });
    observer.observe(target);
    return () => {
      observer.disconnect();
      if (pending !== null) {
        clearTimeout(pending);
      }
    };
  });
  const layout = $derived(layoutNetwork(sizes, width, height));
  let hover = $state<number | null>(null);

  const fill = (layer: LayerLayout): string => COLORS[layer.kind];

  function label(i: number): string {
    if (i === 0) {
      return t("diagrama_etiqueta_entrada");
    }
    if (i === rows.length - 1) {
      return t("diagrama_etiqueta_salida");
    }
    return t("diagrama_etiqueta_oculta").replace("{i}", String(i));
  }

  function tooltip(i: number): string {
    const row = rows[i];
    if (!row) {
      return "";
    }
    const hidden = layout.layers[i]?.hidden ?? 0;
    const more = hidden > 0 ? ` ${t("diagrama_resumidas").replace("{n}", String(hidden))} (no se dibujan).` : "";
    if (i === 0) {
      return `${row.name}: ${row.neurons} neuronas (características) · 0 parámetros.${more}`;
    }
    return (
      t("diagrama_tooltip_capa")
        .replace("{capa}", row.name)
        .replaceAll("{n}", String(row.neurons))
        .replace("{parametros:,}", thousands(row.params))
        .replace("{entradas}", String(row.inputs)) + more
    );
  }

  const hitWidth = $derived(rows.length > 1 ? (width - 80) / (rows.length - 1) : width);
  const tip = $derived.by(() => {
    if (hover === null) {
      return null;
    }
    const layer = layout.layers[hover];
    if (!layer) {
      return null;
    }
    const left = Math.min(Math.max(layer.x, 110), width - 110);
    return { text: tooltip(hover), left };
  });
</script>

<!-- Escape dismisses the hover tooltip (WCAG 1.4.13). -->
<svelte:window onkeydown={(event) => { if (event.key === "Escape") hover = null; }} />

<div class="network" data-network-diagram data-layers={sizes.join(",")}>
  <div class="box">
    <strong>{t("diagrama_caja_titulo")}</strong>
    <span>{t("diagrama_caja_texto").replace("{tarea}", task)}</span>
  </div>
  <div class="canvas" bind:this={canvas}>
    <svg
      viewBox="0 0 {layout.width} {layout.height}"
      width={layout.width}
      height={layout.height}
      role="img"
      aria-label="{t('diagrama_caja_titulo')}: {sizes.join(' → ')}"
    >
      {#each layout.links as link, i (i)}
        <path
          d={link.d}
          stroke={COLORS.connection}
          stroke-width="1"
          fill="none"
          opacity={hover === null || hover === i || hover === i + 1 ? link.opacity : link.opacity * 0.35}
        />
      {/each}
      {#each layout.layers as layer, i (i)}
        <g class="layer" class:dim={hover !== null && hover !== i} data-layer={i}>
          {#each layer.nodes as node (node.index)}
            <circle cx={layer.x} cy={node.y} r={layout.radius} fill={fill(layer)} stroke={COLORS.border} stroke-width="2" />
          {/each}
          {#if layer.gapY !== null}
            <text class="gap" x={layer.x} y={layer.gapY} font-size={layout.radius * 1.8} text-anchor="middle" dominant-baseline="central">⋮</text>
            <text class="more" x={layer.x} y={layer.gapY} dx={layout.radius + 4} font-size="10.5" dominant-baseline="central">+{layer.hidden}</text>
          {/if}
          <text class="label" x={layer.x} y={layout.labelY} font-size={layout.fontSize} text-anchor="middle">{label(i)}</text>
          <text class="label" x={layer.x} y={layout.labelY + 15} font-size={layout.fontSize} text-anchor="middle">({layer.neurons})</text>
          <rect
            class="hit"
            x={layer.x - hitWidth / 2}
            y="0"
            width={hitWidth}
            height={layout.height}
            role="presentation"
            onpointerenter={() => (hover = i)}
            onpointerleave={() => (hover = null)}
          />
        </g>
      {/each}
    </svg>
    {#if tip}
      <div class="tooltip" style:left="{tip.left}px" role="tooltip" data-network-tooltip>{tip.text}</div>
    {/if}
  </div>
  <div class="legend">
    <span><i style:background={COLORS.input}></i>{t("diagrama_leyenda_entrada")}</span>
    <span><i style:background={COLORS.hidden}></i>{t("diagrama_leyenda_ocultas").replace("{activacion}", activation.toUpperCase())}</span>
    <span><i style:background={COLORS.output}></i>{t("diagrama_leyenda_salida").replace("{activacion}", outputActivation.toUpperCase())}</span>
  </div>
  <p class="total" data-network-total>{t("metrica_parametros")}: <strong>{thousands(totalParams)}</strong></p>
</div>

<style>
  .network {
    margin: 4px 0 12px;
  }

  .box {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-bottom: 10px;
    padding: 10px 14px;
    background: var(--color-info-bg);
    border-left: 4px solid var(--color-info-border, #2196f3);
    border-radius: 4px;
    font-size: 14px;
  }

  .canvas {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, var(--nn-bg-from, #f8f9fa), var(--nn-bg-to, #e9ecef));
    border: 1px solid var(--color-border);
    border-radius: 8px;
  }

  svg {
    display: block;
    width: 100%;
    height: auto;
  }

  .label {
    fill: var(--nn-text, #2c3e50);
    font-weight: 700;
  }

  .gap {
    fill: var(--nn-text, #2c3e50);
    font-weight: 700;
  }

  .more {
    fill: var(--color-text-muted);
  }

  .layer circle {
    transition: opacity 0.12s;
  }

  .layer.dim circle {
    opacity: 0.45;
  }

  .hit {
    fill: transparent;
    cursor: help;
  }

  .tooltip {
    position: absolute;
    top: 8px;
    max-width: 240px;
    padding: 6px 10px;
    transform: translateX(-50%);
    background: var(--color-tooltip-bg);
    color: var(--color-text);
    border: 1px solid var(--color-border);
    border-radius: 6px;
    box-shadow: 0 4px 14px rgb(0 0 0 / 14%);
    font-size: 12.5px;
    line-height: 1.4;
    pointer-events: none;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px 12px;
    margin-top: 10px;
    font-size: 12.5px;
  }

  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: var(--color-chip-bg);
    border: 1px solid var(--color-border);
    border-radius: 15px;
  }

  .legend i {
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }

  .total {
    margin: 8px 0 0;
    text-align: center;
    color: var(--color-text-muted);
    font-size: 13px;
  }

  :global(:root[data-theme="dark"]) .canvas {
    --nn-bg-from: #1b2534;
    --nn-bg-to: #131c29;
    --nn-text: #dbe4f0;
  }
</style>
