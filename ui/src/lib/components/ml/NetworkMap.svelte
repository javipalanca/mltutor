<script lang="ts">
  /**
   * C24 (D80), map mode: "🗺️ Mapa de Neuronas Muertas (💀) y Constantes (😴)" of 📉
   * Activaciones, with every drawing option of Python's canvas (deterministic jitter, % of
   * connections with the same hash, adaptive radius, sub-columns for big layers) and its
   * tooltip (bias, w̄, σ, |w|max and state). Drawn on a HiDPI `<canvas>` because a 5 × 256
   * network has up to 262 144 connections, which an SVG cannot keep fluid.
   */
  import { untrack } from "svelte";
  import type { LayerActivationsView } from "../../bindings/LayerActivationsView";
  import type { NetworkMapView } from "../../bindings/NetworkMapView";
  import { nnList, nnText } from "../../content";
  import { theme } from "../../theme.svelte";
  import { formatMapValue, hitNeuron, layoutMap, MAP_COLORS, mapHash, type MapLayout, type MapOptions } from "./networkLayout";

  interface Props {
    map: NetworkMapView;
    /** Hidden layers, with their dead and constant neurons. */
    hidden: readonly LayerActivationsView[];
    options: MapOptions;
  }

  let { map, hidden, options }: Props = $props();
  const t = (key: string): string => nnText("activaciones", key);
  const states = nnList("activaciones", "estados");

  let wrapper: HTMLDivElement | undefined = $state();
  let canvas: HTMLCanvasElement | undefined = $state();
  let width = $state(900);
  let tip = $state<{ x: number; y: number; title: string; lines: string[] } | null>(null);

  // Width of the box, outside the ResizeObserver callback (WebKit "ResizeObserver loop").
  $effect(() => {
    const target = wrapper;
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
        pending = setTimeout(apply, 60);
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

  const arch = $derived(map.layers);
  const layout = $derived<MapLayout>(layoutMap(arch, width, options));
  const dead = $derived(hidden.map((l) => new Set(l.dead)));
  const constant = $derived(hidden.map((l) => new Set(l.constant)));
  const counts = $derived({
    dead: hidden.reduce((s, l) => s + l.dead.length, 0),
    constant: hidden.reduce((s, l) => s + l.constant.length, 0),
  });

  function layerLabel(l: number): string {
    if (l === 0) {
      return nnText("arquitectura", "detalles_entrada");
    }
    if (l === arch.length - 1) {
      return nnText("arquitectura", "detalles_salida");
    }
    return nnText("arquitectura", "detalles_oculta").replace("{i}", String(l));
  }

  /** Python's canvas: connections at alpha 0.15, then the neurons, emojis and the labels on top. */
  function draw(target: HTMLCanvasElement, lay: MapLayout): void {
    const ctx = target.getContext("2d");
    if (!ctx) {
      return;
    }
    const dpr = Math.min(3, Math.max(1, window.devicePixelRatio || 1));
    target.width = Math.round(lay.width * dpr);
    target.height = Math.round(lay.height * dpr);
    target.style.width = `${lay.width}px`;
    target.style.height = `${lay.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, lay.width, lay.height);
    const style = getComputedStyle(target);
    const textColor = style.getPropertyValue("--map-text").trim() || "#333";
    const lineColor = style.getPropertyValue("--map-line").trim() || MAP_COLORS.connection;
    const pct = options.connectionPct;
    ctx.globalAlpha = 0.15;
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1;
    for (let l = 0; l < lay.nodes.length - 1; l++) {
      const left = lay.nodes[l] ?? [];
      const right = lay.nodes[l + 1] ?? [];
      // One path per source neuron: overlapping lines of different sources still add up
      // (Python stroked every line on its own) while the canvas does ~1 000 strokes, not 262 144.
      for (let i = 0; i < left.length; i++) {
        const a = left[i];
        if (!a) {
          continue;
        }
        ctx.beginPath();
        for (let j = 0; j < right.length; j++) {
          if (pct < 99.9 && mapHash(l, i, j) > pct / 100) {
            continue;
          }
          const b = right[j];
          if (b) {
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
          }
        }
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    const last = lay.nodes.length - 1;
    for (let l = 0; l < lay.nodes.length; l++) {
      const layer = lay.nodes[l] ?? [];
      const fill = l === 0 ? MAP_COLORS.input : l === last ? MAP_COLORS.output : MAP_COLORS.hidden;
      for (let i = 0; i < layer.length; i++) {
        const c = layer[i];
        if (!c) {
          continue;
        }
        let alpha = 1;
        let emoji = "";
        if (l > 0 && l < last) {
          if (dead[l - 1]?.has(i)) {
            alpha = 0.25;
            emoji = "💀";
          } else if (constant[l - 1]?.has(i)) {
            alpha = 0.45;
            emoji = "😴";
          }
        }
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        ctx.fillStyle = fill;
        ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = MAP_COLORS.border;
        ctx.lineWidth = 1.3;
        ctx.stroke();
        ctx.globalAlpha = 1;
        if (emoji) {
          ctx.font = "14px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(emoji, c.x, c.y + 1);
        }
      }
      ctx.font = "600 13px 'Inter Variable', Inter, sans-serif";
      ctx.fillStyle = textColor;
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      const x = Math.min(Math.max(lay.marginX + l * lay.layerGap, 50), lay.width - 50);
      ctx.fillText(t("mapa_etiqueta").replace("{capa}", layerLabel(l)).replace("{n}", String(arch[l] ?? 0)), x, 20);
    }
  }

  $effect(() => {
    const target = canvas;
    const lay = layout;
    // Read the reactive inputs of `draw` so that the effect follows them (the colours of the
    // light/dark mode come from the computed style, so the system theme is followed as well).
    void dead;
    void constant;
    void options.connectionPct;
    void theme.chart;
    if (target && lay.width > 0) {
      untrack(() => draw(target, lay));
    }
  });

  function stateName(l: number, i: number): string {
    if (l === arch.length - 1) {
      return states[3] ?? "salida";
    }
    if (dead[l - 1]?.has(i)) {
      return states[1] ?? "muerta";
    }
    if (constant[l - 1]?.has(i)) {
      return states[2] ?? "constante";
    }
    return states[0] ?? "activa";
  }

  /** `tooltip_neurona` / `tooltip_entrada` of the content: the first line in bold. */
  function tooltipFor(l: number, i: number): { title: string; lines: string[] } {
    const template = l === 0 ? t("tooltip_entrada") : t("tooltip_neurona");
    const info = l > 0 ? map.neurons[l - 1] : undefined;
    const value = (v: number[] | undefined): string => formatMapValue(v?.[i] ?? Number.NaN);
    const text = template
      .replace("{capa}", l === 0 ? "Entrada" : layerLabel(l))
      .replace("{i}", String(i))
      .replace("{bias}", value(info?.bias))
      .replace("{media_w}", value(info?.mean_w))
      .replace("{std_w}", value(info?.std_w))
      .replace("{max_w}", value(info?.max_abs_w))
      .replace("{estado}", l === 0 ? "" : stateName(l, i));
    const [first = "", ...rest] = text.split("\n");
    return { title: first.replaceAll("**", ""), lines: rest };
  }

  function onmove(event: PointerEvent): void {
    const target = canvas;
    if (!target) {
      return;
    }
    const rect = target.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const hit = hitNeuron(layout, x, y);
    if (!hit) {
      tip = null;
      return;
    }
    const content = tooltipFor(hit.layer, hit.index);
    tip = { x: Math.min(x + 14, width - 230), y: Math.min(y + 14, layout.height - 110), ...content };
  }

  const legend = $derived(t("mapa_leyenda").split(/\s{2,}/));
</script>

<div class="map" data-network-map data-dead={counts.dead} data-constant={counts.constant} data-layers={arch.join(",")}>
  <div class="box" bind:this={wrapper}>
    <!-- No `width`/`height` attributes: `draw` owns the bitmap. Any assignment to them, even
         of the same value, clears the canvas, and Svelte may apply them after `draw` ran. -->
    <canvas
      bind:this={canvas}
      aria-label="{t('mapa_titulo').replace(/^#+\s*/, '')}: {arch.join(' → ')}, {counts.dead} muertas, {counts.constant} constantes"
      onpointermove={onmove}
      onpointerleave={() => (tip = null)}
    ></canvas>
    {#if tip}
      <div class="tooltip" style:left="{tip.x}px" style:top="{tip.y}px" role="tooltip" data-map-tooltip>
        <strong>{tip.title}</strong>
        {#each tip.lines as line, i (i)}
          <br />{line}
        {/each}
      </div>
    {/if}
  </div>
  <p class="legend">
    {#each legend as part, i (i)}
      {#if i === 0 && part.includes(":")}
        <span>{part.slice(0, part.indexOf(":") + 1)} <span class="active">{part.slice(part.indexOf(":") + 1).trim()}</span></span>
      {:else}
        <span class:inactive={i === 1}>{part}</span>
      {/if}
    {/each}
  </p>
</div>

<style>
  .map {
    margin: 6px 0 14px;
  }

  .box {
    position: relative;
    overflow: hidden;
    padding: 0;
    background: var(--map-bg, #fafafa);
    border: 1px solid var(--color-border);
    border-radius: 8px;
  }

  canvas {
    display: block;
    cursor: crosshair;
    --map-text: #333333;
    --map-line: #777777;
  }

  .tooltip {
    position: absolute;
    max-width: 220px;
    padding: 6px 8px;
    background: var(--color-tooltip-bg, #fff);
    color: var(--color-text);
    border: 1px solid var(--color-border);
    border-radius: 6px;
    box-shadow: 0 2px 6px rgb(0 0 0 / 15%);
    font-size: 12px;
    line-height: 1.35;
    pointer-events: none;
    white-space: nowrap;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 14px;
    margin: 6px 2px 0;
    font-size: 12.5px;
    color: var(--color-text);
  }

  .legend .active {
    color: #1976d2;
  }

  .legend .inactive {
    color: var(--color-text);
    opacity: 0.45;
  }

  @media (prefers-color-scheme: dark) {
    .box {
      --map-bg: #151d29;
    }

    canvas {
      --map-text: #dbe4f0;
      --map-line: #a9b6c8;
    }
  }
</style>
