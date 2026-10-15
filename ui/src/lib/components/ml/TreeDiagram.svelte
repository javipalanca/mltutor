<script lang="ts">
  /**
   * C13 (D41/D42): the decision tree as an SVG with the boxes, texts, colours and layout of
   * `plot_tree`, a fixed legible font, zoom (wheel and buttons), panning (drag), a tooltip
   * per node and PNG/SVG downloads. The "Ancho/Alto de figura" sliders (inches of matplotlib)
   * set the size of the viewport: `ancho × 96 px × 0.66` (Python shows the figure in the
   * central column of `[1, 4, 1]`), capped by the container, with the same aspect ratio.
   */
  import { onMount, untrack } from "svelte";
  import { logToBackend } from "../../backend";
  import type { TreeDiagram } from "../../bindings/TreeDiagram";
  import type { TreeDiagramNode } from "../../bindings/TreeDiagramNode";
  import { dataUrlToBytes, saveBytes, saveText } from "../../download";
  import { errorMessage } from "../../backend";
  import { status } from "../../status.svelte";
  import { diagramSvg, drawDiagram, fitTransform, initialTransform, layoutTree, lineY, pathTo, pngScale, textMeasure, type BoxLayout } from "./treeLayout";

  interface Props {
    diagram: TreeDiagram;
    /** "Ancho de figura:" (8–20 in) and "Alto de figura:" (6–15 in). */
    widthIn?: number;
    heightIn?: number;
    /** Leaf whose path is highlighted (🔮 Predicciones). */
    highlightLeaf?: number | null;
    pngLabel: string;
    svgLabel: string;
    help?: string;
  }

  let { diagram, widthIn = 14, heightIn = 10, highlightLeaf = null, pngLabel, svgLabel, help }: Props = $props();
  /** Start of the component (layout + DOM of every box), for the log measure. */
  const created = performance.now();

  const layout = $derived.by(() => {
    const started = performance.now();
    const result = layoutTree(diagram, textMeasure());
    const ms = performance.now() - started;
    if (diagram.nodes.length > 100) {
      void logToBackend("debug", `diagrama del árbol: ${diagram.nodes.length} nodos, disposición en ${ms.toFixed(1)} ms`).catch(() => undefined);
    }
    return result;
  });
  const highlighted = $derived(pathTo(diagram, highlightLeaf));

  let container: HTMLDivElement | undefined = $state();
  let svgElement: SVGSVGElement | undefined = $state();
  let available = $state(900);
  const viewWidth = $derived(Math.max(320, Math.min(available, widthIn * 96 * 0.66)));
  const viewHeight = $derived(Math.max(260, Math.round((viewWidth * heightIn) / widthIn)));

  let view = $state({ k: 1, x: 0, y: 0 });
  let drag: { x: number; y: number; vx: number; vy: number; id: number } | null = null;
  let dragging = $state(false);
  let hover = $state<{ node: TreeDiagramNode; x: number; y: number } | null>(null);
  let frame = 0;

  function fit(): void {
    view = fitTransform(layout, viewWidth, viewHeight);
  }

  // First view again when the tree or the viewport changes.
  $effect(() => {
    const current = layout;
    const width = viewWidth;
    const height = viewHeight;
    untrack(() => {
      view = initialTransform(current, width, height);
    });
  });

  onMount(() => {
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver((entries) => {
            const width = entries[0]?.contentRect.width;
            if (width && Math.abs(width - available) > 1) {
              available = width;
            }
          });
    if (container) {
      available = container.clientWidth || available;
      observer?.observe(container);
    }
    // DOM of every box created; reading the size forces style and layout synchronously
    // (rAF does not run in hidden windows, so the paint itself is not measured).
    void svgElement?.getBoundingClientRect();
    const ms = performance.now() - created;
    void logToBackend("info", `diagrama del árbol: ${diagram.nodes.length} nodos creados y maquetados en ${ms.toFixed(1)} ms`).catch(() => undefined);
    return () => observer?.disconnect();
  });

  function zoomAt(factor: number, px: number, py: number): void {
    const k = Math.min(6, Math.max(0.05, view.k * factor));
    const f = k / view.k;
    view = { k, x: px - (px - view.x) * f, y: py - (py - view.y) * f };
  }

  function wheel(event: WheelEvent): void {
    event.preventDefault();
    const rect = (event.currentTarget as SVGSVGElement).getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const factor = Math.exp(-event.deltaY * (event.deltaMode === 1 ? 0.05 : 0.0018));
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => zoomAt(factor, px, py));
  }

  function down(event: PointerEvent): void {
    if (event.button !== 0) {
      return;
    }
    drag = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y, id: event.pointerId };
    dragging = true;
    (event.currentTarget as Element).setPointerCapture(event.pointerId);
  }

  function move(event: PointerEvent): void {
    if (drag && drag.id === event.pointerId) {
      const d = drag;
      view = { ...view, x: d.vx + event.clientX - d.x, y: d.vy + event.clientY - d.y };
      hover = null;
    }
  }

  function up(event: PointerEvent): void {
    if (drag?.id === event.pointerId) {
      drag = null;
      dragging = false;
    }
  }

  function key(event: KeyboardEvent): void {
    const step = 40;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [step, 0],
      ArrowRight: [-step, 0],
      ArrowUp: [0, step],
      ArrowDown: [0, -step],
    };
    const delta = moves[event.key];
    if (delta) {
      event.preventDefault();
      view = { ...view, x: view.x + delta[0], y: view.y + delta[1] };
    } else if (event.key === "+" || event.key === "=") {
      zoomAt(1.25, viewWidth / 2, viewHeight / 2);
    } else if (event.key === "-") {
      zoomAt(0.8, viewWidth / 2, viewHeight / 2);
    } else if (event.key === "0") {
      fit();
    }
  }

  function enter(box: BoxLayout, event: PointerEvent): void {
    if (drag || !container) {
      return;
    }
    const rect = container.getBoundingClientRect();
    hover = { node: box.node, x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function tooltipLines(node: TreeDiagramNode): string[] {
    const lines = [`Nodo ${node.id}${node.leaf ? " (hoja)" : ""}`, ...node.lines];
    const root = diagram.nodes[0]?.n_samples ?? node.n_samples;
    lines.push(`samples = ${node.n_samples} de ${root}`);
    if (node.value_counts) {
      lines.push(`value = [${node.value_counts.join(", ")}]`);
    }
    return lines;
  }

  async function savePng(): Promise<void> {
    try {
      const scale = pngScale(layout);
      const canvas = document.createElement("canvas");
      canvas.width = Math.ceil(layout.width * scale);
      canvas.height = Math.ceil(layout.height * scale);
      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("No se pudo crear la imagen.");
      }
      drawDiagram(context, layout, scale);
      const saved = await saveBytes("arbol_decision.png", dataUrlToBytes(canvas.toDataURL("image/png")), "png");
      if (saved) {
        status.flash(`Guardado: ${saved}`);
      }
    } catch (error) {
      status.flash(errorMessage(error instanceof Error ? error.message : error), 8000);
    }
  }

  async function saveSvg(): Promise<void> {
    try {
      const saved = await saveText("arbol_decision.svg", diagramSvg(layout), "svg");
      if (saved) {
        status.flash(`Guardado: ${saved}`);
      }
    } catch (error) {
      status.flash(errorMessage(error), 8000);
    }
  }
</script>

<div class="tree-diagram" data-tree-diagram data-nodes={diagram.nodes.length} bind:this={container}>
  <div class="toolbar">
    <button type="button" class="btn icon" title="Acercar (+)" aria-label="Acercar" onclick={() => zoomAt(1.25, viewWidth / 2, viewHeight / 2)}>＋</button>
    <button type="button" class="btn icon" title="Alejar (−)" aria-label="Alejar" onclick={() => zoomAt(0.8, viewWidth / 2, viewHeight / 2)}>－</button>
    <button type="button" class="btn" title="Ver el árbol entero (0)" onclick={fit}>Ajustar</button>
    <span class="zoom">{Math.round(view.k * 100)} %</span>
  </div>
  <!-- Zoom, panning and arrow keys: the diagram is a focusable interactive surface. -->
  <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
  <svg
    bind:this={svgElement}
    class:dragging
    width={viewWidth}
    height={viewHeight}
    role="application"
    aria-roledescription="diagrama interactivo"
    aria-label="Diagrama del árbol de decisión ({diagram.nodes.length} nodos)"
    tabindex="0"
    onwheel={wheel}
    onpointerdown={down}
    onpointermove={move}
    onpointerup={up}
    onpointercancel={up}
    onkeydown={key}
    onpointerleave={() => (hover = null)}
  >
    <defs>
      <marker id="tree-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" class="arrow-head" />
      </marker>
    </defs>
    <g transform="translate({view.x} {view.y}) scale({view.k})">
      {#each layout.edges as e (e.to)}
        <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} class="edge" class:on={highlighted.has(e.to)} marker-end="url(#tree-arrow)" />
        {#if e.label}
          <text
            class="edge-label"
            x={(e.x1 + e.x2) / 2 + (e.label === "True" ? -6 : 6)}
            y={(e.y1 + e.y2) / 2}
            text-anchor={e.label === "True" ? "end" : "start"}
            dominant-baseline="middle">{e.label}</text
          >
        {/if}
      {/each}
      {#each layout.boxes as b (b.node.id)}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <g class="node" class:on={highlighted.has(b.node.id)} data-node={b.node.id} onpointerenter={(event) => enter(b, event)} onpointerleave={() => (hover = null)}>
          <rect x={b.x} y={b.y} width={b.w} height={b.h} rx="6" fill={b.node.fill} />
          {#each b.node.lines as text, i (i)}
            <text x={b.cx} y={lineY(b, i)} text-anchor="middle" dominant-baseline="central">{text}</text>
          {/each}
        </g>
      {/each}
    </g>
  </svg>
  {#if hover}
    <div class="tooltip" role="tooltip" style:left="{Math.min(hover.x + 14, viewWidth - 220)}px" style:top="{hover.y + 44}px">
      {#each tooltipLines(hover.node) as text, i (i)}
        <div class:first={i === 0}>{text}</div>
      {/each}
    </div>
  {/if}
  {#if help}<p class="help">{help}</p>{/if}
  <div class="downloads">
    <button type="button" class="btn" onclick={() => void savePng()} data-tree-png>{pngLabel}</button>
    <button type="button" class="btn" onclick={() => void saveSvg()} data-tree-svg>{svgLabel}</button>
  </div>
</div>

<style>
  .tree-diagram {
    position: relative;
    margin: 12px 0;
  }

  .toolbar {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
  }

  .btn.icon {
    min-width: 34px;
    padding-inline: 8px;
  }

  .zoom {
    margin-left: 6px;
    color: var(--color-text-muted);
    font-size: 13px;
    font-variant-numeric: tabular-nums;
  }

  svg {
    display: block;
    max-width: 100%;
    margin: 0 auto;
    background: #ffffff;
    border: 1px solid var(--color-border);
    border-radius: 8px;
    cursor: grab;
    touch-action: none;
    user-select: none;
    font-family: "Inter Variable", Inter, Helvetica, Arial, sans-serif;
    font-size: 12px;
  }

  svg.dragging {
    cursor: grabbing;
  }

  svg:focus-visible {
    outline: 2px solid var(--color-focus);
    outline-offset: 2px;
  }

  .edge {
    stroke: #000000;
    stroke-width: 1.2;
  }

  .edge.on {
    stroke: #1e88e5;
    stroke-width: 3;
  }

  .arrow-head {
    fill: #000000;
  }

  .edge-label {
    fill: #000000;
  }

  .node rect {
    stroke: #000000;
    stroke-width: 1;
  }

  .node:hover rect {
    stroke: #1e88e5;
    stroke-width: 2;
  }

  .node.on rect {
    stroke: #1e88e5;
    stroke-width: 3;
  }

  .node text {
    fill: #000000;
    pointer-events: none;
  }

  .tooltip {
    position: absolute;
    z-index: 5;
    max-width: 320px;
    padding: 8px 10px;
    background: var(--color-bg);
    color: var(--color-text);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    box-shadow: 0 6px 20px rgb(0 0 0 / 0.16);
    font-size: 12.5px;
    line-height: 1.45;
    pointer-events: none;
    white-space: nowrap;
  }

  .tooltip .first {
    font-weight: 700;
  }

  .help {
    margin: 8px 2px;
    color: var(--color-text-muted);
    font-size: 13px;
  }

  .downloads {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 8px;
  }
</style>
