<script lang="ts" module>
  import type { KnnMetric } from "../../bindings/KnnMetric";
  import type { KnnWeights } from "../../bindings/KnnWeights";

  /** A test point of the canvas (stable `id` for the table and the drag). */
  export interface CanvasPoint {
    id: number;
    x: number;
    y: number;
  }

  /** The state of the canvas, kept by the lesson (it survives navigation, D13). */
  export interface CanvasState {
    pair: [number, number];
    k: number;
    weights: KnnWeights;
    metric: KnnMetric;
    showConfidence: boolean;
    showDistances: boolean;
    points: CanvasPoint[];
  }

  export interface CanvasController {
    readonly playground: CanvasState;
    /** `null` when the canvas already has `MAX_PLAYGROUND_POINTS` points. */
    addPoint(x: number, y: number): CanvasPoint | null;
    movePoint(id: number, x: number, y: number): void;
    removePoint(id: number): void;
    clearPoints(): void;
    setPair(pair: [number, number]): void;
  }
</script>

<script lang="ts">
  /**
   * C18 (D54/D55): the interactive KNN canvas, native (Plotly in the page, no embedded
   * HTML). Click to add a test point, drag to move it, or type exact values; every change
   * asks Rust (`knn_playground`, sklearn's rules on a 2D KNN over every row, D63), debounced
   * and cancelling the previous request. The lines to the K neighbours appear one by one,
   * nearest first (`prefers-reduced-motion` and the automated mode show them at once).
   */
  import { onDestroy, untrack } from "svelte";
  import type { Matrix } from "../../bindings/Matrix";
  import type { PlaygroundView } from "../../bindings/PlaygroundView";
  import Checkbox from "../../../components/Checkbox.svelte";
  import Expander from "../../../components/Expander.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Notice from "../../../components/Notice.svelte";
  import NumberInput from "../../../components/NumberInput.svelte";
  import Select from "../../../components/Select.svelte";
  import Slider from "../../../components/Slider.svelte";
  import Spinner from "../../../components/Spinner.svelte";
  import * as backend from "../../backend";
  import type { Cancellable } from "../../backend";
  import { knnRaw, knnText, md } from "../../content";
  import { fmt, fmtText } from "../../fmt";
  import { fixed } from "../../format";
  import { loadPlotly, type PlotlyModule } from "../../plotly";
  import { theme } from "../../theme.svelte";
  import { MAX_PLAYGROUND_POINTS, MAX_POINTS_MESSAGE } from "../../lessons/knn/state.svelte";
  import { classColor, clampPoint, percent, playgroundFigure, type PixelScale } from "./knnFigures";
  import MetricGrid from "./MetricGrid.svelte";

  interface Props {
    modelId: number;
    features: readonly string[];
    controller: CanvasController;
    /** Bounds of the "Vecinos (K)" slider (D55: 1..20). */
    kMax?: number;
    onstale: () => void;
  }

  let { modelId, features, controller, kMax = 20, onstale }: Props = $props();

  const t = (key: string): string => knnText("interactiva", key);
  const pg = $derived(controller.playground);
  const featureOptions = $derived(features.map((label, value) => ({ label, value })));
  const same = $derived(pg.pair[0] === pg.pair[1]);
  type Pair = readonly (readonly [string, string])[];
  const pairs = (key: string): Pair => (knnRaw("interactiva", key) as Pair | undefined) ?? [];
  const metricOptions = pairs("metrica_opciones").map(([value, label]) => ({ value: value as KnnMetric, label }));
  const weightOptions = pairs("pesos_opciones").map(([value, label]) => ({ value: value as KnnWeights, label }));

  // --- Requests to the backend ---

  let view = $state<PlaygroundView | null>(null);
  /** Data of the pair (sent only when the pair changes). */
  let cache = $state<{ key: string; data: Matrix; codes: number[] } | null>(null);
  let error = $state<string | null>(null);
  let inflight: Cancellable<PlaygroundView> | null = null;
  let dragging = $state<number | null>(null);
  /** Round trips of the last requests (ms), for `data-latency` and the log. */
  const latencies: number[] = [];
  let latency = $state<number | null>(null);

  const pairKey = $derived(`${modelId}|${pg.pair[0]}|${pg.pair[1]}`);

  $effect(() => {
    if (same) {
      return;
    }
    const key = pairKey;
    const request = {
      model_id: modelId,
      feature_x: pg.pair[0],
      feature_y: pg.pair[1],
      n_neighbors: pg.k,
      weights: pg.weights,
      metric: pg.metric,
      points: pg.points.map((p) => [p.x, p.y] as [number, number]),
      include_data: untrack(() => cache?.key) !== key,
    };
    // While dragging, one request per frame; otherwise a short debounce.
    const delay = untrack(() => dragging) !== null ? 0 : 40;
    const timer = setTimeout(() => send(key, request), delay);
    return () => clearTimeout(timer);
  });

  function send(key: string, request: Parameters<typeof backend.knnPlayground>[0]): void {
    inflight?.cancel();
    const started = performance.now();
    const call = backend.knnPlayground(request);
    inflight = call;
    call.promise
      .then((next) => {
        if (inflight !== call) {
          return;
        }
        inflight = null;
        if (next.data && next.codes) {
          cache = { key, data: next.data, codes: next.codes };
        }
        view = next;
        error = null;
        const elapsed = performance.now() - started;
        latencies.push(elapsed);
        if (latencies.length > 60) {
          latencies.shift();
        }
        latency = elapsed;
      })
      .catch((reason: unknown) => {
        if (inflight !== call) {
          return;
        }
        inflight = null;
        if (backend.isStale(reason)) {
          onstale();
        } else if (!backend.isCancelled(reason)) {
          error = backend.errorMessage(reason);
        }
      });
  }

  /** Median round trip of the last requests (smoke test and log). */
  export function medianLatency(): number | null {
    if (latencies.length === 0) {
      return null;
    }
    const sorted = [...latencies].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)] ?? null;
  }

  const ready = $derived(view !== null && cache !== null && cache.key === pairKey);

  // --- Animation of the lines to the neighbours (D54) ---

  const STEP_MS = 60;
  const animated = new Map<number, number>();
  let tick = $state(0);
  let frame = 0;

  function reducedMotion(): boolean {
    return (
      document.documentElement.dataset["automation"] === "true" ||
      (typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches)
    );
  }

  function animate(id: number): void {
    if (reducedMotion()) {
      return;
    }
    animated.set(id, performance.now());
    if (frame === 0) {
      const loop = (): void => {
        const now = performance.now();
        for (const [key, start] of animated) {
          if (now - start > STEP_MS * (kMax + 1)) {
            animated.delete(key);
          }
        }
        tick += 1;
        frame = animated.size > 0 ? requestAnimationFrame(loop) : 0;
      };
      frame = requestAnimationFrame(loop);
    }
  }

  const revealed = $derived.by(() => {
    void tick;
    const now = performance.now();
    return pg.points.map((p) => {
      const start = animated.get(p.id);
      return start === undefined ? Number.POSITIVE_INFINITY : Math.floor((now - start) / STEP_MS) + 1;
    });
  });

  // --- Plotly ---

  let container: HTMLDivElement | undefined = $state();
  let plotly: PlotlyModule | null = null;
  let rendered = $state(false);
  let failed = $state<string | null>(null);
  let width = $state(900);
  const height = $derived(Math.round(Math.max(420, Math.min(640, (width * 2) / 3))));
  let scale = $state<PixelScale>({ dx: 0.01, dy: 0.01 });

  interface Axis {
    _offset: number;
    _length: number;
    range: [number, number];
    p2l(px: number): number;
    l2p(v: number): number;
  }

  function axes(): { xa: Axis; ya: Axis } | null {
    const full = (container as unknown as { _fullLayout?: { xaxis?: Axis; yaxis?: Axis } } | undefined)?._fullLayout;
    return full?.xaxis && full.yaxis ? { xa: full.xaxis, ya: full.yaxis } : null;
  }

  function measure(): void {
    const a = axes();
    if (!a || a.xa._length <= 0 || a.ya._length <= 0) {
      return;
    }
    const next = {
      dx: (a.xa.range[1] - a.xa.range[0]) / a.xa._length,
      dy: (a.ya.range[1] - a.ya.range[0]) / a.ya._length,
    };
    if (Math.abs(next.dx - scale.dx) > 1e-12 || Math.abs(next.dy - scale.dy) > 1e-12) {
      scale = next;
    }
  }

  /** The view with the points where the user has them now (the drag runs ahead). */
  const shownView = $derived.by(() => {
    if (!view) {
      return null;
    }
    const local = pg.points;
    if (local.length !== view.points.length) {
      return view;
    }
    return { ...view, points: view.points.map((p, i) => ({ ...p, point: [local[i]?.x ?? p.point[0], local[i]?.y ?? p.point[1]] as [number, number] })) };
  });

  const figure = $derived.by(() => {
    if (!shownView || !cache || cache.key !== pairKey) {
      return null;
    }
    return playgroundFigure(
      shownView,
      cache.data,
      cache.codes,
      {
        showConfidence: pg.showConfidence,
        showDistances: pg.showDistances,
        revealed,
        dragging,
        scale,
        webgl: cache.data.rows > 3000,
      },
      theme.chart,
    );
  });

  let observer: ResizeObserver | undefined;

  $effect(() => {
    const target = container;
    const fig = figure;
    if (!target || !fig) {
      return;
    }
    const layout = { ...fig.layout, height, autosize: true };
    void (async () => {
      try {
        plotly ??= await loadPlotly();
        await plotly.react(target, fig.data, layout, {
          responsive: false,
          displaylogo: false,
          displayModeBar: "hover",
          modeBarButtonsToRemove: ["toImage", "sendDataToCloud", "lasso2d", "select2d", "zoom2d", "pan2d", "autoScale2d"],
          scrollZoom: false,
          doubleClick: false,
          locale: "es",
        });
        if (!rendered) {
          rendered = true;
          (target as unknown as { on?: (event: string, cb: () => void) => void }).on?.("plotly_relayout", () => measure());
        }
        failed = null;
        target.dataset["renderedAt"] = String(performance.now());
        measure();
      } catch (reason) {
        failed = reason instanceof Error ? reason.message : String(reason);
      }
    })();
  });

  $effect(() => {
    const target = container;
    if (!target || observer) {
      return;
    }
    width = target.clientWidth || width;
    observer = new ResizeObserver(() => {
      width = target.clientWidth || width;
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => {
        if (plotly && rendered) {
          void Promise.resolve(plotly.Plots.resize(target)).then(measure);
        }
      });
    });
    observer.observe(target);
  });
  let resizeFrame = 0;

  onDestroy(() => {
    inflight?.cancel();
    observer?.disconnect();
    cancelAnimationFrame(frame);
    cancelAnimationFrame(resizeFrame);
    if (plotly && container) {
      plotly.purge(container);
    }
  });

  // --- Pointer: click to add, drag to move ---

  const HIT_PX = 16;
  let moveFrame = 0;
  let moveTimer: ReturnType<typeof setTimeout> | undefined;
  let moveScheduled = false;
  let pendingMove: [number, number] | null = null;

  function toData(event: PointerEvent): [number, number] | null {
    const a = axes();
    if (!a || !container) {
      return null;
    }
    const rect = container.getBoundingClientRect();
    const px = event.clientX - rect.left - a.xa._offset;
    const py = event.clientY - rect.top - a.ya._offset;
    if (px < 0 || py < 0 || px > a.xa._length || py > a.ya._length) {
      return null;
    }
    return [a.xa.p2l(px), a.ya.p2l(py)];
  }

  function clampToBounds(point: [number, number]): [number, number] {
    return view ? clampPoint(point, view.bounds) : point;
  }

  function hit(event: PointerEvent): number | null {
    const a = axes();
    if (!a || !container) {
      return null;
    }
    const rect = container.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    let best: number | null = null;
    let bestDistance = HIT_PX;
    for (const p of pg.points) {
      const dx = a.xa._offset + a.xa.l2p(p.x) - px;
      const dy = a.ya._offset + a.ya.l2p(p.y) - py;
      const d = Math.hypot(dx, dy);
      if (d <= bestDistance) {
        best = p.id;
        bestDistance = d;
      }
    }
    return best;
  }

  function pointerdown(event: PointerEvent): void {
    if (event.button !== 0 || !ready) {
      return;
    }
    const position = toData(event);
    if (!position) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const existing = hit(event);
    // A full canvas only lets the user drag the existing points (the click adds nothing).
    const id = existing ?? controller.addPoint(...clampToBounds(position))?.id ?? null;
    if (id === null) {
      return;
    }
    dragging = id;
    capture(event.pointerId, true);
  }

  /** Pointer capture keeps the drag when the cursor leaves the plot; it throws for an unknown pointer. */
  function capture(pointerId: number, on: boolean): void {
    try {
      if (on) {
        container?.setPointerCapture(pointerId);
      } else if (container?.hasPointerCapture(pointerId)) {
        container.releasePointerCapture(pointerId);
      }
    } catch {
      // Synthetic pointers (tests) cannot be captured: the drag still works inside the plot.
    }
  }

  function pointermove(event: PointerEvent): void {
    if (dragging === null) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const a = axes();
    if (!a || !container) {
      return;
    }
    const rect = container.getBoundingClientRect();
    const px = Math.min(a.xa._length, Math.max(0, event.clientX - rect.left - a.xa._offset));
    const py = Math.min(a.ya._length, Math.max(0, event.clientY - rect.top - a.ya._offset));
    pendingMove = [a.xa.p2l(px), a.ya.p2l(py)];
    if (!moveScheduled) {
      moveScheduled = true;
      // One update per frame; the timer covers windows where frames do not run.
      let done = false;
      const apply = (): void => {
        if (done) {
          return;
        }
        done = true;
        cancelAnimationFrame(moveFrame);
        clearTimeout(moveTimer);
        moveScheduled = false;
        if (dragging !== null && pendingMove) {
          controller.movePoint(dragging, ...clampToBounds(pendingMove));
        }
      };
      moveFrame = requestAnimationFrame(apply);
      moveTimer = setTimeout(apply, 32);
    }
  }

  function pointerup(event: PointerEvent): void {
    if (dragging === null) {
      return;
    }
    event.stopPropagation();
    if (moveScheduled) {
      cancelAnimationFrame(moveFrame);
      clearTimeout(moveTimer);
      moveScheduled = false;
      if (pendingMove) {
        controller.movePoint(dragging, ...clampToBounds(pendingMove));
      }
    }
    const id = dragging;
    dragging = null;
    pendingMove = null;
    capture(event.pointerId, false);
    animate(id);
  }

  // --- Exact values (Python's fallback, D54) ---

  let exact = $state<[number | null, number | null]>([null, null]);
  const centre = $derived(view ? [(view.bounds[0] + view.bounds[1]) / 2, (view.bounds[2] + view.bounds[3]) / 2] : [0, 0]);
  const exactValues = $derived<[number, number]>([exact[0] ?? round1(centre[0] ?? 0), exact[1] ?? round1(centre[1] ?? 0)]);

  function round3(v: number): number {
    return Math.round(v * 1000) / 1000;
  }

  function round1(v: number): number {
    return Math.round(v * 10) / 10;
  }

  function addExact(): void {
    const point = controller.addPoint(...clampToBounds(exactValues));
    if (point) {
      animate(point.id);
    }
  }

  const full = $derived(pg.points.length >= MAX_PLAYGROUND_POINTS);

  // --- Table of the test points ---

  const featureNames = $derived([features[pg.pair[0]] ?? "", features[pg.pair[1]] ?? ""] as const);
  const stats = $derived.by(() => {
    const n = pg.points.length;
    const confidence = view?.mean_confidence ?? null;
    return [
      { label: t("stat_puntos"), value: n },
      { label: t("stat_confianza"), value: confidence === null || n === 0 ? "0%" : percent(confidence) },
      { label: t("stat_k"), value: view?.n_neighbors ?? pg.k },
    ];
  });
  const columns = $derived(
    (knnRaw("interactiva", "columnas") as readonly string[] | undefined)?.map((c) =>
      fmtText(c, { f1: featureNames[0], f2: featureNames[1] }),
    ) ?? [],
  );

  function editPoint(id: number, axis: 0 | 1, value: number): void {
    const p = pg.points.find((q) => q.id === id);
    if (!p || !Number.isFinite(value)) {
      return;
    }
    // Typed values are kept inside the canvas, as the drag and the backend do.
    const [x, y] = clampToBounds([axis === 0 ? value : p.x, axis === 1 ? value : p.y]);
    controller.movePoint(id, x, y);
    animate(id);
  }

  function setControl(update: Partial<CanvasState>): void {
    Object.assign(controller.playground, update);
    for (const p of pg.points) {
      animate(p.id);
    }
  }
</script>

<section class="playground" data-playground data-playground-ready={ready} data-points={pg.points.length} data-latency={latency === null ? "" : latency.toFixed(1)}>
  <Markdown source={t("titulo")} />
  <p>{t("texto")}</p>
  <div class="pair">
    <Select id="knn-interactiva-x" label={t("f1_label")} options={featureOptions} value={pg.pair[0]} onchange={(x) => controller.setPair([x, pg.pair[1]])} />
    <Select id="knn-interactiva-y" label={t("f2_label")} options={featureOptions} value={pg.pair[1]} onchange={(y) => controller.setPair([pg.pair[0], y])} />
  </div>
  {#if same}
    <Notice severity="warning" markdown={t("misma_caracteristica")} />
  {:else}
    <Notice severity="info" markdown={`**${t("caja_titulo")}**\n\n${t("caja_texto")}`} />
    <div class="controls">
      <div class="k">
        <Slider id="knn-lienzo-k" label={t("k_label")} min={1} max={kMax} step={1} value={pg.k} onchange={(v) => setControl({ k: Math.round(v) })} />
      </div>
      <Select id="knn-lienzo-metrica" label={t("metrica_label")} options={metricOptions} value={pg.metric} onchange={(metric) => setControl({ metric })} />
      <Select id="knn-lienzo-pesos" label={t("pesos_label")} options={weightOptions} value={pg.weights} onchange={(weights) => setControl({ weights })} />
      <div class="toggle">
        <label class="toggle-label" for="knn-lienzo-confianza">{t("confianza_label")}</label>
        <Checkbox id="knn-lienzo-confianza" label="" checked={pg.showConfidence} onchange={(v) => (controller.playground.showConfidence = v)} />
      </div>
      <div class="toggle">
        <label class="toggle-label" for="knn-lienzo-distancias">{t("distancias_label")}</label>
        <Checkbox id="knn-lienzo-distancias" label="" checked={pg.showDistances} onchange={(v) => (controller.playground.showDistances = v)} />
      </div>
      <button type="button" class="btn clear" disabled={pg.points.length === 0} onclick={() => controller.clearPoints()}>{t("btn_limpiar")}</button>
    </div>

    {#if error}
      <Notice severity="error" markdown={fmt(t("error_html"), { error })} />
    {/if}
    {#if full}
      <Notice severity="warning" markdown={MAX_POINTS_MESSAGE} />
    {/if}
    <div class="canvas" class:dragging={dragging !== null} class:full data-canvas-full={full}>
      <div
        class="plot"
        bind:this={container}
        style:height="{height}px"
        role="img"
        aria-label="{t('caja_titulo').replace(/[*#]/g, '')}: {pg.points.length} puntos de prueba. Con el teclado, usa «{t('modo_exacto')}»."
        data-plot-rendered={failed ? "error" : rendered}
        onpointerdowncapture={pointerdown}
        onpointermovecapture={pointermove}
        onpointerupcapture={pointerup}
        onpointercancelcapture={pointerup}
      ></div>
      {#if !ready && !failed}
        <div class="loading"><Spinner label="Preparando el lienzo…" /></div>
      {/if}
      {#if failed}
        <p class="error">No se pudo dibujar el gráfico: {failed}</p>
      {/if}
    </div>
    <p class="caption">{t("punto_editar")}</p>
    <p class="caption"><Markdown inline source={t("nota_modelo_2d")} /></p>

    <MetricGrid items={stats} />

    <Expander id="knn-lienzo-exacto" label={t("modo_exacto")}>
      <div class="exact">
        <NumberInput
          id="knn-exacto-x"
          label={fmtText(t("valor_label"), { feature: featureNames[0] })}
          min={view?.bounds[0] ?? -1e9}
          max={view?.bounds[1] ?? 1e9}
          step={0.1}
          value={exactValues[0]}
          onchange={(v) => (exact = [v, exact[1]])}
        />
        <NumberInput
          id="knn-exacto-y"
          label={fmtText(t("valor_label"), { feature: featureNames[1] })}
          min={view?.bounds[2] ?? -1e9}
          max={view?.bounds[3] ?? 1e9}
          step={0.1}
          value={exactValues[1]}
          onchange={(v) => (exact = [exact[0], v])}
        />
        <button type="button" class="btn primary add" disabled={!ready || full} title={full ? MAX_POINTS_MESSAGE : undefined} onclick={addExact}>{t("btn_agregar")}</button>
      </div>
    </Expander>

    {#if pg.points.length === 0}
      <Notice severity="info" markdown={md("knn/interactiva-como-usar.md")} />
    {:else if view && view.points.length === pg.points.length}
      <Markdown source={t("analisis_titulo")} />
      <div class="table-wrap">
        <table class="points" data-test-points>
          <thead>
            <tr>
              {#each columns as column (column)}<th>{column}</th>{/each}
              <th aria-label="Quitar"></th>
            </tr>
          </thead>
          <tbody>
            {#each pg.points as point, i (point.id)}
              {@const result = view.points[i]}
              <tr>
                <td>{i + 1}</td>
                <td>
                  <input type="number" step="any" value={round3(point.x)} aria-label="{featureNames[0]} del punto {i + 1}" onchange={(e) => editPoint(point.id, 0, Number(e.currentTarget.value))} />
                </td>
                <td>
                  <input type="number" step="any" value={round3(point.y)} aria-label="{featureNames[1]} del punto {i + 1}" onchange={(e) => editPoint(point.id, 1, Number(e.currentTarget.value))} />
                </td>
                <td>
                  {#if result}
                    <span class="swatch" style:background={classColor(result.class_index, view.colors)}></span>{view.class_names[result.class_index] ?? result.class_index}
                  {/if}
                </td>
                <td class="num">{result ? `${fixed(result.confidence * 100, 1)}%` : ""}</td>
                <td class="num">{result ? fixed(result.mean_distance, 3) : ""}</td>
                <td><button type="button" class="remove" aria-label="Quitar el punto {i + 1}" title="Quitar el punto" onclick={() => controller.removePoint(point.id)}>✕</button></td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      <MetricGrid
        items={[
          { label: t("metrica_total"), value: pg.points.length },
          { label: t("metrica_confianza"), value: view.mean_confidence === null ? "—" : `${fixed(view.mean_confidence * 100, 1)}%` },
          { label: t("metrica_parametros"), value: fmtText(t("metrica_parametros_valor"), { k: view.n_neighbors, metric: pg.metric }) },
        ]}
      />
    {/if}
    <Expander id="knn-lienzo-guia" label={t("guia_expander")}>
      <Markdown source={md("knn/interactiva-guia.md")} />
    </Expander>
  {/if}
</section>

<style>
  .pair {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 28px;
  }

  .controls {
    display: grid;
    grid-template-columns: minmax(200px, 1.4fr) repeat(2, minmax(140px, 1fr)) repeat(2, auto) auto;
    gap: 18px;
    align-items: end;
    margin: 6px 0 4px;
  }

  /* 1024 px windows: K, metric and weights on one row; the toggles and "Limpiar" below. */
  @container page (max-width: 900px) {
    .controls {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }

  .toggle {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    margin: 12px 0;
  }

  .toggle-label {
    font-size: 14px;
  }

  .clear {
    margin: 12px 0;
    min-height: 46px;
    /* #ff4444 in Python; darker so the white text reaches 4.5:1 (WCAG AA). */
    background: #d32f2f;
    border-color: #d32f2f;
    color: #ffffff;
    font-weight: 600;
  }

  .clear:hover:not(:disabled) {
    background: #c62828;
    border-color: #c62828;
  }

  .clear:disabled {
    opacity: 0.55;
  }

  .canvas {
    position: relative;
    margin: 10px 0 8px;
    padding: 10px 12px 4px;
    border: 2px solid #d6dbe4;
    border-radius: 12px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    box-shadow: inset 0 1px 3px rgba(20, 30, 60, 0.08);
  }

  .plot {
    width: 100%;
    cursor: crosshair;
    touch-action: none;
  }

  .full .plot {
    cursor: not-allowed;
  }

  .dragging .plot {
    cursor: grabbing;
  }

  .loading {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
  }

  .caption {
    margin: 4px 2px;
    color: var(--color-text-muted);
    font-size: 13px;
    line-height: 1.45;
  }

  .exact {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr)) auto;
    gap: 18px;
    align-items: end;
  }

  .add {
    margin: 12px 0;
    min-height: 46px;
  }

  .table-wrap {
    overflow-x: auto;
  }

  table.points {
    width: 100%;
    margin: 8px 0 12px;
    border-collapse: collapse;
    font-size: 14px;
  }

  .points th,
  .points td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--color-border, #e3e7ee);
    text-align: left;
    white-space: nowrap;
  }

  .points th {
    color: var(--color-text-muted);
    font-weight: 600;
  }

  .points td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  .points input {
    width: 110px;
    padding: 4px 8px;
    background: var(--color-input-bg);
    border: 1px solid var(--color-input-border);
    border-radius: 6px;
    font-variant-numeric: tabular-nums;
  }

  .swatch {
    display: inline-block;
    width: 11px;
    height: 11px;
    margin-right: 7px;
    border: 1px solid #333333;
    border-radius: 50%;
    vertical-align: -1px;
  }

  .remove {
    padding: 2px 8px;
    border: 0;
    background: transparent;
    color: var(--color-text-muted);
    cursor: pointer;
  }

  .remove:hover {
    color: #e53935;
  }

  .error {
    color: var(--color-text-muted);
  }

  :global(:root[data-theme="dark"]) .canvas {
    border-color: #3a4558;
    background: linear-gradient(135deg, #1f2733 0%, #2c3a52 100%);
  }
</style>
