// Interactive K-Nearest Neighbors demo for the hero (plain canvas 2D, no libraries).
//
// The visitor moves a probe point, adds points of a chosen class, drags or removes them,
// and changes K. The background shows the majority-vote prediction of every small cell,
// rendered at low resolution and scaled up with smoothing for soft boundaries.
import { reducedMotion, onReducedMotionChange } from "./motion.js";

const CLASS_NAMES = ["Clase A", "Clase B", "Clase C"];
const CLASS_VARS = ["--class-a", "--class-b", "--class-c"];
const CELL = 5;            // CSS px per background cell
const POINT_R = 6;         // CSS px
const HIT_R = 12;          // CSS px for picking a point
const MAX_POINTS = 160;
const SEED = 42;           // like random_state=42 in the app

/** Small deterministic PRNG (mulberry32). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  // Box-Muller; 1 - rand() avoids log(0).
  const u = 1 - rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function initialPoints() {
  const rand = mulberry32(SEED);
  const centers = [
    [0.26, 0.34, 0.11],
    [0.72, 0.3, 0.1],
    [0.56, 0.72, 0.12],
  ];
  const points = [];
  centers.forEach(([cx, cy, sd], c) => {
    for (let i = 0; i < 16; i++) {
      const u = Math.min(0.97, Math.max(0.03, cx + gaussian(rand) * sd));
      const v = Math.min(0.97, Math.max(0.03, cy + gaussian(rand) * sd * 0.9));
      points.push({ u, v, c });
    }
  });
  return points;
}

function parseColor(value) {
  const hex = value.trim().replace("#", "");
  if (/^[0-9a-f]{6}$/i.test(hex)) {
    return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  }
  return [128, 128, 128];
}

const rgba = ([r, g, b], a) => `rgba(${r}, ${g}, ${b}, ${a})`;

export function initKnnDemo({ canvas, kInput, kOutput, classInputs, resetButton, readout }) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const bg = document.createElement("canvas");
  const bgCtx = bg.getContext("2d");

  let points = initialPoints();
  let k = Number(kInput.value) || 5;
  let selectedClass = 0;
  let width = 0;
  let height = 0;
  let dpr = 1;
  let colors = [];
  let surface = [255, 255, 255];
  let text = [15, 23, 42];
  let bgDirty = true;

  // Probe: where it is drawn, and where it is heading.
  const probe = { u: 0.5, v: 0.5 };
  const target = { u: 0.5, v: 0.5 };
  let mode = "auto"; // auto | pointer | keyboard
  let idleTimer = 0;
  let t = 0;
  let dragging = -1;
  let hover = -1;
  let visible = true;
  let frame = 0;
  let lastAnnounced = "";

  function readColors() {
    const style = getComputedStyle(document.documentElement);
    colors = CLASS_VARS.map((name) => parseColor(style.getPropertyValue(name)));
    surface = parseColor(style.getPropertyValue("--surface-2"));
    text = parseColor(style.getPropertyValue("--text"));
    bgDirty = true;
  }

  // ---- KNN maths -------------------------------------------------------
  // Distances are measured in CSS pixels so that what looks closest is closest.

  /** Indices of the k nearest points to (x, y), nearest first (insertion into a sorted list). */
  function nearest(x, y, kk, skip = -1) {
    const idx = [];
    const dist = [];
    for (let i = 0; i < points.length; i++) {
      if (i === skip) continue;
      const dx = points[i].u * width - x;
      const dy = points[i].v * height - y;
      const d = dx * dx + dy * dy;
      if (idx.length === kk && d >= dist[kk - 1]) continue;
      let j = idx.length < kk ? idx.length : kk - 1;
      while (j > 0 && dist[j - 1] > d) {
        dist[j] = dist[j - 1];
        idx[j] = idx[j - 1];
        j--;
      }
      dist[j] = d;
      idx[j] = i;
    }
    return idx;
  }

  /** Majority vote; a tie goes to the class of the nearest tied neighbour. */
  function vote(neighbours) {
    const counts = [0, 0, 0];
    for (const i of neighbours) counts[points[i].c]++;
    const best = Math.max(...counts);
    let cls = 0;
    for (const i of neighbours) {
      if (counts[points[i].c] === best) {
        cls = points[i].c;
        break;
      }
    }
    return { cls, votes: best, total: neighbours.length };
  }

  // ---- Rendering -------------------------------------------------------

  function renderBackground() {
    const gw = Math.max(1, Math.ceil(width / CELL));
    const gh = Math.max(1, Math.ceil(height / CELL));
    bg.width = gw;
    bg.height = gh;
    const image = bgCtx.createImageData(gw, gh);
    const data = image.data;
    const kk = Math.min(k, points.length);
    for (let gy = 0; gy < gh; gy++) {
      for (let gx = 0; gx < gw; gx++) {
        const o = (gy * gw + gx) * 4;
        if (kk === 0) {
          data[o + 3] = 0;
          continue;
        }
        const { cls, votes, total } = vote(nearest((gx + 0.5) * CELL, (gy + 0.5) * CELL, kk));
        const conf = votes / total;
        const [r, g, b] = colors[cls];
        data[o] = r;
        data[o + 1] = g;
        data[o + 2] = b;
        data[o + 3] = Math.round(40 + 120 * conf * conf);
      }
    }
    bgCtx.putImageData(image, 0, 0);
    bgDirty = false;
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function draw() {
    if (!width || !height) return;
    if (bgDirty) renderBackground();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = rgba(surface, 1);
    ctx.fillRect(0, 0, width, height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bg, 0, 0, bg.width * CELL, bg.height * CELL);

    // Light grid, like the app's plots.
    ctx.strokeStyle = rgba(text, 0.07);
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < 10; i++) {
      const x = Math.round((i * width) / 10) + 0.5;
      const y = Math.round((i * height) / 10) + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
    }
    ctx.stroke();

    const px = probe.u * width;
    const py = probe.v * height;
    const kk = Math.min(k, points.length);
    const neighbours = kk ? nearest(px, py, kk) : [];
    const result = kk ? vote(neighbours) : null;
    const neighbourSet = new Set(neighbours);

    // Lines from the probe to its neighbours.
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.6;
    for (const i of neighbours) {
      ctx.strokeStyle = rgba(colors[points[i].c], 0.9);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(points[i].u * width, points[i].v * height);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Points.
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const x = p.u * width;
      const y = p.v * height;
      const r = i === hover || i === dragging ? POINT_R + 2 : POINT_R;
      if (neighbourSet.has(i)) {
        ctx.beginPath();
        ctx.arc(x, y, r + 4, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(text, 0.75);
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = rgba(colors[p.c], 1);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = rgba(surface, 1);
      ctx.stroke();
    }

    // Probe.
    if (result) {
      const color = colors[result.cls];
      const pulse = reducedMotion() ? 0 : (Math.sin(t * 3) + 1) * 3;
      ctx.beginPath();
      ctx.arc(px, py, 16 + pulse, 0, Math.PI * 2);
      ctx.fillStyle = rgba(color, 0.18);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(px, py, 10, 0, Math.PI * 2);
      ctx.fillStyle = rgba(color, 1);
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = rgba(text, 0.9);
      ctx.stroke();
      // Star-ish cross in the centre.
      ctx.strokeStyle = rgba(surface, 1);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px - 4, py);
      ctx.lineTo(px + 4, py);
      ctx.moveTo(px, py - 4);
      ctx.lineTo(px, py + 4);
      ctx.stroke();

      // Label.
      const pct = Math.round((100 * result.votes) / result.total);
      const label = `${CLASS_NAMES[result.cls]} · ${pct} %`;
      ctx.font = "600 13px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
      const tw = ctx.measureText(label).width;
      const lw = tw + 18;
      const lh = 24;
      let lx = px + 16;
      let ly = py - 36;
      if (lx + lw > width - 4) lx = px - 16 - lw;
      if (ly < 4) ly = py + 14;
      roundRect(lx, ly, lw, lh, 8);
      ctx.fillStyle = rgba(surface, 0.94);
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = rgba(color, 1);
      ctx.stroke();
      ctx.fillStyle = rgba(text, 1);
      ctx.textBaseline = "middle";
      ctx.fillText(label, lx + 9, ly + lh / 2 + 0.5);
    }

    if (mode !== "auto" && result) announce(result);
  }

  function announce(result) {
    const message = `Predicción: ${CLASS_NAMES[result.cls]} (${result.votes} de ${result.total} vecinos). ${points.length} puntos en el lienzo.`;
    if (message !== lastAnnounced) {
      lastAnnounced = message;
      readout.textContent = message;
    }
  }

  // ---- Animation loop --------------------------------------------------

  function step() {
    frame = 0;
    if (mode === "auto" && !reducedMotion()) {
      t += 1 / 60;
      target.u = 0.5 + 0.34 * Math.sin(t * 0.37);
      target.v = 0.5 + 0.3 * Math.sin(t * 0.53 + 1.1);
    }
    if (reducedMotion()) {
      probe.u = target.u;
      probe.v = target.v;
    } else {
      const ease = mode === "auto" ? 0.06 : 0.35;
      probe.u += (target.u - probe.u) * ease;
      probe.v += (target.v - probe.v) * ease;
    }
    draw();
    // The probe pulses, so keep animating while visible (the background is cached).
    if (!reducedMotion()) schedule();
  }

  function schedule() {
    if (!frame && visible && !document.hidden) frame = requestAnimationFrame(step);
  }

  function requestDraw() {
    if (reducedMotion()) {
      // On demand only.
      if (!frame) frame = requestAnimationFrame(step);
    } else {
      schedule();
    }
  }

  // ---- Geometry & resize -----------------------------------------------

  function resize() {
    const rect = canvas.getBoundingClientRect();
    width = Math.max(1, rect.width);
    height = Math.max(1, rect.height);
    dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    bgDirty = true;
    requestDraw();
  }

  function eventPos(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function pointAt(x, y) {
    let best = -1;
    let bestD = HIT_R * HIT_R;
    for (let i = 0; i < points.length; i++) {
      const dx = points[i].u * width - x;
      const dy = points[i].v * height - y;
      const d = dx * dx + dy * dy;
      if (d <= bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  const clamp01 = (x) => Math.min(0.99, Math.max(0.01, x));

  function userMode(next) {
    mode = next;
    clearTimeout(idleTimer);
  }

  function backToAutoLater() {
    clearTimeout(idleTimer);
    if (reducedMotion()) return;
    idleTimer = setTimeout(() => {
      if (dragging < 0 && document.activeElement !== canvas) {
        // Restart the path from where the probe is, so it does not jump.
        t = Math.asin(Math.max(-1, Math.min(1, (probe.u - 0.5) / 0.34))) / 0.37;
        mode = "auto";
        schedule();
      }
    }, 2500);
  }

  // ---- Pointer ---------------------------------------------------------

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const { x, y } = eventPos(event);
    userMode("pointer");
    const hit = pointAt(x, y);
    if (hit >= 0 && event.shiftKey) {
      points.splice(hit, 1);
      hover = -1;
      bgDirty = true;
    } else if (hit >= 0) {
      dragging = hit;
      canvas.classList.add("dragging");
    } else if (points.length < MAX_POINTS) {
      points.push({ u: clamp01(x / width), v: clamp01(y / height), c: selectedClass });
      bgDirty = true;
    }
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch {
      /* synthetic pointers */
    }
    target.u = clamp01(x / width);
    target.v = clamp01(y / height);
    requestDraw();
    event.preventDefault();
  });

  canvas.addEventListener("pointermove", (event) => {
    const { x, y } = eventPos(event);
    if (dragging >= 0) {
      points[dragging].u = clamp01(x / width);
      points[dragging].v = clamp01(y / height);
      bgDirty = true;
    } else {
      // Mouse hover moves the probe; touch only moves it while pressed.
      if (event.pointerType === "touch" && event.buttons === 0) return;
      const over = pointAt(x, y);
      if (over !== hover) {
        hover = over;
        canvas.classList.toggle("over-point", over >= 0);
      }
      target.u = clamp01(x / width);
      target.v = clamp01(y / height);
    }
    userMode("pointer");
    requestDraw();
  });

  const endDrag = () => {
    if (dragging >= 0) {
      dragging = -1;
      canvas.classList.remove("dragging");
      requestDraw();
    }
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", () => {
    if (dragging < 0) {
      hover = -1;
      canvas.classList.remove("over-point");
      backToAutoLater();
    }
  });

  // ---- Keyboard --------------------------------------------------------

  canvas.addEventListener("keydown", (event) => {
    const stepSize = event.shiftKey ? 0.06 : 0.02;
    let handled = true;
    switch (event.key) {
      case "ArrowLeft": target.u = clamp01(target.u - stepSize); break;
      case "ArrowRight": target.u = clamp01(target.u + stepSize); break;
      case "ArrowUp": target.v = clamp01(target.v - stepSize); break;
      case "ArrowDown": target.v = clamp01(target.v + stepSize); break;
      case "Enter":
      case " ":
        if (points.length < MAX_POINTS) {
          points.push({ u: target.u, v: target.v, c: selectedClass });
          bgDirty = true;
        }
        break;
      case "Delete":
      case "Backspace": {
        const hit = nearest(target.u * width, target.v * height, 1)[0];
        if (hit !== undefined) {
          const dx = points[hit].u * width - target.u * width;
          const dy = points[hit].v * height - target.v * height;
          if (dx * dx + dy * dy < 40 * 40) {
            points.splice(hit, 1);
            bgDirty = true;
          }
        }
        break;
      }
      case "Escape":
        mode = "auto";
        handled = false;
        break;
      default:
        handled = false;
    }
    if (handled) {
      event.preventDefault();
      if (mode === "auto") {
        // Start from where the probe is drawn.
        target.u = probe.u;
        target.v = probe.v;
      }
      userMode("keyboard");
    }
    requestDraw();
  });
  canvas.addEventListener("focus", () => {
    if (mode === "auto") {
      target.u = probe.u;
      target.v = probe.v;
    }
    userMode("keyboard");
    requestDraw();
  });
  canvas.addEventListener("blur", backToAutoLater);

  // ---- Controls --------------------------------------------------------

  kInput.addEventListener("input", () => {
    k = Number(kInput.value);
    kOutput.textContent = String(k);
    bgDirty = true;
    lastAnnounced = "";
    requestDraw();
  });
  classInputs.forEach((input) =>
    input.addEventListener("change", () => {
      if (input.checked) selectedClass = Number(input.value);
    }),
  );
  resetButton.addEventListener("click", () => {
    points = initialPoints();
    k = 5;
    kInput.value = "5";
    kOutput.textContent = "5";
    bgDirty = true;
    lastAnnounced = "";
    readout.textContent = "Lienzo reiniciado: 48 puntos y K = 5.";
    requestDraw();
  });

  // ---- Lifecycle -------------------------------------------------------

  readColors();
  document.addEventListener("themechange", () => {
    readColors();
    requestDraw();
  });
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver((entries) => {
    visible = entries.some((e) => e.isIntersecting);
    if (visible) requestDraw();
  }).observe(canvas);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) requestDraw();
  });
  onReducedMotionChange(() => requestDraw());
  resize();
}
