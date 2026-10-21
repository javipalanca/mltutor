// Downloads: detects the visitor's OS, asks the GitHub REST API for the releases and points
// each button at the matching asset of the newest release that has the Rust downloads.
//
// States: loading → ready (a v0.4.0+ release exists) | soon (no such release yet) | error.
// For local checks, `?releases=none|rust|error|ratelimit|offline` replaces the API call.

const REPO = "javipalanca/mltutor";
const API = `https://api.github.com/repos/${REPO}/releases?per_page=30`;
const RELEASES_PAGE = `https://github.com/${REPO}/releases`;
const CACHE_KEY = "mltutor-releases-v1";
const CACHE_MS = 10 * 60 * 1000;
const TIMEOUT_MS = 8000;

export const ASSETS = {
  windows: "mltutor-windows-x86_64.exe",
  macos: "mltutor-macos-arm64.zip",
  linux: "mltutor-linux-x86_64.AppImage",
};

const OS_NAMES = { windows: "Windows", macos: "macOS", linux: "Linux" };

/** "windows" | "macos" | "linux" | "mobile" | "unknown" */
export function detectOs(nav = navigator) {
  const ua = nav.userAgent || "";
  const platform = (nav.userAgentData && nav.userAgentData.platform) || nav.platform || "";
  const mobileHint = nav.userAgentData && nav.userAgentData.mobile;
  if (/android|iphone|ipad|ipod/i.test(ua) || mobileHint) return "mobile";
  // iPadOS presents itself as a Mac with a touch screen.
  if (/mac/i.test(platform) && nav.maxTouchPoints > 1) return "mobile";
  if (/win/i.test(platform) || /windows/i.test(ua)) return "windows";
  if (/mac/i.test(platform) || /mac os x/i.test(ua)) return "macos";
  if (/linux|x11|cros/i.test(platform) || /linux|x11/i.test(ua)) return "linux";
  return "unknown";
}

// ---------- Release data ----------

class ReleaseError extends Error {
  constructor(kind, message, resetAt = null) {
    super(message);
    this.kind = kind; // ratelimit | http | network | timeout
    this.resetAt = resetAt;
  }
}

function slim(releases) {
  return releases.map((r) => ({
    tag_name: r.tag_name,
    name: r.name,
    html_url: r.html_url,
    draft: r.draft,
    prerelease: r.prerelease,
    published_at: r.published_at,
    assets: (r.assets || []).map((a) => ({ name: a.name, size: a.size, browser_download_url: a.browser_download_url })),
  }));
}

function mock(kind) {
  const python = {
    tag_name: "v0.3.0-rc.4",
    name: "v0.3.0-rc.4",
    html_url: `${RELEASES_PAGE}/tag/v0.3.0-rc.4`,
    draft: false,
    prerelease: true,
    published_at: "2026-10-01T10:00:00Z",
    // Same asset names as the real Python v0.3.0-rc.4: they must NOT count as a Rust release.
    assets: [
      { name: "mltutor-linux-x86_64", size: 881e6, browser_download_url: RELEASES_PAGE },
      { name: ASSETS.macos, size: 476e6, browser_download_url: RELEASES_PAGE },
      { name: ASSETS.windows, size: 675e6, browser_download_url: RELEASES_PAGE },
    ],
  };
  const rust = {
    tag_name: "v0.4.0-rc.1",
    name: "v0.4.0-rc.1",
    html_url: `${RELEASES_PAGE}/tag/v0.4.0-rc.1`,
    draft: false,
    prerelease: true,
    published_at: "2026-10-21T10:00:00Z",
    assets: [
      { name: ASSETS.windows, size: 22.4e6, browser_download_url: `${RELEASES_PAGE}/download/v0.4.0-rc.1/${ASSETS.windows}` },
      { name: ASSETS.macos, size: 8.7e6, browser_download_url: `${RELEASES_PAGE}/download/v0.4.0-rc.1/${ASSETS.macos}` },
      { name: ASSETS.linux, size: 86.1e6, browser_download_url: `${RELEASES_PAGE}/download/v0.4.0-rc.1/${ASSETS.linux}` },
      { name: "SHA256SUMS.txt", size: 300, browser_download_url: `${RELEASES_PAGE}/download/v0.4.0-rc.1/SHA256SUMS.txt` },
    ],
  };
  switch (kind) {
    case "rust": return Promise.resolve([rust, python]);
    case "none": return Promise.resolve([python]);
    case "ratelimit": return Promise.reject(new ReleaseError("ratelimit", "rate limit", new Date(Date.now() + 25 * 60000)));
    case "offline": return Promise.reject(new ReleaseError("network", "offline"));
    default: return Promise.reject(new ReleaseError("http", "HTTP 500"));
  }
}

async function fetchReleases() {
  try {
    const cached = JSON.parse(sessionStorage.getItem(CACHE_KEY) || "null");
    if (cached && Date.now() - cached.t < CACHE_MS && Array.isArray(cached.data)) return cached.data;
  } catch {
    /* storage blocked or corrupt: ask the API */
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let response;
  try {
    response = await fetch(API, {
      headers: { Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28" },
      signal: controller.signal,
      credentials: "omit",
      referrerPolicy: "no-referrer",
    });
  } catch (error) {
    throw new ReleaseError(error.name === "AbortError" ? "timeout" : "network", String(error));
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    const remaining = response.headers.get("x-ratelimit-remaining");
    const reset = Number(response.headers.get("x-ratelimit-reset"));
    if (response.status === 429 || (response.status === 403 && remaining === "0")) {
      throw new ReleaseError("ratelimit", `HTTP ${response.status}`, reset ? new Date(reset * 1000) : null);
    }
    throw new ReleaseError("http", `HTTP ${response.status}`);
  }
  const json = await response.json();
  if (!Array.isArray(json)) throw new ReleaseError("http", "respuesta inesperada");
  const data = slim(json);
  try {
    sessionStorage.setItem(CACHE_KEY, JSON.stringify({ t: Date.now(), data }));
  } catch {
    /* no cache */
  }
  return data;
}

/** [major, minor, patch] of a tag like "v0.4.0-rc.1", or null. */
export function tagVersion(tag) {
  const m = /^v?(\d+)\.(\d+)\.(\d+)/.exec(tag || "");
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
}

/** The Rust version starts at v0.4.0 (the Python releases reuse some of the same asset names). */
export function isRustTag(tag) {
  const v = tagVersion(tag);
  return !!v && (v[0] > 0 || v[1] >= 4);
}

/** Newest published release (the API lists newest first) from v0.4.0 on with at least one Rust asset. */
export function pickRustRelease(releases) {
  const names = new Set(Object.values(ASSETS));
  return releases.find((r) => !r.draft && isRustTag(r.tag_name) && r.assets.some((a) => names.has(a.name))) || null;
}

// ---------- Rendering ----------

const dateFmt = new Intl.DateTimeFormat("es-ES", { day: "numeric", month: "long", year: "numeric" });
const timeFmt = new Intl.DateTimeFormat("es-ES", { hour: "2-digit", minute: "2-digit" });

function formatSize(bytes) {
  const mb = bytes / 1e6;
  return `${new Intl.NumberFormat("es-ES", { maximumFractionDigits: mb < 10 ? 1 : 0 }).format(mb)} MB`;
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const child of children) node.append(child);
  return node;
}

function link(href, text) {
  return el("a", { href, textContent: text });
}

function setStatus(box, kind, ...content) {
  box.className = `release-status is-${kind}`;
  box.replaceChildren(el("p", {}, ...content));
}

export function initDownloads({ cards, status, heroButton, heroLabel, osNote, tabs }) {
  const os = detectOs();
  const byOs = Object.fromEntries(cards.map((card) => [card.dataset.os, card]));

  // Highlight the visitor's system.
  if (byOs[os]) {
    const card = byOs[os];
    card.classList.add("is-recommended");
    card.querySelector(".dl-badge").hidden = false;
    card.setAttribute("aria-describedby", "os-note");
    heroLabel.textContent = `Descargar para ${OS_NAMES[os]}`;
    tabs?.selectOs(os);
    osNote.textContent =
      os === "macos"
        ? "Parece que usas macOS: te resaltamos su descarga. Es para Mac con Apple Silicon (M1 o posterior); no hay versión para Intel."
        : `Parece que usas ${OS_NAMES[os]}: te resaltamos su descarga.`;
    osNote.hidden = false;
  } else if (os === "mobile") {
    heroLabel.textContent = "Ver las descargas";
    osNote.textContent = "Estás en un móvil o una tableta: MLTutor es una aplicación de escritorio para Windows, macOS y Linux. Abre esta página en tu ordenador para descargarla.";
    osNote.hidden = false;
  }
  heroButton.dataset.os = os;

  const param = new URLSearchParams(location.search).get("releases");
  const source = param ? mock(param) : fetchReleases();

  source
    .then((releases) => {
      const release = pickRustRelease(releases);
      if (release) showReady(release);
      else showSoon();
    })
    .catch((error) => showError(error instanceof ReleaseError ? error : new ReleaseError("http", String(error))));

  function showReady(release) {
    const date = release.published_at ? dateFmt.format(new Date(release.published_at)) : null;
    const content = [
      "✅ Última versión: ",
      el("strong", { textContent: release.tag_name }),
      release.prerelease ? " (versión de pruebas)" : "",
      date ? ` · publicada el ${date} · ` : " · ",
      link(release.html_url, "notas de la versión"),
    ];
    const sums = release.assets.find((a) => a.name === "SHA256SUMS.txt");
    if (sums) content.push(" · ", link(sums.browser_download_url, "SHA256SUMS.txt"));
    setStatus(status, "ready", ...content);

    for (const [key, card] of Object.entries(byOs)) {
      const asset = release.assets.find((a) => a.name === ASSETS[key]);
      const action = card.querySelector("[data-action]");
      const meta = card.querySelector("[data-meta]");
      card.classList.remove("is-soon");
      if (asset) {
        card.href = asset.browser_download_url;
        card.classList.add("is-available");
        action.textContent = `⬇️ Descargar ${release.tag_name}`;
        meta.textContent = formatSize(asset.size);
        card.setAttribute("aria-label", `Descargar MLTutor ${release.tag_name} para ${OS_NAMES[key]}, ${ASSETS[key]}, ${formatSize(asset.size)}`);
      } else {
        card.href = release.html_url;
        action.textContent = "No disponible en esta versión";
        meta.textContent = "Consulta la página de la versión";
      }
    }
  }

  function showSoon() {
    const content = [
      "🚧 ",
      el("strong", { textContent: "Próximamente." }),
      " La versión 1.0.0 de MLTutor está a punto de publicarse.",
    ];
    content.push(" ", link(RELEASES_PAGE, "Ver todas las releases"));
    setStatus(status, "soon", ...content);
    for (const [key, card] of Object.entries(byOs)) {
      card.href = RELEASES_PAGE;
      card.classList.add("is-soon");
      card.querySelector("[data-action]").textContent = "Próximamente · ver releases";
      card.setAttribute("aria-label", `MLTutor para ${OS_NAMES[key]}: próximamente. Abre la página de releases de GitHub.`);
    }
  }

  function showError(error) {
    let reason;
    if (error.kind === "ratelimit") {
      reason = error.resetAt
        ? `GitHub limita las consultas sin cuenta y se ha alcanzado el límite; vuelve a intentarlo a partir de las ${timeFmt.format(error.resetAt)}.`
        : "GitHub limita las consultas sin cuenta y se ha alcanzado el límite; vuelve a intentarlo más tarde.";
    } else if (error.kind === "network" || error.kind === "timeout") {
      reason = "No se ha podido conectar con GitHub (¿sin conexión?).";
    } else {
      reason = `GitHub ha respondido con un error (${error.message}).`;
    }
    setStatus(status, "error", `⚠️ No se ha podido consultar la última versión. ${reason} Las descargas están en la `, link(RELEASES_PAGE, "página de Releases"), ".");
    for (const card of Object.values(byOs)) {
      card.href = RELEASES_PAGE;
      card.querySelector("[data-action]").textContent = "Ver releases en GitHub";
    }
  }
}
