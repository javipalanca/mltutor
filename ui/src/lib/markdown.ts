/**
 * Markdown → HTML for `content/*.md`, with KaTeX formulas (`$…$`, `$$…$$`) and Python
 * highlighting. Everything is bundled locally; the content is ours, so raw HTML is allowed
 * except comments (they document where each text comes from).
 */
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import katex from "katex";
import { Marked, type TokenizerAndRendererExtension } from "marked";

hljs.registerLanguage("python", python);

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function highlightPython(code: string): string {
  try {
    return hljs.highlight(code, { language: "python", ignoreIllegals: true }).value;
  } catch {
    return escapeHtml(code);
  }
}

function renderMath(tex: string, displayMode: boolean): string {
  return katex.renderToString(tex, { displayMode, throwOnError: false, output: "htmlAndMathml" });
}

const blockMath: TokenizerAndRendererExtension = {
  name: "blockMath",
  level: "block",
  start: (src) => src.indexOf("$$"),
  tokenizer(src) {
    const match = /^\$\$([\s\S]+?)\$\$(?:\n|$)/.exec(src);
    return match ? { type: "blockMath", raw: match[0], text: (match[1] ?? "").trim() } : undefined;
  },
  renderer: (token) => `<div class="math-block">${renderMath(String(token["text"]), true)}</div>`,
};

const inlineMath: TokenizerAndRendererExtension = {
  name: "inlineMath",
  level: "inline",
  start: (src) => src.indexOf("$"),
  tokenizer(src) {
    // `$x$`, not `$ 5` (money) nor `$$`.
    const match = /^\$(?!\s|\$)((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)/.exec(src);
    return match ? { type: "inlineMath", raw: match[0], text: match[1] ?? "" } : undefined;
  },
  renderer: (token) => renderMath(String(token["text"]), false),
};

const marked = new Marked({
  gfm: true,
  breaks: false,
  extensions: [blockMath, inlineMath],
  renderer: {
    code({ text, lang }) {
      const language = (lang ?? "").trim().toLowerCase();
      const body = language === "python" || language === "py" ? highlightPython(text) : escapeHtml(text);
      return `<pre class="md-code"><code class="hljs">${body}</code></pre>`;
    },
  },
});

export function stripComments(markdown: string): string {
  return markdown.replace(/<!--[\s\S]*?-->\s*/g, "");
}

export function renderMarkdown(markdown: string): string {
  return marked.parse(stripComments(markdown), { async: false });
}

export function renderInline(markdown: string): string {
  return marked.parseInline(stripComments(markdown), { async: false });
}
