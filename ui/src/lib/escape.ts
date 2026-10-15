/**
 * Escaping of values that come from data (column names, class values, file names, error
 * messages) before they are mixed with our Markdown/HTML or Plotly labels. Our content
 * may contain raw HTML; data values must always be shown as plain text.
 */

const ENTITIES: Readonly<Record<string, string>> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

/** HTML special characters → entities. */
export function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => ENTITIES[char] ?? char);
}

/**
 * A value that renders literally inside Markdown: HTML is neutralised (entities), every
 * ASCII punctuation mark Markdown may interpret is backslash-escaped (no emphasis, links,
 * autolinks, `$…$` math or headings) and line breaks become spaces (no new blocks).
 */
export function escapeMarkdown(text: string): string {
  return text
    .replace(/\r\n?|\n/g, " ")
    .replace(/[&<>"']|[\\`*_{}[\]()#+\-.!|~$:/=%@^,;?]/g, (char) => ENTITIES[char] ?? `\\${char}`);
}

/**
 * Plotly text (titles, legend, ticks, hover labels) understands a few HTML tags
 * (`<a href>`, `<b>`…) and entities: data values are shown literally with entities.
 */
export function escapePlotly(text: string): string {
  return text.replace(/[&<>]/g, (char) => ENTITIES[char] ?? char);
}

/**
 * A value inserted into a Plotly `hovertemplate`/`texttemplate`: [`escapePlotly`] plus `%`
 * as the entity `&#37;`. Plotly has no escape for `%{…}` (`%%` does not work): it scans the
 * template with `/%{([^\s%{}:]*)…}/`, and a column called `precio %{x}` would show the x
 * value. The entity is converted back to `%` when the label is drawn. Values passed
 * through `customdata`/`text` are not scanned again, so they only need [`escapePlotly`].
 */
export function escapePlotlyTemplate(text: string): string {
  return escapePlotly(text).replace(/%/g, "&#37;");
}
