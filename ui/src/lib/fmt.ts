/**
 * Python `str.format`-style templates used by `content/**` (`{ratio:.1f}`, `{n:,}`).
 * Supports `{name}`, `{name:.Nf}`, `{name:,}` and `{name:,.Nf}`; unknown names are left as is.
 *
 * The templates are Markdown, so `fmt` escapes the interpolated strings by default
 * (`escapeMarkdown`): a column called `<a href=…>` is shown as text. Use `fmtText` only
 * where the result is plain text (Svelte text nodes and attributes, which escape HTML).
 */
import { escapeMarkdown } from "./escape";

export type FmtValue = string | number | boolean | null | undefined;

/** Shown for a missing value (`null`, e.g. a statistic that is not finite). */
export const MISSING = "—";

const PLACEHOLDER = /\{([A-Za-z_][A-Za-z0-9_]*)(?::(,)?(?:\.(\d+)f)?)?\}/g;

/**
 * `value.toFixed(decimals)` with Python's rounding: exact ties go to the even digit
 * (`f"{0.25:.1f}"` is "0.2"; JavaScript gives "0.3").
 */
export function toFixedPython(value: number, decimals: number): string {
  const exact = Math.abs(value).toFixed(Math.min(100, decimals + 30));
  const point = exact.indexOf(".");
  const tail = exact.slice(point + 1 + decimals);
  if (!/^50*$/.test(tail)) {
    return value.toFixed(decimals);
  }
  const kept = exact.slice(0, point + 1 + decimals).replace(/\.$/, "");
  const lastDigit = Number(kept.replace(".", "").slice(-1));
  if (lastDigit % 2 === 0) {
    const sign = value < 0 && Number(kept) !== 0 ? "-" : "";
    return `${sign}${kept}`;
  }
  return value.toFixed(decimals);
}

function formatNumber(value: number, grouping: boolean, decimals: number | undefined): string {
  if (!Number.isFinite(value)) {
    return Number.isNaN(value) ? "nan" : value > 0 ? "inf" : "-inf";
  }
  const fixed = decimals === undefined ? String(value) : toFixedPython(value, decimals);
  if (!grouping) {
    return fixed;
  }
  const [int = "", frac] = fixed.split(".");
  const sign = int.startsWith("-") ? "-" : "";
  const digits = sign ? int.slice(1) : int;
  const grouped = digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return `${sign}${grouped}${frac !== undefined ? `.${frac}` : ""}`;
}

function interpolate(
  template: string,
  values: Readonly<Record<string, FmtValue>>,
  escape: (text: string) => string,
): string {
  return template.replace(PLACEHOLDER, (whole, name: string, comma?: string, decimals?: string) => {
    if (!(name in values)) {
      return whole;
    }
    const value = values[name];
    if (typeof value === "number") {
      return formatNumber(value, comma === ",", decimals === undefined ? undefined : Number(decimals));
    }
    if (value === null) {
      return MISSING;
    }
    return value === undefined ? "" : escape(String(value));
  });
}

/** Template → Markdown: interpolated strings are escaped (HTML and Markdown). */
export function fmt(template: string, values: Readonly<Record<string, FmtValue>>): string {
  return interpolate(template, values, escapeMarkdown);
}

/** Template → plain text (no escaping): only for Svelte text nodes and attributes. */
export function fmtText(template: string, values: Readonly<Record<string, FmtValue>>): string {
  return interpolate(template, values, (text) => text);
}
