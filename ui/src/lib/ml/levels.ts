/**
 * Quality levels stored as tables in `content/componentes/mensajes.toml`
 * (`accuracy_niveles`, `binaria_auc`, `separacion`…): `[threshold, severity, …texts]`.
 * The threshold may be a number or, for the average precision, an expression of the
 * baseline (`"baseline + 0.3"`, `"baseline"`, `"-"` = otherwise).
 */
import type { Severity } from "../../components/Notice.svelte";
import { fmt, type FmtValue } from "../fmt";

export interface LevelNotice {
  severity: Severity;
  /** Markdown (values interpolated with `fmt`, so escaped). */
  markdown: string;
}

const SEVERITIES: readonly Severity[] = ["info", "warning", "error", "success"];

function threshold(spec: unknown, baseline: number): number {
  if (typeof spec === "number") {
    return spec;
  }
  if (typeof spec !== "string" || spec.trim() === "-") {
    return Number.NEGATIVE_INFINITY;
  }
  const match = /^baseline\s*(?:\+\s*([0-9.]+))?$/.exec(spec.trim());
  return match ? baseline + Number(match[1] ?? 0) : Number.NEGATIVE_INFINITY;
}

/**
 * The first row whose threshold `value` reaches (`>=`, the default) or stays below (`<`).
 * Texts after the severity are joined with a space (D34: the icon is shown).
 */
export function pickLevel(
  table: unknown,
  value: number,
  values: Readonly<Record<string, FmtValue>> = {},
  options: { below?: boolean; baseline?: number; strict?: boolean } = {},
): LevelNotice | null {
  if (!Array.isArray(table)) {
    return null;
  }
  for (const row of table as unknown[]) {
    if (!Array.isArray(row) || row.length < 3) {
      continue;
    }
    const [spec, severity, ...texts] = row as unknown[];
    const limit = threshold(spec, options.baseline ?? 0);
    const reached = options.below ? value < limit : options.strict ? value > limit : value >= limit;
    if (reached || (!options.below && limit === Number.NEGATIVE_INFINITY)) {
      const sev = SEVERITIES.includes(severity as Severity) ? (severity as Severity) : "info";
      const text = texts.filter((t): t is string => typeof t === "string").join(" ");
      return { severity: sev, markdown: fmt(text, values) };
    }
  }
  return null;
}
