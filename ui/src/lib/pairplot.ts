/** Pairplot option rules of `dataset_tab.py` (fase1-shell-datos.md §2.6). */

export const PAIRPLOT_MAX_FEATURES = 6;
export const PAIRPLOT_DEFAULT_FEATURES = 4;

export interface PairplotDefaults {
  /** Number of features of the dataset. */
  n: number;
  /** `null` when n ≤ 2 (Python shows an info instead of the slider). */
  slider: { min: number; max: number } | null;
  defaultMax: number;
}

export function pairplotDefaults(features: readonly string[]): PairplotDefaults {
  const n = features.length;
  return {
    n,
    slider: n > 2 ? { min: 2, max: Math.min(PAIRPLOT_MAX_FEATURES, n) } : null,
    defaultMax: Math.min(PAIRPLOT_DEFAULT_FEATURES, n),
  };
}

export interface PairplotNotice {
  key: "cuidado" | "nota";
  severity: "warning" | "info";
  values: Record<string, number>;
}

/** "⚠️ Cuidado" above 25 plots, "ℹ️ Nota" above 16 (with Python's time estimate). */
export function pairplotNotices(k: number, rows: number): PairplotNotice[] {
  const total = k * k;
  if (total > 25) {
    return [{ key: "cuidado", severity: "warning", values: { n: k, total } }];
  }
  if (total > 16) {
    return [{ key: "nota", severity: "info", values: { n: k, total, t: (total * rows) / 50000 } }];
  }
  return [];
}
