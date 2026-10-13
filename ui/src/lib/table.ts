/** Helpers of `DataTable.svelte` (formatting, sorting and TSV copy). */
import type { Cell } from "./bindings/Cell";
import { formatG } from "./format";

/** Text shown in a cell: numbers with `.6g` (Qt), bools as pandas, NaN empty-ish. */
export function cellText(cell: Cell): string {
  switch (cell.kind) {
    case "number":
      return formatG(cell.v);
    case "text":
      return cell.v;
    case "bool":
      return cell.v ? "True" : "False";
    case "missing":
      return "NaN";
  }
}

/** Original value for the clipboard (full precision, like Qt). */
export function cellRaw(cell: Cell): string {
  switch (cell.kind) {
    case "number":
      return String(cell.v);
    case "text":
      return cell.v;
    case "bool":
      return cell.v ? "True" : "False";
    case "missing":
      return "";
  }
}

/** Numbers before text, missing last. */
export function compareCells(a: Cell | undefined, b: Cell | undefined): number {
  const rank = (cell: Cell | undefined): number =>
    cell === undefined || cell.kind === "missing" ? 2 : cell.kind === "text" ? 1 : 0;
  const ra = rank(a);
  const rb = rank(b);
  if (ra !== rb || a === undefined || b === undefined) {
    return ra - rb;
  }
  if (a.kind === "text" && b.kind === "text") {
    return a.v.localeCompare(b.v, "es");
  }
  const value = (cell: Cell): number => (cell.kind === "number" ? cell.v : cell.kind === "bool" ? Number(cell.v) : 0);
  return value(a) - value(b);
}

export function toTsv(columns: readonly string[], rows: readonly (readonly Cell[])[], index?: readonly string[]): string {
  const header = (index ? [""] : []).concat(columns).join("\t");
  const lines = rows.map((row, i) => (index ? [index[i] ?? ""] : []).concat(row.map(cellRaw)).join("\t"));
  return [header, ...lines].join("\n");
}

/** Converts a `DescribeTable`-like matrix (row-major, `null` = NaN) into cells. */
export function matrixCells(values: readonly (number | null)[], rows: number, cols: number): Cell[][] {
  const out: Cell[][] = [];
  for (let r = 0; r < rows; r += 1) {
    const row: Cell[] = [];
    for (let c = 0; c < cols; c += 1) {
      const v = values[r * cols + c];
      row.push(v === null || v === undefined ? { kind: "missing" } : { kind: "number", v });
    }
    out.push(row);
  }
  return out;
}
