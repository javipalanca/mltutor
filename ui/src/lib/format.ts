/** Number formatting shared by tables and charts. */
import { toFixedPython } from "./fmt";

/** Python's `format(v, ".6g")`: 6 significant digits, no trailing zeros, exponent outside 1e-4..1e6. */
export function formatG(value: number, precision = 6): string {
  if (Number.isNaN(value)) {
    return "nan";
  }
  if (!Number.isFinite(value)) {
    return value > 0 ? "inf" : "-inf";
  }
  if (value === 0) {
    return Object.is(value, -0) ? "-0" : "0";
  }
  const exponent = Math.floor(Math.log10(Math.abs(Number(value.toPrecision(precision)))));
  if (exponent < -4 || exponent >= precision) {
    const [mantissa = "", exp = "0"] = value.toExponential(precision - 1).split("e");
    const trimmed = mantissa.includes(".") ? mantissa.replace(/\.?0+$/, "") : mantissa;
    const expNumber = Number(exp);
    const sign = expNumber < 0 ? "-" : "+";
    return `${trimmed}e${sign}${String(Math.abs(expNumber)).padStart(2, "0")}`;
  }
  const decimals = Math.max(0, precision - 1 - exponent);
  const fixed = value.toFixed(decimals);
  return fixed.includes(".") ? fixed.replace(/\.?0+$/, "") : fixed;
}

/** `{:.Nf}` */
export function fixed(value: number, decimals: number): string {
  return Number.isFinite(value) ? toFixedPython(value, decimals) : formatG(value);
}

/** Thousands with a thin separator, as Python's `{:,}`. */
export function grouped(value: number): string {
  return String(Math.trunc(value)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
