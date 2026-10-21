/**
 * Classes of a trained classifier versus classes of its dataset. `class_names` is
 * indexed by the dataset's class code; the model only knows the codes it saw in the
 * train set (`classes_`, `model_classes`). Row `i` of `coef_` belongs to
 * `model_classes[i]`, and with two seen classes the model is binary and its positive
 * class is `classes_[1]`, whatever the number of classes of the dataset.
 */
import type { TrainedModelInfo } from "../bindings/TrainedModelInfo";

export interface CoefficientClasses {
  /** One option per row of `coef_` (multiclass): label = class name, value = row. */
  options: { label: string; value: number }[];
  /** More than two classes seen: one row of coefficients per class. */
  multiclass: boolean;
  /** Name of the positive class of a binary model (`classes_[1]`). */
  positive: string;
}

type ClassInfo = Pick<TrainedModelInfo, "class_names" | "model_classes">;

/** Name of the dataset class with code `code` ("Clase i" if unnamed, as elsewhere). */
function nameOf(info: ClassInfo, code: number): string {
  return info.class_names?.[code] ?? String(code);
}

export function coefficientClasses(info: ClassInfo): CoefficientClasses {
  const names = info.class_names ?? [];
  const seen = info.model_classes ?? names.map((_, i) => i);
  return {
    options: seen.map((code, row) => ({ label: nameOf(info, code), value: row })),
    multiclass: seen.length > 2,
    positive: seen.length >= 2 ? nameOf(info, seen[1] ?? 1) : (names[1] ?? "1"),
  };
}

/** Name of the class of row `row` of `coef_`. */
export function classOfRow(info: ClassInfo, row: number): string {
  const seen = info.model_classes;
  return seen ? nameOf(info, seen[row] ?? row) : nameOf(info, row);
}

/** Number of rows of `coef_` the class selector can choose among. */
export function coefficientRows(info: ClassInfo): number {
  return (info.model_classes ?? info.class_names ?? []).length;
}
