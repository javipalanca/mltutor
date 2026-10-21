import { describe, expect, it } from "vitest";
import { classOfRow, coefficientClasses, coefficientRows } from "./modelClasses";

const IRIS = ["setosa", "versicolor", "virginica"];

describe("classes of a classifier", () => {
  it("multiclass: one selector option per row of coef_", () => {
    const c = coefficientClasses({ class_names: IRIS, model_classes: [0, 1, 2] });
    expect(c.multiclass).toBe(true);
    expect(c.options).toEqual([
      { label: "setosa", value: 0 },
      { label: "versicolor", value: 1 },
      { label: "virginica", value: 2 },
    ]);
  });

  it("a class absent from the train set shifts the rows", () => {
    const info = { class_names: ["a", "b", "c", "d"], model_classes: [0, 2, 3] };
    const c = coefficientClasses(info);
    expect(c.options.map((o) => o.label)).toEqual(["a", "c", "d"]);
    expect(c.options.map((o) => o.value)).toEqual([0, 1, 2]);
    expect(classOfRow(info, 1)).toBe("c");
    expect(coefficientRows(info)).toBe(3);
  });

  it("2 of 3 classes seen: binary model, positive class = classes_[1]", () => {
    const c = coefficientClasses({ class_names: IRIS, model_classes: [0, 2] });
    expect(c.multiclass).toBe(false);
    expect(c.positive).toBe("virginica");
  });

  it("binary dataset: positive class is class 1", () => {
    const c = coefficientClasses({ class_names: ["malignant", "benign"], model_classes: [0, 1] });
    expect(c.multiclass).toBe(false);
    expect(c.positive).toBe("benign");
  });
});
