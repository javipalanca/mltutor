import { describe, expect, it } from "vitest";
import { criterionOptions, trees } from "./state.svelte";

describe("state of 🌲 Árboles", () => {
  it("offers the criteria of scikit-learn for each tree type", () => {
    expect(criterionOptions("classification")).toEqual(["gini", "entropy"]);
    expect(criterionOptions("regression")).toEqual(["squared_error", "friedman_mse", "absolute_error"]);
  });

  it("goes back to the first criterion when the type changes (selectbox without key)", () => {
    trees.reset();
    trees.criterion = "entropy";
    trees.setTask("regression");
    expect(trees.criterion).toBe("squared_error");
    trees.criterion = "absolute_error";
    trees.setTask("regression");
    expect(trees.criterion).toBe("absolute_error");
  });

  it("preselects the task of each dataset once (D37)", () => {
    trees.reset();
    trees.preset("builtin:diabetes", "regression");
    expect(trees.task).toBe("regression");
    // The user may switch to classification; re-entering the same dataset keeps it.
    trees.setTask("classification");
    trees.preset("builtin:diabetes", "regression");
    expect(trees.task).toBe("classification");
    trees.preset("builtin:iris", "classification");
    expect(trees.config()).toEqual({ kind: "tree", task: "classification", criterion: "gini", max_depth: 3, min_samples_split: 2 });
  });
});
