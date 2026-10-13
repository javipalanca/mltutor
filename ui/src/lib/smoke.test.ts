import { describe, expect, it } from "vitest";
import { ROUTES, routeById, type RouteId } from "./routes";
import { runSmokeTest, type SmokeDeps } from "./smoke";

function fakeDeps(overrides: Partial<SmokeDeps> = {}) {
  let current: RouteId | null = null;
  const reported: string[] = [];
  const failures: string[] = [];
  const deps: SmokeDeps = {
    panicProbe: () => Promise.resolve(true),
    navigate: (id) => {
      current = id;
    },
    waitForRender: () => Promise.resolve(),
    renderedTitle: (id) => (id === current ? routeById(id).title : null),
    report: (id) => {
      reported.push(id);
      return Promise.resolve();
    },
    fail: (message) => {
      failures.push(message);
      return Promise.resolve();
    },
    ...overrides,
  };
  return { deps, reported, failures };
}

const ALL = ROUTES.map((route) => route.id);

describe("runSmokeTest", () => {
  it("visits and reports every route in order", async () => {
    const { deps, reported, failures } = fakeDeps();
    expect(await runSmokeTest(ALL, deps)).toBe(true);
    expect(reported).toEqual(ALL);
    expect(failures).toEqual([]);
  });

  it("fails when frontend and backend route lists differ", async () => {
    const unknown = fakeDeps();
    expect(await runSmokeTest([...ALL, "extra"], unknown.deps)).toBe(false);
    expect(unknown.failures[0]).toContain("extra");

    const missing = fakeDeps();
    expect(await runSmokeTest(ALL.slice(1), missing.deps)).toBe(false);
    expect(missing.failures[0]).toContain("inicio");
  });

  it("fails when a page does not render its heading", async () => {
    const { deps, reported, failures } = fakeDeps({ renderedTitle: () => null });
    expect(await runSmokeTest(ALL, deps)).toBe(false);
    expect(reported).toEqual([]);
    expect(failures).toHaveLength(1);
  });

  it("fails when the backend did not survive the panic probe", async () => {
    const { deps, failures } = fakeDeps({ panicProbe: () => Promise.resolve(false) });
    expect(await runSmokeTest(ALL, deps)).toBe(false);
    expect(failures[0]).toContain("panic");
  });
});
