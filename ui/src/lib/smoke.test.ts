import { describe, expect, it } from "vitest";
import { ROUTES, routeById, type RouteId } from "./routes";
import { runSmokeTest, type SmokeDeps } from "./smoke";

const CHECKS = ["datos_vistas", "datos_correlacion", "datos_pairplot"];

function fakeDeps(overrides: Partial<SmokeDeps> = {}) {
  let current: RouteId | null = null;
  const reported: string[] = [];
  const checks: string[] = [];
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
    runCheck: () => Promise.resolve(),
    reportCheck: (name) => {
      checks.push(name);
      return Promise.resolve();
    },
    fail: (message) => {
      failures.push(message);
      return Promise.resolve();
    },
    ...overrides,
  };
  return { deps, reported, checks, failures };
}

const ALL = ROUTES.map((route) => route.id);

describe("runSmokeTest", () => {
  it("visits every route in order and then runs the data checks", async () => {
    const { deps, reported, checks, failures } = fakeDeps();
    expect(await runSmokeTest(ALL, CHECKS, deps)).toBe(true);
    expect(reported).toEqual(ALL);
    expect(checks).toEqual(CHECKS);
    expect(failures).toEqual([]);
  });

  it("fails when frontend and backend route lists differ", async () => {
    const unknown = fakeDeps();
    expect(await runSmokeTest([...ALL, "extra"], CHECKS, unknown.deps)).toBe(false);
    expect(unknown.failures[0]).toContain("extra");

    const missing = fakeDeps();
    expect(await runSmokeTest(ALL.slice(1), CHECKS, missing.deps)).toBe(false);
    expect(missing.failures[0]).toContain("inicio");
  });

  it("fails when a page does not render its heading", async () => {
    const { deps, reported, failures } = fakeDeps({ renderedTitle: () => null });
    expect(await runSmokeTest(ALL, CHECKS, deps)).toBe(false);
    expect(reported).toEqual([]);
    expect(failures).toHaveLength(1);
  });

  it("fails when the backend did not survive the panic probe", async () => {
    const { deps, failures } = fakeDeps({ panicProbe: () => Promise.resolve(false) });
    expect(await runSmokeTest(ALL, CHECKS, deps)).toBe(false);
    expect(failures[0]).toContain("panic");
  });

  it("names the data check that failed and stops", async () => {
    const { deps, checks, failures } = fakeDeps({
      runCheck: (name) => (name === "datos_correlacion" ? Promise.reject(new Error("sin heatmap")) : Promise.resolve()),
    });
    expect(await runSmokeTest(ALL, CHECKS, deps)).toBe(false);
    expect(checks).toEqual(["datos_vistas"]);
    expect(failures[0]).toContain("datos_correlacion");
    expect(failures[0]).toContain("sin heatmap");
  });
});
