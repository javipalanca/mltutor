import { ROUTES, isRouteId, routeById, type RouteId } from "./routes";

/** Side effects needed by the smoke test, injected so the logic is unit-testable. */
export interface SmokeDeps {
  panicProbe(): Promise<boolean>;
  navigate(id: RouteId): void;
  /** Resolves once the DOM reflects the last navigation. */
  waitForRender(): Promise<void>;
  /** Heading currently rendered for `id`, or `null` if that page is not on screen. */
  renderedTitle(id: RouteId): string | null;
  report(id: RouteId, title: string): Promise<void>;
  /** Runs a data check (`SMOKE_CHECKS` of `src-tauri/src/state.rs`); rejects on failure. */
  runCheck(name: string): Promise<void>;
  reportCheck(name: string): Promise<void>;
  fail(message: string): Promise<void>;
}

/**
 * Visits every route requested by the backend (`--smoke-test`) and reports each one
 * once its heading is rendered, then runs the data checks. The backend exits with 0 when
 * everything has been reported. Returns `false` (after calling `deps.fail`) on the first problem.
 */
export async function runSmokeTest(
  requested: readonly string[],
  checks: readonly string[],
  deps: SmokeDeps,
): Promise<boolean> {
  try {
    const unknown = requested.filter((id) => !isRouteId(id));
    if (unknown.length > 0) {
      throw new Error(`El backend pide rutas que la interfaz no conoce: ${unknown.join(", ")}`);
    }
    const missing = ROUTES.filter((route) => !requested.includes(route.id)).map((route) => route.id);
    if (missing.length > 0) {
      throw new Error(`El backend no comprueba estas rutas: ${missing.join(", ")}`);
    }
    if (!(await deps.panicProbe())) {
      throw new Error("Un panic en el backend no se recuperó correctamente");
    }
    for (const id of requested as readonly RouteId[]) {
      deps.navigate(id);
      await deps.waitForRender();
      const expected = routeById(id).title;
      const rendered = deps.renderedTitle(id);
      if (rendered !== expected) {
        throw new Error(`La ruta «${id}» no se ha renderizado (título: ${rendered ?? "ninguno"})`);
      }
      await deps.report(id, rendered);
    }
    for (const check of checks) {
      try {
        await deps.runCheck(check);
      } catch (error) {
        throw new Error(`Comprobación «${check}»: ${error instanceof Error ? error.message : String(error)}`);
      }
      await deps.reportCheck(check);
    }
    return true;
  } catch (error) {
    await deps.fail(error instanceof Error ? error.message : String(error));
    return false;
  }
}
