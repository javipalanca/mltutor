import { DEFAULT_ROUTE, type RouteId } from "./routes";

/** Minimal in-memory router: the app has a fixed set of top-level pages. */
class Router {
  current: RouteId = $state(DEFAULT_ROUTE);

  go(id: RouteId): void {
    this.current = id;
  }
}

export const router = new Router();
