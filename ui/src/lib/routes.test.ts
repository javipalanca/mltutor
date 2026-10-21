import { describe, expect, it } from "vitest";
import { DEFAULT_ROUTE, NAV_SECTIONS, ROUTES, isRouteId, routeById } from "./routes";

describe("navigation", () => {
  it("keeps the sidebar order and labels of the Python version", () => {
    expect(NAV_SECTIONS.map((section) => section.heading)).toEqual(["Navegación:", "🔧 Herramientas:"]);
    expect(NAV_SECTIONS.map((section) => section.routes.map((route) => route.label))).toEqual([
      ["🏠 Inicio", "🌲 Árboles de Decisión", "📊 Regresión", "🔍 K-Nearest Neighbors", "🧠 Redes Neuronales"],
      ["📁 Cargar CSV Personalizado"],
    ]);
  });

  it("has unique ids and starts at Inicio", () => {
    const ids = ROUTES.map((route) => route.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(DEFAULT_ROUTE).toBe("inicio");
    expect(isRouteId("knn")).toBe(true);
    expect(isRouteId("nope")).toBe(false);
    expect(routeById("csv").title).toBe("📁 Cargar CSV Personalizado");
  });
});
