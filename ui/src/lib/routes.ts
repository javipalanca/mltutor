/**
 * Navigation model of the app. Order, emojis and labels mirror the Python
 * version (`mltutor/app.py`, sidebar buttons) and must stay in sync with it.
 */

export type RouteId = "inicio" | "arboles" | "regresion" | "knn" | "redes" | "csv";

export interface RouteDef {
  readonly id: RouteId;
  /** Sidebar button text, exactly as in the Python version. */
  readonly label: string;
  /** Page heading. */
  readonly title: string;
  /** Short line shown under the heading. */
  readonly subtitle: string;
}

export interface NavSection {
  readonly heading: string;
  readonly routes: readonly RouteDef[];
}

export const NAV_SECTIONS: readonly NavSection[] = [
  {
    heading: "Navegación:",
    routes: [
      {
        id: "inicio",
        label: "🏠 Inicio",
        title: "🌟 Bienvenido a 🎓 MLTutor",
        subtitle: "Una plataforma interactiva para aprender Machine Learning",
      },
      {
        id: "arboles",
        label: "🌲 Árboles de Decisión",
        title: "🌲 Árboles de Decisión",
        subtitle: "Clasificación y regresión con árboles de decisión",
      },
      {
        id: "regresion",
        label: "📊 Regresión",
        title: "📊 Regresión",
        subtitle: "Regresión lineal y regresión logística",
      },
      {
        id: "knn",
        label: "🔍 K-Nearest Neighbors",
        title: "🔍 K-Nearest Neighbors (KNN)",
        subtitle: "Clasificación y regresión por vecinos más cercanos",
      },
      {
        id: "redes",
        label: "🧠 Redes Neuronales",
        title: "🧠 Redes Neuronales",
        subtitle: "Diseña, entrena y analiza redes neuronales",
      },
    ],
  },
  {
    heading: "🔧 Herramientas:",
    routes: [
      {
        id: "csv",
        label: "📁 Cargar CSV Personalizado",
        title: "📁 Cargar CSV Personalizado",
        subtitle: "Carga tu propio dataset en formato CSV para análisis personalizado",
      },
    ],
  },
];

export const ROUTES: readonly RouteDef[] = NAV_SECTIONS.flatMap((section) => section.routes);

export const DEFAULT_ROUTE: RouteId = "inicio";

export function isRouteId(value: string): value is RouteId {
  return ROUTES.some((route) => route.id === value);
}

export function routeById(id: RouteId): RouteDef {
  const route = ROUTES.find((candidate) => candidate.id === id);
  if (!route) {
    throw new Error(`Ruta desconocida: ${id}`);
  }
  return route;
}
