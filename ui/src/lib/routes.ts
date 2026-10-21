/**
 * Navigation model of the app. Order, emojis and labels come from
 * `content/inicio/navegacion.toml` (Python: `mltutor/app.py`, sidebar buttons).
 */
import { NAVIGATION, headingText, type LessonId } from "./content";

export type RouteId = "inicio" | LessonId | "csv";

export interface RouteDef {
  readonly id: RouteId;
  /** Sidebar button text, exactly as in the Python version. */
  readonly label: string;
  /** Page heading (`st.header`), checked by the smoke test. */
  readonly title: string;
  /** Line under the heading. */
  readonly subtitle: string;
}

export interface NavSection {
  readonly heading: string;
  readonly routes: readonly RouteDef[];
}

const SPECIAL: Readonly<Record<"inicio" | "csv", { title: string; subtitle: string }>> = {
  inicio: {
    title: "🌟 Bienvenido a 🎓 MLTutor",
    subtitle: "Una plataforma interactiva para aprender Machine Learning",
  },
  csv: {
    title: "📁 Cargar CSV Personalizado",
    subtitle: "Carga tu propio dataset en formato CSV para análisis personalizado",
  },
};

export function isLessonId(id: string): id is LessonId {
  return id in NAVIGATION.lessons;
}

function routeDef(id: string, label: string): RouteDef {
  if (isLessonId(id)) {
    const lesson = NAVIGATION.lessons[id];
    return { id, label, title: lesson.title, subtitle: lesson.subtitle };
  }
  if (id === "inicio" || id === "csv") {
    return { id, label, ...SPECIAL[id] };
  }
  throw new Error(`Ruta desconocida en navegacion.toml: ${id}`);
}

export const NAV_SECTIONS: readonly NavSection[] = NAVIGATION.sections.map((section) => ({
  heading: headingText(section.heading),
  routes: section.routes.map((route) => routeDef(route.id, route.label)),
}));

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
