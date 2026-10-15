/**
 * Pedagogical texts from `content/` (AGENTS.md rule 8), bundled at build time: Markdown as
 * raw strings and TOML parsed by the Vite plugin in `vite.config.ts`.
 */
import navegacion from "@content/inicio/navegacion.toml";
import tarjetas from "@content/inicio/tarjetas.toml";
import datosMensajes from "@content/datos/mensajes.toml";
import csvMensajes from "@content/csv/mensajes.toml";
import csvEjemplos from "@content/csv/ejemplos.toml";
import componentesMensajes from "@content/componentes/mensajes.toml";
import regresionMensajes from "@content/regresion/mensajes.toml";
import arbolesMensajes from "@content/arboles/mensajes.toml";

const markdown = import.meta.glob<string>("../../../content/**/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
});
const templates = import.meta.glob<string>("../../../content/code/*.j2", {
  query: "?raw",
  import: "default",
  eager: true,
});

function lookup(files: Record<string, string>, path: string): string {
  const key = Object.keys(files).find((candidate) => candidate.endsWith(`/content/${path}`));
  if (key === undefined) {
    throw new Error(`Falta el contenido content/${path}`);
  }
  return files[key] ?? "";
}

/** `content/<path>` (e.g. `"datos/correlacion.md"`). */
export function md(path: string): string {
  return lookup(markdown, path);
}

export function codeTemplate(path: string): string {
  return lookup(templates, path);
}

export type LessonId = "arboles" | "regresion" | "knn" | "redes";

export interface LessonContent {
  readonly title: string;
  readonly subtitle: string;
  readonly intro_expander: string;
  readonly tabs: readonly string[];
  readonly tab_state_key: string;
}

export interface NavigationContent {
  readonly sections: readonly {
    readonly heading: string;
    readonly routes: readonly { readonly id: string; readonly label: string; readonly key: string }[];
  }[];
  readonly lessons: Readonly<Record<LessonId, LessonContent>>;
}

export interface HomeCard {
  readonly route: LessonId;
  readonly title: string;
  readonly text: string;
  readonly status?: string;
  readonly background: string;
  readonly border: string;
  readonly title_color: string;
  readonly button: string;
}

export const NAVIGATION = navegacion as unknown as NavigationContent;
export const HOME_CARDS = (tarjetas as unknown as { cards: HomeCard[] }).cards;

type Messages = Readonly<Record<string, Readonly<Record<string, string | readonly string[]>>>>;

function messages(source: unknown, file: string) {
  const data = source as Messages;
  return (section: string, key: string): string => {
    const value = data[section]?.[key];
    if (typeof value !== "string") {
      throw new Error(`Falta el texto [${section}].${key} en ${file}`);
    }
    return value;
  };
}

function messageList(source: unknown, file: string) {
  const data = source as Messages;
  return (section: string, key: string): readonly string[] => {
    const value = data[section]?.[key];
    if (!Array.isArray(value)) {
      throw new Error(`Falta la lista [${section}].${key} en ${file}`);
    }
    return value as readonly string[];
  };
}

/** `content/datos/mensajes.toml` */
export const datosText = messages(datosMensajes, "datos/mensajes.toml");
export const datosList = messageList(datosMensajes, "datos/mensajes.toml");
/** `content/csv/mensajes.toml` */
export const csvText = messages(csvMensajes, "csv/mensajes.toml");
export const csvList = messageList(csvMensajes, "csv/mensajes.toml");

/** `content/componentes/mensajes.toml` (shared evaluation/visualisation components). */
export const compText = messages(componentesMensajes, "componentes/mensajes.toml");
export const compList = messageList(componentesMensajes, "componentes/mensajes.toml");
/** `content/regresion/mensajes.toml` */
export const regText = messages(regresionMensajes, "regresion/mensajes.toml");
export const regList = messageList(regresionMensajes, "regresion/mensajes.toml");

/** `content/arboles/mensajes.toml` */
export const arbText = messages(arbolesMensajes, "arboles/mensajes.toml");
export const arbList = messageList(arbolesMensajes, "arboles/mensajes.toml");

function rawValue(source: unknown) {
  const data = source as Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  return (section: string, key: string): unknown => data[section]?.[key];
}

/** Any value (tables of thresholds, panels…) of `componentes/mensajes.toml`; `undefined` if absent. */
export const compRaw = rawValue(componentesMensajes);
/** Any value of `regresion/mensajes.toml`; `undefined` if absent. */
export const regRaw = rawValue(regresionMensajes);
/** Any value of `arboles/mensajes.toml` (panels of subviews); `undefined` if absent. */
export const arbRaw = rawValue(arbolesMensajes);

/** Optional text (new texts of the content author may not exist yet): `fallback` if absent. */
export function optionalText(raw: (section: string, key: string) => unknown, section: string, key: string, fallback: string): string {
  const value = raw(section, key);
  return typeof value === "string" ? value : fallback;
}

/** Whether `content/<path>` exists (optional texts). */
export function hasMd(path: string): boolean {
  return Object.keys(markdown).some((candidate) => candidate.endsWith(`/content/${path}`));
}

export interface CsvExample {
  readonly name: string;
  readonly description: string;
  readonly task: string;
  readonly icon: string;
}

export const CSV_EXAMPLES = csvEjemplos as unknown as { heading: string; examples: CsvExample[] };

/** `"### Título"` → `"Título"` (headings stored with their Markdown level). */
export function headingText(markdownHeading: string): string {
  return markdownHeading.replace(/^#+\s*/, "");
}

/** Footer lines of `content/inicio/pie.md`. */
export function footerLines(): string[] {
  return md("inicio/pie.md")
    .replace(/<!--[\s\S]*?-->/g, "")
    .split("\n")
    .map((line) => line.replace(/^-\s*/, "").trim())
    .filter(Boolean);
}
