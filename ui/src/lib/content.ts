/**
 * Pedagogical texts from `content/` (AGENTS.md rule 8), bundled at build time: Markdown as
 * raw strings and TOML parsed by the Vite plugin in `vite.config.ts`.
 */
import navegacion from "@content/inicio/navegacion.toml";
import tarjetas from "@content/inicio/tarjetas.toml";
import datosMensajes from "@content/datos/mensajes.toml";
import csvMensajes from "@content/csv/mensajes.toml";
import csvEjemplos from "@content/csv/ejemplos.toml";

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
