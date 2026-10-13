/**
 * Screens prepared by `--screenshot <png> --scenario <name>` (and the smoke test), named
 * after the Qt reference captures of `docs/migration/capturas/qt/` (fase1 §0).
 * CSV scenarios take the file path after a colon: `csv_cargado_clasificacion:/ruta/tips.csv`.
 * The backend reads that path from its own command line (`load_scenario_csv`): the
 * frontend never sends file paths.
 */
import type { DatasetId } from "./bindings/DatasetId";
import type { DatasetRef } from "./bindings/DatasetRef";
import type { DiagKind } from "./bindings/DiagKind";
import type { LessonId } from "./content";
import { loadScenarioCsv, setTarget } from "./csvFlow";
import { router } from "./router.svelte";
import { chooseDataset } from "./selection";
import { session, type Analysis } from "./session.svelte";

interface DatosOptions {
  lesson?: LessonId;
  dataset?: DatasetId;
  tab?: number;
  analysis?: Analysis;
  diag?: DiagKind;
  generate?: boolean;
  expanders?: (lesson: LessonId) => string[];
}

async function datos(options: DatosOptions): Promise<void> {
  const lesson = options.lesson ?? "regresion";
  const dataset: DatasetRef = { kind: "builtin", id: options.dataset ?? "iris" };
  await chooseDataset(dataset);
  session.activeTab[lesson] = options.tab ?? 0;
  const state = session.datasetTab(lesson, dataset);
  state.analysis = options.analysis ?? null;
  state.diag = options.diag ?? "kde";
  state.generated = options.generate ?? false;
  for (const id of options.expanders?.(lesson) ?? []) {
    session.setExpander(id, true);
  }
  router.go(lesson);
}

const EXPANDED = (lesson: LessonId): string[] => [
  `intro-${lesson}`,
  `datos-que-significan-${lesson}`,
  `datos-codigo-${lesson}`,
];

const SCENARIOS: Readonly<Record<string, (arg: string | undefined) => Promise<void>>> = {
  inicio: () => Promise.resolve(router.go("inicio")),
  datos_iris: () => datos({}),
  datos_iris_expandido: () => datos({ expanders: EXPANDED }),
  datos_correlacion_iris: () => datos({ analysis: "correlation" }),
  datos_pairplot_opciones: () =>
    datos({ analysis: "pairplot", expanders: (l) => [`pairplot-consejos-${l}`] }),
  datos_pairplot_kde_iris: () => datos({ analysis: "pairplot", generate: true }),
  datos_pairplot_hist_iris: () => datos({ analysis: "pairplot", diag: "hist", generate: true }),
  datos_diabetes: () => datos({ dataset: "diabetes" }),
  datos_diabetes_expandido: () =>
    datos({ dataset: "diabetes", expanders: (l) => [`datos-estadisticas-objetivo-${l}`] }),
  datos_pairplot_diabetes: () => datos({ dataset: "diabetes", analysis: "pairplot", generate: true }),
  datos_correlacion_digitos: () => datos({ dataset: "digits", analysis: "correlation" }),
  datos_titanic_desbalance: () => datos({ dataset: "titanic" }),
  datos_moons: () => datos({ dataset: "moons" }),
  datos_redes_tips: () => datos({ lesson: "redes", dataset: "tips" }),
  regresion_pestana_evaluacion_dataset_actual: () => datos({ tab: 2 }),
  csv_vacio: () => Promise.resolve(router.go("csv")),
  csv_cargado_clasificacion: async (path) => {
    router.go("csv");
    await loadCsv(path);
  },
  csv_cargado_clasificacion_expandido: async (path) => {
    router.go("csv");
    await loadCsv(path);
    session.setExpander("csv-estadisticas", true);
    session.setExpander("csv-distribucion", true);
  },
  csv_cargado_regresion_expandido: async (path) => {
    router.go("csv");
    await loadCsv(path);
    await setTarget("tip");
    session.setExpander("csv-estadisticas", true);
    session.setExpander("csv-distribucion", true);
  },
};

async function loadCsv(path: string | undefined): Promise<void> {
  if (!path) {
    throw new Error("Este escenario necesita la ruta del CSV: <nombre>:<ruta.csv>");
  }
  await loadScenarioCsv();
  if (session.csv.error) {
    throw new Error(session.csv.error);
  }
}

export const SCENARIO_NAMES = Object.keys(SCENARIOS);

export async function runScenario(spec: string): Promise<void> {
  const colon = spec.indexOf(":");
  const name = colon < 0 ? spec : spec.slice(0, colon);
  const arg = colon < 0 ? undefined : spec.slice(colon + 1);
  const scenario = SCENARIOS[name];
  if (!scenario) {
    throw new Error(`Escenario desconocido: ${name}. Disponibles: ${SCENARIO_NAMES.join(", ")}`);
  }
  await scenario(arg);
}
