/**
 * Lesson structure: tabs from `content/inicio/navegacion.toml` and the navbar messages
 * of `apps/navbar.py` (fase1-shell-datos.md §1.5).
 */
import { NAVIGATION, type LessonContent, type LessonId } from "./content";

export const LESSON_IDS: readonly LessonId[] = ["arboles", "regresion", "knn", "redes"];

export function lesson(id: LessonId): LessonContent {
  return NAVIGATION.lessons[id];
}

/** "📊 Datos" → "Datos". */
export function tabName(tab: string): string {
  return tab.replace(/^[^\p{L}\p{N}]+/u, "").trim();
}

export interface NavbarMessages {
  prev?: string;
  next?: string;
  prevNote?: string;
  nextNote?: string;
}

const NN_READY_NOTE = "**¿Listo para entrenar?** ¡Tu arquitectura está configurada!";

/** Messages of the navbar at the bottom of tab `index` of `id`. */
export function navbarMessages(id: LessonId, index: number, trained = false): NavbarMessages {
  const tabs = lesson(id).tabs.map(tabName);
  const last = tabs.length - 1;
  if (id === "redes") {
    const messages: NavbarMessages[] = [
      { next: `Continuar a ${tabs[1]}` },
      { prev: `Volver a ${tabs[0]}`, next: `Continuar a ${tabs[2]}`, nextNote: NN_READY_NOTE },
      trained ? { prev: `Volver a ${tabs[1]}`, next: `Ver ${tabs[3]}` } : { prev: `Volver a ${tabs[1]}` },
      { prev: `Volver a ${tabs[2]}`, next: `Ver ${tabs[4]}` },
      { prev: `Volver a ${tabs[3]}`, next: `Ver ${tabs[5]}` },
      { prev: `Volver a ${tabs[4]}`, next: `Ver ${tabs[6]}` },
      { prev: `Volver a ${tabs[5]}` },
    ];
    return messages[index] ?? {};
  }
  const out: NavbarMessages = {};
  if (index > 0) {
    out.prev = `Volver a ${tabs[index - 1]}`;
  }
  if (index < last) {
    out.next = `Continuar a ${tabs[index + 1]}`;
  }
  return out;
}

/** What tab `index` shows of the 📊 Datos block (§2.1). */
export type DataBlock = "full" | "selector" | "current";

export function dataBlock(id: LessonId, index: number): DataBlock {
  if (index === 0) {
    return "full";
  }
  if (index === 1 && id !== "redes") {
    return "selector";
  }
  return "current";
}

/** Phase of PLAN.md where each lesson is ported (placeholder of the other tabs). */
export const LESSON_PHASE: Readonly<Record<LessonId, string>> = {
  regresion: "Fase 2",
  arboles: "Fase 3",
  knn: "Fase 4",
  redes: "Fase 5",
};
