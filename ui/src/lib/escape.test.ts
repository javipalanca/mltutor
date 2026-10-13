import { cleanup, render } from "@testing-library/svelte";
import { afterEach, describe, expect, it } from "vitest";
import Notice from "../components/Notice.svelte";
import type { PairplotData } from "./bindings/PairplotData";
import { csvTargetFigure, pairplotFigure } from "./charts";
import { csvText, datosText } from "./content";
import { escapeMarkdown, escapePlotly } from "./escape";
import { fmt, fmtText, MISSING } from "./fmt";
import { renderMarkdown } from "./markdown";

afterEach(() => cleanup());

const LINK = '<a href="https://example.com">pulsa</a>';
const STYLED = '<div style="position:fixed;inset:0">tapa</div>';

/** Renders `markdown` like a `Notice` and returns its element. */
function noticeOf(markdown: string): HTMLElement {
  const { container } = render(Notice, { markdown });
  const box = container.querySelector<HTMLElement>(".notice");
  if (!box) {
    throw new Error("no notice");
  }
  return box;
}

describe("data values in Markdown are text", () => {
  it("shows column headers with HTML as text (finding 3)", () => {
    for (const header of [LINK, STYLED]) {
      const box = noticeOf(fmt(csvText("selector", "columna"), { col: header }));
      expect(box.querySelector("a, div[style]")).toBeNull();
      expect(box.textContent).toContain(header);
    }
  });

  it("does not turn data into Markdown, links or math", () => {
    const tricky = "**negrita** _x_ [enlace](https://a.b) https://c.d $x^2$ # h `c` <b>";
    const html = renderMarkdown(fmt("Valor: {v}", { v: tricky }));
    expect(html).not.toMatch(/<(strong|em|a|code|b)\b/);
    expect(html).not.toContain("katex");
    const box = noticeOf(fmt("Valor: {v}", { v: tricky }));
    expect(box.textContent?.trim()).toBe(`Valor: ${tricky}`);
  });

  it("keeps a value on one line and escapes whole messages", () => {
    const box = noticeOf(escapeMarkdown(`❌ Columna objetivo '${STYLED}'\n# no es título`));
    expect(box.querySelector("h1, div[style]")).toBeNull();
    expect(box.textContent?.trim()).toBe(`❌ Columna objetivo '${STYLED}' # no es título`);
  });

  it("escapes the dataset label and the file name of the 📊 Datos and CSV notices", () => {
    const label = `📄 ${LINK}.csv`;
    for (const markdown of [
      fmt(datosText("info", "seleccionado"), { dataset: label }),
      fmt(datosText("seleccion", "dataset_actual"), { dataset: label }),
      fmt(csvText("selector", "anadido"), { nombre: `${LINK}.csv` }),
    ]) {
      const box = noticeOf(markdown);
      expect(box.querySelector("a")).toBeNull();
      expect(box.textContent).toContain(LINK);
    }
  });

  it("fmtText leaves text untouched and both show missing values as a dash", () => {
    expect(fmtText("• {valor}: {n}", { valor: "a_b*", n: 2 })).toBe("• a_b*: 2");
    expect(fmt("Media: {v:.4f}", { v: null })).toBe(`Media: ${MISSING}`);
    expect(fmtText("Media: {v:.4f}", { v: null })).toBe(`Media: ${MISSING}`);
  });
});

describe("data values in Plotly labels are text", () => {
  it("escapes class names, columns and tooltips and keeps them out of the templates", () => {
    const plot: PairplotData = {
      features: [LINK, "b"],
      n_points: 2,
      n_total: 2,
      sampled: false,
      points: { rows: 2, cols: 2, values: [1, 2, 3, 4] },
      hue: { kind: "categorical", levels: ["%{x}", STYLED], codes: [0, 1] },
      tooltips: [LINK, "ok"],
      diagonal: [
        { kind: "hist", edges: [0, 1], series: [{ level: "%{x}", counts: [1] }] },
        { kind: "hist", edges: [0, 1], series: [{ level: STYLED, counts: [1] }] },
      ],
      high_quality: false,
      title: "Matriz",
    };
    const figure = pairplotFigure(plot);
    // `legendgroup` is an identifier, never rendered.
    const text = JSON.stringify(figure, (key, value: unknown) => (key === "legendgroup" ? undefined : value));
    expect(text).not.toContain("<a ");
    expect(text).not.toContain("<div");
    expect(text).toContain(JSON.stringify(escapePlotly(LINK)).slice(1, -1));
    for (const trace of figure.data as { hovertemplate?: string }[]) {
      expect(trace.hovertemplate ?? "").not.toContain("%{x}<br>%{x}");
      expect(trace.hovertemplate ?? "").not.toContain("<div");
    }
    const csv = csvTargetFigure(STYLED, { kind: "classes", value_counts: [[LINK, 3]] });
    expect(JSON.stringify(csv)).not.toMatch(/<a |<div/);
  });

  it("draws a continuous hue with missing values and no range", () => {
    const plot: PairplotData = {
      features: ["a", "b"],
      n_points: 2,
      n_total: 2,
      sampled: false,
      points: { rows: 2, cols: 2, values: [1, 2, 3, 4] },
      hue: { kind: "continuous", values: [1, null], min: null, max: null },
      tooltips: null,
      diagonal: [],
      high_quality: false,
      title: "Matriz",
    };
    const scatter = pairplotFigure(plot).data[0] as { marker: { cmin?: number; color: unknown } };
    expect(scatter.marker.cmin).toBeUndefined();
    expect(scatter.marker.color).toEqual([1, null]);
  });
});
