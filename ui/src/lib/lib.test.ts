import { describe, expect, it } from "vitest";
import type { PairplotData } from "./bindings/PairplotData";
import {
  correlationFigure,
  levelColor,
  pairplotFigure,
  pairplotHeight,
  TAB10,
  targetDistributionFigure,
  withAlpha,
} from "./charts";
import { csvText, datosText, footerLines, headingText, md, NAVIGATION } from "./content";
import { fmt } from "./fmt";
import { formatG } from "./format";
import { dataBlock, navbarMessages, tabName } from "./lessons";
import { renderMarkdown, stripComments } from "./markdown";
import { pairplotDefaults, pairplotNotices } from "./pairplot";
import { routeById } from "./routes";
import { cellText, compareCells, matrixCells, toTsv } from "./table";

describe("fmt (Python str.format)", () => {
  it("formats placeholders like the content templates", () => {
    expect(fmt("{ratio:.1f}x", { ratio: 2.147 })).toBe("2.1x");
    expect(fmt("~{t:.1f}s", { t: 0.25 })).toBe("~0.2s");
    expect(fmt("{n:,} filas", { n: 20640 })).toBe("20,640 filas");
    expect(fmt("• {valor}: {n} ({pct:.1f}%)", { valor: "2", n: 156, pct: 63.934 })).toBe("• 2: 156 (63.9%)");
    expect(fmt("{a} y {b}", { a: "x" })).toBe("x y {b}");
    expect(fmt("• Media: {v:.4f}", { v: 2.998279 })).toBe("• Media: 2.9983");
  });
});

describe("formatG (.6g)", () => {
  it("matches Python's general format", () => {
    expect(formatG(5.843333333)).toBe("5.84333");
    expect(formatG(150)).toBe("150");
    expect(formatG(0.828066)).toBe("0.828066");
    expect(formatG(1234567)).toBe("1.23457e+06");
    expect(formatG(0.00001234)).toBe("1.234e-05");
    expect(formatG(-0.5)).toBe("-0.5");
    expect(formatG(0)).toBe("0");
    expect(formatG(Number.NaN)).toBe("nan");
    expect(formatG(100000)).toBe("100000");
  });
});

describe("markdown", () => {
  it("drops source comments, renders lists, bold and KaTeX", () => {
    const html = renderMarkdown("<!-- origen -->\n**Hola** $x^2$\n\n- a\n- b\n\n$$\\frac{1}{2}$$");
    expect(html).not.toContain("origen");
    expect(html).toContain("<strong>Hola</strong>");
    expect(html).toContain("katex");
    expect(html).toContain("<li>a</li>");
    expect(html).toContain("math-block");
  });

  it("does not take money for math and highlights Python", () => {
    expect(renderMarkdown("Cuesta $5 y $6")).not.toContain("katex");
    expect(renderMarkdown("```python\nimport numpy as np\n```")).toContain("hljs-keyword");
    expect(stripComments("a<!-- b -->c")).toBe("ac");
  });
});

describe("content", () => {
  it("loads the texts of content/", () => {
    expect(md("datos/correlacion.md")).toContain("**Correlación**");
    expect(datosText("pairplot", "boton_generar")).toBe("🚀 Generar Matriz de Dispersión");
    expect(csvText("selector", "vacio")).toBe("❌ El archivo está vacío");
    expect(headingText("### 📊 Información del Dataset")).toBe("📊 Información del Dataset");
    expect(footerLines()).toEqual([
      "Desarrollado por Javi Palanca",
      "Departament de Sistemes Informàtics i Computació",
      "Universitat Politècnica de València",
    ]);
  });

  it("uses the lesson titles and subtitles of navegacion.toml", () => {
    expect(routeById("regresion").title).toBe("📊 Regresión");
    expect(routeById("regresion").subtitle).toBe(NAVIGATION.lessons.regresion.subtitle);
    expect(routeById("knn").title).toBe("🔍 K-Nearest Neighbors (KNN)");
  });
});

describe("lessons", () => {
  it("builds the navbar messages of apps/navbar.py", () => {
    expect(tabName("🏋️ Entrenamiento")).toBe("Entrenamiento");
    expect(navbarMessages("arboles", 0)).toEqual({ next: "Continuar a Entrenamiento" });
    expect(navbarMessages("regresion", 3)).toEqual({ prev: "Volver a Evaluación", next: "Continuar a Coeficientes" });
    expect(navbarMessages("knn", 6)).toEqual({ prev: "Volver a Predicciones" });
    expect(navbarMessages("redes", 1).nextNote).toContain("¿Listo para entrenar?");
    expect(navbarMessages("redes", 2)).toEqual({ prev: "Volver a Arquitectura" });
  });

  it("shows the data block of §2.1", () => {
    expect(dataBlock("knn", 0)).toBe("full");
    expect(dataBlock("knn", 1)).toBe("selector");
    expect(dataBlock("redes", 1)).toBe("current");
    expect(dataBlock("arboles", 4)).toBe("current");
  });
});

describe("pairplot options", () => {
  it("follows the slider and warning rules", () => {
    expect(pairplotDefaults(["a", "b"])).toEqual({ n: 2, slider: null, defaultMax: 2 });
    expect(pairplotDefaults(["a", "b", "c", "d", "e", "f", "g", "h"])).toEqual({
      n: 8,
      slider: { min: 2, max: 6 },
      defaultMax: 4,
    });
    expect(pairplotNotices(4, 150)).toEqual([]);
    expect(pairplotNotices(5, 1000)[0]).toMatchObject({ key: "nota", values: { t: 0.5 } });
    expect(pairplotNotices(6, 1000)[0]).toMatchObject({ key: "cuidado", severity: "warning" });
  });
});

describe("table helpers", () => {
  it("formats, sorts and copies cells", () => {
    expect(cellText({ kind: "number", v: 3.14159265 })).toBe("3.14159");
    expect(cellText({ kind: "bool", v: true })).toBe("True");
    expect(compareCells({ kind: "number", v: 2 }, { kind: "number", v: 10 })).toBeLessThan(0);
    expect(compareCells({ kind: "missing" }, { kind: "text", v: "a" })).toBeGreaterThan(0);
    expect(toTsv(["a", "b"], [[{ kind: "number", v: 0.123456789 }, { kind: "text", v: "x" }]], ["0"])).toBe(
      "\ta\tb\n0\t0.123456789\tx",
    );
    expect(matrixCells([1, null, 3, 4], 2, 2)[0]).toEqual([{ kind: "number", v: 1 }, { kind: "missing" }]);
  });
});

describe("charts", () => {
  it("uses the Python colours", () => {
    expect(levelColor(1, 3)).toBe(TAB10[1]);
    expect(levelColor(11, 12)).toMatch(/^hsl/);
    expect(withAlpha("#1f77b4", 0.5)).toBe("rgba(31, 119, 180, 0.5)");
    const bars = targetDistributionFigure({
      kind: "bar",
      labels: ["a", "b", "c", "d"],
      counts: [1, 2, 3, 4],
      balance_ratio: 4,
      balanced: false,
    });
    expect(bars.layout.xaxis?.tickangle).toBe(-45);
    expect(bars.layout.title).toMatchObject({ text: "Distribución de Clases" });
  });

  it("masks the upper triangle and the diagonal of the correlation (np.triu)", () => {
    const figure = correlationFigure({
      labels: ["a", "b", "target"],
      values: { rows: 3, cols: 3, values: [1, 0.5, -0.2, 0.5, 1, null, -0.2, null, 1] },
      visible_min: -0.2,
      visible_max: 0.5,
      truncated: false,
      total_columns: 3,
      truncation_message: null,
    });
    const trace = figure.data[0] as unknown as { z: (number | null)[][]; zmin: number; zmax: number };
    expect(trace.z).toEqual([
      [null, null, null],
      [0.5, null, null],
      [-0.2, null, null],
    ]);
    expect([trace.zmin, trace.zmax]).toEqual([-1, 1]);
  });

  it("draws the pairplot with WebGL scatter and one diagonal per feature", () => {
    const plot: PairplotData = {
      features: ["x", "y"],
      n_points: 3,
      n_total: 3,
      sampled: false,
      points: { rows: 3, cols: 2, values: [1, 2, 3, 4, 5, 6] },
      hue: { kind: "categorical", levels: ["p", "q"], codes: [0, 1, 0] },
      tooltips: null,
      diagonal: [
        { kind: "kde", curves: [{ level: "p", n: 2, bandwidth: 1, grid_min: 0, grid_max: 1, density: [0, 1, 0] }] },
        { kind: "hist", edges: [0, 1, 2], series: [{ level: "p", counts: [1, 1] }, { level: "q", counts: [0, 1] }] },
      ],
      high_quality: false,
      title: "Matriz de Dispersión de Características",
    };
    const figure = pairplotFigure(plot);
    const types = figure.data.map((trace) => (trace as { type?: string }).type);
    expect(types.filter((t) => t === "scattergl")).toHaveLength(4); // 2 panels × 2 levels
    expect(types.filter((t) => t === "bar")).toHaveLength(2);
    expect(figure.layout.showlegend).toBe(true);
    expect((figure.layout as Record<string, unknown>)["xaxis4"]).toBeDefined();
    expect(pairplotHeight(6)).toBeGreaterThan(pairplotHeight(2));
  });

  it("uses a continuous colour scale for regression (D11)", () => {
    const figure = pairplotFigure({
      features: ["x", "y"],
      n_points: 2,
      n_total: 2,
      sampled: false,
      points: { rows: 2, cols: 2, values: [1, 2, 3, 4] },
      hue: { kind: "continuous", values: [10, 20], min: 10, max: 20 },
      tooltips: ["a", "b"],
      diagonal: [],
      high_quality: true,
      title: "t",
    });
    const scatter = figure.data.filter((trace) => (trace as { type?: string }).type === "scattergl");
    expect(scatter).toHaveLength(2);
    expect((scatter[0] as { marker: { color: number[] } }).marker.color).toEqual([10, 20]);
    expect(figure.layout.showlegend).toBe(false);
  });
});

describe("toFixedPython", () => {
  it("rounds exact ties to even like Python", async () => {
    const { toFixedPython } = await import("./fmt");
    expect(toFixedPython(0.25, 1)).toBe("0.2");
    expect(toFixedPython(0.35, 1)).toBe("0.3"); // 0.35 is 0.34999… in binary
    expect(toFixedPython(2.5, 0)).toBe("2");
    expect(toFixedPython(3.5, 0)).toBe("4");
    expect(toFixedPython(1.005, 2)).toBe("1.00");
    expect(toFixedPython(-0.125, 2)).toBe("-0.12");
  });
});

describe("pairplot without WebGL", () => {
  it("falls back to SVG scatter", () => {
    const figure = pairplotFigure(
      {
        features: ["x", "y"],
        n_points: 1,
        n_total: 1,
        sampled: false,
        points: { rows: 1, cols: 2, values: [1, 2] },
        hue: { kind: "categorical", levels: ["p"], codes: [0] },
        tooltips: null,
        diagonal: [],
        high_quality: false,
        title: "t",
      },
      undefined,
      false,
    );
    expect(figure.data.every((trace) => (trace as { type?: string }).type !== "scattergl")).toBe(true);
  });
});
