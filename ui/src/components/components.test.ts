import { fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup } from "@testing-library/svelte";
import { session } from "../lib/session.svelte";
import ButtonPanel from "./ButtonPanel.svelte";
import DataTable from "./DataTable.svelte";
import FilePicker from "./FilePicker.svelte";
import LessonNavbar from "./LessonNavbar.svelte";
import LessonTabs from "./LessonTabs.svelte";
import Metric from "./Metric.svelte";
import MultiSelect from "./MultiSelect.svelte";
import Notice from "./Notice.svelte";
import Radio from "./Radio.svelte";
import Select from "./Select.svelte";
import Slider from "./Slider.svelte";
import CodeBlock from "./CodeBlock.svelte";
import ExpanderHarness from "./test/ExpanderHarness.svelte";

afterEach(() => cleanup());

describe("Notice", () => {
  it("renders Markdown with the severity style", () => {
    const { container } = render(Notice, { severity: "warning", markdown: "⚠️ **Cuidado**: lento" });
    const box = container.querySelector(".notice");
    expect(box?.classList.contains("warning")).toBe(true);
    expect(box?.querySelector("strong")?.textContent).toBe("Cuidado");
  });
});

describe("Metric", () => {
  it("shows label, value and help", () => {
    render(Metric, { label: "📏 Filas", value: 150, help: "Número total" });
    expect(screen.getByText("150")).toBeTruthy();
    expect(screen.getByText("📏 Filas").closest(".metric")?.getAttribute("title")).toBe("Número total");
  });
});

describe("Expander", () => {
  it("toggles and remembers its state by id", async () => {
    render(ExpanderHarness, { id: "test-expander", label: "💡 ¿Qué significan?" });
    const button = screen.getByRole("button", { name: /Qué significan/ });
    expect(screen.queryByText("cuerpo")).toBeNull();
    await fireEvent.click(button);
    expect(screen.getByText("cuerpo")).toBeTruthy();
    expect(session.expander("test-expander", false)).toBe(true);
  });
});

describe("ButtonPanel and LessonTabs", () => {
  it("marks the active option as primary and reports clicks", async () => {
    const onchange = vi.fn();
    render(ButtonPanel, {
      options: [
        { label: "🔗 Matriz de Correlación", value: "correlation" },
        { label: "📈 Pairplot", value: "pairplot" },
      ],
      value: "pairplot",
      onchange,
    });
    expect(screen.getByRole("button", { name: "📈 Pairplot" }).classList.contains("primary")).toBe(true);
    await fireEvent.click(screen.getByRole("button", { name: "🔗 Matriz de Correlación" }));
    expect(onchange).toHaveBeenCalledWith("correlation");
  });

  it("has seven tabs with arrow-key navigation", async () => {
    const onchange = vi.fn();
    const tabs = ["📊 Datos", "🏋️ Entrenamiento", "📈 Evaluación", "📉 Visualización", "🔍 Coeficientes", "🔮 Predicciones", "💾 Exportar"];
    render(LessonTabs, { tabs, active: 0, onchange, idPrefix: "lesson-test" });
    expect(screen.getAllByRole("tab")).toHaveLength(7);
    expect(screen.getByRole("tab", { name: "📊 Datos" }).getAttribute("aria-selected")).toBe("true");
    await fireEvent.keyDown(screen.getByRole("tablist"), { key: "ArrowLeft" });
    expect(onchange).toHaveBeenCalledWith(6);
  });
});

describe("LessonNavbar", () => {
  it("shows 🔙/🔜 buttons and notes", async () => {
    const onnext = vi.fn();
    render(LessonNavbar, { prev: "Volver a Datos", next: "Continuar a Evaluación", nextNote: "**nota**", onnext });
    expect(screen.getByText("🧭 Navegación")).toBeTruthy();
    expect(screen.getByRole("button", { name: "🔙 Volver a Datos" })).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "🔜 Continuar a Evaluación" }));
    expect(onnext).toHaveBeenCalled();
  });
});

describe("DataTable", () => {
  const rows = [
    [{ kind: "number", v: 5.1 }, { kind: "text", v: "setosa" }],
    [{ kind: "number", v: 4.9 }, { kind: "text", v: "virginica" }],
    [{ kind: "number", v: 6.25 }, { kind: "missing" }],
  ] as const;

  it("formats numbers, shows the index and the caption", () => {
    const { container } = render(DataTable, { columns: ["sepal", "target"], rows: rows.map((r) => [...r]), index: ["0", "1", "2"] });
    expect(container.querySelectorAll("tbody tr")).toHaveLength(3);
    expect(container.querySelector("td.number")?.textContent?.trim()).toBe("5.1");
    expect(screen.getByText(/3 filas · 2 columnas/)).toBeTruthy();
  });

  it("sorts by a column when sortable", async () => {
    const { container } = render(DataTable, { columns: ["sepal", "target"], rows: rows.map((r) => [...r]), sortable: true });
    await fireEvent.click(screen.getByText("sepal"));
    const first = () => container.querySelector("tbody tr td")?.textContent?.trim();
    expect(first()).toBe("4.9");
    await fireEvent.click(screen.getByText("sepal"));
    expect(first()).toBe("6.25");
  });
});

describe("Form controls", () => {
  it("MultiSelect respects max_selections and shows the summary", async () => {
    const onchange = vi.fn();
    render(MultiSelect, { label: "Características:", options: ["a", "b", "c"], value: ["a", "b"], max: 2, onchange });
    expect(screen.getByText(/2 de 3 seleccionadas · Máximo 2/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "c" }) as HTMLButtonElement).disabled).toBe(true);
    await fireEvent.click(screen.getByRole("button", { name: /a/ }));
    expect(onchange).toHaveBeenCalledWith(["b"]);
  });

  it("Select emits the chosen value", async () => {
    const onchange = vi.fn();
    render(Select, {
      id: "s",
      label: "Dataset:",
      options: [
        { label: "🌸 Iris", value: { kind: "builtin", id: "iris" } },
        { label: "🍷 Vino", value: { kind: "builtin", id: "wine" } },
      ],
      value: { kind: "builtin", id: "iris" },
      onchange,
    });
    await fireEvent.change(screen.getByLabelText("Dataset:"), { target: { value: "1" } });
    expect(onchange).toHaveBeenCalledWith({ kind: "builtin", id: "wine" });
  });

  it("Radio and Slider emit on change", async () => {
    const onradio = vi.fn();
    render(Radio, {
      name: "diag",
      label: "Tipo de gráfico en la diagonal:",
      horizontal: true,
      options: [
        { label: "Histograma", value: "hist" },
        { label: "KDE", value: "kde" },
      ],
      value: "kde",
      onchange: onradio,
    });
    await fireEvent.click(screen.getByLabelText("Histograma"));
    expect(onradio).toHaveBeenCalledWith("hist");

    const onslider = vi.fn();
    render(Slider, { id: "max", label: "Número máximo:", min: 2, max: 6, value: 4, onchange: onslider });
    await fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "9" } });
    expect(onslider).toHaveBeenCalledWith(6);
  });

  it("FilePicker shows the file name and the clear button", async () => {
    const onclear = vi.fn();
    render(FilePicker, {
      label: "Selecciona un archivo CSV:",
      fileName: "tips.csv",
      buttonLabel: "Seleccionar archivo CSV…",
      clearLabel: "Quitar archivo",
      onpick: vi.fn(),
      onclear,
    });
    expect(screen.getByRole("button", { name: "📄 tips.csv" })).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Quitar archivo" }));
    expect(onclear).toHaveBeenCalled();
  });
});

describe("CodeBlock", () => {
  it("highlights Python and offers copy and download", () => {
    const { container } = render(CodeBlock, { code: "import pandas as pd\nprint(1)", title: "Código", downloadName: "a.py" });
    expect(screen.getByText("Código")).toBeTruthy();
    expect(screen.getByRole("button", { name: /Ver código/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: "📥 Descargar código" })).toBeTruthy();
    expect(container.querySelector(".code-panel")).toBeNull();
  });
});
