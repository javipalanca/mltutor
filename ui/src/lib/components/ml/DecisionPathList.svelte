<script lang="ts">
  /**
   * C16: "### Camino de decisión" of 🔮 Predicciones (`show_prediction_path`): one line per
   * internal node ("- petal length (cm) = 4.7000 > 2.4500 ✅"), plus D46: the label of the
   * categorical values ("Sexo = 0 (Mujer)") and a last line with the leaf it reaches.
   */
  import Markdown from "../../../components/Markdown.svelte";
  import type { DecisionPath } from "../../bindings/DecisionPath";
  import { arbText } from "../../content";
  import { escapeMarkdown } from "../../escape";
  import { fmt } from "../../fmt";

  interface Props {
    path: DecisionPath;
    /** Class names of the dataset (indexed by code); `null` for regressors. */
    classNames: readonly string[] | null;
    /** The tree uses medians in its leaves (`absolute_error`). */
    median?: boolean;
  }

  let { path, classNames, median = false }: Props = $props();
  const t = (key: string): string => arbText("predicciones", key);

  const lines = $derived.by(() => {
    const out = path.steps.map((step) => {
      const template = t(step.goes_left ? "camino_menor_igual" : "camino_mayor");
      if (step.display !== null) {
        // D46: the code and its label instead of "0.0000".
        return fmt(template.replace("{valor:.4f}", "{valor_cat}"), {
          nombre: step.name,
          valor_cat: step.display,
          umbral: step.threshold,
        });
      }
      return fmt(template, { nombre: step.name, valor: step.value, umbral: step.threshold });
    });
    if (path.leaf_class !== null) {
      const index = path.leaf_value.reduce((best, v, i, all) => (v > (all[best] ?? -1) ? i : best), 0);
      out.push(
        fmt(t("camino_hoja_clasificacion"), {
          nodo: path.leaf,
          clase: classNames?.[path.leaf_class] ?? String(path.leaf_class),
          probabilidad: path.leaf_value[index] ?? 0,
          muestras: path.leaf_samples,
        }),
      );
    } else {
      out.push(
        fmt(t(median ? "camino_hoja_regresion_mediana" : "camino_hoja_regresion"), {
          nodo: path.leaf,
          valor: path.leaf_value[0] ?? Number.NaN,
          muestras: path.leaf_samples,
        }),
      );
    }
    return out.join("\n");
  });
</script>

<div class="path" data-decision-path={path.steps.length}>
  <Markdown source={lines || escapeMarkdown("")} />
</div>

<style>
  .path {
    margin: 6px 0 14px;
    user-select: text;
  }
</style>
