<script lang="ts">
  /**
   * C9 (`ui.py::create_prediction_interface`): one control per feature built in Rust
   * from X_train and the dataset metadata (checkbox, select with the metadata labels or
   * slider from min to max with the mean as default), in two columns. Emits the codes the
   * model receives (e.g. "Mujer" → 0). Shared by the four lessons.
   */
  import type { FormControl } from "../../bindings/FormControl";
  import Checkbox from "../../../components/Checkbox.svelte";
  import Markdown from "../../../components/Markdown.svelte";
  import Select from "../../../components/Select.svelte";
  import Slider from "../../../components/Slider.svelte";
  import { compText } from "../../content";

  interface Props {
    controls: readonly FormControl[];
    /** One code per control, in column order. */
    values: readonly number[];
    onchange: (values: number[]) => void;
    idPrefix: string;
  }

  let { controls, values, onchange, idPrefix }: Props = $props();

  const half = $derived(Math.ceil(controls.length / 2));
  const columns = $derived([controls.slice(0, half), controls.slice(half)]);

  function set(index: number, value: number): void {
    const next = [...values];
    next[index] = value;
    onchange(next);
  }

  /** Values used by the slider (more decimals than its step, as Qt's spin boxes). */
  function decimals(step: number): number {
    return Math.min(6, Math.max(2, Math.ceil(-Math.log10(Math.abs(step) || 1)) + 2));
  }
</script>

<Markdown source={compText("prediccion", "titulo")} />
<div class="columns" data-prediction-form>
  {#each columns as column, c (c)}
    <div class="column">
      {#each column as control (control.index)}
        {@const value = values[control.index] ?? 0}
        {@const id = `${idPrefix}-feature-${control.index}`}
        {@const strong = control.description !== null}
        <div class="control">
          {#if control.control.kind === "checkbox"}
            <Checkbox {id} label={control.label} checked={value !== 0} onchange={(checked) => set(control.index, checked ? 1 : 0)} />
            {#if control.description}<p class="description">{control.description}</p>{/if}
          {:else if control.control.kind === "select"}
            <Select
              {id}
              {strong}
              hint={control.description ?? undefined}
              label={control.label}
              options={control.control.options.map((o) => ({ label: o.label, value: o.code }))}
              {value}
              onchange={(code) => set(control.index, code)}
            />
          {:else}
            {@const slider = control.control}
            <Slider
              {id}
              {strong}
              hint={control.description ?? undefined}
              label={control.label}
              min={slider.min}
              max={slider.max}
              step={slider.step > 0 ? slider.step : 0.1}
              value={Number(value.toFixed(decimals(slider.step)))}
              onchange={(v) => set(control.index, v)}
            />
          {/if}
        </div>
      {/each}
    </div>
  {/each}
</div>

<style>
  .columns {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 28px;
    align-items: start;
  }

  .control {
    margin-bottom: 6px;
  }

  .description {
    margin: -6px 0 8px;
    color: var(--color-text-muted);
    font-size: 13px;
    font-style: italic;
  }
</style>
