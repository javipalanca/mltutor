<script lang="ts">
  import LessonTabs from "../LessonTabs.svelte";

  let { tabs }: { tabs: readonly string[] } = $props();
  let active = $state(0);
</script>

<LessonTabs {tabs} {active} onchange={(i) => (active = i)} idPrefix="t" label="Pestañas" />
<div role="tabpanel" id="t-panel" aria-labelledby="t-tab-{active}">contenido {active}</div>
