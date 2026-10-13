<script lang="ts">
  import Expander from "../Expander.svelte";

  let { id, label }: { id: string; label: string } = $props();
</script>

<Expander {id} {label}><p>cuerpo</p></Expander>
