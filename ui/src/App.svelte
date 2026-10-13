<script lang="ts">
  import { onMount } from "svelte";
  import Sidebar from "./components/Sidebar.svelte";
  import PlaceholderPage from "./pages/PlaceholderPage.svelte";
  import { router } from "./lib/router.svelte";
  import { routeById } from "./lib/routes";
  import { startup } from "./lib/startup";

  const route = $derived(routeById(router.current));

  onMount(() => {
    void startup();
  });
</script>

<div class="shell">
  <Sidebar current={router.current} onnavigate={(id) => router.go(id)} />
  <main class="content">
    {#key route.id}
      <PlaceholderPage {route} />
    {/key}
  </main>
</div>

<style>
  .shell {
    display: grid;
    grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
    height: 100%;
  }

  .content {
    overflow-y: auto;
    background: var(--color-bg);
  }
</style>
