<script lang="ts">
  import { onMount } from "svelte";
  import Footer from "./components/Footer.svelte";
  import PageHeader from "./components/PageHeader.svelte";
  import Sidebar from "./components/Sidebar.svelte";
  import StatusBar from "./components/StatusBar.svelte";
  import CsvPage from "./pages/CsvPage.svelte";
  import HomePage from "./pages/HomePage.svelte";
  import LessonPage from "./pages/LessonPage.svelte";
  import { router } from "./lib/router.svelte";
  import { isLessonId } from "./lib/routes";
  import { startup } from "./lib/startup";

  let content: HTMLElement | undefined = $state();

  function scrollTop(): void {
    content?.scrollTo({ top: 0 });
  }

  // The scroll goes back to the top when the page changes (Qt `present()`).
  $effect(() => {
    void router.current;
    scrollTop();
  });

  onMount(() => {
    void startup();
  });
</script>

<div class="shell">
  <Sidebar current={router.current} onnavigate={(id) => router.go(id)} />
  <main class="content" bind:this={content} data-content>
    <div class="page">
      <PageHeader />
      {#key router.current}
        {#if router.current === "inicio"}
          <HomePage onnavigate={(id) => router.go(id)} />
        {:else if router.current === "csv"}
          <CsvPage />
        {:else if isLessonId(router.current)}
          <LessonPage id={router.current} onscrolltop={scrollTop} />
        {/if}
      {/key}
      <Footer />
    </div>
  </main>
  <div class="status"><StatusBar /></div>
</div>

<style>
  .shell {
    display: grid;
    grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
    grid-template-rows: minmax(0, 1fr) auto;
    height: 100%;
  }

  .content {
    overflow-y: auto;
    background: var(--color-bg);
    scrollbar-gutter: stable;
  }

  .page {
    max-width: var(--content-max-width);
    margin: 0 auto;
    padding: 26px 36px 24px;
  }

  .status {
    grid-column: 1 / -1;
  }
</style>
