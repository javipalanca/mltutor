<script lang="ts">
  import Expander from "../components/Expander.svelte";
  import LessonNavbar from "../components/LessonNavbar.svelte";
  import LessonTabs from "../components/LessonTabs.svelte";
  import Markdown from "../components/Markdown.svelte";
  import Notice from "../components/Notice.svelte";
  import { md, type LessonId } from "../lib/content";
  import { dataBlock, lesson as lessonContent, LESSON_PHASE, navbarMessages, tabName } from "../lib/lessons";
  import { session } from "../lib/session.svelte";
  import DatasetTab from "./DatasetTab.svelte";

  interface Props {
    id: LessonId;
    onscrolltop: () => void;
  }

  let { id, onscrolltop }: Props = $props();

  const content = $derived(lessonContent(id));
  const active = $derived(session.activeTab[id]);
  const block = $derived(dataBlock(id, active));
  const navbar = $derived(navbarMessages(id, active));
  const tab = $derived(content.tabs[active] ?? "");

  function go(index: number): void {
    const next = Math.max(0, Math.min(content.tabs.length - 1, index));
    if (next !== session.activeTab[id]) {
      session.activeTab[id] = next;
      onscrolltop();
    }
  }
</script>

<article class="lesson" data-route-page={id}>
  <h1 data-page-title>{content.title}</h1>
  <p class="subtitle">{content.subtitle}</p>
  <Expander id="intro-{id}" label={content.intro_expander}>
    <Markdown source={md(`${id}/intro.md`)} />
  </Expander>

  <LessonTabs tabs={content.tabs} {active} onchange={go} />

  <div class="tab-body" data-lesson-tab={active}>
    <DatasetTab lesson={id} {block} />
    {#if active > 0}
      <h2>{tab}</h2>
      <Notice
        severity="info"
        markdown={`🚧 La pestaña **${tabName(tab)}** de esta lección se está portando a la nueva versión de MLTutor (${LESSON_PHASE[id]} del plan de migración).`}
      />
    {/if}
  </div>

  <LessonNavbar
    prev={navbar.prev}
    next={navbar.next}
    prevNote={navbar.prevNote}
    nextNote={navbar.nextNote}
    disabledPrev={active === 0}
    onprev={() => go(active - 1)}
    onnext={() => go(active + 1)}
  />
</article>

<style>
  h1 {
    margin-top: 8px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .subtitle {
    margin: 18px 4px 22px;
  }

  h2 {
    margin: 24px 0 12px;
    color: var(--color-heading);
    font-size: 23px;
  }
</style>
