<script lang="ts">
  import Card from "../components/Card.svelte";
  import Markdown from "../components/Markdown.svelte";
  import { HOME_CARDS, md, type LessonId } from "../lib/content";

  interface Props {
    onnavigate: (id: LessonId) => void;
  }

  let { onnavigate }: Props = $props();

  function split(path: string): { title: string; body: string } {
    const text = md(path).replace(/<!--[\s\S]*?-->\s*/g, "");
    const [first = "", ...rest] = text.split("\n");
    return { title: first.replace(/^#+\s*/, ""), body: rest.join("\n") };
  }

  const welcome = split("inicio/bienvenida.md");
  const whatIs = split("inicio/que-es-ml.md");
  const algorithms = split("inicio/algoritmos.md");
  const howTo = split("inicio/como-usar.md");
  // "## 📝 Cómo usar MLTutor" + "### Guía de uso" + list: the card holds the last two.
  const guideBody = howTo.body.replace(/^\s*###\s*(.+)\n/, "");
  const guideTitle = /###\s*(.+)/.exec(howTo.body)?.[1] ?? "Guía de uso";
</script>

<!-- Card title: #0288D1 in Python (3.4:1 on #E1F5FE); #0270B5 reaches WCAG AA (4.7:1). -->
<article class="home" data-route-page="inicio">
  <h1 data-page-title>{welcome.title}</h1>
  <div class="lead"><Markdown source={welcome.body} /></div>

  <Card background="#E1F5FE" titleColor="#0270B5" title={whatIs.title}>
    <Markdown source={whatIs.body} />
  </Card>

  <div class="section"><Markdown source={`## ${algorithms.title}\n${algorithms.body}`} /></div>
  <div class="cards">
    {#each HOME_CARDS as card (card.route)}
      <div class="card-column">
        <Card background={card.background} border={card.border} titleColor={card.title_color} title={card.title} height={200}>
          <p class="card-text">{card.text}</p>
          {#if card.status}
            <p class="status"><strong>Estado:</strong> <span>{card.status}</span></p>
          {/if}
        </Card>
        <button type="button" class="btn block" onclick={() => onnavigate(card.route)}>{card.button}</button>
      </div>
    {/each}
  </div>

  <div class="section"><Markdown source={`## ${howTo.title}`} /></div>
  <Card background="#F5F5F5" titleColor="#0d47a1" title={guideTitle}>
    <Markdown source={guideBody} />
  </Card>
</article>

<style>
  h1 {
    margin-top: 8px;
    color: var(--color-heading);
    font-size: 23px;
  }

  .lead {
    margin: 18px 6px 22px;
  }

  .lead :global(h3) {
    font-size: 17px;
  }

  .section {
    margin: 26px 4px 10px;
  }

  .cards {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 22px;
    margin: 14px 0 10px;
  }

  /* 1024 px windows: 2 × 2 cards instead of four narrow columns. */
  @container page (max-width: 820px) {
    .cards {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  .card-column {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .card-text {
    margin: 0 0 12px;
    line-height: 1.7;
  }

  .status {
    margin: 0;
  }

  .status span {
    color: #2e7d32;
  }

  .home :global(.card .md li) {
    line-height: 1.6;
  }
</style>
