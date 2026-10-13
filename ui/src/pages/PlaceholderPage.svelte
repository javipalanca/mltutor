<script lang="ts">
  import type { RouteDef } from "../lib/routes";

  interface Props {
    route: RouteDef;
  }

  let { route }: Props = $props();
</script>

<article class="page" data-route-page={route.id}>
  <h1>{route.title}</h1>
  <p class="subtitle">{route.subtitle}</p>
  <div class="info-box" role="note">
    🚧 Esta sección se está portando a la nueva versión de MLTutor.
  </div>
</article>

<style>
  .page {
    max-width: 1100px;
    padding: 32px 40px;
  }

  h1 {
    color: var(--color-title);
    font-size: 30px;
    font-weight: 700;
  }

  .subtitle {
    margin: 10px 0 24px;
    color: var(--color-heading);
    font-size: 19px;
    font-weight: 600;
  }

  .info-box {
    padding: 14px 18px;
    background: var(--color-info-bg);
    border: 1px solid var(--color-info-border);
    border-radius: var(--radius-box);
  }
</style>
