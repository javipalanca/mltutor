<script lang="ts">
  import { md } from "../lib/content";

  // content/inicio/cabecera.md: "# 🎓 MLTutor" + subtitle line.
  const lines = md("inicio/cabecera.md")
    .replace(/<!--[\s\S]*?-->/g, "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const title = (lines[0] ?? "# 🎓 MLTutor").replace(/^#\s*/, "");
  const subtitle = lines[1] ?? "";
</script>

<header class="app-header">
  <p class="main-header">{title}</p>
  <p class="sub-header">{subtitle}</p>
</header>

<style>
  .app-header {
    margin-bottom: 18px;
  }

  .main-header {
    margin: 0;
    color: var(--color-title);
    font-size: 32px;
    font-weight: 700;
    text-align: center;
  }

  .sub-header {
    margin: 26px 0 0;
    color: var(--color-heading);
    font-size: 20px;
  }
</style>
