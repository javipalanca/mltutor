<script lang="ts">
  import Markdown from "./Markdown.svelte";

  export type Severity = "info" | "warning" | "error" | "success";

  interface Props {
    severity?: Severity;
    /** Body in Markdown (`st.info("💡 **…**")`). */
    markdown: string;
  }

  let { severity = "info", markdown }: Props = $props();
</script>

<div class="notice {severity}" role={severity === "error" ? "alert" : "status"}>
  <Markdown source={markdown} />
</div>

<style>
  .notice {
    margin: 10px 0;
    padding: 14px 16px;
    border-radius: var(--radius-notice);
    border: 1px solid transparent;
    user-select: text;
  }

  .info {
    background: var(--color-info-bg);
    border-color: var(--color-info-border);
  }

  .warning {
    background: var(--color-warning-bg);
    border-color: var(--color-warning-border);
  }

  .error {
    background: var(--color-error-bg);
    border-color: var(--color-error-border);
  }

  .success {
    background: var(--color-success-bg);
    border-color: var(--color-success-border);
  }
</style>
