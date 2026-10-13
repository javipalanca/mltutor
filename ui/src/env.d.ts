/// <reference types="svelte" />
/// <reference types="vite/client" />

declare module "*.toml" {
  const data: Record<string, unknown>;
  export default data;
}

declare module "plotly.js-strict-dist-min" {
  export * from "plotly.js";
  import Plotly from "plotly.js";
  export default Plotly;
}
