import "@fontsource-variable/inter/wght.css";
import "katex/dist/katex.min.css";
import "./styles/theme.css";

import { mount } from "svelte";
import App from "./App.svelte";

const target = document.getElementById("app");
if (!target) {
  throw new Error("No se encontró el contenedor #app");
}

export default mount(App, { target });
