import "./styles/fonts.css";
import "katex/dist/katex.min.css";
import "./styles/theme.css";
// Sets <html data-theme> before the first paint (dark tokens hang from it).
import "./lib/theme.svelte";

import { mount } from "svelte";
import App from "./App.svelte";
import { installMacTabbing } from "./lib/keyboard";

const target = document.getElementById("app");
if (!target) {
  throw new Error("No se encontró el contenedor #app");
}

const app = mount(App, { target });
installMacTabbing(target);

export default app;
