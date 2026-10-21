# Comparación visual Qt ↔ Rust — Fase 7 (revisión pantalla por pantalla, accesibilidad, HiDPI y tamaños)

Fecha: 2026-10-20. Capturas Qt de referencia en [`qt/`](qt/) y Rust en [`rust/`](rust/), regeneradas todas
(219 escenarios) con un binario de depuración con el frontend embebido
(`npm --prefix ui run tauri build -- --debug --no-bundle`) y `RealFacade`:

```sh
target/debug/mltutor --screenshot rust/fase3/05_arbol_iris.png --scenario arboles_arbol_iris --full-page
# F7: modificadores de escenario (ui/src/lib/scenarioSpec.ts), combinables y antes de «:ruta.csv»
target/debug/mltutor --screenshot x.png --scenario knn_interactiva_iris@vista            # ventana normal, vista principal arriba
target/debug/mltutor --screenshot x.png --scenario knn_interactiva_iris@vista@1024x668   # app en una caja de 1024×668 (recortar el PNG)
target/debug/mltutor --screenshot x.png --scenario redes_pesos_iris@vista@oscuro         # tema oscuro forzado
```

Cada escenario se capturó cuatro veces: página entera (`--full-page`, las del repositorio), ventana normal
1400×888 con la vista principal arriba (`@vista`, para ver fallos que `--full-page` esconde, como el del
lienzo de F5c), 1024×668 (`@1024x668`, el interior de una ventana de 1024×700) y tema oscuro (`@oscuro`).
Una selección está en [`rust/fase7/`](rust/fase7/) (`vista_*`, `1024x700_*`, `1280x800_*`, `oscuro_*` y
`katex_*`). Las `*_ventana.png` de fases anteriores se han regenerado a tamaño de ventana. Todo se cuantizó a
256 colores con Pillow (las capturas Rust pasan de 31 a 40 MB: las de página entera regeneradas, 3 nuevas y las 24 de
`fase7/`).

## Fallos encontrados y corregidos

| Dónde | Fallo (antes) | Corrección |
|---|---|---|
| Formularios de 🔮 (todas las lecciones; Diabetes, Cáncer) | El campo numérico del deslizador (96 px + flechas de WebKit) cortaba «-0,00056…» | 112 px, sin flechas (las flechas del teclado siguen funcionando), cifras tabulares |
| Pestañas de lección en 1024 px | «Entrenamie», «Visualizacio», «Característic»: texto cortado por los dos lados (el `text-overflow` no actúa en `inline-flex`) | Con la página < 900 px el texto pasa a dos líneas con guiones (`hyphens: auto`, `lang="es"`): «Entrena-miento» |
| 🎮 Lienzo KNN en 1024 px | La fila de controles medía ≥ 860 px: «Limpiar Puntos» quedaba fuera y aparecía una barra horizontal en toda la página | `@container page (max-width: 900px)`: K, métrica y pesos en una fila; casillas y «Limpiar» debajo |
| 🏠 Inicio en 1024 px | Cuatro tarjetas de ~150 px, títulos partidos | 2 × 2 tarjetas con la página < 820 px |
| Barra lateral en 1024 px | 260 px fijos (25 % de la ventana); «Cargar CSV Person…» | `clamp(210px, 22cqw, 260px)` y el texto del botón puede ocupar dos líneas |
| Consultas `@media (max-width: …)` de 🧠 (pesos, activaciones, arquitectura, evaluación, exportar) | Medían la ventana, no el contenido, y nunca se activaban (la ventana mínima es de 1000 px) | `@container page`, con umbrales sobre el ancho de la página |
| Tema oscuro | Solo seguía al sistema: imposible capturarlo o probarlo sin cambiar macOS | `<html data-theme>` lo pone `theme.svelte.ts` antes del primer pintado (sigue al sistema en vivo); `@oscuro` lo fuerza |
| Lienzo del mapa de neuronas | Al pasar la ventana de una pantalla 1× a una 2× se quedaba con el mapa de bits de 1× (borroso) | `theme.dpr` escucha `(resolution: Ndppx)` y el mapa se redibuja |

## Accesibilidad

- **Teclado:** anillo de foco único (`outline` 2 px `--color-focus` con separación de 2 px) en todos los
  controles, también en los botones azules (antes era un borde azul sobre fondo azul) y en los campos
  (antes solo cambiaba el color del borde). Enlace «Saltar al contenido» al principio. Pestañas WAI-ARIA
  (`tablist`/`tab`/`tabpanel`, `aria-controls`/`aria-labelledby`, tabindex móvil, ← → Inicio Fin que
  mueven también el foco). Desplegables con `aria-controls` y **Escape** para cerrarlos desde dentro
  (el foco vuelve a la cabecera). Escape cierra los tooltips del árbol, del diagrama y del mapa de la red.
  Cabeceras ordenables de las tablas como botones. El lienzo KNN tiene alternativa de teclado
  («✏️ Añadir un punto con valores exactos») y lo dice su `aria-label`.
- **macOS:** WKWebView solo recorre con Tab campos de texto y desplegables si no está activada la
  «Navegación con teclado» del sistema. `lib/keyboard.ts` pone `tabindex="0"` a botones, enlaces,
  casillas, radios y deslizadores (solo en WKWebView); WebView2 y WebKitGTK ya lo hacen.
- **Etiquetas:** todos los campos tienen nombre accesible (las dos casillas del lienzo KNN no tenían; el
  `aria-labelledby` del multiselect usaba la etiqueta como id, con espacios y emojis, y no resolvía). Las
  pistas en cursiva de los formularios van en `aria-describedby`. `PlotlyChart` exige `label`
  (`aria-label` del `<figure>`; faltaba en las dos gráficas de la página CSV) y su botón «Guardar
  imagen…» dice de qué gráfico es. Mapa de neuronas y lienzo KNN con `role="img"`.
- **Regiones vivas:** la barra de estado anuncia solo su texto (`role="status"`), no el botón Cancelar;
  el entrenamiento de 🧠 anuncia el progreso cada 10 % (la línea de época cambia hasta 20 veces por
  segundo); el multiselect anuncia «n de m seleccionadas».
- **Movimiento:** `prefers-reduced-motion` ya quitaba transiciones y animaciones; la barra indeterminada
  de la barra de estado se queda llena y atenuada en lugar de inmóvil a la izquierda.
- **Contraste (WCAG 2.1 AA):** calculado para todos los pares de tokens de `theme.css` en los dos temas
  (test `a11y.test.ts`, que falla si alguno baja del mínimo). Cambios:

| Token / elemento | Antes | Ahora | Razón |
|---|---|---|---|
| `--color-primary` (botón activo, texto blanco) claro | #1e88e5 (3,68:1) | #1976d2 (4,60:1) | texto normal ≥ 4,5 |
| `--color-primary-hover` claro | #1572c4 | #1565c0 (5,75:1) | |
| `--color-primary` / hover oscuro | #2f8ee8 (3,41) / #4aa0f0 (2,77) | #1d6fc9 (5,04) / #1a65b8 (5,84) | |
| `--color-text-muted` y barra de estado (claro) | #64748b (4,17 sobre info, 4,39 sobre la barra lateral) | #5b6b80 (≥ 4,75) | |
| `--color-footer` | #888888 (3,54) | #707070 (4,95) | |
| `--color-focus` claro | #60a5fa (2,54) | #2563eb (5,17) | indicador de foco ≥ 3 |
| `--color-input-border` claro / oscuro | #d7e2f0 (1,25) / #2c3d57 (1,45) | #7f8fa4 (3,15) / #5b6e8a (3,08) | borde que identifica el campo ≥ 3 |
| `--chart-axis` claro | #8a99ad (2,90) | #7b8aa0 (3,51) | |
| Título «¿Qué es el Machine Learning?» | #0288D1 (3,43) | #0270B5 (4,69) | 17 px negrita no es «texto grande» |
| Título de la tarjeta «📊 Regresión» (`content/inicio/tarjetas.toml`) | #E65100 (3,46) | #C43E00 (4,74) | |
| «Limpiar Puntos» del lienzo KNN | #ff4444 (3,41) | #d32f2f (4,98) | |
| Leyenda «activas» del mapa (oscuro) | #1976d2 (3,68) | #64b5f6 (7,65) | |

  Los bordes de los botones (`--color-border`) no se han oscurecido: el texto identifica el botón
  (1.4.11 no lo exige) y mantienen el aspecto de Qt. Los colores de datos (tab10, paleta del lienzo KNN,
  colores de nodo del árbol) se conservan por paridad.

## HiDPI y tamaños de ventana

- **Tamaño mínimo:** `tauri.conf.json` tiene `minWidth: 1000, minHeight: 700` (y 1400×920 por defecto).
  Con los cambios de F7 la interfaz cabe sin barra horizontal desde 1000 px; recomendación para
  `src-tauri` (no tocado): **subir `minWidth` a 1024** para que coincida con lo probado, y
  `backgroundColor` `#ffffff` produce un destello blanco al abrir en tema oscuro (habría que quitarlo o
  ponerlo según el tema).
- **1024×700 y 1280×800:** revisados los 219 escenarios en 1024×668 y una muestra en 1280×768 (emulados con
  `@WxH`: la app se dibuja en una caja de ese tamaño; los `@container`, `cqw` y los `ResizeObserver` de
  Plotly responden a la caja igual que a la ventana). **1920×1080:** no se puede emular (la ventana de
  captura mide 1400); el contenido tiene `max-width: 1180px` centrado, así que solo crecen los márgenes.
- **2× DPR:** las dos pantallas de este Mac son de 1×, así que no hay captura a 2×. Revisión de código:
  Plotly es SVG (nítido a cualquier escala) y scattergl usa `devicePixelRatio`; el mapa de neuronas
  dibuja a `min(3, dpr)` y ahora se redibuja al cambiar de pantalla; el PNG del árbol sale a 3× y las
  imágenes de Plotly a 2×. **Pendiente:** comprobarlo en una pantalla Retina.

## Tamaño (petición del coordinador)

- KaTeX: un plugin de Vite (`katexWoff2Only` en `ui/vite.config.ts`) quita los `.woff`/`.ttf` de
  `katex.min.css`: 40 archivos menos, 817 KB de recursos (~480 KB en el binario comprimido). Las tres
  WebViews leen woff2. Comprobado en [`rust/fase7/katex_coeficientes_logistica_iris.png`](rust/fase7/katex_coeficientes_logistica_iris.png)
  (fracciones, sumatorio, paréntesis grandes con KaTeX_Size*).
- Inter: `ui/src/styles/fonts.css` declara solo los subconjuntos `latin` y `latin-ext` (85 KB menos).
  Las letras griegas de algunos textos (β, σ) usan la fuente del sistema.

## Pantalla por pantalla

Estado: **OK** = mismo contenido, orden, emojis y colores que Qt (las diferencias deliberadas de fases
anteriores están en `comparacion-fase1.md` … `comparacion-fase5d.md`); **OK (corregido en F7)**;
**Nueva** = no existe en Python; **Sin equivalente** = pantalla de Python que no existe en Rust a propósito.
En todas: barra lateral y pestañas iguales; Rust usa Inter, tablas con índice gris y gráficos Plotly
interactivos en lugar de matplotlib.

| Pantalla | Qt | Rust | Estado | Notas |
|---|---|---|---|---|
| fase1 · 01_inicio | [qt](qt/01_inicio.png) | [01_inicio.png](rust/01_inicio.png) | OK (corregido en F7) | F7: título de la tarjeta «Regresión» #C43E00 y «¿Qué es…?» #0270B5 (AA); 2×2 tarjetas en 1024 px |
| fase1 · 01_inicio_ventana | [qt](qt/01_inicio_ventana.png) | [01_inicio_ventana.png](rust/01_inicio_ventana.png) | OK |  |
| fase1 · 02_datos_iris | [qt](qt/02_datos_iris.png) | [02_datos_iris.png](rust/02_datos_iris.png) | OK (corregido en F7) | F7: cabeceras ordenables como botones (teclado) |
| fase1 · 02_datos_iris_ventana | [qt](qt/02_datos_iris_ventana.png) | [02_datos_iris_ventana.png](rust/02_datos_iris_ventana.png) | OK |  |
| fase1 · 03_datos_iris_expandido | [qt](qt/03_datos_iris_expandido.png) | [03_datos_iris_expandido.png](rust/03_datos_iris_expandido.png) | OK |  |
| fase1 · 04_datos_correlacion_iris | [qt](qt/04_datos_correlacion_iris.png) | [04_datos_correlacion_iris.png](rust/04_datos_correlacion_iris.png) | OK |  |
| fase1 · 05_datos_pairplot_opciones | [qt](qt/05_datos_pairplot_opciones.png) | [05_datos_pairplot_opciones.png](rust/05_datos_pairplot_opciones.png) | OK |  |
| fase1 · 06_datos_pairplot_kde_iris | [qt](qt/06_datos_pairplot_kde_iris.png) | [06_datos_pairplot_kde_iris.png](rust/06_datos_pairplot_kde_iris.png) | OK |  |
| fase1 · 07_datos_pairplot_hist_iris | [qt](qt/07_datos_pairplot_hist_iris.png) | [07_datos_pairplot_hist_iris.png](rust/07_datos_pairplot_hist_iris.png) | OK |  |
| fase1 · 08_datos_diabetes | [qt](qt/08_datos_diabetes.png) | [08_datos_diabetes.png](rust/08_datos_diabetes.png) | OK |  |
| fase1 · 09_datos_diabetes_expandido | [qt](qt/09_datos_diabetes_expandido.png) | [09_datos_diabetes_expandido.png](rust/09_datos_diabetes_expandido.png) | OK |  |
| fase1 · 10_datos_pairplot_diabetes | [qt](qt/10_datos_pairplot_diabetes.png) | [10_datos_pairplot_diabetes.png](rust/10_datos_pairplot_diabetes.png) | OK |  |
| fase1 · 11_datos_correlacion_digitos | [qt](qt/11_datos_correlacion_digitos.png) | [11_datos_correlacion_digitos.png](rust/11_datos_correlacion_digitos.png) | OK |  |
| fase1 · 12_datos_titanic_desbalance | [qt](qt/12_datos_titanic_desbalance.png) | [12_datos_titanic_desbalance.png](rust/12_datos_titanic_desbalance.png) | OK |  |
| fase1 · 13_datos_moons | [qt](qt/13_datos_moons.png) | [13_datos_moons.png](rust/13_datos_moons.png) | OK |  |
| fase1 · 14_datos_redes_tips | [qt](qt/14_datos_redes_tips.png) | [14_datos_redes_tips.png](rust/14_datos_redes_tips.png) | OK |  |
| fase1 · 15_regresion_pestana_evaluacion_dataset_actual | [qt](qt/15_regresion_pestana_evaluacion_dataset_actual.png) | [15_regresion_pestana_evaluacion_dataset_actual.png](rust/15_regresion_pestana_evaluacion_dataset_actual.png) | OK |  |
| fase1 · 16_csv_vacio | [qt](qt/16_csv_vacio.png) | [16_csv_vacio.png](rust/16_csv_vacio.png) | OK |  |
| fase1 · 16_csv_vacio_ventana | [qt](qt/16_csv_vacio_ventana.png) | [16_csv_vacio_ventana.png](rust/16_csv_vacio_ventana.png) | OK |  |
| fase1 · 17_csv_cargado_clasificacion | [qt](qt/17_csv_cargado_clasificacion.png) | [17_csv_cargado_clasificacion.png](rust/17_csv_cargado_clasificacion.png) | OK |  |
| fase1 · 18_csv_cargado_clasificacion_expandido | [qt](qt/18_csv_cargado_clasificacion_expandido.png) | [18_csv_cargado_clasificacion_expandido.png](rust/18_csv_cargado_clasificacion_expandido.png) | OK |  |
| fase1 · 19_csv_cargado_regresion_expandido | [qt](qt/19_csv_cargado_regresion_expandido.png) | [19_csv_cargado_regresion_expandido.png](rust/19_csv_cargado_regresion_expandido.png) | OK |  |
| fase2 · 00_evaluacion_sin_entrenar | [qt](qt/fase2/00_evaluacion_sin_entrenar.png) | [00_evaluacion_sin_entrenar.png](rust/fase2/00_evaluacion_sin_entrenar.png) | OK |  |
| fase2 · 00b_visualizacion_sin_entrenar | [qt](qt/fase2/00b_visualizacion_sin_entrenar.png) | [00b_visualizacion_sin_entrenar.png](rust/fase2/00b_visualizacion_sin_entrenar.png) | OK |  |
| fase2 · 01_entrenamiento_logistica_iris | [qt](qt/fase2/01_entrenamiento_logistica_iris.png) | [01_entrenamiento_logistica_iris.png](rust/fase2/01_entrenamiento_logistica_iris.png) | OK |  |
| fase2 · 02_entrenado_logistica_iris | [qt](qt/fase2/02_entrenado_logistica_iris.png) | [02_entrenado_logistica_iris.png](rust/fase2/02_entrenado_logistica_iris.png) | OK |  |
| fase2 · 02_entrenado_logistica_iris_ventana | [qt](qt/fase2/02_entrenado_logistica_iris_ventana.png) | [02_entrenado_logistica_iris_ventana.png](rust/fase2/02_entrenado_logistica_iris_ventana.png) | OK |  |
| fase2 · 03_evaluacion_clasificacion_iris | [qt](qt/fase2/03_evaluacion_clasificacion_iris.png) | [03_evaluacion_clasificacion_iris.png](rust/fase2/03_evaluacion_clasificacion_iris.png) | OK |  |
| fase2 · 04_evaluacion_clasificacion_iris_expandido | [qt](qt/fase2/04_evaluacion_clasificacion_iris_expandido.png) | [04_evaluacion_clasificacion_iris_expandido.png](rust/fase2/04_evaluacion_clasificacion_iris_expandido.png) | OK |  |
| fase2 · 05_roc_multiclase_iris | [qt](qt/fase2/05_roc_multiclase_iris.png) | [05_roc_multiclase_iris.png](rust/fase2/05_roc_multiclase_iris.png) | OK |  |
| fase2 · 05_roc_multiclase_iris_ventana | [qt](qt/fase2/05_roc_multiclase_iris_ventana.png) | [05_roc_multiclase_iris_ventana.png](rust/fase2/05_roc_multiclase_iris_ventana.png) | OK |  |
| fase2 · 06_roc_multiclase_iris_expandido | [qt](qt/fase2/06_roc_multiclase_iris_expandido.png) | [06_roc_multiclase_iris_expandido.png](rust/fase2/06_roc_multiclase_iris_expandido.png) | OK |  |
| fase2 · 07_roc_micro_sin_clases_iris | [qt](qt/fase2/07_roc_micro_sin_clases_iris.png) | [07_roc_micro_sin_clases_iris.png](rust/fase2/07_roc_micro_sin_clases_iris.png) | OK |  |
| fase2 · 08_frontera_multiclase_iris | [qt](qt/fase2/08_frontera_multiclase_iris.png) | [08_frontera_multiclase_iris.png](rust/fase2/08_frontera_multiclase_iris.png) | OK |  |
| fase2 · 09_frontera_petalos_iris | [qt](qt/fase2/09_frontera_petalos_iris.png) | [09_frontera_petalos_iris.png](rust/fase2/09_frontera_petalos_iris.png) | OK |  |
| fase2 · 10_frontera_misma_caracteristica | [qt](qt/fase2/10_frontera_misma_caracteristica.png) | [10_frontera_misma_caracteristica.png](rust/fase2/10_frontera_misma_caracteristica.png) | OK |  |
| fase2 · 11_probabilidades_multiclase_iris | [qt](qt/fase2/11_probabilidades_multiclase_iris.png) | [11_probabilidades_multiclase_iris.png](rust/fase2/11_probabilidades_multiclase_iris.png) | OK |  |
| fase2 · 12_coeficientes_logistica_iris | [qt](qt/fase2/12_coeficientes_logistica_iris.png) | [12_coeficientes_logistica_iris.png](rust/fase2/12_coeficientes_logistica_iris.png) | OK |  |
| fase2 · 13_coeficientes_logistica_iris_expandido | [qt](qt/fase2/13_coeficientes_logistica_iris_expandido.png) | [13_coeficientes_logistica_iris_expandido.png](rust/fase2/13_coeficientes_logistica_iris_expandido.png) | OK |  |
| fase2 · 14_coeficientes_clase_versicolor | [qt](qt/fase2/14_coeficientes_clase_versicolor.png) | [14_coeficientes_clase_versicolor.png](rust/fase2/14_coeficientes_clase_versicolor.png) | OK |  |
| fase2 · 15_prediccion_formulario_iris | [qt](qt/fase2/15_prediccion_formulario_iris.png) | [15_prediccion_formulario_iris.png](rust/fase2/15_prediccion_formulario_iris.png) | OK (corregido en F7) | F7: campo numérico del deslizador más ancho |
| fase2 · 16_prediccion_resultado_iris | [qt](qt/fase2/16_prediccion_resultado_iris.png) | [16_prediccion_resultado_iris.png](rust/fase2/16_prediccion_resultado_iris.png) | OK |  |
| fase2 · 17_exportar | [qt](qt/fase2/17_exportar.png) | [17_exportar.png](rust/fase2/17_exportar.png) | OK |  |
| fase2 · 18_exportar_codigo | [qt](qt/fase2/18_exportar_codigo.png) | [18_exportar_codigo.png](rust/fase2/18_exportar_codigo.png) | OK |  |
| fase2 · 19_exportar_pickle | [qt](qt/fase2/19_exportar_pickle.png) | — | Sin equivalente | Sin Pickle en Rust: paquete Python (.zip) con JSON + script sklearn y ONNX (decisión de F2–F4) |
| fase2 · 20_entrenado_cancer | [qt](qt/fase2/20_entrenado_cancer.png) | [20_entrenado_cancer.png](rust/fase2/20_entrenado_cancer.png) | OK |  |
| fase2 · 21_evaluacion_binaria_cancer | [qt](qt/fase2/21_evaluacion_binaria_cancer.png) | [21_evaluacion_binaria_cancer.png](rust/fase2/21_evaluacion_binaria_cancer.png) | OK |  |
| fase2 · 22_roc_binaria_cancer | [qt](qt/fase2/22_roc_binaria_cancer.png) | [22_roc_binaria_cancer.png](rust/fase2/22_roc_binaria_cancer.png) | OK |  |
| fase2 · 23_frontera_binaria_cancer | [qt](qt/fase2/23_frontera_binaria_cancer.png) | [23_frontera_binaria_cancer.png](rust/fase2/23_frontera_binaria_cancer.png) | OK |  |
| fase2 · 24_probabilidades_umbrales_cancer | [qt](qt/fase2/24_probabilidades_umbrales_cancer.png) | [24_probabilidades_umbrales_cancer.png](rust/fase2/24_probabilidades_umbrales_cancer.png) | OK |  |
| fase2 · 25_probabilidades_umbrales_cancer_expandido | [qt](qt/fase2/25_probabilidades_umbrales_cancer_expandido.png) | [25_probabilidades_umbrales_cancer_expandido.png](rust/fase2/25_probabilidades_umbrales_cancer_expandido.png) | OK |  |
| fase2 · 26_prediccion_formulario_titanic | [qt](qt/fase2/26_prediccion_formulario_titanic.png) | [26_prediccion_formulario_titanic.png](rust/fase2/26_prediccion_formulario_titanic.png) | OK |  |
| fase2 · 27_prediccion_resultado_titanic | [qt](qt/fase2/27_prediccion_resultado_titanic.png) | [27_prediccion_resultado_titanic.png](rust/fase2/27_prediccion_resultado_titanic.png) | OK |  |
| fase2 · 28_coeficientes_logistica_titanic | [qt](qt/fase2/28_coeficientes_logistica_titanic.png) | [28_coeficientes_logistica_titanic.png](rust/fase2/28_coeficientes_logistica_titanic.png) | OK |  |
| fase2 · 29_entrenamiento_lineal_diabetes | [qt](qt/fase2/29_entrenamiento_lineal_diabetes.png) | [29_entrenamiento_lineal_diabetes.png](rust/fase2/29_entrenamiento_lineal_diabetes.png) | OK |  |
| fase2 · 30_entrenado_lineal_diabetes | [qt](qt/fase2/30_entrenado_lineal_diabetes.png) | [30_entrenado_lineal_diabetes.png](rust/fase2/30_entrenado_lineal_diabetes.png) | OK |  |
| fase2 · 31_evaluacion_regresion_diabetes | [qt](qt/fase2/31_evaluacion_regresion_diabetes.png) | [31_evaluacion_regresion_diabetes.png](rust/fase2/31_evaluacion_regresion_diabetes.png) | OK |  |
| fase2 · 32_evaluacion_regresion_diabetes_expandido | [qt](qt/fase2/32_evaluacion_regresion_diabetes_expandido.png) | [32_evaluacion_regresion_diabetes_expandido.png](rust/fase2/32_evaluacion_regresion_diabetes_expandido.png) | OK |  |
| fase2 · 33_residuos_diabetes | [qt](qt/fase2/33_residuos_diabetes.png) | [33_residuos_diabetes.png](rust/fase2/33_residuos_diabetes.png) | OK |  |
| fase2 · 33_residuos_diabetes_ventana | [qt](qt/fase2/33_residuos_diabetes_ventana.png) | [33_residuos_diabetes_ventana.png](rust/fase2/33_residuos_diabetes_ventana.png) | OK | Rust tiene la de página entera (`33_residuos_diabetes`) |
| fase2 · 34_residuos_diabetes_expandido | [qt](qt/fase2/34_residuos_diabetes_expandido.png) | [34_residuos_diabetes_expandido.png](rust/fase2/34_residuos_diabetes_expandido.png) | OK |  |
| fase2 · 35_superficie_diabetes | [qt](qt/fase2/35_superficie_diabetes.png) | [35_superficie_diabetes.png](rust/fase2/35_superficie_diabetes.png) | OK |  |
| fase2 · 36_superficie_bmi_s5_diabetes | [qt](qt/fase2/36_superficie_bmi_s5_diabetes.png) | [36_superficie_bmi_s5_diabetes.png](rust/fase2/36_superficie_bmi_s5_diabetes.png) | OK |  |
| fase2 · 37_coeficientes_lineal_diabetes | [qt](qt/fase2/37_coeficientes_lineal_diabetes.png) | [37_coeficientes_lineal_diabetes.png](rust/fase2/37_coeficientes_lineal_diabetes.png) | OK |  |
| fase2 · 38_coeficientes_lineal_diabetes_expandido | [qt](qt/fase2/38_coeficientes_lineal_diabetes_expandido.png) | [38_coeficientes_lineal_diabetes_expandido.png](rust/fase2/38_coeficientes_lineal_diabetes_expandido.png) | OK |  |
| fase2 · 39_prediccion_formulario_diabetes | [qt](qt/fase2/39_prediccion_formulario_diabetes.png) | [39_prediccion_formulario_diabetes.png](rust/fase2/39_prediccion_formulario_diabetes.png) | OK (corregido en F7) | F7: valores negativos de Diabetes ya no se cortan |
| fase2 · 40_prediccion_resultado_diabetes | [qt](qt/fase2/40_prediccion_resultado_diabetes.png) | [40_prediccion_resultado_diabetes.png](rust/fase2/40_prediccion_resultado_diabetes.png) | OK |  |
| fase2 · 41_aviso_lineal_en_clasificacion_iris | [qt](qt/fase2/41_aviso_lineal_en_clasificacion_iris.png) | [41_aviso_lineal_en_clasificacion_iris.png](rust/fase2/41_aviso_lineal_en_clasificacion_iris.png) | OK |  |
| fase2 · 42_aviso_logistica_en_regresion_diabetes | [qt](qt/fase2/42_aviso_logistica_en_regresion_diabetes.png) | [42_aviso_logistica_en_regresion_diabetes.png](rust/fase2/42_aviso_logistica_en_regresion_diabetes.png) | OK |  |
| fase2 · 43_logistica_entrenada_en_diabetes | [qt](qt/fase2/43_logistica_entrenada_en_diabetes.png) | — | Sin equivalente | Bloqueado a propósito (D26): ver `43_logistica_bloqueada_diabetes` |
| fase2 · 44_error_roc_logistica_diabetes | [qt](qt/fase2/44_error_roc_logistica_diabetes.png) | — | Sin equivalente | No se puede llegar (D26) |
| fase3 · 00_evaluacion_sin_entrenar | [qt](qt/fase3/00_evaluacion_sin_entrenar.png) | [00_evaluacion_sin_entrenar.png](rust/fase3/00_evaluacion_sin_entrenar.png) | OK |  |
| fase3 · 00b_visualizacion_sin_entrenar | [qt](qt/fase3/00b_visualizacion_sin_entrenar.png) | [00b_visualizacion_sin_entrenar.png](rust/fase3/00b_visualizacion_sin_entrenar.png) | OK |  |
| fase3 · 01_entrenamiento_iris | [qt](qt/fase3/01_entrenamiento_iris.png) | [01_entrenamiento_iris.png](rust/fase3/01_entrenamiento_iris.png) | OK |  |
| fase3 · 02_entrenado_iris | [qt](qt/fase3/02_entrenado_iris.png) | [02_entrenado_iris.png](rust/fase3/02_entrenado_iris.png) | OK |  |
| fase3 · 02_entrenado_iris_ventana | [qt](qt/fase3/02_entrenado_iris_ventana.png) | [02_entrenado_iris_ventana.png](rust/fase3/02_entrenado_iris_ventana.png) | OK |  |
| fase3 · 03_evaluacion_clasificacion_iris | [qt](qt/fase3/03_evaluacion_clasificacion_iris.png) | [03_evaluacion_clasificacion_iris.png](rust/fase3/03_evaluacion_clasificacion_iris.png) | OK |  |
| fase3 · 04_evaluacion_clasificacion_iris_expandido | [qt](qt/fase3/04_evaluacion_clasificacion_iris_expandido.png) | [04_evaluacion_clasificacion_iris_expandido.png](rust/fase3/04_evaluacion_clasificacion_iris_expandido.png) | OK |  |
| fase3 · 05_arbol_iris | [qt](qt/fase3/05_arbol_iris.png) | [05_arbol_iris.png](rust/fase3/05_arbol_iris.png) | OK (corregido en F7) | F7: tooltip del nodo se cierra con Escape |
| fase3 · 05_arbol_iris_ventana | [qt](qt/fase3/05_arbol_iris_ventana.png) | [05_arbol_iris_ventana.png](rust/fase3/05_arbol_iris_ventana.png) | OK |  |
| fase3 · 06_arbol_iris_20x15 | [qt](qt/fase3/06_arbol_iris_20x15.png) | [06_arbol_iris_20x15.png](rust/fase3/06_arbol_iris_20x15.png) | OK |  |
| fase3 · 07_arbol_iris_codigo | [qt](qt/fase3/07_arbol_iris_codigo.png) | [07_arbol_iris_codigo.png](rust/fase3/07_arbol_iris_codigo.png) | OK |  |
| fase3 · 08_texto_iris | [qt](qt/fase3/08_texto_iris.png) | [08_texto_iris.png](rust/fase3/08_texto_iris.png) | OK |  |
| fase3 · 09_frontera_iris | [qt](qt/fase3/09_frontera_iris.png) | [09_frontera_iris.png](rust/fase3/09_frontera_iris.png) | OK |  |
| fase3 · 10_frontera_petalos_iris | [qt](qt/fase3/10_frontera_petalos_iris.png) | [10_frontera_petalos_iris.png](rust/fase3/10_frontera_petalos_iris.png) | OK |  |
| fase3 · 11_roc_iris | [qt](qt/fase3/11_roc_iris.png) | [11_roc_iris.png](rust/fase3/11_roc_iris.png) | OK |  |
| fase3 · 12_caracteristicas_iris | [qt](qt/fase3/12_caracteristicas_iris.png) | [12_caracteristicas_iris.png](rust/fase3/12_caracteristicas_iris.png) | OK |  |
| fase3 · 13_caracteristicas_iris_expandido | [qt](qt/fase3/13_caracteristicas_iris_expandido.png) | [13_caracteristicas_iris_expandido.png](rust/fase3/13_caracteristicas_iris_expandido.png) | OK |  |
| fase3 · 14_prediccion_formulario_iris | [qt](qt/fase3/14_prediccion_formulario_iris.png) | [14_prediccion_formulario_iris.png](rust/fase3/14_prediccion_formulario_iris.png) | OK |  |
| fase3 · 15_prediccion_camino_iris | [qt](qt/fase3/15_prediccion_camino_iris.png) | [15_prediccion_camino_iris.png](rust/fase3/15_prediccion_camino_iris.png) | OK |  |
| fase3 · 16_prediccion_camino_iris_codigo | [qt](qt/fase3/16_prediccion_camino_iris_codigo.png) | [16_prediccion_camino_iris_codigo.png](rust/fase3/16_prediccion_camino_iris_codigo.png) | OK |  |
| fase3 · 17_exportar_iris | [qt](qt/fase3/17_exportar_iris.png) | [17_exportar_iris.png](rust/fase3/17_exportar_iris.png) | OK |  |
| fase3 · 18_exportar_iris_expandido | [qt](qt/fase3/18_exportar_iris_expandido.png) | [18_exportar_iris_expandido.png](rust/fase3/18_exportar_iris_expandido.png) | OK |  |
| fase3 · 19_exportar_pickle | [qt](qt/fase3/19_exportar_pickle.png) | — | Sin equivalente | Sin Pickle en Rust: paquete Python (.zip) con JSON + script sklearn y ONNX (decisión de F2–F4) |
| fase3 · 20_exportar_reglas | [qt](qt/fase3/20_exportar_reglas.png) | — | Sin equivalente | Reglas en 💾 «Código de entrenamiento» y en 📉 Texto |
| fase3 · 21_exportar_codigo_python | [qt](qt/fase3/21_exportar_codigo_python.png) | — | Sin equivalente | `export_tree_as_python_code` nunca compilaba en Python; sustituido por `21b_exportar_codigo_entrenamiento` |
| fase3 · 22_entrenamiento_cancer_entropy_d10 | [qt](qt/fase3/22_entrenamiento_cancer_entropy_d10.png) | [22_entrenamiento_cancer_entropy_d10.png](rust/fase3/22_entrenamiento_cancer_entropy_d10.png) | OK |  |
| fase3 · 23_arbol_cancer_d10 | [qt](qt/fase3/23_arbol_cancer_d10.png) | [23_arbol_cancer_d10.png](rust/fase3/23_arbol_cancer_d10.png) | OK |  |
| fase3 · 24_roc_binaria_cancer | [qt](qt/fase3/24_roc_binaria_cancer.png) | [24_roc_binaria_cancer.png](rust/fase3/24_roc_binaria_cancer.png) | OK |  |
| fase3 · 25_prediccion_formulario_titanic | [qt](qt/fase3/25_prediccion_formulario_titanic.png) | [25_prediccion_formulario_titanic.png](rust/fase3/25_prediccion_formulario_titanic.png) | OK |  |
| fase3 · 26_prediccion_camino_titanic | [qt](qt/fase3/26_prediccion_camino_titanic.png) | [26_prediccion_camino_titanic.png](rust/fase3/26_prediccion_camino_titanic.png) | OK |  |
| fase3 · 27_aviso_clasificacion_en_diabetes | [qt](qt/fase3/27_aviso_clasificacion_en_diabetes.png) | [27_aviso_clasificacion_en_diabetes.png](rust/fase3/27_aviso_clasificacion_en_diabetes.png) | OK |  |
| fase3 · 28_entrenamiento_regresion_diabetes | [qt](qt/fase3/28_entrenamiento_regresion_diabetes.png) | [28_entrenamiento_regresion_diabetes.png](rust/fase3/28_entrenamiento_regresion_diabetes.png) | OK |  |
| fase3 · 29_entrenado_regresion_diabetes | [qt](qt/fase3/29_entrenado_regresion_diabetes.png) | [29_entrenado_regresion_diabetes.png](rust/fase3/29_entrenado_regresion_diabetes.png) | OK |  |
| fase3 · 30_evaluacion_regresion_diabetes | [qt](qt/fase3/30_evaluacion_regresion_diabetes.png) | [30_evaluacion_regresion_diabetes.png](rust/fase3/30_evaluacion_regresion_diabetes.png) | OK |  |
| fase3 · 31_arbol_regresion_diabetes | [qt](qt/fase3/31_arbol_regresion_diabetes.png) | [31_arbol_regresion_diabetes.png](rust/fase3/31_arbol_regresion_diabetes.png) | OK |  |
| fase3 · 31_arbol_regresion_diabetes_ventana | [qt](qt/fase3/31_arbol_regresion_diabetes_ventana.png) | [31_arbol_regresion_diabetes_ventana.png](rust/fase3/31_arbol_regresion_diabetes_ventana.png) | OK |  |
| fase3 · 32_texto_regresion_diabetes | [qt](qt/fase3/32_texto_regresion_diabetes.png) | [32_texto_regresion_diabetes.png](rust/fase3/32_texto_regresion_diabetes.png) | OK |  |
| fase3 · 33_residuos_diabetes | [qt](qt/fase3/33_residuos_diabetes.png) | [33_residuos_diabetes.png](rust/fase3/33_residuos_diabetes.png) | OK |  |
| fase3 · 33_residuos_diabetes_ventana | [qt](qt/fase3/33_residuos_diabetes_ventana.png) | — | Sin equivalente | Rust tiene la de página entera (`33_residuos_diabetes`) |
| fase3 · 34_superficie_diabetes | [qt](qt/fase3/34_superficie_diabetes.png) | [34_superficie_diabetes.png](rust/fase3/34_superficie_diabetes.png) | OK |  |
| fase3 · 35_superficie_bmi_s5_diabetes | [qt](qt/fase3/35_superficie_bmi_s5_diabetes.png) | [35_superficie_bmi_s5_diabetes.png](rust/fase3/35_superficie_bmi_s5_diabetes.png) | OK |  |
| fase3 · 36_caracteristicas_regresion_diabetes | [qt](qt/fase3/36_caracteristicas_regresion_diabetes.png) | [36_caracteristicas_regresion_diabetes.png](rust/fase3/36_caracteristicas_regresion_diabetes.png) | OK |  |
| fase3 · 37_caracteristicas_regresion_diabetes_expandido | [qt](qt/fase3/37_caracteristicas_regresion_diabetes_expandido.png) | [37_caracteristicas_regresion_diabetes_expandido.png](rust/fase3/37_caracteristicas_regresion_diabetes_expandido.png) | OK |  |
| fase3 · 38_prediccion_regresion_diabetes | [qt](qt/fase3/38_prediccion_regresion_diabetes.png) | [38_prediccion_regresion_diabetes.png](rust/fase3/38_prediccion_regresion_diabetes.png) | OK (corregido en F7) | F7: «-0,000568» ya no se corta |
| fase3 · 39_exportar_regresion_diabetes | [qt](qt/fase3/39_exportar_regresion_diabetes.png) | [39_exportar_regresion_diabetes.png](rust/fase3/39_exportar_regresion_diabetes.png) | OK |  |
| fase3 · 40_arbol_california_d8 | [qt](qt/fase3/40_arbol_california_d8.png) | [40_arbol_california_d8.png](rust/fase3/40_arbol_california_d8.png) | OK |  |
| fase3 · 41_error_clasificacion_california | [qt](qt/fase3/41_error_clasificacion_california.png) | [41_error_clasificacion_california.png](rust/fase3/41_error_clasificacion_california.png) | OK |  |
| fase3 · 42_clasificacion_entrenada_en_diabetes | [qt](qt/fase3/42_clasificacion_entrenada_en_diabetes.png) | — | Sin equivalente | Bloqueado a propósito: ver `42_clasificacion_bloqueada_diabetes` |
| fase3 · 43_error_evaluacion_clasificacion_diabetes | [qt](qt/fase3/43_error_evaluacion_clasificacion_diabetes.png) | — | Sin equivalente | No se puede llegar (bloqueado) |
| fase3 · 44_aviso_regresion_en_iris | [qt](qt/fase3/44_aviso_regresion_en_iris.png) | [44_aviso_regresion_en_iris.png](rust/fase3/44_aviso_regresion_en_iris.png) | OK |  |
| fase4 · 00_evaluacion_sin_entrenar | [qt](qt/fase4/00_evaluacion_sin_entrenar.png) | [00_evaluacion_sin_entrenar.png](rust/fase4/00_evaluacion_sin_entrenar.png) | OK |  |
| fase4 · 00b_visualizacion_sin_entrenar | [qt](qt/fase4/00b_visualizacion_sin_entrenar.png) | [00b_visualizacion_sin_entrenar.png](rust/fase4/00b_visualizacion_sin_entrenar.png) | OK |  |
| fase4 · 00c_caracteristicas_sin_entrenar | [qt](qt/fase4/00c_caracteristicas_sin_entrenar.png) | [00c_caracteristicas_sin_entrenar.png](rust/fase4/00c_caracteristicas_sin_entrenar.png) | OK |  |
| fase4 · 01_entrenamiento_iris | [qt](qt/fase4/01_entrenamiento_iris.png) | [01_entrenamiento_iris.png](rust/fase4/01_entrenamiento_iris.png) | OK |  |
| fase4 · 02_entrenado_iris | [qt](qt/fase4/02_entrenado_iris.png) | [02_entrenado_iris.png](rust/fase4/02_entrenado_iris.png) | OK |  |
| fase4 · 02_entrenado_iris_ventana | [qt](qt/fase4/02_entrenado_iris_ventana.png) | [02_entrenado_iris_ventana.png](rust/fase4/02_entrenado_iris_ventana.png) | OK |  |
| fase4 · 03_evaluacion_clasificacion_iris | [qt](qt/fase4/03_evaluacion_clasificacion_iris.png) | [03_evaluacion_clasificacion_iris.png](rust/fase4/03_evaluacion_clasificacion_iris.png) | OK |  |
| fase4 · 04_evaluacion_clasificacion_iris_expandido | [qt](qt/fase4/04_evaluacion_clasificacion_iris_expandido.png) | [04_evaluacion_clasificacion_iris_expandido.png](rust/fase4/04_evaluacion_clasificacion_iris_expandido.png) | OK |  |
| fase4 · 05_probabilidades_multiclase_iris | [qt](qt/fase4/05_probabilidades_multiclase_iris.png) | [05_probabilidades_multiclase_iris.png](rust/fase4/05_probabilidades_multiclase_iris.png) | OK |  |
| fase4 · 06_roc_multiclase_iris | [qt](qt/fase4/06_roc_multiclase_iris.png) | [06_roc_multiclase_iris.png](rust/fase4/06_roc_multiclase_iris.png) | OK |  |
| fase4 · 07_frontera_iris | [qt](qt/fase4/07_frontera_iris.png) | [07_frontera_iris.png](rust/fase4/07_frontera_iris.png) | OK |  |
| fase4 · 08_frontera_petalos_iris | [qt](qt/fase4/08_frontera_petalos_iris.png) | [08_frontera_petalos_iris.png](rust/fase4/08_frontera_petalos_iris.png) | OK |  |
| fase4 · 09_interactiva_iris | [qt](qt/fase4/09_interactiva_iris.png) | [09_interactiva_iris.png](rust/fase4/09_interactiva_iris.png) | OK (corregido en F7) | F7: casillas con etiqueta, «Limpiar Puntos» #D32F2F (AA); en 1024 px los controles pasan a 2 filas (antes se salían) |
| fase4 · 09_interactiva_iris_ventana | [qt](qt/fase4/09_interactiva_iris_ventana.png) | [09_interactiva_iris_ventana.png](rust/fase4/09_interactiva_iris_ventana.png) | OK |  |
| fase4 · 10_interactiva_iris_expandido | [qt](qt/fase4/10_interactiva_iris_expandido.png) | [10_interactiva_iris_expandido.png](rust/fase4/10_interactiva_iris_expandido.png) | OK |  |
| fase4 · 11_interactiva_misma_caracteristica | [qt](qt/fase4/11_interactiva_misma_caracteristica.png) | [11_interactiva_misma_caracteristica.png](rust/fase4/11_interactiva_misma_caracteristica.png) | OK |  |
| fase4 · 12_distancias_iris | [qt](qt/fase4/12_distancias_iris.png) | [12_distancias_iris.png](rust/fase4/12_distancias_iris.png) | OK |  |
| fase4 · 13_distancias_iris_expandido | [qt](qt/fase4/13_distancias_iris_expandido.png) | [13_distancias_iris_expandido.png](rust/fase4/13_distancias_iris_expandido.png) | OK |  |
| fase4 · 14_caracteristicas_iris | [qt](qt/fase4/14_caracteristicas_iris.png) | [14_caracteristicas_iris.png](rust/fase4/14_caracteristicas_iris.png) | OK |  |
| fase4 · 15_caracteristicas_iris_expandido | [qt](qt/fase4/15_caracteristicas_iris_expandido.png) | [15_caracteristicas_iris_expandido.png](rust/fase4/15_caracteristicas_iris_expandido.png) | OK |  |
| fase4 · 16_prediccion_formulario_iris | [qt](qt/fase4/16_prediccion_formulario_iris.png) | [16_prediccion_formulario_iris.png](rust/fase4/16_prediccion_formulario_iris.png) | OK (corregido en F7) | F7: campo numérico del deslizador 112 px sin flechas (antes cortaba «-0,00056») |
| fase4 · 17_prediccion_resultado_iris | [qt](qt/fase4/17_prediccion_resultado_iris.png) | [17_prediccion_resultado_iris.png](rust/fase4/17_prediccion_resultado_iris.png) | OK |  |
| fase4 · 18_exportar_iris | [qt](qt/fase4/18_exportar_iris.png) | [18_exportar_iris.png](rust/fase4/18_exportar_iris.png) | OK |  |
| fase4 · 19_exportar_codigo | [qt](qt/fase4/19_exportar_codigo.png) | [19_exportar_codigo.png](rust/fase4/19_exportar_codigo.png) | OK |  |
| fase4 · 20_exportar_pickle | [qt](qt/fase4/20_exportar_pickle.png) | — | Sin equivalente | Sin Pickle en Rust: paquete Python (.zip) + ONNX |
| fase4 · 21_entrenado_cancer | [qt](qt/fase4/21_entrenado_cancer.png) | [21_entrenado_cancer.png](rust/fase4/21_entrenado_cancer.png) | OK |  |
| fase4 · 22_probabilidades_umbrales_cancer | [qt](qt/fase4/22_probabilidades_umbrales_cancer.png) | [22_probabilidades_umbrales_cancer.png](rust/fase4/22_probabilidades_umbrales_cancer.png) | OK |  |
| fase4 · 23_probabilidades_umbrales_cancer_expandido | [qt](qt/fase4/23_probabilidades_umbrales_cancer_expandido.png) | [23_probabilidades_umbrales_cancer_expandido.png](rust/fase4/23_probabilidades_umbrales_cancer_expandido.png) | OK |  |
| fase4 · 24_roc_binaria_cancer | [qt](qt/fase4/24_roc_binaria_cancer.png) | [24_roc_binaria_cancer.png](rust/fase4/24_roc_binaria_cancer.png) | OK |  |
| fase4 · 25_frontera_binaria_cancer | [qt](qt/fase4/25_frontera_binaria_cancer.png) | [25_frontera_binaria_cancer.png](rust/fase4/25_frontera_binaria_cancer.png) | OK |  |
| fase4 · 26_distancias_cancer | [qt](qt/fase4/26_distancias_cancer.png) | [26_distancias_cancer.png](rust/fase4/26_distancias_cancer.png) | OK |  |
| fase4 · 27_prediccion_formulario_titanic | [qt](qt/fase4/27_prediccion_formulario_titanic.png) | [27_prediccion_formulario_titanic.png](rust/fase4/27_prediccion_formulario_titanic.png) | OK |  |
| fase4 · 28_prediccion_resultado_titanic | [qt](qt/fase4/28_prediccion_resultado_titanic.png) | [28_prediccion_resultado_titanic.png](rust/fase4/28_prediccion_resultado_titanic.png) | OK |  |
| fase4 · 29_entrenamiento_k20_distance_manhattan | [qt](qt/fase4/29_entrenamiento_k20_distance_manhattan.png) | [29_entrenamiento_k20_distance_manhattan.png](rust/fase4/29_entrenamiento_k20_distance_manhattan.png) | OK |  |
| fase4 · 30_interactiva_k20_distance_manhattan | [qt](qt/fase4/30_interactiva_k20_distance_manhattan.png) | [30_interactiva_k20_distance_manhattan.png](rust/fase4/30_interactiva_k20_distance_manhattan.png) | OK (corregido en F7) | F7: igual que 09 |
| fase4 · 31_frontera_k20_distance_manhattan | [qt](qt/fase4/31_frontera_k20_distance_manhattan.png) | [31_frontera_k20_distance_manhattan.png](rust/fase4/31_frontera_k20_distance_manhattan.png) | OK |  |
| fase4 · 32_entrenamiento_regresion_diabetes | [qt](qt/fase4/32_entrenamiento_regresion_diabetes.png) | [32_entrenamiento_regresion_diabetes.png](rust/fase4/32_entrenamiento_regresion_diabetes.png) | OK |  |
| fase4 · 33_entrenado_regresion_diabetes | [qt](qt/fase4/33_entrenado_regresion_diabetes.png) | [33_entrenado_regresion_diabetes.png](rust/fase4/33_entrenado_regresion_diabetes.png) | OK |  |
| fase4 · 34_evaluacion_regresion_diabetes | [qt](qt/fase4/34_evaluacion_regresion_diabetes.png) | [34_evaluacion_regresion_diabetes.png](rust/fase4/34_evaluacion_regresion_diabetes.png) | OK |  |
| fase4 · 35_predicciones_residuos_diabetes | [qt](qt/fase4/35_predicciones_residuos_diabetes.png) | [35_predicciones_residuos_diabetes.png](rust/fase4/35_predicciones_residuos_diabetes.png) | OK |  |
| fase4 · 35_predicciones_residuos_diabetes_ventana | [qt](qt/fase4/35_predicciones_residuos_diabetes_ventana.png) | [35_predicciones_residuos_diabetes_ventana.png](rust/fase4/35_predicciones_residuos_diabetes_ventana.png) | OK |  |
| fase4 · 36_predicciones_residuos_diabetes_expandido | [qt](qt/fase4/36_predicciones_residuos_diabetes_expandido.png) | [36_predicciones_residuos_diabetes_expandido.png](rust/fase4/36_predicciones_residuos_diabetes_expandido.png) | OK |  |
| fase4 · 37_superficie_diabetes | [qt](qt/fase4/37_superficie_diabetes.png) | [37_superficie_diabetes.png](rust/fase4/37_superficie_diabetes.png) | OK |  |
| fase4 · 38_superficie_bmi_s5_diabetes | [qt](qt/fase4/38_superficie_bmi_s5_diabetes.png) | [38_superficie_bmi_s5_diabetes.png](rust/fase4/38_superficie_bmi_s5_diabetes.png) | OK |  |
| fase4 · 39_caracteristicas_regresion_diabetes | [qt](qt/fase4/39_caracteristicas_regresion_diabetes.png) | [39_caracteristicas_regresion_diabetes.png](rust/fase4/39_caracteristicas_regresion_diabetes.png) | OK |  |
| fase4 · 40_prediccion_regresion_diabetes | [qt](qt/fase4/40_prediccion_regresion_diabetes.png) | [40_prediccion_regresion_diabetes.png](rust/fase4/40_prediccion_regresion_diabetes.png) | OK (corregido en F7) | F7: valores negativos de Diabetes ya no se cortan |
| fase4 · 41_exportar_regresion_diabetes | [qt](qt/fase4/41_exportar_regresion_diabetes.png) | [41_exportar_regresion_diabetes.png](rust/fase4/41_exportar_regresion_diabetes.png) | OK |  |
| fase4 · 42_superficie_california | [qt](qt/fase4/42_superficie_california.png) | [42_superficie_california.png](rust/fase4/42_superficie_california.png) | OK |  |
| fase4 · 43_caracteristicas_california | [qt](qt/fase4/43_caracteristicas_california.png) | [43_caracteristicas_california.png](rust/fase4/43_caracteristicas_california.png) | OK |  |
| fase5 · 00_entrenamiento_sin_configurar | [qt](qt/fase5/00_entrenamiento_sin_configurar.png) | — | Sin equivalente | En Rust la arquitectura tiene valores por defecto: no existe el estado «sin configurar» |
| fase5 · 00b_evaluacion_sin_entrenar | [qt](qt/fase5/00b_evaluacion_sin_entrenar.png) | [00b_evaluacion_sin_entrenar.png](rust/fase5/00b_evaluacion_sin_entrenar.png) | OK |  |
| fase5 · 00c_visualizacion_sin_entrenar | [qt](qt/fase5/00c_visualizacion_sin_entrenar.png) | — | Sin equivalente | Mismo aviso que `00b` |
| fase5 · 00d_predicciones_sin_entrenar | [qt](qt/fase5/00d_predicciones_sin_entrenar.png) | [00d_predicciones_sin_entrenar.png](rust/fase5/00d_predicciones_sin_entrenar.png) | OK |  |
| fase5 · 00e_exportar_sin_entrenar | [qt](qt/fase5/00e_exportar_sin_entrenar.png) | [00e_exportar_sin_entrenar.png](rust/fase5/00e_exportar_sin_entrenar.png) | OK |  |
| fase5 · 01_datos_iris | [qt](qt/fase5/01_datos_iris.png) | — | Sin equivalente | Igual que `fase1/02_datos_iris` (pestaña 📊 compartida) |
| fase5 · 02_arquitectura_iris | [qt](qt/fase5/02_arquitectura_iris.png) | [02_arquitectura_iris.png](rust/fase5/02_arquitectura_iris.png) | OK |  |
| fase5 · 02_arquitectura_iris_ventana | [qt](qt/fase5/02_arquitectura_iris_ventana.png) | [02_arquitectura_iris_ventana.png](rust/fase5/02_arquitectura_iris_ventana.png) | OK |  |
| fase5 · 03_arquitectura_iris_expandido | [qt](qt/fase5/03_arquitectura_iris_expandido.png) | [03_arquitectura_iris_expandido.png](rust/fase5/03_arquitectura_iris_expandido.png) | OK |  |
| fase5 · 04_arquitectura_salida_linear | [qt](qt/fase5/04_arquitectura_salida_linear.png) | [04_arquitectura_salida_linear.png](rust/fase5/04_arquitectura_salida_linear.png) | OK |  |
| fase5 · 05_arquitectura_salida_sigmoid_multiclase | [qt](qt/fase5/05_arquitectura_salida_sigmoid_multiclase.png) | [05_arquitectura_salida_sigmoid_multiclase.png](rust/fase5/05_arquitectura_salida_sigmoid_multiclase.png) | OK |  |
| fase5 · 06_arquitectura_oculta_sigmoid | [qt](qt/fase5/06_arquitectura_oculta_sigmoid.png) | [06_arquitectura_oculta_sigmoid.png](rust/fase5/06_arquitectura_oculta_sigmoid.png) | OK |  |
| fase5 · 07_arquitectura_1_capa | [qt](qt/fase5/07_arquitectura_1_capa.png) | [07_arquitectura_1_capa.png](rust/fase5/07_arquitectura_1_capa.png) | OK |  |
| fase5 · 08_arquitectura_5_capas | [qt](qt/fase5/08_arquitectura_5_capas.png) | [08_arquitectura_5_capas.png](rust/fase5/08_arquitectura_5_capas.png) | OK |  |
| fase5 · 09_arquitectura_codigo | [qt](qt/fase5/09_arquitectura_codigo.png) | [09_arquitectura_codigo.png](rust/fase5/09_arquitectura_codigo.png) | OK |  |
| fase5 · 10_entrenamiento_iris | [qt](qt/fase5/10_entrenamiento_iris.png) | [10_entrenamiento_iris.png](rust/fase5/10_entrenamiento_iris.png) | OK |  |
| fase5 · 11_entrenamiento_iris_avanzado | [qt](qt/fase5/11_entrenamiento_iris_avanzado.png) | [11_entrenamiento_iris_avanzado.png](rust/fase5/11_entrenamiento_iris_avanzado.png) | OK |  |
| fase5 · 12_entrenamiento_sin_callbacks | [qt](qt/fase5/12_entrenamiento_sin_callbacks.png) | [12_entrenamiento_sin_callbacks.png](rust/fase5/12_entrenamiento_sin_callbacks.png) | OK |  |
| fase5 · 13_entrenado_iris | [qt](qt/fase5/13_entrenado_iris.png) | [13_entrenado_iris.png](rust/fase5/13_entrenado_iris.png) | OK |  |
| fase5 · 13_entrenado_iris_ventana | [qt](qt/fase5/13_entrenado_iris_ventana.png) | [13_entrenado_iris_ventana.png](rust/fase5/13_entrenado_iris_ventana.png) | OK |  |
| fase5 · 14_evaluacion_iris | [qt](qt/fase5/14_evaluacion_iris.png) | [14_evaluacion_iris.png](rust/fase5/14_evaluacion_iris.png) | OK |  |
| fase5 · 15_evaluacion_iris_expandido | [qt](qt/fase5/15_evaluacion_iris_expandido.png) | [15_evaluacion_iris_expandido.png](rust/fase5/15_evaluacion_iris_expandido.png) | OK |  |
| fase5 · 16_historial_iris | [qt](qt/fase5/16_historial_iris.png) | [16_historial_iris.png](rust/fase5/16_historial_iris.png) | OK |  |
| fase5 · 17_pesos_iris | [qt](qt/fase5/17_pesos_iris.png) | [17_pesos_iris.png](rust/fase5/17_pesos_iris.png) | OK (corregido en F7) | Mapa/histogramas OK también en ventana normal (@vista) |
| fase5 · 18_pesos_iris_expandido | [qt](qt/fase5/18_pesos_iris_expandido.png) | [18_pesos_iris_expandido.png](rust/fase5/18_pesos_iris_expandido.png) | OK |  |
| fase5 · 19_superficie_iris | [qt](qt/fase5/19_superficie_iris.png) | [19_superficie_iris.png](rust/fase5/19_superficie_iris.png) | OK |  |
| fase5 · 20_superficie_petalos_iris | [qt](qt/fase5/20_superficie_petalos_iris.png) | [20_superficie_petalos_iris.png](rust/fase5/20_superficie_petalos_iris.png) | OK |  |
| fase5 · 21_activaciones_iris | [qt](qt/fase5/21_activaciones_iris.png) | [21_activaciones_iris.png](rust/fase5/21_activaciones_iris.png) | OK (corregido en F7) | F7: lienzo con role=img; tooltip se cierra con Escape; redibujo al cambiar de pantalla 1×/2× |
| fase5 · 21_activaciones_iris_ventana | [qt](qt/fase5/21_activaciones_iris_ventana.png) | [21_activaciones_iris_ventana.png](rust/fase5/21_activaciones_iris_ventana.png) | OK |  |
| fase5 · 22_activaciones_iris_expandido | [qt](qt/fase5/22_activaciones_iris_expandido.png) | [22_activaciones_iris_expandido.png](rust/fase5/22_activaciones_iris_expandido.png) | OK |  |
| fase5 · 23_prediccion_formulario_iris | [qt](qt/fase5/23_prediccion_formulario_iris.png) | [23_prediccion_formulario_iris.png](rust/fase5/23_prediccion_formulario_iris.png) | OK |  |
| fase5 · 24_prediccion_resultado_iris | [qt](qt/fase5/24_prediccion_resultado_iris.png) | [24_prediccion_resultado_iris.png](rust/fase5/24_prediccion_resultado_iris.png) | OK |  |
| fase5 · 25_exportar_tensorflow | [qt](qt/fase5/25_exportar_tensorflow.png) | — | Sin equivalente | D75: `25_exportar_keras` (paquete Keras .zip) |
| fase5 · 26_exportar_savedmodel | [qt](qt/fase5/26_exportar_savedmodel.png) | — | Sin equivalente | D75: dentro del paquete Keras |
| fase5 · 27_exportar_completo | [qt](qt/fase5/27_exportar_completo.png) | — | Sin equivalente | D75: dentro del paquete Keras |
| fase5 · 28_exportar_python | [qt](qt/fase5/28_exportar_python.png) | [28_exportar_python.png](rust/fase5/28_exportar_python.png) | OK |  |
| fase5 · 29_exportar_metadatos | [qt](qt/fase5/29_exportar_metadatos.png) | [29_exportar_metadatos.png](rust/fase5/29_exportar_metadatos.png) | OK |  |
| fase5 · 30_arquitectura_cancer_softmax_heredado | [qt](qt/fase5/30_arquitectura_cancer_softmax_heredado.png) | [30_arquitectura_cancer_softmax_heredado.png](rust/fase5/30_arquitectura_cancer_softmax_heredado.png) | OK |  |
| fase5 · 30b_error_softmax_heredado_cancer | [qt](qt/fase5/30b_error_softmax_heredado_cancer.png) | — | Sin equivalente | Fallo de Python (selector sin key) no reproducible en Rust |
| fase5 · 30c_arquitectura_cancer | [qt](qt/fase5/30c_arquitectura_cancer.png) | [30c_arquitectura_cancer.png](rust/fase5/30c_arquitectura_cancer.png) | OK |  |
| fase5 · 31_entrenado_cancer | [qt](qt/fase5/31_entrenado_cancer.png) | [31_entrenado_cancer.png](rust/fase5/31_entrenado_cancer.png) | OK |  |
| fase5 · 32_evaluacion_cancer | [qt](qt/fase5/32_evaluacion_cancer.png) | [32_evaluacion_cancer.png](rust/fase5/32_evaluacion_cancer.png) | OK |  |
| fase5 · 33_superficie_cancer | [qt](qt/fase5/33_superficie_cancer.png) | [33_superficie_cancer.png](rust/fase5/33_superficie_cancer.png) | OK |  |
| fase5 · 34_prediccion_resultado_cancer | [qt](qt/fase5/34_prediccion_resultado_cancer.png) | [34_prediccion_resultado_cancer.png](rust/fase5/34_prediccion_resultado_cancer.png) | OK |  |
| fase5 · 35_arquitectura_diabetes | [qt](qt/fase5/35_arquitectura_diabetes.png) | [35_arquitectura_diabetes.png](rust/fase5/35_arquitectura_diabetes.png) | OK |  |
| fase5 · 36_entrenado_diabetes | [qt](qt/fase5/36_entrenado_diabetes.png) | [36_entrenado_diabetes.png](rust/fase5/36_entrenado_diabetes.png) | OK |  |
| fase5 · 37_evaluacion_diabetes | [qt](qt/fase5/37_evaluacion_diabetes.png) | [37_evaluacion_diabetes.png](rust/fase5/37_evaluacion_diabetes.png) | OK |  |
| fase5 · 38_historial_diabetes | [qt](qt/fase5/38_historial_diabetes.png) | [38_historial_diabetes.png](rust/fase5/38_historial_diabetes.png) | OK |  |
| fase5 · 39_superficie_diabetes | [qt](qt/fase5/39_superficie_diabetes.png) | [39_superficie_diabetes.png](rust/fase5/39_superficie_diabetes.png) | OK |  |
| fase5 · 39b_activaciones_diabetes | [qt](qt/fase5/39b_activaciones_diabetes.png) | [39b_activaciones_diabetes.png](rust/fase5/39b_activaciones_diabetes.png) | OK |  |
| fase5 · 40_prediccion_resultado_diabetes | [qt](qt/fase5/40_prediccion_resultado_diabetes.png) | [40_prediccion_resultado_diabetes.png](rust/fase5/40_prediccion_resultado_diabetes.png) | OK |  |
| fase5 · 41_superficie_moons | [qt](qt/fase5/41_superficie_moons.png) | [41_superficie_moons.png](rust/fase5/41_superficie_moons.png) | OK |  |
| fase5 · 42_error_salida_sigmoid_iris | [qt](qt/fase5/42_error_salida_sigmoid_iris.png) | [42_error_salida_sigmoid_iris.png](rust/fase5/42_error_salida_sigmoid_iris.png) | OK |  |
| fase5 · 43_evaluacion_tras_error | [qt](qt/fase5/43_evaluacion_tras_error.png) | — | Sin equivalente | Fallo de Python (`model_trained_nn` sin modelo) no reproducible |
| fase2 · 43_logistica_bloqueada_diabetes | — | [43_logistica_bloqueada_diabetes.png](rust/fase2/43_logistica_bloqueada_diabetes.png) | Nueva |  |
| fase2 · 45_evaluacion_california | — | [45_evaluacion_california.png](rust/fase2/45_evaluacion_california.png) | Nueva |  |
| fase2 · 46_superficie_california | — | [46_superficie_california.png](rust/fase2/46_superficie_california.png) | Nueva |  |
| fase3 · 13b_caracteristicas_permutacion_iris | — | [13b_caracteristicas_permutacion_iris.png](rust/fase3/13b_caracteristicas_permutacion_iris.png) | Nueva |  |
| fase3 · 21b_exportar_codigo_entrenamiento | — | [21b_exportar_codigo_entrenamiento.png](rust/fase3/21b_exportar_codigo_entrenamiento.png) | Nueva |  |
| fase3 · 40_arbol_california_d8_ventana | — | [40_arbol_california_d8_ventana.png](rust/fase3/40_arbol_california_d8_ventana.png) | Nueva |  |
| fase3 · 42_clasificacion_bloqueada_diabetes | — | [42_clasificacion_bloqueada_diabetes.png](rust/fase3/42_clasificacion_bloqueada_diabetes.png) | Nueva |  |
| fase3 · 44b_regresion_entrenada_en_iris | — | [44b_regresion_entrenada_en_iris.png](rust/fase3/44b_regresion_entrenada_en_iris.png) | Nueva |  |
| fase4 · 09d_interactiva_iris_con_puntos | — | [09d_interactiva_iris_con_puntos.png](rust/fase4/09d_interactiva_iris_con_puntos.png) | Nueva |  |
| fase4 · 25b_interactiva_cancer | — | [25b_interactiva_cancer.png](rust/fase4/25b_interactiva_cancer.png) | Nueva |  |
| fase4 · 31b_prediccion_resultado_k20_distance_manhattan | — | [31b_prediccion_resultado_k20_distance_manhattan.png](rust/fase4/31b_prediccion_resultado_k20_distance_manhattan.png) | Nueva |  |
| fase5 · 08b_arquitectura_5x256 | — | [08b_arquitectura_5x256.png](rust/fase5/08b_arquitectura_5x256.png) | Nueva |  |
| fase5 · 13b_entrenado_iris_lr01 | — | [13b_entrenado_iris_lr01.png](rust/fase5/13b_entrenado_iris_lr01.png) | Nueva |  |
| fase5 · 13c_entrenamiento_cancelado | — | [13c_entrenamiento_cancelado.png](rust/fase5/13c_entrenamiento_cancelado.png) | Nueva |  |
| fase5 · 18b_pesos_5x256 | — | [18b_pesos_5x256.png](rust/fase5/18b_pesos_5x256.png) | Nueva |  |
| fase5 · 22b_activaciones_5x256 | — | [22b_activaciones_5x256.png](rust/fase5/22b_activaciones_5x256.png) | Nueva |  |
| fase5 · 25_exportar_keras | — | [25_exportar_keras.png](rust/fase5/25_exportar_keras.png) | Nueva |  |
| fase5 · 25b_exportar_onnx | — | [25b_exportar_onnx.png](rust/fase5/25b_exportar_onnx.png) | Nueva |  |
| fase5 · 43b_evaluacion_divergencia | — | [43b_evaluacion_divergencia.png](rust/fase5/43b_evaluacion_divergencia.png) | Nueva |  |
| fase5 · 43c_evaluacion_divergencia_recuperada | — | [43c_evaluacion_divergencia_recuperada.png](rust/fase5/43c_evaluacion_divergencia_recuperada.png) | Nueva |  |
| fase5 · 44_lotes_formato_iris | — | [44_lotes_formato_iris.png](rust/fase5/44_lotes_formato_iris.png) | Nueva |  |
| fase5 · 45_lotes_iris | — | [45_lotes_iris.png](rust/fase5/45_lotes_iris.png) | Nueva |  |
| fase5 · 45b_lotes_diabetes | — | [45b_lotes_diabetes.png](rust/fase5/45b_lotes_diabetes.png) | Nueva |  |
| fase5 · 46_exploracion_iris | — | [46_exploracion_iris.png](rust/fase5/46_exploracion_iris.png) | Nueva |  |
| fase5 · 46b_exploracion_iris_expandido | — | [46b_exploracion_iris_expandido.png](rust/fase5/46b_exploracion_iris_expandido.png) | Nueva |  |
| fase5 · 47_exploracion_cancer | — | [47_exploracion_cancer.png](rust/fase5/47_exploracion_cancer.png) | Nueva |  |
| fase5 · 48_exploracion_diabetes | — | [48_exploracion_diabetes.png](rust/fase5/48_exploracion_diabetes.png) | Nueva |  |
| fase5 · 49_prediccion_divergencia | — | [49_prediccion_divergencia.png](rust/fase5/49_prediccion_divergencia.png) | Nueva |  |
| fase5 · 49b_exportar_divergencia | — | [49b_exportar_divergencia.png](rust/fase5/49b_exportar_divergencia.png) | Nueva |  |

## Diferencias visuales que quedan

- Paleta del lienzo KNN de Python: versicolor y virginica (dos turquesas) se distinguen poco, sobre todo
  en tema oscuro; se mantiene por paridad (decidir si se cambia).
- Los campos numéricos muestran coma decimal («0,00289») porque WebKit localiza `input[type=number]` con
  `lang="es"`, como Qt; el resto de la app (tablas, textos) usa punto como Python.
- Los nodos del árbol solo muestran su detalle al pasar el ratón; con teclado, la vista 📝 Texto da la
  misma información.
- No probado con lector de pantalla real (VoiceOver/Narrator/Orca) ni con Tab real en WKWebView sin la
  «Navegación con teclado» de macOS (los eventos sintéticos no mueven el foco).
