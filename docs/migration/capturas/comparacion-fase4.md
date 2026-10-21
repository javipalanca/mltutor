# Comparación visual Qt ↔ Rust — Fase 4 (🔍 K-Nearest Neighbors)

Fecha: 2026-10-15. Capturas Qt en [`qt/fase4/`](qt/fase4/) (`tools/capture_qt.py --phase 4`) y Rust en
[`rust/fase4/`](rust/fase4/), con los mismos números y nombres. Las de Rust se han hecho con el binario de
release y `RealFacade`, así que los cálculos son de `mltutor-core::knn` (kd-tree exacto, D52):

```sh
target/release/mltutor --screenshot rust/fase4/09d_interactiva_iris_con_puntos.png --scenario knn_interactiva_iris_con_puntos --full-page
```

El escenario es el nombre Qt sin el número y con el prefijo `knn_` (`ui/src/lib/lessons/knn/scenarios.ts`).
Las capturas `_ventana` se hacen sin `--full-page`. Están cuantizadas a 256 colores con Pillow: 51 capturas,
8,7 MB. Como en F1–F3, las `NNN.png` de Qt muestran solo el contenido (1135 px) y las de Rust, la ventana
entera a 1400 px con la barra lateral.

**Sin equivalente en Rust:** `20_exportar_pickle` (D57: no hay pickle directo; el paquete Python lo
reconstruye). Los diálogos nativos de guardar no se capturan.

**Nuevas:**
- `09d_interactiva_iris_con_puntos`: el lienzo nativo con los tres puntos de la captura de Chrome
  `qt/fase4/09d_interactiva_iris_lienzo_con_puntos.png` (en Qt el lienzo sale en blanco, D65);
- `25b_interactiva_cancer` (lienzo en binaria);
- `31b_prediccion_resultado_k20_distance_manhattan` (pesos por distancia en la tabla de vecinos).

## Resumen

| Pantalla | Qt | Rust | Diferencias |
|---|---|---|---|
| 00, 00b, 00c sin entrenar | `st.info` | igual | — |
| 01/02 🏋️ Entrenamiento | 4 controles; la caja «Información del Dataset» desaparece tras entrenar | 4 controles en una fila (K con `− +`), info siempre (D60), nota «sin escalar» y 3 expanders nuevos (parámetros, escala, empates) | texto nuevo del autor de contenido |
| 03/04 📈 Evaluación | C1 + C3 | igual (componentes de F2) | — |
| 05 📊 Probabilidades | «Probabilidades para Clase 0» | nombres de clase (D61) | — |
| 06 📉 ROC | «Clase 0 (AUC = …)» | nombres de clase (D61) | — |
| 07/08 🌈 Frontera | matplotlib, todas las filas | Plotly, todas las filas y `knn_step(n)` (D63); celdas idénticas a sklearn | aviso 2D de F2 debajo |
| 09/10 🎮 Interactiva | lienzo HTML **en blanco** en Qt; en Chrome: sin ejes, JS propio | lienzo nativo Plotly: ejes con nombre, leyenda, duplicados ×n, clic/arrastre, valores exactos, tabla y métricas de puntos (D54), K 1–20 (D55) | ver abajo |
| 11 misma característica | `st.warning` | igual | — |
| 12/13 📏 Distancias | heatmap viridis con la diagonal a 0 y texto blanco | heatmap Plotly; diagonal = distancia media dentro de cada clase en cursiva (D67); texto negro o blanco según la celda; nota con las muestras usadas (D56) | métricas idénticas (1.903 / 4.808 / 2.53) |
| 14/15 🔍 Características | permutación | igual (C10) con el texto propio de KNN y la nota de escala | — |
| 16/17 🔮 Predicciones | tarjeta + «Información del modelo KNN» | igual + probabilidades + tabla de los K vecinos + expander (D58) | — |
| 18/19 💾 Exportar | «Generar Código» (fragmento roto de Regresión) + pickle | 📄 Código de Entrenamiento, 💾 ONNX, 📦 Paquete (D57) | — |
| 21–26 Cáncer | ratio 1,00 → 🔴 siempre | D67: `st.info` de binaria y diagonal con las distancias internas (802 / 246) | — |
| 27/28 Titanic | formulario con selects | igual; tabla de vecinos con desplazamiento horizontal (15 columnas) | — |
| 29–31 K20 distance manhattan | deslizador del lienzo en 15 con K = 20 | deslizador en 20 | D55 |
| 32–41 Diabetes | sin botones de tipo, C2, C7, superficie age/sex | igual; la superficie age/sex coincide celda a celda (5 340 celdas sensibles a empates) | — |
| 42/43 California | paso 0,1, permutación 0,9 s | igual; permutación 125 ms | — |

## 🎮 Lienzo interactivo (C18)

Rust conserva lo que se ve en el HTML de Python (caja azul, fila de controles con «Valor K:», Métrica,
Pesos, los dos checkboxes y el botón rojo «🗑️ Limpiar Puntos», fondo degradado `#f5f7fa → #c3cfe2`, paleta
`colors[código % 8]`, borde `#333`, estrella dorada, líneas discontinuas `#FF4444`, arco verde de confianza,
«NN%» y el nombre de la clase en negrita, tarjetas «Puntos de Prueba», «Confianza Promedio» y «Valor K
Actual») y añade:

- ejes con los nombres de las características y rejilla suave; leyenda de Plotly («Leyenda:», un punto por
  clase y el rombo dorado de «Puntos de Prueba»);
- clic para añadir, **arrastre** para mover (las líneas siguen al punto), tooltip «Vecino n: clase,
  distancia d» y borde rojo grueso en los vecinos; las líneas aparecen una a una (60 ms) al soltar;
- «✏️ Añadir un punto con valores exactos» y la tabla «📊 Análisis de Puntos de Prueba», editable, con las
  tres métricas de Python, que en Python eran inalcanzables;
- cálculo en Rust con las reglas de sklearn (en Python lo hacía el JS con su propia regla de distancia 0).

Diferencias que quedan:
- Los vecinos muy cercanos quedan ocultos bajo el anillo del punto de prueba cuando hay muchos duplicados
  (Iris en 2D); el tooltip los lista.
- El área de datos usa `min − 1 … max + 1` como Python, así que en Iris la nube ocupa la parte central.
- En modo oscuro el degradado pasa a `#1f2733 → #2c3a52` (Python solo tenía modo claro).

## Medidas (Mac arm64, release)

- `knn_playground`: 0,0–0,1 ms en el backend con 3 y con 20 puntos (incluye el ajuste del KNN 2D sobre las
  150 filas); ida y vuelta por IPC medida en la interfaz ≤ 1 ms (p95; WebKit redondea `performance.now()` a
  1 ms). Al arrastrar se envía una petición por fotograma y se cancela la anterior.
- Arrastre simulado en el smoke test (eventos de puntero reales sobre el gráfico, 20 movimientos): de mover
  el puntero a terminar el redibujado de Plotly, mediana 22–31 ms y máximo 38 ms (2 de 3 ejecuciones). En
  la tercera, con la ventana en segundo plano, WebKit retrasa los temporizadores a 1 s y salió 1 s: no es
  representativo de una ventana visible.
- Entrenar: 0–2 ms (California, kd-tree de 14 448 filas). Frontera de Iris (40 000 celdas): 10,7 ms;
  superficie de California: 20 ms. Permutación de California: 125 ms (sklearn 0,9 s). Distancias de Cáncer:
  0,8 ms.
- Smoke test completo (13 comprobaciones): 37 s. `.app` de 15,8 MiB.
