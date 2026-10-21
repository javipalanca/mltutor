# Comparación visual Qt ↔ Rust — Fase 5b (🧠 Redes Neuronales: 🏗️ Arquitectura y ⚙️ Entrenamiento)

Fecha: 2026-10-16. Capturas Qt en [`qt/fase5/`](qt/fase5/) (con las `NNN_web<k>.png` de Chrome para los lienzos
WebEngine, que en Qt salen en blanco) y Rust en [`rust/fase5/`](rust/fase5/), con los mismos números y nombres.
Se han hecho con el binario de release y `RealFacade`, así que los entrenamientos son del motor real
(`mltutor-nn`, semilla 42):

```sh
target/release/mltutor --screenshot rust/fase5/13_entrenado_iris.png --scenario redes_entrenado_iris --full-page
```

El escenario es el nombre Qt sin el número y con el prefijo `redes_` (`ui/src/lib/lessons/redes/scenarios.ts`).
Las `_ventana` se hacen sin `--full-page`. Cuantizadas a 256 colores con Pillow: 24 capturas, 4,8 MB. Como en
F1–F4, las `NNN.png` de Qt muestran solo el contenido (1135 px) y las de Rust la ventana entera a 1400 px.

**Nuevas:** `08b_arquitectura_5x256` (Dígitos con 5 × 256 neuronas: el diagrama resume las capas),
`13b_entrenado_iris_lr01` (LR 0,1 y 100 épocas, con la mejor época y las reducciones del LR marcadas) y
`13c_entrenamiento_cancelado` (California, 500 épocas, cancelado en vivo).
**Sin captura Rust en F5b:** `00_entrenamiento_sin_configurar` (D90: ya no existe ese estado), `01_datos_iris`
(pestaña 📊 de F1, sin cambios) y las pestañas 3–6 (F5c/F5d; hoy muestran «⚠️ Primero debes entrenar un
modelo.» o un aviso 🚧 si hay red; `00b_evaluacion_sin_entrenar`).

## Resumen

| Pantalla | Qt | Rust | Diferencias |
|---|---|---|---|
| 02/03 🏗️ Iris | controles, diagrama HTML (en blanco en Qt; `web1` en Chrome), resumen, tabla desplazada | mismos controles, textos y orden; diagrama SVG nativo con tooltip por capa y total de parámetros; tabla con una fila por capa (D76) | captions en Markdown (Qt mostraba los `**` literales); expander nuevo «¿Qué activación de salida…?» |
| 04 salida linear, 05 sigmoid multiclase | `st.error` / `st.warning` | el mismo aviso y, debajo, el `st.error` 🚫 de D71; el botón de ⚙️ y el del código quedan deshabilitados | D71 |
| 06 oculta sigmoid | `st.warning` | igual | — |
| 07 1 capa, 08 5 capas | sliders por capa; 5 capas: 7 columnas de nodos | igual; con 7 capas los nodos encogen para no solaparse | — |
| 08b 5 × 256 (nueva) | Python dibujaba 256 círculos apelotonados y 65 536 líneas | 12 + ⋮ + 12 nodos por capa con «+232», una ruta SVG por par de capas con opacidad decreciente; el tooltip dice «… y 232 neuronas más» | D80 |
| 09 código | código literal (con `X` sin definir) | `red_entrenamiento.py.j2` con la configuración de 🏗️ y ⚙️ (D74), descarga `red_neuronal_arquitectura_{tarea}.py`, nota y aviso si la configuración cambia después | D74 |
| 10/11/12 ⚙️ | LR, épocas, validación con la distribución errónea (96/30/24) | distribución real 96/24/30 (D73) con ayuda, actualizaciones por época (D88), nota y expander del escalado (D72), semilla en el expander avanzado (D70), expander de valores por defecto | contenido nuevo de `ml-lesson-author` |
| 13 entrenado Iris | «✅ ¡Entrenamiento completado!» + «🎉 …»; los pasos 1–4 no se llegan a ver | pasos 1–4 y barra por época en vivo, «Época n/N · pérdida · validación», botón «⏹️ Cancelar entrenamiento»; al terminar, los dos avisos de Python, la mejor época restaurada y la gráfica del historial (C27) | la gráfica en ⚙️ es NUEVA (Python solo la mostraba en 📈) |
| 13c cancelado (nueva) | — (Python: «Operación cancelada» en la barra de estado) | «⏹️ Entrenamiento cancelado en la época n de N…», la gráfica parcial y el modelo anterior intacto (D83) | — |
| 30 Cáncer tras softmax | el selector conserva «softmax» y el entrenamiento falla en inglés (`30b`) | la salida vuelve a «sigmoid» (D85) | — |
| 30c/31 Cáncer, 35/36 Diabetes | entrenamiento | igual, con los tamaños reales y el historial; Diabetes con R² negativo como en Python (D88) | — |
| 42 sigmoid multiclase al entrenar | error de TF en inglés y 📈 rota (`43`) | botón deshabilitado con el 🚫 de D71 | D71 |

## Diagrama de la red (C24)

Mismos colores (`#4ECDC4`, `#45B7D1`, `#FF6B6B`, bordes `#2C3E50`, conexiones `#BDC3C7`), fondo degradado,
etiquetas «Entrada/Oculta i/Salida (n)» y leyenda con las activaciones en mayúsculas. Diferencias deliberadas:
alto adaptado al número de nodos (300–460 px) para que no se solapen; tooltip por capa con neuronas y
parámetros (`[arquitectura] diagrama_tooltip_capa`); total de parámetros bajo la leyenda; capas de más de 24
neuronas resumidas; modo oscuro (fondo oscuro y texto claro). En Cáncer la entrada de 30 neuronas también se
resume (Python dibujaba las 30).

## Gráfica del historial (C27)

1×2 de Python (títulos de `[evaluacion]`, azul/rojo, altura 400, eje «Época»), con las épocas desde 1 (Python
usaba el índice desde 0), la mejor época restaurada (verde discontinua) y las reducciones del LR (morado
punteado) marcadas con su leyenda. En vivo el eje x abarca todas las épocas desde el principio.

## Diferencias que quedan

- Las captions (`st.caption`) se renderizan como Markdown; en Qt se veían los `**` literales.
- La pestaña 📊 Datos y las 3–6 no cambian en F5b.
- El código de 🏗️ se genera con la semilla 42 fija en el script (`keras.utils.set_random_seed(42)`): si el
  estudiante cambia la semilla de la app, el script no la sigue (la plantilla no tiene esa variable).
