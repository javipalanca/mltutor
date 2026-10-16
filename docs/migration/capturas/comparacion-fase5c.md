# Comparación visual Qt ↔ Rust — Fase 5c (🧠 Redes Neuronales: 📈 Evaluación y 🎯 Visualización)

Fecha: 2026-10-16. Capturas Qt en [`qt/fase5/`](qt/fase5/) (con las `NNN_web<k>.png` de Chrome para las gráficas
Plotly y el mapa de neuronas, que en Qt salen en blanco) y Rust en [`rust/fase5/`](rust/fase5/), con los mismos
números y nombres. Binario de release con `RealFacade` (motor `mltutor-nn`, semilla 42):

```sh
target/release/mltutor --screenshot rust/fase5/17_pesos_iris.png --scenario redes_pesos_iris --full-page
```

Escenario = nombre Qt sin el número, con el prefijo `redes_` (`ui/src/lib/lessons/redes/scenarios.ts`). Cuantizadas
a 256 colores con Pillow: 20 capturas, 3,7 MB. Las de Qt muestran solo el contenido (1135 px); las de Rust, la
ventana entera a 1400 px. Los entrenamientos de Qt no tienen semilla, así que las cifras difieren entre ambas
(y entre dos ejecuciones de Qt); la estructura y los textos son los que se comparan.

**Nuevas:** `18b_pesos_5x256` y `22b_activaciones_5x256` (Dígitos, 5 × 256: la red más grande), `43b_evaluacion_divergencia`
(Diabetes con tanh y SGD a LR 0,3: diverge en la época 7, K1) y `21_activaciones_iris_ventana`.

## Resumen

| Pantalla | Qt | Rust | Diferencias |
|---|---|---|---|
| 14/15 📈 Iris | «🎯 Precisión en Test», «📉 Pérdida en Test», historial Plotly (`web1`), C1 + C3 y el diagnóstico completo | mismo orden: resultados, «### 📈 Progreso del Entrenamiento» con C27, C1 (métricas, matriz, informe, muestras, precisión por clase, código), C3 e «🔍 Diagnóstico del Entrenamiento» con el árbol del core (checklist, puntuación, tendencias por fase, análisis comparativo, patrones, contexto, alertas y recomendaciones) | «Exactitud (accuracy)» (D86); el historial marca la mejor época restaurada y las reducciones del LR; banner del modelo (D91) |
| 16 📊 Historial | gráfica 1×2 y 4 tarjetas | igual; las tarjetas salen del core (`history_cards`) con los mismos formatos (`{:.6f}`, «Gap: … (…%)», «x% CV») | el eje x empieza en 1 (Python en 0) |
| 17/18 🧠 Pesos | captions «✅ Capa 2: (4, 10)…», tabla con «2, 4, 6», histogramas «Capa 1, 2, 3» | «Oculta 1», «Oculta 2», «Salida» en los tres sitios (D77); tabla, 4 métricas, veredicto, dos histogramas por capa (`#1f77b4`/`#ff7f0e`, línea gris en 0, media verde discontinua con «Media», `bargap` 0,02), pies de figura idénticos y «🩺 Evaluación General» | los bins los calcula Rust con el algoritmo de `nbinsx` de Plotly (K2): mismas barras, pero la respuesta no lleva los valores crudos |
| 18b 🧠 5 × 256 (nueva) | — | 6 capas densas, 281 000 pesos, histogramas en 22 ms (respuesta de 9 KB) | — |
| 19/20 🎯 Superficie Iris | matplotlib RdYlBu en el **espacio escalado** (226 × 246 celdas), puntos de prueba, «✅ Superficie generada exitosamente» | C8 en Plotly, **ejes en unidades originales** (D78), ≤ 200 × 200 celdas, puntos de prueba con los colores de clase de F2–F4, el aviso de éxito y la nota «Las demás características se fijan en su media…» | otra paleta (la de C8 en todas las lecciones); sin el código de la frontera (la red no se reentrena en 2D) |
| 33 Cáncer, 41 Moons | igual que Iris | igual que Iris; Moons sin aviso de modelo 2D | — |
| 39 🎯 Diabetes | viridis 100 niveles escalado a las **predicciones** (0,76–0,99 con los valores por defecto), puntos con su propia escala | la superficie con su escala («Valor predicho») y los puntos con la suya («Valor real»), dos barras de color | Python no tenía barra para los puntos |
| 21/22 📉 Activaciones | tabla «Capa 2, 4», aviso global, expanders «Capa 2 (dense_1)», mapa en canvas (`web1`) | mismos 10 controles, tabla con «Oculta i», aviso, expanders «Oculta 1 (dense_1)» (D77) con 4 métricas e histograma/medias, mapa en canvas HiDPI con las mismas opciones (dispersión, % de conexiones, radio adaptativo, subcolumnas, espaciado), 💀/😴 y tooltip con bias, w̄, σ, \|w\|max y estado | «Se usan las 30 muestras de prueba…» si hay menos filas que «Muestras» (K3) |
| 22b 📉 5 × 256 (nueva) | — | 1 354 neuronas en subcolumnas, unas 197 000 conexiones dibujadas (el 70 % de 281 000), 💀 en las capas profundas; 15 ms en el backend, 89 KB | — |
| 32 📈 Cáncer | binaria | igual | — |
| 37 📈 Diabetes | «📉 Error en Test», R² −3,4 y el diagnóstico «✅ Error de regresión aceptable» | «📉 Error en Test», C2 con R² negativo, el aviso NUEVO de R² negativo (D84) con el expander «¿Por qué la red acierta poco con los valores por defecto?» y el diagnóstico con los textos de regresión corregidos (`convergio_regresion`, `salud_bueno_regresion`, `estado_bueno_regresion`) | D84 |
| 38 📊 Diabetes | MSE / MAE | igual | — |
| 39b 📉 Diabetes | tabla y mapa | igual | — |
| 43b divergencia (nueva) | (Python: métricas `nan` y diagnóstico con NaN) | aviso 🚨 con la época de la divergencia, el historial para verla y nada más; 🎯 bloquea Pesos, Superficie y Activaciones con el mismo aviso | K1 |

## Diferencias que quedan

- **Mapa de neuronas:** el hash de Python (`(h ^ (h >> 13)) * 1274126177` en dobles de JavaScript) nunca pasaba de 0,5,
  así que «% conexiones dibujadas» ≥ 50 dibujaba todas las líneas (el valor por defecto es 70) y la «dispersión»
  movía siempre los nodos a la izquierda. Rust usa la misma mezcla en enteros de 32 bits (`Math.imul`): el 70 %
  dibuja el 70 % y la dispersión es simétrica. Además, con «Máx subcolumnas» Python limitaba las columnas pero no las
  neuronas por columna y las que no cabían salían en columnas extra; Rust las reparte en las permitidas.
- **Histogramas de activaciones:** color `#636efa` (el de Plotly por defecto, como en Python), sin título de ejes.
- **Superficie:** paleta y estilo de C8 (Plotly) en lugar de matplotlib RdYlBu; con 2 características (Moons) siguen
  los dos selectores, como en las otras lecciones.
- **Tarjetas de 📊 Historial y métricas del diagnóstico:** el `delta` sale en gris, como en Qt (Streamlit lo coloreaba).
- **Interpretación (C3):** dice «Tu modelo de regresión lineal…» también para la red, igual que Python.
- **Modo claro/oscuro:** el mapa y los histogramas siguen el tema; las capturas son en modo claro.
