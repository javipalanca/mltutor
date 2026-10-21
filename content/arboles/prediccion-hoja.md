<!-- NUEVO (D46, Fase 3): no existe en la versión Python, que no mostraba probabilidades ni la hoja. Pestaña 🔮 Predicciones, expander cerrado debajo de "### Camino de decisión" (y de la tabla de probabilidades de la hoja en clasificación). Título: mensajes.toml [predicciones] hoja_expander. Hojas de tests/parity/trees/iris_gini_d3.json (nodos 7 y 8). -->
**El camino de decisión** es la lista de preguntas que ha respondido tu muestra, de la raíz a la hoja.
Cada línea dice el valor de tu muestra, el umbral y por qué rama ha seguido: `≤` es la rama izquierda
(se cumple la condición) y `>` la derecha. Las características que no aparecen en el camino **no han
influido** en esta predicción, aunque el árbol las use en otras ramas.

**La predicción es la respuesta de la hoja.** Todas las muestras que llegan a la misma hoja reciben
exactamente la misma predicción:

- En **clasificación**, la hoja guarda cuántas muestras de entrenamiento de cada clase llegaron a ella. La
  **probabilidad** de cada clase es esa proporción, y la clase predicha es la más frecuente.
- En **regresión**, el valor predicho es la media de las muestras de entrenamiento de la hoja (la mediana
  con `absolute_error`), así que el árbol nunca predice un valor fuera del rango visto al entrenar.

Ejemplo con 🌸 Iris (gini, profundidad 3): una flor con pétalos de más de 4,75 cm de largo y de más de
1,75 cm de ancho llega a una hoja con 1 *versicolor* y 32 *virginica*: predice *virginica* con
probabilidad $32/33 = 0{,}970$. Si el ancho es de 1,75 cm o menos, llega a otra con 4 y 4: **empate**,
probabilidad 0,5 para cada una, y el árbol elige *versicolor* solo porque va antes en la lista de clases.

⚠️ **Errores típicos**

- **Leer la probabilidad como certeza.** Un 1,000 significa que *todas* las muestras de entrenamiento de
  esa hoja eran de esa clase, no que el modelo no pueda fallar. Si la hoja tiene 2 muestras, ese 1,000 se
  basa en dos ejemplos. Mira siempre cuántas muestras tiene la hoja.
- **Esperar probabilidades finas.** Un árbol solo tiene tantas probabilidades distintas como hojas; por eso
  sus curvas ROC (📉 Visualización) tienen pocos escalones y el umbral de decisión apenas se puede afinar.
- **Comparar valores muy cercanos al umbral con la calculadora.** scikit-learn convierte cada valor a
  `float32` antes de compararlo: una flor con el pétalo de 2,45 cm exactos va por la rama de la **derecha**
  de `petal length (cm) <= 2.45`, porque el umbral real es 2,44999999 y, en `float32`, 2,45 es un poco
  mayor. MLTutor hace la misma conversión.
