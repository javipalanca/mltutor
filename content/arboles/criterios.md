<!-- NUEVO (Fase 3): no existe en la versión Python, que solo tenía la ayuda corta del selector ("Medida de la calidad de una división..."). Pestaña 🏋️ Entrenamiento, expander cerrado debajo de "### Parámetros del Modelo". Título: mensajes.toml [entrenamiento] criterios_expander. Números comprobados con scikit-learn 1.7.2 y los fixtures tests/parity/trees/iris_gini_d3.json y diabetes_*.json (split de la app, random_state=42). CORREGIDO (revisión ML de F3, 2026-10-15): «sale el mismo árbol» pasa a «salvo empates numéricos»: en sklearn friedman_mse y squared_error difieren en 10 de 60 árboles con muchos empates; en Diabetes e Iris coinciden (MEMORY.md). -->
Un árbol aprende haciendo **preguntas de sí o no** sobre una característica: «¿largo del pétalo ≤ 2,45 cm?».
Cada pregunta divide las muestras de un **nodo** en dos grupos (los hijos izquierdo y derecho). Para elegir la
pregunta, el árbol prueba todas las características y todos los umbrales posibles y se queda con la que deja
los dos grupos más **puros**, es decir, más homogéneos. El **criterio de división** es la forma de medir esa
pureza.

**Clasificación: impureza de un nodo**

Si en un nodo la proporción de muestras de la clase $k$ es $p_k$:

- **Gini** (`gini`): $G = 1 - \sum_k p_k^2$. Es la probabilidad de equivocarse si se etiqueta una muestra
  al azar según las proporciones del nodo. Vale 0 si todas las muestras son de la misma clase y como
  máximo $1 - 1/K$ con $K$ clases repartidas a partes iguales (0,5 con dos clases y 0,667 con tres).
- **Entropía** (`entropy`): $H = -\sum_k p_k \log_2 p_k$. Mide, en bits, la incertidumbre sobre la clase.
  Vale 0 en un nodo puro y como máximo $\log_2 K$ (1 bit con dos clases y 1,585 con tres).

Ejemplo con 🌸 Iris: la raíz tiene las 105 flores de entrenamiento, 31 *setosa*, 37 *versicolor* y
37 *virginica*:

$$G = 1 - \left(\tfrac{31}{105}\right)^2 - 2\left(\tfrac{37}{105}\right)^2 = 0{,}664 \qquad H = 1{,}580 \text{ bits}$$

**Cómo se elige la división**

Se calcula la impureza media de los dos hijos, ponderada por su número de muestras, y se elige la división
que más la **reduce** respecto al padre:

$$\Delta = I(\text{padre}) - \frac{n_{izq}}{n}\, I(\text{izq}) - \frac{n_{der}}{n}\, I(\text{der})$$

Con la entropía, esta reducción se llama **ganancia de información**. En Iris, «largo del pétalo ≤ 2,45»
deja a la izquierda las 31 *setosa* (nodo puro, impureza 0) y a la derecha 37 + 37 flores (Gini 0,5;
entropía 1 bit):

- Gini: $0{,}664 - \tfrac{74}{105} \cdot 0{,}5 = 0{,}312$
- Ganancia de información: $1{,}580 - \tfrac{74}{105} \cdot 1 = 0{,}875$ bits

Gini y entropía casi siempre eligen las mismas divisiones. Gini se calcula un poco más deprisa (no usa logaritmos).
Prueba los dos con 🍷 Vino y compara los árboles.

**Regresión: error de un nodo**

En regresión la hoja no predice una clase sino un número, y la impureza es el error de esa predicción con
las muestras de entrenamiento del nodo:

- **`squared_error`** (error cuadrático medio, MSE): la hoja predice la **media** $\bar y$ y la impureza es
  $\frac{1}{n}\sum (y_i - \bar y)^2$, la **varianza** de $y$ en el nodo. En 🩺 Diabetes, la raíz tiene
  309 pacientes, media 153,9 y varianza 6148,3; el primer corte, `bmi ≤ 0,005`, es el que más la reduce.
- **`friedman_mse`**: misma impureza (MSE), pero cada división se puntúa con la fórmula de Friedman
  $\frac{n_{izq}\, n_{der}}{n}\,(\bar y_{izq} - \bar y_{der})^2$, pensada para el *gradient boosting*. En un
  solo árbol esa puntuación es exactamente la reducción del error cuadrático, así que en la práctica sale
  **el mismo árbol** que con `squared_error`, salvo empates numéricos: si dos cortes empatan, el redondeo
  de cada fórmula puede decidir otro (compruébalo en Diabetes con cualquier profundidad: coinciden).
- **`absolute_error`** (error absoluto medio, MAE): la hoja predice la **mediana** y la impureza es
  $\frac{1}{n}\sum |y_i - \text{mediana}|$. Le afectan menos los valores extremos (*outliers*), pero es
  bastante más lento de entrenar. En Diabetes, la raíz tiene mediana 142 y error absoluto medio 65,9.

⚠️ **Errores típicos**

- **Pensar que una impureza baja en entrenamiento es buena señal.** Casi siempre se puede llegar a hojas puras
  (impureza 0) dividiendo hasta tener una muestra por hoja: el árbol memoriza los datos. Lo que importa es
  el resultado en el conjunto de prueba (📈 Evaluación).
- **Comparar impurezas de criterios distintos.** Un Gini de 0,3 y una entropía de 0,3 no significan lo
  mismo; tampoco se pueden comparar los MSE de dos datasets con escalas diferentes.
