<!-- NUEVO (D41, Fase 3): no existe en la versión Python (plot_tree sin explicación). Pestaña 📉 Visualización → 🌲 Árbol, expander cerrado debajo del diagrama y de mensajes.toml [visualizacion] arbol_ayuda. Título: [visualizacion] leer_arbol_expander. Las líneas de cada caja son las de plot_tree(filled=True, proportion=True, impurity=True), fase3-arboles.md §3 C13. Ejemplo: tests/parity/trees/iris_gini_d3.json. -->
Cada caja es un **nodo**. La de arriba es la **raíz**, con todas las muestras de entrenamiento; las cajas
sin flechas debajo son las **hojas**, donde el árbol da su respuesta. Las líneas de una caja son:

1. **La pregunta** (solo en los nodos internos), por ejemplo `petal length (cm) <= 2.45`. Las muestras que
   cumplen la condición bajan por la flecha de la **izquierda** (*True*) y el resto por la de la
   **derecha** (*False*). El umbral es el punto medio entre dos valores consecutivos de entrenamiento.
2. **La impureza** del nodo con el criterio elegido (`gini = 0.664`, `entropy = …`, `squared_error = …`):
   0 significa que todas sus muestras son de la misma clase o tienen el mismo valor.
3. **`samples`**: el porcentaje de las muestras de entrenamiento que llegan a ese nodo (100 % en la raíz).
4. **`value`**: en clasificación, la **proporción de cada clase** en el nodo, en el orden de las clases
   (`[0.0, 0.5, 0.5]`); en regresión, el **valor que predice**: la media de sus muestras (la mediana con
   `absolute_error`).
5. **`class`** (solo clasificación): la clase mayoritaria, la que predice el nodo. Si hay empate, la
   primera en el orden de las clases.

**El color** indica la clase mayoritaria (un color por clase) y su **intensidad**, cuánto domina esa clase: cuanto
más pura es la caja, más intenso es el color; si las dos clases principales empatan, la caja sale blanca
(la raíz de 🌸 Iris, con 37 *versicolor* y 37 *virginica*, es blanca). En regresión hay un solo color,
más intenso cuanto mayor es el valor predicho.

**Para predecir** una muestra nueva, empieza en la raíz, responde a cada pregunta y sigue la flecha hasta
llegar a una hoja. 🔮 Predicciones hace ese recorrido por ti y lo muestra en **Camino de decisión**.

Pasa el ratón por una caja para ver sus datos completos, como el número exacto de muestras de cada clase.

⚠️ **Errores típicos**

- **Leer la profundidad como importancia.** Que una característica salga en la raíz no la convierte en la
  única importante: otra muy parecida pudo empatar con ella (🔍 Características lo matiza).
- **Fiarse de una hoja con pocas muestras.** Una hoja con `samples = 1.0%` y `value = [0.0, 1.0, 0.0]`
  parece muy segura, pero se apoya en una o dos flores: es una señal típica de sobreajuste.
