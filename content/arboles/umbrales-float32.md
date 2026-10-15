<!-- NUEVO (D44, Fase 3): no existe en la versión Python. Pestaña 💾 Exportar, dentro del expander "Ver código Python del árbol", debajo del st.info "Información del código generado". Título (expander anidado o bloque plegable): mensajes.toml [exportar] umbrales_expander. Explica los umbrales de content/code/arbol_python_corregido.py.j2 (f64_switch_threshold, fase3-arboles.md §3 C15). Ejemplo verificado con scikit-learn 1.7.2 en tests/parity/trees/iris_gini_d3.json (umbral de la raíz 2.449999988079071). -->
En el código verás condiciones como `if petal_length_cm <= 2.4499999284744263:` donde el diagrama dice
`petal length (cm) <= 2.45`. No es un descuido: es lo que hace falta para que la función responda
**exactamente igual** que el modelo de scikit-learn.

**Por qué.** scikit-learn convierte los datos a `float32` (números con unas 7 cifras significativas) para
entrenar y para predecir: cada valor se pasa a `float32` antes de compararlo con el umbral. Tu función, en
cambio, recibe números normales de Python (`float64`, unas 16 cifras). Si se copiara el umbral tal cual,
algunos valores muy cercanos a él irían por la otra rama.

Ejemplo con 🌸 Iris: el umbral de la raíz es 2,449999988… (el punto medio, en `float32`, entre 1,9 y 3,0).
Una flor con el pétalo de **2,45 cm** exactos:

- scikit-learn la convierte a `float32` → 2,4500000477, que es **mayor** que el umbral → rama derecha, *no*
  es *setosa*.
- Con el umbral redondeado a 6 decimales, como hacía la versión anterior de MLTutor (`<= 2.450000`), la
  condición se cumple → rama izquierda → *setosa*. Una predicción distinta.

**Cómo se calcula.** Para cada umbral $t$, MLTutor busca el número `float64` más grande, $w$, tal que

$$x \le w \iff \text{float32}(x) \le t$$

para cualquier valor $x$. Es el punto exacto donde cambia la rama (en la raíz de Iris,
2,4499999284744263), y por eso la función no necesita ni numpy ni conversiones: compara números de Python
y va siempre por las mismas ramas que scikit-learn. Se ha comprobado con todas las muestras de
entrenamiento y de prueba de los datasets de la app.

⚠️ **Error típico:** redondear los umbrales «para que se lean mejor». Para leer el árbol usa 📝 Texto o el
diagrama; el código es para ejecutarlo.
