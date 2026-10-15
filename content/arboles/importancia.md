<!-- NUEVO (D38, Fase 3): no existe en la versión Python, que solo mostraba la importancia nativa. Pestaña 🔍 Características, expander cerrado debajo del panel "🌳 Importancia del modelo" / "🔀 Importancia por permutación" (en las dos subvistas, después de "📖 Cómo interpretar estos resultados"). Título: mensajes.toml [caracteristicas] comparar_expander. Cifras de tests/parity/trees/{breast_cancer_gini_d3,diabetes_squared_error_d10,iris_gini_d3}.json (feature_importances y permutation_importance, n_repeats=10, random_state=42). -->
MLTutor calcula la importancia de dos maneras, y a menudo **no coinciden**. No es un error: responden a
preguntas distintas.

**🌳 Importancia del modelo (nativa)** — *¿cuánto ha usado el árbol cada característica al aprender?*
Para cada nodo que pregunta por una característica se suma cuánto reduce la impureza, ponderado por las
muestras que pasan por él:

$$\text{imp}(f) \propto \sum_{\text{nodos que usan } f} \left( n \cdot I - n_{izq} \cdot I_{izq} - n_{der} \cdot I_{der} \right)$$

y al final se normaliza para que la suma sea 1. Se calcula con los datos de **entrenamiento**, gratis y
al instante, pero tiene dos sesgos:

- **Favorece las características con muchos valores distintos.** Una variable continua (edad, tarifa, una
  medida) ofrece cientos de umbrales posibles y una binaria solo uno, así que la continua tiene muchas más
  ocasiones de encontrar, por casualidad, un corte que reduzca la impureza **en entrenamiento**. En los
  árboles profundos esos cortes casuales se acumulan: la característica parece importante aunque solo
  esté ajustando ruido.
- **Reparte mal entre características parecidas.** Si dos están muy correlacionadas, el árbol usa una y la
  otra ya no aporta nada, así que su importancia es casi 0 aunque contenga la misma información.

**🔀 Importancia por permutación** — *¿cuánto empeora el modelo si le quito la información de una
característica?* Con los datos de **prueba**, se barajan al azar los valores de una columna (así deja de
estar relacionada con el objetivo, pero conserva su distribución) y se mide cuánto baja la exactitud
(clasificación) o el R² (regresión). Se repite 10 veces y se da la media y la desviación estándar.
Es más lenta, pero mide lo que importa: la dependencia del modelo con datos que no ha visto.

**Ejemplos con los datasets de la app**

- 🔬 **Cáncer** (gini, profundidad 3): la nativa da 0,810 a *mean concave points*, la característica de la
  raíz. Por permutación, en cambio, las que más hacen caer la exactitud son *worst area* (0,153),
  *worst radius* (0,142) y *area error* (0,134); *mean concave points* se queda en 0,075. La pregunta de
  la raíz se lleva mucha importancia nativa porque actúa sobre todas las muestras de entrenamiento, pero
  eso no dice cuánto se equivoca el modelo sin ella en la prueba.
- 🩺 **Diabetes** (squared_error, profundidad 10, un árbol sobreajustado): *s1* tiene una importancia nativa
  de 0,067, pero por permutación sale **negativa** (−0,050): al barajarla, el modelo **mejora** en prueba.
  Sus cortes solo servían para memorizar el entrenamiento. Lo mismo pasa con *s4* y *s3*.
- 🌸 **Iris** (gini, profundidad 3): *petal length* (0,925) y *petal width* (0,075) empatan en la raíz y el
  sorteo eligió la primera. Por permutación quedan mucho más igualadas (0,520 y 0,300).

**Cuál usar**

- Para entender **cómo está construido** el árbol, la nativa.
- Para decidir **qué características necesita el modelo** (o cuáles podrías quitar), la de permutación, y
  mejor con un árbol que no esté sobreajustado.
- Si las dos coinciden, puedes fiarte bastante del orden; si no, mira si hay características correlacionadas
  o si el árbol es demasiado profundo.

⚠️ **Errores típicos**

- **Leer la importancia como causa.** Que el modelo dependa de una característica no significa que esa
  característica *cause* el resultado.
- **Quitar una característica porque su importancia es 0.** Puede valer 0 porque otra correlacionada hizo
  su papel; si quitas la otra, esta pasa a ser imprescindible. Comprueba siempre el efecto reentrenando.
- **Comparar importancias de modelos distintos.** La nativa suma 1 (salvo en un árbol de una sola hoja, en el
  que todas valen 0), así que solo indica el reparto dentro de un árbol; la de permutación depende de la métrica y del conjunto de prueba.
