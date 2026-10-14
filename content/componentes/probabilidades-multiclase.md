<!-- viz/roc.py::plot_threshold_analysis (multiclase), tras el st.info de nota. Expander cerrado. Título del expander: "ℹ️ ¿Cómo interpretar la Distribución de Probabilidades Multiclase?" -->
La distribución de probabilidades para cada clase muestra la confianza del modelo al predecir la clase a la que pertenece una muestra. 
En un escenario de clasificación multiclase, el modelo asigna una probabilidad a cada clase posible, y la clase predicha suele ser la de mayor probabilidad.

**Interpretación:**
- Cada barra en cada histograma representa la frecuencia de las probabilidades predichas para una clase específica.
- Un modelo bien calibrado tendrá una alta concentración de muestras en los extremos (cerca de 0 o 1) para la clase correcta.
- Distribuciones más planas o con picos intermedios sugieren que el modelo presenta incertidumbre o confusión entre clases.

**Consejos:**
- Presta atención a las clases con solapamiento en sus distribuciones, ya que esto puede indicar áreas de incertidumbre.
- Considera ajustar el umbral de decisión para mejorar la precisión en clases específicas.
