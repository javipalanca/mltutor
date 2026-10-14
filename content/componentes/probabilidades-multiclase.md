<!-- viz/roc.py::plot_threshold_analysis (multiclase), tras el st.info de nota. Expander cerrado. Título del expander: "ℹ️ ¿Cómo interpretar la Distribución de Probabilidades Multiclase?". CORREGIDO en Rust (2026-10-14): "Un modelo bien calibrado tendrá … en los extremos" confundía calibración con seguridad: un modelo está calibrado si, de las muestras a las que asigna un 70 % a una clase, en torno al 70 % son de esa clase, aunque sus probabilidades no estén cerca de 0 o 1. -->
La distribución de probabilidades para cada clase muestra la confianza del modelo al predecir la clase a la que pertenece una muestra. 
En un escenario de clasificación multiclase, el modelo asigna una probabilidad a cada clase posible, y la clase predicha suele ser la de mayor probabilidad.

**Interpretación:**
- Cada barra en cada histograma representa la frecuencia de las probabilidades predichas para una clase específica.
- Un modelo que separa bien las clases concentra las muestras en los extremos (cerca de 0 o 1). Que esté seguro no significa que acierte: un modelo está **bien calibrado** si, de las muestras a las que asigna un 70 % a una clase, alrededor del 70 % son realmente de esa clase.
- Distribuciones más planas o con picos intermedios sugieren que el modelo presenta incertidumbre o confusión entre clases.

**Consejos:**
- Presta atención a las clases con solapamiento en sus distribuciones, ya que esto puede indicar áreas de incertidumbre.
- Considera ajustar el umbral de decisión para mejorar la precisión en clases específicas.
