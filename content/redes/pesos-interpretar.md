<!-- mltutor/viz/nn.py — expander "💡 Cómo interpretar esta pestaña" (cerrado) en 🧠 Análisis de Pesos. -->
Esta sección te ayuda a entender **qué está aprendiendo** la red:

• **Histograma de Pesos:** Muestra la distribución de las conexiones entre neuronas.
  - Distribución centrada en 0 y con forma aproximadamente normal suele indicar un entrenamiento saludable.
  - Pesos extremadamente grandes pueden provocar inestabilidad o sobreajuste.
  - Pesos casi todos muy cercanos a 0 pueden indicar que la red no aprendió (underfitting) o exceso de regularización.

• **Histograma de Sesgos:** Indica desplazamientos aprendidos por cada neurona.
  - Sesgos suelen estar más concentrados; valores extremos pueden saturar activaciones.

• **Líneas verticales:**
  - Línea gris: valor 0 (referencia neutra)
  - Línea verde discontinua: media actual

• **% Casi Cero:** Conexiones cuyo valor absoluto < 1e-3. Un porcentaje moderado (<60%) es normal; demasiado alto puede indicar red infraentrenada.
