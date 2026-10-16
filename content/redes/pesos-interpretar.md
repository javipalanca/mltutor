<!-- mltutor/viz/nn.py — expander "💡 Cómo interpretar esta pestaña" (cerrado) en 🧠 Análisis de Pesos. CORREGIDO en Rust (revisión de contenido de F5, 2026-10-16): (1) NUEVO: la viñeta del punto de partida (inicialización Glorot uniforme; con los valores por defecto el histograma sigue casi plano: en tests/parity/nn/exact/iris_default.json la std de la capa Oculta 1 es 0,42 frente a 0,38 de la inicialización, l = 0,65); (2) Python: "  - Pesos casi todos muy cercanos a 0 pueden indicar que la red no aprendió (underfitting) o exceso de regularización.". Una red que no aprende conserva los pesos iniciales, que no están cerca de 0; (3) Python: "• **% Casi Cero:** Conexiones cuyo valor absoluto < 1e-3. Un porcentaje moderado (<60%) es normal; demasiado alto puede indicar red infraentrenada.". Con Glorot casi ningún peso empieza por debajo de 1e-3 (0 % en Iris y 0,15 % en Cáncer en los fixtures): un 60 % sería extremo. -->
Esta sección te ayuda a entender **qué está aprendiendo** la red:

• **Histograma de Pesos:** Muestra la distribución de las conexiones entre neuronas.
  - Punto de partida: los pesos se sortean con la inicialización Glorot uniforme, al azar entre $-l$ y $l$ con $l = \sqrt{6/(\text{entradas} + \text{neuronas})}$ (en la primera capa de 🌸 Iris, $l \approx 0{,}65$). Si la red ha entrenado poco, el histograma sigue casi plano, como un rectángulo: no es un error, es que los pesos apenas se han movido.
  - Distribución centrada en 0 y con forma aproximadamente normal suele indicar un entrenamiento saludable.
  - Pesos extremadamente grandes pueden provocar inestabilidad o sobreajuste.
  - Pesos casi todos muy cercanos a 0 suelen deberse a una regularización fuerte (MLTutor no la usa). Una red que no aprende no tiene pesos cercanos a 0, sino los iniciales.

• **Histograma de Sesgos:** Indica desplazamientos aprendidos por cada neurona.
  - Sesgos suelen estar más concentrados; valores extremos pueden saturar activaciones.

• **Líneas verticales:**
  - Línea gris: valor 0 (referencia neutra)
  - Línea verde discontinua: media actual

• **% Casi Cero:** Conexiones cuyo valor absoluto < 1e-3. Lo normal es que sean casi ninguna (menos del 1 %); un porcentaje alto indica conexiones anuladas, por ejemplo por una regularización L1.
