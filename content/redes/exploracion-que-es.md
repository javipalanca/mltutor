<!-- mltutor/viz/nn.py — INALCANZABLE en Python (viz/nn.py::show_neural_network_predictions, pestaña "🎲 Exploración Interactiva"); se porta como NUEVO por D81. Expander "ℹ️ ¿Qué es la Exploración Interactiva?" (cerrado). CORREGIDO en Rust (revisión de contenido de F5, 2026-10-16): la gráfica muestra la respuesta del MODELO al variar una característica en UNA muestra (una curva ICE), no una correlación de los datos. Python: "- **Líneas ascendentes**: La característica tiene correlación positiva", "- **Líneas descendentes**: La característica tiene correlación negativa", "- **Líneas planas**: La característica tiene poco impacto" y "**La exploración interactiva** te permite entender cómo el modelo neural toma decisiones:" ("neural" → "neuronal"). NUEVO: la viñeta sobre combinaciones irreales. NUEVO (revisión de F5d, 2026-10-16): los tres bloques finales («Una muestra, una curva», «No es la importancia de la característica» y «Fuera de los datos, la red extrapola»). Comportamiento (Python, viz/nn.py y C29): 50 valores equiespaciados entre el mínimo y el máximo de la característica en TODO el dataset, las demás características fijas en los valores de la muestra base, cada fila escalada con el escalador del entrenamiento antes de pasar por la red. Cifras del ejemplo de extrapolación: pesos finales de tests/parity/nn/exact/iris_default.json en NumPy, entrada [5.8, 3.0, 20, 1.2] → [0.000, 0.004, 0.996]; con pétalo 6,9 → [0.137, 0.245, 0.618]; la probabilidad máxima de virginica en las 150 flores reales es 0,790 (de cualquier clase, 0,820). -->
**La exploración interactiva** te permite entender cómo el modelo neuronal toma decisiones:

🔍 **¿Para qué sirve?**
- Ver cómo cada característica influye en las predicciones
- Identificar patrones y comportamientos del modelo
- Detectar posibles sesgos o comportamientos inesperados
- Comprender la sensibilidad del modelo a cambios en los datos

📊 **¿Cómo interpretar los resultados?**
- **Líneas ascendentes**: Para esta muestra, el modelo predice más al aumentar la característica
- **Líneas descendentes**: Para esta muestra, el modelo predice menos al aumentar la característica  
- **Líneas planas**: Con los demás valores de esta muestra, la característica apenas cambia la predicción (con otra muestra puede ser distinto)
- **Cambios abruptos**: Puntos de decisión críticos del modelo

💡 **Consejos de uso:**
- Prueba diferentes muestras base para ver patrones generales
- Observa qué características causan mayores cambios
- Busca comportamientos inesperados o poco realistas
- Recuerda que se cambia una sola característica y las demás se quedan fijas: algunas combinaciones (un pétalo muy largo y muy estrecho) no existen en los datos, y ahí la red extrapola. Describe al modelo, no a la realidad

📍 **Una muestra, una curva.** La gráfica es una **curva *ceteris paribus*** («lo demás, igual»; también se
llama **curva ICE**, de *Individual Conditional Expectation*): MLTutor toma la muestra base, cambia solo la
característica elegida por 50 valores entre su mínimo y su máximo en el dataset, deja las demás tal cual,
escala cada fila igual que al entrenar y anota la salida de la red. Describe **cómo responde la red alrededor
de esa muestra**, no cómo se comportan los datos: en una red, el efecto de una característica depende de los
valores de las demás, así que con otra muestra base la curva puede tener otra forma o incluso ir en sentido
contrario. Si promedias las curvas de muchas muestras obtienes la *dependencia parcial* (PDP), que sí resume el
comportamiento medio del modelo.

⚖️ **No es la importancia de la característica.** Que la curva sea plana para esta muestra no significa que la
red ignore esa característica (puede ser decisiva para otras muestras), y una curva muy empinada en una zona
donde casi no hay datos importa poco en la práctica. La importancia mide el efecto en **todo** el conjunto de
prueba: por ejemplo, la importancia por permutación (la de 🌲 Árboles de Decisión y 🔍 KNN) desordena una
columna y mide cuánto empeora la métrica. Las etiquetas «muy influyente», «moderada» o «poco influyente» del
análisis se refieren solo a esta muestra.

🚧 **Fuera de los datos, la red extrapola.** Cada valor del eje está dentro del rango de su característica,
pero la **combinación** puede no existir: en 🌸 Iris, un pétalo de 6,9 cm de largo con 0,1 cm de ancho no se
parece a ninguna flor. Ahí la red no ha visto ejemplos y responde con lo que dicte su forma: con ReLU es
lineal a trozos y, lejos de los datos, sigue la última tendencia sin límite (en regresión) o se vuelve cada
vez más segura de una clase (en clasificación); con tanh o sigmoid tiende a quedarse plana. Una probabilidad
alta en esa zona **no** indica que la predicción sea fiable. Lo mismo pasa si escribes en 🔮 valores fuera del
rango de entrenamiento: con una red de 🌸 Iris entrenada con la configuración por defecto, una flor con
sépalo de 5,8 × 3,0 cm y pétalo de 20 × 1,2 cm (el pétalo más largo de los datos mide 6,9 cm) sale *virginica*
con un 99,6 % de probabilidad, más segura que con cualquier flor real (los números exactos dependen de tu
entrenamiento).
