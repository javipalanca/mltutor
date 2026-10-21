<!-- mltutor/apps/knn_app.py (viz "Distancias") — expander "ℹ️ Cómo interpretar este análisis" (cerrado), debajo de "### Análisis de Distancias entre Clases". {metrica} = model.metric tal cual ("minkowski", "euclidean" o "manhattan").
     Literal salvo las líneas marcadas aquí (cifras de tests/parity/knn/iris_k5_uniform_minkowski.json distance_analysis: entre clases 3,24 / 4,81 / 1,90; dentro de cada clase, within_class_mean_D67, 0,69 / 1,02 / 1,20; ratio 2,53):
     - CORREGIDO (D67 y rigor): Python: "- Valores más altos implican mayor separación entre las clases; valores bajos indican solapamiento o proximidad.". Una distancia media entre clases solo indica separación comparada con la distancia media DENTRO de cada clase (dos clases muy dispersas tienen una distancia media alta aunque se solapen). D67 muestra esa distancia en la diagonal.
     - NUEVO (D67): la línea "- La diagonal muestra la distancia media entre las muestras de una misma clase…" (en Python la diagonal valía 0 y no se explicaba).
     - CORREGIDO (rigor): Python: "- *Ratio Max/Min*: una medida heurística de separabilidad. En esta app usamos umbrales sencillos:". El ratio compara el par de clases más alejado con el más próximo: mide lo desiguales que son las distancias entre clases, no si las clases se solapan. En Iris da 2,53 («moderada») porque setosa está lejos de las otras dos, aunque versicolor y virginica se solapan.
     - NUEVO (D67): la línea "- Con dos clases solo hay una distancia…".
     - CORREGIDO (J2, revisión ML de F4): los umbrales del ratio (> 5 bien separadas, 2–5 moderada, ≤ 2 muy próximas) ya no deciden el veredicto; se sustituyen por la separación = min_{i≠j} M[i,j] / max(W_i, W_j) (≥ 2 separadas, ≥ 1 moderadas, < 1 próximas), igual en binaria y multiclase. Cifras con el split de la app (euclídea, K = 5, prueba 0,3, todas las filas de entrenamiento, scikit-learn 1.7.2): 🔢 Dígitos, ratio 1,35 y separación 1,11 (clases 1 y 8: 45,3 entre ellas frente a 41,0 y 38,2 dentro) con un 99,3 % de acierto en prueba; tres nubes de make_blobs(n_samples=300, centers=[[0,0],[10,0],[5,8.66]], cluster_std=1, random_state=42), ratio 1,02 y separación 5,77.
     - CORREGIDO (D62): Python: "- Escala las características (StandardScaler) antes de entrenar — KNN es sensible a la escala.". MLTutor no escala (D62); se dice dónde verlo y cómo hacerlo sin fuga de datos.
     - CORREGIDO (métricas): Python: "- Prueba otra métrica (Manhattan / Minkowski) en caso de que la forma de los clusters sea no circular.". Con p = 2 (el valor de la app), Minkowski es la euclídea: solo Manhattan es otra métrica.
     - NUEVO (D56): la frase final sobre las muestras usadas. -->
**¿Qué representa la matriz de distancias?**

- Cada celda (i, j) contiene la distancia promedio entre las muestras de la clase *i* y las de la clase *j* calculada usando la métrica seleccionada por el modelo (actualmente: **{metrica}**).
- Una distancia alta entre dos clases solo indica separación si es mayor que la distancia típica **dentro** de cada una: dos clases muy dispersas pueden estar lejos en promedio y aun así solaparse.
- La diagonal muestra la distancia media entre las muestras de una misma clase. Compárala con las celdas de su fila: en 🌸 Iris (euclídea, prueba 0,3), entre versicolor y virginica hay 1,90 de media, poco más que dentro de cada una (1,02 y 1,20): son las dos especies que pueden confundirse. Setosa está a 3,24 y 4,81 de ellas, con 0,69 dentro.

**Interpretación rápida de las estadísticas mostradas:**
- *Distancia Mínima*: el menor valor promedio observado entre pares de clases (ignorando la diagonal).
- *Distancia Máxima*: el mayor valor promedio observado.
- *Ratio Max/Min*: el cociente entre las dos anteriores; es el criterio de la versión original y se muestra solo como dato. Compara el par de clases más alejado con el más próximo, así que mide lo desiguales que son las distancias entre clases, no si las clases se solapan. Falla en los dos sentidos: en 🌸 Iris vale 2,53 aunque versicolor y virginica se solapan, y tres nubes de puntos bien separadas, colocadas en los vértices de un triángulo equilátero, dan un ratio de 1,02 («muy próximas» con los umbrales antiguos) porque todas están igual de lejos entre sí. Con dos clases solo hay una distancia entre clases y el ratio siempre vale 1.

**Separación: el criterio del veredicto.** Para cada par de clases se divide su distancia media por la dispersión de la más dispersa de las dos (su distancia media interna, la de la diagonal), y se toma el par peor:

$$\text{separación} = \min_{i \neq j} \frac{M_{ij}}{\max(W_i,\, W_j)}$$

donde $M_{ij}$ es la celda (i, j) de la matriz y $W_i$ la distancia media dentro de la clase *i*. Vale igual con dos clases que con diez:
- Separación ≥ 2 → 🟢 clases separadas: incluso el par más próximo está, en promedio, al menos el doble de lejos entre sí que dentro de cada clase. Las tres nubes del ejemplo dan 5,77.
- 1 ≤ Separación < 2 → 🟡 separación moderada: el par más próximo está más lejos entre sí que dentro de cada clase, pero sus nubes probablemente se tocan. En 🌸 Iris sale 1,58 (versicolor y virginica: 1,90 frente a 1,20).
- Separación < 1 → 🔴 clases muy próximas: la distancia media entre las dos clases es menor que la distancia media dentro de una de ellas; se solapan.

⚠️ **Es una medida global con promedios, no una predicción del acierto.** El KNN decide con los pocos vecinos más cercanos, no con medias: en 🔢 Dígitos la separación es 1,11 (🟡, por el 1 y el 8) y aun así, con $K = 5$, el modelo acierta el 99,3 % de las imágenes de prueba, porque cada dígito tiene cerca otros iguales aunque su clase esté muy dispersa. Para saber cuánto se confunden dos clases, mira la matriz de confusión de 📈 Evaluación.

**Sugerencias si las clases están próximas:**
- Escala las características antes de entrenar — KNN es sensible a la escala. MLTutor usa los datos sin escalar (ver «📏 KNN sin escalar» en 🏋️ Entrenamiento); fuera de la app, usa un `Pipeline` con `StandardScaler` para que el escalado se ajuste solo con el entrenamiento.
- Prueba la métrica Manhattan en caso de que la forma de los clusters sea no circular (en la app, Minkowski usa p = 2 y es la misma que la euclídea).
- Reduce o transforma características irrelevantes (PCA, selección de características).
- Aumenta el tamaño del conjunto de entrenamiento o prueba con más datos si es posible.

**Nota:** La matriz muestra promedios; examina además la distribución de distancias individuales si necesitas un diagnóstico más fino. Se calcula con las muestras de entrenamiento del modelo (en las clases muy numerosas, con una muestra de ellas, como indica la nota debajo de la matriz).
