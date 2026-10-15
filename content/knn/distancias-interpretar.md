<!-- mltutor/apps/knn_app.py (viz "Distancias") — expander "ℹ️ Cómo interpretar este análisis" (cerrado), debajo de "### Análisis de Distancias entre Clases". {metrica} = model.metric tal cual ("minkowski", "euclidean" o "manhattan").
     Literal salvo las líneas marcadas aquí (cifras de tests/parity/knn/iris_k5_uniform_minkowski.json distance_analysis: entre clases 3,24 / 4,81 / 1,90; dentro de cada clase, within_class_mean_D67, 0,69 / 1,02 / 1,20; ratio 2,53):
     - CORREGIDO (D67 y rigor): Python: "- Valores más altos implican mayor separación entre las clases; valores bajos indican solapamiento o proximidad.". Una distancia media entre clases solo indica separación comparada con la distancia media DENTRO de cada clase (dos clases muy dispersas tienen una distancia media alta aunque se solapen). D67 muestra esa distancia en la diagonal.
     - NUEVO (D67): la línea "- La diagonal muestra la distancia media entre las muestras de una misma clase…" (en Python la diagonal valía 0 y no se explicaba).
     - CORREGIDO (rigor): Python: "- *Ratio Max/Min*: una medida heurística de separabilidad. En esta app usamos umbrales sencillos:". El ratio compara el par de clases más alejado con el más próximo: mide lo desiguales que son las distancias entre clases, no si las clases se solapan. En Iris da 2,53 («moderada») porque setosa está lejos de las otras dos, aunque versicolor y virginica se solapan.
     - NUEVO (D67): la línea "- Con dos clases solo hay una distancia…".
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
- *Ratio Max/Min*: compara el par de clases más alejado con el más próximo. Es una heurística: un ratio alto dice que hay clases mucho más cerca entre sí que otras, no que todas estén bien separadas (en Iris vale 2,53 aunque versicolor y virginica se solapan). En esta app usamos umbrales sencillos:
    - Ratio > 5 → clases bien separadas
    - 2 < Ratio ≤ 5 → separación moderada
    - Ratio ≤ 2 → clases muy próximas
- Con dos clases solo hay una distancia entre clases, así que el ratio siempre vale 1: fíjate en la diagonal.

**Sugerencias si las clases están próximas:**
- Escala las características antes de entrenar — KNN es sensible a la escala. MLTutor usa los datos sin escalar (ver «📏 KNN sin escalar» en 🏋️ Entrenamiento); fuera de la app, usa un `Pipeline` con `StandardScaler` para que el escalado se ajuste solo con el entrenamiento.
- Prueba la métrica Manhattan en caso de que la forma de los clusters sea no circular (en la app, Minkowski usa p = 2 y es la misma que la euclídea).
- Reduce o transforma características irrelevantes (PCA, selección de características).
- Aumenta el tamaño del conjunto de entrenamiento o prueba con más datos si es posible.

**Nota:** La matriz muestra promedios; examina además la distribución de distancias individuales si necesitas un diagnóstico más fino. Se calcula con las muestras de entrenamiento del modelo.
