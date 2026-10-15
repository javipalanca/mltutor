<!-- mltutor/apps/knn_app.py (viz "Distancias") — expander "ℹ️ Cómo interpretar este análisis" (cerrado), debajo de "### Análisis de Distancias entre Clases". Literal. {metrica} = model.metric tal cual ("minkowski", "euclidean" o "manhattan"). -->
**¿Qué representa la matriz de distancias?**

- Cada celda (i, j) contiene la distancia promedio entre las muestras de la clase *i* y las de la clase *j* calculada usando la métrica seleccionada por el modelo (actualmente: **{metrica}**).
- Valores más altos implican mayor separación entre las clases; valores bajos indican solapamiento o proximidad.

**Interpretación rápida de las estadísticas mostradas:**
- *Distancia Mínima*: el menor valor promedio observado entre pares de clases (ignorando la diagonal).
- *Distancia Máxima*: el mayor valor promedio observado.
- *Ratio Max/Min*: una medida heurística de separabilidad. En esta app usamos umbrales sencillos:
    - Ratio > 5 → clases bien separadas
    - 2 < Ratio ≤ 5 → separación moderada
    - Ratio ≤ 2 → clases muy próximas

**Sugerencias si las clases están próximas:**
- Escala las características (StandardScaler) antes de entrenar — KNN es sensible a la escala.
- Prueba otra métrica (Manhattan / Minkowski) en caso de que la forma de los clusters sea no circular.
- Reduce o transforma características irrelevantes (PCA, selección de características).
- Aumenta el tamaño del conjunto de entrenamiento o prueba con más datos si es posible.

**Nota:** La matriz muestra promedios; examina además la distribución de distancias individuales si necesitas un diagnóstico más fino.
