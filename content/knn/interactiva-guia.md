<!-- mltutor/apps/knn_app.py::create_interactive_knn_visualization — expander "📚 Guía de la Visualización Interactiva" (cerrado), al final de 🎮 Visualización Interactiva. Literal.
     Literal salvo las líneas marcadas aquí:
     - CORREGIDO (D55): Python: "- **Valor K (1-15)**: Número de vecinos más cercanos a considerar". El deslizador de Python iba de 1 a 15 aunque el modelo admite K hasta 20; el componente nativo va de 1 a 20.
     - CORREGIDO (aclaración): Python: "- **Métrica de Distancia**: Euclidean, Manhattan, o Minkowski". Con p = 2 (el valor de scikit-learn y de la app), Minkowski ES la euclídea: el estudiante esperaba ver una diferencia que no existe.
     - CORREGIDO (D63): Python: "- **Puntos de colores**: Datos de entrenamiento (cada color = una clase)". El lienzo usa TODAS las muestras del dataset, no solo las de entrenamiento.
     - CORREGIDO (geometría): Python: "- **Distancia Manhattan**: Cuadrados de influencia (movimiento tipo taxi)". Los puntos a la misma distancia Manhattan forman un rombo (un cuadrado girado 45°), no un cuadrado con los lados paralelos a los ejes.
     - NUEVO (D54): la sección "#### 🖱️ **Novedades del lienzo**" (arrastrar, valores exactos, animación de vecinos, tabla y puntos repetidos), que Python no tenía. -->
**🎯 Visualización Interactiva de K-Nearest Neighbors**

Esta herramienta te permite experimentar directamente con el algoritmo KNN:

#### 🔧 **Controles Disponibles:**
- **Valor K (1-20)**: Número de vecinos más cercanos a considerar
- **Métrica de Distancia**: Euclidean, Manhattan, o Minkowski (con p = 2, la que usa la app, Minkowski es la euclídea)
- **Tipo de Pesos**: Uniforme (todos iguales) o por distancia (más cercanos pesan más)
- **Mostrar Confianza**: Visualiza qué tan seguro está el modelo (anillo verde)
- **Mostrar Distancias**: Líneas punteadas rojas conectando con vecinos

#### 🎨 **Elementos Visuales:**
- **Puntos de colores**: Todas las muestras del dataset (cada color = una clase)
- **Estrellas doradas**: Puntos de prueba que agregues
- **Líneas punteadas**: Conexiones a los K vecinos más cercanos
- **Anillo verde**: Nivel de confianza de la predicción
- **Porcentaje**: Confianza numérica de la clasificación

#### 💡 **Consejos para Experimentar:**
- Prueba diferentes valores de K en la misma ubicación
- Haz click cerca de las fronteras entre clases
- Observa cómo cambia la confianza en diferentes regiones
- Compara las métricas Euclidean vs Manhattan
- Experimenta con pesos uniformes vs por distancia

#### 🎓 **¿Qué Aprender?**
- **K bajo**: Más sensible a ruido, fronteras más irregulares
- **K alto**: Más suave, pero puede perder detalles importantes
- **Distancia Euclidean**: Círculos de influencia
- **Distancia Manhattan**: Rombos de influencia (cuadrados girados 45°; movimiento tipo taxi)
- **Pesos por distancia**: Vecinos más cercanos tienen más influencia

#### 🖱️ **Novedades del lienzo**
- **Arrastra** un punto de prueba para moverlo y mira cómo cambian sus vecinos y su clase al cruzar una frontera
- **✏️ Añadir un punto con valores exactos**: escribe los dos valores en lugar de hacer clic
- **Líneas a los vecinos**: aparecen una a una, del vecino más cercano al más lejano
- **📊 Análisis de Puntos de Prueba**: la tabla con la predicción, la confianza y la distancia media de cada punto
- **Puntos repetidos**: varias muestras con los mismos dos valores caen en el mismo sitio; el tamaño del punto y el texto al pasar el ratón indican cuántas hay

**¡La mejor forma de aprender es experimentando directamente con la visualización!**
