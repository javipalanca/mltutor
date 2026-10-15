<!-- mltutor/apps/knn_app.py::create_interactive_knn_visualization — expander "📚 Guía de la Visualización Interactiva" (cerrado), al final de 🎮 Visualización Interactiva. Literal.
     Ojo (D55): Python limita el deslizador del lienzo a 1–15 aunque el modelo admite K hasta 20; si Rust usa 1–20, cambiar "Valor K (1-15)" con marca CORREGIDO. -->
**🎯 Visualización Interactiva de K-Nearest Neighbors**

Esta herramienta te permite experimentar directamente con el algoritmo KNN:

#### 🔧 **Controles Disponibles:**
- **Valor K (1-15)**: Número de vecinos más cercanos a considerar
- **Métrica de Distancia**: Euclidean, Manhattan, o Minkowski
- **Tipo de Pesos**: Uniforme (todos iguales) o por distancia (más cercanos pesan más)
- **Mostrar Confianza**: Visualiza qué tan seguro está el modelo (anillo verde)
- **Mostrar Distancias**: Líneas punteadas rojas conectando con vecinos

#### 🎨 **Elementos Visuales:**
- **Puntos de colores**: Datos de entrenamiento (cada color = una clase)
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
- **Distancia Manhattan**: Cuadrados de influencia (movimiento tipo taxi)
- **Pesos por distancia**: Vecinos más cercanos tienen más influencia

**¡La mejor forma de aprender es experimentando directamente con la visualización!**
