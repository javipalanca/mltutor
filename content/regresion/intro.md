<!-- mltutor/apps/linear_regression_app.py — expander "ℹ️ ¿Qué es la Regresión?" (cerrado por defecto), encima de las pestañas. CORREGIDO en Rust (2026-10-14): la viñeta del escalado decía "No requieren escalado de datos, aunque puede mejorar la convergencia", pero la logística de scikit-learn está regularizada (L2), así que el escalado también cambia el modelo, no solo la convergencia; y los coeficientes solo son comparables entre sí si las características están en escalas parecidas. -->
La regresión es un conjunto de algoritmos de aprendizaje supervisado utilizados para predecir valores numéricos continuos (regresión lineal) o clasificar elementos en categorías (regresión logística).

**Características principales:**
- **Regresión Lineal**: Establece una relación lineal entre las variables independientes (características) y la variable dependiente (objetivo)
- **Regresión Logística**: Utiliza la función logística para modelar la probabilidad de pertenencia a una clase
- Ambos tipos minimizan funciones de costo específicas para encontrar los mejores parámetros
- Son interpretables: los coeficientes indican la importancia y dirección del efecto de cada característica (para comparar su tamaño, las características deben estar en escalas parecidas)
- La regresión lineal no necesita escalar los datos para predecir bien; en la logística, escalar (por ejemplo con `StandardScaler`) ayuda al optimizador a converger y cambia el modelo, porque su regularización penaliza igual todos los coeficientes

**Tipos de regresión:**
- **Regresión Lineal Simple**: Una sola característica predictora
- **Regresión Lineal Múltiple**: Múltiples características predictoras
- **Regresión Logística**: Para problemas de clasificación binaria o multiclase

**Limitaciones:**
- Asume una relación lineal entre variables (regresión lineal)
- Sensible a valores atípicos (outliers)
- Puede sufrir de multicolinealidad cuando las características están correlacionadas
