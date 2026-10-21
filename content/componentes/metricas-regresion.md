<!-- model_evaluation.py::show_detailed_evaluation (regresión), tras el histograma de errores. Expander cerrado. Título del expander: "📊 Explicación de métricas de regresión". CORREGIDO en Rust (2026-10-14): (1) "Rango: 0 a 1 (valores negativos…)" se contradecía: R² va de −∞ a 1 y en el conjunto de prueba es negativo si el modelo predice peor que la media; (2) faltaba el MSE, que es la primera métrica de la vista; (3) se añaden las fórmulas y un ejemplo con Diabetes (lineal, test_size 0.3: MSE 2821.75, RMSE 53.12, MAE 41.92, R² 0.4773). -->
**R² Score (Coeficiente de Determinación):**
- Mide qué tan bien el modelo explica la variabilidad de los datos
- Indica qué proporción de la varianza en la variable dependiente es predecible. 1 es predicción perfecta, 0 significa que el modelo no es mejor que predecir la media.
- Fórmula: $R^2 = 1 - \dfrac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$, donde $\bar{y}$ es la media de los valores reales
- Rango: de $-\infty$ a 1. Con los datos de prueba puede ser negativo: el modelo predice peor que si diera siempre la media
- **Interpretación:**
    - R² = 1.0: El modelo explica perfectamente toda la variabilidad
    - R² = 0.8: El modelo explica el 80% de la variabilidad (muy bueno)
    - R² = 0.5: El modelo explica el 50% de la variabilidad (moderado)
    - R² = 0.0: El modelo no explica nada de la variabilidad

**MSE (Error Cuadrático Medio):**
- Promedio de los errores al cuadrado: $\text{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$
- Penaliza mucho los errores grandes: un error de 10 pesa 100 veces más que un error de 1
- Se expresa en las unidades de la variable objetivo **al cuadrado**, por eso cuesta interpretarlo directamente
- **Interpretación:** Valores más bajos = mejor modelo

**MAE (Error Absoluto Medio):**
- Promedio de las diferencias absolutas entre valores reales y predichos: $\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|$
- Se expresa en las mismas unidades que la variable objetivo
- **Interpretación:** Valores más bajos = mejor modelo

**RMSE (Raíz del Error Cuadrático Medio):**
- Similar al MAE pero penaliza más los errores grandes: $\text{RMSE} = \sqrt{\text{MSE}}$
- Se expresa en las mismas unidades que la variable objetivo
- Siempre RMSE ≥ MAE; cuanto más se separan, más pesan unos pocos errores grandes
- **Interpretación:** Valores más bajos = mejor modelo

**Ejemplo (🩺 Diabetes, regresión lineal):** MSE ≈ 2821.75, RMSE ≈ 53.12 y MAE ≈ 41.92 puntos de 
progresión de la enfermedad, con R² ≈ 0.48: el modelo explica algo menos de la mitad de la 
variabilidad y suele equivocarse en unos 40–50 puntos.

⚠️ Calcula siempre estas métricas con el **conjunto de prueba**. Con los datos de 
entrenamiento salen demasiado optimistas, sobre todo si el modelo está sobreajustado.
