<!-- model_evaluation.py::show_detailed_evaluation (regresión), tras el histograma de errores. Expander cerrado. Título del expander: "📊 Explicación de métricas de regresión" -->
**R² Score (Coeficiente de Determinación):**
- Mide qué tan bien el modelo explica la variabilidad de los datos
- Indica qué proporción de la varianza en la variable dependiente es predecible. 1 es predicción perfecta, 0 significa que el modelo no es mejor que predecir la media.
- Rango: 0 a 1 (valores negativos indican un modelo muy malo)
- **Interpretación:**
    - R² = 1.0: El modelo explica perfectamente toda la variabilidad
    - R² = 0.8: El modelo explica el 80% de la variabilidad (muy bueno)
    - R² = 0.5: El modelo explica el 50% de la variabilidad (moderado)
    - R² = 0.0: El modelo no explica nada de la variabilidad

**MAE (Error Absoluto Medio):**
- Promedio de las diferencias absolutas entre valores reales y predichos
- Se expresa en las mismas unidades que la variable objetivo
- **Interpretación:** Valores más bajos = mejor modelo

**RMSE (Raíz del Error Cuadrático Medio):**
- Similar al MAE pero penaliza más los errores grandes
- Se expresa en las mismas unidades que la variable objetivo
- **Interpretación:** Valores más bajos = mejor modelo
