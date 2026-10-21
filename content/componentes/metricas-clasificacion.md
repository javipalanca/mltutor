<!-- mltutor/algorithms/model_evaluation.py::show_detailed_evaluation (clasificación), columna derecha, bajo el informe por clase. Expander cerrado. OJO: la sangría irregular del original (primera línea a 17 espacios, el resto a 20) hace que, tras dedent, las viñetas queden sangradas 3 espacios; se conserva tal cual. Título del expander: "📊 Explicación de métricas" -->
**Accuracy (Exactitud):**
   - Porcentaje de predicciones correctas del total
   - Rango: 0 a 1 (0% a 100%)
   - **Interpretación:** Valores más altos = mejor modelo
   - **Cuidado:** Puede ser engañosa con clases desbalanceadas

   **Precision (Precisión):**
   - De todas las predicciones positivas, cuántas fueron correctas
   - Fórmula: VP / (VP + FP)
   - Importante cuando los falsos positivos son costosos
   - **Interpretación:** Valores más altos = mejor modelo

   **Recall (Sensibilidad o Exhaustividad):**
   - De todos los casos positivos reales, cuántos detectó el modelo
   - Fórmula: VP / (VP + FN)
   - Importante cuando los falsos negativos son costosos
   - **Interpretación:** Valores más altos = mejor modelo

   **F1-Score:**
   - Media armónica entre precisión y recall
   - Fórmula: 2 × (Precisión × Recall) / (Precisión + Recall)
   - Útil cuando necesitas balance entre precisión y recall
   - **Interpretación:** Valores más altos = mejor balance

   **Curva ROC:**
   - Muestra el rendimiento en diferentes umbrales de decisión
   - AUC (Área bajo la curva): 0.5 = aleatorio, 1.0 = perfecto

   **Curva Precision-Recall:**
   - Especialmente útil para clases desbalanceadas
   - Muestra el trade-off entre precisión y recall

   **VP = Verdaderos Positivos, FP = Falsos Positivos, FN = Falsos Negativos**
