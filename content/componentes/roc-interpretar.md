<!-- mltutor/viz/roc.py::_add_detailed_explanation, al final de la vista ROC (binaria y multiclase). Expander cerrado. Título del expander: "ℹ️ ¿Cómo interpretar las Curvas de Rendimiento?" -->
**Curva ROC (Receiver Operating Characteristic)**

**¿Qué muestra?**
- **Eje X:** Tasa de Falsos Positivos (FPR) = FP / (FP + TN)
- **Eje Y:** Tasa de Verdaderos Positivos (TPR) = TP / (TP + FN) = Sensibilidad/Recall
- **Línea diagonal:** Rendimiento de un clasificador aleatorio
- **AUC (Área Bajo la Curva):** Métrica resumen del rendimiento

**Interpretación:**
- **AUC = 1.0:** Clasificador perfecto
- **AUC = 0.9-1.0:** Excelente discriminación
- **AUC = 0.8-0.9:** Buena discriminación
- **AUC = 0.7-0.8:** Discriminación aceptable
- **AUC = 0.5:** Equivalente a adivinar al azar
- **AUC < 0.5:** Peor que adivinar (pero se puede invertir)

**¿Cuándo usar ROC?**
- Cuando las clases están relativamente balanceadas
- Para comparar modelos rápidamente
- Cuando te importa el rendimiento general

---

**Curva Precision-Recall (P-R)**

**¿Qué muestra?**
- **Eje X:** Recall (Sensibilidad) = TP / (TP + FN)
- **Eje Y:** Precision (Precisión) = TP / (TP + FP)
- **Línea horizontal:** Baseline (proporción de casos positivos)
- **AP (Average Precision):** Métrica resumen del rendimiento

**Interpretación:**
- **AP alto:** Buen balance entre precisión y recall
- **Curva cerca del ángulo superior derecho:** Excelente rendimiento
- **Por encima del baseline:** Mejor que una predicción aleatoria

**¿Cuándo usar P-R?**
- ✅ **Clases desbalanceadas** (muchos más negativos que positivos)
- ✅ Cuando los **falsos positivos son costosos**
- ✅ Para datasets con **pocos casos positivos**
- ✅ En problemas como **detección de fraude, diagnóstico médico**

**Comparación ROC vs P-R:**
- **ROC** es más optimista con clases desbalanceadas
- **P-R** es más conservadora y realista
- **P-R** se enfoca más en el rendimiento de la clase minoritaria
- Usar **ambas** para una evaluación completa

---

**Multi-clase: Macro vs Micro**

**Estrategia One-vs-Rest:**
- Cada clase se trata como positiva vs todas las demás como negativas
- Se calculan métricas para cada clase por separado
- Luego se combinan usando macro o micro promedio

**Macro-promedio:**
- Promedio simple de las métricas de cada clase
- Todas las clases tienen el mismo peso
- Más sensible a clases minoritarias
- Útil cuando todas las clases son igualmente importantes

**Micro-promedio:**
- Agregación global de TP, FP, FN de todas las clases
- Las clases con más muestras tienen más peso
- Menos sensible a desbalance de clases
- Útil cuando algunas clases son más importantes
