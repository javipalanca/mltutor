<!-- mltutor/viz/roc.py::_add_detailed_explanation, al final de la vista ROC (binaria y multiclase). Expander cerrado. Título del expander: "ℹ️ ¿Cómo interpretar las Curvas de Rendimiento?". CORREGIDO en Rust (2026-10-14): (1) el micro-promedio "Útil cuando algunas clases son más importantes" y "Menos sensible a desbalance" era incorrecto: micro da el mismo peso a cada muestra, así que lo dominan las clases frecuentes y puede ocultar un mal rendimiento en las minoritarias; (2) se añaden la lectura probabilística del AUC y la fórmula de AP. -->
**Curva ROC (Receiver Operating Characteristic)**

**¿Qué muestra?**
- **Eje X:** Tasa de Falsos Positivos (FPR) = FP / (FP + TN)
- **Eje Y:** Tasa de Verdaderos Positivos (TPR) = TP / (TP + FN) = Sensibilidad/Recall
- **Línea diagonal:** Rendimiento de un clasificador aleatorio
- **AUC (Área Bajo la Curva):** Métrica resumen del rendimiento. Es la probabilidad de que el modelo dé más puntuación a un positivo elegido al azar que a un negativo elegido al azar
- Cada punto de la curva es un umbral de decisión distinto: al bajar el umbral se detectan más positivos (sube TPR) a costa de más falsas alarmas (sube FPR)

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
- **AP (Average Precision):** Métrica resumen del rendimiento: la precisión media a lo largo de la curva, $\text{AP} = \sum_n (R_n - R_{n-1})\, P_n$, donde $P_n$ y $R_n$ son la precisión y el recall del umbral *n*

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
- Puede ocultar un mal rendimiento en las clases minoritarias
- Útil cuando cada muestra cuenta lo mismo, sea de la clase que sea
