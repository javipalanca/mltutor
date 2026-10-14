<!-- viz/roc.py::plot_threshold_analysis (binaria), antes del histograma. Expander cerrado. Título del expander: "ℹ️ ¿Cómo interpretar la Distribución de Probabilidades?" -->
**¿Qué muestra este gráfico?**

Este histograma muestra cómo el modelo asigna probabilidades a cada muestra del conjunto de prueba, 
separado por la clase real a la que pertenece cada muestra.

**Elementos del gráfico:**
- **Histograma azul:** Distribución de probabilidades para muestras que realmente pertenecen a la clase positiva
- **Histograma rojo:** Distribución de probabilidades para muestras que realmente pertenecen a la clase negativa  
- **Línea roja vertical:** Umbral de decisión (0.5 por defecto)
- **Eje X:** Probabilidad asignada por el modelo (0 = clase negativa, 1 = clase positiva)
- **Eje Y:** Cantidad de muestras

**Interpretación ideal:**
- ✅ **Buena separación:** Los histogramas no se superponen mucho
- ✅ **Clase negativa:** Concentrada cerca de 0 (izquierda)
- ✅ **Clase positiva:** Concentrada cerca de 1 (derecha)
- ✅ **Pocas muestras cerca del umbral (0.5):** Indica confianza en las predicciones

**Problemas a identificar:**
- ⚠️ **Mucha superposición:** Indica dificultad para separar las clases
- ⚠️ **Concentración en el centro (0.3-0.7):** El modelo está inseguro
- ⚠️ **Distribución uniforme:** El modelo no está aprendiendo patrones útiles

**Aplicaciones prácticas:**
- Identificar si el modelo está confiado en sus predicciones
- Evaluar si cambiar el umbral de decisión podría mejorar el rendimiento
- Detectar casos donde el modelo necesita más datos o características
