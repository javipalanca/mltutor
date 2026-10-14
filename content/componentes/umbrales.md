<!-- viz/roc.py::plot_threshold_analysis, bajo '### 🎯 Análisis de Umbrales de Decisión' (solo binaria). Expander cerrado. Título del expander: "ℹ️ ¿Cómo interpretar el Análisis de Umbrales?". CORREGIDO en Rust (2026-10-14): (1) el ejemplo del spam estaba al revés: un falso positivo (un email bueno en spam) es el error caro, así que se prefiere un umbral alto; (2) el diagnóstico médico aparecía como ejemplo de umbral ALTO y más abajo como caso en el que se prefieren falsos positivos (umbral BAJO), que es lo correcto; (3) "No se pierde tantos"; (4) NUEVO: aviso de fuga de datos al elegir el umbral con el conjunto de prueba (D21 enseña ahora tres filas). -->
**¿Qué es el umbral de decisión?**

El umbral de decisión es el valor que determina cuándo el modelo clasifica una muestra como 
positiva o negativa. Por defecto, este umbral es **0.5**:
- **Probabilidad ≥ 0.5** → Clase Positiva
- **Probabilidad < 0.5** → Clase Negativa

**¿Por qué cambiar el umbral?**

El umbral por defecto (0.5) no siempre es óptimo. Dependiendo del problema, 
puede ser beneficioso ajustarlo:

**📈 Umbral más alto (0.6, 0.7, 0.8):**
- ✅ **Mayor Precisión:** Menos falsos positivos
- ✅ **Predicciones más conservadoras:** Solo clasifica como positivo cuando está muy seguro
- ⚠️ **Menor Recall:** Puede perder casos positivos reales
- **Útil cuando:** Los falsos positivos son muy costosos (ej: filtro de spam, inversiones)

**📉 Umbral más bajo (0.3, 0.4):**
- ✅ **Mayor Recall:** Detecta más casos positivos reales
- ✅ **Predicciones más sensibles:** No se pierden tantos casos positivos
- ⚠️ **Menor Precisión:** Más falsos positivos
- **Útil cuando:** Los falsos negativos son muy costosos (ej: cribado médico, detección de fraude, seguridad)

**Métricas mostradas:**
- **Accuracy:** Porcentaje total de predicciones correctas
- **Precision:** De las predicciones positivas, cuántas son correctas
- **Recall:** De los casos positivos reales, cuántos detectamos
- **F1-Score:** Balance entre precisión y recall

**¿Cómo elegir el umbral óptimo?**
1. **Maximizar F1-Score:** Balance general entre precisión y recall
2. **Maximizar Precision:** Si los falsos positivos son costosos
3. **Maximizar Recall:** Si los falsos negativos son costosos
4. **Considerar el contexto:** Costos reales de errores en tu dominio

⚠️ **Cuidado con la fuga de datos:** aquí el umbral óptimo se busca con el mismo conjunto de 
prueba con el que se miden las métricas, así que su F1 sale algo optimista. En un proyecto real, 
elige el umbral con datos de **validación** (o validación cruzada) y usa la prueba solo al final.

**Ejemplo práctico:**
- **Email spam:** Prefiere falsos negativos (algo de spam en la bandeja de entrada) que falsos positivos (un email importante en spam): umbral más alto
- **Diagnóstico médico:** Prefiere falsos positivos (más pruebas) que falsos negativos (enfermedad no detectada): umbral más bajo
- **Recomendaciones:** Balance entre no molestar (precisión) y no perder oportunidades (recall)
