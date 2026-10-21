<!-- viz/roc.py::plot_threshold_analysis, bajo '### 🎯 Análisis de Umbrales de Decisión' (solo binaria). Expander cerrado. Título del expander: "ℹ️ ¿Cómo interpretar el Análisis de Umbrales?". CORREGIDO en Rust (2026-10-14): (1) el ejemplo del spam estaba al revés: un falso positivo (un email bueno en spam) es el error caro, así que se prefiere un umbral alto; (2) el diagnóstico médico aparecía como ejemplo de umbral ALTO y más abajo como caso en el que se prefieren falsos positivos (umbral BAJO), que es lo correcto; (3) "No se pierde tantos"; (4) NUEVO: aviso de fuga de datos al elegir el umbral con el conjunto de prueba (D21 enseña ahora tres filas). CORREGIDO (revisión ML de F4, 2026-10-15): Python decía "**Probabilidad ≥ 0.5** → Clase Positiva / **Probabilidad < 0.5** → Clase Negativa", que no es la regla de `predict`: en la logística es `decision_function > 0`, es decir, p > 0,5; en árboles y KNN es la clase de mayor probabilidad y, en un empate exacto a 0,5, gana la de código menor (la negativa). La tabla (crates/mltutor-core/src/probabilities.rs::threshold_analysis) usa p ≥ t con umbrales exactos i/10, salvo la fila 0,5, que usa p > 0,5 para coincidir con predict (revisión F4). NUEVO: el párrafo sobre las probabilidades discretas de KNN y árboles (🔬 Cáncer, KNN con K = 10 uniform: 4 de las 171 muestras de prueba tienen p = 0,5 exacto y predict las da como malignant). -->
**¿Qué es el umbral de decisión?**

El umbral de decisión es el valor que determina cuándo el modelo clasifica una muestra como 
positiva o negativa. Por defecto, este umbral es **0.5**:
- **Probabilidad > 0.5** → Clase Positiva
- **Probabilidad ≤ 0.5** → Clase Negativa

Es la regla de `predict` en scikit-learn: la regresión logística da la clase positiva solo si la 
probabilidad supera 0,5, y los árboles y KNN eligen la clase más probable y, si empatan justo en 0,5, 
la de código menor (la negativa). La tabla de abajo cuenta como positiva toda muestra con **probabilidad ≥ umbral**, salvo 
en la fila de 0,5, que usa **probabilidad > 0,5** para coincidir siempre con `predict` y con la 
📈 Evaluación. Por eso, una muestra con probabilidad exactamente 0,3 o 0,7 sí cuenta como positiva en 
esa fila.

La diferencia importa en árboles y KNN, cuyas probabilidades solo toman unos pocos valores: con un KNN 
de $K = 10$ y pesos `uniform` solo puede salir 0, 0,1, 0,2… hasta 1, y muchas muestras caen justo en un 
umbral (en 🔬 Cáncer de mama, 4 de las 171 muestras de prueba tienen exactamente 0,5). Por eso, al mover 
el umbral, las métricas cambian a saltos: dos umbrales que caen entre los mismos dos valores de probabilidad dan las mismas métricas.

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
