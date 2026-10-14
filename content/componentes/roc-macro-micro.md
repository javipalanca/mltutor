<!-- viz/roc.py::_display_multiclass_interpretation, solo multiclase, tras la tabla 📊 Métricas por Clase. Expander cerrado. Título del expander: "ℹ️ ¿Macro vs Micro promedio?". CORREGIDO en Rust (2026-10-14): "Micro cuando algunas clases son más importantes que otras" y "Menos sensible a clases desbalanceadas" eran incorrectos (micro pondera por número de muestras, no por importancia); se añade cómo se calcula la curva macro, cuyo AUC no es exactamente la media de la tabla. -->
**Macro-promedio:**
- Calcula la métrica para cada clase independientemente
- Luego promedia los resultados (todas las clases tienen el mismo peso)
- Útil cuando todas las clases son igualmente importantes
- Más sensible al rendimiento en clases minoritarias

**Micro-promedio:**
- Agrega las contribuciones de todas las clases para calcular la métrica
- Da más peso a las clases con más muestras
- Útil cuando las clases más frecuentes son más importantes
- Con clases desbalanceadas lo dominan las frecuentes: puede ocultar que una clase minoritaria va mal

**¿Cuál usar?**
- **Macro** cuando todas las clases son igualmente importantes
- **Micro** cuando cada muestra cuenta lo mismo y las clases frecuentes deben pesar más
- **Ambos** para una evaluación completa: si el macro es mucho peor que el micro, alguna clase pequeña falla

**¿Cómo se dibuja la curva macro?** Se promedian las curvas ROC de las clases (interpolando la 
TPR en los mismos valores de FPR). Su AUC es parecido, pero no siempre idéntico, a la media de 
los AUC de la tabla «📊 Métricas por Clase», sobre todo con pocas muestras de prueba. El AP 
macro sí es la media exacta de los AP de cada clase.
