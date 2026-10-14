<!-- viz/roc.py::_display_multiclass_interpretation, solo multiclase, tras la tabla 📊 Métricas por Clase. Expander cerrado. Título del expander: "ℹ️ ¿Macro vs Micro promedio?" -->
**Macro-promedio:**
- Calcula la métrica para cada clase independientemente
- Luego promedia los resultados (todas las clases tienen el mismo peso)
- Útil cuando todas las clases son igualmente importantes
- Más sensible al rendimiento en clases minoritarias

**Micro-promedio:**
- Agrega las contribuciones de todas las clases para calcular la métrica
- Da más peso a las clases con más muestras
- Útil cuando las clases más frecuentes son más importantes
- Menos sensible a clases desbalanceadas

**¿Cuál usar?**
- **Macro** cuando todas las clases son igualmente importantes
- **Micro** cuando algunas clases son más importantes que otras
- **Ambos** para una evaluación completa
