<!-- mltutor/viz/nn.py — INALCANZABLE en Python (viz/nn.py::show_neural_network_predictions); se porta como NUEVO por D81. Expander "💡 Consejos para Interpretar los Resultados" (cerrado) debajo del gráfico de exploración. Variables: {caracteristica}. CORREGIDO en Rust (revisión de contenido de F5, 2026-10-16): (1) Python: "- Relaciones monotónicas (siempre creciente o decreciente)" como buena señal: una relación no monótona puede ser correcta; solo es buena señal si el dominio la espera; (2) Python: "2. Explora otras características para comparar importancias": el efecto en una sola muestra no es la importancia de la característica en el modelo. NUEVO (revisión de F5d, 2026-10-16): la señal de alerta sobre la confianza en los extremos (extrapolación). -->
**🔍 Analizando '{caracteristica}':**

✅ **Buenas señales:**
- Cambios graduales y suaves en las predicciones
- Comportamiento consistente con el conocimiento del dominio
- Relaciones monótonas (siempre crecientes o decrecientes) cuando el conocimiento del dominio las espera

⚠️ **Señales de alerta:**
- Cambios muy abruptos sin explicación lógica
- Comportamientos contradictorios al conocimiento experto
- Excesiva sensibilidad a pequeños cambios
- Probabilidades cada vez más altas hacia los extremos del eje: suele ser extrapolación (zona sin datos), no certeza

**🎯 Próximos pasos:**
1. Prueba con diferentes muestras base para confirmar patrones
2. Explora otras características para comparar su efecto (en esta muestra; la importancia global requiere muchas muestras)
3. Si encuentras comportamientos extraños, considera reentrenar el modelo
4. Documenta los insights para mejorar futuras versiones del modelo
