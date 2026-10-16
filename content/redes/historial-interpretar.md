<!-- mltutor/viz/nn.py — expander "💡 Interpretación del Entrenamiento" (cerrado) en 📊 Historial. CORREGIDO en Rust (revisión de contenido de F5, 2026-10-16): (1) Python: "- **Subiendo:** 🚨 Posible sobreajuste o learning rate muy alto". El sobreajuste no hace subir la pérdida de ENTRENAMIENTO: hace subir la de validación mientras la de entrenamiento baja; (2) Python: "- **Validación mejor que entrenamiento:** 🤔 Posible error en datos". Con dropout es lo normal: Keras (y MLTutor) miden la pérdida de entrenamiento con el dropout activo y como media de la época, y la de validación sin dropout y con los pesos del final de la época (en tests/parity/nn/exact/diabetes_default.json, sin dropout, la validación también es menor porque son otras filas). -->
📚 **Guía para Interpretar los Gráficos:**

**🔴 Pérdida (Loss):**
- **Bajando:** ✅ El modelo está aprendiendo correctamente
- **Estable:** 🎯 El modelo ha convergido (terminó de aprender)
- **Subiendo:** 🚨 Learning rate muy alto (si sube la de entrenamiento) o sobreajuste (si solo sube la de validación)

**🟡 Validación vs Entrenamiento:**
- **Líneas cercanas:** ✅ Generalización saludable
- **Gap creciente:** 🚨 Sobreajuste (memorización vs aprendizaje)
- **Validación mejor que entrenamiento:** 🤔 Normal con dropout (la pérdida de entrenamiento se mide con el dropout activo) o con pocas filas de validación; si la diferencia es grande, revisa cómo se separaron los datos

**📈 Accuracy/Métricas:**
- **Crecimiento sostenido:** ✅ Aprendizaje progresivo
- **Plateau:** 🎯 Límite del modelo alcanzado
- **Fluctuaciones grandes:** ⚠️ Batch size muy pequeño o datos ruidosos

**🎯 Señales de Calidad:**
- ✅ Loss decreciente y suave
- ✅ Gap train/val menor al 10%
- ✅ Métricas estables al final
