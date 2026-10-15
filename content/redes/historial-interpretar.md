<!-- mltutor/viz/nn.py — expander "💡 Interpretación del Entrenamiento" (cerrado) en 📊 Historial. -->
📚 **Guía para Interpretar los Gráficos:**

**🔴 Pérdida (Loss):**
- **Bajando:** ✅ El modelo está aprendiendo correctamente
- **Estable:** 🎯 El modelo ha convergido (terminó de aprender)
- **Subiendo:** 🚨 Posible sobreajuste o learning rate muy alto

**🟡 Validación vs Entrenamiento:**
- **Líneas cercanas:** ✅ Generalización saludable
- **Gap creciente:** 🚨 Sobreajuste (memorización vs aprendizaje)
- **Validación mejor que entrenamiento:** 🤔 Posible error en datos

**📈 Accuracy/Métricas:**
- **Crecimiento sostenido:** ✅ Aprendizaje progresivo
- **Plateau:** 🎯 Límite del modelo alcanzado
- **Fluctuaciones grandes:** ⚠️ Batch size muy pequeño o datos ruidosos

**🎯 Señales de Calidad:**
- ✅ Loss decreciente y suave
- ✅ Gap train/val menor al 10%
- ✅ Métricas estables al final
