<!-- mltutor/algorithms/model_evaluation.py — expander "📚 ¿Qué significan las tendencias de pérdida?" (cerrado) en 📈 Evaluación → 📊 Tendencias. -->
### 🎯 **Interpretación de Tendencias por Fase:**

**🟢 Fase Inicial (Primeras épocas):**
- **Descendente rápido:** ✅ Excelente - El modelo está aprendiendo patrones básicos
- **Descendente lento:** ⚠️ Learning rate muy bajo o datos complejos
- **Ascendente:** 🚨 Learning rate muy alto o problema en los datos

**🟡 Fase Media (Épocas intermedias):**
- **Descendente sostenido:** ✅ Aprendizaje progresivo saludable
- **Plateau temprano:** ⚠️ Posible saturación o learning rate muy bajo
- **Fluctuaciones:** 📊 Normal, pero pueden indicar batch size pequeño

**🔴 Fase Final (Últimas épocas):**
- **Estabilizado:** 🎯 Ideal - Convergencia alcanzada
- **Descendente:** 📈 Aún aprendiendo - Considera más épocas
- **Ascendente:** 🚨 Sobreajuste - Detener entrenamiento antes

### 📊 **Patrones de Calidad:**
- **Curva logarítmica suave:** Patrón ideal de aprendizaje
- **Escalones descendentes:** Learning rate scheduling efectivo
- **Zigzag descendente:** Normal con batch gradient descent
- **Valle en U:** Posible learning rate muy alto inicialmente
