<!-- mltutor/algorithms/model_evaluation.py — expander "📚 ¿Qué significan las tendencias de pérdida?" (cerrado) en 📈 Evaluación → 📊 Tendencias. CORREGIDO en Rust (revisión de contenido de F5, 2026-10-16): las tendencias se calculan sobre la pérdida de ENTRENAMIENTO (history["loss"]), que el sobreajuste no hace subir. (1) Python: "- **Ascendente:** 🚨 Sobreajuste - Detener entrenamiento antes" (fase final); (2) Python: "- **Zigzag descendente:** Normal con batch gradient descent". El zigzag es propio de los mini-lotes (descenso de gradiente estocástico); con todos los datos a la vez la curva es suave; (3) Python: "- **Valle en U:** Posible learning rate muy alto inicialmente". -->
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
- **Ascendente:** 🚨 Inestabilidad (learning rate demasiado alto). El sobreajuste se ve en la pérdida de validación, no en esta

### 📊 **Patrones de Calidad:**
- **Curva logarítmica suave:** Patrón ideal de aprendizaje
- **Escalones descendentes:** Learning rate scheduling efectivo
- **Zigzag descendente:** Normal con mini-lotes (cada actualización usa solo una parte de los datos)
- **Valle en U:** En la pérdida de entrenamiento, learning rate demasiado alto; en la de validación, sobreajuste (el fondo del valle es la mejor época)
