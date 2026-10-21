<!-- mltutor/algorithms/model_evaluation.py — expander "🎓 Contexto Educativo: ¿Cómo interpretar estos números?" (cerrado) al final de 📈 Evaluación → 🔄 Patrones. CORREGIDO en Rust (revisión de contenido de F5, 2026-10-16): (1) NUEVO: el párrafo sobre la escala de la pérdida. Los umbrales de pendiente son absolutos y solo tienen sentido con pérdidas del orden de 1 (entropía cruzada): en tests/parity/nn/exact/diabetes_default.json (MSE de unos 30 000) la pérdida baja unas 180 unidades por época y el diagnóstico la clasifica como "🚀 Excelente" con un R² de prueba de −3; (2) errata: Python "- Optimal → …" (dos veces) pasa a "- Adecuado → …". -->
### 📐 **Entendiendo los Valores de Tendencia:**

**Valores Negativos (Descendente):**
- `-0.1` o menor: 🚀 Aprendizaje muy rápido
- `-0.01` a `-0.1`: ✅ Aprendizaje saludable  
- `-0.001` a `-0.01`: 📊 Aprendizaje gradual
- `-0.0001` a `-0.001`: 🎯 Convergencia fina

**Valores Cerca de Cero:**
- `-0.0001` a `+0.0001`: 🎯 Convergencia ideal

**Valores Positivos (Ascendente):**
- `+0.0001` a `+0.001`: ⚠️ Ligero deterioro
- `+0.001` a `+0.01`: 🚨 Problema moderado
- `+0.01` o mayor: 💥 Problema grave

⚠️ **Estos umbrales dependen de la escala de la pérdida.** Valen para la entropía cruzada de clasificación, que suele estar entre 0 y 2. En regresión la pérdida es el MSE, en las unidades del objetivo al cuadrado: en 🩺 Diabetes empieza en unos 30 000 y una pendiente de −100 por época puede significar que la red todavía predice muy mal.

### 🔬 **Factores que Afectan las Tendencias:**

**Learning Rate:**
- Muy alto → Oscilaciones o divergencia
- Muy bajo → Convergencia lenta
- Adecuado → Descenso suave y rápido

**Batch Size:**
- Pequeño → Más ruido, convergencia irregular
- Grande → Menos ruido, convergencia suave
- Adecuado → Balance entre velocidad y estabilidad

**Arquitectura del Modelo:**
- Muy simple → Plateau temprano (underfitting)
- Muy compleja → Sobreajuste tardío
- Apropiada → Convergencia saludable sin sobreajuste
