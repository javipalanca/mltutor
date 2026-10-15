<!-- mltutor/algorithms/model_evaluation.py — expander "🎓 Contexto Educativo: ¿Cómo interpretar estos números?" (cerrado) al final de 📈 Evaluación → 🔄 Patrones. -->
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

### 🔬 **Factores que Afectan las Tendencias:**

**Learning Rate:**
- Muy alto → Oscilaciones o divergencia
- Muy bajo → Convergencia lenta
- Optimal → Descenso suave y rápido

**Batch Size:**
- Pequeño → Más ruido, convergencia irregular
- Grande → Menos ruido, convergencia suave
- Optimal → Balance entre velocidad y estabilidad

**Arquitectura del Modelo:**
- Muy simple → Plateau temprano (underfitting)
- Muy compleja → Sobreajuste tardío
- Apropiada → Convergencia saludable sin sobreajuste
