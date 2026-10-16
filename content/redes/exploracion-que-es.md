<!-- mltutor/viz/nn.py — INALCANZABLE en Python (viz/nn.py::show_neural_network_predictions, pestaña "🎲 Exploración Interactiva"); se porta como NUEVO por D81. Expander "ℹ️ ¿Qué es la Exploración Interactiva?" (cerrado). CORREGIDO en Rust (revisión de contenido de F5, 2026-10-16): la gráfica muestra la respuesta del MODELO al variar una característica en UNA muestra (una curva ICE), no una correlación de los datos. Python: "- **Líneas ascendentes**: La característica tiene correlación positiva", "- **Líneas descendentes**: La característica tiene correlación negativa", "- **Líneas planas**: La característica tiene poco impacto" y "**La exploración interactiva** te permite entender cómo el modelo neural toma decisiones:" ("neural" → "neuronal"). NUEVO: la viñeta sobre combinaciones irreales. -->
**La exploración interactiva** te permite entender cómo el modelo neuronal toma decisiones:

🔍 **¿Para qué sirve?**
- Ver cómo cada característica influye en las predicciones
- Identificar patrones y comportamientos del modelo
- Detectar posibles sesgos o comportamientos inesperados
- Comprender la sensibilidad del modelo a cambios en los datos

📊 **¿Cómo interpretar los resultados?**
- **Líneas ascendentes**: Para esta muestra, el modelo predice más al aumentar la característica
- **Líneas descendentes**: Para esta muestra, el modelo predice menos al aumentar la característica  
- **Líneas planas**: Con los demás valores de esta muestra, la característica apenas cambia la predicción (con otra muestra puede ser distinto)
- **Cambios abruptos**: Puntos de decisión críticos del modelo

💡 **Consejos de uso:**
- Prueba diferentes muestras base para ver patrones generales
- Observa qué características causan mayores cambios
- Busca comportamientos inesperados o poco realistas
- Recuerda que se cambia una sola característica y las demás se quedan fijas: algunas combinaciones (un pétalo muy largo y muy estrecho) no existen en los datos, y ahí la red extrapola. Describe al modelo, no a la realidad
