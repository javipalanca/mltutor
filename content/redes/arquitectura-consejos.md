<!-- mltutor/apps/neural_network_app.py — expander "💡 Tips para optimizar tu configuración" (cerrado) en 🏗️ Arquitectura. Variables: {batch_recomendado}, {n_muestras}. CORREGIDO en Rust (revisión de contenido de F5, 2026-10-16): Python: "- Dataset balanceado: 0.2-0.3". Que las clases estén equilibradas no tiene relación con el dropout; el rango 0.2-0.3 es un punto de partida habitual. -->
**Para tu dataset específico ({n_muestras} muestras):**

🎯 **Batch Size recomendado:**
- Dataset pequeño (<1000): 16-32
- Dataset mediano (1000-10000): 32-64
- Dataset grande (>10000): 64-128
- Tu dataset: {n_muestras} muestras → Recomendado: {batch_recomendado}

🛡️ **Dropout recomendado:**
- Pocos datos: 0.3-0.5 (más regularización)
- Muchos datos: 0.1-0.2 (menos regularización)
- Punto de partida habitual: 0.2-0.3

🚀 **Optimizador:**
- **Adam**: Mejor opción general, se adapta automáticamente
- **SGD**: Úsalo solo si tienes experiencia ajustando learning rates
- **RMSprop**: Alternativa a Adam, a veces funciona mejor en problemas específicos
