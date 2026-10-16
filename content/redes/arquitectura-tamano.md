<!-- mltutor/apps/neural_network_app.py — expander "💡 ¿Cómo decidir el tamaño de la red?" (cerrado) en 🏗️ Arquitectura. Variables: {cuadruple}, {doble}, {max_parametros}, {mitad}, {n_caracteristicas}, {n_muestras}. CORREGIDO en Rust (revisión de contenido de F5, 2026-10-16): Python: "- 1-2 capas: Problemas simples, linealmente separables". Un problema linealmente separable no necesita capas ocultas (basta una regresión logística); una sola capa oculta ya aprende fronteras curvas (🌙 Moons) y, con suficientes neuronas, puede aproximar cualquier función continua. -->
**Reglas generales para tu dataset ({n_muestras} muestras, {n_caracteristicas} características):**

🔢 **Neuronas por capa oculta:**
- Pequeño: {mitad} - {n_caracteristicas} neuronas
- Mediano: {n_caracteristicas} - {doble} neuronas  
- Grande: {doble} - {cuadruple} neuronas

📚 **Número de capas:**
- 1-2 capas: Problemas simples (una sola capa oculta ya aprende fronteras curvas; si las clases se separan con una recta, ni siquiera hace falta)
- 2-3 capas: Problemas moderadamente complejos (recomendado para empezar)
- 4+ capas: Problemas muy complejos (requiere muchos datos)

⚖️ **Balance capacidad vs. datos:**
- Más parámetros que datos → riesgo de sobreajuste
- Tu dataset: {n_muestras} muestras, mantén parámetros < {max_parametros}
