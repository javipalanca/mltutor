<!-- mltutor/viz/nn.py — expander "💡 ¿Qué verás aquí?" (cerrado) en 📉 Análisis de Activaciones. -->
Esta pestaña muestra **cómo se activan las neuronas internas** y detecta:

* 💀 **Neuronas muertas:** Nunca se activan (todas sus salidas ≈ 0 en ReLU) → No aportan nada.
* 😴 **Neuronas casi constantes:** Producen (casi) siempre el mismo valor → Muy poca contribución.
* 🔴 **Saturación:** Muchas activaciones muy altas (por ejemplo >0.99 en sigmoides) → Gradientes débiles.

**¿Por qué ocurren las neuronas muertas?**
- Learning rate alto que empuja los pesos a regiones sin recuperación
- Inicialización poco favorable
- Demasiada regularización (L1/L2 / dropout mal configurado)
- Arquitectura sobredimensionada

**Cómo mitigarlo:** Reduce LR, elimina capas innecesarias, cambia inicialización o usa activaciones como LeakyReLU / ELU.
