<!-- mltutor/viz/nn.py — expander "💡 ¿Qué verás aquí?" (cerrado) en 📉 Análisis de Activaciones. NUEVO en Rust (revisión de contenido de F5, 2026-10-16, §12.10 de fase5-redes.md): el párrafo final sobre tanh/sigmoid y las muestras analizadas. La regla de la app es "todas sus activaciones ≤ umbral (0)": con tanh significa "siempre negativa" (no está muerta: su gradiente no es 0) y con sigmoid no ocurre nunca (siempre > 0); la saturación solo se mide si la activación máxima es ≤ 1,2, así que no cuenta la de tanh hacia −1. -->
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

**Ten en cuenta:** el análisis usa solo las primeras muestras de prueba (las que elijas en ⚙️ Parámetros de Análisis), así que una neurona «muerta» puede activarse con otras muestras. La regla «muerta = todas sus salidas ≤ umbral» está pensada para ReLU: con **tanh** marca neuronas que siempre dan un valor negativo (siguen aprendiendo, porque su gradiente no es 0) y con **sigmoid** no marca ninguna, porque su salida siempre es mayor que 0. La saturación solo se mide si la activación máxima es como mucho 1,2 (sigmoid y tanh), y no cuenta los valores cercanos a −1 de tanh.
