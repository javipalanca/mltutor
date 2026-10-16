<!-- mltutor/algorithms/export.py — st.info bajo st.subheader("💡 Información Adicional") al final de 💾 Exportar. CORREGIDO (D75, revisión de F5d, 2026-10-16): Python recomendaba los botones "Modelo TensorFlow" y "Modelo Completo", que en Rust no existen (ahora van dentro del paquete Keras), y "Código Python", que era una plantilla sin los pesos. Se mantienen la estructura, los cuatro puntos y las consideraciones de versión. Python:
**Recomendaciones para el uso del modelo:**

1. **Modelo TensorFlow**: Ideal para integrar en aplicaciones que ya usan TensorFlow
2. **Modelo Completo**: Incluye preprocesadores, perfecto para producción
3. **Código Python**: Para entender la implementación o crear versiones optimizadas
4. **Metadatos**: Para documentación y seguimiento del modelo

**Consideraciones de versión:**
- TensorFlow versión utilizada en entrenamiento
- Compatibilidad con versiones futuras
- Dependencias del entorno de producción
-->
**Recomendaciones para el uso del modelo:**

1. **Paquete Keras**: Ideal para integrar en aplicaciones que ya usan TensorFlow; al ejecutarlo genera también el modelo completo con el escalador (`modelo_completo.pkl`)
2. **ONNX**: Incluye el escalado y no necesita Python ni TensorFlow, perfecto para usar la red en otras aplicaciones
3. **Código Python**: Para entender la implementación (el predictor NumPy) o repetir el entrenamiento con Keras
4. **Metadatos**: Para documentación y seguimiento del modelo

**Consideraciones de versión:**
- `reconstruir_modelo.py` se ha probado con TensorFlow 2.16 y Keras 3.12, y antes de guardar comprueba él mismo que la red da las mismas salidas que MLTutor; si cambias de versión, fíjate en ese mensaje
- Los archivos `.pkl` dependen de la versión de scikit-learn con la que se crean: cárgalos con la misma
- Dependencias del entorno de producción: el ONNX solo necesita `onnxruntime`
