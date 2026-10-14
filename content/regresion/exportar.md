<!-- NUEVO (D1/D28, Fase 2): no existe en la versión Python, que solo ofrecía "📄 Generar Código" y "📥 Descargar Modelo (Pickle)". Pestaña 💾 Exportar, debajo de los tres botones. Expander cerrado. Título: "ℹ️ ¿Qué formato elijo?" (mensajes.toml [exportar] ayuda_expander). Nombres de entrada/salida del ONNX: contrato C11 de fase2-regresion.md (float_input; label + probabilities o variable). -->
**📄 Generar Código** — un script de Python que **repite el entrenamiento** con scikit-learn:
carga los mismos datos, hace el mismo split (`random_state=42`), entrena el mismo modelo y
calcula las mismas métricas. Úsalo para aprender a hacerlo tú y como punto de partida para
probar cambios (escalar, otras características, otro modelo).

**💾 Exportar a ONNX** — el modelo **ya entrenado**, en un formato abierto que no depende de
MLTutor ni de scikit-learn. Sirve para usarlo en otra aplicación (Python, C#, Java, JavaScript…).
En Python se carga con `onnxruntime`:

```python
import numpy as np
import onnxruntime as ort

sesion = ort.InferenceSession("regresion_logistica.onnx")
entrada = sesion.get_inputs()[0].name  # "float_input"
# Una fila por muestra, con las características en el mismo orden que en MLTutor
X = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)

etiquetas, probabilidades = sesion.run(None, {entrada: X})  # regresión logística
print(etiquetas, probabilidades)
# Regresión lineal: (prediccion,) = sesion.run(None, {entrada: X})
```

ONNX calcula en `float32`, así que los resultados pueden diferir de MLTutor a partir de la
sexta o séptima cifra significativa.

**📦 Paquete Python (.zip)** — los parámetros aprendidos (`modelo.json`) y un script,
`reconstruir_modelo.py`, que crea el objeto `LinearRegression` o `LogisticRegression` de
scikit-learn **sin reentrenar**, comprueba que predice lo mismo que MLTutor y lo guarda como
`modelo.pkl`. Elígelo si quieres seguir trabajando con el modelo en scikit-learn.

⚠️ **Errores típicos al usar el modelo fuera de la app**

- **Cambiar el orden o el número de columnas.** El modelo solo sabe de posiciones: si
  intercambias dos características, predice con los coeficientes equivocados sin dar ningún error.
- **Olvidar el preprocesado.** Las columnas de texto (sexo, puerto de embarque…) se pasaron a
  números en orden alfabético; los datos nuevos deben codificarse igual.
- **Abrir archivos `.pkl` de origen desconocido.** Cargar un pickle puede ejecutar código:
  carga solo los que hayas generado tú.
