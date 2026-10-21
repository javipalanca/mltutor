<!-- NUEVO (D43, Fase 3): no existe en la versión Python, que ofrecía Pickle, reglas .txt y código .py sin explicar la diferencia. Pestaña 💾 Exportar, expander cerrado debajo de los botones de "#### Formatos de Exportación". Título: mensajes.toml [exportar] ayuda_expander. Nombres de entrada/salida del ONNX: contrato C11 de fase2-regresion.md (float_input; label + probabilities o variable) y D47 de fase3-arboles.md. -->
**💾 Exportar a ONNX** — el árbol **ya entrenado**, en un formato abierto que no depende de MLTutor ni de
scikit-learn. Sirve para usarlo en otra aplicación (Python, C#, Java, JavaScript…). En Python se carga con
`onnxruntime`:

```python
import numpy as np
import onnxruntime as ort

sesion = ort.InferenceSession("arbol_decision.onnx")
entrada = sesion.get_inputs()[0].name  # "float_input"
# Una fila por muestra, con las características en el mismo orden que en MLTutor
X = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)

etiquetas, probabilidades = sesion.run(None, {entrada: X})  # árbol de clasificación
print(etiquetas, probabilidades)
# Árbol de regresión: (prediccion,) = sesion.run(None, {entrada: X})
```

ONNX trabaja en `float32`, igual que scikit-learn al recorrer el árbol, y MLTutor guarda cada umbral como el
mayor `float32` que no lo supera: las muestras van por las mismas ramas y las clases coinciden exactamente.
Los valores de regresión y las probabilidades pueden diferir de MLTutor a partir de la séptima cifra
significativa.

**📦 Paquete Python (.zip)** — el árbol completo (`modelo.json`: nodos, umbrales, impurezas y valores) y un
script, `reconstruir_modelo.py`, que crea el `DecisionTreeClassifier` o `DecisionTreeRegressor` de
scikit-learn **sin reentrenar**, comprueba que predice lo mismo que MLTutor y lo guarda en
`decision_tree_model.pkl`, el archivo que usa el «Código para Usar el Modelo» de esta pestaña. Elígelo si
quieres seguir trabajando con el árbol en scikit-learn (dibujarlo con `plot_tree`, calcular otras métricas…).

**📝 Exportar Reglas como Texto** — el árbol como una lista de reglas «si… entonces…» (`export_text`), para
leerlo, imprimirlo o incluirlo en un informe. No sirve para predecir.

**🐍 Exportar como Código Python** — una función `predecir_clase(...)` o `predecir_valor(...)` hecha solo de
`if`/`else`, sin scikit-learn ni ninguna otra biblioteca. Se puede copiar en cualquier programa de Python y
da exactamente las mismas predicciones que el modelo (los umbrales tienen muchos decimales por eso; lo
explica la vista previa).

**📄 Código de Entrenamiento** — un script que **repite el entrenamiento** con scikit-learn: los mismos
datos, el mismo split (`random_state=42`), el mismo árbol, las mismas métricas e importancias y la predicción
de una muestra con su camino de decisión. Úsalo para aprender a hacerlo tú y como punto de partida para
probar cambios.

⚠️ **Errores típicos al usar el modelo fuera de la app**

- **Cambiar el orden o el número de columnas.** El árbol solo sabe de posiciones: si intercambias dos
  características, sigue respondiendo, pero a preguntas equivocadas y sin dar ningún error.
- **Olvidar el preprocesado.** Las columnas de texto (sexo, puerto de embarque…) se pasaron a números en orden
  alfabético; los datos nuevos deben codificarse igual.
- **Cargar archivos `.pkl` de otras personas.** Abrir un pickle puede ejecutar código: carga solo los que
  hayas generado tú.
