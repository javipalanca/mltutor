<!-- NUEVO (D57, Fase 4; como content/regresion/exportar.md y content/arboles/exportar.md): no existe en la versión Python, que ofrecía "📄 Generar Código" (el fragmento de Regresión, con un `model` sin definir) y un pickle llamado linear_regression_model.pkl. Pestaña 💾 Exportar, expander cerrado debajo de los tres botones. Título: mensajes.toml [exportar] ayuda_expander. ONNX: grafo propio de fase4-knn.md D57 (entrada float_input float32 [N, n]; salidas label + probabilities o variable; las muestras de entrenamiento van dentro en float64). Cifras de tests/parity/knn/*.json onnx_proposal_D57: 0 etiquetas distintas de scikit-learn en 18 casos y 2 en Titanic (empates); knn.onnx de California ≈ 1 MB. -->
**📄 Código de Entrenamiento** — un script que **repite el entrenamiento** con scikit-learn: los mismos datos
(sin escalar), el mismo split (`random_state=42`), el mismo `KNeighborsClassifier` o `KNeighborsRegressor`,
los mismos vecinos, métricas, distancias entre clases e importancias, y la predicción de una muestra con sus
$K$ vecinos. Úsalo para aprender a hacerlo tú y como punto de partida para probar cambios: escalar con un
`Pipeline` (ver «📏 KNN sin escalar» en 🏋️ Entrenamiento) o elegir $K$ con validación cruzada.

**💾 Exportar a ONNX** — el modelo listo para usar, en un formato abierto que no depende de MLTutor ni de
scikit-learn (Python, C#, Java, JavaScript…). Un KNN no tiene parámetros que guardar: **el archivo lleva dentro
todas las muestras de entrenamiento** y, al predecir, calcula las distancias a todas ellas. Por eso ocupa más
cuantos más datos tengas (unos 6 KB con 🌸 Iris y alrededor de 1 MB con 🏠 California). En Python se carga
con `onnxruntime`:

```python
import numpy as np
import onnxruntime as ort

sesion = ort.InferenceSession("knn.onnx")
entrada = sesion.get_inputs()[0].name  # "float_input"
# Una fila por muestra, con las características en el mismo orden que en MLTutor y SIN escalar
X = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)

etiquetas, probabilidades = sesion.run(None, {entrada: X})  # KNN de clasificación
print(etiquetas, probabilidades)
# KNN de regresión: (prediccion,) = sesion.run(None, {entrada: X})
```

La entrada es `float32` (5.1 se guarda como 5.0999999…), así que las distancias cambian en la séptima cifra.
Casi nunca importa, pero si dos muestras de entrenamiento estaban empatadas a la distancia del $K$-ésimo
vecino, ONNX puede quedarse con la otra (ver «🤝 Empates entre vecinos»): en los casos de prueba de MLTutor, las
etiquetas coinciden con scikit-learn en todos salvo en 2 muestras de 🚢 Titanic. El ONNX solo
predice: no devuelve los vecinos.

**📦 Paquete Python (.zip)** — las muestras de entrenamiento y los hiperparámetros (`modelo.json`) y un script,
`reconstruir_modelo.py`, que crea el KNN de scikit-learn con esas mismas muestras **en el mismo orden** (el
orden decide los empates), comprueba que elige los mismos vecinos y predice lo mismo que MLTutor y lo guarda
como `modelo_knn.pkl`. Elígelo si quieres seguir trabajando con el modelo en scikit-learn (por ejemplo, para
usar `kneighbors`).

⚠️ **Errores típicos al usar el modelo fuera de la app**

- **Cambiar el orden o el número de columnas.** El modelo solo sabe de posiciones: si intercambias dos
  características, calcula las distancias con las columnas equivocadas sin dar ningún error.
- **Escalar los datos nuevos** (o no hacerlo, si escalas al reentrenar). Este modelo se entrenó **sin
  escalar**: pásale los valores en sus unidades originales.
- **Olvidar el preprocesado.** Las columnas de texto (sexo, puerto de embarque…) se pasaron a números en orden
  alfabético; los datos nuevos deben codificarse igual.
- **Compartir el modelo sin pensar en los datos.** El ONNX y el paquete contienen las muestras de
  entrenamiento completas: si los datos son privados, el modelo también lo es.
- **Abrir archivos `.pkl` de origen desconocido.** Cargar un pickle puede ejecutar código: carga solo los que
  hayas generado tú.
