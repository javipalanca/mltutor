<!-- NUEVO (D57, Fase 4; como content/regresion/exportar.md y content/arboles/exportar.md): no existe en la versión Python, que ofrecía "📄 Generar Código" (el fragmento de Regresión, con un `model` sin definir) y un pickle llamado linear_regression_model.pkl. Pestaña 💾 Exportar, expander cerrado debajo de los tres botones. Título: mensajes.toml [exportar] ayuda_expander. ONNX: grafo propio de fase4-knn.md D57 (entrada float_input float32 [N, n]; salidas label + probabilities o variable; las muestras de entrenamiento van dentro en float64). Cifras de tests/parity/knn/*.json onnx_proposal_D57: 0 etiquetas distintas de scikit-learn en 18 casos y 2 en Titanic (empates); knn.onnx de California ≈ 1 MB. CORREGIDO (revisión ML de F4, 2026-10-15): «ONNX puede quedarse con la otra… salvo en 2 muestras de Titanic» presentaba el caso de prueba como cota general. El grafo usa TopK, que es estable: entre muestras empatadas se queda siempre con la de índice menor, que no es la regla de la app ni la de scikit-learn (kd-tree o montículo de la fuerza bruta). Con datos discretos (🚢 Titanic, `sex` de 🩺 Diabetes, 🔢 Dígitos) los empates son frecuentes y puede discrepar a menudo; el float32 de la entrada se añade como segunda causa. También se añade la recomendación de usar threadpool_limits(1) con el paquete. -->
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

⚠️ **El ONNX desempata de otra forma.** Cuando varias muestras de entrenamiento están a la misma distancia que
el $K$-ésimo vecino y no caben todas, el ONNX se queda siempre con **la que aparece antes** en los datos de
entrenamiento. Ni la app ni scikit-learn usan esa regla (ver «🤝 Empates entre vecinos»), así que en esas
muestras el ONNX puede elegir otros vecinos y dar otra predicción. Con medidas continuas (🌸 Iris, 🏠 California)
casi nunca pasa, pero con datos discretos o repetidos (🚢 Titanic, el `sex` de 🩺 Diabetes, los píxeles de
🔢 Dígitos) los empates son frecuentes y las discrepancias pueden ser habituales. Además, la entrada es
`float32` (5.1 se guarda como 5.0999999…), así que las distancias cambian en la séptima cifra y un empate casi
exacto puede romperse hacia el otro lado. Si necesitas los mismos vecinos que la app, usa el
📦 Paquete Python. El ONNX solo predice: no devuelve los vecinos.

**📦 Paquete Python (.zip)** — las muestras de entrenamiento y los hiperparámetros (`modelo.json`) y un script,
`reconstruir_modelo.py`, que crea el KNN de scikit-learn con esas mismas muestras **en el mismo orden** (el
orden decide los empates), comprueba que elige los mismos vecinos y predice lo mismo que MLTutor y lo guarda
como `modelo_knn.pkl`. Elígelo si quieres seguir trabajando con el modelo en scikit-learn (por ejemplo, para
usar `kneighbors`). Para que los empates se resuelvan como en la app, predice dentro de
`with threadpool_limits(1):` (de `threadpoolctl`), como hace el script.

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
