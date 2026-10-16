<!-- NUEVO (D75, Fase 5; como content/regresion/exportar.md, content/arboles/exportar.md y content/knn/exportar.md): no existe en la versión Python, que tenía "🤖 Modelo TensorFlow" (SavedModel/HDF5/TFLite), "📊 Modelo Completo" (pickle), "📝 Código Python" (una plantilla que devolvía ceros) y "📋 Metadatos". Pestaña 💾 Exportar, expander cerrado debajo de los cuatro botones. Título: mensajes.toml [exportar_rust] ayuda_expander. ONNX: grafo de fase5-redes.md §3 C28 (entrada float_input float32 [N, F] SIN escalar → Sub/Div con el escalador → Gemm + activación por capa; salidas label + probabilities o variable; revisión ML de F5a, 2026-10-16: label = código de la clase en el dataset mediante Gather sobre el argmax o sobre p > 0,5, y en binaria probabilities [N, 2] = [1 − p, p]). El fragmento se ha ejecutado con onnxruntime sobre ese grafo construido con los pesos finales de tests/parity/nn/exact/iris_default.json (tools/golden_nn.py::onnx_proposal_model): da [0] y [[0.7971 0.1325 0.0704]], igual que la red en NumPy. Paquete: content/code/paquete_modelo_red.py.j2, red_predictor_numpy.py.j2 y paquete_leeme_red.txt.j2 (verificados en expected.export_package). Revisión de F5d (2026-10-16): el resumen inicial «En resumen» (también en mensajes.toml [exportar_rust] *_cuando) y la precisión de que los metadatos no contienen los pesos; el fragmento ONNX se ha vuelto a ejecutar con onnxruntime y da la misma salida. -->
**En resumen:** para seguir trabajando en Python con TensorFlow/Keras, el **📦 paquete Keras**; para predecir
desde otra aplicación o sin instalar TensorFlow, el **💾 ONNX**; para entender lo que calcula la red o repetir
el entrenamiento, el **📝 código Python**; para documentar el modelo, los **📋 metadatos**. El ONNX y el
predictor NumPy llevan dentro el escalado y reciben los datos en sus unidades originales; el modelo `.keras`
que genera el paquete, no (ver los errores típicos).

**📦 Paquete Keras (.zip)** — para seguir trabajando con la red en TensorFlow/Keras. Contiene `modelo.json` (la
arquitectura, los pesos de cada capa, la media y la desviación del escalador y unas filas de comprobación) y
`reconstruir_modelo.py`, que construye **la misma red** en Keras con esos pesos, comprueba que da las mismas
salidas que MLTutor y la guarda como `.keras`, `.h5`, SavedModel y `.tflite`, junto con `scaler.pkl` y
`modelo_completo.pkl`. Es la única opción que necesita TensorFlow instalado. No vuelve a entrenar: si quieres
reentrenar, usa el script de entrenamiento de 📝 Código Python.

**💾 ONNX** — la red lista para usar en un formato abierto, sin Python ni TensorFlow (onnxruntime existe para
Python, C#, Java y JavaScript, y `onnx2tf` lo convierte a TensorFlow Lite). **El escalado va dentro**: el
archivo recibe los valores en sus unidades originales, lo resta y divide por la media y la desviación de la
app, y aplica cada capa (una multiplicación de matrices, `Gemm`, y su activación). Las redes de clasificación
devuelven dos salidas:

- `label`: la clase predicha, con el **mismo código que usa MLTutor** para esa clase (el número de la lista de
  clases del `LEEME.txt` y de 📋 Metadatos). En multiclase es la neurona con la salida más alta; en binaria,
  la clase 1 si la salida de la neurona supera 0,5.
- `probabilities`: una columna por clase. En multiclase, la salida de cada neurona; en binaria, donde la red
  tiene una sola neurona con salida `p`, las dos columnas `[1 − p, p]`.

En Python:

```python
import numpy as np
import onnxruntime as ort

sesion = ort.InferenceSession("red_neuronal.onnx")
entrada = sesion.get_inputs()[0].name  # "float_input"
# Una fila por muestra, con las características en el mismo orden que en MLTutor y SIN escalar:
# el ONNX ya incluye el escalado de la app
X = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)

etiquetas, probabilidades = sesion.run(None, {entrada: X})  # red de clasificación
print(etiquetas, np.round(probabilidades, 4))
# Red de regresión: (prediccion,) = sesion.run(None, {entrada: X})
```

Con la red por defecto de 🌸 Iris (que, como explica ⚙️ Entrenamiento, ha entrenado poco) sale algo como
`[0] [[0.7971 0.1325 0.0704]]`: clase 0 (setosa), con una probabilidad de solo el 80 % para una flor que es
claramente setosa. Los números exactos dependen de tu entrenamiento.

**📝 Código Python** — dos scripts para aprender:

- **El predictor en NumPy**: la red entrenada escrita en 30 líneas, sin TensorFlow. Lee `modelo.json`, escala la
  entrada y, capa a capa, calcula `a @ kernel + bias` y la activación. Es la mejor forma de ver que una red
  neuronal, una vez entrenada, solo son multiplicaciones de matrices y funciones sencillas. El dropout no
  aparece: solo actúa al entrenar.
- **El script de entrenamiento con Keras**: repite lo que hace la app (mismos datos, mismo escalado, mismo
  split, misma red, mismos callbacks) y calcula las métricas de 📈 Evaluación. Keras parte de otros pesos
  iniciales (ver «¿Por qué hay una semilla?»), así que sus cifras serán parecidas, no idénticas.

**📋 Metadatos** — un JSON con la arquitectura, los hiperparámetros, las características, las clases y la fecha
del entrenamiento. Sirve para documentar qué modelo es cuál; no contiene los pesos, así que no permite predecir.

⚠️ **Errores típicos al usar la red fuera de la app**

- **Escalar dos veces o ninguna.** El ONNX y el predictor NumPy esperan los valores **sin escalar** (escalan
  ellos). El modelo de Keras del paquete, en cambio, espera los datos **ya escalados**: aplica antes
  `scaler.transform`.
- **Cambiar el orden o el número de columnas.** La red solo conoce posiciones: si intercambias dos
  características, multiplica cada valor por los pesos equivocados sin dar ningún error.
- **Olvidar el preprocesado.** Las columnas de texto (sexo, puerto de embarque…) se pasaron a números en orden
  alfabético; los datos nuevos deben codificarse igual.
- **Esperar exactamente las cifras de la app con valores muy grandes.** El ONNX calcula en `float32` (unos 7
  dígitos), también el escalado. Con columnas de valores grandes y muy desplazados del 0 (años, *timestamps*,
  códigos postales), restar la media pierde precisión y las salidas pueden diferir algo de las de MLTutor, que
  escala en `float64`.
- **Leer la salida tanh como una probabilidad.** Si elegiste tanh en la salida, `probabilities` es la salida
  de la red, que puede ser negativa.
- **Usar tensorflow-metal en un Mac.** Con la GPU de Apple, `model.predict` puede dar salidas incorrectas tras
  cargar pesos (comprobado con tensorflow-metal 1.2.0). Los scripts del paquete fuerzan la CPU; haz lo mismo en
  tu código (`tf.config.set_visible_devices([], "GPU")`).
- **Abrir archivos `.pkl` de origen desconocido.** Cargar un pickle puede ejecutar código: carga solo los que
  hayas generado tú.
