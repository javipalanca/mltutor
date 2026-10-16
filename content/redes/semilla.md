<!-- NUEVO (D70, Fase 5): no existe en la versión Python, que no fijaba ninguna semilla (cada entrenamiento daba otro resultado). Pestaña ⚙️ Entrenamiento, expander cerrado debajo del control "🎲 Semilla" del expander avanzado. Título: mensajes.toml [entrenamiento] semilla_expander. Cifras: tests/parity/nn/stat/iris_default.json (la app Python con keras.utils.set_random_seed(0..4), dropout 0,2, barajado): exactitud en prueba 0,633 / 0,733 / 0,767 / 0,667 / 0,733. El fragmento se ha ejecutado tal cual (TensorFlow 2.16.2, Keras 3.12, scikit-learn 1.7.2, CPU) y da exactamente esas cinco cifras. Fuentes de azar del motor: fase5-redes.md §3 C21 (Glorot, máscaras de dropout y permutación de cada época, con un generador propio sembrado). -->
**¿Por qué hay una semilla?** Entrenar una red tiene tres pasos que dependen del azar:

1. **Los pesos iniciales.** Antes de aprender, cada peso se sortea (inicialización Glorot uniforme: un valor
   al azar entre $-l$ y $l$, con $l = \sqrt{6/(\text{entradas} + \text{salidas})}$; los sesgos empiezan en 0).
2. **El dropout.** En cada lote se apaga al azar una fracción de las neuronas ocultas (la *tasa de dropout*).
3. **El orden de los lotes.** En cada época se barajan las filas de entrenamiento antes de repartirlas en
   lotes (la validación no se baraja: son siempre las últimas filas).

La **semilla** es el número con el que arranca el generador de números aleatorios. Con la misma semilla, los
tres sorteos salen iguales y obtienes **exactamente la misma red** cada vez que entrenas: puedes cambiar un
hiperparámetro y saber que la diferencia se debe a ese cambio y no a la suerte. MLTutor usa 42 por defecto,
como el `random_state=42` del resto de lecciones.

**Cambiarla sirve para ver cuánto pesa el azar.** Con la configuración por defecto de 🌸 Iris, la exactitud
en prueba de cinco entrenamientos que solo se diferencian en la semilla va del 63 % al 77 %:

```python
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # en la CPU, como MLTutor
import keras  # noqa: E402
from sklearn.datasets import load_iris  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)  # como MLTutor (ver «Sobre el escalado de los datos»)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
Y_train = keras.utils.to_categorical(y_train, 3)
Y_test = keras.utils.to_categorical(y_test, 3)


def entrenar(semilla):
    keras.utils.set_random_seed(semilla)  # pesos iniciales, dropout y orden de los lotes
    modelo = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(10, activation="relu"), keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="relu"), keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation="softmax"),
    ])
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                   loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
    ]
    modelo.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.2,
               callbacks=callbacks, verbose=0)
    return modelo.evaluate(X_test, Y_test, verbose=0)[1]  # exactitud en prueba


for semilla in [0, 1, 2, 3, 4]:
    print(f"semilla {semilla}: exactitud en prueba = {entrenar(semilla):.3f}")
```

```
semilla 0: exactitud en prueba = 0.633
semilla 1: exactitud en prueba = 0.733
semilla 2: exactitud en prueba = 0.767
semilla 3: exactitud en prueba = 0.667
semilla 4: exactitud en prueba = 0.733
```

Son 30 muestras de prueba, así que cada acierto vale 3,3 puntos: entre la peor y la mejor semilla hay 4 flores
de diferencia, sin tocar nada más.

⚠️ **Errores típicos**

- **Elegir la semilla que mejor sale** y presentarla como el resultado del modelo. Eso es ajustar al azar: con
  datos nuevos, el resultado volverá a la media. Si comparas dos configuraciones, entrena cada una con varias
  semillas y compara las medias (y cuánto varían).
- **Esperar las mismas cifras en Keras.** El script de 💾 Exportar usa la semilla que elijas aquí, pero Keras
  tiene su propio generador: sus pesos iniciales, su dropout y su orden de lotes son otros, así que sus cifras
  serán parecidas a las de MLTutor, no idénticas. Incluso en Keras, otra versión, la GPU o varios hilos pueden cambiar los
  últimos decimales.
- **Creer que una semilla fija hace el modelo mejor.** Solo lo hace **repetible**.
