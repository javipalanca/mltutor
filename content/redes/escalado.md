<!-- NUEVO (D72, Fase 5): no existe en la versión Python. Pestaña ⚙️ Entrenamiento, expander cerrado debajo de la caption "📊 Distribución" (y enlazado desde la nota escalado_nota). Título: mensajes.toml [entrenamiento] escalado_expander. Comportamiento de la app (Python y Rust, D72 por paridad): StandardScaler ajustado con TODAS las filas y después train_test_split(test_size=0.2, random_state=42, stratify=y en clasificación); el objetivo de regresión NO se escala (fase5-redes.md §1.2 y C22). Los dos fragmentos se han ejecutado tal cual (scikit-learn 1.7.2) y dan exactamente las salidas mostradas. -->
**Por qué se escala.** Una neurona suma sus entradas multiplicadas por sus pesos. Si una característica va de
0 a 1 y otra de 0 a 1000, la segunda domina la suma al principio y el descenso de gradiente avanza a saltos:
con un mismo learning rate, unos pesos se mueven demasiado y otros casi nada. Por eso MLTutor **estandariza**
cada característica antes de entrenar: le resta su media y la divide por su desviación típica,
$z = (x - \mu)/\sigma$, de modo que todas quedan con media 0 y desviación 1. Al predecir (🔮) hace lo mismo con
los valores que escribes, con la $\mu$ y la $\sigma$ guardadas.

**Lo que hace MLTutor no es lo correcto.** Para que los resultados coincidan con los de la versión original
de la app, MLTutor calcula $\mu$ y $\sigma$ con **todas las filas, incluidas las de prueba**, y divide los datos
después. Es una **fuga de datos**: la prueba debería simular datos que el modelo no ha visto nunca, y aquí
una parte de su información (su media y su dispersión) ya ha entrado en el preprocesado. En 🌸 Iris la
diferencia es pequeña, pero existe:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True, as_frame=True)

# Como MLTutor: el escalador aprende la media y la desviación con TODAS las filas
escalador_app = StandardScaler().fit(X)

# Lo correcto: dividir primero y ajustar el escalador SOLO con el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
escalador = StandardScaler().fit(X_train)
X_train_esc = escalador.transform(X_train)
X_test_esc = escalador.transform(X_test)  # la prueba se transforma, nunca se ajusta

for nombre, m1, m2, s1, s2 in zip(X.columns, escalador_app.mean_, escalador.mean_,
                                  escalador_app.scale_, escalador.scale_):
    print(f"{nombre:18s} media {m1:.3f} → {m2:.3f}   desviación {s1:.3f} → {s2:.3f}")
```

```
sepal length (cm)  media 5.843 → 5.842   desviación 0.825 → 0.837
sepal width (cm)   media 3.057 → 3.048   desviación 0.434 → 0.447
petal length (cm)  media 3.758 → 3.770   desviación 1.759 → 1.761
petal width (cm)   media 1.199 → 1.205   desviación 0.760 → 0.759
```

La división es la misma en los dos casos (con `random_state=42` y `stratify=y`, las filas de prueba solo
dependen del número de filas y de las clases), así que solo cambia el escalado. Con 150 flores la fuga apenas
mueve las cifras; con pocos datos, valores extremos en la prueba o un preprocesado más complejo (rellenar
huecos con la media, elegir características), puede hacer que la exactitud en prueba sea **optimista**.

**Con scikit-learn, un `Pipeline` lo hace solo.** Encadena el escalador y el modelo: `fit` ajusta el escalador
con los datos que recibe (solo los de entrenamiento) y `predict`/`score` lo reutilizan. También funciona
dentro de la validación cruzada, donde hacerlo a mano es fácil de olvidar. Este ejemplo usa la red de
scikit-learn (`MLPClassifier`), no Keras, así que su resultado no es el de MLTutor:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# El Pipeline ajusta el escalador con X_train dentro de fit() y lo reutiliza en score() y predict()
modelo = make_pipeline(StandardScaler(),
                       MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000, random_state=42))
modelo.fit(X_train, y_train)
print(f"Exactitud en prueba: {modelo.score(X_test, y_test):.3f}")
```

```
Exactitud en prueba: 0.967
```

Con Keras no hay `Pipeline`: ajusta el escalador con `X_train` como en el primer fragmento y guárdalo junto al
modelo (el script de 💾 Exportar guarda `scaler.pkl`).

⚠️ **Errores típicos**

- **Ajustar el escalador con todos los datos** (lo que hace MLTutor a propósito, para que puedas compararlo
  con la versión original): la prueba deja de ser independiente.
- **Volver a ajustar el escalador con los datos nuevos** al predecir (`fit_transform` en lugar de `transform`):
  cada lote se escalaría distinto y la red recibiría números que no significan lo mismo.
- **Olvidar el objetivo en regresión.** MLTutor escala las características, pero **no** el valor que hay que
  predecir. En 🩺 Diabetes va de 25 a 346 y la red empieza dando valores cercanos a 0: con un learning rate
  pequeño necesita muchísimas actualizaciones para llegar ahí (ver «¿Por qué la red acierta poco con los
  valores por defecto?»). Fuera de la app es habitual escalar también el objetivo y deshacer el escalado en
  las predicciones.
