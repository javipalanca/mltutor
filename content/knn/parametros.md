<!-- NUEVO (Fase 4, como content/arboles/parametros.md en F3): no existe en la versión Python, que solo tenía el expander "ℹ️ ¿Qué es K-Nearest Neighbors?" de la cabecera. Pestaña 🏋️ Entrenamiento, expander cerrado debajo de los 4 controles de "### Parámetros del Modelo". Título: mensajes.toml [entrenamiento] parametros_expander. Cifras medidas con scikit-learn 1.7.2 y el split de la app (test_size=0.3, random_state=42); los dos fragmentos se han ejecutado tal cual y coinciden con tests/parity/knn/breast_cancer_k5_uniform_minkowski.json (prueba 0,959) e iris_k5_*.json (prueba 1,000). CORREGIDO (revisión ML de F4, 2026-10-15): «Al subir K, la exactitud en entrenamiento baja y la de prueba sube» pasa a «Aquí, al subir K…» y se añade que a partir de cierto K la de prueba vuelve a bajar. Reglas de pesos y de distancia 0: fase4-knn.md §3 C17.5, igual que crates/mltutor-core. -->
**Cómo predice un KNN.** Para una muestra nueva, el modelo calcula su distancia a **todas** las muestras de
entrenamiento, se queda con las $K$ más cercanas (sus *vecinos*) y decide con ellas: en clasificación, cada
vecino vota por su clase; en regresión, se promedian sus valores. «Entrenar» no aprende nada: solo guarda las
muestras. Por eso se le llama modelo *perezoso*: todo el trabajo se hace al predecir.

#### 🔢 Vecinos (K): sesgo frente a varianza

- **K pequeño** (1, 2, 3): cada predicción depende de muy pocas muestras. El modelo sigue cada detalle de
  los datos, también el ruido: fronteras irregulares, islas alrededor de muestras aisladas y **sobreajuste**
  (*varianza* alta: con otro split saldría otro modelo).
- **K grande**: se promedia sobre muchas muestras. Las fronteras se suavizan y el ruido pesa menos, pero si $K$
  crece demasiado el modelo deja de ver los detalles reales (*sesgo* alto) y acaba prediciendo casi siempre la
  clase mayoritaria o la media.

Con $K = 1$ la exactitud en entrenamiento es casi siempre del 100 %: cada muestra de entrenamiento es su
propio vecino más cercano. Ese 100 % no dice nada; lo que importa es la exactitud en **prueba**. En
🔬 Cáncer de mama:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for k in [1, 5, 20]:
    modelo = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    print(f"K={k:2d}  entrenamiento {modelo.score(X_train, y_train):.3f}  "
          f"prueba {modelo.score(X_test, y_test):.3f}")
```

```
K= 1  entrenamiento 1.000  prueba 0.936
K= 5  entrenamiento 0.922  prueba 0.959
K=20  entrenamiento 0.915  prueba 0.965
```

Aquí, al subir $K$, la exactitud en entrenamiento baja y la de prueba sube: el modelo memoriza menos y
generaliza mejor. No es una regla general: a partir de cierto $K$ la de prueba vuelve a bajar, porque el modelo
promedia demasiado (el sesgo de arriba). **Cuidado:** elegir $K$ mirando la exactitud de prueba es una forma de **fuga de datos**, porque el
conjunto de prueba deja de ser independiente. Fuera de la app, $K$ se elige con validación cruzada sobre el
entrenamiento (`GridSearchCV`) y la prueba se usa una sola vez al final. En clasificación binaria se suele
preferir un $K$ impar para que no haya empates de votos (si los hay, scikit-learn elige la clase de código más
bajo).

#### ⚖️ Pesos: `uniform` o `distance`

- **`uniform`**: los $K$ vecinos votan igual. La probabilidad de una clase es la proporción de vecinos de esa
  clase (con $K = 5$, solo puede valer 0, 0,2, 0,4, 0,6, 0,8 o 1).
- **`distance`**: cada vecino pesa $w_i = 1/d_i$, así que los más cercanos deciden más. La probabilidad de una
  clase es la suma de los pesos de sus vecinos dividida por la suma de todos los pesos; en regresión, la
  predicción es la media ponderada $\hat y = \sum_i w_i\,y_i \,/\, \sum_i w_i$.
- **Distancia 0** (regla de scikit-learn): $1/0$ no existe. Si la muestra coincide exactamente con alguna de
  entrenamiento, **solo cuentan las que están a distancia 0** (pesan 1) y el resto pesa 0.

Por esa regla, con `distance` la exactitud en **entrenamiento** es del 100 % con cualquier $K$ (salvo
muestras repetidas con clases distintas): cada muestra está a distancia 0 de sí misma y decide sola. En
🌸 Iris, con $K = 5$, `uniform` da 0,952 en entrenamiento y `distance`, 1,000; en prueba, ambas aciertan
el 100 %. De nuevo, compara siempre en prueba.

#### 📐 Métrica: cómo se mide la distancia

Entre dos muestras $\mathbf{x}$ y $\mathbf{z}$ con $n$ características:

- **`euclidean`** (euclídea, «en línea recta»): $d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{j=1}^{n} (x_j - z_j)^2}$.
- **`manhattan`** (de taxi, por calles en cuadrícula): $d(\mathbf{x}, \mathbf{z}) = \sum_{j=1}^{n} |x_j - z_j|$.
  Una diferencia grande en una sola característica pesa menos que en la euclídea, que la eleva al cuadrado.
- **`minkowski`** generaliza las dos: $d(\mathbf{x}, \mathbf{z}) = \left(\sum_{j=1}^{n} |x_j - z_j|^p\right)^{1/p}$,
  con $p = 1$ (Manhattan) o $p = 2$ (euclídea). La app usa el valor por defecto de scikit-learn, $p = 2$, así
  que **`minkowski` y `euclidean` dan exactamente el mismo modelo**.

⚠️ **Las tres métricas suman diferencias en las unidades de cada columna**: la característica con los números
más grandes domina la distancia. Lo explica «📏 KNN sin escalar».
