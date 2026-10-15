<!-- NUEVO (D62, Fase 4): no existe en la versión Python, que entrena el KNN SIN escalar (no hay StandardScaler en ninguna parte de la lección) mientras intro.md y distancias-interpretar.md avisan de que KNN es sensible a la escala. D62 mantiene los datos sin escalar por paridad; este texto explica qué supone. Pestaña 🏋️ Entrenamiento, expander cerrado debajo de parametros.md. Título: mensajes.toml [entrenamiento] escala_expander. Cifras medidas con scikit-learn 1.7.2 y el split de la app (test_size=0.3, random_state=42): exactitud 0,741 = tests/parity/knn/wine_k5_uniform_minkowski.json; importancias por permutación del mismo fixture (proline 0,370, magnesium 0,043, el resto ≤ 0,002 en valor absoluto); rangos de load_wine(). Los tres fragmentos se han ejecutado tal cual; Dígitos con K=5: 0,993 en prueba. -->
**MLTutor entrena el KNN con los datos tal cual, sin escalar**, igual que la versión original de la app.
Así puedes ver un error muy común en la práctica.

#### Por qué la escala importa tanto en KNN

La distancia suma las diferencias de cada característica **en sus propias unidades**. En 🍷 Vino, `proline`
va de 278 a 1680 (un rango de 1402), `magnesium` tiene un rango de 92 y `hue`, de 1,23. Una diferencia
corriente en `proline` (unos cientos) pesa miles de veces más en la distancia que la mayor diferencia
posible en `hue`. En la práctica, el KNN busca los vecinos casi solo por `proline` e ignora las otras 12
características.

Se ve en 🔍 Características: con $K = 5$, la importancia por permutación de `proline` es 0,370, la de
`magnesium` 0,043 y la de las otras 11 está entre −0,002 y 0,002: barajarlas apenas cambia las
predicciones. No significa
que no sirvan para distinguir los vinos: el modelo no las «ve».

#### Qué pasa al escalar

`StandardScaler` resta a cada columna su media y la divide por su desviación estándar, $z = (x - \mu)/\sigma$,
de modo que todas pasan a pesar parecido:

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sin_escalar = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
# El Pipeline aprende la media y la desviación SOLO con X_train y las aplica a X_test
escalado = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
escalado.fit(X_train, y_train)

print(f"Sin escalar: {sin_escalar.score(X_test, y_test):.3f}")
print(f"Escalado:    {escalado.score(X_test, y_test):.3f}")
```

```
Sin escalar: 0.741
Escalado:    0.963
```

La exactitud en prueba pasa del 74 % (lo que ves en la app) al 96 %. Escalar no siempre mejora: en
🔬 Cáncer de mama, con $K = 5$, sale 0,959 con y sin escalar. Pero sin escalar, el resultado depende de
algo tan arbitrario como las unidades (medir en gramos o en kilos cambia los vecinos).

⚠️ **Errores típicos**

- **Escalar antes de dividir** (`StandardScaler().fit(X)` con todos los datos): la media y la desviación se
  calculan también con las muestras de prueba y la información de prueba se filtra al modelo (**fuga de
  datos**). Ajusta el escalador solo con el entrenamiento; un `Pipeline` lo hace por ti, también dentro de
  la validación cruzada.
- **Olvidar escalar los datos nuevos** al predecir. Con un `Pipeline` guardado se aplica solo.
- **Tratar como números las columnas que son códigos** (en 🚢 Titanic, el puerto de embarque es 0, 1 o 2 por
  orden alfabético): escaladas o no, el KNN considera que los puertos 0 y 2 están el doble de lejos que el 0
  y el 1. Para categorías sin orden es mejor `OneHotEncoder`.

#### 🌌 La maldición de la dimensionalidad

Con muchas características aparece otro problema: **todas las distancias se parecen**. En un cubo con puntos
al azar, la distancia al vecino más cercano se acerca a la distancia al más lejano a medida que crece el
número de dimensiones $d$:

```python
import numpy as np
from sklearn.metrics import pairwise_distances

rng = np.random.default_rng(42)
for d in [2, 13, 30, 64, 1000]:
    X = rng.uniform(size=(1000, d))   # 1000 puntos al azar en un cubo de d dimensiones
    Q = rng.uniform(size=(100, d))    # 100 consultas
    D = pairwise_distances(Q, X)
    print(f"d={d:4d}  más cercano / más lejano = {np.mean(D.min(axis=1) / D.max(axis=1)):.3f}")
```

```
d=   2  más cercano / más lejano = 0.016
d=  13  más cercano / más lejano = 0.328
d=  30  más cercano / más lejano = 0.509
d=  64  más cercano / más lejano = 0.630
d=1000  más cercano / más lejano = 0.892
```

En 2 dimensiones el vecino más cercano está 60 veces más cerca que el más lejano; con 1000, casi a la misma
distancia. Si «cercano» y «lejano» apenas se distinguen, los vecinos dejan de ser informativos. Por eso KNN
funciona mejor con pocas características relevantes: quita las que no aportan, o reduce la dimensión (PCA)
antes, y necesitarás muchas más muestras cuantas más dimensiones tengas. (Los datos reales no están
repartidos al azar: 🔢 Dígitos, con 64 píxeles, funciona bien con KNN porque las imágenes de un mismo dígito
ocupan una región pequeña del espacio: con $K = 5$ acierta el 99 % en prueba.)
