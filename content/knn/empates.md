<!-- NUEVO (D52, Fase 4): no existe en la versión Python. Explica por qué MLTutor reproduce el desempate de scikit-learn (kd-tree portado, tools/knn_reference.py). Pestaña 🏋️ Entrenamiento, expander cerrado debajo de escala.md (y enlazado desde la nota vecinos_nota_empate de 🔮 Predicciones). Título: mensajes.toml [entrenamiento] empates_expander. Cifras de los fixtures (scikit-learn 1.7.2, split de la app): titanic_k5_uniform_minkowski.json (neighbours_test.ties: 55 de 268 filas con empate en el 5.º vecino; en 40 la regla "gana el índice menor" daría otros vecinos), diabetes_k5_uniform_minkowski.json (rejilla age/sex: 5 340 de 11 655 celdas cambian de predicción con esa regla), iris_k5_uniform_minkowski.json (rejilla sepal length/width: 179 de 40 000 celdas cambian de clase); duplicados de Iris en 2D contados con load_iris() (33 filas repetidas, 10 puntos con especies distintas). Regla del motor: fase4-knn.md §3 C17.1. CORREGIDO (revisión ML de F4, 2026-10-15): (1) la fuerza bruta NO desempata por el índice menor: decide el montículo de scikit-learn (un valor menor expulsa la raíz, y cuál de los máximos empatados es la raíz depende de la disposición del montículo); (2) con varios hilos, scikit-learn reparte las filas de entrenamiento entre hilos y fusiona los montículos, así que el resultado depende del número de hilos incluso en el mismo equipo (🔢 Dígitos, K = 20, manhattan, scikit-learn 1.7.2 en un Mac de 12 núcleos: 104 de las 540 filas de prueba tienen otros vecinos que con un hilo; 110 con 2 hilos, 105 con 4, 104 con 8; con threadpool_limits(1) los conjuntos de vecinos coinciden con tools/knn_reference.py en las 540); (3) la frase «como mucho, puede cambiar algún empate en otro sistema operativo» era falsa: puede cambiar en el mismo equipo y con frecuencia en datos discretos. -->
**¿Qué es un empate entre vecinos?** Con $K = 5$, el modelo necesita exactamente cinco vecinos. Si la quinta
muestra más cercana está a la misma distancia que otras (por ejemplo, tres muestras a distancia 2,0 y solo
queda un hueco), hay que elegir cuál entra. **No existe una respuesta «correcta»**: todas están igual de
cerca. Pero la elegida decide la votación y puede cambiar la predicción.

**¿Es raro?** Con medidas continuas casi nunca pasa, pero con características discretas o repetidas es muy
frecuente:

- En 🚢 Titanic muchos pasajeros tienen exactamente los mismos valores (misma clase, sexo, puerto, edad
  rellenada con la mediana…). Con $K = 5$, **55 de las 268** muestras de prueba tienen un empate en el quinto
  vecino.
- En 🩺 Diabetes, `sex` solo toma dos valores. En la 🌐 Superficie de Predicción con `age` y `sex`, casi
  la mitad de los puntos de la rejilla tienen empates que deciden la predicción.
- En 🌸 Iris, las medidas se redondean a 0,1 cm: con solo el largo y el ancho del sépalo, 33 flores repiten
  las medidas de otra, y en 10 de esos puntos hay flores de especies distintas (por ejemplo, 5,8 × 2,7 cm).

**¿Quién gana?** Depende de cómo busque los vecinos scikit-learn (`algorithm="auto"`):

- **Árbol k-d** (si hay 15 características o menos y $K$ es menor que la mitad de las muestras de
  entrenamiento, redondeada hacia abajo): gana la primera muestra que encuentra el recorrido del árbol, y ese
  orden depende de cómo se construyó.
- **Fuerza bruta** (en el resto de casos; por ejemplo, 🔬 Cáncer de mama, con 30 características, o
  🔢 Dígitos, con 64): scikit-learn recorre las muestras en orden y las va metiendo en un *montículo* (una
  estructura que guarda las $K$ mejores y expulsa la peor cuando llega una más cercana). Entre varias
  muestras empatadas, cuál se expulsa depende de cómo esté colocado el montículo en ese momento, así que
  **no siempre gana la que aparece antes**. Además, si el ordenador tiene varios núcleos, scikit-learn
  reparte las muestras entre ellos y luego junta los resultados: **el desempate cambia con el número de
  hilos**, incluso en el mismo equipo. En 🔢 Dígitos con $K = 20$ y Manhattan, 104 de las 540 muestras de
  prueba reciben otros vecinos con 8 hilos que con uno solo. Con la distancia euclídea, los redondeos
  del cálculo rápido que usa scikit-learn pueden deshacer algún empate.

MLTutor copia estas dos reglas, la de la fuerza bruta tal como la ejecuta scikit-learn con **un solo
hilo**. Con una regla más simple («gana siempre la que aparece antes»)
cambiarían los vecinos de 40 de esas 55 muestras de Titanic, la clase de 179 de las 40 000 celdas de la
frontera de Iris y el valor predicho en 5 340 de las 11 655 celdas de la superficie `age`/`sex` de Diabetes
(el 46 %). Gracias a ello, el «📄 Código de Entrenamiento» de 💾 Exportar da en scikit-learn los mismos
vecinos y las mismas predicciones que ves en la app. Para lograrlo, el script limita scikit-learn a un hilo
con `threadpool_limits(1)` (del paquete `threadpoolctl`, que se instala con scikit-learn); sin esa línea, en
un equipo con varios núcleos los empates de la fuerza bruta pueden resolverse de otra forma. Aun así, el
árbol k-d puede elegir otro vecino empatado en otro sistema operativo (ver abajo).

**Empate de votos ≠ empate de distancias.** Aunque los vecinos estén claros, la votación también puede
empatar (con $K = 4$, dos votos para cada clase). Entonces scikit-learn, y MLTutor, eligen la clase con el
código más bajo (la primera de la lista). Un $K$ impar evita este empate en clasificación binaria, pero no
el de distancias.

⚠️ **Lo que hay que recordar**

- Un empate no es un error del modelo: es una ambigüedad de los datos. Si cambia la predicción, esa muestra
  está justo en la frontera y la predicción no es fiable.
- El desempate depende de detalles de implementación: con el árbol k-d, el mismo código de scikit-learn
  puede elegir otro vecino empatado en otro sistema operativo o con otra versión; con la fuerza bruta,
  también con otro número de hilos en el mismo equipo. Con datos discretos esto pasa a menudo, no de
  forma excepcional. Los vecinos no empatados y las distancias sí coinciden siempre.
- Barajar o reordenar las filas de entrenamiento puede cambiar qué muestra gana el empate. Si comparas dos
  modelos, usa los mismos datos en el mismo orden.
