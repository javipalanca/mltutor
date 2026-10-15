<!-- NUEVO (Fase 3): no existe en la versión Python, que solo tenía las ayudas cortas de los deslizadores. Pestaña 🏋️ Entrenamiento, expander cerrado debajo de criterios.md. Título: mensajes.toml [entrenamiento] parametros_expander. Cifras medidas con scikit-learn 1.7.2, split de la app (test_size=0.3, random_state=42), criterio por defecto (gini / squared_error). CORREGIDO (revisión ML de F3, 2026-10-15): «para que obtengas exactamente el árbol de scikit-learn» pasa a «el mismo árbol que scikit-learn, salvo empates numéricos en algunos equipos»: la wheel de scikit-learn para Apple Silicon usa FMA, y en algunos empates exactos eso rompe el empate y cambia la estructura (MEMORY.md). -->
**Profundidad máxima** (`max_depth`) es el número máximo de preguntas seguidas desde la raíz hasta una
hoja. Con profundidad 1 el árbol hace una sola pregunta (un «tocón» con dos hojas); con profundidad
$d$ puede tener hasta $2^d$ hojas.

**Muestras mínimas para división** (`min_samples_split`) es el número de muestras de entrenamiento que
necesita un nodo para que se le permita hacer otra pregunta. Si tiene menos, se queda como hoja aunque
no sea puro. Con el valor por defecto, 2, el árbol puede seguir dividiendo hasta tener hojas de una sola
muestra.

Los dos controlan lo mismo: **cuánto se adapta el árbol a los datos de entrenamiento**. Un árbol muy
profundo memoriza los detalles, incluido el ruido, y se **sobreajusta**: acierta casi todo en
entrenamiento y falla con datos nuevos. Un árbol demasiado corto **infraajusta**: es tan simple que no
capta el patrón.

En 🩺 Diabetes (📈 Regresión, `squared_error`) se ve muy claro:

| Profundidad | Hojas | R² en entrenamiento | R² en prueba |
|---|---|---|---|
| 1 | 2 | 0,299 | 0,207 |
| 2 | 4 | 0,440 | **0,355** |
| 3 | 8 | 0,525 | 0,330 |
| 5 | 30 | 0,680 | 0,293 |
| 10 | 176 | 0,951 | 0,035 |

A partir de profundidad 2 el árbol mejora en entrenamiento pero **empeora en prueba**: con 10 niveles
explica el 95 % de la varianza de los datos que ha visto y casi nada (3,5 %) de los nuevos. Subir las
muestras mínimas a 10 con profundidad 10 lo frena un poco (123 nodos en lugar de 351; R² en prueba 0,064).
En 🔬 Cáncer pasa lo mismo, más suave: con profundidad 3 acierta el 96,5 % de la prueba; con 10, el
100 % del entrenamiento y el 94,2 % de la prueba.

**Cómo elegirlos**

1. Empieza con poca profundidad (3 o 4) y súbela de uno en uno mirando la métrica de **prueba** en
   📈 Evaluación, no la de entrenamiento.
2. Quédate con el árbol más sencillo a partir del cual la métrica de prueba deja de mejorar: se
   interpreta mejor y suele generalizar mejor.
3. Si el árbol tiene muchas hojas con muy pocas muestras (míralo en 🌲 Árbol), sube las muestras mínimas.

⚠️ **Errores típicos**

- **Elegir los hiperparámetros mirando el conjunto de prueba muchas veces.** Si pruebas veinte
  combinaciones y te quedas con la que mejor puntúa en prueba, esa puntuación ya es optimista: has
  ajustado el modelo a ese conjunto concreto. Fuera de la app se usa validación cruzada
  (`GridSearchCV`) sobre el entrenamiento y la prueba se reserva para el final.
- **Escalar los datos «por si acaso».** Un árbol solo compara cada característica con un umbral, así que
  multiplicar una columna por 1000 mueve el umbral pero no cambia el árbol. No hace falta escalar.
- **Creer que el árbol entiende las categorías.** Las columnas de texto se convierten en números por orden
  alfabético (en 🚢 Titanic, Puerto_Embarque: C = 0, Q = 1, S = 2), y el árbol las trata como números
  ordenados: con una sola pregunta («Puerto_Embarque ≤ 0,5») puede separar C de {Q, S}, pero para aislar Q
  necesita dos.
- **Esperar el mismo árbol con otros datos.** Los árboles son **inestables**: cambiar unas pocas muestras de
  entrenamiento (otro tamaño de prueba, por ejemplo) puede cambiar la primera pregunta y, con ella, todo el
  árbol. Si dos características empatan, scikit-learn elige al azar con `random_state=42`; MLTutor hace lo
  mismo para que obtengas el mismo árbol que scikit-learn, salvo empates numéricos en algunos equipos (en
  un empate exacto, una diferencia en la última cifra decimal puede decidir otra pregunta).
