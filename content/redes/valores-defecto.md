<!-- NUEVO (D88, Fase 5): no existe en la versión Python. Pestaña ⚙️ Entrenamiento, expander cerrado debajo de "### 🚀 Iniciar Entrenamiento"; también en 📈 Evaluación debajo del aviso aviso_r2_negativo (D84). Título: mensajes.toml [entrenamiento] defecto_expander. Los valores por defecto se mantienen por paridad (D88). Tamaños: fase5-redes.md §2.3.1 (test = ceil(0,2·n), validación = últimas filas de entrenamiento). Cifras con Keras 3.12 / TensorFlow 2.16.2 en CPU, la red por defecto ([entrada, 10, 10, salida], relu, dropout 0,2, Adam, early stopping 10, reducción de LR 0,5) y keras.utils.set_random_seed(0..4); mín–máx de 5 semillas: Iris por defecto 0,633–0,767 (tests/parity/nn/stat/iris_default.json), Iris LR 0,01 0,933–0,967, Iris 200 épocas 0,800–0,867, Iris batch 16 0,700–0,833, Iris LR 0,01 + batch 16 + 100 épocas 0,933–0,967 (para entre las épocas 56 y 75); Diabetes por defecto R² −3,77 a −2,83 (nn/stat/diabetes_default.json), LR 0,01 0,19–0,43, 200 épocas −0,07–0,27, LR 0,1 0,417–0,448 (nn/stat/diabetes_adam_lr01.json, para entre las épocas 21 y 33); Cáncer por defecto 0,939–0,974 (nn/stat/breast_cancer_default.json). MLTutor usa otro generador de números aleatorios: sus cifras caen en esas bandas, pero no son idénticas. -->
**¿Por qué la red acierta poco con los valores por defecto?** No es un fallo de la app: los valores por
defecto (learning rate 0,001, batch 64, unas 50 épocas) son prudentes, pero con datasets pequeños **se quedan
cortos**. La clave es contar cuántas veces se actualizan los pesos.

En cada época, la red recorre las filas de entrenamiento en **lotes** (*batches*) y actualiza los pesos una
vez por lote. En 🌸 Iris quedan 96 filas para ajustar los pesos (150 − 30 de prueba − 24 de validación); con
lotes de 64, son **2 actualizaciones por época** y **100 en total** en 50 épocas. Adam mueve cada peso, en cada
actualización, aproximadamente lo que marca el learning rate: 0,001. Después de 100 pasos, cada peso se ha
movido del orden de una décima, cuando los pesos iniciales son del orden de ±0,6. La red casi no ha salido de su
punto de partida: la pérdida sigue bajando al final, y la exactitud en prueba se queda entre el 63 % y el 77 %
según la semilla.

En 🩺 Diabetes se suma otro problema: el objetivo **no está escalado** (va de 25 a 346, con media 152) y la red
empieza dando valores cercanos a 0. Con 5 actualizaciones por época y un learning rate de 0,001, en 50 épocas
no llega ni a acercarse: predice peor que si respondiera siempre la media y el **R² en prueba sale negativo**
(de −3,8 a −2,8). En cambio, en 🔬 Cáncer de mama (6 actualizaciones por época y una frontera más sencilla)
los mismos valores dan un 94–97 %.

**Qué probar** (cifras con Keras y cinco semillas; las de MLTutor caen en el mismo rango):

| Cambio | 🌸 Iris (exactitud en prueba) | 🩺 Diabetes (R² en prueba) |
|---|---|---|
| Valores por defecto | 0,63 – 0,77 | −3,8 – −2,8 |
| **Learning rate 0,01** | **0,93 – 0,97** | 0,19 – 0,43 |
| Learning rate 0,1 | — | **0,42 – 0,45** |
| 200 épocas (LR 0,001) | 0,80 – 0,87 | −0,07 – 0,27 |
| Batch 16 (LR 0,001) | 0,70 – 0,83 | — |

1. **Sube el learning rate a 0,01** (o a 0,1 en regresión con el objetivo sin escalar). Es el cambio que más
   ayuda: cada actualización avanza diez veces más.
2. **Más épocas o lotes más pequeños** también dan más actualizaciones, pero cuestan más tiempo y ayudan menos.
   Con early stopping activado puedes poner muchas épocas sin miedo: el entrenamiento para solo cuando la
   validación deja de mejorar (con LR 0,01, batch 16 y 100 épocas, Iris para entre las épocas 56 y 75).
3. **Mira la gráfica de la pérdida**: si al final todavía baja con fuerza, a la red le falta entrenamiento,
   no capacidad. Añadir capas o neuronas no lo arregla.

⚠️ **Errores típicos**

- **Subir demasiado el learning rate.** Con 0,3 la pérdida puede oscilar o explotar y muchas neuronas ReLU
  quedan muertas (en Iris, con 3 capas y LR 0,3, 15 de 36 neuronas mueren y la exactitud baja al 67 %).
- **Mirar solo la puntuación del diagnóstico.** En regresión, el diagnóstico compara el error con el de las
  primeras épocas: si la red empezó muy mal, cualquier mejora le parece «aceptable». Comprueba siempre el
  R² en prueba de 📈 Evaluación.
- **Comparar configuraciones con una sola semilla.** En Iris la semilla sola mueve la exactitud 13 puntos
  (ver «¿Por qué hay una semilla?»).
