<!-- NUEVO (D71, Fase 5): no existe en la versión Python. Pestaña 🏗️ Arquitectura, expander cerrado debajo del selector "Función de activación de salida" (y enlazado desde los st.error no_entrenable_*). Título: mensajes.toml [arquitectura] salidas_expander. Tabla de combinaciones: fase5-redes.md §2.2.1 y tools/nn_reference.py::resolve_training (lo que hace la app). Cifras: tests/parity/nn/exact/diabetes_softmax_output.json (salida 1,0 en todas las muestras de prueba y pesos sin cambiar tras 10 épocas), diabetes_sigmoid_output.json (salidas entre 0,993 y 1,0 con un objetivo de 25 a 346; MSE de prueba 26 258), iris_tanh_output.json (salidas entre −0,09 y 0,99; exactitud 0,933) y breast_cancer_sgd_tanh_output.json (salidas entre −0,22 y 1,0). Pérdidas desde los logits: MEMORY.md (Keras 3.12 frente a burn). -->
**¿Qué hace una función de activación?** Cada neurona calcula una suma ponderada de sus entradas más un
sesgo, $z = w_1 x_1 + \dots + w_n x_n + b$, y le aplica una función $f(z)$. Sin esa función, apilar capas no
serviría de nada: una combinación lineal de combinaciones lineales sigue siendo lineal, y la red entera
equivaldría a una sola capa. La **no linealidad** es lo que permite aprender fronteras curvas, como las de
🌙 Moons.

**En las capas ocultas:**

- **ReLU**, $f(z) = \max(0, z)$: barata y con derivada 1 para $z > 0$, así que el gradiente no se debilita al
  atravesar muchas capas. Su riesgo: si una neurona recibe $z \le 0$ con todas las muestras, su derivada es 0,
  deja de aprender y queda **muerta** (lo verás en 📉 Activaciones).
- **tanh**, entre $-1$ y $1$, centrada en 0; y **sigmoid**, $\sigma(z) = 1/(1 + e^{-z})$, entre 0 y 1. Las dos se
  **saturan**: con $|z|$ grande la curva es casi plana, la derivada casi 0 y las capas anteriores apenas
  aprenden (el *gradiente que desaparece*). Por eso hoy se prefiere ReLU en las capas ocultas.

**En la capa de salida** la activación decide **qué significa** lo que devuelve la red, y cada una va con su
**función de pérdida**, la medida del error que el entrenamiento intenta reducir:

| Problema | Salida | Neuronas | Pérdida |
|---|---|---|---|
| Clasificación binaria (🔬 Cáncer, 🌙 Moons) | **sigmoid**: la probabilidad $p$ de la clase positiva | 1 | Entropía cruzada binaria: $-[\,y \log p + (1-y)\log(1-p)\,]$ |
| Multiclase (🌸 Iris, 🍷 Vino) | **softmax**: $p_k = e^{z_k} / \sum_j e^{z_j}$, una probabilidad por clase que suman 1 | una por clase | Entropía cruzada categórica: $-\log p_{\text{clase real}}$ (la etiqueta va en *one-hot*: un 1 en su clase y 0 en las demás) |
| Regresión (🩺 Diabetes, 🏠 California) | **linear**: $f(z) = z$, cualquier número | 1 | Error cuadrático medio (MSE): media de $(y - \hat y)^2$ |

La entropía cruzada castiga mucho estar **seguro y equivocado**: si la clase real tiene $p = 0{,}01$, la
pérdida es $-\log 0{,}01 \approx 4{,}6$; con $p = 0{,}9$, solo $0{,}1$. Keras (y MLTutor) la calculan a partir
de $z$ y no de $p$, que es más estable numéricamente. En regresión la métrica que se muestra es el MAE (error
absoluto medio), que está en las unidades del objetivo y es más fácil de leer que el MSE.

**Por qué MLTutor no entrena cuatro combinaciones** (el botón se desactiva):

- **Binaria + softmax.** Con una sola neurona, la softmax divide $e^z$ entre sí mismo: vale **siempre 1**, sea
  cual sea la entrada. La red no puede distinguir las dos clases.
- **Binaria o multiclase + linear.** La salida puede ser cualquier número, negativo o mayor que 1, y la
  entropía cruzada necesita probabilidades: $\log$ de un número negativo no existe. (En Keras se puede
  entrenar una salida lineal con `from_logits=True`, que aplica la sigmoid o la softmax dentro de la pérdida;
  MLTutor no ofrece esa opción.)
- **Multiclase + sigmoid.** Cada neurona daría una probabilidad **independiente** (sí/no para su clase), sin
  sumar 1: es la salida de los problemas *multietiqueta*, donde una muestra puede tener varias clases a la vez.
  Necesitaría una etiqueta 0/1 por neurona, y aquí cada muestra tiene un único número de clase.

**Combinaciones que se entrenan, pero dan resultados raros** (por eso los avisos):

- **tanh en clasificación.** Se entrena con MSE contra las etiquetas 0/1 (o el one-hot) y la clase se decide
  con el umbral 0,5 o con la neurona mayor. Puede acertar (🌸 Iris con salida tanh y learning rate 0,01: 93 %
  en prueba), pero su salida **no es una probabilidad**: va de $-1$ a $1$ y en 🔬 Cáncer llega a $-0{,}22$.
- **sigmoid o tanh en regresión.** La salida queda encerrada en $(0, 1)$ o $(-1, 1)$. MLTutor no escala el
  objetivo, así que en 🩺 Diabetes (de 25 a 346) la red se queda pegada al techo: todas las predicciones de
  prueba salen entre 0,993 y 1,0 y el R² es muy negativo.
- **softmax en regresión.** Con una neurona vale siempre 1 y su gradiente es 0: la red **no aprende nada**
  (tras 10 épocas, los pesos de 🩺 Diabetes son exactamente los iniciales).

⚠️ **Errores típicos**

- Poner **softmax con una neurona** en un problema binario (con dos neuronas sí funcionaría: equivale a una
  sigmoid).
- Leer la salida de una red con tanh como si fuera una **probabilidad**.
- Usar una salida acotada (sigmoid, tanh) en regresión **sin escalar el objetivo** a ese rango.
- Confundir **multiclase** (una clase por muestra: softmax) con **multietiqueta** (varias a la vez: sigmoid por
  neurona).
