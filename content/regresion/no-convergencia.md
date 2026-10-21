<!-- NUEVO (D19, Fase 2): no existe en la versión Python, que ignoraba el ConvergenceWarning. Pestaña 🏋️ Entrenamiento, debajo del aviso "⚠️ El optimizador no ha convergido…" (mensajes.toml [entrenamiento] aviso_no_convergencia). Expander cerrado. Título: "ℹ️ ¿Qué significa que el modelo no converja?" CORREGIDO (revisión ML de F2, 2026-10-14): el texto describía L-BFGS de scikit-learn; la app usa Newton (L-BFGS solo con más de 300 parámetros) sobre características estandarizadas internamente, y escalar con StandardScaler da otro modelo. -->
**¿Qué significa «converger»?**

La regresión logística no tiene una fórmula cerrada como la lineal: sus coeficientes se buscan
paso a paso. Un **optimizador** parte de todos los coeficientes a 0 y, en cada **iteración**,
los mueve cuesta abajo en la función de pérdida. Se dice que **converge** cuando ya casi no
puede mejorar: la pendiente (el gradiente) es prácticamente cero y ha llegado al mínimo.

$$
\text{pérdida}(\beta) = \underbrace{\tfrac{1}{2}\lVert \beta \rVert^2}_{\text{regularización}} + C \sum_{i=1}^{n} \text{log-loss}_i(\beta)
$$

(la regularización no incluye el intercepto; en MLTutor, $C = 1$).

MLTutor usa el **método de Newton**, un optimizador de *segundo orden*: además de la pendiente
mira la curvatura de la pérdida, y por eso suele llegar al mínimo en menos de 30 iteraciones.
Cuando hay muchos parámetros (características × clases, como en 🔢 Dígitos), calcular la
curvatura sale demasiado caro y usa **L-BFGS**, el optimizador por defecto de scikit-learn:
cada paso es más barato, pero necesita muchos más.

Si se agota el **máximo de iteraciones** antes de llegar, el entrenamiento se detiene donde
esté. El modelo funciona, pero sus coeficientes no son los óptimos: pueden cambiar si entrenas
con más iteraciones, y las probabilidades pueden diferir de las de otro programa
(scikit-learn, por ejemplo).

**¿Por qué pasa?**

- **Máximo de iteraciones bajo** para el problema: con L-BFGS, muchas características o clases
  muy mezcladas necesitan cientos de pasos.
- **Características en escalas muy distintas** (la causa más habitual en scikit-learn). En
  🔬 Cáncer, *mean area* vale cientos o miles y *mean smoothness* ronda 0,1. La pérdida forma
  entonces un «valle» muy alargado y estrecho, y un optimizador que solo mira la pendiente,
  como L-BFGS, avanza en zigzag con pasos diminutos. MLTutor lo evita: por dentro trabaja con
  las características estandarizadas (la penalización se sigue aplicando a los coeficientes
  originales, así que el modelo es el mismo) y Newton corrige la forma del valle con la
  curvatura.
- **Clases casi separables**: si una recta separa casi perfectamente las clases, la pérdida
  sigue bajando al hacer los coeficientes más y más grandes, y el optimizador tarda mucho en
  pararse (la regularización lo frena).

**¿Qué hacer?**

1. **Subir el máximo de iteraciones** con el deslizador. En MLTutor suele bastar, porque el
   problema de escala ya está resuelto por dentro.
2. **Fuera de la app, en scikit-learn**, el aviso aparece mucho más: con el solver por defecto
   (`lbfgs`) y los datos sin escalar, 🔬 Cáncer necesita unas 2300 iteraciones, así que ni
   2000 bastan. Tienes dos salidas, y no son equivalentes:

   - **Cambiar de solver** a `solver="newton-cholesky"`, el equivalente al de MLTutor: llega
     en unas 10 iteraciones al **mismo modelo** que la app (es lo que hace el código de
     💾 Exportar).
   - **Escalar las características** con `StandardScaler` (cada columna con media 0 y
     desviación 1): `lbfgs` converge en menos de 20 iteraciones, pero obtienes **otro modelo**.
     La penalización $\tfrac{1}{2}\lVert \beta \rVert^2$ se aplica ahora a los coeficientes de las
     características estandarizadas, que están en otra escala, y el mínimo cambia: en
     🔬 Cáncer, algunas probabilidades difieren en más de 0,9 de las de MLTutor (la exactitud
     sube del 97,7 % al 98,2 %). Ajusta el escalador **solo con el conjunto de
     entrenamiento**; si usas también la prueba, se filtra información del test al modelo
     (*fuga de datos*). Un `Pipeline` lo hace bien automáticamente:

   ```python
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression

   modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
   modelo.fit(X_train, y_train)  # el escalador aprende la media y la desviación de X_train
   ```

   Con los datos escalados, los coeficientes miden el efecto de **una desviación estándar** de
   cada característica, no de una unidad. Así son comparables entre sí.

**¿Es grave?** No siempre: en scikit-learn, el modelo de 🔬 Cáncer que se queda a medias con
`lbfgs` (1000 iteraciones) acierta el mismo 97,7 % que el óptimo, aunque algunas de sus
probabilidades difieren en 0,2. Pero no interpretes los coeficientes al detalle ni compares
números exactos con otra herramienta hasta que el modelo converja.
