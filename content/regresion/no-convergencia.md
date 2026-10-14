<!-- NUEVO (D19, Fase 2): no existe en la versión Python, que ignoraba el ConvergenceWarning. Pestaña 🏋️ Entrenamiento, debajo del aviso "⚠️ El optimizador no ha convergido…" (mensajes.toml [entrenamiento] aviso_no_convergencia). Expander cerrado. Título: "ℹ️ ¿Qué significa que el modelo no converja?" -->
**¿Qué significa «converger»?**

La regresión logística no tiene una fórmula cerrada como la lineal: sus coeficientes se buscan
paso a paso. Un **optimizador** (L-BFGS, el mismo que usa scikit-learn por defecto) parte de
todos los coeficientes a 0 y, en cada **iteración**, los mueve un poco cuesta abajo en la
función de pérdida. Se dice que **converge** cuando ya casi no puede mejorar: la pendiente
(el gradiente) es prácticamente cero y ha llegado al mínimo.

$$
\text{pérdida}(\beta) = \underbrace{\tfrac{1}{2}\lVert \beta \rVert^2}_{\text{regularización}} + C \sum_{i=1}^{n} \text{log-loss}_i(\beta)
$$

Si se agota el **máximo de iteraciones** antes de llegar, el entrenamiento se detiene donde
esté. El modelo funciona, pero sus coeficientes no son los óptimos: pueden cambiar si entrenas
con más iteraciones, y las probabilidades pueden diferir de las de otro programa
(scikit-learn, por ejemplo).

**¿Por qué pasa?**

- **Características en escalas muy distintas** (la causa más habitual). En 🔬 Cáncer, *mean area*
  vale cientos o miles y *mean smoothness* ronda 0,1. La pérdida forma entonces un «valle» muy
  alargado y estrecho, y el optimizador avanza en zigzag con pasos diminutos.
- **Máximo de iteraciones bajo** para el problema: muchas características o clases muy
  mezcladas necesitan más pasos.
- **Clases casi separables**: si una recta separa casi perfectamente las clases, la pérdida
  sigue bajando al hacer los coeficientes más y más grandes, y el optimizador tarda mucho en
  pararse (la regularización lo frena).

**¿Qué hacer?**

1. **Escalar las características** con `StandardScaler` (cada columna con media 0 y
   desviación 1). Es la solución de fondo: en 🔬 Cáncer, scikit-learn necesita unas
   2300 iteraciones sin escalar y menos de 20 con los datos escalados. Ajusta el escalador
   **solo con el conjunto de entrenamiento**; si usas también la prueba, se filtra información
   del test al modelo (*fuga de datos*). Un `Pipeline` lo hace bien automáticamente:

   ```python
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression

   modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
   modelo.fit(X_train, y_train)  # el escalador aprende la media y la desviación de X_train
   ```

   Escalar cambia los coeficientes: pasan a medir el efecto de **una desviación estándar** de
   cada característica, no de una unidad. Así son comparables entre sí.
2. **Subir el máximo de iteraciones** con el deslizador. Puede bastar si el modelo se ha
   quedado cerca; con datos sin escalar a veces ni 2000 iteraciones son suficientes (es lo que
   ocurre en 🔬 Cáncer).

**¿Es grave?** No siempre: en 🔬 Cáncer la exactitud ya es del 97,7 % aunque no haya
convergido. Pero no interpretes los coeficientes al detalle ni compares números exactos con
otra herramienta hasta que el modelo converja.
