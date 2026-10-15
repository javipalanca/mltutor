<!-- NUEVO (D58, Fase 4): no existe en la versión Python, que solo mostraba la clase o el valor y el st.info "Información del modelo KNN". Pestaña 🔮 Predicciones, expander cerrado debajo de la tabla "#### 👥 Los {k} vecinos más cercanos" (mensajes.toml [predicciones] vecinos_titulo). Título: mensajes.toml [predicciones] vecinos_expander. Reglas: fase4-knn.md §3 C17.5 (iguales en crates/mltutor-core). Ejemplo de Iris: tests/parity/knn/iris_k5_uniform_minkowski.json prediction_form.custom (entrada 7,9 / 4,4 / 6,9 / 2,5, vecino más cercano a 0,728, P(virginica) = 1). -->
**Así ha decidido el modelo.** La tabla muestra las $K$ muestras de **entrenamiento** más cercanas a la que has
introducido, de la más cercana a la más lejana, con su distancia (en la métrica del modelo), su peso y su
clase o valor.

- **Clasificación, pesos `uniform`**: cada vecino vale un voto. La probabilidad de una clase es la fracción de
  vecinos que la tienen: $P(c) = \dfrac{\text{n.º de vecinos de la clase } c}{K}$. Gana la clase más votada.
- **Clasificación, pesos `distance`**: cada vecino pesa $w_i = 1/d_i$ y
  $P(c) = \sum_{i \in c} w_i \,/\, \sum_i w_i$. Un vecino muy cercano puede ganar a varios lejanos.
- **Regresión**: la predicción es la media de los valores de los vecinos (`uniform`) o su media ponderada
  por $1/d_i$ (`distance`).
- **Distancia 0**: si tu muestra coincide exactamente con alguna de entrenamiento y los pesos son `distance`,
  solo cuentan las que coinciden (peso 1) y las demás pesan 0.

⚠️ **Cómo no leer esta tabla**

- **Una probabilidad del 100 % no significa «seguro».** Solo dice que los $K$ vecinos coinciden. En 🌸 Iris,
  con los valores máximos de las cuatro medidas (7,9 / 4,4 / 6,9 / 2,5 cm), el vecino más cercano está a
  0,73 cm y aun así $P(\text{virginica}) = 1$. Mira también **las distancias**: si son mucho mayores que las
  habituales, la muestra está lejos de todo lo que el modelo conoce.
- **Un KNN no extrapola.** En regresión la predicción es una media de valores de entrenamiento, así que nunca
  sale del rango de esos valores, por raros que sean los datos de entrada.
- **Las distancias están en las unidades de los datos, sin escalar** (ver «📏 KNN sin escalar» en
  🏋️ Entrenamiento): la característica con los números más grandes decide casi sola quiénes son los vecinos.
- Con $K$ pequeño, la probabilidad solo puede tomar unos pocos valores ($0, \frac{1}{K}, \frac{2}{K}, \dots, 1$
  con `uniform`), así que es una estimación muy gruesa.
