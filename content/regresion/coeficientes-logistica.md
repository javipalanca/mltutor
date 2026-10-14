<!-- linear_regression_app.py, pestaña 🔍 Coeficientes, modelo logístico. Expander cerrado, primero de la pestaña. Título del expander: "ℹ️ ¿Cómo interpretar los coeficientes?". CORREGIDO en Rust (2026-10-14): (1) la escala de magnitudes solo daba los odds ratio de los coeficientes positivos; ahora es simétrica, como la columna "Fuerza del Efecto" (D27); (2) se dice cuál es la clase positiva; (3) nueva sección de multiclase: con softmax, e^β de una clase no es el odds ratio de "ser de esa clase frente a no serlo" (en Python se interpretaba como en binaria); (4) "negativos altos" era ambiguo. -->
# 🎯 **Coeficientes en Regresión Logística**

## 🎯 **¿Qué representan los coeficientes?**

En regresión logística, los coeficientes representan el **cambio en log-odds** 
(logaritmo de las probabilidades) por cada unidad de cambio en la característica. 
La fórmula es:

```
log(odds) = β₀ + β₁×X₁ + β₂×X₂ + ... + βₙ×Xₙ
odds = e^(β₀ + β₁×X₁ + β₂×X₂ + ... + βₙ×Xₙ)
probabilidad = odds / (1 + odds)
```

En un problema binario, la **clase positiva** es la de código 1, la segunda de la lista de 
clases (en 🚢 Titanic, *Sobrevivió*; en 🔬 Cáncer, *benign*).

## 🔍 **Interpretación de Log-Odds:**

### **Valor del Coeficiente:**
- **Positivo (+):** 📈 **Aumenta** la probabilidad de la clase positiva
- **Negativo (-):** 📉 **Disminuye** la probabilidad de la clase positiva
- **Cero (0.0):** **No afecta** la probabilidad

### **Magnitud del Coeficiente:**
- **|β| > 2:** **Efecto muy fuerte** (odds ratio > 7.4, o < 0.14 si β es negativo)
- **1 < |β| ≤ 2:** **Efecto fuerte** (odds ratio entre 2.7 y 7.4, o entre 0.14 y 0.37)
- **0.5 < |β| ≤ 1:** **Efecto moderado** (odds ratio entre 1.6 y 2.7, o entre 0.37 y 0.61)
- **|β| ≤ 0.5:** **Efecto débil** (odds ratio entre 0.61 y 1.6)

Un coeficiente negativo es tan fuerte como el positivo del mismo tamaño: en 🚢 Titanic, 
`Hombre_Adulto ≈ −2.5` (OR ≈ 0.08) es un efecto muy fuerte que **divide** las odds de 
sobrevivir por 12. La magnitud depende también de la unidad: `Edad ≈ −0.013` parece 
débil, pero es por cada año; 30 años de diferencia suman −0.39 en el log-odds.

## 📊 **Conversión a Odds Ratios (Más Intuitivo):**

Los **Odds Ratios** se calculan como `e^β` y son más fáciles de interpretar:

### **Interpretación de Odds Ratios:**
- **OR = 1:** La característica **no tiene efecto**
- **OR > 1:** La característica **aumenta** las odds de la clase positiva
- **OR < 1:** La característica **disminuye** las odds de la clase positiva

### **Ejemplos de Odds Ratios:**
- **OR = 2.0:** Duplica las odds (100% de aumento)
- **OR = 1.5:** Aumenta las odds en 50%
- **OR = 0.5:** Reduce las odds a la mitad (50% de reducción)
- **OR = 0.2:** Reduce las odds en 80%

## 💡 **Ejemplos Prácticos:**

**🏥 Diagnóstico Médico (Predicción de Enfermedad):**
- `Edad = +0.05 (OR=1.05)`: Cada año adicional aumenta las odds en 5%
- `Ejercicio = -1.2 (OR=0.30)`: Hacer ejercicio reduce las odds en 70%
- `Fumador = +1.8 (OR=6.05)`: Ser fumador multiplica las odds por 6

**📧 Clasificación de Spam:**
- `Palabras_Sospechosas = +2.3 (OR=10.0)`: Multiplica las odds de spam por 10
- `Remitente_Conocido = -1.5 (OR=0.22)`: Reduce las odds de spam en 78%

**💳 Detección de Fraude:**
- `Hora_Inusual = +1.1 (OR=3.0)`: Triplica las odds de fraude
- `Ubicacion_Habitual = -2.0 (OR=0.14)`: Reduce las odds de fraude en 86%

## 🎨 **En multiclase (🌸 Iris, 🍷 Vino, 🐧 Pingüinos…):**

scikit-learn usa una regresión logística **multinomial**: cada clase *k* tiene su propio 
vector de coeficientes y su propia puntuación $z_k = \beta_{k0} + \beta_{k1} x_1 + \dots$, 
y las probabilidades salen de comparar todas las puntuaciones (función **softmax**):

$$
p_k = \frac{e^{z_k}}{\sum_{l} e^{z_l}}
$$

Por eso un coeficiente de una clase **no se lee igual que en binaria**:

- Lo que es lineal es la **diferencia entre dos clases**: 
  $\log\left(\dfrac{p_k}{p_l}\right) = z_k - z_l$. Al aumentar $x_j$ en una unidad, las odds de 
  *k* frente a *l* se multiplican por $e^{\beta_{kj} - \beta_{lj}}$.
- Con la regularización de scikit-learn, los coeficientes de una característica suman 0 
  entre todas las clases. Así, $e^{\beta_{kj}}$ multiplica las odds de la clase *k* frente a 
  una **clase promedio** (la media geométrica de las probabilidades de todas las clases).
- Un coeficiente positivo **favorece** a la clase *k*, pero su probabilidad puede bajar si 
  otra clase tiene un coeficiente aún mayor para esa característica.

En 🌸 Iris, la longitud del pétalo tiene coeficiente ≈ −2.3 en *setosa* y ≈ +2.5 en 
*virginica*: los pétalos largos alejan de *setosa* y acercan a *virginica*.

## ⚠️ **Limitaciones y Consideraciones:**

1. **Interpretación no lineal:** El efecto en probabilidades no es constante
2. **Asunciones de linealidad:** En el espacio log-odds, no en probabilidades
3. **Interacciones ignoradas:** Los efectos pueden depender de otras variables
4. **Escalas importantes:** Normalizar puede facilitar la interpretación

## 🎛️ **Cómo usar esta información:**

### **Para el Negocio:**
- **Identificar factores de riesgo:** Coeficientes positivos altos
- **Encontrar factores protectores:** Coeficientes muy negativos
- **Priorizar intervenciones:** Enfocarse en variables con mayor impacto

### **Para el Modelo:**
- **Validar coherencia:** ¿Los signos tienen sentido del dominio?
- **Detectar overfitting:** ¿Coeficientes extremadamente grandes?
- **Selección de características:** ¿Coeficientes cerca de cero?

## 🧮 **Fórmula de Conversión:**
```python
# De coeficiente a odds ratio
odds_ratio = np.exp(coeficiente)

# Cambio porcentual en odds
cambio_porcentual = (odds_ratio - 1) * 100
```
