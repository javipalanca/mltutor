<!-- linear_regression_app.py, pestaña 🔍 Coeficientes, modelo logístico. Expander cerrado, primero de la pestaña. Título del expander: "ℹ️ ¿Cómo interpretar los coeficientes?" -->
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

## 🔍 **Interpretación de Log-Odds:**

### **Valor del Coeficiente:**
- **Positivo (+):** 📈 **Aumenta** la probabilidad de la clase positiva
- **Negativo (-):** 📉 **Disminuye** la probabilidad de la clase positiva
- **Cero (0.0):** **No afecta** la probabilidad

### **Magnitud del Coeficiente:**
- **|β| > 2:** **Efecto muy fuerte** (odds ratio > 7.4)
- **1 < |β| < 2:** **Efecto fuerte** (odds ratio entre 2.7 y 7.4)
- **0.5 < |β| < 1:** **Efecto moderado** (odds ratio entre 1.6 y 2.7)
- **|β| < 0.5:** **Efecto débil** (odds ratio < 1.6)

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

## ⚠️ **Limitaciones y Consideraciones:**

1. **Interpretación no lineal:** El efecto en probabilidades no es constante
2. **Asunciones de linealidad:** En el espacio log-odds, no en probabilidades
3. **Interacciones ignoradas:** Los efectos pueden depender de otras variables
4. **Escalas importantes:** Normalizar puede facilitar la interpretación

## 🎛️ **Cómo usar esta información:**

### **Para el Negocio:**
- **Identificar factores de riesgo:** Coeficientes positivos altos
- **Encontrar factores protectores:** Coeficientes negativos altos
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
