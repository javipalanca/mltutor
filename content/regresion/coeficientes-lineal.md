<!-- mltutor/apps/linear_regression_app.py, pestaña 🔍 Coeficientes, modelo lineal. Expander cerrado, primero de la pestaña. Título del expander: "ℹ️ ¿Cómo interpretar los coeficientes?" -->
# 📈 **Coeficientes en Regresión Lineal**

## 🎯 **¿Qué representan los coeficientes?**

Los coeficientes son los **parámetros aprendidos** por el modelo que determinan cómo cada característica 
influye en la predicción final. La fórmula de regresión lineal es:

```
Predicción = β₀ + β₁×X₁ + β₂×X₂ + ... + βₙ×Xₙ
```

Donde cada **βᵢ** es un coeficiente que indica el cambio en la variable objetivo 
por cada unidad de cambio en la característica correspondiente.

## 🔍 **Interpretación Detallada:**

### **Valor del Coeficiente (Magnitud):**
- **Valor absoluto grande (ej: |5.2|):** La característica tiene **alta influencia**
- **Valor absoluto pequeño (ej: |0.1|):** La característica tiene **baja influencia**
- **Valor cero (0.0):** La característica **no influye** en la predicción

### **Signo del Coeficiente (Dirección):**
- **Positivo (+):** 📈 **Relación directa** - A mayor valor de X, mayor valor de Y
- **Negativo (-):** 📉 **Relación inversa** - A mayor valor de X, menor valor de Y

### **Unidades:**
Los coeficientes mantienen las **unidades originales**. Si predices precios en euros 
y una característica está en metros², un coeficiente de 150 significa 
**+150 euros por cada metro² adicional**.

## 💡 **Ejemplos Prácticos:**

**🏠 Predicción de Precios de Casas:**
- `Tamaño = +150`: Cada m² adicional aumenta el precio en 150€
- `Antigüedad = -500`: Cada año adicional reduce el precio en 500€
- `Habitaciones = +2000`: Cada habitación adicional aumenta el precio en 2000€

**📊 Predicción de Ventas:**
- `Presupuesto_Marketing = +1.5`: Cada euro en marketing genera 1.5€ en ventas
- `Competencia = -0.8`: Cada competidor adicional reduce ventas en 0.8€

## ⚠️ **Limitaciones y Consideraciones:**

1. **Correlación ≠ Causalidad:** Un coeficiente alto no implica que la característica cause el resultado
2. **Escalas diferentes:** Características con diferentes escalas pueden tener coeficientes incomparables
3. **Multicolinealidad:** Características correlacionadas pueden tener coeficientes inestables
4. **Outliers:** Valores extremos pueden afectar significativamente los coeficientes

## 🎛️ **Cómo usar esta información:**

- **Identificar factores clave:** Coeficientes con mayor valor absoluto
- **Validar intuición:** ¿Los signos coinciden con el conocimiento del dominio?
- **Tomar decisiones:** ¿En qué características enfocar esfuerzos?
- **Detectar problemas:** ¿Hay coeficientes que no tienen sentido?
