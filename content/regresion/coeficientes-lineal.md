<!-- mltutor/apps/linear_regression_app.py, pestaña 🔍 Coeficientes, modelo lineal. Expander cerrado, primero de la pestaña. Título del expander: "ℹ️ ¿Cómo interpretar los coeficientes?". CORREGIDO en Rust (2026-10-14): (1) cada coeficiente mide el cambio manteniendo fijas las demás características (faltaba, y es lo que distingue la regresión múltiple); (2) comparar magnitudes solo vale con escalas parecidas (ejemplo de Diabetes); (3) "genera 1.5€ en ventas" afirmaba causalidad, en contra de la propia sección de limitaciones; (4) ejemplos de multicolinealidad y escala con los datasets de la app. -->
# 📈 **Coeficientes en Regresión Lineal**

## 🎯 **¿Qué representan los coeficientes?**

Los coeficientes son los **parámetros aprendidos** por el modelo que determinan cómo cada característica 
influye en la predicción final. La fórmula de regresión lineal es:

```
Predicción = β₀ + β₁×X₁ + β₂×X₂ + ... + βₙ×Xₙ
```

Donde cada **βᵢ** es un coeficiente que indica el cambio en la variable objetivo 
por cada unidad de cambio en la característica correspondiente, **manteniendo fijas 
las demás características**.

## 🔍 **Interpretación Detallada:**

### **Valor del Coeficiente (Magnitud):**
- **Valor absoluto grande (ej: |5.2|):** La característica tiene **alta influencia**
- **Valor absoluto pequeño (ej: |0.1|):** La característica tiene **baja influencia**
- ⚠️ Solo puedes comparar magnitudes si las características están en **escalas parecidas**: 
  el coeficiente depende de la unidad de medida (no es lo mismo metros que centímetros)
- **Valor cero (0.0):** La característica **no influye** en la predicción

### **Signo del Coeficiente (Dirección):**
- **Positivo (+):** 📈 **Relación directa** - A mayor valor de X, mayor valor de Y (con las demás características fijas)
- **Negativo (-):** 📉 **Relación inversa** - A mayor valor de X, menor valor de Y (con las demás características fijas)

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
- `Presupuesto_Marketing = +1.5`: Cada euro más en marketing se asocia con 1.5€ más en ventas
- `Competencia = -0.8`: Cada competidor adicional se asocia con 0.8€ menos en ventas

## ⚠️ **Limitaciones y Consideraciones:**

1. **Correlación ≠ Causalidad:** Un coeficiente alto no implica que la característica cause el resultado
2. **Escalas diferentes:** Características con diferentes escalas pueden tener coeficientes incomparables. 
   En 🩺 Diabetes las características vienen ya escaladas (cada una varía más o menos entre −0.1 y 0.2), 
   así que «una unidad» es mayor que todo su rango: por eso salen coeficientes de cientos
3. **Multicolinealidad:** Características correlacionadas pueden tener coeficientes inestables. 
   En 🩺 Diabetes, `s1` y `s2` (colesterol total y LDL) tienen una correlación de 0.90 y reciben 
   coeficientes enormes de signo contrario (unos −900 y +500) que en parte se compensan
4. **Outliers:** Valores extremos pueden afectar significativamente los coeficientes

## 🎛️ **Cómo usar esta información:**

- **Identificar factores clave:** Coeficientes con mayor valor absoluto
- **Validar intuición:** ¿Los signos coinciden con el conocimiento del dominio?
- **Tomar decisiones:** ¿En qué características enfocar esfuerzos?
- **Detectar problemas:** ¿Hay coeficientes que no tienen sentido?
