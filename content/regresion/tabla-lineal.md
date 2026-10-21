<!-- linear_regression_app.py, bajo la tabla de coeficientes (lineal). Expander cerrado. Título del expander: "📖 ¿Cómo leer esta tabla?". CORREGIDO en Rust (2026-10-14): faltaba "manteniendo fijas las demás" y que la importancia (por cuantiles de |coef|) depende de la escala de cada característica. -->
**Columnas de la tabla:**

- **Coeficiente:** Cambio en la variable objetivo por unidad de cambio en la característica
- **Efecto:** Dirección de la relación (positiva/negativa)
- **Importancia:** Magnitud relativa del efecto (según |coeficiente|; solo es justa si las características están en escalas parecidas)

**Interpretación directa:**
- Un coeficiente de **+50** significa que cada unidad adicional de esa característica 
  aumenta la predicción en 50 unidades, si las demás no cambian
- Un coeficiente de **-20** significa que cada unidad adicional de esa característica 
  disminuye la predicción en 20 unidades, si las demás no cambian
