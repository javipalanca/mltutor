<!-- linear_regression_app.py, bajo la tabla de coeficientes (logística). Expander cerrado. Título del expander: "📖 ¿Cómo leer esta tabla?". CORREGIDO en Rust (2026-10-14): el coeficiente es el cambio del log-odds por unidad (no "el log-odds"); la fuerza del efecto se explica con los umbrales simétricos de D27; nota de multiclase. -->
**Columnas de la tabla:**

- **Coeficiente:** Valor original del modelo: cambio del log-odds por cada unidad de la característica
- **Odds Ratio:** `e^coeficiente` - Más fácil de interpretar
- **Cambio en Odds:** Cambio porcentual en las odds
- **Efecto:** Dirección del efecto (positivo/negativo)
- **Fuerza del Efecto:** Magnitud del impacto según |coeficiente| (> 2 muy fuerte, > 1 fuerte, > 0.5 moderado); 
  un coeficiente de −2.5 es tan fuerte como uno de +2.5, aunque en sentido contrario

**Interpretación de Odds Ratio:**
- **OR = 1.0:** Sin efecto
- **OR > 1.0:** Aumenta las odds (efecto positivo)
- **OR < 1.0:** Disminuye las odds (efecto negativo)

**Ejemplos:**
- **OR = 2.0:** Duplica las odds (+100%)
- **OR = 1.5:** Aumenta las odds en 50%
- **OR = 0.5:** Reduce las odds a la mitad (-50%)
- **OR = 0.2:** Reduce las odds en 80%

**En multiclase** (🌸 Iris, 🍷 Vino…) la tabla es la de la clase elegida y las odds se miden 
frente a una clase promedio, no frente a «no ser de esa clase»: lee la nota de multiclase de 
«¿Cómo interpretar los coeficientes?».
