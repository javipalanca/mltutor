<!-- linear_regression_app.py, bajo la tabla de coeficientes (logística). Expander cerrado. Título del expander: "📖 ¿Cómo leer esta tabla?" -->
**Columnas de la tabla:**

- **Coeficiente:** Valor original del modelo (log-odds)
- **Odds Ratio:** `e^coeficiente` - Más fácil de interpretar
- **Cambio en Odds:** Cambio porcentual en las odds
- **Efecto:** Dirección del efecto (positivo/negativo)
- **Fuerza del Efecto:** Magnitud del impacto

**Interpretación de Odds Ratio:**
- **OR = 1.0:** Sin efecto
- **OR > 1.0:** Aumenta las odds (efecto positivo)
- **OR < 1.0:** Disminuye las odds (efecto negativo)

**Ejemplos:**
- **OR = 2.0:** Duplica las odds (+100%)
- **OR = 1.5:** Aumenta las odds en 50%
- **OR = 0.5:** Reduce las odds a la mitad (-50%)
- **OR = 0.2:** Reduce las odds en 80%
