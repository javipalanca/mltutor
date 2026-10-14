<!-- viz/residual.py::plot_residuals, antes de los dos gráficos. Expander cerrado. Título del expander: "ℹ️ ¿Cómo interpretar el análisis de residuos?". NUEVO en Rust (2026-10-14): la última línea, que explica qué comprueban las viñetas automáticas (ver CORREGIDO en mensajes.toml [residuos]). -->
**¿Qué son los residuos?**
Los residuos son las diferencias entre los valores reales y las predicciones del modelo:
`Residuo = Valor Real - Predicción`

**Gráfico de Residuos vs Predicciones:**
- **Ideal:** Los puntos deben estar distribuidos aleatoriamente alrededor de la línea y=0
- **Problema:** Si ves patrones (curvas, abanicos), puede indicar:
    - El modelo no captura relaciones no lineales
    - Heterocedasticidad (varianza no constante)
    - Variables importantes omitidas

**Histograma de Residuos:**
- **Ideal:** Distribución normal (campana) centrada en 0
- **Problema:** Si la distribución está sesgada o tiene múltiples picos:
    - Puede indicar que el modelo no es apropiado
    - Sugiere la presencia de outliers o datos problemáticos

**Línea roja punteada:** Marca el residuo = 0 (predicción perfecta)
**Media de residuos:** Debería estar cerca de 0 para un modelo bien calibrado

**Las comprobaciones automáticas** (media, correlación y asimetría) son orientativas: la correlación solo detecta una tendencia en línea recta y una curva en U puede tener correlación 0. Mira siempre los gráficos.
