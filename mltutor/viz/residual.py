import matplotlib.pyplot as plt
from mltutor.desktop import ui as st


def plot_predictions(y_true, y_pred):
    """Crea un gráfico de dispersión de las predicciones frente a los valores reales."""
    with st.expander("ℹ️ ¿Cómo interpretar estas visualizaciones?", expanded=False):
        st.markdown("""
        **Gráfico de Predicciones vs Valores Reales:**
        - Cada punto representa una predicción del modelo
        - La línea roja diagonal representa predicciones perfectas
        - **Interpretación:**
            - Puntos cerca de la línea roja = buenas predicciones
            - Puntos dispersos = predicciones menos precisas
            - Patrones sistemáticos fuera de la línea pueden indicar problemas del modelo
        
        **Gráfico de Residuos:**
        - Muestra la diferencia entre valores reales y predicciones
        - **Interpretación:**
            - Residuos cerca de cero = buenas predicciones
            - Patrones en los residuos pueden indicar que el modelo lineal no es adecuado
            - Distribución aleatoria alrededor de cero es ideal
        """)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot con mejor estilo
    ax.scatter(y_true, y_pred, alpha=0.6, s=50,
               edgecolors='black', linewidth=0.5)

    # Línea de predicción perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', lw=2, label='Predicción Perfecta')

    # Personalización del gráfico
    ax.set_xlabel('Valores Reales', fontsize=12)
    ax.set_ylabel('Predicciones', fontsize=12)
    ax.set_title('Predicciones vs Valores Reales',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Añadir estadísticas al gráfico
    r2_value = st.session_state.get('metrics_lr', {}).get('r2', 0)
    ax.text(0.05, 0.95, f'R² = {r2_value:.4f}',
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Mostrar con 80% del ancho
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.pyplot(fig, use_container_width=True)


def plot_residuals(y_true, y_pred):
    # Información explicativa sobre los residuos
    with st.expander("ℹ️ ¿Cómo interpretar el análisis de residuos?", expanded=False):
        st.markdown("""
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
        """)

    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Residuos vs Predicciones
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--',
                lw=2, label='Residuo = 0')
    ax1.set_xlabel('Predicciones', fontsize=12)
    ax1.set_ylabel('Residuos (Real - Predicción)', fontsize=12)
    ax1.set_title('Residuos vs Predicciones',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Añadir estadísticas al gráfico
    residual_std = residuals.std()
    ax1.text(0.05, 0.95, f'Desv. Estándar: {residual_std:.3f}',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Histograma de residuos
    ax2.hist(residuals, bins=20, alpha=0.7,
             edgecolor='black', color='skyblue')
    ax2.axvline(residuals.mean(), color='red', linestyle='--',
                lw=2, label=f'Media: {residuals.mean():.3f}')
    ax2.axvline(0, color='green', linestyle='-',
                lw=2, alpha=0.7, label='Ideal (0)')
    ax2.set_xlabel('Residuos', fontsize=12)
    ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.set_title('Distribución de Residuos',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Mostrar con 80% del ancho
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.pyplot(fig, use_container_width=True)

    # Interpretación automática de los residuos
    st.markdown("### 🔍 Interpretación de los Residuos")

    mean_residual = abs(residuals.mean())
    std_residual = residuals.std()

    interpretation = []

    if mean_residual < 0.1 * std_residual:
        interpretation.append(
            "✅ **Media de residuos cercana a 0:** El modelo está bien calibrado")
    else:
        interpretation.append(
            "⚠️ **Media de residuos alejada de 0:** El modelo puede tener sesgo sistemático")

    # Calcular R² de los residuos para detectar patrones
    from scipy import stats
    if len(residuals) > 10:
        slope, _, r_value, _, _ = stats.linregress(
            y_pred, residuals)
        if abs(r_value) < 0.1:
            interpretation.append(
                "✅ **Sin correlación entre residuos y predicciones:** Buen ajuste lineal")
        else:
            interpretation.append(
                "⚠️ **Correlación detectada en residuos:** Puede haber relaciones no lineales")

    # Test de normalidad simplificado (basado en asimetría)
    skewness = abs(stats.skew(residuals))
    if skewness < 1:
        interpretation.append(
            "✅ **Distribución de residuos aproximadamente normal**")
    else:
        interpretation.append(
            "⚠️ **Distribución de residuos sesgada:** Revisar outliers o transformaciones")

    for item in interpretation:
        st.markdown(f"- {item}")

    if mean_residual >= 0.1 * std_residual or abs(r_value) >= 0.1 or skewness >= 1:
        st.info(
            "💡 **Sugerencias de mejora:** Considera probar transformaciones de variables, añadir características polinómicas, o usar modelos no lineales.")
