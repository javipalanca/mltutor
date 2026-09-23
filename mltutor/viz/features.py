from mltutor.desktop import ui as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance


def display_feature_importance(model, feature_names, X_test=None, y_test=None, task_type="Clasificación"):
    """
    Muestra la importancia de las características de forma inteligente según el modelo.
    Funciona con árboles de decisión, KNN, regresión y cualquier modelo de sklearn.
    """
    st.markdown("### Importancia de Características")

    # Determinar qué método usar para calcular importancia
    if hasattr(model, 'feature_importances_'):
        # Modelos con importancia nativa (árboles, random forest, etc.)
        importances = model.feature_importances_
        method = "Importancia Nativa del Modelo"
        method_description = "Basada en la reducción de impureza (Gini/Entropy) o varianza (MSE) durante la construcción del árbol."

    elif X_test is not None and y_test is not None:
        # Usar permutation importance para modelos sin importancia nativa (KNN, etc.)
        st.info(
            "🔄 Calculando Permutation Importance... Esto puede tomar unos segundos.")

        try:
            

            # Calcular permutation importance
            with st.spinner("Calculando importancia por permutación..."):
                perm_imp = permutation_importance(
                    model, X_test, y_test,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1  # Usar todos los cores disponibles
                )

            importances = perm_imp.importances_mean
            importances_std = perm_imp.importances_std
            method = "Permutation Importance"
            method_description = "Mide cuánto disminuye el rendimiento del modelo cuando se permutan aleatoriamente los valores de cada característica."

            # Mostrar información adicional sobre la desviación estándar
            st.markdown(f"""
            **Método utilizado:** {method}
            
            {method_description}
            
            📊 **Interpretación:**
            - Valores más altos = mayor importancia
            - Se calculó con 10 repeticiones para mayor robustez
            - Las barras de error muestran la variabilidad entre repeticiones
            """)

        except Exception as e:
            st.error(f"Error al calcular Permutation Importance: {str(e)}")
            st.info(
                "💡 Asegúrate de que el modelo esté entrenado y los datos de prueba sean válidos.")
            return
    else:
        st.warning("""
        ⚠️ **No se puede calcular la importancia de características**
        
        Este modelo no tiene importancia nativa y no se proporcionaron datos de prueba.
        
        **Para ver la importancia de características necesitas:**
        - Modelos con importancia nativa (Árboles de Decisión, Random Forest, etc.), O
        - Datos de prueba (X_test, y_test) para calcular Permutation Importance
        """)
        return

    # Crear DataFrame para ordenar
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)

    # Añadir desviación estándar si está disponible
    if 'importances_std' in locals():
        feature_importance_df['std'] = importances_std[feature_importance_df.index]

    # Crear visualización
    fig, ax = plt.subplots(figsize=(12, max(8, len(feature_names) * 0.4)))

    # Barplot horizontal con barras de error si están disponibles
    if 'importances_std' in locals():
        bars = ax.barh(
            feature_importance_df['feature'],
            feature_importance_df['importance'],
            xerr=feature_importance_df['std'],
            capsize=3,
            alpha=0.8
        )
    else:
        bars = ax.barh(
            feature_importance_df['feature'],
            feature_importance_df['importance'],
            alpha=0.8
        )

    # Colorear barras con gradiente
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Configuración del gráfico
    ax.set_xlabel('Importancia')
    ax.set_title(f'Importancia de Características\n({method})')
    ax.grid(True, alpha=0.3, axis='x')

    # Añadir valores en las barras
    for i, (feature, importance) in enumerate(zip(feature_importance_df['feature'], feature_importance_df['importance'])):
        # Posicionar el texto ligeramente a la derecha de la barra
        text_x = importance + (ax.get_xlim()[1] * 0.01)
        ax.text(text_x, i, f'{importance:.3f}', va='center', fontsize=9)

    # Ajustar layout
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Mostrar tabla con los valores
    st.markdown("### Valores Detallados")
    importance_table = feature_importance_df.sort_values(
        'importance', ascending=False)
    importance_table['importance'] = importance_table['importance'].round(4)

    # Añadir columna de desviación estándar si está disponible
    if 'std' in importance_table.columns:
        importance_table['std'] = importance_table['std'].round(4)
        importance_table = importance_table.rename(columns={
            'feature': 'Característica',
            'importance': 'Importancia',
            'std': 'Desv. Estándar'
        })
    else:
        importance_table = importance_table.rename(columns={
            'feature': 'Característica',
            'importance': 'Importancia'
        })

    st.dataframe(importance_table, use_container_width=True)

    # Interpretación específica según el modelo y tarea
    with st.expander("📖 Cómo interpretar estos resultados", expanded=False):
        if hasattr(model, 'feature_importances_'):
            if task_type == "Clasificación":
                st.markdown("""
                **Importancia Nativa - Clasificación:**
                - Mide cuánto contribuye cada característica a reducir la impureza (Gini o Entropy)
                - Valores más altos = más útiles para separar las clases
                - La suma de todas las importancias = 1.0
                """)
            else:  # Regresión
                st.markdown("""
                **Importancia Nativa - Regresión:**
                - Mide cuánto contribuye cada característica a reducir la varianza (MSE)
                - Valores más altos = más útiles para predecir el valor objetivo
                - La suma de todas las importancias = 1.0
                """)
        else:
            st.markdown("""
            **Permutation Importance:**
            - Mide cuánto empeora el modelo cuando se "rompe" cada característica
            - Valores más altos = el modelo depende más de esa característica
            - Valores pueden ser negativos (característica confunde al modelo)
            - La desviación estándar indica la estabilidad de la medición
            """)

        st.markdown("""
        **Consejos generales:**
        - Características con importancia muy baja pueden ser eliminadas
        - Características con alta importancia son críticas para el modelo
        - Usar estos resultados para selección de características o interpretabilidad
        """)

