import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.inspection import DecisionBoundaryDisplay

from utils import get_image_download_link, show_code_with_download
from algorithms.code_examples import generate_decision_boundary_code


def plot_decision_boundary(model_2d, X_train, y_train,
                           feature_names, class_names):
    """
    Ejecuta la visualización de frontera de decisión para el modelo entrenado usando solo librerías de terceros.

    Parameters:
    -----------
    model_2d : sklearn model
        Modelo entrenado
    X_train : array-like or DataFrame
        Datos de entrenamiento
    y_train : array-like
        Etiquetas de entrenamiento
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases
    """

    # Verificar que hay al menos 2 características
    if len(feature_names) < 2:
        st.warning(
            "Se necesitan al menos 2 características para visualizar la frontera de decisión.")
        return

    st.markdown("### Visualización de Frontera de Decisión")

    st.info("""
    **Cómo interpretar esta visualización:**
    - Las áreas coloreadas muestran las regiones de decisión para cada clase
    - Los puntos representan las muestras de entrenamiento
    - Las líneas entre colores son las fronteras de decisión
    - Solo se muestran las primeras dos características para crear la visualización 2D
    """)

    # Selección de características para la visualización
    if len(feature_names) > 2:
        cols = st.columns(2)
        with cols[0]:
            feature1 = st.selectbox(
                "Primera característica:",
                feature_names,
                index=0,
                key="feature1_boundary_viz"
            )
        with cols[1]:
            feature2 = st.selectbox(
                "Segunda característica:",
                feature_names,
                index=1,
                key="feature2_boundary_viz"
            )

        # Obtener índices de las características seleccionadas
        feature_names_list = list(feature_names)
        f1_idx = feature_names_list.index(feature1)
        f2_idx = feature_names_list.index(feature2)

        # Crear array con solo las dos características seleccionadas
        if hasattr(X_train, 'iloc'):
            # Es un DataFrame, usar iloc para indexación posicional
            X_boundary = X_train.iloc[:, [f1_idx, f2_idx]]
        else:
            # Es un numpy array, usar indexación normal
            X_boundary = X_train[:, [f1_idx, f2_idx]]
        feature_names_boundary = [feature1, feature2]
    else:
        # Si solo hay dos características, usarlas directamente
        X_boundary = X_train
        feature_names_boundary = feature_names

    # Entrenar el modelo simplificado solo con las 2 características
    model_2d.fit(X_boundary, y_train)

    # Crear figura y dibujar frontera de decisión usando sklearn
    try:
        fig, ax = plt.subplots(figsize=(14, 10))

        # Usar DecisionBoundaryDisplay de sklearn con el modelo simplificado
        disp = DecisionBoundaryDisplay.from_estimator(
            model_2d,
            X_boundary,
            response_method="predict",
            alpha=0.6,
            ax=ax,
            grid_resolution=200
        )

        # Añadir puntos de datos sobre la frontera
        scatter = ax.scatter(
            X_boundary.iloc[:, 0] if hasattr(
                X_boundary, 'iloc') else X_boundary[:, 0],
            X_boundary.iloc[:, 1] if hasattr(
                X_boundary, 'iloc') else X_boundary[:, 1],
            c=y_train,
            edgecolors='black',
            s=60,
            alpha=0.8
        )

        # Configurar etiquetas de los ejes
        if len(feature_names_boundary) >= 2:
            ax.set_xlabel(feature_names_boundary[0])
            ax.set_ylabel(feature_names_boundary[1])
        else:
            ax.set_xlabel("Característica 1")
            ax.set_ylabel("Característica 2")

        # Añadir leyenda de clases
        if class_names:
            legend_labels = class_names
        else:
            legend_labels = [f"Clase {i}" for i in range(
                len(np.unique(y_train)))]

        # Crear leyenda para las clases
        handles, _ = scatter.legend_elements()
        ax.legend(handles, legend_labels, title="Clases", loc="best")

        ax.set_title("Frontera de Decisión")

        # Mostrar la figura
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.pyplot(fig, use_container_width=True)

        # Enlace para descargar
        st.markdown(
            get_image_download_link(
                fig, "frontera_decision", "📥 Descargar visualización de frontera"),
            unsafe_allow_html=True
        )

        # Explicación adicional
        st.markdown("""
        **Nota:** Esta visualización muestra cómo el árbol de decisión divide el espacio de características
        en regiones de decisión. Cada color representa una clase diferente.
        """)

        # Advertencia sobre dimensionalidad
        if len(feature_names) > 2:
            st.warning("""
            ⚠️ Esta visualización solo muestra 2 características seleccionadas. Se entrena un nuevo modelo 
            simplificado solo con estas 2 características para poder visualizar la frontera de decisión. 
            El modelo real utiliza todas las características y puede tener diferentes decisiones.
            """)

        # Mostrar código para generar esta visualización
        code_boundary = generate_decision_boundary_code(
            feature_names_boundary, class_names
        )

        show_code_with_download(
            code_boundary, "Código para generar la frontera de decisión", "frontera_decision.py"
        )

    except Exception as e:
        st.error(
            f"Error al crear la visualización de frontera de decisión: {str(e)}")
        st.error(
            "Asegúrate de que el modelo sea compatible con DecisionBoundaryDisplay de sklearn.")
