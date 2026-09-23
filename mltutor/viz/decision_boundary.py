import pandas as pd
from mltutor.desktop import ui as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.inspection import DecisionBoundaryDisplay

from mltutor.utils import get_image_download_link, show_code_with_download
from mltutor.algorithms.code_examples import generate_decision_boundary_code


def plot_decision_boundary(model_2d, X, y, feature_names, class_names):
    st.markdown("### Selección de Características")
    st.markdown("Selecciona 2 características para visualizar:")

    col1, col2 = st.columns(2)

    with col1:
        feature1 = st.selectbox(
            "Primera característica:",
            feature_names,
            index=0,
            key="viz_feature1"
        )

    with col2:
        feature2 = st.selectbox(
            "Segunda característica:",
            feature_names,
            index=min(1, len(feature_names) - 1),
            key="viz_feature2"
        )

    if feature1 != feature2:
        # Obtener índices de las características seleccionadas
        feature_idx = [feature_names.index(
            feature1), feature_names.index(feature2)]

        with st.spinner("Generando frontera de decisión..."):

            try:
                # Extraer las características seleccionadas
                if hasattr(X, 'iloc'):  # DataFrame
                    X_2d = X.iloc[:, feature_idx].values
                else:  # numpy array
                    X_2d = X[:, feature_idx]

                # Entrenar un modelo con solo estas 2 características
                model_2d.fit(X_2d, y)

                # Crear la visualización
                fig, ax = plt.subplots(figsize=(10, 8))

                # Crear malla de puntos con resolución adaptativa
                # Reducir resolución para KNN para evitar problemas de memoria
                n_samples = len(X_2d)
                if hasattr(model_2d, '_estimator_type') and 'KNeighbors' in str(type(model_2d)):
                    # Para KNN, usar resolución más baja para evitar OOM
                    if n_samples > 1000:
                        h = 0.1  # Resolución muy baja para datasets grandes
                    elif n_samples > 500:
                        h = 0.05  # Resolución baja para datasets medianos
                    else:
                        h = 0.02  # Resolución normal para datasets pequeños
                else:
                    h = 0.02  # Resolución normal para otros algoritmos

                x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

                # Limitar el rango para evitar mallas gigantes
                x_range = x_max - x_min
                y_range = y_max - y_min
                max_range = max(x_range, y_range)

                # Si el rango es muy grande, limitarlo
                if max_range > 100:
                    # Máximo 200 puntos por dimensión
                    h = max(h, max_range / 200)

                # Asegurar que el paso no sea demasiado grande
                min_points = 10  # Mínimo 10 puntos por dimensión
                h = min(h, x_range / min_points, y_range / min_points)

                # Crear arrays para la malla
                x_points = np.arange(x_min, x_max, h)
                y_points = np.arange(y_min, y_max, h)

                # Asegurar que tenemos al menos 2 puntos en cada dimensión
                if len(x_points) < 2:
                    x_points = np.linspace(x_min, x_max, 10)
                if len(y_points) < 2:
                    y_points = np.linspace(y_min, y_max, 10)

                xx, yy = np.meshgrid(x_points, y_points)

                # Verificar el tamaño de la malla antes de continuar
                mesh_size = xx.shape[0] * xx.shape[1]
                st.info(
                    f"Malla creada: {xx.shape[0]} x {xx.shape[1]} = {mesh_size:,} puntos")

                if mesh_size > 50000:  # Límite de seguridad
                    st.warning(
                        f"⚠️ La malla es muy grande ({mesh_size:,} puntos). Reduciendo resolución...")
                    # Reducir a máximo 200x200
                    max_dim = 200
                    x_points = np.linspace(
                        x_min, x_max, min(max_dim, len(x_points)))
                    y_points = np.linspace(
                        y_min, y_max, min(max_dim, len(y_points)))
                    xx, yy = np.meshgrid(x_points, y_points)
                    mesh_size = xx.shape[0] * xx.shape[1]
                    st.info(
                        f"Nueva resolución: {xx.shape[0]} x {xx.shape[1]} = {mesh_size:,} puntos")

                # Predecir en la malla con manejo de memoria
                try:
                    with st.spinner(f"Calculando predicciones en {mesh_size:,} puntos..."):
                        mesh_points = np.c_[xx.ravel(), yy.ravel()]
                        Z = model_2d.predict(mesh_points)
                        Z = Z.reshape(xx.shape)

                        # Verificar las dimensiones antes de continuar
                        if Z.shape[0] < 2 or Z.shape[1] < 2:
                            st.error(
                                f"❌ Error: La malla resultante es demasiado pequeña: {Z.shape}")
                            st.info(
                                "💡 Aumenta el rango de datos o ajusta los parámetros.")
                            return
                except MemoryError:
                    st.error(
                        "❌ Error de memoria. El dataset es demasiado grande para esta resolución.")
                    st.info(
                        "💡 Sugerencias: Prueba con un dataset más pequeño o usa un algoritmo menos intensivo como Decision Trees.")
                    return

                # Frontera de decisión para clasificación
                #########################################
                n_classes = len(np.unique(y))

                # Usar diferentes mapas de colores según el número de clases
                if n_classes == 2:
                    contour = ax.contourf(
                        xx, yy, Z, alpha=0.3, cmap='RdBu', levels=50)
                    scatter_cmap = 'RdBu'
                else:
                    contour = ax.contourf(
                        xx, yy, Z, alpha=0.3, cmap='Set3', levels=n_classes)
                    scatter_cmap = 'Set3'

                # Scatter plot de los datos
                scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y,
                                     cmap=scatter_cmap, edgecolor='black', s=50, alpha=0.8)

                # Leyenda para clasificación
                if class_names and n_classes <= 10:
                    import matplotlib.patches as mpatches
                    if n_classes == 2:
                        colors = ['#d7191c', '#2c7bb6']  # RdBu colors
                    else:
                        colors = plt.cm.Set3(
                            np.linspace(0, 1, n_classes))

                    patches = [mpatches.Patch(color=colors[i],
                                              label=class_names[i])
                               for i in range(min(len(class_names), n_classes))]
                    ax.legend(handles=patches,
                              loc='best', title='Clases')

                ax.set_title(
                    f'Frontera de Decisión para {feature1} vs {feature2}')

                ax.set_xlabel(feature1)
                ax.set_ylabel(feature2)
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

                # Enlace para descargar
                st.markdown(
                    get_image_download_link(
                        fig, "frontera_decision", "📥 Descargar visualización de frontera"),
                    unsafe_allow_html=True
                )

                # Advertencia sobre dimensionalidad
                if len(feature_names) > 2:
                    st.warning("""
                    ⚠️ Esta visualización solo muestra 2 características seleccionadas. Se entrena un nuevo modelo 
                    simplificado solo con estas 2 características para poder visualizar la frontera de decisión. 
                    El modelo real utiliza todas las características y puede tener diferentes decisiones.
                    """)

                # Mostrar código para generar esta visualización
                code_boundary = generate_decision_boundary_code(
                    feature_names, class_names)

                show_code_with_download(
                    code_boundary, "Código para generar la frontera de decisión", "frontera_decision.py"
                )
            except Exception as e:
                st.error(
                    f"Error al crear la visualización de frontera de decisión: {str(e)}"
                )


def plot_decision_surface(model_2d, feature_names, X, y):
    st.markdown("### Selección de Características")
    st.markdown("Selecciona 2 características para visualizar:")

    col1, col2 = st.columns(2)

    with col1:
        feature1 = st.selectbox(
            "Primera característica:",
            feature_names,
            index=0,
            key="viz_feature1"
        )

    with col2:
        feature2 = st.selectbox(
            "Segunda característica:",
            feature_names,
            index=min(1, len(feature_names) - 1),
            key="viz_feature2"
        )

    if feature1 != feature2:
        with st.spinner("Generando superficie de decisión..."):
            # Obtener índices de las características seleccionadas
            feature_idx = [feature_names.index(
                feature1), feature_names.index(feature2)]

            # Extraer las características seleccionadas
            if hasattr(X, 'iloc'):  # DataFrame
                X_2d = X.iloc[:, feature_idx].values
            else:  # numpy array
                X_2d = X[:, feature_idx]

            # Entrenar un modelo con solo estas 2 características
            model_2d.fit(X_2d, y)

            # Crear la visualización
            fig, ax = plt.subplots(figsize=(10, 8))

            # Crear malla de puntos con resolución adaptativa
            # Reducir resolución para KNN para evitar problemas de memoria
            n_samples = len(X_2d)
            if hasattr(model_2d, '_estimator_type') and 'KNeighbors' in str(type(model_2d)):
                # Para KNN, usar resolución más baja para evitar OOM
                if n_samples > 1000:
                    h = 0.1  # Resolución muy baja para datasets grandes
                elif n_samples > 500:
                    h = 0.05  # Resolución baja para datasets medianos
                else:
                    h = 0.02  # Resolución normal para datasets pequeños
            else:
                h = 0.02  # Resolución normal para otros algoritmos

            x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
            y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

            # Limitar el rango para evitar mallas gigantes
            x_range = x_max - x_min
            y_range = y_max - y_min
            max_range = max(x_range, y_range)

            # Si el rango es muy grande, limitarlo
            if max_range > 100:
                h = max(h, max_range / 200)  # Máximo 200 puntos por dimensión

            # Asegurar que el paso no sea demasiado grande
            min_points = 10  # Mínimo 10 puntos por dimensión
            h = min(h, x_range / min_points, y_range / min_points)

            # Crear arrays para la malla
            x_points = np.arange(x_min, x_max, h)
            y_points = np.arange(y_min, y_max, h)

            # Asegurar que tenemos al menos 2 puntos en cada dimensión
            if len(x_points) < 2:
                x_points = np.linspace(x_min, x_max, 10)
            if len(y_points) < 2:
                y_points = np.linspace(y_min, y_max, 10)

            xx, yy = np.meshgrid(x_points, y_points)

            # Verificar el tamaño de la malla antes de continuar
            mesh_size = xx.shape[0] * xx.shape[1]
            msg = f"- Malla creada: {xx.shape[0]} x {xx.shape[1]} = {mesh_size:,} puntos"

            if mesh_size > 50000:  # Límite de seguridad

                msg += f"\n - ⚠️ La malla es muy grande ({mesh_size:,} puntos). Reduciendo resolución..."
                # Reducir a máximo 200x200
                max_dim = 200
                x_points = np.linspace(
                    x_min, x_max, min(max_dim, len(x_points)))
                y_points = np.linspace(
                    y_min, y_max, min(max_dim, len(y_points)))
                xx, yy = np.meshgrid(x_points, y_points)
                mesh_size = xx.shape[0] * xx.shape[1]

                msg += f"\n - Nueva resolución: {xx.shape[0]} x {xx.shape[1]} = {mesh_size:,} puntos"
            st.info(msg)

            # Predecir en la malla con manejo de memoria
            try:
                with st.spinner(f"Calculando predicciones en {mesh_size:,} puntos..."):
                    mesh_points = np.c_[xx.ravel(), yy.ravel()]
                    Z = model_2d.predict(mesh_points)
                    Z = Z.reshape(xx.shape)

                    # Verificar las dimensiones antes de continuar
                    if Z.shape[0] < 2 or Z.shape[1] < 2:
                        st.error(
                            f"❌ Error: La malla resultante es demasiado pequeña: {Z.shape}")
                        st.info(
                            "💡 Aumenta el rango de datos o ajusta los parámetros.")
                        return
            except MemoryError:
                st.error(
                    "❌ Error de memoria. El dataset es demasiado grande para KNN con esta resolución.")
                st.info(
                    "💡 Sugerencias: Prueba con un dataset más pequeño o usa un algoritmo menos intensivo como Decision Trees.")
                return

            # Superficie de predicción para regresión
            #########################################
            contour = ax.contourf(
                xx, yy, Z, alpha=0.6, cmap='viridis', levels=20)
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y,
                                 cmap='viridis', edgecolor='black', s=50, alpha=0.8)

            # Barra de colores
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Valor Objetivo')

            ax.set_title(
                f'Superficie de Predicción para {feature1} vs {feature2}')

            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
