from utils import (
    get_image_download_link, generate_model_code, export_model_pickle, export_model_onnx,
    create_info_box, format_number, show_code_with_download
)
from tree_visualizer import (
    render_tree_visualization,
    create_tree_visualization, get_tree_text
)
from ui import (
    setup_page, init_session_state, show_welcome_page,
    display_feature_importance, display_model_export_options, create_prediction_interface
)
from sklearn.model_selection import train_test_split
from decision_boundary import plot_decision_boundary
from model_evaluation import evaluate_classification_model, evaluate_regression_model, show_detailed_evaluation
from model_training import train_decision_tree, predict_sample, train_linear_model, train_knn_model
from dataset_manager import load_data, preprocess_data, create_dataset_selector, load_dataset_from_file
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def run_knn_app():
    """Ejecuta la aplicación específica de K-Nearest Neighbors (KNN) para clasificación y regresión."""
    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    st.header("🔍 K-Nearest Neighbors (KNN)")
    st.markdown("Aprende sobre K-vecinos más cercanos de forma interactiva.")

    # Explicación teórica
    with st.expander("ℹ️ ¿Qué es K-Nearest Neighbors?", expanded=False):
        st.markdown("""
        K-Nearest Neighbors (KNN) es un algoritmo supervisado que predice la clase o el valor de una muestra basándose en las muestras más cercanas en el espacio de características.

        **Características principales:**
        - No requiere entrenamiento explícito (modelo perezoso)
        - Puede usarse para clasificación y regresión
        - La predicción depende de la distancia a los vecinos más cercanos
        - Sensible a la escala de los datos y a la elección de K
        """)

    # Variables para almacenar datos
    dataset_loaded = False
    X, y, feature_names, class_names, dataset_info, task_type = None, None, None, None, None, None

    # Inicializar el estado de la pestaña activa si no existe
    if 'active_tab_knn' not in st.session_state:
        st.session_state.active_tab_knn = 0

    # Crear pestañas para organizar la información
    tab_options = [
        "📊 Datos",
        "🏋️ Entrenamiento",
        "📈 Evaluación",
        "📉 Visualización",
        "🔮 Predicciones"
    ]

    tab_cols = st.columns(len(tab_options))

    # Estilo CSS para los botones de pestañas (KNN)
    st.markdown("""
    <style>
    div.tab-button-knn > button {
        border-radius: 4px 4px 0 0;
        padding: 10px;
        width: 100%;
        white-space: nowrap;
        background-color: #F0F2F6;
        border-bottom: 2px solid #E0E0E0;
        color: #333333;
    }
    div.tab-button-knn-active > button {
        background-color: #E3F2FD !important;
        border-bottom: 2px solid #1E88E5 !important;
        font-weight: bold !important;
        color: #1E88E5 !important;
    }
    div.tab-button-knn > button:hover {
        background-color: #E8EAF6;
    }
    </style>
    """, unsafe_allow_html=True)

    for i, (tab_name, col) in enumerate(zip(tab_options, tab_cols)):
        button_key = f"tab_knn_{i}"
        button_style = "tab-button-knn-active" if st.session_state.active_tab_knn == i else "tab-button-knn"
        is_active = st.session_state.active_tab_knn == i
        with col:
            st.markdown(f"<div class='{button_style}'>",
                        unsafe_allow_html=True)
            if st.button(tab_name, key=button_key, use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.active_tab_knn = i
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Separador visual
    st.markdown("---")

    # SELECTOR UNIFICADO DE DATASET (solo mostrar en pestañas que lo necesiten)
    # Exploración y Entrenamiento
    if st.session_state.active_tab_knn in [0, 1]:
        st.markdown("### 📊 Selección de Dataset")

        # Inicializar dataset seleccionado si no existe
        if 'selected_dataset_knn' not in st.session_state:
            st.session_state.selected_dataset_knn = "🌸 Iris - Clasificación de flores"

        # Lista base de datasets predefinidos
        builtin_datasets = [
            "🌸 Iris - Clasificación de flores",
            "🍷 Vino - Clasificación de vinos",
            "🔬 Cáncer - Diagnóstico binario",
            "🚢 Titanic - Supervivencia",
            "💰 Propinas - Predicción de propinas",
            "🏠 Viviendas California - Precios",
            "🐧 Pingüinos - Clasificación de especies"
        ]

        # Añadir datasets CSV cargados si existen
        available_datasets = builtin_datasets.copy()
        if 'csv_datasets' in st.session_state:
            available_datasets.extend(st.session_state.csv_datasets.keys())

        # Asegurar que el dataset seleccionado esté en la lista disponible
        if st.session_state.selected_dataset_knn not in available_datasets:
            st.session_state.selected_dataset_knn = builtin_datasets[0]

        # Selector unificado
        dataset_option = st.selectbox(
            "Dataset:",
            available_datasets,
            index=available_datasets.index(
                st.session_state.selected_dataset_knn),
            key="unified_dataset_selector_knn",
            help="El dataset seleccionado se mantendrá entre las pestañas de Exploración y Entrenamiento"
        )

        # Actualizar la variable de sesión
        st.session_state.selected_dataset_knn = dataset_option

        # Separador después del selector
        st.markdown("---")
    else:
        # Para otras pestañas, mostrar qué dataset está seleccionado actualmente
        if hasattr(st.session_state, 'selected_dataset_knn'):
            st.info(
                f"📊 **Dataset actual:** {st.session_state.selected_dataset_knn}")
            st.markdown("---")

    # Lógica de cada pestaña (similar a los otros algoritmos)
    from dataset_manager import create_dataset_selector, load_data
    from model_training import train_knn_model
    from model_evaluation import evaluate_classification_model, evaluate_regression_model, show_detailed_evaluation
    from ui import create_prediction_interface
    import seaborn as sns

    # Estado de datos y modelo
    if 'knn_dataset' not in st.session_state:
        st.session_state.knn_dataset = None
    if 'knn_X' not in st.session_state:
        st.session_state.knn_X = None
    if 'knn_y' not in st.session_state:
        st.session_state.knn_y = None
    if 'knn_feature_names' not in st.session_state:
        st.session_state.knn_feature_names = None
    if 'knn_class_names' not in st.session_state:
        st.session_state.knn_class_names = None
    if 'knn_task_type' not in st.session_state:
        st.session_state.knn_task_type = 'Clasificación'
    if 'knn_model' not in st.session_state:
        st.session_state.knn_model = None
    if 'knn_metrics' not in st.session_state:
        st.session_state.knn_metrics = None
    if 'knn_trained' not in st.session_state:
        st.session_state.knn_trained = False

    tab = st.session_state.active_tab_knn

    # Pestaña 0: Datos
    if tab == 0:
        st.header("Exploración de Datos")

        try:
            # Cargar datos para exploración usando el dataset seleccionado
            X, y, feature_names, class_names, info, task_type = load_data(
                st.session_state.selected_dataset_knn)

            # Crear DataFrame para mostrar los datos
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                data = X.copy()
            else:
                data = pd.DataFrame(X, columns=feature_names)

            # Añadir la variable objetivo
            target_name = 'target'
            if isinstance(info, dict):
                target_name = info.get('target', 'target')
            data[target_name] = y

            # Actualizar el estado de la sesión
            st.session_state.knn_dataset = st.session_state.selected_dataset_knn
            st.session_state.knn_X = X
            st.session_state.knn_y = y
            st.session_state.knn_feature_names = feature_names
            st.session_state.knn_class_names = class_names
            st.session_state.knn_task_type = task_type

            # Mostrar información del dataset
            st.markdown("### Información del Dataset")
            from utils import create_info_box
            st.markdown(create_info_box(info), unsafe_allow_html=True)

            # Mostrar las primeras filas de los datos
            st.markdown("### Vista previa de datos")
            st.dataframe(data.head(20), use_container_width=True)
            st.markdown(f"**Características:** {', '.join(feature_names)}")

            # Determinar el nombre de la variable objetivo
            if isinstance(info, dict):
                target_display = info.get('target', 'target')
            else:
                target_display = 'target'
            st.markdown(f"**Variable objetivo:** {target_display}")

            st.markdown(f"**Tipo de tarea:** {task_type}")

            # Mostrar clases correctamente
            if class_names is not None:
                if isinstance(class_names, list):
                    st.markdown(
                        f"**Clases:** {', '.join(map(str, class_names))}")
                else:
                    st.markdown(f"**Clases:** {class_names}")
            else:
                st.markdown("**Clases:** N/A (regresión)")

            if X is not None and hasattr(X, 'shape') and len(X.shape) >= 2:
                st.markdown(
                    f"**Tamaño del dataset:** {X.shape[0]} muestras, {X.shape[1]} características")
            elif X is not None and hasattr(X, 'shape') and len(X.shape) == 1:
                st.markdown(
                    f"**Tamaño del dataset:** {X.shape[0]} muestras, 1 característica")
            else:
                st.markdown("**Tamaño del dataset:** No disponible")
            st.session_state.knn_trained = False

        except Exception as e:
            st.error(f"Error al cargar el dataset: {str(e)}")
            st.info(
                "Por favor, selecciona un dataset válido para continuar con la exploración.")

    # Pestaña 1: Entrenamiento
    elif tab == 1:
        st.header("Configuración del Modelo KNN")

        # Inicializar variables de sesión necesarias
        if 'dataset_option_knn' not in st.session_state:
            st.session_state.dataset_option_knn = st.session_state.selected_dataset_knn

        # Cargar datos para la vista previa si cambia el dataset o si no se ha cargado
        if st.session_state.selected_dataset_knn != st.session_state.dataset_option_knn or st.session_state.knn_X is None:
            try:
                X, y, feature_names, class_names, info, task_type = load_data(
                    st.session_state.selected_dataset_knn)

                st.session_state.dataset_option_knn = st.session_state.selected_dataset_knn

                # Actualizar información del dataset
                st.session_state.knn_dataset = st.session_state.selected_dataset_knn
                st.session_state.knn_X = X
                st.session_state.knn_y = y
                st.session_state.knn_feature_names = feature_names
                st.session_state.knn_class_names = class_names
                st.session_state.knn_task_type = task_type

                # Mostrar información del dataset
                st.markdown("### Información del Dataset")
                from utils import create_info_box
                st.markdown(create_info_box(info), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error al cargar el dataset: {str(e)}")

        # Verificar que los datos estén disponibles
        if st.session_state.knn_X is not None and st.session_state.knn_y is not None:
            st.markdown("### Parámetros del Modelo")
            st.markdown("Configura los hiperparámetros del modelo KNN:")

            col1, col2, col3 = st.columns(3)
            with col1:
                n_neighbors = st.number_input(
                    "Vecinos (K)", min_value=1, max_value=20, value=5, step=1, key="knn_n_neighbors")
            with col2:
                weights = st.selectbox(
                    "Pesos", options=["uniform", "distance"], key="knn_weights")
            with col3:
                metric = st.selectbox("Métrica", options=[
                                      "minkowski", "euclidean", "manhattan"], key="knn_metric")

            if st.button("🚀 Entrenar Modelo KNN", key="train_knn_button", type="primary"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        result = train_knn_model(
                            st.session_state.knn_X,
                            st.session_state.knn_y,
                            task_type=st.session_state.knn_task_type,
                            n_neighbors=n_neighbors,
                            weights=weights,
                            metric=metric
                        )
                        st.session_state.knn_model = result["model"]
                        st.session_state.knn_metrics = result["evaluation"]
                        st.session_state.knn_trained = True
                        st.success("¡Modelo KNN entrenado correctamente!")

                        # Sugerir ir a la pestaña de evaluación
                        st.info(
                            "👉 Ve a la pestaña '📈 Evaluación' para ver los resultados del modelo.")

                    except Exception as e:
                        st.error(f"Error al entrenar el modelo: {str(e)}")
        else:
            st.info("Primero selecciona y carga un dataset en la pestaña de Datos.")

    # Pestaña 2: Evaluación
    elif tab == 2:
        st.header("📈 Evaluación del Modelo KNN")
        if st.session_state.knn_trained and st.session_state.knn_metrics is not None:
            # Obtener las predicciones del modelo
            if hasattr(st.session_state, 'knn_X') and hasattr(st.session_state, 'knn_y'):
                from sklearn.model_selection import train_test_split

                # Recrear el split de entrenamiento/prueba con los mismos parámetros
                X_train, X_test, y_train, y_test = train_test_split(
                    st.session_state.knn_X,
                    st.session_state.knn_y,
                    test_size=0.3,
                    random_state=42
                )

                # Obtener las predicciones
                y_pred = st.session_state.knn_model.predict(X_test)

                # Mostrar evaluación detallada del modelo usando la misma función que otros algoritmos
                show_detailed_evaluation(
                    y_test,
                    y_pred,
                    st.session_state.knn_class_names if st.session_state.knn_task_type == "Clasificación" else None,
                    st.session_state.knn_task_type
                )
            else:
                st.error(
                    "No se encontraron los datos necesarios para la evaluación.")
        else:
            st.info("Primero entrena un modelo KNN.")

    # Pestaña 3: Visualización
    elif tab == 3:
        st.header("📉 Visualización de KNN")
        if st.session_state.knn_trained and st.session_state.knn_model is not None:
            X = st.session_state.knn_X
            y = st.session_state.knn_y
            model = st.session_state.knn_model
            feature_names = st.session_state.knn_feature_names
            task_type = st.session_state.knn_task_type

            st.markdown("### Opciones de Visualización")

            # Selector de tipo de visualización
            viz_options = []

            # Siempre disponible: Distribución de predicciones
            viz_options.append("📊 Distribución de Predicciones")

            # Si hay al menos 2 características: Frontera de decisión/superficie
            if X.shape[1] >= 2:
                if task_type == "Clasificación":
                    viz_options.append("🎯 Frontera de Decisión")
                else:
                    viz_options.append("🏔️ Superficie de Predicción")

            # Si es clasificación: Matriz de distancias
            if task_type == "Clasificación":
                viz_options.append("📏 Análisis de Distancias")

            viz_type = st.selectbox("Tipo de visualización:", viz_options)

            if viz_type == "📊 Distribución de Predicciones":
                # Recrear el split para obtener predicciones
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                y_pred = model.predict(X_test)

                if task_type == "Clasificación":
                    # Mostrar distribución de clases predichas vs reales
                    import pandas as pd

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Distribución Real")
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        real_counts = pd.Series(
                            y_test).value_counts().sort_index()
                        ax1.bar(range(len(real_counts)),
                                real_counts.values, alpha=0.7, color='skyblue')
                        ax1.set_xlabel('Clase')
                        ax1.set_ylabel('Frecuencia')
                        ax1.set_title('Distribución Real')
                        if st.session_state.knn_class_names:
                            ax1.set_xticks(range(len(real_counts)))
                            ax1.set_xticklabels(
                                [st.session_state.knn_class_names[i] for i in real_counts.index], rotation=45)
                        st.pyplot(fig1)

                    with col2:
                        st.markdown("#### Distribución Predicha")
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        pred_counts = pd.Series(
                            y_pred).value_counts().sort_index()
                        ax2.bar(range(len(pred_counts)), pred_counts.values,
                                alpha=0.7, color='lightcoral')
                        ax2.set_xlabel('Clase')
                        ax2.set_ylabel('Frecuencia')
                        ax2.set_title('Distribución Predicha')
                        if st.session_state.knn_class_names:
                            ax2.set_xticks(range(len(pred_counts)))
                            ax2.set_xticklabels(
                                [st.session_state.knn_class_names[i] for i in pred_counts.index], rotation=45)
                        st.pyplot(fig2)

                else:  # Regresión
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Valores Reales vs Predichos")
                        fig1, ax1 = plt.subplots(figsize=(6, 5))
                        ax1.scatter(y_test, y_pred, alpha=0.6)
                        ax1.plot([y_test.min(), y_test.max()], [
                                 y_test.min(), y_test.max()], 'r--', lw=2)
                        ax1.set_xlabel('Valores Reales')
                        ax1.set_ylabel('Valores Predichos')
                        ax1.set_title('Predicciones vs Realidad')
                        st.pyplot(fig1)

                    with col2:
                        st.markdown("#### Distribución de Errores")
                        fig2, ax2 = plt.subplots(figsize=(6, 5))
                        errors = y_test - y_pred
                        ax2.hist(errors, bins=20, alpha=0.7,
                                 color='lightgreen', edgecolor='black')
                        ax2.axvline(x=0, color='red',
                                    linestyle='--', linewidth=2)
                        ax2.set_xlabel('Error (Real - Predicho)')
                        ax2.set_ylabel('Frecuencia')
                        ax2.set_title('Distribución de Errores')
                        st.pyplot(fig2)

            elif viz_type in ["🎯 Frontera de Decisión", "🏔️ Superficie de Predicción"]:
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

                    # Extraer las características seleccionadas
                    if hasattr(X, 'iloc'):  # DataFrame
                        X_2d = X.iloc[:, feature_idx].values
                    else:  # numpy array
                        X_2d = X[:, feature_idx]

                    # Entrenar un modelo KNN con solo estas 2 características
                    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

                    if task_type == "Clasificación":
                        model_2d = KNeighborsClassifier(
                            n_neighbors=model.n_neighbors,
                            weights=model.weights,
                            metric=model.metric
                        )
                    else:
                        model_2d = KNeighborsRegressor(
                            n_neighbors=model.n_neighbors,
                            weights=model.weights,
                            metric=model.metric
                        )

                    model_2d.fit(X_2d, y)

                    # Crear la visualización
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Crear malla de puntos
                    h = 0.02
                    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                         np.arange(y_min, y_max, h))

                    # Predecir en la malla
                    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    if task_type == "Clasificación":
                        # Frontera de decisión para clasificación
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
                        if st.session_state.knn_class_names and n_classes <= 10:
                            import matplotlib.patches as mpatches
                            if n_classes == 2:
                                colors = ['#d7191c', '#2c7bb6']  # RdBu colors
                            else:
                                colors = plt.cm.Set3(
                                    np.linspace(0, 1, n_classes))

                            patches = [mpatches.Patch(color=colors[i],
                                                      label=st.session_state.knn_class_names[i])
                                       for i in range(min(len(st.session_state.knn_class_names), n_classes))]
                            ax.legend(handles=patches,
                                      loc='best', title='Clases')

                        ax.set_title(
                            f'Frontera de Decisión KNN (K={model.n_neighbors})')

                    else:
                        # Superficie de predicción para regresión
                        contour = ax.contourf(
                            xx, yy, Z, alpha=0.6, cmap='viridis', levels=20)
                        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y,
                                             cmap='viridis', edgecolor='black', s=50, alpha=0.8)

                        # Barra de colores
                        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                        cbar.set_label('Valor Objetivo')

                        ax.set_title(
                            f'Superficie de Predicción KNN (K={model.n_neighbors})')

                    ax.set_xlabel(feature1)
                    ax.set_ylabel(feature2)
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)

                    # Información adicional
                    st.info(f"""
                    🔍 **Información de la visualización:**
                    - **Características:** {feature1} vs {feature2}
                    - **Número de vecinos (K):** {model.n_neighbors}
                    - **Tipo de peso:** {model.weights}
                    - **Métrica de distancia:** {model.metric}
                    - **Tipo de tarea:** {task_type}
                    """)

                else:
                    st.warning(
                        "Por favor selecciona dos características diferentes.")

            elif viz_type == "📏 Análisis de Distancias":
                st.markdown("### Análisis de Distancias entre Clases")

                # Calcular distancias promedio entre clases
                from sklearn.metrics.pairwise import pairwise_distances

                # Recrear el split para análisis
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                classes = np.unique(y_train)
                n_classes = len(classes)

                if n_classes > 1:
                    # Matriz de distancias promedio entre clases
                    dist_matrix = np.zeros((n_classes, n_classes))

                    for i, class1 in enumerate(classes):
                        for j, class2 in enumerate(classes):
                            if i != j:
                                X_class1 = X_train[y_train == class1]
                                X_class2 = X_train[y_train == class2]

                                # Calcular distancia promedio entre las dos clases
                                distances = pairwise_distances(
                                    X_class1, X_class2, metric=model.metric)
                                dist_matrix[i, j] = np.mean(distances)

                    # Visualizar matriz de distancias
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto')

                    # Añadir valores a las celdas
                    for i in range(n_classes):
                        for j in range(n_classes):
                            if i != j:
                                text = ax.text(j, i, f'{dist_matrix[i, j]:.2f}',
                                               ha="center", va="center", color="white")

                    # Configurar etiquetas
                    if st.session_state.knn_class_names:
                        class_labels = [
                            st.session_state.knn_class_names[int(c)] for c in classes]
                    else:
                        class_labels = [f'Clase {int(c)}' for c in classes]

                    ax.set_xticks(range(n_classes))
                    ax.set_yticks(range(n_classes))
                    ax.set_xticklabels(class_labels, rotation=45)
                    ax.set_yticklabels(class_labels)
                    ax.set_title(
                        f'Distancias Promedio entre Clases\n(Métrica: {model.metric})')

                    plt.colorbar(im, ax=ax, label='Distancia Promedio')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Análisis de separabilidad
                    min_dist = np.min(dist_matrix[dist_matrix > 0])
                    max_dist = np.max(dist_matrix)

                    st.markdown("### Análisis de Separabilidad")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Distancia Mínima", f"{min_dist:.3f}")
                    with col2:
                        st.metric("Distancia Máxima", f"{max_dist:.3f}")
                    with col3:
                        ratio = max_dist / min_dist if min_dist > 0 else 0
                        st.metric("Ratio Max/Min", f"{ratio:.2f}")

                    if ratio > 5:
                        st.success(
                            "🟢 Las clases están bien separadas (ratio > 5)")
                    elif ratio > 2:
                        st.warning(
                            "🟡 Separación moderada entre clases (2 < ratio ≤ 5)")
                    else:
                        st.error("🔴 Las clases están muy próximas (ratio ≤ 2)")
                else:
                    st.info(
                        "Se necesitan al menos 2 clases para el análisis de distancias.")

        else:
            st.info("Primero entrena un modelo KNN.")

    # Pestaña 4: Predicciones
    elif tab == 4:
        st.header("🔮 Predicciones con KNN")
        if st.session_state.knn_trained and st.session_state.knn_model is not None:
            create_prediction_interface(
                st.session_state.knn_model,
                st.session_state.knn_feature_names,
                st.session_state.knn_class_names,
                st.session_state.knn_task_type,
                st.session_state.knn_X,
                st.session_state.knn_dataset
            )
        else:
            st.info("Primero entrena un modelo KNN.")


#!/usr/bin/env python3
"""
MLTutor: Plataforma educativa para el aprendizaje de Machine Learning.
"""


# Importar módulos refactorizados


def main():
    """Función principal que ejecuta la aplicación MLTutor."""
    # Configuración de la página
    setup_page()

    # Inicializar estado de la sesión
    init_session_state()

    # Configurar navegación principal con botones en lugar de radio
    st.sidebar.markdown("### Navegación:")

    # Estilo para los botones de navegación
    button_style = """
    <style>
    div.stButton > button {
        width: 100%;
        text-align: left;
        padding: 10px;
        margin-bottom: 5px;
        border-radius: 5px;
        font-weight: normal;
    }
    div.stButton > button:hover {
        background-color: #E3F2FD;
        border-color: #90CAF9;
    }
    </style>
    """
    st.sidebar.markdown(button_style, unsafe_allow_html=True)

    # Crear los botones de navegación
    if st.sidebar.button("🏠 Inicio",
                         key="nav_home",
                         use_container_width=True,
                         type="secondary" if st.session_state.navigation != "🏠 Inicio" else "primary"):
        st.session_state.navigation = "🏠 Inicio"
        st.rerun()

    if st.sidebar.button("🌲 Árboles de Decisión",
                         key="nav_trees",
                         use_container_width=True,
                         type="secondary" if st.session_state.navigation != "🌲 Árboles de Decisión" else "primary"):
        st.session_state.navigation = "🌲 Árboles de Decisión"
        st.rerun()

    if st.sidebar.button("📊 Regresión",
                         key="nav_linear",
                         use_container_width=True,
                         type="secondary" if st.session_state.navigation != "📊 Regresión" else "primary"):
        st.session_state.navigation = "📊 Regresión"
        st.rerun()

    if st.sidebar.button("🔍 K-Nearest Neighbors",
                         key="nav_knn",
                         use_container_width=True,
                         type="secondary" if st.session_state.navigation != "🔍 K-Nearest Neighbors" else "primary"):
        st.session_state.navigation = "🔍 K-Nearest Neighbors"
        st.rerun()

    if st.sidebar.button("🧠 Redes Neuronales (próximamente)",
                         key="nav_nn",
                         use_container_width=True,
                         disabled=True):
        st.session_state.navigation = "🧠 Redes Neuronales (próximamente)"
        st.rerun()

    # Separador
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Herramientas:")

    if st.sidebar.button("📁 Cargar CSV Personalizado",
                         key="nav_csv",
                         use_container_width=True):
        st.session_state.navigation = "📁 Cargar CSV Personalizado"
        st.rerun()

    # Página de inicio
    if st.session_state.navigation == "🏠 Inicio":
        show_welcome_page()
        return

    # Páginas de algoritmos
    if st.session_state.navigation == "🌲 Árboles de Decisión":
        run_decision_trees_app()
    elif st.session_state.navigation == "📊 Regresión":
        run_linear_regression_app()
    elif st.session_state.navigation == "🔍 K-Nearest Neighbors":
        run_knn_app()
    elif st.session_state.navigation == "📁 Cargar CSV Personalizado":
        run_csv_loader_app()
    elif st.session_state.navigation in ["🧠 Redes Neuronales (próximamente)"]:
        algorithm_name = st.session_state.navigation.split(" ")[1]
        st.header(f"{algorithm_name} (próximamente)")
        st.info(
            f"La funcionalidad de {algorithm_name} estará disponible próximamente. Por ahora, puedes explorar los Árboles de Decisión.")

        # Mostrar una imagen ilustrativa según el algoritmo
        if "Regresión Logística" in st.session_state.navigation:
            st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_logistic_001.png",
                     caption="Ilustración de Regresión Logística")
        elif "K-Nearest Neighbors" in st.session_state.navigation:
            st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png",
                     caption="Ilustración de K-Nearest Neighbors")
        elif "Redes Neuronales" in st.session_state.navigation:
            st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_mlp_001.png",
                     caption="Ilustración de Redes Neuronales")


def run_decision_trees_app():
    """Ejecuta la aplicación específica de árboles de decisión."""
    st.header("🌲 Árboles de Decisión")
    st.markdown("Aprende sobre los árboles de decisión de forma interactiva")

    # Información sobre árboles de decisión
    with st.expander("ℹ️ ¿Qué son los Árboles de Decisión?", expanded=False):
        st.markdown("""
        Los árboles de decisión son algoritmos de aprendizaje supervisado que se pueden usar tanto para tareas de clasificación como de regresión.

        **Características principales:**
        - Funcionan dividiendo el conjunto de datos en subconjuntos más pequeños basándose en condiciones sobre las características
        - Son fáciles de interpretar y visualizar
        - Pueden manejar tanto datos numéricos como categóricos
        - No requieren escalado de datos

        **Limitaciones:**
        - Pueden sobreajustarse fácilmente (esto se puede controlar con parámetros como max_depth)
        - Pueden ser inestables (pequeños cambios en los datos pueden generar árboles muy diferentes)
        - No son tan precisos como algoritmos más complejos para algunas tareas

        Experimenta con diferentes configuraciones para ver cómo afectan al rendimiento del modelo.
        """)

    # Variables para almacenar datos
    dataset_loaded = False
    X, y, feature_names, class_names, dataset_info, task_type = None, None, None, None, None, None

    # Inicializar el estado de la pestaña activa si no existe
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    # Crear pestañas para organizar la información
    tab_options = [
        "📊 Datos",
        "🏋️ Entrenamiento",
        "📈 Evaluación",
        "🌲 Visualización",
        "🔍 Características",
        "🔮 Predicciones",
        "💾 Exportar Modelo"
    ]

    # Crear contenedor para los botones de las pestañas
    tab_cols = st.columns(len(tab_options))

    # Estilo CSS para los botones de pestañas (Decision Trees)
    st.markdown("""
    <style>
    div.tab-button-dt > button {
        border-radius: 4px 4px 0 0;
        padding: 10px;
        width: 100%;
        white-space: nowrap;
        background-color: #F0F2F6;
        border-bottom: 2px solid #E0E0E0;
        color: #333333;
    }
    div.tab-button-dt-active > button {
        background-color: #E3F2FD !important;
        border-bottom: 2px solid #1E88E5 !important;
        font-weight: bold !important;
        color: #1E88E5 !important;
    }
    div.tab-button-dt > button:hover {
        background-color: #E8EAF6;
    }
    </style>
    """, unsafe_allow_html=True)

    # Crear botones para las pestañas
    for i, (tab_name, col) in enumerate(zip(tab_options, tab_cols)):
        button_key = f"tab_{i}"
        button_style = "tab-button-dt-active" if st.session_state.active_tab == i else "tab-button-dt"

    # Crear botones para las pestañas
    for i, (tab_name, col) in enumerate(zip(tab_options, tab_cols)):
        button_key = f"tab_{i}"
        button_style = "tab-button-dt-active" if st.session_state.active_tab == i else "tab-button-dt"
        is_active = st.session_state.active_tab == i

        with col:
            st.markdown(f"<div class='{button_style}'>",
                        unsafe_allow_html=True)
            # Usar type="primary" para el botón activo
            if st.button(tab_name, key=button_key, use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active_tab = i
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Separador visual
    st.markdown("---")

    # SELECTOR UNIFICADO DE DATASET (solo mostrar en pestañas que lo necesiten)
    if st.session_state.active_tab in [0, 1]:  # Exploración y Entrenamiento
        st.markdown("### 📊 Selección de Dataset")

        # Inicializar dataset seleccionado si no existe
        if 'selected_dataset' not in st.session_state:
            st.session_state.selected_dataset = "🌸 Iris - Clasificación de flores"

        # Lista base de datasets predefinidos
        builtin_datasets = [
            "🌸 Iris - Clasificación de flores",
            "🍷 Vino - Clasificación de vinos",
            "🔬 Cáncer - Diagnóstico binario",
            "🚢 Titanic - Supervivencia",
            "💰 Propinas - Predicción de propinas",
            "🏠 Viviendas California - Precios",
            "🐧 Pingüinos - Clasificación de especies"
        ]

        # Añadir datasets CSV cargados si existen
        available_datasets = builtin_datasets.copy()
        if 'csv_datasets' in st.session_state:
            available_datasets.extend(st.session_state.csv_datasets.keys())

        # Asegurar que el dataset seleccionado esté en la lista disponible
        if st.session_state.selected_dataset not in available_datasets:
            st.session_state.selected_dataset = builtin_datasets[0]

        # Selector unificado
        dataset_option = st.selectbox(
            "Dataset:",
            available_datasets,
            index=available_datasets.index(st.session_state.selected_dataset),
            key="unified_dataset_selector",
            help="El dataset seleccionado se mantendrá entre las pestañas de Exploración y Entrenamiento"
        )

        # Actualizar la variable de sesión
        st.session_state.selected_dataset = dataset_option

        # Separador después del selector
        st.markdown("---")
    else:
        # Para otras pestañas, mostrar qué dataset está seleccionado actualmente
        if hasattr(st.session_state, 'selected_dataset'):
            st.info(
                f"📊 **Dataset actual:** {st.session_state.selected_dataset}")
            st.markdown("---")

    # Pestaña de Datos
    if st.session_state.active_tab == 0:
        st.header("Exploración de Datos")

        try:
            # Cargar datos para exploración
            X, y, feature_names, class_names, dataset_info, task_type = load_data(
                st.session_state.selected_dataset)

            # Convertir a DataFrames para facilitar el manejo
            # FIXED: No sobrescribir las columnas si X ya es DataFrame para evitar NaN
            if isinstance(X, pd.DataFrame):
                X_df = X.copy()  # Usar las columnas originales
                # Crear mapeo de nombres para mostrar
                column_mapping = {}
                if len(feature_names) == len(X_df.columns):
                    column_mapping = dict(zip(X_df.columns, feature_names))
            else:
                X_df = pd.DataFrame(X, columns=feature_names)
                column_mapping = {}

            y_df = pd.Series(y, name="target")

            # Mapear nombres de clases si están disponibles
            if task_type == "Clasificación" and class_names is not None:
                y_df = y_df.map(
                    {i: name for i, name in enumerate(class_names)})

            df = pd.concat([X_df, y_df], axis=1)

            # Renombrar columnas para mostrar nombres amigables si existe el mapeo
            if column_mapping:
                df_display = df.rename(columns=column_mapping)
            else:
                df_display = df

            # Mostrar información del dataset
            st.markdown("### Información del Dataset")
            st.markdown(create_info_box(dataset_info), unsafe_allow_html=True)

            # Mostrar las primeras filas de los datos
            st.markdown("### Vista previa de datos")
            st.dataframe(df_display.head(10))

            # Estadísticas descriptivas
            st.markdown("### Estadísticas Descriptivas")
            st.dataframe(df_display.describe())

            # Distribución de clases o valores objetivo
            st.markdown("### Distribución del Objetivo")

            fig, ax = plt.subplots(figsize=(10, 6))

            if task_type == "Clasificación":
                # Gráfico de barras para clasificación
                value_counts = y_df.value_counts().sort_index()
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                ax.set_title("Distribución de Clases")
                ax.set_xlabel("Clase")
                ax.set_ylabel("Cantidad")

                # Rotar etiquetas si son muchas
                if len(value_counts) > 3:
                    plt.xticks(rotation=45, ha='right')
            else:
                # Histograma para regresión
                # Convertir a numérico en caso de que sean strings
                y_numeric = pd.to_numeric(y_df, errors='coerce')
                # Usar matplotlib directamente para evitar problemas de tipo
                ax.hist(y_numeric.dropna(), bins=30,
                        alpha=0.7, edgecolor='black')
                ax.set_title("Distribución de Valores Objetivo")
                ax.set_xlabel("Valor")
                ax.set_ylabel("Frecuencia")

            # Mostrar la figura con tamaño reducido pero expandible
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.pyplot(fig, use_container_width=True)

            # Análisis de correlación
            st.markdown("### Matriz de Correlación")

            # Matriz de correlación
            corr = X_df.corr()

            # Generar máscara para el triángulo superior
            mask = np.triu(np.ones_like(corr, dtype=bool))

            # Generar mapa de calor
            fig_corr, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                        square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
            ax.set_title("Matriz de Correlación de Características")

            # Mostrar la figura con tamaño reducido pero expandible
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.pyplot(fig_corr, use_container_width=True)

            # Matriz de dispersión (Scatterplot Matrix)
            st.markdown("### Matriz de Dispersión (Pairplot)")

            # Opciones de visualización
            st.markdown("#### Opciones de visualización")
            col1, col2 = st.columns(2)

            with col1:
                # Seleccionar tipo de gráfico para la diagonal
                diag_kind = st.radio(
                    "Tipo de gráfico en la diagonal:",
                    ["Histograma", "KDE (Estimación de Densidad)"],
                    index=1,
                    horizontal=True
                )
                diag_kind = "hist" if diag_kind == "Histograma" else "kde"

            with col2:
                # Seleccionar número máximo de características
                max_features_selected = st.slider(
                    "Número máximo de características:",
                    min_value=2,
                    max_value=min(6, len(X_df.columns)),
                    value=min(4, len(X_df.columns)),
                    help="Un número mayor de características puede hacer que el gráfico sea más difícil de interpretar."
                )

            # Permitir al usuario seleccionar las características específicas
            st.markdown("#### Selecciona las características para visualizar")

            # Limitar a max_features_selected
            # Usar nombres amigables si están disponibles, sino usar originales
            if column_mapping:
                available_features = list(
                    column_mapping.values())  # Nombres amigables
                display_to_original = {
                    v: k for k, v in column_mapping.items()}  # Mapeo inverso
            else:
                available_features = X_df.columns.tolist()
                display_to_original = {}

            # Usar multiselect para seleccionar características
            selected_features = st.multiselect(
                "Características a incluir en la matriz de dispersión:",
                available_features,
                default=available_features[:max_features_selected],
                max_selections=max_features_selected,
                help=f"Selecciona hasta {max_features_selected} características para incluir en la visualización."
            )

            # Si no se seleccionó ninguna característica, usar las primeras por defecto
            if not selected_features:
                selected_features = available_features[:max_features_selected]
                st.info(
                    f"No se seleccionaron características. Usando las primeras {max_features_selected} por defecto.")

            # Convertir nombres amigables a nombres originales si es necesario
            if column_mapping:
                original_features = [display_to_original[feat]
                                     for feat in selected_features]
            else:
                original_features = selected_features

            # Crear el dataframe para la visualización
            plot_df = X_df[original_features].copy()
            # Renombrar a nombres amigables para visualización
            if column_mapping:
                plot_df = plot_df.rename(columns=column_mapping)
            # Añadir la variable objetivo para colorear
            plot_df['target'] = y_df

            # Generar el pairplot
            with st.spinner("Generando matriz de dispersión..."):
                pair_plot = sns.pairplot(
                    plot_df,
                    hue='target',
                    diag_kind=diag_kind,
                    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
                    diag_kws={'alpha': 0.5},
                    height=2.0  # Reducir altura para que sea más compacto
                )
                pair_plot.fig.suptitle(
                    "Matriz de Dispersión de Características", y=1.02, fontsize=14)

                # Mostrar la figura con tamaño reducido pero expandible
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(pair_plot.fig, use_container_width=True)

                # Enlace para descargar
                st.markdown(
                    get_image_download_link(
                        pair_plot.fig, "matriz_dispersion", "📥 Descargar matriz de dispersión"),
                    unsafe_allow_html=True
                )

            # Generar código para este análisis
            code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar tus datos (reemplaza esto con tu método de carga)
# df = pd.read_csv('tu_archivo.csv')

# Separar características y objetivo
X = df.iloc[:, :-1]  # Todas las columnas excepto la última
y = df.iloc[:, -1]   # Última columna como objetivo

# Estadísticas descriptivas
print(df.describe())

# Distribución del objetivo
fig, ax = plt.subplots(figsize=(10, 6))

# Para clasificación:
if len(np.unique(y)) <= 10:  # Si hay pocas clases únicas
    value_counts = y.value_counts().sort_index()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
    ax.set_title("Distribución de Clases")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad")
else:  # Para regresión
    sns.histplot(y, kde=True, ax=ax)
    ax.set_title("Distribución de Valores Objetivo")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# Matriz de correlación
corr = X.corr()
# Máscara para triángulo superior
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
           square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Matriz de Correlación de Características")

plt.tight_layout()
plt.show()

# Matriz de dispersión (Scatterplot Matrix)
# Seleccionar características específicas para visualizar
# Reemplaza con tus características de interés
selected_features = ['feature1', 'feature2', 'feature3']
max_features = min(6, len(selected_features))

# Crear el dataframe para la visualización
plot_df = X[selected_features].copy()
plot_df['target'] = y  # Añadir la variable objetivo para colorear

# Generar el pairplot
pair_plot = sns.pairplot(
    plot_df,
    hue='target',
    diag_kind='kde',  # Opciones: 'hist' para histograma o 'kde' para densidad
    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
    diag_kws={'alpha': 0.5},
    height=2.5
)
pair_plot.fig.suptitle(
    "Matriz de Dispersión de Características", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
"""

            show_code_with_download(
                code, "Código para análisis exploratorio", "analisis_exploratorio.py")

        except Exception as e:
            st.error(f"Error al cargar el dataset: {str(e)}")
            st.info(
                "Por favor, selecciona un dataset válido para continuar con la exploración.")

    # Pestaña de Entrenamiento
    elif st.session_state.active_tab == 1:
        st.header("Configuración del Modelo")

        # Inicializar session state variables
        if 'dataset_option' not in st.session_state:
            st.session_state.dataset_option = st.session_state.selected_dataset
        if 'tree_type' not in st.session_state:
            st.session_state.tree_type = "Clasificación"
        if 'is_trained' not in st.session_state:
            st.session_state.is_trained = False

        # Cargar datos para la vista previa si cambia el dataset o si no se ha cargado
        if st.session_state.selected_dataset != st.session_state.dataset_option or not dataset_loaded:
            try:
                X, y, feature_names, class_names, dataset_info, task_type = load_data(
                    st.session_state.selected_dataset)

                st.session_state.dataset_option = st.session_state.selected_dataset
                dataset_loaded = True

                # Mostrar información del dataset
                st.markdown("### Información del Dataset")
                st.markdown(create_info_box(dataset_info),
                            unsafe_allow_html=True)

                # Usar el tipo de tarea detectado por la función load_data
                st.markdown("### Tipo de Árbol de Decisión")

                # Usar botones en lugar de radio para seleccionar el tipo de árbol
                tipo_col1, tipo_col2 = st.columns(2)

                with tipo_col1:
                    is_classification = True
                    if "tree_type" in st.session_state:
                        is_classification = st.session_state.tree_type == "Clasificación"

                    if st.button("🏷️ Clasificación",
                                 key="btn_classification",
                                 type="primary" if is_classification else "secondary",
                                 use_container_width=True,
                                 help="Para predecir categorías o clases"):
                        tree_type = "Clasificación"
                        st.session_state.tree_type = tree_type
                        st.rerun()

                with tipo_col2:
                    is_regression = False
                    if "tree_type" in st.session_state:
                        is_regression = st.session_state.tree_type == "Regresión"

                    if st.button("📈 Regresión",
                                 key="btn_regression",
                                 type="primary" if is_regression else "secondary",
                                 use_container_width=True,
                                 help="Para predecir valores numéricos continuos"):
                        tree_type = "Regresión"
                        st.session_state.tree_type = tree_type
                        st.rerun()

                # Obtener el valor actual del tipo de árbol
                tree_type = st.session_state.get('tree_type', task_type)

                # Si no hay tree_type definido, usar el detectado
                if 'tree_type' not in st.session_state:
                    st.session_state.tree_type = task_type

                # Mostrar advertencia si la selección no coincide con el tipo de tarea detectado
                if tree_type != task_type:
                    st.warning(
                        f"Este dataset parece ser más adecuado para {task_type}. La selección actual podría no ofrecer resultados óptimos.")

            except Exception as e:
                st.error(f"Error al cargar el dataset: {str(e)}")
                dataset_loaded = False

        # Parámetros del modelo
        st.markdown("### Parámetros del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            criterion = st.selectbox(
                "Criterio de División:",
                ["gini", "entropy"] if st.session_state.get('tree_type', 'Clasificación') == "Clasificación" else [
                    "squared_error", "friedman_mse", "absolute_error"],
                index=0,
                help="Medida de la calidad de una división. Gini o Entropy para clasificación, MSE para regresión."
            )

            max_depth = st.slider(
                "Profundidad Máxima:",
                min_value=1,
                max_value=10,
                value=3,
                help="La profundidad máxima del árbol. A mayor profundidad, más complejo el modelo."
            )

        with col2:
            min_samples_split = st.slider(
                "Muestras Mínimas para División:",
                min_value=2,
                max_value=10,
                value=2,
                help="Número mínimo de muestras para dividir un nodo."
            )

            test_size = st.slider(
                "Tamaño del Conjunto de Prueba:",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Proporción de datos que se usará para evaluar el modelo."
            )

        # Guardar parámetros en el estado de la sesión
        st.session_state.criterion = criterion
        st.session_state.max_depth = max_depth
        st.session_state.min_samples_split = min_samples_split
        st.session_state.test_size = test_size

        # Botón para entrenar el modelo
        train_button = st.button("Entrenar Modelo", type="primary")

        if train_button:
            with st.spinner("Entrenando modelo..."):
                try:
                    # Cargar y preprocesar datos
                    X, y, feature_names, class_names, dataset_info, task_type = load_data(
                        st.session_state.selected_dataset)
                    X_train, X_test, y_train, y_test = preprocess_data(
                        X, y, test_size=test_size)

                    # Entrenar el modelo
                    model_results = train_decision_tree(
                        X_train, y_train,
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        tree_type=tree_type
                    )

                    # Extraer el modelo del diccionario de resultados
                    tree_model = model_results["model"]

                    # Guardar en el estado de la sesión
                    st.session_state.tree_model = tree_model
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = feature_names
                    st.session_state.class_names = class_names
                    st.session_state.dataset_info = dataset_info
                    st.session_state.is_trained = True

                    # Evaluar el modelo
                    if tree_type == "Clasificación":
                        # Obtener predicciones del modelo
                        y_pred = tree_model.predict(X_test)
                        st.session_state.test_results = evaluate_classification_model(
                            y_test, y_pred, class_names
                        )
                    else:
                        # Obtener predicciones del modelo
                        y_pred = tree_model.predict(X_test)
                        st.session_state.test_results = evaluate_regression_model(
                            y_test, y_pred
                        )

                    st.success(
                        "¡Modelo entrenado con éxito! Ahora puedes explorar las otras pestañas.")

                    # Sugerir ir a la pestaña de visualización
                    st.info(
                        "👉 Ve a la pestaña '🌲 Visualización' para ver el árbol generado.")

                except Exception as e:
                    st.error(f"Error al entrenar el modelo: {str(e)}")

    # Pestaña de Evaluación
    elif st.session_state.active_tab == 2:
        st.header("Evaluación del Modelo")

        if not st.session_state.get('is_trained', False):
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'.")
        else:
            # Obtener predicciones del modelo
            y_pred = st.session_state.tree_model.predict(
                st.session_state.X_test)

            # Mostrar evaluación detallada del modelo
            show_detailed_evaluation(
                st.session_state.y_test,
                y_pred,
                st.session_state.class_names if st.session_state.get(
                    'tree_type', 'Clasificación') == "Clasificación" else None,
                st.session_state.get('tree_type', 'Clasificación')
            )

            # Decisión boundary si es clasificación y tiene 2 características
            if st.session_state.get('tree_type', 'Clasificación') == "Clasificación" and st.session_state.X_train.shape[1] <= 2:
                st.markdown("### Frontera de Decisión")

                if st.session_state.X_train.shape[1] == 1:
                    # Caso especial: solo una característica - agregar una segunda artificial
                    if hasattr(st.session_state.X_train, 'values'):
                        # DataFrame
                        X_values = st.session_state.X_train.values
                    else:
                        # numpy array
                        X_values = st.session_state.X_train
                    X_plot = np.column_stack(
                        [X_values, np.zeros_like(X_values)])
                    feature_idx = [0]
                    feature_names_plot = [
                        st.session_state.feature_names[0], "Característica artificial"]
                else:
                    # Dos características - dejar seleccionar cuáles usar
                    st.markdown(
                        "Selecciona las características para visualizar la frontera de decisión:")
                    col1, col2 = st.columns(2)

                    with col1:
                        feature1 = st.selectbox(
                            "Primera característica:",
                            st.session_state.feature_names,
                            index=0
                        )

                    with col2:
                        feature2 = st.selectbox(
                            "Segunda característica:",
                            st.session_state.feature_names,
                            index=min(
                                1, len(st.session_state.feature_names) - 1)
                        )

                    feature_idx = [st.session_state.feature_names.index(feature1),
                                   st.session_state.feature_names.index(feature2)]

                    # Manejar DataFrame vs numpy array
                    if hasattr(st.session_state.X_train, 'iloc'):
                        # DataFrame
                        X_plot = st.session_state.X_train.iloc[:,
                                                               feature_idx].values
                    else:
                        # numpy array
                        X_plot = st.session_state.X_train[:, feature_idx]
                    feature_names_plot = [feature1, feature2]

                # Generar y mostrar el plot en tamaño reducido
                try:
                    fig, ax = plot_decision_boundary(
                        st.session_state.tree_model,
                        X_plot,
                        st.session_state.y_train,
                        feature_names=feature_names_plot,
                        class_names=st.session_state.class_names
                    )

                    # Mostrar en columnas para reducir el tamaño al 75%
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        # Mostrar la figura directamente
                        plt.tight_layout()
                        st.pyplot(fig, clear_figure=True,
                                  use_container_width=True)

                    # Enlace para descargar
                    st.markdown(
                        get_image_download_link(
                            fig, "frontera_decision", "📥 Descargar visualización de la frontera"),
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(
                        f"Error al mostrar la visualización de frontera de decisión: {str(e)}")
                    st.info(
                        "La frontera de decisión requiere exactamente 2 características para visualizarse.")

                # Mostrar código para generar esta visualización
                code_boundary = """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

def plot_decision_boundary(model, X, y, feature_names=None, class_names=None):
    \"\"\"
    Visualiza la frontera de decisión para un modelo con 2 características.

    Parameters:
    -----------
    model : Modelo de scikit-learn
        Modelo entrenado con método predict
    X : array-like
        Datos de características (solo se usan las primeras 2 columnas)
    y : array-like
        Etiquetas de clase
    feature_names : list, opcional
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases

    Returns:
    --------
    fig : Figura de matplotlib
    \"\"\"
    # Asegurar que solo usamos 2 características
    X_plot = X[:, :2] if X.shape[1] > 2 else X

    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # Crear objeto de visualización de frontera
    disp = DecisionBoundaryDisplay.from_estimator(
        model,
        X_plot,
        alpha=0.5,
        ax=ax,
        response_method="predict"
    )

    # Colorear los puntos según su clase
    scatter = ax.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=y,
        edgecolor="k",
        s=50
    )

    # Configurar etiquetas
    if feature_names and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("Característica 1")
        ax.set_ylabel("Característica 2")

    # Configurar leyenda
    if class_names:
        legend_labels = class_names
    else:
        legend_labels = [f"Clase {i}" for i in range(len(np.unique(y)))]

    legend = ax.legend(
        handles=scatter.legend_elements()[0],
        labels=legend_labels,
        title="Clases"
    )

    ax.add_artist(legend)
    ax.set_title("Frontera de Decisión")

    return fig

# Para usar:
# fig = plot_decision_boundary(model, X, y, feature_names, class_names)
# plt.show()
"""

                show_code_with_download(
                    code_boundary, "Código para generar la frontera de decisión",
                    "frontera_decision.py"
                )

    # Pestaña de Visualización
    elif st.session_state.active_tab == 3:
        st.header("Visualización del Árbol")

        if not st.session_state.get('is_trained', False):
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'.")
        else:
            # Configuración de la visualización
            st.markdown("### Tipo de visualización")

            # Usar botones para seleccionar el tipo de visualización
            if "viz_type" not in st.session_state:
                st.session_state.viz_type = "Estándar"

            # Determinar si mostrar la opción de frontera de decisión
            show_boundary = (st.session_state.get('tree_type', 'Clasificación') == "Clasificación"
                             and len(st.session_state.get('feature_names', [])) >= 2)

            if show_boundary:
                viz_col1, viz_col2, viz_col3 = st.columns(3)
            else:
                viz_col1, viz_col2 = st.columns(2)
                viz_col3 = None

            with viz_col1:
                if st.button("📊 Estándar",
                             key="viz_standard",
                             type="primary" if st.session_state.viz_type == "Estándar" else "secondary",
                             use_container_width=True):
                    st.session_state.viz_type = "Estándar"
                    st.rerun()

            with viz_col2:
                if st.button("📝 Texto",
                             key="viz_text",
                             type="primary" if st.session_state.viz_type == "Texto" else "secondary",
                             use_container_width=True):
                    st.session_state.viz_type = "Texto"
                    st.rerun()

            if show_boundary and viz_col3:
                with viz_col3:
                    if st.button("🌈 Frontera",
                                 key="viz_boundary",
                                 type="primary" if st.session_state.viz_type == "Frontera" else "secondary",
                                 use_container_width=True):
                        st.session_state.viz_type = "Frontera"
                        st.rerun()

            viz_type = st.session_state.viz_type

            # Opciones de tamaño para la visualización
            col1, col2 = st.columns(2)

            with col1:
                fig_width = st.slider("Ancho de figura:", 8, 20, 14)

            with col2:
                fig_height = st.slider("Alto de figura:", 6, 15, 10)

            # Mostrar la visualización según el tipo seleccionado
            if viz_type == "Estándar":
                # Visualización estándar de scikit-learn
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                from sklearn.tree import plot_tree

                plot_tree(
                    st.session_state.tree_model,
                    feature_names=st.session_state.feature_names,
                    class_names=st.session_state.class_names if st.session_state.tree_type == "Clasificación" else None,
                    filled=True,
                    rounded=True,
                    ax=ax,
                    proportion=True,
                    impurity=True
                )

                # Mostrar con tamaño reducido pero expandible
                col1, col2, col3 = st.columns([1, 4, 1])
                with col2:
                    st.pyplot(fig, use_container_width=True)

                # Enlace para descargar
                st.markdown(
                    get_image_download_link(
                        fig, "arbol_decision", "📥 Descargar visualización del árbol"),
                    unsafe_allow_html=True
                )

                # Mostrar código para generar esta visualización
                code_tree = """
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Suponiendo que ya tienes un modelo entrenado (tree_model)
# y los nombres de las características (feature_names) y clases (class_names)

fig, ax = plt.subplots(figsize=(14, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=class_names,  # Solo para clasificación
    filled=True,
    rounded=True,
    ax=ax,
    proportion=True,
    impurity=True
)

plt.tight_layout()
plt.show()

# Para guardar a un archivo:
# plt.savefig('arbol_decision.png', dpi=300, bbox_inches='tight')
"""

                show_code_with_download(
                    code_tree, "Código para visualizar el árbol", "visualizar_arbol.py")

            elif viz_type == "Texto":
                # Obtener representación de texto
                tree_text = get_tree_text(
                    st.session_state.tree_model,
                    st.session_state.feature_names,
                )

                # Mostrar en un área de texto
                st.text(tree_text)

                # Botón para descargar
                text_bytes = tree_text.encode()
                b64 = base64.b64encode(text_bytes).decode()
                href = f'<a href="data:text/plain;base64,{b64}" download="arbol_texto.txt">📥 Descargar texto del árbol</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Mostrar código para generar esta visualización
                code_text = """
from sklearn.tree import export_text

def get_tree_text(model, feature_names, show_class_name=True):
    \"\"\"
    Obtiene una representación de texto de un árbol de decisión.

    Parameters:
    -----------
    model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol entrenado
    feature_names : list
        Nombres de las características
    show_class_name : bool
        Si es True, muestra los nombres de las clases (para clasificación)

    Returns:
    --------
    str
        Representación de texto del árbol
    \"\"\"
    return export_text(
        model,
        feature_names=feature_names,
        show_weights=True
    )

# Ejemplo de uso:
tree_text = get_tree_text(tree_model, feature_names)
print(tree_text)

# Para guardar a un archivo:
# with open('arbol_texto.txt', 'w') as f:
#     f.write(tree_text)
"""

                show_code_with_download(
                    code_text, "Código para obtener el texto del árbol", "texto_arbol.py")

            elif viz_type == "Frontera":
                # Visualización de frontera de decisión
                st.markdown("### Visualización de Frontera de Decisión")

                st.info("""
                **Cómo interpretar esta visualización:**
                - Las áreas coloreadas muestran las regiones de decisión para cada clase
                - Los puntos representan las muestras de entrenamiento
                - Las líneas entre colores son las fronteras de decisión
                - Solo se muestran las primeras dos características para crear la visualización 2D
                """)

                # Selección de características para la visualización
                if len(st.session_state.feature_names) > 2:
                    cols = st.columns(2)
                    with cols[0]:
                        feature1 = st.selectbox(
                            "Primera característica:",
                            st.session_state.feature_names,
                            index=0,
                            key="feature1_boundary_viz"
                        )
                    with cols[1]:
                        feature2 = st.selectbox(
                            "Segunda característica:",
                            st.session_state.feature_names,
                            index=1,
                            key="feature2_boundary_viz"
                        )

                    # Obtener índices de las características seleccionadas
                    feature_names_list = list(st.session_state.feature_names)
                    f1_idx = feature_names_list.index(feature1)
                    f2_idx = feature_names_list.index(feature2)

                    # Crear array con solo las dos características seleccionadas
                    # Verificar si X_train es DataFrame o numpy array
                    if hasattr(st.session_state.X_train, 'iloc'):
                        # Es un DataFrame, usar iloc para indexación posicional
                        X_boundary = st.session_state.X_train.iloc[:, [
                            f1_idx, f2_idx]].values
                    else:
                        # Es un numpy array, usar indexación normal
                        X_boundary = st.session_state.X_train[:, [
                            f1_idx, f2_idx]]
                    feature_names_boundary = [feature1, feature2]
                else:
                    # Si solo hay dos características, usarlas directamente
                    if hasattr(st.session_state.X_train, 'values'):
                        # Es un DataFrame, convertir a numpy array
                        X_boundary = st.session_state.X_train.values
                    else:
                        # Es un numpy array
                        X_boundary = st.session_state.X_train
                    feature_names_boundary = st.session_state.feature_names

                # Crear figura y dibujar frontera de decisión
                try:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    plot_decision_boundary(
                        st.session_state.tree_model,
                        X_boundary,
                        st.session_state.y_train,
                        ax=ax,
                        feature_names=feature_names_boundary,
                        class_names=st.session_state.class_names,
                        show_code=False
                    )

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
                    
                    Para crear esta visualización 2D, se entrena un nuevo árbol utilizando solo las dos características 
                    seleccionadas, por lo que puede diferir ligeramente del modelo completo que utiliza todas las características.
                    """)

                    # Advertencia sobre dimensionalidad
                    if len(st.session_state.feature_names) > 2:
                        st.warning("""
                        ⚠️ Esta visualización solo muestra 2 características seleccionadas. El modelo real utiliza todas 
                        las características para hacer predicciones. Las fronteras pueden variar si se seleccionan 
                        diferentes pares de características.
                        """)

                    # Mostrar código para generar esta visualización
                    code_boundary = f"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from decision_boundary import plot_decision_boundary

# Datos de entrenamiento (solo las primeras 2 características)
X_2d = X_train[:, [0, 1]]  # Usar las características seleccionadas
y_train = y_train

# Crear figura
fig, ax = plt.subplots(figsize=({fig_width}, {fig_height}))

# Visualizar frontera de decisión
plot_decision_boundary(
    tree_model,
    X_2d,
    y_train,
    ax=ax,
    feature_names={feature_names_boundary},
    class_names={st.session_state.class_names if st.session_state.class_names else None}
)

plt.tight_layout()
plt.show()

# Para guardar a un archivo:
# plt.savefig('frontera_decision.png', dpi=300, bbox_inches='tight')
"""

                    show_code_with_download(
                        code_boundary, "Código para generar la frontera de decisión", "frontera_decision.py")

                except Exception as e:
                    st.error(
                        f"Error al mostrar la visualización de frontera de decisión: {str(e)}")
                    st.info("""
                    La frontera de decisión requiere:
                    - Un modelo de clasificación entrenado
                    - Exactamente 2 características para visualizar
                    - Datos de entrenamiento válidos
                    """)
                    st.exception(
                        e)  # Mostrar detalles del error para debugging

    # Pestaña de Características
    elif st.session_state.active_tab == 4:
        st.header("Importancia de Características")

        if not st.session_state.get('is_trained', False):
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'.")
        else:
            # Mostrar importancia de características
            display_feature_importance(
                st.session_state.tree_model,
                st.session_state.feature_names
            )

    # Pestaña de Predicciones
    elif st.session_state.active_tab == 5:
        st.header("Predicciones con Nuevos Datos")

        if not st.session_state.get('is_trained', False):
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'.")
        else:
            # Interfaz para hacer predicciones
            create_prediction_interface(
                st.session_state.tree_model,
                st.session_state.feature_names,
                st.session_state.class_names,
                st.session_state.get('tree_type', 'Clasificación'),
                # Pasar datos de entrenamiento para rangos dinámicos
                st.session_state.get('X_train', None),
                # Pasar nombre del dataset para metadata
                st.session_state.get('selected_dataset', 'Titanic')
            )

    # Pestaña de Exportar Modelo
    elif st.session_state.active_tab == 6:
        st.header("Exportar Modelo")

        if not st.session_state.get('is_trained', False):
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'.")
        else:
            # Opciones para exportar el modelo
            display_model_export_options(
                st.session_state.tree_model,
                st.session_state.feature_names,
                st.session_state.class_names,
                st.session_state.get('tree_type', 'Clasificación'),
                st.session_state.get('max_depth', 3),
                st.session_state.get('min_samples_split', 2),
                st.session_state.get('criterion', 'gini')
            )


def run_linear_regression_app():
    """Ejecuta la aplicación específica de regresión (lineal y logística)."""
    st.header("📊 Regresión")
    st.markdown(
        "Aprende sobre regresión lineal y logística de forma interactiva")

    # Información sobre regresión
    with st.expander("ℹ️ ¿Qué es la Regresión?", expanded=False):
        st.markdown("""
        La regresión es un conjunto de algoritmos de aprendizaje supervisado utilizados para predecir valores numéricos continuos (regresión lineal) o clasificar elementos en categorías (regresión logística).

        **Características principales:**
        - **Regresión Lineal**: Establece una relación lineal entre las variables independientes (características) y la variable dependiente (objetivo)
        - **Regresión Logística**: Utiliza la función logística para modelar la probabilidad de pertenencia a una clase
        - Ambos tipos minimizan funciones de costo específicas para encontrar los mejores parámetros
        - Son interpretables: los coeficientes indican la importancia y dirección del efecto de cada característica
        - No requieren escalado de datos, aunque puede mejorar la convergencia

        **Tipos de regresión:**
        - **Regresión Lineal Simple**: Una sola característica predictora
        - **Regresión Lineal Múltiple**: Múltiples características predictoras
        - **Regresión Logística**: Para problemas de clasificación binaria o multiclase

        **Limitaciones:**
        - Asume una relación lineal entre variables (regresión lineal)
        - Sensible a valores atípicos (outliers)
        - Puede sufrir de multicolinealidad cuando las características están correlacionadas
        """)

    # Variables para almacenar datos
    dataset_loaded = False
    X, y, feature_names, class_names, dataset_info, task_type = None, None, None, None, None, None

    # Inicializar el estado de la pestaña activa si no existe
    if 'active_tab_lr' not in st.session_state:
        st.session_state.active_tab_lr = 0

    # Crear pestañas para organizar la información
    tab_options = [
        "📊 Datos",
        "🏋️ Entrenamiento",
        "📈 Evaluación",
        "📉 Visualización",
        "🔍 Coeficientes",
        "🔮 Predicciones",
        "💾 Exportar Modelo"
    ]

    # Crear contenedor para los botones de las pestañas
    tab_cols = st.columns(len(tab_options))

    # Estilo CSS para los botones de pestañas (Regresión)
    st.markdown("""
    <style>
    div.tab-button-lr > button {
        border-radius: 4px 4px 0 0;
        padding: 10px;
        width: 100%;
        white-space: nowrap;
        background-color: #F0F2F6;
        border-bottom: 2px solid #E0E0E0;
        color: #333333;
    }
    div.tab-button-lr-active > button {
        background-color: #E3F2FD !important;
        border-bottom: 2px solid #1E88E5 !important;
        font-weight: bold !important;
        color: #1E88E5 !important;
    }
    div.tab-button-lr > button:hover {
        background-color: #E8EAF6;
    }
    </style>
    """, unsafe_allow_html=True)

    # Crear botones para las pestañas
    for i, (tab_name, col) in enumerate(zip(tab_options, tab_cols)):
        button_key = f"tab_lr_{i}"
        button_style = "tab-button-lr-active" if st.session_state.active_tab_lr == i else "tab-button-lr"
        is_active = st.session_state.active_tab_lr == i

        with col:
            st.markdown(f"<div class='{button_style}'>",
                        unsafe_allow_html=True)
            # Usar type="primary" para el botón activo
            if st.button(tab_name, key=button_key, use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active_tab_lr = i
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Separador visual
    st.markdown("---")

    # SELECTOR UNIFICADO DE DATASET (solo mostrar en pestañas que lo necesiten)
    if st.session_state.active_tab_lr in [0, 1]:  # Exploración y Entrenamiento
        st.markdown("### 📊 Selección de Dataset")

        # Inicializar dataset seleccionado si no existe
        if 'selected_dataset_lr' not in st.session_state:
            st.session_state.selected_dataset_lr = "💰 Propinas - Predicción de propinas"

        # Lista de datasets adecuados para regresión
        regression_datasets = [
            "💰 Propinas - Predicción de propinas",
            "🏠 Viviendas California - Precios",
            "🌸 Iris - Clasificación de flores",  # También útil para regresión logística
            "🍷 Vino - Clasificación de vinos",   # También útil para regresión logística
            "🔬 Cáncer - Diagnóstico binario",   # También útil para regresión logística
            "🚢 Titanic - Supervivencia",        # También útil para regresión logística
            # También útil para regresión logística
            "🐧 Pingüinos - Clasificación de especies"
        ]

        # Añadir datasets CSV cargados si existen
        available_datasets = regression_datasets.copy()
        if 'csv_datasets' in st.session_state:
            available_datasets.extend(st.session_state.csv_datasets.keys())

        # Asegurar que el dataset seleccionado esté en la lista disponible
        if st.session_state.selected_dataset_lr not in available_datasets:
            st.session_state.selected_dataset_lr = regression_datasets[0]

        # Selector unificado
        dataset_option = st.selectbox(
            "Dataset:",
            available_datasets,
            index=available_datasets.index(
                st.session_state.selected_dataset_lr),
            key="unified_dataset_selector_lr",
            help="El dataset seleccionado se mantendrá entre las pestañas de Exploración y Entrenamiento"
        )

        # Actualizar la variable de sesión
        st.session_state.selected_dataset_lr = dataset_option

        # Separador después del selector
        st.markdown("---")
    else:
        # Para otras pestañas, mostrar qué dataset está seleccionado actualmente
        if hasattr(st.session_state, 'selected_dataset_lr'):
            st.info(
                f"📊 **Dataset actual:** {st.session_state.selected_dataset_lr}")
            st.markdown("---")

    # Pestaña de Datos
    if st.session_state.active_tab_lr == 0:
        st.header("Exploración de Datos")

        try:
            # Cargar datos para exploración
            X, y, feature_names, class_names, dataset_info, task_type = load_data(
                st.session_state.selected_dataset_lr)

            # Convertir a DataFrames para facilitar el manejo
            if isinstance(X, pd.DataFrame):
                X_df = X.copy()
                column_mapping = {}
                if len(feature_names) == len(X_df.columns):
                    column_mapping = dict(zip(X_df.columns, feature_names))
            else:
                X_df = pd.DataFrame(X, columns=feature_names)
                column_mapping = {}

            y_df = pd.Series(y, name="target")

            # Para regresión logística, mapear nombres de clases si están disponibles
            if task_type == "Clasificación" and class_names is not None:
                y_df = y_df.map(
                    {i: name for i, name in enumerate(class_names)})

            df = pd.concat([X_df, y_df], axis=1)

            # Renombrar columnas para mostrar nombres amigables si existe el mapeo
            if column_mapping:
                df_display = df.rename(columns=column_mapping)
            else:
                df_display = df

            # Mostrar información del dataset
            st.markdown("### Información del Dataset")
            st.markdown(create_info_box(dataset_info), unsafe_allow_html=True)

            # Mostrar las primeras filas de los datos
            st.markdown("### Vista previa de datos")
            st.dataframe(df_display.head(10))

            # Estadísticas descriptivas
            st.markdown("### Estadísticas Descriptivas")
            st.dataframe(df_display.describe())

            # Distribución de clases o valores objetivo
            st.markdown("### Distribución del Objetivo")

            fig, ax = plt.subplots(figsize=(10, 6))

            if task_type == "Clasificación":
                # Gráfico de barras para clasificación (regresión logística)
                value_counts = y_df.value_counts().sort_index()
                try:
                    import seaborn as sns
                    sns.barplot(x=value_counts.index,
                                y=value_counts.values, ax=ax)
                except ImportError:
                    # Fallback to matplotlib if seaborn is not available
                    ax.bar(value_counts.index, value_counts.values)
                ax.set_title("Distribución de Clases")
                ax.set_xlabel("Clase")
                ax.set_ylabel("Cantidad")

                # Rotar etiquetas si son muchas
                if len(value_counts) > 3:
                    plt.xticks(rotation=45, ha='right')
            else:
                # Histograma para regresión
                # Convertir a numérico en caso de que sean strings
                y_numeric = pd.to_numeric(y_df, errors='coerce')
                # Usar matplotlib directamente para evitar problemas de tipo
                ax.hist(y_numeric.dropna(), bins=30,
                        alpha=0.7, edgecolor='black')
                ax.set_title("Distribución de Valores Objetivo")
                ax.set_xlabel("Valor")
                ax.set_ylabel("Frecuencia")

            # Mostrar la figura
            col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
            with col2:
                st.pyplot(fig, use_container_width=True)

            # Análisis de correlación
            st.markdown("### Matriz de Correlación")

            # Matriz de correlación
            corr = X_df.corr()

            # Generar máscara para el triángulo superior
            mask = np.triu(np.ones_like(corr, dtype=bool))

            # Generar mapa de calor
            fig_corr, ax = plt.subplots(figsize=(10, 8))
            try:
                import seaborn as sns
                sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                            square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
            except ImportError:
                # Fallback to matplotlib if seaborn is not available
                im = ax.imshow(corr.where(~mask),
                               cmap="coolwarm", aspect="auto")
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha='right')
                ax.set_yticklabels(corr.columns)
                # Add text annotations
                for i in range(len(corr.columns)):
                    for j in range(len(corr.columns)):
                        if not mask[i, j]:
                            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                           ha="center", va="center", color="black")
            ax.set_title("Matriz de Correlación de Características")

            # Mostrar la figura
            col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
            with col2:
                st.pyplot(fig_corr, use_container_width=True)

            # Matriz de dispersión (Scatterplot Matrix)
            st.markdown("### Matriz de Dispersión (Pairplot)")

            # Opciones de visualización
            st.markdown("#### Opciones de visualización")
            col1, col2 = st.columns(2)

            with col1:
                # Seleccionar tipo de gráfico para la diagonal
                diag_kind = st.radio(
                    "Tipo de gráfico en la diagonal:",
                    ["Histograma", "KDE (Estimación de Densidad)"],
                    index=1,
                    horizontal=True
                )
                diag_kind = "hist" if diag_kind == "Histograma" else "kde"

            with col2:
                # Seleccionar número máximo de características
                max_features_selected = st.slider(
                    "Número máximo de características:",
                    min_value=2,
                    max_value=min(6, len(X_df.columns)),
                    value=min(4, len(X_df.columns)),
                    help="Un número mayor de características puede hacer que el gráfico sea más difícil de interpretar."
                )

            # Permitir al usuario seleccionar las características específicas
            st.markdown("#### Selecciona las características para visualizar")

            # Usar nombres amigables si están disponibles
            if column_mapping:
                available_features = list(column_mapping.values())
                display_to_original = {v: k for k, v in column_mapping.items()}
            else:
                available_features = X_df.columns.tolist()
                display_to_original = {}

            # Usar multiselect para seleccionar características
            selected_features = st.multiselect(
                "Características a incluir en la matriz de dispersión:",
                available_features,
                default=available_features[:max_features_selected],
                max_selections=max_features_selected,
                help=f"Selecciona hasta {max_features_selected} características para incluir en la visualización."
            )

            # Si no se seleccionó ninguna característica, usar las primeras por defecto
            if not selected_features:
                selected_features = available_features[:max_features_selected]
                st.info(
                    f"No se seleccionaron características. Usando las primeras {max_features_selected} por defecto.")

            # Convertir nombres amigables a nombres originales si es necesario
            if column_mapping:
                original_features = [display_to_original[feat]
                                     for feat in selected_features]
            else:
                original_features = selected_features

            # Crear el dataframe para la visualización
            plot_df = X_df[original_features].copy()
            # Renombrar a nombres amigables para visualización
            if column_mapping:
                plot_df = plot_df.rename(columns=column_mapping)
            # Añadir la variable objetivo para colorear
            plot_df['target'] = y_df

            # Generar el pairplot
            with st.spinner("Generando matriz de dispersión..."):
                try:
                    import seaborn as sns
                    pair_plot = sns.pairplot(
                        plot_df,
                        hue='target' if task_type == "Clasificación" else None,
                        diag_kind=diag_kind,
                        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
                        diag_kws={'alpha': 0.5},
                        height=2.0
                    )
                    pair_plot.fig.suptitle(
                        "Matriz de Dispersión de Características", y=1.02, fontsize=14)

                    # Mostrar la figura
                    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                    with col2:
                        st.pyplot(pair_plot.fig, use_container_width=True)

                    # Enlace para descargar
                    st.markdown(
                        get_image_download_link(
                            pair_plot.fig, "matriz_dispersion_lr", "📥 Descargar matriz de dispersión"),
                        unsafe_allow_html=True
                    )
                except ImportError:
                    st.error(
                        "Seaborn no está disponible. Por favor, instala seaborn para usar la matriz de dispersión.")
                    st.info(
                        "Puedes instalar seaborn ejecutando: pip install seaborn")

        except Exception as e:
            st.error(f"Error al cargar el dataset: {str(e)}")
            st.info(
                "Por favor, selecciona un dataset válido para continuar con la exploración.")

    # Pestaña de Entrenamiento
    elif st.session_state.active_tab_lr == 1:
        st.header("Configuración del Modelo")

        # Inicializar session state variables
        if 'dataset_option_lr' not in st.session_state:
            st.session_state.dataset_option_lr = st.session_state.selected_dataset_lr
        if 'model_type_lr' not in st.session_state:
            st.session_state.model_type_lr = "Linear"
        if 'is_trained_lr' not in st.session_state:
            st.session_state.is_trained_lr = False

        # Cargar datos para la vista previa si cambia el dataset o si no se ha cargado
        if st.session_state.selected_dataset_lr != st.session_state.dataset_option_lr or not dataset_loaded:
            try:
                X, y, feature_names, class_names, dataset_info, task_type = load_data(
                    st.session_state.selected_dataset_lr)

                st.session_state.dataset_option_lr = st.session_state.selected_dataset_lr
                dataset_loaded = True

                # Mostrar información del dataset
                st.markdown("### Información del Dataset")
                st.markdown(create_info_box(dataset_info),
                            unsafe_allow_html=True)

                # Determinar el tipo de modelo según el task_type detectado
                st.markdown("### Tipo de Modelo Lineal")

                # Usar botones para seleccionar el tipo de modelo
                tipo_col1, tipo_col2 = st.columns(2)

                with tipo_col1:
                    is_linear = True
                    if "model_type_lr" in st.session_state:
                        is_linear = st.session_state.model_type_lr == "Linear"

                    if st.button("📈 Regresión Lineal",
                                 key="btn_linear",
                                 type="primary" if is_linear else "secondary",
                                 use_container_width=True,
                                 help="Para predecir valores numéricos continuos"):
                        model_type = "Linear"
                        st.session_state.model_type_lr = model_type
                        st.rerun()

                with tipo_col2:
                    is_logistic = False
                    if "model_type_lr" in st.session_state:
                        is_logistic = st.session_state.model_type_lr == "Logistic"

                    if st.button("🏷️ Regresión Logística",
                                 key="btn_logistic",
                                 type="primary" if is_logistic else "secondary",
                                 use_container_width=True,
                                 help="Para predecir categorías o clases"):
                        model_type = "Logistic"
                        st.session_state.model_type_lr = model_type
                        st.rerun()

                # Obtener el valor actual del tipo de modelo
                model_type = st.session_state.get('model_type_lr', "Linear")

                # Mostrar sugerencia basada en el tipo de tarea detectado
                if task_type == "Clasificación" and model_type == "Linear":
                    st.warning(
                        "Este dataset parece ser más adecuado para Regresión Logística. La selección actual podría no ofrecer resultados óptimos.")
                elif task_type == "Regresión" and model_type == "Logistic":
                    st.warning(
                        "Este dataset parece ser más adecuado para Regresión Lineal. La selección actual podría no ofrecer resultados óptimos.")

            except Exception as e:
                st.error(f"Error al cargar el dataset: {str(e)}")
                dataset_loaded = False

        # Parámetros del modelo
        st.markdown("### Parámetros del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            # Para regresión logística, mostrar parámetro max_iter
            if st.session_state.get('model_type_lr', 'Linear') == "Logistic":
                max_iter = st.slider(
                    "Máximo de Iteraciones:",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    step=100,
                    help="Número máximo de iteraciones para el optimizador de regresión logística."
                )
            else:
                st.info(
                    "La regresión lineal no requiere configuración adicional de iteraciones.")

        with col2:
            # Configuraciones adicionales pueden ir aquí en el futuro
            st.empty()

        if st.button("🚀 Entrenar Modelo", key="train_lr_button", type="primary"):
            if dataset_loaded and X is not None and y is not None:
                with st.spinner("Entrenando modelo..."):
                    try:
                        # Entrenar el modelo
                        if model_type == "Logistic":
                            result = train_linear_model(
                                X, y,
                                model_type=model_type,
                                max_iter=max_iter,
                                test_size=0.2,
                                random_state=42
                            )
                        else:
                            result = train_linear_model(
                                X, y,
                                model_type=model_type,
                                test_size=0.2,
                                random_state=42
                            )

                        # Extraer el modelo y métricas del resultado
                        model = result["model"]
                        metrics = result["test_results"]

                        # Guardar en session state
                        st.session_state.model_lr = model
                        st.session_state.metrics_lr = metrics
                        st.session_state.X_train_lr = result["X_train"]
                        st.session_state.X_test_lr = result["X_test"]
                        st.session_state.y_train_lr = result["y_train"]
                        st.session_state.y_test_lr = result["y_test"]
                        st.session_state.feature_names_lr = feature_names
                        st.session_state.class_names_lr = class_names
                        st.session_state.task_type_lr = task_type
                        st.session_state.model_trained_lr = True

                        st.success("¡Modelo entrenado exitosamente!")

                    except Exception as e:
                        st.error(f"Error al entrenar el modelo: {str(e)}")
            else:
                st.error("Por favor, carga un dataset válido primero.")

    # Pestaña de Evaluación
    elif st.session_state.active_tab_lr == 2:
        st.header("Evaluación del Modelo")

        if st.session_state.get('model_trained_lr', False):
            metrics = st.session_state.get('metrics_lr', {})
            model_type = st.session_state.get('model_type_lr', 'Linear')

            # Análisis de balance de clases para regresión logística
            if model_type == "Logistic":
                y_test = st.session_state.get('y_test_lr')
                if y_test is not None:
                    # Verificar balance de clases
                    class_counts = pd.Series(y_test).value_counts()
                    total_samples = len(y_test)

                    st.markdown("### ⚖️ Análisis de Balance de Clases")

                    col1, col2 = st.columns(2)
                    with col1:
                        # Mostrar distribución de clases
                        for class_val, count in class_counts.items():
                            percentage = (count / total_samples) * 100
                            class_name = st.session_state.get(
                                'class_names_lr', [])
                            display_name = class_name[int(class_val)] if class_name and int(
                                class_val) < len(class_name) else f"Clase {class_val}"
                            st.metric(f"{display_name}",
                                      f"{count} ({percentage:.1f}%)")

                    with col2:
                        # Evaluar balance
                        min_class_ratio = class_counts.min() / class_counts.max()
                        if min_class_ratio < 0.1:
                            st.error(
                                "❌ **Clases muy desbalanceadas** (ratio < 10%)")
                            st.markdown("**Recomendaciones:**")
                            st.markdown(
                                "• Considera técnicas de balanceo (SMOTE, undersampling)")
                            st.markdown(
                                "• Usa métricas como F1-Score en lugar de Accuracy")
                            st.markdown(
                                "• Ajusta los pesos de las clases en el modelo")
                        elif min_class_ratio < 0.3:
                            st.warning(
                                "⚠️ **Clases moderadamente desbalanceadas**")
                            st.markdown(
                                "• Presta especial atención a Precision y Recall")
                            st.markdown(
                                "• Considera la curva Precision-Recall")
                        else:
                            st.success(
                                "✅ **Clases relativamente balanceadas**")
                            st.markdown("• Accuracy es una métrica confiable")
                            st.markdown(
                                "• Todas las métricas son representativas")

            # Información sobre las métricas
            with st.expander("ℹ️ ¿Qué significan estas métricas?", expanded=False):
                if model_type == "Linear":
                    st.markdown("""
                    **Métricas de Regresión Lineal:**
                    
                    **R² Score (Coeficiente de Determinación):**
                    - Mide qué tan bien el modelo explica la variabilidad de los datos
                    - Rango: 0 a 1 (valores negativos indican un modelo muy malo)
                    - **Interpretación:**
                      - R² = 1.0: El modelo explica perfectamente toda la variabilidad
                      - R² = 0.8: El modelo explica el 80% de la variabilidad (muy bueno)
                      - R² = 0.5: El modelo explica el 50% de la variabilidad (moderado)
                      - R² = 0.0: El modelo no explica nada de la variabilidad
                    
                    **MAE (Error Absoluto Medio):**
                    - Promedio de las diferencias absolutas entre valores reales y predichos
                    - Se expresa en las mismas unidades que la variable objetivo
                    - **Interpretación:** Valores más bajos = mejor modelo
                    
                    **RMSE (Raíz del Error Cuadrático Medio):**
                    - Similar al MAE pero penaliza más los errores grandes
                    - Se expresa en las mismas unidades que la variable objetivo
                    - **Interpretación:** Valores más bajos = mejor modelo
                    """)
                else:
                    st.markdown("""
                    **Métricas de Regresión Logística:**
                    
                    **Accuracy (Exactitud):**
                    - Porcentaje de predicciones correctas del total
                    - Rango: 0 a 1 (0% a 100%)
                    - **Interpretación:** Valores más altos = mejor modelo
                    - **Cuidado:** Puede ser engañosa con clases desbalanceadas
                    
                    **Precision (Precisión):**
                    - De todas las predicciones positivas, cuántas fueron correctas
                    - Fórmula: VP / (VP + FP)
                    - Importante cuando los falsos positivos son costosos
                    - **Interpretación:** Valores más altos = mejor modelo
                    
                    **Recall (Sensibilidad o Exhaustividad):**
                    - De todos los casos positivos reales, cuántos detectó el modelo
                    - Fórmula: VP / (VP + FN)
                    - Importante cuando los falsos negativos son costosos
                    - **Interpretación:** Valores más altos = mejor modelo
                    
                    **F1-Score:**
                    - Media armónica entre precisión y recall
                    - Fórmula: 2 × (Precisión × Recall) / (Precisión + Recall)
                    - Útil cuando necesitas balance entre precisión y recall
                    - **Interpretación:** Valores más altos = mejor balance
                    
                    **Curva ROC:**
                    - Muestra el rendimiento en diferentes umbrales de decisión
                    - AUC (Área bajo la curva): 0.5 = aleatorio, 1.0 = perfecto
                    
                    **Curva Precision-Recall:**
                    - Especialmente útil para clases desbalanceadas
                    - Muestra el trade-off entre precisión y recall
                    
                    **VP = Verdaderos Positivos, FP = Falsos Positivos, FN = Falsos Negativos**
                    """)

            st.markdown("### 📊 Resultados de Evaluación")

            if model_type == "Linear":
                col1, col2, col3 = st.columns(3)
                with col1:
                    r2_value = metrics.get('r2', 0)
                    st.metric("R² Score", f"{r2_value:.4f}")
                    # Indicador de calidad
                    if r2_value >= 0.8:
                        st.success("🎯 Excelente ajuste")
                    elif r2_value >= 0.6:
                        st.info("👍 Buen ajuste")
                    elif r2_value >= 0.4:
                        st.warning("⚠️ Ajuste moderado")
                    else:
                        st.error("❌ Ajuste pobre")

                with col2:
                    mae_value = metrics.get('mae', 0)
                    st.metric("MAE", f"{mae_value:.4f}")

                with col3:
                    rmse_value = metrics.get('rmse', 0)
                    st.metric("RMSE", f"{rmse_value:.4f}")

            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    acc_value = metrics.get('accuracy', 0)
                    st.metric("Accuracy", f"{acc_value:.4f}")
                    # Indicador de calidad
                    if acc_value >= 0.9:
                        st.success("🎯 Excelente")
                    elif acc_value >= 0.8:
                        st.info("👍 Muy bueno")
                    elif acc_value >= 0.7:
                        st.warning("⚠️ Bueno")
                    else:
                        st.error("❌ Necesita mejora")

                with col2:
                    st.metric(
                        "Precision", f"{metrics.get('precision', 0):.4f}")

                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")

                # Añadir F1-Score si está disponible
                if 'report' in metrics and 'weighted avg' in metrics['report']:
                    f1_score = metrics['report']['weighted avg'].get(
                        'f1-score', 0)
                    st.markdown("### 🎯 Métricas Adicionales")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("F1-Score", f"{f1_score:.4f}")
                        if f1_score >= 0.8:
                            st.success("🎯 Excelente balance")
                        elif f1_score >= 0.7:
                            st.info("👍 Buen balance")
                        else:
                            st.warning("⚠️ Balance mejorable")

                    # Mostrar métricas por clase si es clasificación multiclase
                    if 'report' in metrics:
                        class_names = st.session_state.get(
                            'class_names_lr', [])
                        if class_names and len(class_names) > 2:
                            st.markdown("### 📊 Métricas por Clase")

                            report_data = []
                            for class_name in class_names:
                                if class_name in metrics['report']:
                                    class_metrics = metrics['report'][class_name]
                                    report_data.append({
                                        'Clase': class_name,
                                        'Precision': f"{class_metrics.get('precision', 0):.3f}",
                                        'Recall': f"{class_metrics.get('recall', 0):.3f}",
                                        'F1-Score': f"{class_metrics.get('f1-score', 0):.3f}",
                                        'Soporte': int(class_metrics.get('support', 0))
                                    })

                            if report_data:
                                df_report = pd.DataFrame(report_data)
                                st.dataframe(
                                    df_report, use_container_width=True, hide_index=True)

            # Mostrar interpretación contextual
            st.markdown("### 🎯 Interpretación de Resultados")

            if model_type == "Linear":
                r2_value = metrics.get('r2', 0)
                mae_value = metrics.get('mae', 0)
                rmse_value = metrics.get('rmse', 0)

                interpretation = f"""
                **Resumen del Modelo:**
                - Tu modelo de regresión lineal explica **{r2_value*100:.1f}%** de la variabilidad en los datos
                - En promedio, las predicciones se desvían **{mae_value:.2f} unidades** del valor real (MAE)
                - La raíz del error cuadrático medio es **{rmse_value:.2f} unidades** (RMSE)
                """

                if rmse_value > mae_value * 1.5:
                    interpretation += "\n- ⚠️ El RMSE es significativamente mayor que el MAE, lo que indica la presencia de algunos errores grandes"

                st.info(interpretation)

            else:
                acc_value = metrics.get('accuracy', 0)
                prec_value = metrics.get('precision', 0)
                rec_value = metrics.get('recall', 0)
                f1_value = 0

                # Obtener F1-score si está disponible
                if 'report' in metrics and 'weighted avg' in metrics['report']:
                    f1_value = metrics['report']['weighted avg'].get(
                        'f1-score', 0)

                interpretation = f"""
                **Resumen del Modelo:**
                - Tu modelo clasifica correctamente **{acc_value*100:.1f}%** de los casos
                - De las predicciones positivas, **{prec_value*100:.1f}%** son correctas (Precisión)
                - Detecta **{rec_value*100:.1f}%** de todos los casos positivos reales (Recall)
                """

                if f1_value > 0:
                    interpretation += f"\n- El F1-Score (balance entre precisión y recall) es **{f1_value:.3f}**"

                # Análisis de balance entre precisión y recall
                if abs(prec_value - rec_value) > 0.1:
                    if prec_value > rec_value:
                        interpretation += "\n\n⚖️ **Balance:** El modelo es más preciso pero menos sensible (más conservador)"
                        interpretation += "\n💡 **Sugerencia:** Si es importante detectar todos los casos positivos, considera ajustar el umbral de decisión"
                    else:
                        interpretation += "\n\n⚖️ **Balance:** El modelo es más sensible pero menos preciso (más liberal)"
                        interpretation += "\n💡 **Sugerencia:** Si es importante evitar falsos positivos, considera ajustar el umbral de decisión"
                else:
                    interpretation += "\n\n⚖️ **Balance:** Bueno equilibrio entre precisión y recall"

                # Análisis específico del accuracy
                if acc_value < 0.6:
                    interpretation += "\n\n🔍 **Recomendaciones para mejorar:**"
                    interpretation += "\n• Revisar la calidad y cantidad de datos de entrenamiento"
                    interpretation += "\n• Considerar ingeniería de características adicionales"
                    interpretation += "\n• Probar diferentes algoritmos de clasificación"
                    interpretation += "\n• Verificar si hay desbalance de clases"

                st.info(interpretation)

        else:
            st.info("Entrena un modelo primero para ver las métricas de evaluación.")

    # Pestaña de Visualización
    elif st.session_state.active_tab_lr == 3:
        st.header("Visualizaciones")

        if st.session_state.get('model_trained_lr', False):
            model_type = st.session_state.get('model_type_lr', 'Linear')
            X_test = st.session_state.get('X_test_lr')
            y_test = st.session_state.get('y_test_lr')
            X_train = st.session_state.get('X_train_lr')
            y_train = st.session_state.get('y_train_lr')
            model = st.session_state.get('model_lr')

            # Información sobre las visualizaciones
            with st.expander("ℹ️ ¿Cómo interpretar estas visualizaciones?", expanded=False):
                if model_type == "Linear":
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
                else:
                    st.markdown("""
                    **Matriz de Confusión:**
                    - Muestra predicciones correctas e incorrectas por clase
                    - **Interpretación:**
                      - Diagonal principal = predicciones correctas
                      - Fuera de la diagonal = errores del modelo
                      - Colores más intensos = mayor cantidad de casos
                    
                    **Curva ROC (si es binaria):**
                    - Muestra el rendimiento del clasificador en diferentes umbrales
                    - **Interpretación:**
                      - Línea más cerca de la esquina superior izquierda = mejor modelo
                      - Área bajo la curva (AUC) cercana a 1 = excelente modelo
                    """)

            if model_type == "Linear" and X_test is not None and y_test is not None and model is not None:
                y_pred = model.predict(X_test)

                # Crear visualizaciones con mejor tamaño
                st.markdown("### 📊 Gráfico de Predicciones vs Valores Reales")

                fig, ax = plt.subplots(figsize=(12, 8))

                # Scatter plot con mejor estilo
                ax.scatter(y_test, y_pred, alpha=0.6, s=50,
                           edgecolors='black', linewidth=0.5)

                # Línea de predicción perfecta
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
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

                # Gráfico de residuos
                st.markdown("### 📈 Análisis de Residuos")

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

                residuals = y_test - y_pred

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

            elif model_type == "Logistic" and X_test is not None and y_test is not None and model is not None:
                from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

                # Matriz de Confusión
                st.markdown("### 📊 Matriz de Confusión")

                cm = confusion_matrix(y_test, y_pred)
                class_names = st.session_state.get('class_names_lr', [])

                fig, ax = plt.subplots(figsize=(10, 8))

                # Crear mapa de calor
                try:
                    import seaborn as sns
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                ax=ax, cbar_kws={'shrink': 0.8},
                                xticklabels=class_names if class_names else True,
                                yticklabels=class_names if class_names else True)
                except ImportError:
                    # Fallback to matplotlib if seaborn is not available
                    im = ax.imshow(cm, cmap='Blues', aspect='auto')
                    # Add text annotations
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            text = ax.text(j, i, f'{cm[i, j]}',
                                           ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
                    ax.set_xticks(range(cm.shape[1]))
                    ax.set_yticks(range(cm.shape[0]))
                    if class_names:
                        ax.set_xticklabels(class_names)
                        ax.set_yticklabels(class_names)
                ax.set_title('Matriz de Confusión',
                             fontsize=14, fontweight='bold')
                ax.set_xlabel('Predicciones', fontsize=12)
                ax.set_ylabel('Valores Reales', fontsize=12)

                # Mostrar con 80% del ancho
                col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                with col2:
                    st.pyplot(fig, use_container_width=True)

                # Análisis detallado de la matriz de confusión
                if len(np.unique(y_test)) == 2:
                    tn, fp, fn, tp = cm.ravel()

                    # Métricas derivadas de la matriz de confusión
                    st.markdown("### 🔍 Análisis Detallado de la Matriz")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Verdaderos Positivos", f"{tp}")
                        st.caption(
                            "Casos positivos correctamente identificados")
                    with col2:
                        st.metric("Falsos Positivos", f"{fp}")
                        st.caption(
                            "Casos negativos clasificados como positivos")
                    with col3:
                        st.metric("Falsos Negativos", f"{fn}")
                        st.caption(
                            "Casos positivos clasificados como negativos")
                    with col4:
                        st.metric("Verdaderos Negativos", f"{tn}")
                        st.caption(
                            "Casos negativos correctamente identificados")

                    # Interpretación de errores
                    if fp > fn:
                        st.warning(
                            "⚠️ El modelo tiende a clasificar más casos como positivos (más falsos positivos que falsos negativos)")
                    elif fn > fp:
                        st.warning(
                            "⚠️ El modelo tiende a ser más conservador (más falsos negativos que falsos positivos)")
                    else:
                        st.success(
                            "✅ El modelo tiene un balance equilibrado entre falsos positivos y negativos")

                # Curvas ROC y Precision-Recall
                if len(np.unique(y_test)) == 2:
                    from sklearn.metrics import roc_curve, auc, average_precision_score

                    # Crear subplots para ROC y Precision-Recall
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                    # Curva ROC
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)

                    ax1.plot(fpr, tpr, color='darkorange', lw=2,
                             label=f'Curva ROC (AUC = {roc_auc:.3f})')
                    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                             label='Clasificador Aleatorio')

                    ax1.set_xlim([0.0, 1.0])
                    ax1.set_ylim([0.0, 1.05])
                    ax1.set_xlabel('Tasa de Falsos Positivos', fontsize=12)
                    ax1.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12)
                    ax1.set_title('Curva ROC', fontsize=14, fontweight='bold')
                    ax1.legend(loc="lower right")
                    ax1.grid(True, alpha=0.3)

                    # Curva Precision-Recall
                    precision, recall, _ = precision_recall_curve(
                        y_test, y_pred_proba[:, 1])
                    avg_precision = average_precision_score(
                        y_test, y_pred_proba[:, 1])

                    ax2.plot(recall, precision, color='darkgreen', lw=2,
                             label=f'Curva P-R (AP = {avg_precision:.3f})')
                    ax2.axhline(y=np.sum(y_test)/len(y_test), color='navy', lw=2, linestyle='--',
                                label=f'Baseline ({np.sum(y_test)/len(y_test):.3f})')

                    ax2.set_xlim([0.0, 1.0])
                    ax2.set_ylim([0.0, 1.05])
                    ax2.set_xlabel('Recall (Sensibilidad)', fontsize=12)
                    ax2.set_ylabel('Precision (Precisión)', fontsize=12)
                    ax2.set_title('Curva Precision-Recall',
                                  fontsize=14, fontweight='bold')
                    ax2.legend(loc="lower left")
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()

                    st.markdown("### 📈 Curvas de Rendimiento")

                    # Explicación detallada sobre las curvas de rendimiento
                    with st.expander("ℹ️ ¿Cómo interpretar las Curvas de Rendimiento?", expanded=False):
                        st.markdown("""
                        **Curva ROC (Receiver Operating Characteristic)**
                        
                        **¿Qué muestra?**
                        - **Eje X:** Tasa de Falsos Positivos (FPR) = FP / (FP + TN)
                        - **Eje Y:** Tasa de Verdaderos Positivos (TPR) = TP / (TP + FN) = Sensibilidad/Recall
                        - **Línea diagonal:** Rendimiento de un clasificador aleatorio
                        - **AUC (Área Bajo la Curva):** Métrica resumen del rendimiento
                        
                        **Interpretación:**
                        - **AUC = 1.0:** Clasificador perfecto
                        - **AUC = 0.9-1.0:** Excelente discriminación
                        - **AUC = 0.8-0.9:** Buena discriminación  
                        - **AUC = 0.7-0.8:** Discriminación aceptable
                        - **AUC = 0.5:** Equivalente a adivinar al azar
                        - **AUC < 0.5:** Peor que adivinar (pero se puede invertir)
                        
                        **¿Cuándo usar ROC?**
                        - Cuando las clases están relativamente balanceadas
                        - Para comparar modelos rápidamente
                        - Cuando te importa el rendimiento general
                        
                        ---
                        
                        **Curva Precision-Recall (P-R)**
                        
                        **¿Qué muestra?**
                        - **Eje X:** Recall (Sensibilidad) = TP / (TP + FN)
                        - **Eje Y:** Precision (Precisión) = TP / (TP + FP)
                        - **Línea horizontal:** Baseline (proporción de casos positivos)
                        - **AP (Average Precision):** Métrica resumen del rendimiento
                        
                        **Interpretación:**
                        - **AP alto:** Buen balance entre precisión y recall
                        - **Curva cerca del ángulo superior derecho:** Excelente rendimiento
                        - **Por encima del baseline:** Mejor que una predicción aleatoria
                        
                        **¿Cuándo usar P-R?**
                        - ✅ **Clases desbalanceadas** (muchos más negativos que positivos)
                        - ✅ Cuando los **falsos positivos son costosos**
                        - ✅ Para datasets con **pocos casos positivos**
                        - ✅ En problemas como **detección de fraude, diagnóstico médico**
                        
                        **Comparación ROC vs P-R:**
                        - **ROC** es más optimista con clases desbalanceadas
                        - **P-R** es más conservadora y realista
                        - **P-R** se enfoca más en el rendimiento de la clase minoritaria
                        - Usar **ambas** para una evaluación completa
                        """)

                    # Mostrar con 80% del ancho
                    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                    with col2:
                        st.pyplot(fig, use_container_width=True)

                    # Interpretación de las curvas
                    st.markdown("### 📋 Interpretación de las Curvas")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Curva ROC:**")
                        if roc_auc >= 0.9:
                            st.success(
                                f"🎯 Excelente discriminación (AUC = {roc_auc:.3f})")
                        elif roc_auc >= 0.8:
                            st.info(
                                f"👍 Buena discriminación (AUC = {roc_auc:.3f})")
                        elif roc_auc >= 0.7:
                            st.warning(
                                f"⚠️ Discriminación moderada (AUC = {roc_auc:.3f})")
                        else:
                            st.error(
                                f"❌ Discriminación pobre (AUC = {roc_auc:.3f})")

                    with col2:
                        st.markdown("**Curva Precision-Recall:**")
                        baseline_precision = np.sum(y_test)/len(y_test)
                        if avg_precision >= baseline_precision + 0.3:
                            st.success(
                                f"🎯 Excelente (AP = {avg_precision:.3f})")
                        elif avg_precision >= baseline_precision + 0.1:
                            st.info(
                                f"👍 Buena mejora sobre baseline (AP = {avg_precision:.3f})")
                        elif avg_precision >= baseline_precision:
                            st.warning(
                                f"⚠️ Mejora marginal (AP = {avg_precision:.3f})")
                        else:
                            st.error(
                                f"❌ Por debajo del baseline (AP = {avg_precision:.3f})")

                # Análisis de probabilidades de predicción
                st.markdown("### 📊 Distribución de Probabilidades")

                # Explicación detallada sobre distribución de probabilidades
                with st.expander("ℹ️ ¿Cómo interpretar la Distribución de Probabilidades?", expanded=False):
                    st.markdown("""
                    **¿Qué muestra este gráfico?**
                    
                    Este histograma muestra cómo el modelo asigna probabilidades a cada muestra del conjunto de prueba, 
                    separado por la clase real a la que pertenece cada muestra.
                    
                    **Elementos del gráfico:**
                    - **Histograma azul:** Distribución de probabilidades para muestras que realmente pertenecen a la clase positiva
                    - **Histograma rojo:** Distribución de probabilidades para muestras que realmente pertenecen a la clase negativa  
                    - **Línea roja vertical:** Umbral de decisión (0.5 por defecto)
                    - **Eje X:** Probabilidad asignada por el modelo (0 = clase negativa, 1 = clase positiva)
                    - **Eje Y:** Cantidad de muestras
                    
                    **Interpretación ideal:**
                    - ✅ **Buena separación:** Los histogramas no se superponen mucho
                    - ✅ **Clase negativa:** Concentrada cerca de 0 (izquierda)
                    - ✅ **Clase positiva:** Concentrada cerca de 1 (derecha)
                    - ✅ **Pocas muestras cerca del umbral (0.5):** Indica confianza en las predicciones
                    
                    **Problemas a identificar:**
                    - ⚠️ **Mucha superposición:** Indica dificultad para separar las clases
                    - ⚠️ **Concentración en el centro (0.3-0.7):** El modelo está inseguro
                    - ⚠️ **Distribución uniforme:** El modelo no está aprendiendo patrones útiles
                    
                    **Aplicaciones prácticas:**
                    - Identificar si el modelo está confiado en sus predicciones
                    - Evaluar si cambiar el umbral de decisión podría mejorar el rendimiento
                    - Detectar casos donde el modelo necesita más datos o características
                    """)

                # Verificar que tenemos datos válidos
                if y_pred_proba is not None and len(y_pred_proba) > 0:
                    unique_classes = np.unique(y_test)

                    # Para clasificación binaria
                    if len(unique_classes) == 2:
                        fig, ax = plt.subplots(figsize=(12, 6))

                        # Obtener probabilidades de la clase positiva
                        prob_class_1 = y_pred_proba[:, 1]

                        # Separar por clase real - usar los valores únicos reales
                        mask_class_0 = (y_test == unique_classes[0])
                        mask_class_1 = (y_test == unique_classes[1])

                        prob_class_0_real = prob_class_1[mask_class_0]
                        prob_class_1_real = prob_class_1[mask_class_1]

                        # Crear histogramas solo si hay datos
                        if len(prob_class_0_real) > 0:
                            ax.hist(prob_class_0_real, bins=20, alpha=0.7,
                                    label=f'Clase {class_names[0] if class_names and len(class_names) > 0 else unique_classes[0]} (Real)',
                                    color='lightcoral', edgecolor='black')

                        if len(prob_class_1_real) > 0:
                            ax.hist(prob_class_1_real, bins=20, alpha=0.7,
                                    label=f'Clase {class_names[1] if class_names and len(class_names) > 1 else unique_classes[1]} (Real)',
                                    color='lightblue', edgecolor='black')

                        # Línea del umbral de decisión
                        ax.axvline(x=0.5, color='red', linestyle='--',
                                   linewidth=2, label='Umbral de decisión (0.5)')

                        # Configurar el gráfico
                        ax.set_xlabel(
                            'Probabilidad de Clase Positiva', fontsize=12)
                        ax.set_ylabel('Frecuencia', fontsize=12)
                        ax.set_title('Distribución de Probabilidades Predichas por Clase Real',
                                     fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        # Asegurar límites apropiados
                        ax.set_xlim(0, 1)

                        plt.tight_layout()

                        # Mostrar el gráfico
                        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                        with col2:
                            st.pyplot(fig, use_container_width=True)

                        # Limpiar la figura
                        plt.close(fig)

                        # Análisis de separación
                        if len(prob_class_0_real) > 0 and len(prob_class_1_real) > 0:
                            # Contar solapamiento en la zona de incertidumbre (0.3-0.7)
                            overlap_0 = np.sum(
                                (prob_class_0_real > 0.3) & (prob_class_0_real < 0.7))
                            overlap_1 = np.sum(
                                (prob_class_1_real > 0.3) & (prob_class_1_real < 0.7))
                            total_overlap = overlap_0 + overlap_1

                            overlap_percentage = total_overlap / len(y_test)

                            # Métricas adicionales de separación
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Muestras en zona incierta",
                                          f"{total_overlap}/{len(y_test)}")
                            with col2:
                                st.metric("Porcentaje de incertidumbre",
                                          f"{overlap_percentage:.1%}")
                            with col3:
                                conf_threshold = 0.8  # 80% de confianza
                                high_conf = np.sum((prob_class_1 < 0.2) | (
                                    prob_class_1 > conf_threshold))
                                st.metric("Predicciones confiables",
                                          f"{high_conf}/{len(y_test)}")

                            # Interpretación
                            if overlap_percentage < 0.2:
                                st.success(
                                    "✅ Excelente separación entre clases - El modelo está muy confiado en sus predicciones")
                            elif overlap_percentage < 0.4:
                                st.info("👍 Buena separación entre clases")
                            else:
                                st.warning(
                                    "⚠️ Las clases se superponen significativamente - Considera ajustar el umbral de decisión")

                    elif len(unique_classes) > 2:
                        # Para clasificación multiclase
                        st.info(
                            "**Nota:** Clasificación multiclase detectada. Mostrando distribución de probabilidades para cada clase.")

                        n_classes = len(unique_classes)
                        n_cols = min(3, n_classes)
                        n_rows = (n_classes + n_cols - 1) // n_cols

                        fig, axes = plt.subplots(
                            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

                        # Manejar caso de una sola clase
                        if n_classes == 1:
                            axes = [axes]
                        elif n_rows == 1:
                            axes = axes if n_cols > 1 else [axes]
                        else:
                            axes = axes.flatten()

                        for i, class_val in enumerate(unique_classes):
                            if i < len(axes):
                                ax_sub = axes[i]

                                # Probabilidades para esta clase
                                class_probs = y_pred_proba[:, i]

                                ax_sub.hist(class_probs, bins=20, alpha=0.7,
                                            color=plt.cm.Set3(i), edgecolor='black')

                                class_label = class_names[i] if class_names and i < len(
                                    class_names) else f"Clase {class_val}"
                                ax_sub.set_title(
                                    f'Probabilidades para {class_label}')
                                ax_sub.set_xlabel('Probabilidad')
                                ax_sub.set_ylabel('Frecuencia')
                                ax_sub.grid(True, alpha=0.3)
                                ax_sub.set_xlim(0, 1)

                        # Ocultar subplots vacíos
                        for i in range(n_classes, len(axes)):
                            axes[i].set_visible(False)

                        plt.tight_layout()

                        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                        with col2:
                            st.pyplot(fig, use_container_width=True)

                        plt.close(fig)
                    else:
                        st.error(
                            "Error: Datos de clasificación insuficientes para crear la visualización")

                else:
                    st.error(
                        "Error: No hay probabilidades predichas disponibles. Asegúrate de que el modelo esté entrenado correctamente.")

                # Análisis de umbrales de decisión para clasificación binaria
                if len(np.unique(y_test)) == 2:
                    st.markdown("### 🎯 Análisis de Umbrales de Decisión")

                    # Explicación detallada sobre umbrales de decisión
                    with st.expander("ℹ️ ¿Cómo interpretar el Análisis de Umbrales?", expanded=False):
                        st.markdown("""
                        **¿Qué es el umbral de decisión?**
                        
                        El umbral de decisión es el valor que determina cuándo el modelo clasifica una muestra como 
                        positiva o negativa. Por defecto, este umbral es **0.5**:
                        - **Probabilidad ≥ 0.5** → Clase Positiva
                        - **Probabilidad < 0.5** → Clase Negativa
                        
                        **¿Por qué cambiar el umbral?**
                        
                        El umbral por defecto (0.5) no siempre es óptimo. Dependiendo del problema, 
                        puede ser beneficioso ajustarlo:
                        
                        **📈 Umbral más alto (0.6, 0.7, 0.8):**
                        - ✅ **Mayor Precisión:** Menos falsos positivos
                        - ✅ **Predicciones más conservadoras:** Solo clasifica como positivo cuando está muy seguro
                        - ⚠️ **Menor Recall:** Puede perder casos positivos reales
                        - **Útil cuando:** Los falsos positivos son muy costosos (ej: diagnóstico médico, inversiones)
                        
                        **📉 Umbral más bajo (0.3, 0.4):**
                        - ✅ **Mayor Recall:** Detecta más casos positivos reales
                        - ✅ **Predicciones más sensibles:** No se pierde tantos casos positivos
                        - ⚠️ **Menor Precisión:** Más falsos positivos
                        - **Útil cuando:** Los falsos negativos son muy costosos (ej: detección de fraude, seguridad)
                        
                        **Métricas mostradas:**
                        - **Accuracy:** Porcentaje total de predicciones correctas
                        - **Precision:** De las predicciones positivas, cuántas son correctas
                        - **Recall:** De los casos positivos reales, cuántos detectamos
                        - **F1-Score:** Balance entre precisión y recall
                        
                        **¿Cómo elegir el umbral óptimo?**
                        1. **Maximizar F1-Score:** Balance general entre precisión y recall
                        2. **Maximizar Precision:** Si los falsos positivos son costosos
                        3. **Maximizar Recall:** Si los falsos negativos son costosos
                        4. **Considerar el contexto:** Costos reales de errores en tu dominio
                        
                        **Ejemplo práctico:**
                        - **Email spam:** Prefiere falsos positivos (email importante en spam) que falsos negativos
                        - **Diagnóstico médico:** Prefiere falsos positivos (más pruebas) que falsos negativos (enfermedad no detectada)
                        - **Recomendaciones:** Balance entre no molestar (precisión) y no perder oportunidades (recall)
                        """)

                    # Calcular métricas para diferentes umbrales
                    thresholds = np.arange(0.1, 1.0, 0.1)
                    threshold_metrics = []

                    for threshold in thresholds:
                        y_pred_thresh = (
                            y_pred_proba[:, 1] >= threshold).astype(int)

                        if len(np.unique(y_pred_thresh)) > 1:  # Evitar división por cero
                            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                            precision = precision_score(
                                y_test, y_pred_thresh, zero_division=0)
                            recall = recall_score(
                                y_test, y_pred_thresh, zero_division=0)
                            f1 = f1_score(y_test, y_pred_thresh,
                                          zero_division=0)
                            accuracy = accuracy_score(y_test, y_pred_thresh)

                            threshold_metrics.append({
                                'Umbral': threshold,
                                'Accuracy': accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1-Score': f1
                            })

                    if threshold_metrics:
                        df_thresholds = pd.DataFrame(threshold_metrics)

                        # Encontrar el mejor umbral por F1-Score
                        best_f1_idx = df_thresholds['F1-Score'].idxmax()
                        best_threshold = df_thresholds.loc[best_f1_idx, 'Umbral']

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Umbral Actual", "0.50")
                            st.metric("Umbral Óptimo (F1)",
                                      f"{best_threshold:.2f}")

                            if abs(best_threshold - 0.5) > 0.1:
                                st.info(
                                    f"💡 Considera ajustar el umbral a {best_threshold:.2f} para mejorar el F1-Score")

                        with col2:
                            # Mostrar tabla de umbrales (seleccionados)
                            display_thresholds = df_thresholds[df_thresholds['Umbral'].isin(
                                [0.3, 0.5, 0.7])].copy()
                            for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                                display_thresholds[col] = display_thresholds[col].apply(
                                    lambda x: f"{x:.3f}")

                            st.markdown("**Comparación de Umbrales:**")
                            st.dataframe(
                                display_thresholds, hide_index=True, use_container_width=True)

        else:
            st.info("Entrena un modelo primero para ver las visualizaciones.")

    # Pestaña de Coeficientes
    elif st.session_state.active_tab_lr == 4:
        st.header("Coeficientes del Modelo")

        if st.session_state.get('model_trained_lr', False):
            model = st.session_state.get('model_lr')
            feature_names = st.session_state.get('feature_names_lr', [])
            model_type = st.session_state.get('model_type_lr', 'Linear')

            # Información sobre los coeficientes
            with st.expander("ℹ️ ¿Cómo interpretar los coeficientes?", expanded=False):
                if model_type == "Linear":
                    st.markdown("""
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
                    """)
                else:
                    st.markdown("""
                    # 🎯 **Coeficientes en Regresión Logística**
                    
                    ## 🎯 **¿Qué representan los coeficientes?**
                    
                    En regresión logística, los coeficientes representan el **cambio en log-odds** 
                    (logaritmo de las probabilidades) por cada unidad de cambio en la característica. 
                    La fórmula es:
                    
                    ```
                    log(odds) = β₀ + β₁×X₁ + β₂×X₂ + ... + βₙ×Xₙ
                    odds = e^(β₀ + β₁×X₁ + β₂×X₂ + ... + βₙ×Xₙ)
                    probabilidad = odds / (1 + odds)
                    ```
                    
                    ## 🔍 **Interpretación de Log-Odds:**
                    
                    ### **Valor del Coeficiente:**
                    - **Positivo (+):** 📈 **Aumenta** la probabilidad de la clase positiva
                    - **Negativo (-):** 📉 **Disminuye** la probabilidad de la clase positiva
                    - **Cero (0.0):** **No afecta** la probabilidad
                    
                    ### **Magnitud del Coeficiente:**
                    - **|β| > 2:** **Efecto muy fuerte** (odds ratio > 7.4)
                    - **1 < |β| < 2:** **Efecto fuerte** (odds ratio entre 2.7 y 7.4)
                    - **0.5 < |β| < 1:** **Efecto moderado** (odds ratio entre 1.6 y 2.7)
                    - **|β| < 0.5:** **Efecto débil** (odds ratio < 1.6)
                    
                    ## 📊 **Conversión a Odds Ratios (Más Intuitivo):**
                    
                    Los **Odds Ratios** se calculan como `e^β` y son más fáciles de interpretar:
                    
                    ### **Interpretación de Odds Ratios:**
                    - **OR = 1:** La característica **no tiene efecto**
                    - **OR > 1:** La característica **aumenta** las odds de la clase positiva
                    - **OR < 1:** La característica **disminuye** las odds de la clase positiva
                    
                    ### **Ejemplos de Odds Ratios:**
                    - **OR = 2.0:** Duplica las odds (100% de aumento)
                    - **OR = 1.5:** Aumenta las odds en 50%
                    - **OR = 0.5:** Reduce las odds a la mitad (50% de reducción)
                    - **OR = 0.2:** Reduce las odds en 80%
                    
                    ## 💡 **Ejemplos Prácticos:**
                    
                    **🏥 Diagnóstico Médico (Predicción de Enfermedad):**
                    - `Edad = +0.05 (OR=1.05)`: Cada año adicional aumenta las odds en 5%
                    - `Ejercicio = -1.2 (OR=0.30)`: Hacer ejercicio reduce las odds en 70%
                    - `Fumador = +1.8 (OR=6.05)`: Ser fumador multiplica las odds por 6
                    
                    **📧 Clasificación de Spam:**
                    - `Palabras_Sospechosas = +2.3 (OR=10.0)`: Multiplica las odds de spam por 10
                    - `Remitente_Conocido = -1.5 (OR=0.22)`: Reduce las odds de spam en 78%
                    
                    **💳 Detección de Fraude:**
                    - `Hora_Inusual = +1.1 (OR=3.0)`: Triplica las odds de fraude
                    - `Ubicacion_Habitual = -2.0 (OR=0.14)`: Reduce las odds de fraude en 86%
                    
                    ## ⚠️ **Limitaciones y Consideraciones:**
                    
                    1. **Interpretación no lineal:** El efecto en probabilidades no es constante
                    2. **Asunciones de linealidad:** En el espacio log-odds, no en probabilidades
                    3. **Interacciones ignoradas:** Los efectos pueden depender de otras variables
                    4. **Escalas importantes:** Normalizar puede facilitar la interpretación
                    
                    ## 🎛️ **Cómo usar esta información:**
                    
                    ### **Para el Negocio:**
                    - **Identificar factores de riesgo:** Coeficientes positivos altos
                    - **Encontrar factores protectores:** Coeficientes negativos altos
                    - **Priorizar intervenciones:** Enfocarse en variables con mayor impacto
                    
                    ### **Para el Modelo:**
                    - **Validar coherencia:** ¿Los signos tienen sentido del dominio?
                    - **Detectar overfitting:** ¿Coeficientes extremadamente grandes?
                    - **Selección de características:** ¿Coeficientes cerca de cero?
                    
                    ## 🧮 **Fórmula de Conversión:**
                    ```python
                    # De coeficiente a odds ratio
                    odds_ratio = np.exp(coeficiente)
                    
                    # Cambio porcentual en odds
                    cambio_porcentual = (odds_ratio - 1) * 100
                    ```
                    """)

            if model is not None and hasattr(model, 'coef_'):
                try:
                    # Preparar datos de coeficientes de forma robusta
                    coef_raw = model.coef_
                    class_names = st.session_state.get('class_names_lr', [])

                    # Para regresión logística binaria, coef_ tiene forma (1, n_features)
                    # Para regresión logística multiclase, coef_ tiene forma (n_classes, n_features)
                    # Para regresión lineal, coef_ tiene forma (n_features,)

                    if len(coef_raw.shape) == 2:
                        # Es una matriz 2D (regresión logística)
                        if coef_raw.shape[0] == 1:
                            # Clasificación binaria: tomar la primera (y única) fila
                            coefficients = coef_raw[0]
                            is_multiclass = False
                        else:
                            # Clasificación multiclase: mostrar nota informativa
                            # Usar primera clase por defecto
                            coefficients = coef_raw[0]
                            is_multiclass = True

                            st.info(
                                f"**Nota:** Este es un modelo de clasificación multiclase con {coef_raw.shape[0]} clases. Mostrando los coeficientes para la primera clase ({class_names[0] if class_names else 'Clase 0'}).")

                            # Opción para seleccionar qué clase mostrar
                            if class_names and len(class_names) == coef_raw.shape[0]:
                                selected_class = st.selectbox(
                                    "Selecciona la clase para mostrar coeficientes:",
                                    options=range(len(class_names)),
                                    format_func=lambda x: class_names[x],
                                    index=0
                                )
                                coefficients = coef_raw[selected_class]
                    else:
                        # Es un vector 1D (regresión lineal)
                        coefficients = coef_raw
                        is_multiclass = False

                    # Asegurar que coefficients es un array 1D
                    coefficients = np.array(coefficients).flatten()

                    # Verificar que las longitudes coincidan
                    if len(coefficients) != len(feature_names):
                        st.error(
                            f"Error: Se encontraron {len(coefficients)} coeficientes pero {len(feature_names)} características.")
                        st.error(f"Forma de coef_: {coef_raw.shape}")
                        st.error(f"Características: {feature_names}")
                        return

                    coef_df = pd.DataFrame({
                        'Característica': feature_names,
                        'Coeficiente': coefficients,
                        'Valor_Absoluto': np.abs(coefficients)
                    })

                except Exception as e:
                    st.error(f"Error al procesar los coeficientes: {str(e)}")
                    st.error(f"Forma de model.coef_: {model.coef_.shape}")
                    st.error(
                        f"Número de características: {len(feature_names)}")
                    return
                coef_df = coef_df.sort_values(
                    'Valor_Absoluto', ascending=False)

                # Añadir interpretación
                coef_df['Efecto'] = coef_df['Coeficiente'].apply(
                    lambda x: '📈 Positivo' if x > 0 else '📉 Negativo'
                )
                coef_df['Importancia'] = coef_df['Valor_Absoluto'].apply(
                    lambda x: '🔥 Alta' if x > coef_df['Valor_Absoluto'].quantile(0.75)
                    else ('🔶 Media' if x > coef_df['Valor_Absoluto'].quantile(0.25) else '🔹 Baja')
                )

                # Mostrar tabla de coeficientes
                st.markdown("### 📊 Tabla de Coeficientes")

                # Crear tabla mejorada con información adicional para regresión logística
                if model_type == "Logistic":
                    # Agregar odds ratios y cambios porcentuales
                    coef_df['Odds_Ratio'] = np.exp(coef_df['Coeficiente'])
                    coef_df['Cambio_Porcentual'] = (
                        coef_df['Odds_Ratio'] - 1) * 100

                    # Interpretación de la magnitud del efecto
                    def interpretar_efecto_logistico(odds_ratio):
                        if odds_ratio > 7.4:  # |coef| > 2
                            return '🔥 Muy Fuerte'
                        elif odds_ratio > 2.7:  # |coef| > 1
                            return '🔥 Fuerte'
                        elif odds_ratio > 1.6 or odds_ratio < 0.625:  # |coef| > 0.5
                            return '🔶 Moderado'
                        else:
                            return '🔹 Débil'

                    coef_df['Fuerza_Efecto'] = coef_df['Odds_Ratio'].apply(
                        interpretar_efecto_logistico)

                    # Formatear la tabla para regresión logística
                    display_df = coef_df[[
                        'Característica', 'Coeficiente', 'Odds_Ratio', 'Cambio_Porcentual', 'Efecto', 'Fuerza_Efecto']].copy()
                    display_df['Coeficiente'] = display_df['Coeficiente'].apply(
                        lambda x: f"{x:.4f}")
                    display_df['Odds_Ratio'] = display_df['Odds_Ratio'].apply(
                        lambda x: f"{x:.3f}")
                    display_df['Cambio_Porcentual'] = display_df['Cambio_Porcentual'].apply(
                        lambda x: f"{x:+.1f}%")

                    # Renombrar columnas para mayor claridad
                    display_df = display_df.rename(columns={
                        'Odds_Ratio': 'Odds Ratio',
                        'Cambio_Porcentual': 'Cambio en Odds',
                        'Fuerza_Efecto': 'Fuerza del Efecto'
                    })

                    st.dataframe(
                        display_df, use_container_width=True, hide_index=True)

                    # Explicación de la tabla para regresión logística
                    with st.expander("📖 ¿Cómo leer esta tabla?", expanded=False):
                        st.markdown("""
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
                        """)

                else:
                    # Tabla estándar para regresión lineal
                    display_df = coef_df[[
                        'Característica', 'Coeficiente', 'Efecto', 'Importancia']].copy()
                    display_df['Coeficiente'] = display_df['Coeficiente'].apply(
                        lambda x: f"{x:.4f}")

                    st.dataframe(
                        display_df, use_container_width=True, hide_index=True)

                    # Explicación de la tabla para regresión lineal
                    with st.expander("📖 ¿Cómo leer esta tabla?", expanded=False):
                        st.markdown("""
                        **Columnas de la tabla:**
                        
                        - **Coeficiente:** Cambio en la variable objetivo por unidad de cambio en la característica
                        - **Efecto:** Dirección de la relación (positiva/negativa)
                        - **Importancia:** Magnitud relativa del efecto
                        
                        **Interpretación directa:**
                        - Un coeficiente de **+50** significa que cada unidad adicional de esa característica 
                          aumenta la predicción en 50 unidades
                        - Un coeficiente de **-20** significa que cada unidad adicional de esa característica 
                          disminuye la predicción en 20 unidades
                        """)

                # Mostrar intercepto si existe
                if hasattr(model, 'intercept_'):
                    st.markdown("### 🎯 Intercepto del Modelo")
                    intercept = model.intercept_[0] if hasattr(
                        model.intercept_, '__len__') else model.intercept_

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Intercepto", f"{intercept:.4f}")

                    if model_type == "Logistic":
                        with col2:
                            intercept_odds_ratio = np.exp(intercept)
                            st.metric("Odds Ratio Base",
                                      f"{intercept_odds_ratio:.3f}")

                        st.info(
                            f"**Interpretación:** Cuando todas las características son 0, el log-odds base es {intercept:.4f} "
                            f"(Odds Ratio = {intercept_odds_ratio:.3f})")
                    else:
                        st.info(
                            f"**Interpretación:** Cuando todas las características son 0, el modelo predice un valor de {intercept:.4f}")

                # Gráfico de coeficientes
                st.markdown("### 📈 Visualización de Coeficientes")

                fig, ax = plt.subplots(
                    figsize=(12, max(6, len(feature_names) * 0.4)))

                # Crear gráfico de barras horizontal
                colors = ['#ff6b6b' if x <
                          0 else '#4ecdc4' for x in coef_df['Coeficiente']]
                bars = ax.barh(range(
                    len(coef_df)), coef_df['Coeficiente'], color=colors, alpha=0.7, edgecolor='black')

                # Personalización
                ax.set_yticks(range(len(coef_df)))
                ax.set_yticklabels(coef_df['Característica'], fontsize=10)
                ax.set_xlabel('Valor del Coeficiente', fontsize=12)
                ax.set_title(
                    'Coeficientes del Modelo (ordenados por importancia)', fontsize=14, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-',
                           alpha=0.8, linewidth=1)
                ax.grid(True, alpha=0.3, axis='x')

                # Añadir valores en las barras
                for i, (bar, coef) in enumerate(zip(bars, coef_df['Coeficiente'])):
                    width = bar.get_width()
                    ax.text(width + (0.01 * (max(coef_df['Coeficiente']) - min(coef_df['Coeficiente']))) if width >= 0
                            else width - (0.01 * (max(coef_df['Coeficiente']) - min(coef_df['Coeficiente']))),
                            bar.get_y() + bar.get_height()/2,
                            f'{coef:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)

                # Leyenda
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#4ecdc4', alpha=0.7,
                          label='Efecto Positivo'),
                    Patch(facecolor='#ff6b6b', alpha=0.7,
                          label='Efecto Negativo')
                ]
                ax.legend(handles=legend_elements, loc='upper right')

                plt.tight_layout()

                # Mostrar con 80% del ancho
                col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                with col2:
                    st.pyplot(fig, use_container_width=True)

                # Análisis de importancia
                st.markdown("### 🔍 Análisis Detallado de Importancia")

                # Identificar las características más importantes
                top_features = coef_df.head(5)  # Mostrar top 5 en lugar de 3

                # Análisis diferenciado por tipo de modelo
                if model_type == "Logistic":
                    st.markdown(
                        "#### 🎯 **Características que MÁS aumentan la probabilidad:**")
                    positive_features = coef_df[coef_df['Coeficiente'] > 0].head(
                        3)

                    if len(positive_features) > 0:
                        for i, row in positive_features.iterrows():
                            odds_ratio = np.exp(row['Coeficiente'])
                            cambio_pct = (odds_ratio - 1) * 100

                            # Determinar la intensidad del efecto
                            if odds_ratio > 3:
                                intensidad = "🔥 **FUERTE**"
                            elif odds_ratio > 1.5:
                                intensidad = "🔶 **MODERADO**"
                            else:
                                intensidad = "🔹 **DÉBIL**"

                            st.markdown(f"""
                            **{i+1}. {row['Característica']}** {intensidad}
                            - Coeficiente: `{row['Coeficiente']:.4f}`
                            - Odds Ratio: `{odds_ratio:.3f}`
                            - **Impacto:** Cada unidad adicional **multiplica las odds por {odds_ratio:.2f}** (aumenta {cambio_pct:+.1f}%)
                            """)
                    else:
                        st.info(
                            "No hay características que aumenten significativamente la probabilidad.")

                    st.markdown(
                        "#### 📉 **Características que MÁS disminuyen la probabilidad:**")
                    negative_features = coef_df[coef_df['Coeficiente'] < 0].head(
                        3)

                    if len(negative_features) > 0:
                        for i, row in negative_features.iterrows():
                            odds_ratio = np.exp(row['Coeficiente'])
                            reduccion_pct = (1 - odds_ratio) * 100

                            # Determinar la intensidad del efecto
                            if odds_ratio < 0.33:  # Reduce a menos de 1/3
                                intensidad = "🔥 **FUERTE**"
                            elif odds_ratio < 0.67:  # Reduce a menos de 2/3
                                intensidad = "🔶 **MODERADO**"
                            else:
                                intensidad = "🔹 **DÉBIL**"

                            st.markdown(f"""
                            **{i+1}. {row['Característica']}** {intensidad}
                            - Coeficiente: `{row['Coeficiente']:.4f}`
                            - Odds Ratio: `{odds_ratio:.3f}`
                            - **Impacto:** Cada unidad adicional **reduce las odds en {reduccion_pct:.1f}%**
                            """)
                    else:
                        st.info(
                            "No hay características que disminuyan significativamente la probabilidad.")

                    # Resumen ejecutivo para regresión logística
                    st.markdown("#### 📋 **Resumen Ejecutivo:**")

                    # Característica más influyente
                    most_influential = coef_df.iloc[0]
                    odds_ratio_most = np.exp(most_influential['Coeficiente'])

                    if most_influential['Coeficiente'] > 0:
                        impacto_desc = f"multiplica las odds por {odds_ratio_most:.2f}"
                    else:
                        reduccion = (1 - odds_ratio_most) * 100
                        impacto_desc = f"reduce las odds en {reduccion:.1f}%"

                    st.success(f"""
                    🎯 **Factor más determinante:** `{most_influential['Característica']}`
                    
                    Esta característica es la que mayor impacto tiene en las predicciones del modelo.
                    Cada unidad adicional {impacto_desc}.
                    """)

                    # Verificar balance de efectos
                    n_positive = len(coef_df[coef_df['Coeficiente'] > 0])
                    n_negative = len(coef_df[coef_df['Coeficiente'] < 0])

                    if n_positive == 0:
                        st.warning(
                            "⚠️ **Atención:** Todas las características reducen la probabilidad. Verifica que el modelo esté bien entrenado.")
                    elif n_negative == 0:
                        st.warning(
                            "⚠️ **Atención:** Todas las características aumentan la probabilidad. Verifica que el modelo esté bien entrenado.")
                    else:
                        st.info(
                            f"✅ **Balance:** {n_positive} características aumentan y {n_negative} disminuyen la probabilidad.")

                else:  # Regresión lineal
                    st.markdown(
                        "#### 📈 **Características que MÁS aumentan el valor predicho:**")
                    positive_features = coef_df[coef_df['Coeficiente'] > 0].head(
                        3)

                    if len(positive_features) > 0:
                        for i, row in positive_features.iterrows():
                            coef_val = row['Coeficiente']

                            # Determinar la intensidad del efecto
                            abs_coef = abs(coef_val)
                            if abs_coef > coef_df['Valor_Absoluto'].quantile(0.8):
                                intensidad = "🔥 **FUERTE**"
                            elif abs_coef > coef_df['Valor_Absoluto'].quantile(0.6):
                                intensidad = "🔶 **MODERADO**"
                            else:
                                intensidad = "🔹 **DÉBIL**"

                            st.markdown(f"""
                            **{i+1}. {row['Característica']}** {intensidad}
                            - Coeficiente: `{coef_val:.4f}`
                            - **Impacto:** Cada unidad adicional **aumenta la predicción en {coef_val:.3f} unidades**
                            """)
                    else:
                        st.info(
                            "No hay características que aumenten significativamente el valor predicho.")

                    st.markdown(
                        "#### 📉 **Características que MÁS disminuyen el valor predicho:**")
                    negative_features = coef_df[coef_df['Coeficiente'] < 0].head(
                        3)

                    if len(negative_features) > 0:
                        for i, row in negative_features.iterrows():
                            coef_val = row['Coeficiente']

                            # Determinar la intensidad del efecto
                            abs_coef = abs(coef_val)
                            if abs_coef > coef_df['Valor_Absoluto'].quantile(0.8):
                                intensidad = "🔥 **FUERTE**"
                            elif abs_coef > coef_df['Valor_Absoluto'].quantile(0.6):
                                intensidad = "🔶 **MODERADO**"
                            else:
                                intensidad = "🔹 **DÉBIL**"

                            st.markdown(f"""
                            **{i+1}. {row['Característica']}** {intensidad}
                            - Coeficiente: `{coef_val:.4f}`
                            - **Impacto:** Cada unidad adicional **disminuye la predicción en {abs(coef_val):.3f} unidades**
                            """)
                    else:
                        st.info(
                            "No hay características que disminuyan significativamente el valor predicho.")

                    # Resumen ejecutivo para regresión lineal
                    st.markdown("#### 📋 **Resumen Ejecutivo:**")

                    most_influential = coef_df.iloc[0]
                    coef_val = most_influential['Coeficiente']

                    if coef_val > 0:
                        impacto_desc = f"aumenta la predicción en {coef_val:.3f} unidades"
                    else:
                        impacto_desc = f"disminuye la predicción en {abs(coef_val):.3f} unidades"

                    st.success(f"""
                    🎯 **Factor más determinante:** `{most_influential['Característica']}`
                    
                    Esta característica tiene el mayor impacto absoluto en las predicciones.
                    Cada unidad adicional {impacto_desc}.
                    """)

                # Recomendaciones generales
                st.markdown("#### 💡 **Recomendaciones para la Acción:**")

                # Identificar características controlables vs no controlables
                top_3 = coef_df.head(3)

                recommendations = []
                for i, row in top_3.iterrows():
                    feature_name = row['Característica']
                    effect_direction = "aumentar" if row['Coeficiente'] > 0 else "reducir"
                    target = "probabilidad positiva" if model_type == "Logistic" else "valor predicho"

                    recommendations.append(
                        f"**{feature_name}:** {effect_direction.capitalize()} esta característica para {'incrementar' if row['Coeficiente'] > 0 else 'decrementar'} la {target}")

                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")

                # Warning para coeficientes extremos
                extreme_coefs = coef_df[coef_df['Valor_Absoluto']
                                        > coef_df['Valor_Absoluto'].quantile(0.95)]
                if len(extreme_coefs) > 0:
                    st.warning(f"""
                    ⚠️ **Atención - Coeficientes Extremos Detectados:**
                    
                    Las siguientes características tienen coeficientes muy altos: {', '.join(extreme_coefs['Característica'].tolist())}
                    
                    Esto podría indicar:
                    - Overfitting del modelo
                    - Escalas muy diferentes entre características
                    - Multicolinealidad entre variables
                    - Outliers en los datos
                    
                    **Recomendación:** Considera normalizar las características o revisar la calidad de los datos.
                    """)

            else:
                st.error("El modelo no tiene coeficientes disponibles.")
        else:
            st.info("Entrena un modelo primero para ver los coeficientes.")

    # Pestaña de Predicciones
    elif st.session_state.active_tab_lr == 5:
        st.header("Hacer Predicciones")

        if st.session_state.get('model_trained_lr', False):
            model = st.session_state.get('model_lr')
            feature_names = st.session_state.get('feature_names_lr', [])
            dataset_name = st.session_state.get('selected_dataset_lr', '')
            X_train = st.session_state.get('X_train_lr')
            class_names = st.session_state.get('class_names_lr', [])
            model_type = st.session_state.get('model_type_lr', 'Linear')
            task_type = st.session_state.get('task_type_lr', 'Regresión')

            # Obtener metadata del dataset para adaptar los inputs
            from dataset_metadata import get_dataset_metadata
            metadata = get_dataset_metadata(dataset_name)
            feature_descriptions = metadata.get('feature_descriptions', {})
            value_mappings = metadata.get('value_mappings', {})
            original_to_display = metadata.get('original_to_display', {})
            categorical_features = metadata.get('categorical_features', [])

            st.markdown("### Ingresa los valores para hacer una predicción")

            # Analizar características si tenemos datos de entrenamiento
            feature_info = {}
            if X_train is not None:
                # Convertir a DataFrame si es necesario
                if hasattr(X_train, 'columns'):
                    column_names = X_train.columns
                else:
                    column_names = range(len(feature_names))

                for i, feature_display_name in enumerate(feature_names):
                    # Obtener el nombre original de la columna
                    if i < len(column_names):
                        original_col_name = column_names[i]
                    else:
                        original_col_name = feature_display_name

                    # Encontrar el nombre original usando reverse mapping
                    reverse_mapping = {v: k for k,
                                       v in original_to_display.items()}
                    if feature_display_name in reverse_mapping:
                        original_col_name = reverse_mapping[feature_display_name]

                    # Usar el índice para acceder a las columnas
                    if hasattr(X_train, 'iloc'):
                        feature_col = X_train.iloc[:, i]
                    else:
                        feature_col = X_train[:, i]

                    # Determinar tipo de característica
                    unique_values = len(set(feature_col)) if hasattr(
                        feature_col, '__iter__') else 10
                    unique_vals = sorted(list(set(feature_col))) if hasattr(
                        feature_col, '__iter__') else [0, 1]

                    # Verificar si es categórica según metadata
                    is_categorical_by_metadata = original_col_name in categorical_features

                    if unique_values <= 2:
                        feature_type = 'binary'
                    elif unique_values <= 10 and (all(isinstance(x, (int, float)) and x == int(x) for x in unique_vals) or is_categorical_by_metadata):
                        feature_type = 'categorical'
                    else:
                        feature_type = 'continuous'

                    # Preparar información de la característica
                    if feature_type in ['binary', 'categorical'] and original_col_name in value_mappings:
                        display_values = []
                        value_to_original = {}
                        for orig_val in unique_vals:
                            if orig_val in value_mappings[original_col_name]:
                                display_val = value_mappings[original_col_name][orig_val]
                                display_values.append(display_val)
                                value_to_original[display_val] = orig_val
                            else:
                                display_values.append(str(orig_val))
                                value_to_original[str(orig_val)] = orig_val

                        feature_info[i] = {
                            'type': feature_type,
                            'values': unique_vals,
                            'display_values': display_values,
                            'value_to_original': value_to_original,
                            'original_column': original_col_name,
                            'display_name': feature_display_name
                        }
                    else:
                        feature_info[i] = {
                            'type': feature_type,
                            'values': unique_vals,
                            'min': float(min(unique_vals)) if feature_type == 'continuous' else min(unique_vals),
                            'max': float(max(unique_vals)) if feature_type == 'continuous' else max(unique_vals),
                            'mean': float(sum(feature_col) / len(feature_col)) if feature_type == 'continuous' and hasattr(feature_col, '__iter__') else None,
                            'original_column': original_col_name,
                            'display_name': feature_display_name
                        }
            else:
                # Valores por defecto si no hay datos de entrenamiento
                for i, feature in enumerate(feature_names):
                    feature_info[i] = {
                        'type': 'continuous',
                        'min': 0.0,
                        'max': 10.0,
                        'mean': 5.0,
                        'original_column': feature,
                        'display_name': feature
                    }

            # Crear controles adaptativos para cada característica
            input_values = []
            cols = st.columns(min(3, len(feature_names)))

            for i, feature in enumerate(feature_names):
                with cols[i % len(cols)]:
                    info = feature_info.get(
                        i, {'type': 'continuous', 'min': 0.0, 'max': 10.0, 'mean': 5.0})

                    # Crear etiqueta con descripción si está disponible
                    original_col = info.get('original_column', feature)
                    description = feature_descriptions.get(original_col, '')
                    if description:
                        label = f"**{feature}**\n\n*{description}*"
                    else:
                        label = feature

                    if info['type'] == 'binary':
                        # Control para características binarias
                        if 'display_values' in info and 'value_to_original' in info:
                            selected_display = st.selectbox(
                                label,
                                options=info['display_values'],
                                index=0,
                                key=f"pred_input_{i}"
                            )
                            value = info['value_to_original'][selected_display]
                        elif len(info['values']) == 2 and 0 in info['values'] and 1 in info['values']:
                            value = st.checkbox(label, key=f"pred_input_{i}")
                            value = 1 if value else 0
                        else:
                            value = st.selectbox(
                                label,
                                options=info['values'],
                                index=0,
                                key=f"pred_input_{i}"
                            )
                        input_values.append(value)

                    elif info['type'] == 'categorical':
                        # Control para características categóricas
                        if 'display_values' in info and 'value_to_original' in info:
                            selected_display = st.selectbox(
                                label,
                                options=info['display_values'],
                                index=0,
                                key=f"pred_input_{i}"
                            )
                            value = info['value_to_original'][selected_display]
                        else:
                            value = st.selectbox(
                                label,
                                options=info['values'],
                                index=0,
                                key=f"pred_input_{i}"
                            )
                        input_values.append(value)

                    else:  # continuous
                        # Control para características continuas
                        if 'min' in info and 'max' in info:
                            step = (info['max'] - info['min']) / \
                                100 if info['max'] != info['min'] else 0.1
                            default_val = info.get(
                                'mean', (info['min'] + info['max']) / 2)
                            value = st.slider(
                                label,
                                min_value=info['min'],
                                max_value=info['max'],
                                value=default_val,
                                step=step,
                                key=f"pred_input_{i}"
                            )
                        else:
                            value = st.number_input(
                                label,
                                value=0.0,
                                key=f"pred_input_{i}"
                            )
                        input_values.append(value)

            if st.button("🔮 Predecir", key="predict_lr_button"):
                try:
                    if model is not None:
                        prediction = model.predict([input_values])[0]

                        # For logistic regression, convert numeric prediction to class label
                        if task_type == 'Clasificación' and class_names is not None:
                            prediction_label = class_names[int(prediction)]
                            st.success(f"Predicción: {prediction_label}")
                        else:
                            # For regression, show numeric prediction
                            st.success(f"Predicción: {prediction:.4f}")
                    else:
                        st.error("Modelo no disponible")
                except Exception as e:
                    st.error(f"Error en la predicción: {str(e)}")
        else:
            st.info("Entrena un modelo primero para hacer predicciones.")

    # Pestaña de Exportar
    elif st.session_state.active_tab_lr == 6:
        st.header("Exportar Modelo")

        if st.session_state.get('model_trained_lr', False):
            model = st.session_state.get('model_lr')

            col1, col2 = st.columns(2)

            with col2:
                if st.button("📥 Descargar Modelo (Pickle)", key="download_pickle_lr"):
                    pickle_data = export_model_pickle(model)
                    st.download_button(
                        label="Descargar modelo.pkl",
                        data=pickle_data,
                        file_name="linear_regression_model.pkl",
                        mime="application/octet-stream"
                    )

            with col1:
                if st.button("📄 Generar Código", key="generate_code_lr"):
                    # Generate complete code for linear models
                    model_type = st.session_state.get(
                        'model_type_lr', 'Linear')
                    feature_names = st.session_state.get(
                        'feature_names_lr', [])
                    class_names = st.session_state.get('class_names_lr', [])

                    if model_type == "Logistic":
                        code = generate_logistic_regression_code(
                            feature_names, class_names)
                    else:
                        code = generate_linear_regression_code(feature_names)

                    st.code(code, language="python")

                    # Download button for the code
                    st.download_button(
                        label="📥 Descargar código",
                        data=code,
                        file_name=f"{'logistic' if model_type == 'Logistic' else 'linear'}_regression_code.py",
                        mime="text/plain"
                    )
        else:
            st.info("Entrena un modelo primero para exportarlo.")


def run_csv_loader_app():
    """Ejecuta la aplicación específica para cargar archivos CSV personalizados."""
    st.header("📁 Cargar CSV Personalizado")
    st.markdown(
        "Carga tu propio dataset en formato CSV para análisis personalizado")

    # Información sobre cargar CSV
    with st.expander("ℹ️ ¿Cómo usar esta herramienta?", expanded=True):
        st.markdown("""
        **Esta herramienta te permite cargar tus propios datasets CSV para análisis con Machine Learning.**

        ### 📋 Requisitos del archivo CSV:
        - Formato CSV con encabezados (primera fila con nombres de columnas)
        - Al menos 2 columnas (características + variable objetivo)
        - Datos limpios y estructurados
        - Codificación UTF-8 preferible

        ### 🔧 Funcionalidades:
        - **Vista previa automática** del dataset cargado
        - **Detección automática** del tipo de tarea (Clasificación/Regresión)
        - **Selección de columna objetivo** personalizable
        - **Estadísticas descriptivas** del dataset
        - **Integración completa** con todos los algoritmos disponibles

        ### 💡 Consejos:
        - Asegúrate de que los datos numéricos estén en formato correcto
        - Para clasificación, la columna objetivo debe contener categorías
        - Para regresión, la columna objetivo debe contener valores numéricos continuos
        """)

    # Usar la función existente de dataset_manager
    st.markdown("---")

    # Llamar a la función de carga de CSV sin mostrar datasets predefinidos
    result = create_dataset_selector(show_predefined=False)

    if result is not None:
        if isinstance(result, tuple):
            # CSV cargado exitosamente
            file_path, target_col, task_type = result

            st.markdown("---")
            st.success("✅ ¡Dataset CSV cargado exitosamente!")

            # Mostrar opciones de análisis
            st.markdown("### 🚀 Próximos pasos:")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🌲 Analizar con Árboles de Decisión",
                             key="analyze_trees",
                             use_container_width=True,
                             type="primary"):
                    st.session_state.navigation = "🌲 Árboles de Decisión"
                    st.rerun()

            with col2:
                if st.button("📊 Analizar con Regresión",
                             key="analyze_linear",
                             use_container_width=True,
                             type="primary"):
                    st.session_state.navigation = "📊 Regresión"
                    st.rerun()

            with col3:
                if st.button("🔄 Cargar otro archivo",
                             key="load_another",
                             use_container_width=True):
                    # Limpiar el estado del CSV cargado
                    if 'csv_datasets' in st.session_state:
                        st.session_state.csv_datasets.clear()
                    if 'selected_dataset' in st.session_state:
                        del st.session_state.selected_dataset
                    st.rerun()

            # Información adicional sobre el dataset cargado
            st.markdown("### 📊 Información del Dataset Cargado:")

            try:
                # Leer el archivo para mostrar estadísticas
                df = pd.read_csv(file_path)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📏 Filas", df.shape[0])
                with col2:
                    st.metric("📊 Columnas", df.shape[1])
                with col3:
                    st.metric("🎯 Variable Objetivo", target_col)
                with col4:
                    task_icon = "🏷️" if task_type == "Clasificación" else "📈"
                    st.metric(f"{task_icon} Tipo de Tarea", task_type)

                # Mostrar estadísticas descriptivas
                with st.expander("📈 Estadísticas Descriptivas", expanded=False):
                    st.dataframe(df.describe(), use_container_width=True)

                # Mostrar distribución de la variable objetivo
                with st.expander("🎯 Distribución de la Variable Objetivo", expanded=False):
                    if task_type == "Clasificación":
                        value_counts = df[target_col].value_counts()

                        col1, col2 = st.columns(2)
                        with col1:
                            st.dataframe(value_counts.to_frame(
                                "Cantidad"), use_container_width=True)

                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            value_counts.plot(kind='bar', ax=ax)
                            ax.set_title(f'Distribución de {target_col}')
                            ax.set_ylabel('Cantidad')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Estadísticas:**")
                            st.write(f"• Mínimo: {df[target_col].min():.4f}")
                            st.write(f"• Máximo: {df[target_col].max():.4f}")
                            st.write(f"• Media: {df[target_col].mean():.4f}")
                            st.write(
                                f"• Mediana: {df[target_col].median():.4f}")
                            st.write(
                                f"• Desv. Estándar: {df[target_col].std():.4f}")

                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            df[target_col].hist(bins=30, ax=ax, alpha=0.7)
                            ax.set_title(f'Distribución de {target_col}')
                            ax.set_xlabel(target_col)
                            ax.set_ylabel('Frecuencia')
                            plt.tight_layout()
                            st.pyplot(fig)

            except Exception as e:
                st.warning(
                    f"No se pudieron cargar las estadísticas adicionales: {str(e)}")

    else:
        # Mostrar consejos mientras no hay archivo cargado
        st.markdown("### 💡 Ejemplos de Datasets que puedes cargar:")

        examples = [
            {
                "name": "Dataset de Ventas",
                "description": "Datos de ventas con características como precio, descuento, temporada → Predictor de ventas",
                "task": "Regresión",
                "icon": "💰"
            },
            {
                "name": "Dataset de Clientes",
                "description": "Datos de clientes con edad, ingresos, historial → Clasificación de segmentos",
                "task": "Clasificación",
                "icon": "👥"
            },
            {
                "name": "Dataset de Productos",
                "description": "Características de productos con ratings → Predicción de popularidad",
                "task": "Regresión",
                "icon": "📦"
            },
            {
                "name": "Dataset Médico",
                "description": "Síntomas y características del paciente → Diagnóstico binario",
                "task": "Clasificación",
                "icon": "🏥"
            }
        ]

        for example in examples:
            with st.container():
                col1, col2 = st.columns([1, 10])
                with col1:
                    st.markdown(f"## {example['icon']}")
                with col2:
                    st.markdown(f"**{example['name']}** ({example['task']})")
                    st.markdown(f"_{example['description']}_")
                st.markdown("---")


def generate_linear_regression_code(feature_names):
    """Generate complete Python code for linear regression."""
    feature_names_str = str(feature_names)

    code = f"""# Código completo para Regresión Lineal
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. CARGAR Y PREPARAR LOS DATOS
# Reemplaza esta sección con tu método de carga de datos
# df = pd.read_csv('tu_archivo.csv')  # Cargar desde CSV
# O usa datos de ejemplo:

# Datos de ejemplo (reemplaza con tus datos reales)
# X = df[{feature_names_str}]  # Características
# y = df['variable_objetivo']  # Variable objetivo

# 2. DIVIDIR LOS DATOS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. CREAR Y ENTRENAR EL MODELO
model = LinearRegression()
model.fit(X_train, y_train)

# 4. HACER PREDICCIONES
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 5. EVALUAR EL MODELO
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("=== RESULTADOS DEL MODELO ===")
print(f"R² Score (Entrenamiento): {{r2_train:.4f}}")
print(f"R² Score (Prueba): {{r2_test:.4f}}")
print(f"MAE (Error Absoluto Medio): {{mae_test:.4f}}")
print(f"RMSE (Raíz Error Cuadrático Medio): {{rmse_test:.4f}}")

# 6. MOSTRAR COEFICIENTES
print("\\n=== COEFICIENTES DEL MODELO ===")
feature_names = {feature_names_str}
for i, coef in enumerate(model.coef_):
    print(f"{{feature_names[i]}}: {{coef:.4f}}")
print(f"Intercepto: {{model.intercept_:.4f}}")

# 7. VISUALIZAR RESULTADOS
plt.figure(figsize=(12, 5))

# Gráfico 1: Predicciones vs Valores Reales
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.grid(True, alpha=0.3)

# Gráfico 2: Residuos
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. FUNCIÓN PARA NUEVAS PREDICCIONES
def predecir_nuevo_valor(nuevo_ejemplo):
    \"\"\"
    Función para hacer predicciones con nuevos datos.
    
    Parámetros:
    nuevo_ejemplo: lista con valores para cada característica
                  en el orden: {feature_names_str}
    \"\"\"
    nuevo_ejemplo = np.array(nuevo_ejemplo).reshape(1, -1)
    prediccion = model.predict(nuevo_ejemplo)[0]
    return prediccion

# Ejemplo de uso para nuevas predicciones:
# nuevo_ejemplo = [valor1, valor2, valor3, ...]  # Reemplaza con tus valores
# resultado = predecir_nuevo_valor(nuevo_ejemplo)
# print(f"Predicción para nuevo ejemplo: {{resultado:.4f}}")

# 9. GUARDAR EL MODELO (OPCIONAL)
import pickle

# Guardar modelo
with open('modelo_regresion_lineal.pkl', 'wb') as f:
    pickle.dump(model, f)

# Cargar modelo guardado
# with open('modelo_regresion_lineal.pkl', 'rb') as f:
#     modelo_cargado = pickle.load(f)
"""
    return code


def generate_logistic_regression_code(feature_names, class_names):
    """Generate complete Python code for logistic regression."""
    feature_names_str = str(feature_names)
    class_names_str = str(class_names)

    code = f"""# Código completo para Regresión Logística
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CARGAR Y PREPARAR LOS DATOS
# Reemplaza esta sección con tu método de carga de datos
# df = pd.read_csv('tu_archivo.csv')  # Cargar desde CSV
# O usa datos de ejemplo:

# Datos de ejemplo (reemplaza con tus datos reales)
# X = df[{feature_names_str}]  # Características
# y = df['variable_objetivo']  # Variable objetivo (0, 1, 2, ...)

# 2. DIVIDIR LOS DATOS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. CREAR Y ENTRENAR EL MODELO
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 4. HACER PREDICCIONES
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 5. EVALUAR EL MODELO
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("=== RESULTADOS DEL MODELO ===")
print(f"Accuracy (Entrenamiento): {{accuracy_train:.4f}}")
print(f"Accuracy (Prueba): {{accuracy_test:.4f}}")

# Reporte detallado de clasificación
class_names = {class_names_str}
print("\\n=== REPORTE DE CLASIFICACIÓN ===")
print(classification_report(y_test, y_pred_test, target_names=class_names))

# 6. MATRIZ DE CONFUSIÓN
plt.figure(figsize=(12, 5))

# Gráfico 1: Matriz de Confusión
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')

# Gráfico 2: Importancia de características (coeficientes)
plt.subplot(1, 2, 2)
feature_names = {feature_names_str}
coef = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
colors = ['red' if x < 0 else 'blue' for x in coef]
plt.barh(range(len(coef)), coef, color=colors, alpha=0.7)
plt.yticks(range(len(coef)), feature_names)
plt.xlabel('Coeficientes')
plt.title('Importancia de Características')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.8)
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# 7. MOSTRAR COEFICIENTES DETALLADOS
print("\\n=== COEFICIENTES DEL MODELO ===")
for i, coef in enumerate(coef):
    effect = "aumenta" if coef > 0 else "disminuye"
    print(f"{{feature_names[i]}}: {{coef:.4f}} ({{effect}} la probabilidad)")
print(f"Intercepto: {{model.intercept_[0]:.4f}}")

# 8. FUNCIÓN PARA NUEVAS PREDICCIONES
def predecir_nueva_muestra(nuevo_ejemplo):
    \"\"\"
    Función para hacer predicciones con nuevos datos.
    
    Parámetros:
    nuevo_ejemplo: lista con valores para cada característica
                  en el orden: {feature_names_str}
    
    Retorna:
    prediccion: clase predicha
    probabilidades: probabilidades para cada clase
    \"\"\"
    nuevo_ejemplo = np.array(nuevo_ejemplo).reshape(1, -1)
    prediccion = model.predict(nuevo_ejemplo)[0]
    probabilidades = model.predict_proba(nuevo_ejemplo)[0]
    
    clase_predicha = class_names[prediccion]
    return clase_predicha, probabilidades

# Ejemplo de uso para nuevas predicciones:
# nuevo_ejemplo = [valor1, valor2, valor3, ...]  # Reemplaza con tus valores
# clase, probas = predecir_nueva_muestra(nuevo_ejemplo)
# print(f"Clase predicha: {{clase}}")
# print("Probabilidades por clase:")
# for i, prob in enumerate(probas):
#     print(f"  {{class_names[i]}}: {{prob:.4f}} ({{prob*100:.1f}}%)")

# 9. GUARDAR EL MODELO (OPCIONAL)
import pickle

# Guardar modelo
with open('modelo_regresion_logistica.pkl', 'wb') as f:
    pickle.dump(model, f)

# Cargar modelo guardado
# with open('modelo_regresion_logistica.pkl', 'rb') as f:
#     modelo_cargado = pickle.load(f)

# 10. FUNCIÓN PARA INTERPRETAR PROBABILIDADES
def interpretar_prediccion(nuevo_ejemplo, umbral_confianza=0.8):
    \"\"\"
    Interpreta una predicción mostrando la confianza del modelo.
    \"\"\"
    clase, probabilidades = predecir_nueva_muestra(nuevo_ejemplo)
    max_prob = max(probabilidades)
    
    print(f"Predicción: {{clase}}")
    print(f"Confianza: {{max_prob:.4f}} ({{max_prob*100:.1f}}%)")
    
    if max_prob >= umbral_confianza:
        print("✅ Alta confianza en la predicción")
    elif max_prob >= 0.6:
        print("⚠️ Confianza moderada en la predicción")
    else:
        print("❌ Baja confianza en la predicción")
    
    return clase, max_prob

# Ejemplo de interpretación:
# interpretar_prediccion([valor1, valor2, valor3, ...])
"""
    return code


if __name__ == "__main__":
    main()
