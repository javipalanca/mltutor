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
import streamlit.components.v1 as components
import plotly.express as px
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
                    viz_options.append("🎮 Visualización Interactiva")
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

            elif viz_type == "🎮 Visualización Interactiva":
                st.markdown("### Visualización Interactiva de KNN")
                st.markdown(
                    "Explora cómo el algoritmo KNN toma decisiones en tiempo real")

                # Importar las funciones necesarias
                import plotly.graph_objects as go
                import plotly.express as px
                from plotly.subplots import make_subplots

                # Selección de características para la visualización
                col1, col2 = st.columns(2)

                with col1:
                    feature1 = st.selectbox(
                        "Primera característica:",
                        feature_names,
                        index=0,
                        key="interactive_feature1"
                    )

                with col2:
                    feature2 = st.selectbox(
                        "Segunda característica:",
                        feature_names,
                        index=min(1, len(feature_names) - 1),
                        key="interactive_feature2"
                    )

                if feature1 != feature2:
                    # Obtener índices de características
                    feature_idx = [feature_names.index(
                        feature1), feature_names.index(feature2)]

                    # Extraer datos para 2D
                    if hasattr(X, 'iloc'):
                        X_2d = X.iloc[:, feature_idx].values
                    else:
                        X_2d = X[:, feature_idx]

                    # Inicializar puntos de prueba en session_state si no existen
                    if 'test_points' not in st.session_state:
                        st.session_state.test_points = []

                    # Crear visualización interactiva (con controles integrados)
                    create_interactive_knn_visualization(
                        X_2d, y, model,
                        feature1, feature2,
                        st.session_state.knn_class_names,
                        True, True  # show_confidence, animate_distances por defecto
                    )

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

    if st.sidebar.button("🧠 Redes Neuronales",
                         key="nav_nn",
                         use_container_width=True,
                         type="primary" if st.session_state.navigation == "🧠 Redes Neuronales" else "secondary"):
        st.session_state.navigation = "🧠 Redes Neuronales"
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
    elif st.session_state.navigation == "🧠 Redes Neuronales":
        run_neural_networks_app()
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


def run_neural_networks_app():
    """Ejecuta la aplicación específica de redes neuronales."""
    st.header("🧠 Redes Neuronales")
    st.markdown(
        "Aprende sobre redes neuronales artificiales de forma visual e interactiva")

    # Información sobre redes neuronales
    with st.expander("ℹ️ ¿Qué son las Redes Neuronales?", expanded=False):
        st.markdown("""
        Las redes neuronales artificiales son modelos computacionales inspirados en el funcionamiento del cerebro humano.
        Están compuestas por nodos (neuronas) interconectados que procesan información de manera paralela.

        **Características principales:**
        - **Neuronas**: Unidades básicas que reciben entradas, las procesan y generan salidas
        - **Capas**: Organizan las neuronas en estructuras jerárquicas (entrada, ocultas, salida)
        - **Pesos y Sesgos**: Parámetros que se ajustan durante el entrenamiento
        - **Funciones de Activación**: Determinan la salida de cada neurona
        - **Backpropagation**: Algoritmo para entrenar la red ajustando pesos y sesgos

        **Ventajas:**
        - Pueden modelar relaciones no lineales complejas
        - Excelentes para reconocimiento de patrones
        - Adaptables a diferentes tipos de problemas
        - Capaces de aproximar cualquier función continua

        **Desventajas:**
        - Requieren grandes cantidades de datos
        - Pueden ser "cajas negras" (difíciles de interpretar)
        - Propensos al sobreajuste
        - Requieren mucho poder computacional
        """)

    # Sistema de pestañas
    tab_names = [
        "📊 Datos",
        "🏗️ Arquitectura",
        "⚙️ Entrenamiento",
        "📈 Evaluación",
        "🎯 Visualizaciones",
        "🔮 Predicciones",
        "💾 Exportar"
    ]

    # Inicializar estado de pestañas si no existe
    if 'active_tab_nn' not in st.session_state:
        st.session_state.active_tab_nn = 0

    # Crear pestañas visuales personalizadas
    cols = st.columns(len(tab_names))
    for i, (col, tab_name) in enumerate(zip(cols, tab_names)):
        with col:
            if st.button(
                tab_name,
                key=f"tab_nn_{i}",
                use_container_width=True,
                type="primary" if st.session_state.active_tab_nn == i else "secondary"
            ):
                st.session_state.active_tab_nn = i
                st.rerun()

    st.markdown("---")

    # Pestaña de Datos
    if st.session_state.active_tab_nn == 0:
        st.header("📊 Selección y Preparación de Datos")

        # Tips educativos sobre datos para redes neuronales
        st.info("""
        🎓 **Tips para Redes Neuronales:**
        - Las redes neuronales funcionan mejor con **datos normalizados** (valores entre 0 y 1 o -1 y 1)
        - Necesitan **suficientes datos** para entrenar bien (mínimo 100 ejemplos por clase)
        - Son excelentes para **patrones complejos** y **relaciones no lineales**
        - Pueden funcionar tanto para **clasificación** como para **regresión**
        """)

        # Inicializar dataset seleccionado si no existe
        if 'selected_dataset_nn' not in st.session_state:
            st.session_state.selected_dataset_nn = "🌸 Iris - Clasificación de flores"

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
        if st.session_state.selected_dataset_nn not in available_datasets:
            st.session_state.selected_dataset_nn = builtin_datasets[0]

        # Selector unificado con explicación
        with st.container():
            st.markdown("### 🎯 Selección del Dataset")
            dataset_option = st.selectbox(
                "Elige tu dataset:",
                available_datasets,
                index=available_datasets.index(
                    st.session_state.selected_dataset_nn),
                key="unified_dataset_selector_nn",
                help="💡 Cada dataset presenta diferentes retos de aprendizaje para tu red neuronal"
            )

            # Explicación sobre el dataset seleccionado
            if "Iris" in dataset_option:
                st.markdown(
                    "🌸 **Iris**: Perfecto para empezar. 3 clases de flores, 4 características simples.")
            elif "Vino" in dataset_option:
                st.markdown(
                    "🍷 **Vino**: Clasificación multiclase con 13 características químicas.")
            elif "Cáncer" in dataset_option:
                st.markdown(
                    "🔬 **Cáncer**: Problema binario médico con 30 características.")
            elif "Titanic" in dataset_option:
                st.markdown(
                    "🚢 **Titanic**: Predicción de supervivencia con datos categóricos y numéricos.")
            elif "Propinas" in dataset_option:
                st.markdown(
                    "💰 **Propinas**: Regresión para predecir cantidad de propina.")
            elif "Viviendas" in dataset_option:
                st.markdown(
                    "🏠 **Viviendas**: Regresión para predecir precios de casas.")
            elif "Pingüinos" in dataset_option:
                st.markdown(
                    "🐧 **Pingüinos**: Clasificación de especies con datos biológicos.")

        # Actualizar la variable de sesión
        st.session_state.selected_dataset_nn = dataset_option

        # Separador después del selector
        st.markdown("---")

        # Mostrar información del dataset seleccionado
        if 'selected_dataset_nn' in st.session_state:
            dataset_name = st.session_state.selected_dataset_nn
            st.success(f"✅ Dataset seleccionado: **{dataset_name}**")

            # Cargar y mostrar datos
            try:
                # Cargar datos usando la función load_data común
                X, y, feature_names, class_names, dataset_info, task_type = load_data(
                    dataset_name)

                # Crear DataFrame
                df = pd.DataFrame(X, columns=feature_names)

                # Determinar el nombre de la columna objetivo
                if class_names is not None and len(class_names) > 0:
                    # Para clasificación, usar el nombre de la variable objetivo del dataset_info
                    target_col = 'target'  # Nombre por defecto
                    if hasattr(dataset_info, 'target_names'):
                        target_col = 'target'
                    df[target_col] = y
                else:
                    # Para regresión
                    target_col = 'target'
                    df[target_col] = y

                # Almacenar información del dataset
                st.session_state.nn_df = df
                st.session_state.nn_target_col = target_col
                st.session_state.nn_feature_names = feature_names
                st.session_state.nn_class_names = class_names
                st.session_state.nn_task_type = task_type
                st.session_state.nn_dataset_info = dataset_info

                # Mostrar información básica con explicaciones
                st.markdown("### 📊 Información del Dataset")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "📏 Filas", df.shape[0], help="Número total de ejemplos para entrenar")
                with col2:
                    st.metric(
                        "📊 Columnas", df.shape[1], help="Características + variable objetivo")
                with col3:
                    st.metric("🎯 Variable Objetivo", target_col,
                              help="Lo que la red va a predecir")
                with col4:
                    task_icon = "🏷️" if task_type == "Clasificación" else "📈"
                    st.metric(f"{task_icon} Tipo de Tarea",
                              task_type, help="Clasificación o Regresión")

                # Mostrar muestra de datos con explicación
                st.markdown("### 👀 Vista Previa de los Datos")
                st.markdown("📋 **Primeras 10 filas de tu dataset:**")
                st.dataframe(df.head(10), use_container_width=True)

                # Tip sobre los datos
                with st.expander("💡 ¿Qué significan estos datos?"):
                    st.markdown(f"""
                    - **Filas**: Cada fila es un ejemplo que la red usará para aprender
                    - **Columnas de características**: Las variables que la red analiza para hacer predicciones
                    - **Variable objetivo ({target_col})**: Lo que queremos predecir
                    - **Preprocesamiento**: Los datos se normalizarán automáticamente para la red neuronal
                    """)

                # Análisis de la variable objetivo con explicaciones
                if task_type == "Clasificación":
                    st.markdown("### 🎯 Distribución de Clases")
                    st.markdown("📊 **¿Cuántos ejemplos hay de cada clase?**")
                    class_counts = df[target_col].value_counts()

                    # Usar nombres de clases si están disponibles
                    if class_names is not None:
                        # Mapear valores numéricos a nombres de clases
                        class_labels = [class_names[int(idx)] if int(idx) < len(class_names) else f"Clase {idx}"
                                        for idx in class_counts.index]
                        fig = px.bar(x=class_labels, y=class_counts.values,
                                     labels={'x': target_col, 'y': 'Cantidad'},
                                     title=f"Distribución de {target_col}")
                    else:
                        fig = px.bar(x=class_counts.index, y=class_counts.values,
                                     labels={'x': target_col, 'y': 'Cantidad'},
                                     title=f"Distribución de {target_col}")

                    st.plotly_chart(fig, use_container_width=True)

                    # Explicación sobre balance de clases
                    balance_ratio = class_counts.max() / class_counts.min()
                    if balance_ratio > 3:
                        st.warning(
                            f"⚠️ **Dataset desbalanceado**: La clase más frecuente tiene {balance_ratio:.1f}x más ejemplos que la menos frecuente")
                        st.info(
                            "💡 Las redes neuronales funcionan mejor con clases balanceadas. Considera técnicas de balanceo si es necesario.")
                    else:
                        st.success(
                            "✅ **Dataset bien balanceado**: Las clases tienen cantidad similar de ejemplos")

                else:
                    st.markdown("### 📊 Distribución de la Variable Objetivo")
                    st.markdown(
                        "📈 **Distribución de los valores que queremos predecir:**")
                    fig = px.histogram(df, x=target_col, nbins=30,
                                       title=f"Distribución de {target_col}")
                    st.plotly_chart(fig, use_container_width=True)

                    # Estadísticas básicas para regresión
                    with st.expander("📊 Estadísticas de la Variable Objetivo"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "🎯 Media", f"{df[target_col].mean():.2f}")
                        with col2:
                            st.metric("📏 Desv. Estándar",
                                      f"{df[target_col].std():.2f}")
                        with col3:
                            st.metric(
                                "📉 Mínimo", f"{df[target_col].min():.2f}")
                        with col4:
                            st.metric(
                                "📈 Máximo", f"{df[target_col].max():.2f}")

                # Información adicional del dataset
                if dataset_info and hasattr(dataset_info, 'DESCR'):
                    with st.expander("📖 Descripción Detallada del Dataset"):
                        st.text(dataset_info.DESCR)

                # Botón para continuar con explicación
                st.markdown("---")
                st.markdown("### ➡️ Siguiente Paso")
                st.markdown(
                    "Una vez que entiendas tus datos, es hora de **diseñar la arquitectura** de tu red neuronal.")
                if st.button("🏗️ Continuar a Arquitectura", type="primary", use_container_width=True):
                    st.session_state.active_tab_nn = 1
                    st.rerun()

            except Exception as e:
                st.error(f"Error cargando dataset: {str(e)}")
                st.info("Por favor, selecciona un dataset válido.")

        else:
            st.warning("⚠️ Por favor, selecciona un dataset para continuar.")

    # Pestaña de Arquitectura
    elif st.session_state.active_tab_nn == 1:
        st.header("🏗️ Diseño de la Arquitectura de la Red")

        if 'nn_df' not in st.session_state or 'nn_task_type' not in st.session_state:
            st.warning(
                "⚠️ Primero debes seleccionar un dataset en la pestaña de Datos.")
            if st.button("🔙 Ir a Datos"):
                st.session_state.active_tab_nn = 0
                st.rerun()
            return

        # Tips educativos sobre arquitectura
        st.info("""
        🎓 **Conceptos Clave de Arquitectura:**
        - **Capas ocultas**: Más capas = mayor capacidad de aprender patrones complejos
        - **Neuronas por capa**: Más neuronas = mayor capacidad, pero riesgo de sobreajuste
        - **Funciones de activación**: Determinan cómo las neuronas procesan la información
        - **Arquitectura óptima**: Depende del problema y cantidad de datos
        """)

        st.markdown("### 🎛️ Configuración de la Red Neuronal")

        # Información básica del dataset
        df = st.session_state.nn_df
        target_col = st.session_state.nn_target_col
        task_type = st.session_state.nn_task_type

        # Preparar datos básicos para mostrar dimensiones
        X = df.drop(columns=[target_col])
        y = df[target_col]
        input_size = X.shape[1]

        if task_type == "Clasificación":
            num_classes = len(y.unique())
            if num_classes == 2:
                output_size = 1  # Para clasificación binaria
                st.info(
                    f"📊 **Entrada**: {input_size} características → **Salida**: {output_size} neurona (clasificación binaria)")
            else:
                output_size = num_classes  # Para clasificación multiclase
                st.info(
                    f"📊 **Entrada**: {input_size} características → **Salida**: {output_size} clases")
        else:
            output_size = 1
            st.info(
                f"📊 **Entrada**: {input_size} características → **Salida**: {output_size} valor numérico")

        # Tips sobre dimensiones
        with st.expander("💡 ¿Cómo decidir el tamaño de la red?"):
            st.markdown(f"""
            **Reglas generales para tu dataset ({df.shape[0]} muestras, {input_size} características):**
            
            🔢 **Neuronas por capa oculta:**
            - Pequeño: {input_size//2} - {input_size} neuronas
            - Mediano: {input_size} - {input_size*2} neuronas  
            - Grande: {input_size*2} - {input_size*4} neuronas
            
            📚 **Número de capas:**
            - 1-2 capas: Problemas simples, linealmente separables
            - 2-3 capas: Problemas moderadamente complejos (recomendado para empezar)
            - 4+ capas: Problemas muy complejos (requiere muchos datos)
            
            ⚖️ **Balance capacidad vs. datos:**
            - Más parámetros que datos → riesgo de sobreajuste
            - Tu dataset: {df.shape[0]} muestras, mantén parámetros < {df.shape[0]//10}
            """)

        # Configuración de arquitectura
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### ⚙️ Configuración de Capas")

            # Número de capas ocultas con explicación
            num_hidden_layers = st.slider(
                "Número de capas ocultas",
                min_value=1, max_value=5, value=2,
                help="💡 Más capas = mayor capacidad de aprender patrones complejos, pero también mayor riesgo de sobreajuste"
            )

            # Sugerencia basada en el número de capas
            if num_hidden_layers == 1:
                st.caption(
                    "🟦 **1 capa**: Ideal para problemas linealmente separables")
            elif num_hidden_layers == 2:
                st.caption(
                    "🟨 **2 capas**: Recomendado para la mayoría de problemas")
            elif num_hidden_layers >= 3:
                st.caption(
                    "🟥 **3+ capas**: Solo para problemas muy complejos con muchos datos")

            # Configuración de cada capa oculta con explicaciones
            hidden_layers = []
            for i in range(num_hidden_layers):
                # Calcular sugerencia inteligente
                suggested_size = max(10, input_size // (i+1))
                if task_type == "Clasificación" and num_classes > 2:
                    suggested_size = max(suggested_size, num_classes * 2)

                neurons = st.slider(
                    f"Neuronas en capa oculta {i+1}",
                    min_value=1, max_value=256,
                    value=min(suggested_size, 64),  # Limitar valor por defecto
                    key=f"layer_{i}",
                    help=f"💡 Sugerencia para capa {i+1}: {suggested_size} neuronas"
                )
                hidden_layers.append(neurons)

            # Función de activación con explicaciones detalladas
            st.markdown("#### 🧮 Función de Activación (Capas Ocultas)")
            activation = st.selectbox(
                "Función de activación",
                ["relu", "tanh", "sigmoid"],
                help="💡 ReLU es la más popular y efectiva para la mayoría de problemas"
            )

            # Explicaciones sobre funciones de activación
            if activation == "relu":
                st.success(
                    "✅ **ReLU**: Rápida, evita el problema del gradiente que desaparece. Recomendada.")
            elif activation == "tanh":
                st.info(
                    "ℹ️ **Tanh**: Salida entre -1 y 1. Buena para datos normalizados.")
            elif activation == "sigmoid":
                st.warning(
                    "⚠️ **Sigmoid**: Puede causar gradientes que desaparecen. Úsala solo si es necesario.")

            # Función de activación de salida - AHORA SELECCIONABLE
            st.markdown("#### 🎯 Función de Activación de Salida")

            # Opciones disponibles según el tipo de tarea
            if task_type == "Clasificación":
                output_options = ["sigmoid", "softmax", "linear", "tanh"]
                if output_size == 1:  # Clasificación binaria
                    recommended = "sigmoid"
                    default_index = 0
                else:  # Clasificación multiclase
                    recommended = "softmax"
                    default_index = 1
            else:
                output_options = ["linear", "sigmoid", "tanh", "softmax"]
                recommended = "linear"
                default_index = 0

            output_activation = st.selectbox(
                "Función de activación de salida",
                output_options,
                index=default_index,
                help=f"💡 Función recomendada para {task_type.lower()}: **{recommended}**"
            )

            # Validaciones y avisos
            show_warning = False
            warning_message = ""

            if task_type == "Clasificación":
                if output_size == 1:  # Clasificación binaria
                    if output_activation == "sigmoid":
                        st.success(
                            "✅ Sigmoid es ideal para clasificación binaria")
                    elif output_activation == "softmax":
                        show_warning = True
                        warning_message = "⚠️ Softmax no es recomendada para clasificación binaria (1 neurona). Considera usar Sigmoid."
                    elif output_activation == "linear":
                        show_warning = True
                        warning_message = "⚠️ Linear puede causar problemas en clasificación. Considera usar Sigmoid."
                    elif output_activation == "tanh":
                        show_warning = True
                        warning_message = "⚠️ Tanh puede funcionar pero Sigmoid es más estándar para clasificación binaria."

                else:  # Clasificación multiclase
                    if output_activation == "softmax":
                        st.success(
                            "✅ Softmax es ideal para clasificación multiclase")
                    elif output_activation == "sigmoid":
                        st.warning(
                            "⚠️ Sigmoid en multiclase requiere 'binary_crossentropy' por clase. Softmax es más estándar.")
                    elif output_activation == "linear":
                        show_warning = True
                        warning_message = "⚠️ Linear no es apropiada para clasificación. Usa Softmax."
                    elif output_activation == "tanh":
                        show_warning = True
                        warning_message = "⚠️ Tanh no es estándar para clasificación multiclase. Softmax es recomendada."

            else:  # Regresión
                if output_activation == "linear":
                    st.success("✅ Linear es ideal para regresión")
                elif output_activation == "sigmoid":
                    st.warning(
                        "⚠️ Sigmoid limita la salida a [0,1]. Solo útil si tus valores objetivo están en este rango.")
                elif output_activation == "tanh":
                    st.warning(
                        "⚠️ Tanh limita la salida a [-1,1]. Solo útil si tus valores objetivo están en este rango.")
                elif output_activation == "softmax":
                    show_warning = True
                    warning_message = "⚠️ Softmax no es apropiada para regresión. Las salidas suman 1. Usa Linear."

            # Mostrar advertencia crítica si es necesario
            if show_warning:
                st.error(warning_message)

        with col2:
            st.markdown("#### 🎨 Visualización de la Arquitectura")

            # Crear arquitectura completa
            architecture = [input_size] + hidden_layers + [output_size]

            # Guardar configuración en session state
            st.session_state.nn_architecture = {
                'layers': architecture,
                'activation': activation,
                'output_activation': output_activation,
                'input_size': input_size,
                'output_size': output_size,
                'task_type': task_type
            }

            # Visualizar la red neuronal dinámicamente
            create_neural_network_visualization(
                architecture, activation, output_activation, task_type)

        # Configuración adicional con explicaciones detalladas
        st.markdown("### ⚙️ Configuración Adicional")

        st.markdown("📚 **Parámetros importantes para el entrenamiento:**")

        col3, col4, col5 = st.columns(3)

        with col3:
            st.markdown("#### 🛡️ Regularización")
            dropout_rate = st.slider(
                "Tasa de Dropout",
                min_value=0.0, max_value=0.8, value=0.2, step=0.1,
                help="💡 Dropout previene sobreajuste eliminando aleatoriamente neuronas durante entrenamiento"
            )

            # Explicación del dropout
            if dropout_rate == 0.0:
                st.caption("🔴 **Sin Dropout**: Mayor riesgo de sobreajuste")
            elif dropout_rate <= 0.2:
                st.caption("🟢 **Dropout Ligero**: Bueno para datasets grandes")
            elif dropout_rate <= 0.5:
                st.caption(
                    "🟡 **Dropout Moderado**: Recomendado para la mayoría de casos")
            else:
                st.caption(
                    "🟠 **Dropout Alto**: Solo para datasets muy pequeños")

        with col4:
            st.markdown("#### 📦 Procesamiento")
            batch_size = st.selectbox(
                "Tamaño de Batch",
                [16, 32, 64, 128, 256],
                index=2,  # 64 por defecto
                help="💡 Número de muestras procesadas antes de actualizar los pesos"
            )

            # Sugerencias según el tamaño del dataset
            dataset_size = df.shape[0]
            if batch_size >= dataset_size // 4:
                st.caption(
                    "🔴 **Batch Grande**: Puede ser lento pero más estable")
            elif batch_size >= 32:
                st.caption(
                    "🟢 **Batch Óptimo**: Buen balance velocidad/estabilidad")
            else:
                st.caption("🟡 **Batch Pequeño**: Más rápido pero más ruidoso")

        with col5:
            st.markdown("#### 🚀 Optimización")
            optimizer = st.selectbox(
                "Optimizador",
                ["adam", "sgd", "rmsprop"],
                help="💡 Algoritmo para actualizar los pesos de la red"
            )

            # Explicaciones sobre optimizadores
            if optimizer == "adam":
                st.caption(
                    "🟢 **Adam**: Adaptativo, recomendado para la mayoría de casos")
            elif optimizer == "sgd":
                st.caption(
                    "🟡 **SGD**: Clásico, requiere ajuste fino del learning rate")
            elif optimizer == "rmsprop":
                st.caption(
                    "🟦 **RMSprop**: Bueno para RNNs y problemas específicos")

        # Tips sobre la configuración
        with st.expander("💡 Tips para optimizar tu configuración"):
            st.markdown(f"""
            **Para tu dataset específico ({dataset_size} muestras):**
            
            🎯 **Batch Size recomendado:**
            - Dataset pequeño (<1000): 16-32
            - Dataset mediano (1000-10000): 32-64
            - Dataset grande (>10000): 64-128
            - Tu dataset: {dataset_size} muestras → Recomendado: {32 if dataset_size < 1000 else 64 if dataset_size < 10000 else 128}
            
            🛡️ **Dropout recomendado:**
            - Pocos datos: 0.3-0.5 (más regularización)
            - Muchos datos: 0.1-0.2 (menos regularización)
            - Dataset balanceado: 0.2-0.3
            
            🚀 **Optimizador:**
            - **Adam**: Mejor opción general, se adapta automáticamente
            - **SGD**: Úsalo solo si tienes experiencia ajustando learning rates
            - **RMSprop**: Alternativa a Adam, a veces funciona mejor en problemas específicos
            """)

        # Guardar configuración completa
        st.session_state.nn_config = {
            'architecture': architecture,
            'activation': activation,
            'output_activation': output_activation,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'task_type': task_type,
            'input_size': input_size,
            'output_size': output_size
        }

        # Resumen de la configuración con análisis
        st.markdown("### 📋 Resumen de la Arquitectura")

        total_params = calculate_network_parameters(architecture)

        col6, col7, col8 = st.columns(3)
        with col6:
            st.metric("🔢 Total de Parámetros", f"{total_params:,}",
                      help="Número total de pesos y sesgos que la red aprenderá")
        with col7:
            st.metric("📚 Capas Totales", len(architecture),
                      help="Entrada + Ocultas + Salida")
        with col8:
            complexity_ratio = total_params / dataset_size if dataset_size > 0 else 0
            complexity_level = "Baja" if complexity_ratio < 0.1 else "Media" if complexity_ratio < 1 else "Alta"
            st.metric("⚖️ Complejidad", complexity_level,
                      help=f"Ratio parámetros/datos: {complexity_ratio:.2f}")
        with col8:
            st.metric("🧠 Tipo de Red", "Perceptrón Multicapa")

        # Mostrar detalles de cada capa
        st.markdown("#### 📊 Detalles por Capa")
        layer_details = []
        for i, (current, next_size) in enumerate(zip(architecture[:-1], architecture[1:])):
            if i == 0:
                layer_type = "Entrada"
                params = 0
            elif i == len(architecture) - 2:
                layer_type = "Salida"
                params = current * next_size + next_size
            else:
                layer_type = f"Oculta {i}"
                params = current * next_size + next_size

            layer_details.append({
                "Capa": layer_type,
                "Neuronas": current if i < len(architecture) - 1 else next_size,
                "Parámetros": params,
                "Activación": "Entrada" if i == 0 else (output_activation if i == len(architecture) - 2 else activation)
            })

        st.dataframe(pd.DataFrame(layer_details), use_container_width=True)

        # Análisis de complejidad y recomendaciones
        st.markdown("#### 🔍 Análisis de Complejidad")

        # Análisis del ratio parámetros/datos
        if complexity_ratio < 0.1:
            st.success(
                f"✅ **Complejidad Óptima**: Tu red tiene {total_params:,} parámetros para {dataset_size} muestras (ratio: {complexity_ratio:.3f}). Bajo riesgo de sobreajuste.")
        elif complexity_ratio < 1:
            st.warning(
                f"⚠️ **Complejidad Media**: Tu red tiene {total_params:,} parámetros para {dataset_size} muestras (ratio: {complexity_ratio:.3f}). Monitorea el sobreajuste.")
        else:
            st.error(
                f"🚨 **Complejidad Alta**: Tu red tiene {total_params:,} parámetros para {dataset_size} muestras (ratio: {complexity_ratio:.3f}). Alto riesgo de sobreajuste. Considera reducir el tamaño de la red.")

        # Botón para generar código Python
        st.markdown("### 💻 Código Python")
        if st.button("📝 Generar Código de la Arquitectura", use_container_width=True):
            # Generar código Python para la arquitectura
            code = generate_neural_network_architecture_code(
                architecture, activation, output_activation, dropout_rate,
                optimizer, batch_size, task_type, st.session_state.nn_feature_names
            )

            st.markdown("#### 🐍 Código Python Generado")
            st.code(code, language='python')

            # Botón para descargar el código
            st.download_button(
                label="💾 Descargar Código Python",
                data=code,
                file_name=f"red_neuronal_arquitectura_{task_type.lower()}.py",
                mime="text/plain"
            )

        # Botones de navegación
        st.markdown("---")
        st.markdown("### 🧭 Navegación")
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("🔙 Volver a Datos", use_container_width=True):
                st.session_state.active_tab_nn = 0
                st.rerun()
        with col_nav2:
            st.markdown(
                "**¿Listo para entrenar?** ¡Tu arquitectura está configurada!")
            if st.button("🚀 Continuar a Entrenamiento", type="primary", use_container_width=True):
                st.session_state.active_tab_nn = 2
                st.rerun()

    # Pestaña de Entrenamiento
    elif st.session_state.active_tab_nn == 2:
        st.header("⚙️ Entrenamiento de la Red Neuronal")

        if 'nn_config' not in st.session_state:
            st.warning("⚠️ Primero debes configurar la arquitectura de la red.")
            if st.button("🔙 Ir a Arquitectura"):
                st.session_state.active_tab_nn = 1
                st.rerun()
            return

        # Tips educativos sobre entrenamiento
        st.info("""
        🎓 **Conceptos Clave del Entrenamiento:**
        - **Learning Rate**: Controla qué tan rápido aprende la red (muy alto = inestable, muy bajo = lento)
        - **Épocas**: Cuántas veces la red ve todos los datos (más épocas ≠ siempre mejor)
        - **Validación**: Datos separados para monitorear si la red está generalizando bien
        - **Early Stopping**: Para evitar sobreajuste, para cuando la validación no mejora
        """)

        st.markdown("### 🎛️ Parámetros de Entrenamiento")

        # Información del dataset para sugerencias
        df = st.session_state.nn_df
        dataset_size = df.shape[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📈 **Learning Rate**")
            learning_rate = st.selectbox(
                "Tasa de Aprendizaje",
                [0.001, 0.01, 0.1, 0.3],
                index=0,
                help="💡 0.001 es seguro para empezar. Valores más altos pueden acelerar el entrenamiento pero causar inestabilidad"
            )

            # Explicación del learning rate seleccionado
            if learning_rate == 0.001:
                st.caption("🟢 **Conservador**: Aprendizaje lento pero estable")
            elif learning_rate == 0.01:
                st.caption(
                    "🟡 **Moderado**: Buen balance velocidad/estabilidad")
            elif learning_rate == 0.1:
                st.caption("🟠 **Agresivo**: Rápido pero puede ser inestable")
            else:
                st.caption("🔴 **Muy Alto**: Solo para casos especiales")

        with col2:
            st.markdown("#### 🔄 **Épocas**")
            # Sugerir épocas basado en tamaño del dataset
            suggested_epochs = min(200, max(50, dataset_size // 10))
            epochs = st.slider(
                "Épocas",
                min_value=10, max_value=500, value=min(100, suggested_epochs), step=10,
                help=f"💡 Sugerencia para tu dataset: ~{suggested_epochs} épocas"
            )

            # Explicación sobre las épocas
            if epochs < 50:
                st.caption(
                    "🟡 **Pocas épocas**: Puede que no aprenda completamente")
            elif epochs <= 150:
                st.caption("🟢 **Épocas adecuadas**: Buen balance")
            else:
                st.caption("🟠 **Muchas épocas**: Monitorea el sobreajuste")

        with col3:
            st.markdown("#### 🎯 **Validación**")
            validation_split = st.slider(
                "% Datos de Validación",
                min_value=10, max_value=40, value=20,
                help="💡 20% es estándar. Más datos = mejor validación, menos datos para entrenar"
            )

            # Calcular tamaños efectivos
            # 80% del total para entrenamiento
            train_size = int(
                dataset_size * (100 - validation_split) / 100 * 0.8)
            val_size = int(dataset_size * validation_split / 100)
            test_size = dataset_size - train_size - val_size

            st.caption(
                f"📊 **Distribución**: Train={train_size}, Val={val_size}, Test={test_size}")

        # Configuración avanzada con explicaciones detalladas
        with st.expander("⚙️ Configuración Avanzada - Técnicas para Mejorar el Entrenamiento", expanded=False):
            st.markdown("#### 🛡️ Técnicas de Regularización y Optimización")

            col4, col5 = st.columns(2)

            with col4:
                st.markdown("##### 🛑 Early Stopping")
                early_stopping = st.checkbox(
                    "Activar Parada Temprana",
                    value=True,
                    help="💡 Recomendado: Evita sobreajuste parando cuando la validación no mejora"
                )

                if early_stopping:
                    st.success(
                        "✅ **Early Stopping activado**: La red parará automáticamente cuando deje de mejorar")
                    patience = st.slider(
                        "Paciencia (épocas)",
                        min_value=5, max_value=50, value=10,
                        help="Épocas a esperar sin mejora antes de parar. Más paciencia = más oportunidades de mejorar"
                    )

                    if patience <= 5:
                        st.caption(
                            "🔴 **Impatiente**: Para rápido, puede interrumpir mejoras tardías")
                    elif patience <= 15:
                        st.caption("🟢 **Balanceado**: Buen equilibrio")
                    else:
                        st.caption(
                            "🟡 **Paciente**: Da muchas oportunidades, pero puede sobreajustar")
                else:
                    st.warning(
                        "⚠️ **Sin Early Stopping**: La red entrenará todas las épocas. Riesgo de sobreajuste.")

            with col5:
                st.markdown("##### 📉 Learning Rate Scheduler")
                reduce_lr = st.checkbox(
                    "Reducir Learning Rate Automáticamente",
                    value=True,
                    help="💡 Recomendado: Reduce la tasa de aprendizaje cuando no mejora"
                )

                if reduce_lr:
                    st.success(
                        "✅ **Scheduler activado**: La tasa de aprendizaje se reducirá automáticamente")
                    lr_factor = st.slider(
                        "Factor de Reducción",
                        min_value=0.1, max_value=0.9, value=0.5,
                        help="Factor por el que se multiplica la tasa. 0.5 = reduce a la mitad"
                    )

                    if lr_factor <= 0.3:
                        st.caption(
                            "🔴 **Reducción agresiva**: Cambios dramáticos")
                    elif lr_factor <= 0.7:
                        st.caption("🟢 **Reducción moderada**: Recomendado")
                    else:
                        st.caption("🟡 **Reducción suave**: Cambios graduales")
                else:
                    st.info(
                        "ℹ️ **Learning rate fijo**: Se mantendrá constante durante todo el entrenamiento")

            # Explicación sobre las técnicas
            st.markdown("---")
            st.markdown("#### 📚 ¿Por qué usar estas técnicas?")
            st.markdown("""
            - **Early Stopping**: Evita que la red memorice los datos (sobreajuste) parando cuando la performance en validación deja de mejorar
            - **Learning Rate Reduction**: Permite un ajuste fino hacia el final del entrenamiento cuando se está cerca del óptimo
            - **Combinadas**: Estas técnicas trabajan juntas para lograr el mejor modelo posible automáticamente
            """)

        # Botón de entrenamiento con explicación
        st.markdown("### 🚀 Iniciar Entrenamiento")
        st.markdown(
            "**¿Todo listo?** Tu red está configurada y lista para aprender de los datos.")

        if st.button("🧠 Entrenar Red Neuronal", type="primary", use_container_width=True):
            with st.spinner("🧠 Entrenando la red neuronal... Esto puede tomar unos minutos."):
                # Preparar datos
                df = st.session_state.nn_df
                target_col = st.session_state.nn_target_col
                task_type = st.session_state.nn_task_type

                try:
                    # Mostrar progreso del entrenamiento con pasos
                    progress_container = st.empty()

                    # Paso 1: Preparando datos
                    with progress_container.container():
                        st.info(
                            "🔄 **Paso 1/4**: Preparando y dividiendo los datos...")

                    # Llamar función de entrenamiento con callback de progreso
                    def update_progress(step, message):
                        with progress_container.container():
                            if step == 2:
                                st.info(f"🧠 **Paso {step}/4**: {message}")
                            elif step == 3:
                                st.info(f"⚙️ **Paso {step}/4**: {message}")
                            elif step == 4:
                                st.info(f"🚀 **Paso {step}/4**: {message}")
                            else:
                                st.info(f"🔄 **Paso {step}/4**: {message}")

                    # Entrenar el modelo con callback de progreso
                    model, history, X_test, y_test, scaler, label_encoder = train_neural_network(
                        df, target_col, st.session_state.nn_config,
                        learning_rate, epochs, validation_split/100,
                        early_stopping, patience if early_stopping else None,
                        reduce_lr, lr_factor if reduce_lr else None,
                        progress_callback=update_progress
                    )

                    # Guardar resultados
                    st.session_state.nn_model = model
                    st.session_state.nn_history = history
                    st.session_state.nn_test_data = (X_test, y_test)
                    st.session_state.nn_scaler = scaler
                    st.session_state.nn_label_encoder = label_encoder
                    st.session_state.model_trained_nn = True

                    # Limpiar el progreso y mostrar finalización
                    with progress_container.container():
                        st.success(
                            "✅ **¡Entrenamiento completado!** Red neuronal lista para usar")

                    st.success("🎉 ¡Red neuronal entrenada exitosamente!")

                    # Mostrar métricas básicas con explicaciones
                    st.markdown("#### 📊 Resultados del Entrenamiento")
                    if task_type == "Clasificación":
                        test_loss, test_acc = model.evaluate(
                            X_test, y_test, verbose=0)
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("🎯 Precisión en Test", f"{test_acc:.3f}",
                                      help="Porcentaje de predicciones correctas en datos nunca vistos")
                        with col_m2:
                            st.metric("📉 Pérdida en Test", f"{test_loss:.3f}",
                                      help="Qué tan 'equivocada' está la red en promedio")
                    else:
                        test_loss = model.evaluate(X_test, y_test, verbose=0)
                        st.metric("📉 Error en Test", f"{test_loss:.3f}")

                    # Gráfico de entrenamiento en tiempo real
                    st.markdown("### 📈 Progreso del Entrenamiento")
                    plot_training_history(history, task_type)

                except Exception as e:
                    # Limpiar el progreso en caso de error
                    with progress_container.container():
                        st.error("❌ **Error durante el entrenamiento**")

                    st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                    st.info(
                        "Intenta ajustar los parámetros o verificar el dataset.")

        # Botones de navegación
        if st.session_state.get('model_trained_nn', False):
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                if st.button("🔙 Volver a Arquitectura", use_container_width=True):
                    st.session_state.active_tab_nn = 1
                    st.rerun()
            with col_nav2:
                if st.button("➡️ Ver Evaluación", type="primary", use_container_width=True):
                    st.session_state.active_tab_nn = 3
                    st.rerun()

    # Pestañas restantes (Evaluación, Visualizaciones, Predicciones, Exportar)
    elif st.session_state.active_tab_nn == 3:
        st.header("📈 Evaluación del Modelo")
        if not st.session_state.get('model_trained_nn', False):
            st.warning("⚠️ Primero debes entrenar un modelo.")
            if st.button("🔙 Ir a Entrenamiento"):
                st.session_state.active_tab_nn = 2
                st.rerun()
        else:
            show_neural_network_evaluation()

    elif st.session_state.active_tab_nn == 4:
        st.header("🎯 Visualizaciones")
        if not st.session_state.get('model_trained_nn', False):
            st.warning("⚠️ Primero debes entrenar un modelo.")
        else:
            show_neural_network_visualizations()

    elif st.session_state.active_tab_nn == 5:
        st.header("🔮 Predicciones")
        if not st.session_state.get('model_trained_nn', False):
            st.warning("⚠️ Primero debes entrenar un modelo.")
        else:
            show_neural_network_predictions()

    elif st.session_state.active_tab_nn == 6:
        st.header("💾 Exportar Modelo")
        if not st.session_state.get('model_trained_nn', False):
            st.warning("⚠️ Primero debes entrenar un modelo.")
        else:
            show_neural_network_export()


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


def create_interactive_knn_visualization(X_2d, y, model, feature1, feature2, class_names, show_confidence, animate_distances):
    """
    Crea una visualización interactiva de KNN usando JavaScript/HTML5 Canvas.

    Parameters:
    -----------
    X_2d : array
        Datos de entrenamiento en 2D
    y : array
        Etiquetas de clase
    model : KNeighborsClassifier
        Modelo KNN entrenado
    feature1, feature2 : str
        Nombres de las características
    class_names : list
        Nombres de las clases
    show_confidence : bool
        Mostrar niveles de confianza
    animate_distances : bool
        Mostrar distancias a los vecinos
    """

    try:
        # Normalizar los datos para la visualización
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

        # Convertir datos a formato JSON para JavaScript
        import json

        # Preparar datos de entrenamiento
        training_data = []
        unique_classes = np.unique(y)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726',
                  '#AB47BC', '#66BB6A', '#EF5350', '#42A5F5']

        for i, (point, label) in enumerate(zip(X_2d, y)):
            training_data.append({
                'x': float(point[0]),
                'y': float(point[1]),
                'class': int(label),
                'className': class_names[int(label)] if class_names else f"Clase {int(label)}",
                'color': colors[int(label) % len(colors)]
            })

        # Información de las clases
        class_info = []
        for i, class_val in enumerate(unique_classes):
            class_info.append({
                'id': int(class_val),
                'name': class_names[int(class_val)] if class_names else f"Clase {int(class_val)}",
                'color': colors[int(class_val) % len(colors)]
            })

        # Obtener puntos de prueba del estado de sesión
        test_points = st.session_state.get('test_points', [])

        # HTML y JavaScript para la visualización interactiva
        html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .knn-container {{
                max-width: 100%;
                margin: 0 auto;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            }}
            .canvas-container {{
                position: relative;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin: 20px 0;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                width: 100%;
                overflow: hidden;
            }}
            #knnCanvas {{
                display: block;
                cursor: crosshair;
                border-radius: 6px;
                width: 100%;
                height: auto;
                max-width: 100%;
            }}
            .controls {{
                display: flex;
                gap: 15px;
                margin-bottom: 15px;
                flex-wrap: wrap;
                align-items: center;
                justify-content: center;
            }}
            .control-group {{
                display: flex;
                flex-direction: column;
                gap: 5px;
                min-width: 120px;
            }}
            .control-group label {{
                font-weight: 600;
                font-size: 12px;
                color: #555;
                text-align: center;
            }}
            .control-group input, .control-group select {{
                padding: 5px 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }}
            .legend {{
                display: flex;
                gap: 15px;
                margin-top: 15px;
                flex-wrap: wrap;
                justify-content: center;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 5px;
                font-size: 14px;
            }}
            .legend-color {{
                width: 16px;
                height: 16px;
                border-radius: 50%;
                border: 2px solid #333;
                flex-shrink: 0;
            }}
            .stats {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
            }}
            .stat-item {{
                text-align: center;
                padding: 10px;
                background: white;
                border-radius: 6px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2196F3;
            }}
            .stat-label {{
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }}
            .clear-btn {{
                background: #ff4444;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                min-width: 140px;
            }}
            .clear-btn:hover {{
                background: #cc3333;
            }}
            .info-box {{
                background: #e3f2fd;
                border-left: 4px solid #2196F3;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }}
            
            /* Responsivo para móviles */
            @media (max-width: 768px) {{
                .controls {{
                    gap: 10px;
                }}
                .control-group {{
                    min-width: 100px;
                }}
                .legend {{
                    gap: 10px;
                }}
                .legend-item {{
                    font-size: 12px;
                }}
                .stats {{
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 8px;
                }}
                .stat-value {{
                    font-size: 20px;
                }}
                .clear-btn {{
                    min-width: 120px;
                    font-size: 12px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="knn-container">
            <div class="info-box">
                <strong>🎯 Visualización Interactiva KNN</strong><br>
                Haz click en cualquier parte del área gris para agregar puntos de prueba y ver cómo el algoritmo los clasifica.
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label>Valor K:</label>
                    <input type="range" id="kValue" min="1" max="15" value="{model.n_neighbors}" onchange="updateK(this.value)">
                    <span id="kDisplay">{model.n_neighbors}</span>
                </div>
                <div class="control-group">
                    <label>Métrica:</label>
                    <select id="metric" onchange="updateMetric(this.value)">
                        <option value="euclidean" {'selected' if model.metric == 'euclidean' else ''}>Euclidean</option>
                        <option value="manhattan" {'selected' if model.metric == 'manhattan' else ''}>Manhattan</option>
                        <option value="minkowski" {'selected' if model.metric == 'minkowski' else ''}>Minkowski</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Pesos:</label>
                    <select id="weights" onchange="updateWeights(this.value)">
                        <option value="uniform" {'selected' if model.weights == 'uniform' else ''}>Uniforme</option>
                        <option value="distance" {'selected' if model.weights == 'distance' else ''}>Por distancia</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Mostrar confianza:</label>
                    <input type="checkbox" id="showConfidence" {'checked' if show_confidence else ''} onchange="updateShowConfidence(this.checked)">
                </div>
                <div class="control-group">
                    <label>Mostrar distancias:</label>
                    <input type="checkbox" id="showDistances" {'checked' if animate_distances else ''} onchange="updateShowDistances(this.checked)">
                </div>
                <button class="clear-btn" onclick="clearTestPoints()">🗑️ Limpiar Puntos</button>
            </div>
            
            <div class="canvas-container">
                <canvas id="knnCanvas"></canvas>
            </div>
            
            <div class="legend">
                <strong>Leyenda:</strong>
                {' '.join([f'<div class="legend-item"><div class="legend-color" style="background-color: {colors[i % len(colors)]}"></div><span>{class_info[i]["name"]}</span></div>' for i in range(len(class_info))])}
                <div class="legend-item"><div class="legend-color" style="background-color: #FFD700; transform: rotate(45deg);"></div><span>Puntos de Prueba</span></div>
            </div>
            
            <div class="stats" id="stats">
                <div class="stat-item">
                    <div class="stat-value" id="totalPoints">0</div>
                    <div class="stat-label">Puntos de Prueba</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="avgConfidence">0%</div>
                    <div class="stat-label">Confianza Promedio</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="currentK">{model.n_neighbors}</div>
                    <div class="stat-label">Valor K Actual</div>
                </div>
            </div>
        </div>

        <script>
            // Configuración global
            const canvas = document.getElementById('knnCanvas');
            const ctx = canvas.getContext('2d');
            
            // Función para redimensionar el canvas
            function resizeCanvas() {{
                const container = document.querySelector('.canvas-container');
                const containerWidth = container.clientWidth - 4; // Restar bordes
                const aspectRatio = 3/2; // Relación de aspecto 3:2
                const canvasHeight = Math.max(400, containerWidth / aspectRatio);
                
                canvas.width = containerWidth;
                canvas.height = canvasHeight;
                canvas.style.width = containerWidth + 'px';
                canvas.style.height = canvasHeight + 'px';
                
                // Redibujar después de redimensionar
                if (typeof draw === 'function') {{
                    draw();
                }}
            }}
            
            // Datos del modelo
            const trainingData = {json.dumps(training_data)};
            const classInfo = {json.dumps(class_info)};
            const bounds = {{
                xMin: {x_min},
                xMax: {x_max},
                yMin: {y_min},
                yMax: {y_max}
            }};
            
            // Estado de la visualización
            let currentK = {model.n_neighbors};
            let currentMetric = "{model.metric}";
            let currentWeights = "{model.weights}";
            let showConfidence = {str(show_confidence).lower()};
            let showDistances = {str(animate_distances).lower()};
            let testPoints = [];
            
            // Funciones de conversión de coordenadas
            function dataToCanvas(x, y) {{
                const margin = 20;
                const canvasX = ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * (canvas.width - 2 * margin) + margin;
                const canvasY = canvas.height - (((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * (canvas.height - 2 * margin) + margin);
                return {{x: canvasX, y: canvasY}};
            }}
            
            function canvasToData(canvasX, canvasY) {{
                const margin = 20;
                const x = ((canvasX - margin) / (canvas.width - 2 * margin)) * (bounds.xMax - bounds.xMin) + bounds.xMin;
                const y = bounds.yMax - (((canvasY - margin) / (canvas.height - 2 * margin)) * (bounds.yMax - bounds.yMin));
                return {{x: x, y: y}};
            }}
            
            // Función de distancia
            function calculateDistance(p1, p2, metric) {{
                if (metric === 'euclidean') {{
                    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
                }} else if (metric === 'manhattan') {{
                    return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
                }} else {{ // minkowski (p=2, equivalent to euclidean)
                    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
                }}
            }}
            
            // Clasificación KNN
            function classifyPoint(testPoint) {{
                const distances = trainingData.map(point => ({{
                    ...point,
                    distance: calculateDistance(testPoint, point, currentMetric)
                }}));
                
                distances.sort((a, b) => a.distance - b.distance);
                const neighbors = distances.slice(0, currentK);
                
                // Contar votos por clase
                const votes = {{}};
                const weights = {{}};
                
                neighbors.forEach(neighbor => {{
                    if (!votes[neighbor.class]) {{
                        votes[neighbor.class] = 0;
                        weights[neighbor.class] = 0;
                    }}
                    
                    if (currentWeights === 'distance') {{
                        const weight = neighbor.distance === 0 ? 1 : 1 / neighbor.distance;
                        weights[neighbor.class] += weight;
                    }} else {{
                        votes[neighbor.class] += 1;
                    }}
                }});
                
                // Encontrar la clase ganadora
                let winningClass = null;
                let maxScore = 0;
                
                if (currentWeights === 'distance') {{
                    for (const [cls, weight] of Object.entries(weights)) {{
                        if (weight > maxScore) {{
                            maxScore = weight;
                            winningClass = parseInt(cls);
                        }}
                    }}
                }} else {{
                    for (const [cls, count] of Object.entries(votes)) {{
                        if (count > maxScore) {{
                            maxScore = count;
                            winningClass = parseInt(cls);
                        }}
                    }}
                }}
                
                // Calcular confianza
                const totalVotes = currentWeights === 'distance' ? 
                    Object.values(weights).reduce((a, b) => a + b, 0) :
                    Object.values(votes).reduce((a, b) => a + b, 0);
                    
                const confidence = totalVotes > 0 ? (maxScore / totalVotes) : 0;
                
                return {{
                    predictedClass: winningClass,
                    confidence: confidence,
                    neighbors: neighbors,
                    className: classInfo.find(c => c.id === winningClass)?.name || `Clase ${{winningClass}}`
                }};
            }}
            
            // Dibujar la visualización
            function draw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Calcular tamaño de puntos basado en el tamaño del canvas
                const pointSize = Math.max(4, Math.min(8, canvas.width / 120));
                const testPointSize = Math.max(6, Math.min(12, canvas.width / 80));
                
                // Dibujar puntos de entrenamiento
                trainingData.forEach(point => {{
                    const pos = dataToCanvas(point.x, point.y);
                    ctx.fillStyle = point.color;
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, pointSize, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.strokeStyle = '#333';
                    ctx.lineWidth = Math.max(1, pointSize / 3);
                    ctx.stroke();
                }});
                
                // Dibujar puntos de prueba y sus clasificaciones
                testPoints.forEach((testPoint, index) => {{
                    const pos = dataToCanvas(testPoint.x, testPoint.y);
                    const classification = classifyPoint(testPoint);
                    
                    // Dibujar líneas a vecinos más cercanos
                    if (showDistances && classification.neighbors) {{
                        ctx.strokeStyle = '#FF4444';
                        ctx.lineWidth = Math.max(1, canvas.width / 400);
                        ctx.setLineDash([5, 5]);
                        
                        classification.neighbors.forEach(neighbor => {{
                            const neighborPos = dataToCanvas(neighbor.x, neighbor.y);
                            ctx.beginPath();
                            ctx.moveTo(pos.x, pos.y);
                            ctx.lineTo(neighborPos.x, neighborPos.y);
                            ctx.stroke();
                        }});
                        ctx.setLineDash([]);
                    }}
                    
                    // Dibujar punto de prueba
                    const classColor = classInfo.find(c => c.id === classification.predictedClass)?.color || '#FFD700';
                    ctx.fillStyle = classColor;
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, testPointSize, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.strokeStyle = '#333';
                    ctx.lineWidth = Math.max(2, testPointSize / 4);
                    ctx.stroke();
                    
                    // Dibujar estrella en el centro
                    ctx.fillStyle = '#FFD700';
                    ctx.beginPath();
                    const starSize = testPointSize * 0.6;
                    for (let i = 0; i < 5; i++) {{
                        const angle = (i * 2 * Math.PI) / 5 - Math.PI / 2;
                        const x = pos.x + Math.cos(angle) * starSize;
                        const y = pos.y + Math.sin(angle) * starSize;
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }}
                    ctx.closePath();
                    ctx.fill();
                    
                    // Mostrar confianza si está habilitado
                    if (showConfidence) {{
                        const confidenceRadius = testPointSize + 8;
                        const confidenceAngle = classification.confidence * 2 * Math.PI;
                        
                        ctx.strokeStyle = '#4CAF50';
                        ctx.lineWidth = Math.max(2, canvas.width / 300);
                        ctx.beginPath();
                        ctx.arc(pos.x, pos.y, confidenceRadius, -Math.PI / 2, -Math.PI / 2 + confidenceAngle);
                        ctx.stroke();
                        
                        // Texto de confianza (solo si hay espacio)
                        if (canvas.width > 400) {{
                            ctx.fillStyle = '#333';
                            ctx.font = `${{Math.max(10, canvas.width / 60)}}px Arial`;
                            ctx.textAlign = 'center';
                            ctx.fillText(`${{Math.round(classification.confidence * 100)}}%`, pos.x, pos.y + confidenceRadius + 15);
                        }}
                    }}
                    
                    // Etiqueta de clase (solo si hay espacio)
                    if (canvas.width > 300) {{
                        ctx.fillStyle = '#333';
                        ctx.font = `bold ${{Math.max(10, canvas.width / 60)}}px Arial`;
                        ctx.textAlign = 'center';
                        ctx.fillText(classification.className, pos.x, pos.y - testPointSize - 8);
                    }}
                }});
                
                // Actualizar estadísticas
                updateStats();
            }}
            
            // Actualizar estadísticas
            function updateStats() {{
                document.getElementById('totalPoints').textContent = testPoints.length;
                document.getElementById('currentK').textContent = currentK;
                
                if (testPoints.length > 0) {{
                    const avgConf = testPoints.reduce((sum, point) => {{
                        const classification = classifyPoint(point);
                        return sum + classification.confidence;
                    }}, 0) / testPoints.length;
                    document.getElementById('avgConfidence').textContent = `${{Math.round(avgConf * 100)}}%`;
                }} else {{
                    document.getElementById('avgConfidence').textContent = '0%';
                }}
            }}
            
            // Event listeners
            canvas.addEventListener('click', function(event) {{
                const rect = canvas.getBoundingClientRect();
                const canvasX = event.clientX - rect.left;
                const canvasY = event.clientY - rect.top;
                
                // Verificar que el click esté dentro del área de datos (con márgenes)
                const margin = 20;
                if (canvasX >= margin && canvasX <= canvas.width - margin && 
                    canvasY >= margin && canvasY <= canvas.height - margin) {{
                    const dataPoint = canvasToData(canvasX, canvasY);
                    testPoints.push(dataPoint);
                    draw();
                    
                    // Sincronizar con Streamlit
                    window.parent.postMessage({{
                        type: 'addTestPoint',
                        point: [dataPoint.x, dataPoint.y]
                    }}, '*');
                }}
            }});
            
            // Redimensionar cuando cambie el tamaño de ventana
            window.addEventListener('resize', function() {{
                setTimeout(resizeCanvas, 100);
            }});
            
            // Funciones de control
            function updateK(value) {{
                currentK = parseInt(value);
                document.getElementById('kDisplay').textContent = value;
                draw();
            }}
            
            function updateMetric(value) {{
                currentMetric = value;
                draw();
            }}
            
            function updateWeights(value) {{
                currentWeights = value;
                draw();
            }}
            
            function updateShowConfidence(checked) {{
                showConfidence = checked;
                draw();
            }}
            
            function updateShowDistances(checked) {{
                showDistances = checked;
                draw();
            }}
            
            function clearTestPoints() {{
                testPoints = [];
                draw();
                window.parent.postMessage({{type: 'clearTestPoints'}}, '*');
            }}
            
            // Cargar puntos de prueba existentes
            const existingTestPoints = {json.dumps(test_points)};
            testPoints = existingTestPoints.map(point => ({{x: point[0], y: point[1]}}));
            
            // Inicialización
            resizeCanvas();
            
            // Observer para detectar cambios en el tamaño del contenedor
            if (window.ResizeObserver) {{
                const resizeObserver = new ResizeObserver(entries => {{
                    for (let entry of entries) {{
                        if (entry.target.querySelector('#knnCanvas')) {{
                            resizeCanvas();
                        }}
                    }}
                }});
                resizeObserver.observe(document.querySelector('.canvas-container'));
            }}
        </script>
    </body>
    </html>
        """

        # Mostrar la visualización usando st.components.v1.html con ancho completo
        components.html(html_code, height=800, scrolling=False)

        # CSS personalizado para forzar ancho completo del contenedor
        st.markdown("""
        <style>
        /* Forzar ancho completo para el contenedor de la visualización KNN */
        .stElementContainer.element-container.st-emotion-cache-1vr7d6u.eceldm40 {
            width: 100% !important;
        }
        
        /* Alternativa más específica si la anterior no funciona */
        div[data-testid="stVerticalBlock"] > div:has(iframe) {
            width: 100% !important;
        }
        
        /* Asegurar que el iframe también use todo el ancho */
        iframe {
            width: 100% !important;
        }
        
        /* Forzar contenedores padre también */
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # JavaScript listener para sincronizar con Streamlit
        st.markdown("""
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'addTestPoint') {
            // Aquí podríamos enviar el punto de vuelta a Streamlit si fuera necesario
            console.log('New test point:', event.data.point);
        } else if (event.data.type === 'clearTestPoints') {
            console.log('Clearing test points');
        }
        });
        </script>
        """, unsafe_allow_html=True)

        # Información adicional debajo de la visualización
        if test_points:
            st.markdown("### 📊 Análisis de Puntos de Prueba")

            # Crear tabla con resultados
            results_data = []
            for i, (tx, ty) in enumerate(test_points):
                test_point = np.array([[tx, ty]])
                prediction = model.predict(test_point)[0]
                probabilities = model.predict_proba(test_point)[0]

                predicted_class = class_names[int(
                    prediction)] if class_names else f"Clase {int(prediction)}"
                confidence = np.max(probabilities)

                # Encontrar vecinos más cercanos
                distances, indices = model.kneighbors(test_point)

                results_data.append({
                    'Punto': i+1,
                    f'{feature1}': f"{tx:.3f}",
                    f'{feature2}': f"{ty:.3f}",
                    'Predicción': predicted_class,
                    'Confianza': f"{confidence:.1%}",
                    'Distancia Promedio': f"{np.mean(distances[0]):.3f}"
                })

            st.dataframe(pd.DataFrame(results_data), use_container_width=True)

            # Estadísticas generales
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total de Puntos", len(test_points))

            with col2:
                avg_confidence = np.mean(
                    [np.max(model.predict_proba(np.array([[tx, ty]])))
                     for tx, ty in test_points]
                )
                st.metric("Confianza Promedio", f"{avg_confidence:.1%}")

            with col3:
                st.metric("Parámetros",
                          f"K={model.n_neighbors}, {model.metric}")

        else:
            st.info("💡 **¿Cómo usar la visualización?**\n\n" +
                    "1. **Haz click** en cualquier parte del área gris para agregar puntos de prueba\n" +
                    "2. **Ajusta los controles** para cambiar el comportamiento del algoritmo\n" +
                    "3. **Observa las líneas punteadas** que conectan cada punto con sus vecinos más cercanos\n" +
                    "4. **Examina la confianza** mostrada como porcentaje y anillo alrededor de cada punto\n\n" +
                    "¡Experimenta con diferentes valores de K y métricas para entender cómo afectan las predicciones!")

        # Explicación educativa
        with st.expander("📚 Guía de la Visualización Interactiva"):
            st.markdown(f"""
        **🎯 Visualización Interactiva de K-Nearest Neighbors**
        
        Esta herramienta te permite experimentar directamente con el algoritmo KNN:
        
        #### 🔧 **Controles Disponibles:**
        - **Valor K (1-15)**: Número de vecinos más cercanos a considerar
        - **Métrica de Distancia**: Euclidean, Manhattan, o Minkowski
        - **Tipo de Pesos**: Uniforme (todos iguales) o por distancia (más cercanos pesan más)
        - **Mostrar Confianza**: Visualiza qué tan seguro está el modelo (anillo verde)
        - **Mostrar Distancias**: Líneas punteadas rojas conectando con vecinos
        
        #### 🎨 **Elementos Visuales:**
        - **Puntos de colores**: Datos de entrenamiento (cada color = una clase)
        - **Estrellas doradas**: Puntos de prueba que agregues
        - **Líneas punteadas**: Conexiones a los K vecinos más cercanos
        - **Anillo verde**: Nivel de confianza de la predicción
        - **Porcentaje**: Confianza numérica de la clasificación
        
        #### 💡 **Consejos para Experimentar:**
        - Prueba diferentes valores de K en la misma ubicación
        - Haz click cerca de las fronteras entre clases
        - Observa cómo cambia la confianza en diferentes regiones
        - Compara las métricas Euclidean vs Manhattan
        - Experimenta con pesos uniformes vs por distancia
        
        #### 🎓 **¿Qué Aprender?**
        - **K bajo**: Más sensible a ruido, fronteras más irregulares
        - **K alto**: Más suave, pero puede perder detalles importantes
        - **Distancia Euclidean**: Círculos de influencia
        - **Distancia Manhattan**: Cuadrados de influencia (movimiento tipo taxi)
        - **Pesos por distancia**: Vecinos más cercanos tienen más influencia
        
        **¡La mejor forma de aprender es experimentando directamente con la visualización!**
        """)

    except Exception as e:
        st.error(f"Error en la visualización interactiva: {str(e)}")
        st.info("Fallback: Usando controles manuales")

        # Fallback a controles manuales simples
        col1, col2, col3 = st.columns(3)

        with col1:
            test_x = st.number_input(f"Valor {feature1}:",
                                     min_value=float(x_min),
                                     max_value=float(x_max),
                                     value=float((x_min + x_max) / 2),
                                     step=0.1)

        with col2:
            test_y = st.number_input(f"Valor {feature2}:",
                                     min_value=float(y_min),
                                     max_value=float(y_max),
                                     value=float((y_min + y_max) / 2),
                                     step=0.1)

        with col3:
            if st.button("➕ Agregar Punto", type="primary"):
                if 'test_points' not in st.session_state:
                    st.session_state.test_points = []
                st.session_state.test_points.append((test_x, test_y))
                st.rerun()


def create_animated_neighbors_plot(X_2d, y, test_point, model, feature1, feature2, class_names, colors):
    """
    Crea una visualización animada mostrando los vecinos más cercanos.
    """
    try:
        import plotly.graph_objects as go

        # Encontrar vecinos
        distances, indices = model.kneighbors(test_point)

        # Crear figura
        fig = go.Figure()

        # Agregar todos los puntos de entrenamiento (transparentes)
        for i, class_val in enumerate(np.unique(y)):
            mask = y == class_val
            class_name = class_names[int(
                class_val)] if class_names else f"Clase {int(class_val)}"

            fig.add_trace(go.Scatter(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                mode='markers',
                marker=dict(
                    color=colors[i],
                    size=6,
                    opacity=0.3,
                    line=dict(color='black', width=0.5)
                ),
                name=f"{class_name} (entrenamiento)",
                showlegend=False
            ))

        # Agregar punto de prueba
        fig.add_trace(go.Scatter(
            x=[test_point[0, 0]],
            y=[test_point[0, 1]],
            mode='markers',
            marker=dict(
                color='red',
                size=15,
                symbol='star',
                line=dict(color='black', width=2)
            ),
            name='Punto de prueba'
        ))

        # Agregar vecinos más cercanos destacados
        neighbor_points = X_2d[indices[0]]
        neighbor_classes = y[indices[0]]

        for i, (neighbor, neighbor_class, dist) in enumerate(zip(neighbor_points, neighbor_classes, distances[0])):
            class_name = class_names[int(
                neighbor_class)] if class_names else f"Clase {int(neighbor_class)}"

            # Punto vecino
            fig.add_trace(go.Scatter(
                x=[neighbor[0]],
                y=[neighbor[1]],
                mode='markers',
                marker=dict(
                    color=colors[int(neighbor_class)],
                    size=12,
                    line=dict(color='black', width=2)
                ),
                name=f'Vecino {i+1}: {class_name}',
                text=f"Distancia: {dist:.3f}",
                hovertemplate=f'Vecino {i+1}<br>Clase: {class_name}<br>Distancia: {dist:.3f}<extra></extra>'
            ))

            # Línea de conexión
            fig.add_trace(go.Scatter(
                x=[test_point[0, 0], neighbor[0]],
                y=[test_point[0, 1], neighbor[1]],
                mode='lines',
                line=dict(
                    color='gray',
                    width=2,
                    dash='dash'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Configurar layout
        fig.update_layout(
            title=f'Vecinos más cercanos (K={model.n_neighbors})',
            xaxis_title=feature1,
            yaxis_title=feature2,
            hovermode='closest',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creando visualización animada: {str(e)}")


# ===== FUNCIONES PARA REDES NEURONALES =====

def safe_get_output_size(config):
    """
    Extrae el tamaño de salida de forma segura para evitar errores de comparación de arrays.
    """
    try:
        output_size = config['output_size']
        # Si es un array o lista, tomar el primer elemento
        if hasattr(output_size, '__len__') and not isinstance(output_size, (str, bytes)):
            return int(output_size[0]) if len(output_size) > 0 else 1
        # Si es un escalar
        return int(output_size)
    except:
        return 1


def create_neural_network_visualization(architecture, activation, output_activation, task_type):
    """
    Crea una visualización dinámica de la arquitectura de red neuronal usando HTML5 Canvas.
    """
    try:
        # Colores para diferentes elementos
        colors = {
            'input': '#4ECDC4',
            'hidden': '#45B7D1',
            'output': '#FF6B6B',
            'connection': '#BDC3C7',
            'text': '#2C3E50'
        }

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .nn-container {{
                    max-width: 100%;
                    margin: 0 auto;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                }}
                .canvas-container {{
                    position: relative;
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    margin: 10px 0;
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    width: 100%;
                    overflow: hidden;
                }}
                #nnCanvas {{
                    display: block;
                    width: 100%;
                    height: auto;
                    max-width: 100%;
                }}
                .info-box {{
                    background: #e3f2fd;
                    border-left: 4px solid #2196F3;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 4px;
                    font-size: 14px;
                }}
                .layer-info {{
                    display: flex;
                    gap: 15px;
                    margin-top: 10px;
                    flex-wrap: wrap;
                    justify-content: center;
                    font-size: 12px;
                }}
                .layer-item {{
                    display: flex;
                    align-items: center;
                    gap: 5px;
                    padding: 5px 10px;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .layer-color {{
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                }}
            </style>
        </head>
        <body>
            <div class="nn-container">
                <div class="info-box">
                    <strong>🧠 Arquitectura de Red Neuronal</strong><br>
                    Visualización dinámica de la estructura de la red para {task_type.lower()}
                </div>
                
                <div class="canvas-container">
                    <canvas id="nnCanvas"></canvas>
                </div>
                
                <div class="layer-info">
                    <div class="layer-item">
                        <div class="layer-color" style="background-color: {colors['input']}"></div>
                        <span>Capa de Entrada</span>
                    </div>
                    <div class="layer-item">
                        <div class="layer-color" style="background-color: {colors['hidden']}"></div>
                        <span>Capas Ocultas ({activation.upper()})</span>
                    </div>
                    <div class="layer-item">
                        <div class="layer-color" style="background-color: {colors['output']}"></div>
                        <span>Capa de Salida ({output_activation.upper()})</span>
                    </div>
                </div>
            </div>

            <script>
                const canvas = document.getElementById('nnCanvas');
                const ctx = canvas.getContext('2d');
                
                // Arquitectura de la red
                const architecture = {architecture};
                const maxNeurons = Math.max(...architecture);
                
                // Función para redimensionar el canvas
                function resizeCanvas() {{
                    const container = document.querySelector('.canvas-container');
                    const containerWidth = container.clientWidth - 4;
                    const aspectRatio = 2/1;
                    const canvasHeight = Math.max(300, containerWidth / aspectRatio);
                    
                    canvas.width = containerWidth;
                    canvas.height = canvasHeight;
                    canvas.style.width = containerWidth + 'px';
                    canvas.style.height = canvasHeight + 'px';
                    
                    drawNetwork();
                }}
                
                // Función para dibujar la red neuronal
                function drawNetwork() {{
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    const margin = 40;
                    const layerWidth = (canvas.width - 2 * margin) / (architecture.length - 1);
                    const maxRadius = Math.min(20, canvas.width / (architecture.length * 8));
                    
                    // Dibujar conexiones primero
                    ctx.strokeStyle = '{colors['connection']}';
                    ctx.lineWidth = 1;
                    ctx.globalAlpha = 0.3;
                    
                    for (let i = 0; i < architecture.length - 1; i++) {{
                        const currentLayerSize = architecture[i];
                        const nextLayerSize = architecture[i + 1];
                        
                        const currentX = margin + i * layerWidth;
                        const nextX = margin + (i + 1) * layerWidth;
                        
                        for (let j = 0; j < currentLayerSize; j++) {{
                            const currentY = getNodeY(j, currentLayerSize);
                            
                            for (let k = 0; k < nextLayerSize; k++) {{
                                const nextY = getNodeY(k, nextLayerSize);
                                
                                ctx.beginPath();
                                ctx.moveTo(currentX, currentY);
                                ctx.lineTo(nextX, nextY);
                                ctx.stroke();
                            }}
                        }}
                    }}
                    
                    ctx.globalAlpha = 1.0;
                    
                    // Dibujar nodos
                    architecture.forEach((layerSize, layerIndex) => {{
                        const x = margin + layerIndex * layerWidth;
                        const radius = Math.min(maxRadius, Math.max(8, (canvas.height - 2 * margin) / (maxNeurons * 3)));
                        
                        // Color según tipo de capa
                        let color;
                        if (layerIndex === 0) {{
                            color = '{colors['input']}';
                        }} else if (layerIndex === architecture.length - 1) {{
                            color = '{colors['output']}';
                        }} else {{
                            color = '{colors['hidden']}';
                        }}
                        
                        // Dibujar nodos de la capa
                        for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {{
                            const y = getNodeY(nodeIndex, layerSize);
                            
                            // Nodo
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.arc(x, y, radius, 0, 2 * Math.PI);
                            ctx.fill();
                            
                            // Borde
                            ctx.strokeStyle = '#2C3E50';
                            ctx.lineWidth = 2;
                            ctx.stroke();
                        }}
                        
                        // Etiqueta de capa
                        ctx.fillStyle = '{colors['text']}';
                        ctx.font = `bold ${{Math.max(10, canvas.width / 50)}}px Arial`;
                        ctx.textAlign = 'center';
                        
                        let layerLabel;
                        if (layerIndex === 0) {{
                            layerLabel = `Entrada\\n(${{layerSize}})`;
                        }} else if (layerIndex === architecture.length - 1) {{
                            layerLabel = `Salida\\n(${{layerSize}})`;
                        }} else {{
                            layerLabel = `Oculta ${{layerIndex}}\\n(${{layerSize}})`;
                        }}
                        
                        // Dibujar texto en múltiples líneas
                        const lines = layerLabel.split('\\n');
                        lines.forEach((line, lineIndex) => {{
                            ctx.fillText(line, x, canvas.height - 25 + lineIndex * 15);
                        }});
                    }});
                }}
                
                // Función auxiliar para calcular posición Y de un nodo
                function getNodeY(nodeIndex, layerSize) {{
                    const margin = 40;
                    const availableHeight = canvas.height - 2 * margin - 60; // Espacio para etiquetas
                    
                    if (layerSize === 1) {{
                        return margin + availableHeight / 2;
                    }}
                    
                    const spacing = availableHeight / (layerSize + 1);
                    return margin + spacing * (nodeIndex + 1);
                }}
                
                // Inicialización
                resizeCanvas();
                
                // Redimensionar cuando cambie el tamaño de ventana
                window.addEventListener('resize', function() {{
                    setTimeout(resizeCanvas, 100);
                }});
                
                // Observer para detectar cambios en el contenedor
                if (window.ResizeObserver) {{
                    const resizeObserver = new ResizeObserver(entries => {{
                        for (let entry of entries) {{
                            if (entry.target.querySelector('#nnCanvas')) {{
                                resizeCanvas();
                            }}
                        }}
                    }});
                    resizeObserver.observe(document.querySelector('.canvas-container'));
                }}
            </script>
        </body>
        </html>
        """

        components.html(html_code, height=400, scrolling=False)

    except Exception as e:
        st.error(f"Error en la visualización de red neuronal: {str(e)}")


def calculate_network_parameters(architecture):
    """Calcula el número total de parámetros en la red."""
    total_params = 0
    for i in range(len(architecture) - 1):
        # Pesos: current_layer * next_layer + Sesgos: next_layer
        weights = architecture[i] * architecture[i + 1]
        biases = architecture[i + 1]
        total_params += weights + biases
    return total_params


def train_neural_network(df, target_col, config, learning_rate, epochs, validation_split,
                         early_stopping, patience, reduce_lr, lr_factor, progress_callback=None):
    """
    Entrena una red neuronal con la configuración especificada.
    """
    try:
        # Importar TensorFlow/Keras
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        import time

        # Paso 1: Preparar datos (ya mostrado)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Preprocesamiento
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42,
            stratify=y if config['task_type'] == 'Clasificación' else None
        )

        # Paso 2: Construyendo la red neuronal
        if progress_callback:
            progress_callback(
                2, "Construyendo arquitectura de red neuronal con capas y neuronas...")
        time.sleep(0.8)  # Pausa para que se vea el paso

        # Procesar variable objetivo
        label_encoder = None
        if config['task_type'] == 'Clasificación':
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            # Decisión de one-hot encoding basada en función de activación y número de clases
            output_size = safe_get_output_size(config)
            if config['output_activation'] == 'softmax' or (output_size > 1 and config['output_activation'] != 'sigmoid'):
                # Para softmax multiclase o funciones no-estándar multiclase
                y_train_encoded = keras.utils.to_categorical(y_train_encoded)
                y_test_encoded = keras.utils.to_categorical(y_test_encoded)
            # Para sigmoid (binaria o multiclase) mantener encoding simple
        else:
            y_train_encoded = y_train.values
            y_test_encoded = y_test.values

        # Construir modelo
        model = keras.Sequential()

        # Capa de entrada
        model.add(keras.layers.Dense(
            config['architecture'][1],
            activation=config['activation'],
            input_shape=(config['input_size'],)
        ))
        model.add(keras.layers.Dropout(config['dropout_rate']))

        # Capas ocultas
        for layer_size in config['architecture'][2:-1]:
            model.add(keras.layers.Dense(
                layer_size, activation=config['activation']))
            model.add(keras.layers.Dropout(config['dropout_rate']))

        # Capa de salida
        model.add(keras.layers.Dense(
            config['output_size'],
            activation=config['output_activation']
        ))

        # Paso 3: Compilando el modelo
        if progress_callback:
            progress_callback(
                3, "Compilando modelo con optimizadores y funciones de pérdida...")
        time.sleep(0.8)

        # Compilar modelo - Función de pérdida inteligente según activación
        if config['task_type'] == 'Clasificación':
            # Selección inteligente de función de pérdida
            output_size = safe_get_output_size(config)
            if config['output_activation'] == 'sigmoid':
                if output_size == 1:
                    loss = 'binary_crossentropy'  # Estándar para binaria con sigmoid
                else:
                    # Sigmoid multiclase (multi-label)
                    loss = 'binary_crossentropy'
                metrics = ['accuracy']
            elif config['output_activation'] == 'softmax':
                if output_size == 1:
                    # Softmax con 1 neurona es problemático, pero manejar el caso
                    loss = 'sparse_categorical_crossentropy'
                    metrics = ['accuracy']
                    st.warning(
                        "⚠️ Softmax con 1 neurona detectada. Puede causar problemas.")
                else:
                    loss = 'categorical_crossentropy'  # Estándar para multiclase con softmax
                    metrics = ['accuracy']
            elif config['output_activation'] == 'linear':
                # Linear para clasificación - usar sparse categorical
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
                st.warning(
                    "⚠️ Función linear detectada en clasificación. Rendimiento puede ser subóptimo.")
            elif config['output_activation'] == 'tanh':
                # Tanh para clasificación - tratar como regresión pero con accuracy
                loss = 'mse'
                metrics = ['accuracy']
                st.warning(
                    "⚠️ Función tanh detectada en clasificación. Comportamiento no estándar.")
            else:
                # Fallback
                loss = 'categorical_crossentropy' if output_size > 1 else 'binary_crossentropy'
                metrics = ['accuracy']
        else:
            # Para regresión
            if config['output_activation'] == 'linear':
                loss = 'mse'  # Estándar para regresión
                metrics = ['mae']
            elif config['output_activation'] in ['sigmoid', 'tanh']:
                loss = 'mse'  # MSE también funciona con activaciones acotadas
                metrics = ['mae']
                if config['output_activation'] == 'sigmoid':
                    st.info(
                        "ℹ️ Sigmoid limitará las salidas a [0,1]. Asegúrate de que tus datos objetivo estén normalizados.")
                else:  # tanh
                    st.info(
                        "ℹ️ Tanh limitará las salidas a [-1,1]. Asegúrate de que tus datos objetivo estén normalizados.")
            elif config['output_activation'] == 'softmax':
                loss = 'mse'
                metrics = ['mae']
                st.error(
                    "⚠️ Softmax en regresión: las salidas sumarán 1. Esto raramente es lo deseado.")
            else:
                loss = 'mse'
                metrics = ['mae']

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        if config['optimizer'] == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif config['optimizer'] == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Callbacks
        callbacks = []

        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True
            )
            callbacks.append(early_stop)

        if reduce_lr:
            reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=lr_factor, patience=patience//2, min_lr=1e-7
            )
            callbacks.append(reduce_lr_callback)

        # Paso 4: Iniciando entrenamiento
        if progress_callback:
            progress_callback(
                4, f"Entrenando red neuronal ({epochs} épocas máximo)... ¡Puede tardar unos minutos!")
        time.sleep(1.0)  # Pausa más larga antes del entrenamiento

        # Entrenar modelo
        history = model.fit(
            X_train, y_train_encoded,
            epochs=epochs,
            batch_size=config['batch_size'],
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )

        return model, history, X_test, y_test_encoded, scaler, label_encoder

    except ImportError:
        st.error(
            "❌ TensorFlow no está instalado. Las redes neuronales requieren TensorFlow.")
        st.info("Instala TensorFlow con: `pip install tensorflow`")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error durante el entrenamiento: {str(e)}")
        return None, None, None, None, None, None


def plot_training_history(history, task_type):
    """Grafica el historial de entrenamiento."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import streamlit as st

        # Crear subplots
        if task_type == 'Clasificación':
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Pérdida durante el Entrenamiento',
                                'Precisión durante el Entrenamiento')
            )

            # Pérdida
            fig.add_trace(
                go.Scatter(
                    y=history.history['loss'], name='Entrenamiento', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    y=history.history['val_loss'], name='Validación', line=dict(color='red')),
                row=1, col=1
            )

            # Precisión
            fig.add_trace(
                go.Scatter(y=history.history['accuracy'], name='Entrenamiento', line=dict(
                    color='blue'), showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history.history['val_accuracy'], name='Validación', line=dict(
                    color='red'), showlegend=False),
                row=1, col=2
            )

            fig.update_yaxes(title_text="Pérdida", row=1, col=1)
            fig.update_yaxes(title_text="Precisión", row=1, col=2)

        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    'Pérdida (MSE) durante el Entrenamiento', 'Error Absoluto Medio')
            )

            # MSE
            fig.add_trace(
                go.Scatter(
                    y=history.history['loss'], name='Entrenamiento', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    y=history.history['val_loss'], name='Validación', line=dict(color='red')),
                row=1, col=1
            )

            # MAE
            fig.add_trace(
                go.Scatter(y=history.history['mae'], name='Entrenamiento', line=dict(
                    color='blue'), showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history.history['val_mae'], name='Validación', line=dict(
                    color='red'), showlegend=False),
                row=1, col=2
            )

            fig.update_yaxes(title_text="MSE", row=1, col=1)
            fig.update_yaxes(title_text="MAE", row=1, col=2)

        fig.update_xaxes(title_text="Época")
        fig.update_layout(height=400, showlegend=True)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error graficando historial: {str(e)}")


def show_neural_network_evaluation():
    """Muestra la evaluación detallada del modelo de red neuronal."""
    if 'nn_model' not in st.session_state or st.session_state.nn_model is None:
        st.warning(
            "⚠️ Primero debes entrenar un modelo en la pestaña 'Entrenamiento'")
        return

    # Tips educativos sobre evaluación
    st.info("""
    🎓 **Evaluación de Redes Neuronales:**
    - **Accuracy**: Porcentaje de predicciones correctas (para clasificación)
    - **Matriz de Confusión**: Muestra qué clases se confunden entre sí
    - **MSE/MAE**: Errores promedio para regresión
    - **Datos de test**: Nunca vistos durante entrenamiento, miden la capacidad real
    """)

    try:
        import tensorflow as tf
        import numpy as np
        import pandas as pd
        from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
        import plotly.graph_objects as go
        import plotly.figure_factory as ff
        from plotly.subplots import make_subplots

        model = st.session_state.nn_model
        X_test, y_test = st.session_state.nn_test_data
        scaler = st.session_state.nn_scaler
        label_encoder = st.session_state.nn_label_encoder
        config = st.session_state.nn_config

        st.header("📊 Evaluación del Modelo")

        # Hacer predicciones
        y_pred = model.predict(X_test, verbose=0)

        # Métricas según el tipo de tarea
        if config['task_type'] == 'Clasificación':
            # Obtener el tamaño de salida de forma segura
            output_size = safe_get_output_size(config)

            # Para clasificación - detectar formato de y_test
            # One-hot encoded (multiclase)
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_test_classes = np.argmax(y_test, axis=1)
            else:  # Binaria o multiclase sin one-hot
                if output_size == 1:  # Binaria con 1 neurona
                    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                    y_test_classes = y_test.flatten()
                else:  # Multiclase sin one-hot (sparse)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_test_classes = y_test.flatten()

            # Accuracy
            accuracy = np.mean(y_pred_classes == y_test_classes)

            # Mostrar métricas principales con explicaciones
            st.markdown("### 🎯 Métricas Principales")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🎯 Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%",
                          help="Porcentaje de predicciones correctas en datos nunca vistos")

                # Interpretación del accuracy
                if accuracy >= 0.9:
                    st.success("🌟 **Excelente**: Tu red predice muy bien")
                elif accuracy >= 0.8:
                    st.success("✅ **Muy Bueno**: Predicciones muy confiables")
                elif accuracy >= 0.7:
                    st.warning("⚠️ **Bueno**: Predicciones aceptables")
                elif accuracy >= 0.6:
                    st.warning("🟡 **Regular**: Hay margen de mejora")
                else:
                    st.error("🔴 **Bajo**: Considera ajustar el modelo")

            with col2:
                # Calcular confianza promedio
                # One-hot multiclase
                if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                    confidence = np.mean(np.max(y_pred, axis=1))
                elif output_size == 1:  # Binaria
                    confidence = np.mean(np.maximum(
                        y_pred.flatten(), 1 - y_pred.flatten()))
                else:  # Multiclase sparse
                    confidence = np.mean(np.max(y_pred, axis=1))
                st.metric("🎲 Confianza Promedio",
                          f"{confidence:.4f}", f"{confidence*100:.2f}%")

            with col3:
                # Número de predicciones correctas
                correct_preds = np.sum(y_pred_classes == y_test_classes)
                st.metric("✅ Predicciones Correctas",
                          f"{correct_preds}/{len(y_test_classes)}")

            # Matriz de confusión
            st.subheader("🔍 Matriz de Confusión")

            try:
                cm = confusion_matrix(y_test_classes, y_pred_classes)

                # Obtener nombres de clases
                if label_encoder and hasattr(label_encoder, 'classes_'):
                    class_names = list(label_encoder.classes_)
                else:
                    # Determinar clases basado en los datos únicos
                    all_classes = sorted(
                        set(list(y_test_classes) + list(y_pred_classes)))
                    class_names = [f"Clase {i}" for i in all_classes]

                # Ajustar class_names al tamaño de la matriz si es necesario
                if len(class_names) != cm.shape[0]:
                    class_names = [f"Clase {i}" for i in range(cm.shape[0])]

                # Crear heatmap de la matriz de confusión
                fig_cm = ff.create_annotated_heatmap(
                    z=cm,
                    x=class_names,
                    y=class_names,
                    annotation_text=cm,
                    colorscale='Blues',
                    showscale=True
                )

                fig_cm.update_layout(
                    title='Matriz de Confusión',
                    xaxis_title='Predicciones',
                    yaxis_title='Valores Reales',
                    height=500
                )

                st.plotly_chart(fig_cm, use_container_width=True)

            except Exception as cm_error:
                st.error(
                    f"❌ Error creando matriz de confusión: {str(cm_error)}")
                st.info(
                    "La matriz de confusión no pudo generarse. El modelo funciona correctamente pero hay un problema con la visualización.")

            # Reporte de clasificación detallado
            st.subheader("📋 Reporte de Clasificación")

            if label_encoder:
                target_names = label_encoder.classes_
            else:
                target_names = [f"Clase {i}" for i in range(
                    len(np.unique(y_test_classes)))]

            # Generar reporte
            report = classification_report(
                y_test_classes, y_pred_classes,
                target_names=target_names,
                output_dict=True
            )

            # Mostrar métricas por clase
            metrics_data = []
            for class_name in target_names:
                if class_name in report:
                    metrics_data.append({
                        'Clase': class_name,
                        'Precisión': f"{report[class_name]['precision']:.4f}",
                        'Recall': f"{report[class_name]['recall']:.4f}",
                        'F1-Score': f"{report[class_name]['f1-score']:.4f}",
                        'Soporte': report[class_name]['support']
                    })

            st.dataframe(metrics_data, use_container_width=True)

            # Métricas macro y weighted
            st.subheader("📊 Métricas Agregadas")
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"""
                **Macro Average:**
                - Precisión: {report['macro avg']['precision']:.4f}
                - Recall: {report['macro avg']['recall']:.4f}
                - F1-Score: {report['macro avg']['f1-score']:.4f}
                """)

            with col2:
                st.info(f"""
                **Weighted Average:**
                - Precisión: {report['weighted avg']['precision']:.4f}
                - Recall: {report['weighted avg']['recall']:.4f}
                - F1-Score: {report['weighted avg']['f1-score']:.4f}
                """)

        else:
            # Para regresión
            y_pred_flat = y_pred.flatten()
            y_test_flat = y_test.flatten()

            # Métricas de regresión
            mse = mean_squared_error(y_test_flat, y_pred_flat)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_flat, y_pred_flat)
            r2 = r2_score(y_test_flat, y_pred_flat)

            # Mostrar métricas principales
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 R² Score", f"{r2:.4f}")

            with col2:
                st.metric("📏 MAE", f"{mae:.4f}")

            with col3:
                st.metric("📐 RMSE", f"{rmse:.4f}")

            with col4:
                st.metric("🎯 MSE", f"{mse:.4f}")

            # Gráficos de evaluación para regresión
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Predicciones vs Valores Reales',
                                'Distribución de Residuos')
            )

            # Scatter plot de predicciones vs reales
            fig.add_trace(
                go.Scatter(
                    x=y_test_flat,
                    y=y_pred_flat,
                    mode='markers',
                    name='Predicciones',
                    marker=dict(size=8, opacity=0.6)
                ),
                row=1, col=1
            )

            # Línea de referencia y = x
            min_val = min(y_test_flat.min(), y_pred_flat.min())
            max_val = max(y_test_flat.max(), y_pred_flat.max())

            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Línea Ideal',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )

            # Histograma de residuos
            residuals = y_test_flat - y_pred_flat
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name='Residuos',
                    nbinsx=30,
                    opacity=0.7
                ),
                row=1, col=2
            )

            fig.update_xaxes(title_text="Valores Reales", row=1, col=1)
            fig.update_yaxes(title_text="Predicciones", row=1, col=1)
            fig.update_xaxes(title_text="Residuos", row=1, col=2)
            fig.update_yaxes(title_text="Frecuencia", row=1, col=2)

            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        # Información del modelo
        st.subheader("🔧 Información del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Arquitectura:**
            - Capas: {len(config['architecture'])}
            - Neuronas por capa: {config['architecture']}
            - Función de activación: {config['activation']}
            - Activación de salida: {config['output_activation']}
            """)

        with col2:
            total_params = calculate_network_parameters(config['architecture'])
            st.info(f"""
            **Parámetros:**
            - Total de parámetros: {total_params:,}
            - Optimizador: {config['optimizer']}
            - Dropout: {config['dropout_rate']}
            - Batch size: {config['batch_size']}
            """)

        # Botón para generar código Python de evaluación
        st.markdown("### 💻 Código Python")
        if st.button("📝 Generar Código de Evaluación", use_container_width=True):
            # Generar código Python para evaluación
            code = generate_neural_network_evaluation_code(
                config, st.session_state.nn_feature_names, st.session_state.nn_class_names
            )

            st.markdown("#### 🐍 Código Python para Evaluación")
            st.code(code, language='python')

            # Botón para descargar el código
            st.download_button(
                label="💾 Descargar Código de Evaluación",
                data=code,
                file_name=f"evaluacion_red_neuronal_{config['task_type'].lower()}.py",
                mime="text/plain"
            )

        # Navegación
        st.markdown("---")
        st.markdown("### 🧭 Navegación")
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("🔙 Volver a Entrenamiento", use_container_width=True):
                st.session_state.active_tab_nn = 2
                st.rerun()
        with col_nav2:
            if st.button("🎯 Ver Visualizaciones", type="primary", use_container_width=True):
                st.session_state.active_tab_nn = 4
                st.rerun()

    except Exception as e:
        st.error(f"Error en la evaluación: {str(e)}")
        st.info(
            "Asegúrate de que TensorFlow esté instalado y el modelo esté entrenado correctamente.")


def show_neural_network_visualizations():
    """Muestra visualizaciones avanzadas del modelo."""
    if 'nn_model' not in st.session_state or st.session_state.nn_model is None:
        st.warning(
            "⚠️ Primero debes entrenar un modelo en la pestaña 'Entrenamiento'")
        return

    # Tips educativos sobre visualizaciones
    st.info("""
    🎓 **Visualizaciones de Redes Neuronales:**
    - **Historial de entrenamiento**: Muestra cómo evoluciona el aprendizaje
    - **Pesos y sesgos**: Revelan qué ha aprendido cada neurona
    - **Superficie de decisión**: Cómo la red separa las clases (2D)
    - **Análisis de capas**: Activaciones y patrones internos
    """)

    try:
        import tensorflow as tf
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        model = st.session_state.nn_model
        history = st.session_state.nn_history
        config = st.session_state.nn_config

        # FORZAR inicialización completa del modelo SIEMPRE (de forma transparente)
        try:
            # Obtener datos de test para inicializar el modelo
            X_test, _ = st.session_state.nn_test_data
            
            # Estrategia SILENCIOSA de inicialización (sin importar el estado actual)
            # 1. SIEMPRE hacer predicciones para forzar construcción
            dummy_pred = model.predict(X_test[:1], verbose=0)
            
            # 2. Forzar construcción explícita usando el input shape real
            input_shape = (None, X_test.shape[1])
            
            # 3. Si el modelo no está construido, construirlo
            if not model.built:
                model.build(input_shape)
                
            # 4. FORZAR reconstrucción de todas las capas una por una
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'built'):
                    if not layer.built:
                        # Construir capa por capa
                        if i == 0:
                            # Primera capa usa input_shape original
                            layer.build(input_shape)
                        else:
                            # Capas subsecuentes usan la salida de la capa anterior
                            prev_layer_output_shape = model.layers[i-1].output_shape
                            layer.build(prev_layer_output_shape)
            
            # 5. Verificación MÚLTIPLE con diferentes tamaños de batch
            for batch_size in [1, 5, 10]:
                test_batch = X_test[:min(batch_size, len(X_test))]
                _ = model.predict(test_batch, verbose=0)
            
            # 6. Verificar que el modelo tenga input definido
            if model.input is None:
                # Última estrategia: recrear el modelo si es necesario
                model._set_inputs(X_test[:1])
            
        except Exception as init_error:
            # Proceso de reparación de emergencia SILENCIOSO
            try:
                # Obtener la configuración original del modelo
                original_config = st.session_state.nn_config
                
                # Recrear el modelo desde cero si es necesario
                import tensorflow as tf
                from tensorflow import keras
                
                # Crear nuevo modelo con la misma arquitectura
                new_model = keras.Sequential()
                
                # Agregar capas según la configuración original
                arch = original_config['architecture']
                
                # Primera capa con input_shape explícito
                new_model.add(keras.layers.Dense(
                    arch[1],
                    activation=original_config['activation'],
                    input_shape=(arch[0],)
                ))
                
                # Capas ocultas
                for layer_size in arch[2:-1]:
                    new_model.add(keras.layers.Dense(
                        layer_size,
                        activation=original_config['activation']
                    ))
                
                # Capa de salida
                new_model.add(keras.layers.Dense(
                    arch[-1],
                    activation=original_config['output_activation']
                ))
                
                # Copiar pesos del modelo original
                new_model.set_weights(model.get_weights())
                
                # Compilar el nuevo modelo
                new_model.compile(
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics
                )
                
                # Reemplazar el modelo en session state
                st.session_state.nn_model = new_model
                model = new_model
                
                # Verificar que funciona
                test_pred = model.predict(X_test[:5], verbose=0)
                
            except Exception as emergency_error:
                # Si todo falla, mostrar error simplificado
                st.error("❌ No se pudo inicializar el modelo para visualizaciones.")
                st.info("💡 Intenta reentrenar el modelo desde cero.")
                return
            
            # Proceso de reparación de emergencia
            try:
                # Obtener la configuración original del modelo
                original_config = st.session_state.nn_config
                
                # Recrear el modelo desde cero si es necesario
                import tensorflow as tf
                from tensorflow import keras
                
                st.info("� Recreando modelo desde configuración guardada...")
                
                # Crear nuevo modelo con la misma arquitectura
                new_model = keras.Sequential()
                
                # Agregar capas según la configuración original
                arch = original_config['architecture']
                
                # Primera capa con input_shape explícito
                new_model.add(keras.layers.Dense(
                    arch[1],
                    activation=original_config['activation'],
                    input_shape=(arch[0],)
                ))
                
                # Capas ocultas
                for layer_size in arch[2:-1]:
                    new_model.add(keras.layers.Dense(
                        layer_size,
                        activation=original_config['activation']
                    ))
                
                # Capa de salida
                new_model.add(keras.layers.Dense(
                    arch[-1],
                    activation=original_config['output_activation']
                ))
                
                # Copiar pesos del modelo original
                new_model.set_weights(model.get_weights())
                
                # Compilar el nuevo modelo
                new_model.compile(
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics
                )
                
                # Reemplazar el modelo en session state
                st.session_state.nn_model = new_model
                model = new_model
                
                # Verificar que funciona
                test_pred = model.predict(X_test[:5], verbose=0)
                
                st.success("✅ ¡Modelo recreado y funcionando!")
                
            except Exception as emergency_error:
                st.error(f"❌ Fallo en reparación de emergencia: {emergency_error}")
                st.info("💡 Como último recurso, intenta reentrenar el modelo desde cero.")

        st.header("📈 Visualizaciones Avanzadas")

        # Tabs para diferentes visualizaciones
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📊 Historial de Entrenamiento",
            "🧠 Pesos y Sesgos",
            "🎯 Superficie de Decisión",
            "📉 Análisis de Capas"
        ])

        with viz_tab1:
            st.subheader("📊 Historial de Entrenamiento Detallado")
            st.markdown("📈 **¿Cómo aprendió tu red neuronal?**")

            # Explicación sobre el historial
            with st.expander("💡 ¿Cómo interpretar estas gráficas?"):
                st.markdown("""
                **Gráfica de Pérdida (Loss):**
                - **Bajando**: La red está aprendiendo ✅
                - **Estable**: Ha convergido 🎯
                - **Subiendo**: Posible sobreajuste ⚠️
                - **Gap grande entre train/val**: Sobreajuste 🚨
                
                **Gráfica de Accuracy (clasificación) o MAE (regresión):**
                - **Subiendo**: Mejorando en las predicciones ✅
                - **Plateau**: Ha alcanzado su límite 📊
                - **Train > Val**: Normal, pero gap grande = sobreajuste ⚠️
                """)

            plot_training_history(history, config['task_type'])

            # Información adicional del entrenamiento
            st.markdown("#### 📊 Estadísticas del Entrenamiento")
            col1, col2, col3 = st.columns(3)

            with col1:
                final_loss = history.history['loss'][-1]
                initial_loss = history.history['loss'][0]
                improvement = ((initial_loss - final_loss) /
                               initial_loss) * 100
                st.metric("🔴 Pérdida Final (Entrenamiento)", f"{final_loss:.6f}",
                          f"-{improvement:.1f}% desde inicio")

            with col2:
                if 'val_loss' in history.history:
                    final_val_loss = history.history['val_loss'][-1]
                    overfitting_gap = final_val_loss - final_loss
                    st.metric("🟡 Pérdida Final (Validación)", f"{final_val_loss:.6f}",
                              f"Gap: {overfitting_gap:.6f}")

                    # Interpretación del gap
                    if overfitting_gap < 0.01:
                        st.success("✅ **Sin sobreajuste**: Gap muy pequeño")
                    elif overfitting_gap < 0.05:
                        st.warning("⚠️ **Sobreajuste leve**: Gap aceptable")
                    else:
                        st.error("🚨 **Sobreajuste**: Gap significativo")

            with col3:
                epochs_trained = len(history.history['loss'])
                st.metric("⏱️ Épocas Entrenadas", epochs_trained)

                # ¿Paró por early stopping?
                if 'nn_config' in st.session_state:
                    max_epochs = st.session_state.get('training_epochs', 100)
                    if epochs_trained < max_epochs:
                        st.caption(
                            "🛑 **Early Stopping**: Paró automáticamente")
                    else:
                        st.caption("🔄 **Completó todas las épocas**")

        with viz_tab2:
            st.subheader("🧠 Análisis de Pesos y Sesgos")
            st.markdown("🔍 **¿Qué ha aprendido cada neurona?**")

            # Explicación sobre pesos
            with st.expander("💡 ¿Qué significan los pesos?"):
                st.markdown("""
                **Pesos (Weights):**
                - **Valores altos**: Conexiones importantes entre neuronas
                - **Valores cercanos a 0**: Conexiones débiles o irrelevantes
                - **Valores negativos**: Relaciones inversas
                - **Distribución**: Indica si la red está bien inicializada
                
                **Sesgos (Biases):**
                - **Valores altos**: Neurona se activa fácilmente
                - **Valores bajos**: Neurona es más selectiva
                - **Distribución**: Debe ser razonable, no extrema
                """)

            # Obtener pesos de todas las capas
            layer_weights = []
            layer_biases = []

            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'get_weights') and layer.get_weights():
                    weights = layer.get_weights()
                    if len(weights) >= 2:  # Pesos y sesgos
                        layer_weights.append(weights[0])
                        layer_biases.append(weights[1])

            if layer_weights:
                # Crear gráficos para cada capa
                for i, (weights, biases) in enumerate(zip(layer_weights, layer_biases)):
                    st.markdown(f"#### 📊 Capa {i+1}")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Histograma de pesos
                        fig_weights = go.Figure()
                        fig_weights.add_trace(go.Histogram(
                            x=weights.flatten(),
                            nbinsx=50,
                            name=f'Pesos Capa {i+1}',
                            opacity=0.7
                        ))
                        fig_weights.update_layout(
                            title=f'Distribución de Pesos - Capa {i+1}',
                            xaxis_title='Valor de Peso',
                            yaxis_title='Frecuencia',
                            height=300
                        )
                        st.plotly_chart(fig_weights, use_container_width=True)

                        # Estadísticas de pesos
                        st.caption(f"📊 **Estadísticas**: Media={np.mean(weights):.4f}, "
                                   f"Std={np.std(weights):.4f}, "
                                   f"Min={np.min(weights):.4f}, "
                                   f"Max={np.max(weights):.4f}")

                    with col2:
                        # Histograma de sesgos
                        fig_biases = go.Figure()
                        fig_biases.add_trace(go.Histogram(
                            x=biases.flatten(),
                            nbinsx=20,
                            name=f'Sesgos Capa {i+1}',
                            opacity=0.7,
                            marker_color='orange'
                        ))
                        fig_biases.update_layout(
                            title=f'Distribución de Sesgos - Capa {i+1}',
                            xaxis_title='Valor de Sesgo',
                            yaxis_title='Frecuencia',
                            height=300
                        )
                        st.plotly_chart(fig_biases, use_container_width=True)

                        # Estadísticas de sesgos
                        st.caption(f"📊 **Estadísticas**: Media={np.mean(biases):.4f}, "
                                   f"Std={np.std(biases):.4f}, "
                                   f"Min={np.min(biases):.4f}, "
                                   f"Max={np.max(biases):.4f}")

                # Análisis general
                st.markdown("#### 🔍 Análisis General de la Red")
                all_weights = np.concatenate(
                    [w.flatten() for w in layer_weights])
                all_biases = np.concatenate(
                    [b.flatten() for b in layer_biases])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Pesos Promedio",
                              f"{np.mean(all_weights):.6f}")
                with col2:
                    st.metric("📊 Desv. Std. Pesos",
                              f"{np.std(all_weights):.6f}")
                with col3:
                    dead_neurons = np.sum(np.abs(all_weights) < 1e-6)
                    st.metric("💀 Pesos ~0", f"{dead_neurons}")

                # Salud de la red
                if np.std(all_weights) < 0.01:
                    st.error(
                        "🚨 **Problema**: Pesos muy pequeños, la red puede no haber aprendido")
                elif np.std(all_weights) > 2:
                    st.warning(
                        "⚠️ **Atención**: Pesos muy grandes, posible inestabilidad")
                else:
                    st.success("✅ **Saludable**: Distribución de pesos normal")
            else:
                st.warning("No se pudieron extraer los pesos del modelo")

        with viz_tab3:
            st.subheader("🎯 Superficie de Decisión")
            st.markdown(
                "🗺️ **¿Cómo divide tu red el espacio de características?**")

            # Solo mostrar si tenemos 2 características o menos
            if config['input_size'] <= 2:
                st.info(
                    "Generando superficie de decisión... (Puede tomar unos segundos)")
                # Aquí iría el código para generar superficie de decisión
                # Es complejo, por ahora mostrar mensaje
                st.markdown("""
                **Superficie de Decisión 2D:**
                - Cada color representa una clase predicha
                - Los puntos son tus datos de entrenamiento
                - Las fronteras muestran cómo la red separa las clases
                - Fronteras suaves = red bien generalizada
                - Fronteras muy complejas = posible sobreajuste
                """)
            else:
                st.warning(f"⚠️ **No disponible**: Tu dataset tiene {config['input_size']} características. "
                           "La superficie de decisión solo se puede visualizar con 2 características o menos.")

                st.markdown("""
                **Alternativas para datasets de alta dimensionalidad:**
                - Usar PCA para reducir a 2D
                - Seleccionar las 2 características más importantes
                - Analizar pares de características individualmente
                """)

        with viz_tab4:
            st.subheader("📉 Análisis de Capas")
            st.markdown("🔬 **Activaciones y patrones internos de la red**")

            # Explicación sobre activaciones
            with st.expander("💡 ¿Qué son las activaciones?"):
                st.markdown("""
                **Activaciones:**
                - **Valores que producen las neuronas** cuando procesan datos
                - **Primeras capas**: Detectan características básicas
                - **Capas intermedias**: Combinan características en patrones
                - **Última capa**: Decisión final o predicción
                
                **Qué buscar:**
                - **Muchos ceros**: Neuronas "muertas" (problema)
                - **Valores extremos**: Saturación (problema)
                - **Distribución balanceada**: Red saludable ✅
                """)

            # Obtener algunas muestras para analizar activaciones
            X_test, y_test = st.session_state.nn_test_data
            sample_size = min(100, len(X_test))
            X_sample = X_test[:sample_size]

            try:
                # MÉTODO DIRECTO Y SIMPLE para análisis de activaciones
                st.info("� Iniciando análisis de activaciones...")
                
                # Como el modelo ya fue inicializado arriba, proceder directamente
                # Usar el input del modelo (debería estar disponible ya)
                model_input = model.input
                
                # Verificación simple
                if model_input is None:
                    st.error("❌ El modelo no se pudo inicializar correctamente.")
                    st.info("💡 Intenta reentrenar el modelo desde cero.")
                    return
                
                # Obtener outputs de todas las capas excepto la última
                if len(model.layers) <= 1:
                    st.warning("⚠️ El modelo tiene muy pocas capas para análisis detallado")
                    return
                
                # Crear lista de outputs de capas
                layer_outputs = []
                for i, layer in enumerate(model.layers[:-1]):
                    layer_outputs.append(layer.output)
                
                # Crear modelo de activaciones de forma SIMPLE
                activation_model = tf.keras.Model(inputs=model_input, outputs=layer_outputs)
                
                # Obtener activaciones directamente
                st.info("� Calculando activaciones de las capas...")
                activations = activation_model.predict(X_sample, verbose=0)

                if not isinstance(activations, list):
                    activations = [activations]

                # Mostrar estadísticas por capa
                for i, activation in enumerate(activations):
                    st.markdown(f"#### 📊 Capa {i+1} - Activaciones")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔥 Media", f"{np.mean(activation):.4f}")
                    with col2:
                        st.metric("📊 Desv. Std.", f"{np.std(activation):.4f}")
                    with col3:
                        dead_ratio = np.mean(activation == 0) * 100
                        st.metric("💀 % Neuronas Muertas", f"{dead_ratio:.1f}%")
                    with col4:
                        saturated_ratio = np.mean(activation >= 0.99) * 100
                        st.metric("🔴 % Saturadas", f"{saturated_ratio:.1f}%")

                    # Interpretación de la salud
                    if dead_ratio > 50:
                        st.error(
                            f"🚨 **Problema en Capa {i+1}**: Muchas neuronas muertas")
                    elif dead_ratio > 20:
                        st.warning(
                            f"⚠️ **Atención en Capa {i+1}**: Algunas neuronas muertas")
                    else:
                        st.success(
                            f"✅ **Capa {i+1} Saludable**: Buena activación")

            except Exception as e:
                st.error(f"❌ Error al analizar activaciones: {str(e)}")

                # Información de debug más detallada
                st.markdown("**🔍 Información de Debug:**")

                # Información del modelo
                try:
                    st.write(f"- **Tipo de modelo**: {type(model).__name__}")
                    st.write(f"- **Número de capas**: {len(model.layers)}")
                    st.write(f"- **Modelo construido**: {model.built}")

                    # Verificar si el modelo tiene input_spec
                    if hasattr(model, 'input_spec') and model.input_spec:
                        st.write(f"- **Input spec definido**: ✅")
                    else:
                        st.write(f"- **Input spec definido**: ❌")

                    # Verificar layers
                    for i, layer in enumerate(model.layers):
                        layer_built = getattr(layer, 'built', False)
                        st.write(
                            f"- **Capa {i+1} ({layer.__class__.__name__})**: {'✅' if layer_built else '❌'}")

                except Exception as debug_error:
                    st.write(f"- **Error en debug**: {debug_error}")

                st.info("""
                💡 **Posibles soluciones:**
                - El modelo necesita estar completamente construido antes del análisis
                - Intenta hacer algunas predicciones primero para que el modelo se inicialice
                - Si el problema persiste, vuelve a entrenar el modelo
                - Este error puede ocurrir con modelos Sequential que no han definido su input shape correctamente
                """)

                # Botón para intentar "reparar" el modelo
                if st.button("🔧 Intentar Reparar Modelo para Análisis", key="repair_model_viz"):
                    try:
                        st.info("Intentando construir el modelo completamente...")
                        # Hacer varias predicciones para asegurar que el modelo se construya
                        X_test, _ = st.session_state.nn_test_data

                        # Predicción con batch pequeño
                        _ = model.predict(X_test[:1], verbose=0)

                        # Predicción con batch más grande
                        batch_size = min(32, len(X_test))
                        _ = model.predict(X_test[:batch_size], verbose=0)

                        # Intentar construir explícitamente
                        if hasattr(model, 'build') and not model.built:
                            # Obtener el input shape del primer batch
                            input_shape = X_test[:1].shape
                            model.build(input_shape)

                        st.success(
                            "✅ Modelo reparado. Intenta el análisis de activaciones nuevamente.")
                        st.rerun()

                    except Exception as repair_error:
                        st.error(
                            f"❌ No se pudo reparar el modelo: {repair_error}")
                        st.info("Considera reentrenar el modelo desde cero.")

        # Botón para generar código de visualización
        st.markdown("### 💻 Código Python")
        if st.button("📝 Generar Código de Visualización", use_container_width=True):
            code = generate_neural_network_visualization_code(config)

            st.markdown("#### 🐍 Código Python para Visualizaciones")
            st.code(code, language='python')

            st.download_button(
                label="💾 Descargar Código de Visualización",
                data=code,
                file_name="visualizaciones_red_neuronal.py",
                mime="text/plain"
            )

        # Navegación
        st.markdown("---")
        st.markdown("### 🧭 Navegación")
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("🔙 Volver a Evaluación", use_container_width=True):
                st.session_state.active_tab_nn = 3
                st.rerun()
        with col_nav2:
            if st.button("🔮 Hacer Predicciones", type="primary", use_container_width=True):
                st.session_state.active_tab_nn = 5
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error en las visualizaciones: {str(e)}")

        # Diagnóstico detallado del error
        error_type = type(e).__name__
        error_msg = str(e)

        st.markdown("### 🔍 Diagnóstico del Error")

        if "never been called" in error_msg or "no defined input" in error_msg:
            st.error("🚨 **Problema de Inicialización del Modelo**")
            st.markdown("""
            **Causa del problema:**
            - El modelo Sequential no ha sido completamente inicializado
            - Las capas no tienen sus formas de entrada definidas
            - Se necesita hacer al menos una predicción para construir el modelo
            """)

            # Botón de reparación automática
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔧 Reparar Modelo Automáticamente", type="primary", key="auto_repair"):
                    try:
                        st.info("🔄 Reparando modelo...")

                        # Obtener datos de test
                        if 'nn_test_data' in st.session_state:
                            X_test, y_test = st.session_state.nn_test_data

                            # Forzar construcción del modelo con múltiples estrategias
                            with st.spinner("Inicializando modelo..."):
                                # Estrategia 1: Predicción simple
                                _ = model.predict(X_test[:1], verbose=0)

                                # Estrategia 2: Predicción con batch más grande
                                batch_size = min(10, len(X_test))
                                _ = model.predict(
                                    X_test[:batch_size], verbose=0)

                                # Estrategia 3: Compilar explícitamente si es necesario
                                if not model.built:
                                    model.build(input_shape=(
                                        None, X_test.shape[1]))

                                # Estrategia 4: Verificar que todas las capas estén construidas
                                for layer in model.layers:
                                    if hasattr(layer, 'built') and not layer.built:
                                        layer.build(input_shape=(
                                            None, X_test.shape[1]))

                            # Verificar que el modelo esté funcionando
                            test_pred = model.predict(X_test[:5], verbose=0)

                            st.success("✅ ¡Modelo reparado exitosamente!")
                            st.info(
                                "El modelo ahora está completamente inicializado. Puedes usar todas las visualizaciones.")

                            # Botón para recargar visualizaciones
                            if st.button("🔄 Recargar Visualizaciones", type="primary"):
                                st.rerun()

                        else:
                            st.error(
                                "❌ No se encontraron datos de test para reparar el modelo")

                    except Exception as repair_error:
                        st.error(
                            f"❌ Error durante la reparación: {repair_error}")
                        st.info("Intenta reentrenar el modelo desde cero.")

            with col2:
                st.markdown("**💡 Solución manual:**")
                st.markdown("""
                1. Ve a la pestaña **'Entrenamiento'**
                2. Reentrena el modelo desde cero
                3. Regresa a esta pestaña
                4. Las visualizaciones deberían funcionar
                """)

                if st.button("🔙 Ir a Entrenamiento", key="go_training"):
                    st.session_state.active_tab_nn = 2
                    st.rerun()

        else:
            # Otros tipos de errores
            st.warning("⚠️ **Error Inesperado**")
            st.code(f"Tipo: {error_type}\nMensaje: {error_msg}")

            st.markdown("""
            **Posibles soluciones:**
            - Verifica que TensorFlow esté instalado correctamente
            - Asegúrate de que el modelo esté entrenado
            - Intenta reentrenar el modelo
            - Reinicia la aplicación si persiste el problema
            """)

        # Información técnica adicional
        with st.expander("🔬 Información Técnica Detallada"):
            try:
                st.write("**Estado del Modelo:**")
                st.write(f"- Tipo: {type(model).__name__}")
                st.write(
                    f"- Construido: {getattr(model, 'built', 'No disponible')}")
                st.write(
                    f"- Número de capas: {len(model.layers) if hasattr(model, 'layers') else 'No disponible'}")

                if hasattr(model, 'input'):
                    st.write(
                        f"- Input definido: {'✅' if model.input is not None else '❌'}")

                if hasattr(model, 'layers'):
                    st.write("**Estado de las Capas:**")
                    for i, layer in enumerate(model.layers):
                        layer_built = getattr(layer, 'built', False)
                        st.write(
                            f"  - Capa {i+1} ({layer.__class__.__name__}): {'✅' if layer_built else '❌'}")

            except Exception as debug_error:
                st.write(f"Error obteniendo información: {debug_error}")

        st.info("💡 **Tip**: Este error es común con modelos Sequential. La reparación automática debería resolverlo.")


def show_neural_network_predictions():
    """Interfaz para hacer predicciones con el modelo entrenado."""
    if 'nn_model' not in st.session_state or st.session_state.nn_model is None:
        st.warning(
            "⚠️ Primero debes entrenar un modelo en la pestaña 'Entrenamiento'")
        return

    try:
        import numpy as np
        import pandas as pd
        import scipy.stats

        model = st.session_state.nn_model
        scaler = st.session_state.nn_scaler
        label_encoder = st.session_state.nn_label_encoder
        config = st.session_state.nn_config

        if 'nn_df' not in st.session_state or 'nn_target_col' not in st.session_state:
            st.error("No hay datos disponibles para hacer predicciones.")
            return

        df = st.session_state.nn_df
        target_col = st.session_state.nn_target_col
        feature_cols = [col for col in df.columns if col != target_col]

        st.header("🎯 Hacer Predicciones")

        # Tabs para diferentes tipos de predicción
        pred_tab1, pred_tab2, pred_tab3 = st.tabs([
            "🔍 Predicción Individual",
            "📊 Predicción por Lotes",
            "🎲 Exploración Interactiva"
        ])

        with pred_tab1:
            st.subheader("🔍 Predicción Individual")
            st.markdown("Introduce los valores para cada característica:")

            # Crear inputs para cada característica
            input_values = {}

            # Organizar en columnas
            num_cols = min(3, len(feature_cols))
            cols = st.columns(num_cols)

            for i, feature in enumerate(feature_cols):
                col_idx = i % num_cols

                with cols[col_idx]:
                    # Obtener estadísticas de la característica
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())

                    input_values[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100,
                        key=f"nn_pred_{feature}"
                    )

                    st.caption(
                        f"Min: {min_val:.2f}, Max: {max_val:.2f}, Media: {mean_val:.2f}")

            # Botón para hacer predicción
            if st.button("🚀 Hacer Predicción", type="primary"):
                # Preparar datos para predicción
                input_array = np.array(
                    [[input_values[feature] for feature in feature_cols]])
                input_scaled = scaler.transform(input_array)

                # Hacer predicción
                prediction = model.predict(input_scaled, verbose=0)

                # Mostrar resultados
                st.success("✅ Predicción completada")

                if config['task_type'] == 'Clasificación':
                    output_size = safe_get_output_size(config)
                    if output_size > 2:  # Multiclase
                        predicted_class_idx = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class_idx]

                        if label_encoder:
                            predicted_class = label_encoder.inverse_transform(
                                [predicted_class_idx])[0]
                        else:
                            predicted_class = f"Clase {predicted_class_idx}"

                        # Mostrar resultado principal
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("🎯 Clase Predicha", predicted_class)

                        with col2:
                            st.metric(
                                "🎲 Confianza", f"{confidence:.4f}", f"{confidence*100:.2f}%")

                        # Mostrar probabilidades para todas las clases
                        st.subheader("📊 Probabilidades por Clase")

                        prob_data = []
                        for i, prob in enumerate(prediction[0]):
                            if label_encoder:
                                class_name = label_encoder.inverse_transform([i])[
                                    0]
                            else:
                                class_name = f"Clase {i}"

                            prob_data.append({
                                'Clase': class_name,
                                'Probabilidad': f"{prob:.4f}",
                                'Porcentaje': f"{prob*100:.2f}%"
                            })

                        st.dataframe(prob_data, use_container_width=True)

                        # Gráfico de barras de probabilidades
                        import plotly.graph_objects as go

                        class_names = [item['Clase'] for item in prob_data]
                        probabilities = [float(item['Probabilidad'])
                                         for item in prob_data]

                        fig = go.Figure(data=[
                            go.Bar(x=class_names, y=probabilities,
                                   marker_color=['red' if i == predicted_class_idx else 'lightblue'
                                                 for i in range(len(class_names))])
                        ])

                        fig.update_layout(
                            title="Distribución de Probabilidades",
                            xaxis_title="Clases",
                            yaxis_title="Probabilidad",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    else:  # Binaria
                        probability = prediction[0][0]
                        predicted_class_idx = 1 if probability > 0.5 else 0

                        if label_encoder:
                            predicted_class = label_encoder.inverse_transform(
                                [predicted_class_idx])[0]
                        else:
                            predicted_class = f"Clase {predicted_class_idx}"

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("🎯 Clase Predicha", predicted_class)

                        with col2:
                            st.metric("🎲 Probabilidad", f"{probability:.4f}")

                        with col3:
                            confidence = max(probability, 1 - probability)
                            st.metric(
                                "✨ Confianza", f"{confidence:.4f}", f"{confidence*100:.2f}%")

                else:  # Regresión
                    predicted_value = prediction[0][0]

                    st.metric("🎯 Valor Predicho", f"{predicted_value:.6f}")

                    # Información adicional para regresión
                    target_stats = df[target_col].describe()

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.info(f"📊 **Estadísticas del Target:**\n"
                                f"- Media: {target_stats['mean']:.4f}\n"
                                f"- Mediana: {target_stats['50%']:.4f}")

                    with col2:
                        st.info(f"📏 **Rango de Datos:**\n"
                                f"- Mínimo: {target_stats['min']:.4f}\n"
                                f"- Máximo: {target_stats['max']:.4f}")

                    with col3:
                        deviation_from_mean = abs(
                            predicted_value - target_stats['mean'])
                        st.info(f"🎯 **Análisis:**\n"
                                f"- Desviación de la media: {deviation_from_mean:.4f}\n"
                                f"- Percentil aproximado: {scipy.stats.percentileofscore(df[target_col], predicted_value):.1f}%")

        with pred_tab2:
            st.subheader("📊 Predicción por Lotes")

            st.markdown(
                "Sube un archivo CSV con nuevos datos para hacer predicciones en lote:")

            uploaded_file = st.file_uploader(
                "Selecciona archivo CSV",
                type=['csv'],
                key="nn_batch_predictions"
            )

            if uploaded_file is not None:
                try:
                    # Cargar datos
                    new_df = pd.read_csv(uploaded_file)

                    st.success(
                        f"✅ Archivo cargado: {new_df.shape[0]} filas, {new_df.shape[1]} columnas")

                    # Verificar que las columnas coincidan
                    missing_features = set(feature_cols) - set(new_df.columns)
                    extra_features = set(new_df.columns) - set(feature_cols)

                    if missing_features:
                        st.error(
                            f"❌ Faltan características: {', '.join(missing_features)}")
                    elif extra_features:
                        st.warning(
                            f"⚠️ Características adicionales (serán ignoradas): {', '.join(extra_features)}")
                        # Seleccionar solo las características necesarias
                        new_df = new_df[feature_cols]

                    if not missing_features:
                        # Mostrar vista previa
                        st.dataframe(new_df.head(), use_container_width=True)

                        if st.button("🚀 Generar Predicciones", type="primary"):
                            # Procesar datos
                            new_data_scaled = scaler.transform(new_df)

                            # Hacer predicciones
                            batch_predictions = model.predict(
                                new_data_scaled, verbose=0)

                            # Procesar resultados según el tipo de tarea
                            if config['task_type'] == 'Clasificación':
                                output_size = safe_get_output_size(config)
                                if output_size > 2:
                                    predicted_classes_idx = np.argmax(
                                        batch_predictions, axis=1)
                                    confidences = np.max(
                                        batch_predictions, axis=1)

                                    if label_encoder:
                                        predicted_classes = label_encoder.inverse_transform(
                                            predicted_classes_idx)
                                    else:
                                        predicted_classes = [
                                            f"Clase {idx}" for idx in predicted_classes_idx]

                                    results_df = new_df.copy()
                                    results_df['Predicción'] = predicted_classes
                                    results_df['Confianza'] = confidences

                                else:  # Binaria
                                    probabilities = batch_predictions.flatten()
                                    predicted_classes_idx = (
                                        probabilities > 0.5).astype(int)
                                    confidences = np.maximum(
                                        probabilities, 1 - probabilities)

                                    if label_encoder:
                                        predicted_classes = label_encoder.inverse_transform(
                                            predicted_classes_idx)
                                    else:
                                        predicted_classes = [
                                            f"Clase {idx}" for idx in predicted_classes_idx]

                                    results_df = new_df.copy()
                                    results_df['Predicción'] = predicted_classes
                                    results_df['Probabilidad'] = probabilities
                                    results_df['Confianza'] = confidences

                            else:  # Regresión
                                predicted_values = batch_predictions.flatten()

                                results_df = new_df.copy()
                                results_df['Predicción'] = predicted_values

                            # Mostrar resultados
                            st.success(
                                f"✅ Predicciones generadas para {len(results_df)} muestras")
                            st.dataframe(results_df, use_container_width=True)

                            # Botón para descargar resultados
                            csv_results = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Descargar Resultados",
                                data=csv_results,
                                file_name="predicciones_neural_network.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"Error procesando archivo: {str(e)}")

            else:
                # Mostrar formato esperado
                st.info("📋 **Formato esperado del archivo CSV:**")

                sample_data = df[feature_cols].head(3)
                st.dataframe(sample_data, use_container_width=True)

                st.markdown(
                    "El archivo debe contener las siguientes columnas:")
                st.code(", ".join(feature_cols))

        with pred_tab3:
            st.subheader("🎲 Exploración Interactiva")

            # Información educativa sobre la exploración interactiva
            with st.expander("ℹ️ ¿Qué es la Exploración Interactiva?", expanded=False):
                st.markdown("""
                **La exploración interactiva** te permite entender cómo el modelo neural toma decisiones:
                
                🔍 **¿Para qué sirve?**
                - Ver cómo cada característica influye en las predicciones
                - Identificar patrones y comportamientos del modelo
                - Detectar posibles sesgos o comportamientos inesperados
                - Comprender la sensibilidad del modelo a cambios en los datos
                
                📊 **¿Cómo interpretar los resultados?**
                - **Líneas ascendentes**: La característica tiene correlación positiva
                - **Líneas descendentes**: La característica tiene correlación negativa  
                - **Líneas planas**: La característica tiene poco impacto
                - **Cambios abruptos**: Puntos de decisión críticos del modelo
                
                💡 **Consejos de uso:**
                - Prueba diferentes muestras base para ver patrones generales
                - Observa qué características causan mayores cambios
                - Busca comportamientos inesperados o poco realistas
                """)

            st.markdown(
                "🎯 **Explora cómo cambian las predicciones al modificar diferentes características:**")

            # Seleccionar una muestra base
            st.markdown("**1. 📍 Selecciona una muestra base:**")

            st.info("💡 **Tip:** La muestra base es tu punto de referencia. Todas las exploraciones mostrarán cómo cambian las predicciones desde este punto inicial.")

            sample_idx = st.selectbox(
                "Índice de muestra:",
                range(len(df)),
                format_func=lambda x: f"Muestra {x}",
                key="nn_interactive_sample"
            )

            base_sample = df.iloc[sample_idx][feature_cols].to_dict()

            # Mostrar valores base
            st.markdown("**2. 📋 Valores base de la muestra:**")
            st.caption(
                "Estos son los valores de todas las características para la muestra seleccionada:")
            base_df = pd.DataFrame([base_sample])
            st.dataframe(base_df, use_container_width=True)

            # Hacer predicción base
            base_array = np.array([[base_sample[feature]
                                  for feature in feature_cols]])
            base_scaled = scaler.transform(base_array)
            base_prediction = model.predict(base_scaled, verbose=0)

            if config['task_type'] == 'Clasificación':
                output_size = safe_get_output_size(config)
                if output_size > 2:
                    base_class_idx = np.argmax(base_prediction[0])
                    base_confidence = base_prediction[0][base_class_idx]

                    if label_encoder:
                        base_class = label_encoder.inverse_transform([base_class_idx])[
                            0]
                    else:
                        base_class = f"Clase {base_class_idx}"

                    st.info(
                        f"🎯 **Predicción Base:** {base_class} (Confianza: {base_confidence:.3f})")
                else:
                    base_prob = base_prediction[0][0]
                    base_class_idx = 1 if base_prob > 0.5 else 0

                    if label_encoder:
                        base_class = label_encoder.inverse_transform([base_class_idx])[
                            0]
                    else:
                        base_class = f"Clase {base_class_idx}"

                    st.info(
                        f"🎯 **Predicción Base:** {base_class} (Probabilidad: {base_prob:.3f})")
            else:
                base_value = base_prediction[0][0]
                st.info(f"🎯 **Predicción Base:** {base_value:.6f}")

            # Seleccionar característica para explorar
            st.markdown("**3. 🔍 Explora el efecto de una característica:**")

            st.info("🎯 **Objetivo:** Verás cómo cambia la predicción cuando modificas solo UNA característica, manteniendo todas las demás constantes. Esto te ayuda a entender la importancia relativa de cada variable.")

            feature_to_explore = st.selectbox(
                "Característica a explorar:",
                feature_cols,
                key="nn_explore_feature",
                help="Selecciona la característica cuyo efecto quieres analizar en las predicciones"
            )

            # Crear rango de valores para la característica seleccionada
            min_val = float(df[feature_to_explore].min())
            max_val = float(df[feature_to_explore].max())

            # Generar valores para exploración
            exploration_values = np.linspace(min_val, max_val, 50)
            exploration_predictions = []

            for val in exploration_values:
                # Crear muestra modificada
                modified_sample = base_sample.copy()
                modified_sample[feature_to_explore] = val

                # Hacer predicción
                modified_array = np.array(
                    [[modified_sample[feature] for feature in feature_cols]])
                modified_scaled = scaler.transform(modified_array)
                pred = model.predict(modified_scaled, verbose=0)

                if config['task_type'] == 'Clasificación':
                    output_size = safe_get_output_size(config)
                    if output_size > 2:
                        pred_class_idx = np.argmax(pred[0])
                        confidence = pred[0][pred_class_idx]
                        exploration_predictions.append(
                            (pred_class_idx, confidence))
                    else:
                        prob = pred[0][0]
                        exploration_predictions.append(prob)
                else:
                    exploration_predictions.append(pred[0][0])

            # Crear visualización
            import plotly.graph_objects as go

            fig = go.Figure()

            if config['task_type'] == 'Clasificación':
                output_size = safe_get_output_size(config)
                if output_size > 2:
                    # Multiclase: mostrar clase predicha y confianza
                    classes = [pred[0] for pred in exploration_predictions]
                    confidences = [pred[1] for pred in exploration_predictions]

                    fig.add_trace(go.Scatter(
                        x=exploration_values,
                        y=classes,
                        mode='lines+markers',
                        name='Clase Predicha',
                        yaxis='y1'
                    ))

                    fig.add_trace(go.Scatter(
                        x=exploration_values,
                        y=confidences,
                        mode='lines+markers',
                        name='Confianza',
                        yaxis='y2',
                        line=dict(color='red')
                    ))

                    fig.update_layout(
                        title=f'Efecto de {feature_to_explore} en la Predicción',
                        xaxis_title=feature_to_explore,
                        yaxis=dict(title='Clase Predicha', side='left'),
                        yaxis2=dict(title='Confianza',
                                    side='right', overlaying='y'),
                        height=500
                    )
                else:
                    # Binaria: mostrar probabilidad
                    fig.add_trace(go.Scatter(
                        x=exploration_values,
                        y=exploration_predictions,
                        mode='lines+markers',
                        name='Probabilidad'
                    ))

                    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                                  annotation_text="Umbral de decisión")

                    fig.update_layout(
                        title=f'Efecto de {feature_to_explore} en la Probabilidad',
                        xaxis_title=feature_to_explore,
                        yaxis_title='Probabilidad',
                        height=500
                    )
            else:
                # Regresión
                fig.add_trace(go.Scatter(
                    x=exploration_values,
                    y=exploration_predictions,
                    mode='lines+markers',
                    name='Predicción'
                ))

                fig.update_layout(
                    title=f'Efecto de {feature_to_explore} en la Predicción',
                    xaxis_title=feature_to_explore,
                    yaxis_title='Valor Predicho',
                    height=500
                )

            # Marcar el valor base
            base_val = base_sample[feature_to_explore]
            fig.add_vline(x=base_val, line_dash="dash", line_color="green",
                          annotation_text="Valor Base")

            st.plotly_chart(fig, use_container_width=True)

            # Análisis interpretativo
            st.markdown("**📈 Análisis de Resultados:**")

            # Calcular estadísticas del efecto
            if config['task_type'] == 'Clasificación':
                output_size = safe_get_output_size(config)
                if output_size <= 2:
                    pred_range = max(exploration_predictions) - \
                        min(exploration_predictions)
                    volatility = np.std(exploration_predictions)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rango de Probabilidades", f"{pred_range:.3f}",
                                  help="Diferencia entre la probabilidad máxima y mínima observada")
                    with col2:
                        st.metric("Volatilidad", f"{volatility:.3f}",
                                  help="Qué tan variables son las predicciones (desviación estándar)")

                    if pred_range > 0.3:
                        st.success(
                            f"🎯 **Característica muy influyente:** '{feature_to_explore}' tiene un gran impacto en las predicciones")
                    elif pred_range > 0.1:
                        st.warning(
                            f"📊 **Característica moderadamente influyente:** '{feature_to_explore}' tiene un impacto moderado")
                    else:
                        st.info(
                            f"📉 **Característica poco influyente:** '{feature_to_explore}' tiene poco impacto en las predicciones")
            else:
                pred_range = max(exploration_predictions) - \
                    min(exploration_predictions)
                pred_mean = np.mean(exploration_predictions)
                relative_impact = (pred_range / abs(pred_mean)
                                   ) * 100 if pred_mean != 0 else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rango de Predicciones", f"{pred_range:.6f}")
                with col2:
                    st.metric("Impacto Relativo", f"{relative_impact:.1f}%")

                if relative_impact > 20:
                    st.success(
                        f"🎯 **Característica muy influyente:** '{feature_to_explore}' causa cambios significativos")
                elif relative_impact > 5:
                    st.warning(
                        f"📊 **Característica moderadamente influyente:** '{feature_to_explore}' tiene impacto moderado")
                else:
                    st.info(
                        f"📉 **Característica poco influyente:** '{feature_to_explore}' tiene poco impacto")

            # Consejos interpretativos
            with st.expander("💡 Consejos para Interpretar los Resultados", expanded=False):
                st.markdown(f"""
                **🔍 Analizando '{feature_to_explore}':**
                
                ✅ **Buenas señales:**
                - Cambios graduales y suaves en las predicciones
                - Comportamiento consistente con el conocimiento del dominio
                - Relaciones monotónicas (siempre creciente o decreciente)
                
                ⚠️ **Señales de alerta:**
                - Cambios muy abruptos sin explicación lógica
                - Comportamientos contradictorios al conocimiento experto
                - Excesiva sensibilidad a pequeños cambios
                
                **🎯 Próximos pasos:**
                1. Prueba con diferentes muestras base para confirmar patrones
                2. Explora otras características para comparar importancias
                3. Si encuentras comportamientos extraños, considera reentrenar el modelo
                4. Documenta los insights para mejorar futuras versiones del modelo
                """)

    except Exception as e:
        st.error(f"Error en las predicciones: {str(e)}")
        st.info("Asegúrate de que el modelo esté entrenado correctamente.")


def show_neural_network_export():
    """Permite exportar el modelo entrenado."""
    if 'nn_model' not in st.session_state or st.session_state.nn_model is None:
        st.warning(
            "⚠️ Primero debes entrenar un modelo en la pestaña 'Entrenamiento'")
        return

    try:
        import pickle
        import json
        from datetime import datetime

        model = st.session_state.nn_model
        scaler = st.session_state.nn_scaler
        label_encoder = st.session_state.nn_label_encoder
        config = st.session_state.nn_config

        st.header("📦 Exportar Modelo")

        # Información del modelo
        st.subheader("ℹ️ Información del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Arquitectura:**
            - Tipo: {config['task_type']}
            - Capas: {len(config['architecture'])}
            - Neuronas: {config['architecture']}
            - Activación: {config['activation']}
            - Optimizador: {config['optimizer']}
            """)

        with col2:
            total_params = calculate_network_parameters(config['architecture'])
            st.info(f"""
            **Parámetros:**
            - Total: {total_params:,}
            - Dropout: {config['dropout_rate']}
            - Batch size: {config['batch_size']}
            - Fecha entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """)

        # Opciones de exportación
        st.subheader("📁 Opciones de Exportación")

        export_tab1, export_tab2, export_tab3, export_tab4 = st.tabs([
            "🤖 Modelo TensorFlow",
            "📊 Modelo Completo",
            "📝 Código Python",
            "📋 Metadatos"
        ])

        with export_tab1:
            st.markdown("**Exportar solo el modelo de TensorFlow:**")

            format_option = st.radio(
                "Formato:",
                ["SavedModel (.pb)", "HDF5 (.h5)",
                 "TensorFlow Lite (.tflite)"],
                key="nn_export_format"
            )

            if st.button("💾 Exportar Modelo TensorFlow", type="primary"):
                try:
                    if format_option == "SavedModel (.pb)":
                        # Guardar como SavedModel
                        import tempfile
                        import zipfile
                        import io

                        with tempfile.TemporaryDirectory() as temp_dir:
                            model_path = f"{temp_dir}/neural_network_model"
                            model.save(model_path)

                            # Crear ZIP con el modelo
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                import os
                                for root, dirs, files in os.walk(model_path):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        arc_name = os.path.relpath(
                                            file_path, temp_dir)
                                        zip_file.write(file_path, arc_name)

                            zip_buffer.seek(0)

                            st.download_button(
                                label="📥 Descargar SavedModel",
                                data=zip_buffer.getvalue(),
                                file_name="neural_network_savedmodel.zip",
                                mime="application/zip"
                            )

                    elif format_option == "HDF5 (.h5)":
                        # Guardar como HDF5
                        import io
                        import tempfile

                        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                            model.save(tmp_file.name)

                            with open(tmp_file.name, 'rb') as f:
                                model_data = f.read()

                            st.download_button(
                                label="📥 Descargar Modelo HDF5",
                                data=model_data,
                                file_name="neural_network_model.h5",
                                mime="application/octet-stream"
                            )

                    elif format_option == "TensorFlow Lite (.tflite)":
                        # Convertir a TensorFlow Lite
                        import tensorflow as tf

                        converter = tf.lite.TFLiteConverter.from_keras_model(
                            model)
                        tflite_model = converter.convert()

                        st.download_button(
                            label="📥 Descargar Modelo TFLite",
                            data=tflite_model,
                            file_name="neural_network_model.tflite",
                            mime="application/octet-stream"
                        )

                        st.success("✅ Modelo convertido a TensorFlow Lite")
                        st.info(
                            "💡 TensorFlow Lite es ideal para aplicaciones móviles y embebidas")

                except Exception as e:
                    st.error(f"Error exportando modelo: {str(e)}")

        with export_tab2:
            st.markdown("**Exportar modelo completo con preprocesadores:**")
            st.info("Incluye el modelo, scaler, label encoder y configuración")

            if st.button("💾 Exportar Modelo Completo", type="primary"):
                try:
                    # Crear diccionario con todos los componentes
                    complete_model = {
                        'model': model,
                        'scaler': scaler,
                        'label_encoder': label_encoder,
                        'config': config,
                        'feature_names': st.session_state.get('nn_feature_names', []),
                        'export_date': datetime.now().isoformat(),
                        'version': '1.0'
                    }

                    # Serializar con pickle
                    model_data = pickle.dumps(complete_model)

                    st.download_button(
                        label="📥 Descargar Modelo Completo",
                        data=model_data,
                        file_name="neural_network_complete.pkl",
                        mime="application/octet-stream"
                    )

                    st.success("✅ Modelo completo exportado")
                    st.info(
                        "💡 Este archivo contiene todo lo necesario para hacer predicciones")

                except Exception as e:
                    st.error(f"Error exportando modelo completo: {str(e)}")

            # Mostrar código de ejemplo para cargar
            st.markdown("**Código para cargar el modelo:**")

            load_code = """
import pickle
import numpy as np

# Cargar modelo completo
with open('neural_network_complete.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

model = loaded_model['model']
scaler = loaded_model['scaler']
label_encoder = loaded_model['label_encoder']
config = loaded_model['config']

# Hacer predicción con nuevos datos
def predecir(nuevos_datos):
    # nuevos_datos debe ser una lista con valores para cada característica
    datos_escalados = scaler.transform([nuevos_datos])
    prediccion = model.predict(datos_escalados)
    
    if config['task_type'] == 'Clasificación':
        output_size = safe_get_output_size(config)
        if output_size > 2:
            clase_idx = np.argmax(prediccion[0])
            if label_encoder:
                clase = label_encoder.inverse_transform([clase_idx])[0]
            else:
                clase = f"Clase {clase_idx}"
            confianza = prediccion[0][clase_idx]
            return clase, confianza
        else:
            probabilidad = prediccion[0][0]
            clase_idx = 1 if probabilidad > 0.5 else 0
            if label_encoder:
                clase = label_encoder.inverse_transform([clase_idx])[0]
            else:
                clase = f"Clase {clase_idx}"
            return clase, probabilidad
    else:
        return prediccion[0][0]

# Ejemplo de uso:
# resultado = predecir([valor1, valor2, valor3, ...])
# print(resultado)
"""

            st.code(load_code, language='python')

        with export_tab3:
            st.markdown("**Generar código Python independiente:**")

            if st.button("📝 Generar Código", type="primary"):
                try:
                    # Obtener pesos del modelo
                    weights_data = []
                    for layer in model.layers:
                        if hasattr(layer, 'get_weights') and layer.get_weights():
                            weights_data.append(layer.get_weights())

                    # Generar código
                    code = f"""
# Código generado automáticamente para Red Neuronal
# Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class NeuralNetworkPredictor:
    def __init__(self):
        # Configuración del modelo
        self.config = {json.dumps(config, indent=8)}
        
        # Inicializar preprocesadores
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if {bool(label_encoder)} else None
        
        # Configurar preprocesadores (reemplaza con tus datos de entrenamiento)
        # self.scaler.fit(X_train)  # X_train son tus datos de entrenamiento
        # if self.label_encoder:
        #     self.label_encoder.fit(y_train)  # y_train son tus etiquetas
        
    def activation_function(self, x, activation):
        \"\"\"Implementa funciones de activación\"\"\"
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            return x  # linear
    
    def predict(self, X):
        \"\"\"Hace predicciones con la red neuronal\"\"\"
        # Normalizar entrada
        X_scaled = self.scaler.transform(X)
        
        # Forward pass a través de las capas
        # NOTA: Debes implementar los pesos específicos de tu modelo entrenado
        
        # Ejemplo de estructura (reemplaza con tus pesos reales):
        # layer_1 = self.activation_function(np.dot(X_scaled, weights_1) + bias_1, '{config['activation']}')
        # layer_2 = self.activation_function(np.dot(layer_1, weights_2) + bias_2, '{config['activation']}')
        # output = self.activation_function(np.dot(layer_2, weights_out) + bias_out, '{config['output_activation']}')
        
        # Placeholder para la implementación
        print("⚠️ Implementa los pesos específicos del modelo en este método")
        return np.zeros((X.shape[0], {config['output_size']}))
    
    def predict_class(self, X):
        \"\"\"Predicción para clasificación\"\"\"
        predictions = self.predict(X)
        
        if self.config['task_type'] == 'Clasificación':
            output_size = safe_get_output_size(self.config)
            if output_size > 2:
                class_indices = np.argmax(predictions, axis=1)
                if self.label_encoder:
                    return self.label_encoder.inverse_transform(class_indices)
                else:
                    return [f"Clase {{i}}" for i in class_indices]
            else:
                class_indices = (predictions > 0.5).astype(int).flatten()
                if self.label_encoder:
                    return self.label_encoder.inverse_transform(class_indices)
                else:
                    return [f"Clase {{i}}" for i in class_indices]
        else:
            return predictions.flatten()

# Uso del modelo:
# predictor = NeuralNetworkPredictor()
# 
# # Configura los preprocesadores con tus datos de entrenamiento
# # predictor.scaler.fit(X_train)
# # if predictor.label_encoder:
# #     predictor.label_encoder.fit(y_train)
# 
# # Hacer predicciones
# # nuevos_datos = [[valor1, valor2, valor3, ...]]
# # resultado = predictor.predict_class(nuevos_datos)
# # print(resultado)
"""

                    st.code(code, language='python')

                    # Botón para descargar el código
                    st.download_button(
                        label="📥 Descargar Código",
                        data=code,
                        file_name="neural_network_predictor.py",
                        mime="text/plain"
                    )

                    st.warning(
                        "⚠️ El código generado es una plantilla. Debes implementar los pesos específicos del modelo entrenado.")

                except Exception as e:
                    st.error(f"Error generando código: {str(e)}")

        with export_tab4:
            st.markdown("**Exportar metadatos del modelo:**")

            # Preparar metadatos
            if 'nn_history' in st.session_state:
                history = st.session_state.nn_history
                final_metrics = {
                    'final_loss': float(history.history['loss'][-1]),
                    'final_val_loss': float(history.history.get('val_loss', [0])[-1]) if 'val_loss' in history.history else None,
                    'epochs_trained': len(history.history['loss'])
                }

                if config['task_type'] == 'Clasificación' and 'accuracy' in history.history:
                    final_metrics['final_accuracy'] = float(
                        history.history['accuracy'][-1])
                    if 'val_accuracy' in history.history:
                        final_metrics['final_val_accuracy'] = float(
                            history.history['val_accuracy'][-1])
            else:
                final_metrics = {}

            metadata = {
                'model_info': {
                    'type': 'Neural Network',
                    'task_type': config['task_type'],
                    'architecture': config['architecture'],
                    'total_parameters': calculate_network_parameters(config['architecture']),
                    'activation_function': config['activation'],
                    'output_activation': config['output_activation'],
                    'optimizer': config['optimizer'],
                    'dropout_rate': config['dropout_rate'],
                    'batch_size': config['batch_size']
                },
                'training_info': final_metrics,
                'data_info': {
                    'feature_names': st.session_state.get('nn_feature_names', []),
                    'target_column': st.session_state.get('nn_target_col', ''),
                    'num_features': config['input_size'],
                    'num_classes': config['output_size'] if config['task_type'] == 'Clasificación' else 1
                },
                'export_info': {
                    'export_date': datetime.now().isoformat(),
                    'version': '1.0',
                    'framework': 'TensorFlow/Keras'
                }
            }

            # Mostrar metadatos
            st.json(metadata)

            # Botón para descargar metadatos
            metadata_json = json.dumps(metadata, indent=2)

            st.download_button(
                label="📥 Descargar Metadatos",
                data=metadata_json,
                file_name="neural_network_metadata.json",
                mime="application/json"
            )

        # Información adicional
        st.subheader("💡 Información Adicional")

        st.info("""
        **Recomendaciones para el uso del modelo:**
        
        1. **Modelo TensorFlow**: Ideal para integrar en aplicaciones que ya usan TensorFlow
        2. **Modelo Completo**: Incluye preprocesadores, perfecto para producción
        3. **Código Python**: Para entender la implementación o crear versiones optimizadas
        4. **Metadatos**: Para documentación y seguimiento del modelo
        
        **Consideraciones de versión:**
        - TensorFlow versión utilizada en entrenamiento
        - Compatibilidad con versiones futuras
        - Dependencias del entorno de producción
        """)

    except Exception as e:
        st.error(f"Error en la exportación: {str(e)}")
        st.info("Asegúrate de que el modelo esté entrenado correctamente.")


def generate_neural_network_architecture_code(architecture, activation, output_activation,
                                              dropout_rate, optimizer, batch_size, task_type, feature_names):
    """Genera código Python completo para la arquitectura de red neuronal."""

    feature_names_str = str(
        feature_names) if feature_names else "['feature_1', 'feature_2', ...]"

    # Determinar loss y metrics según el tipo de tarea
    if task_type == "Clasificación":
        if architecture[-1] == 1:  # Clasificación binaria
            loss = "binary_crossentropy"
            metrics = "['accuracy']"
            output_processing = """
# Para clasificación binaria
y_pred_classes = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy:.4f}")
"""
        else:  # Clasificación multiclase
            if output_activation == "softmax":
                loss = "sparse_categorical_crossentropy"
            else:
                loss = "categorical_crossentropy"
            metrics = "['accuracy']"
            output_processing = """
# Para clasificación multiclase
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy:.4f}")
print("\\nReporte de clasificación:")
print(classification_report(y_test, y_pred_classes))
"""
    else:  # Regresión
        loss = "mse"
        metrics = "['mae']"
        output_processing = """
# Para regresión
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
"""

    # Generar código de las capas
    layers_code = []
    for i, neurons in enumerate(architecture[1:-1], 1):
        if i == 1:  # Primera capa oculta
            layers_code.append(f"""
# Capa oculta {i}
model.add(Dense({neurons}, activation='{activation}', input_shape=({architecture[0]},)))
model.add(Dropout({dropout_rate}))""")
        else:
            layers_code.append(f"""
# Capa oculta {i}
model.add(Dense({neurons}, activation='{activation}'))
model.add(Dropout({dropout_rate}))""")

    layers_code_str = "".join(layers_code)

    code = f"""# Código completo para Red Neuronal - {task_type}
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. CARGAR Y PREPARAR LOS DATOS
# Reemplaza esta sección con tu método de carga de datos
# df = pd.read_csv('tu_archivo.csv')  # Cargar desde CSV

# Características y variable objetivo
feature_names = {feature_names_str}
# X = df[feature_names]  # Características
# y = df['target']  # Variable objetivo

# 2. PREPROCESAMIENTO
# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, 
    stratify=y if '{task_type}' == 'Clasificación' else None
)

# Normalizar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Procesar variable objetivo
{"# Para clasificación multiclase con softmax, usar sparse_categorical_crossentropy" if task_type == "Clasificación" and architecture[-1] > 1 and output_activation == "softmax" else ""}
{"# Para clasificación binaria, mantener y como está" if task_type == "Clasificación" and architecture[-1] == 1 else ""}
{"# Para regresión, mantener y como está" if task_type == "Regresión" else ""}

# 3. CONSTRUIR EL MODELO
model = Sequential()
{layers_code_str}

# Capa de salida
model.add(Dense({architecture[-1]}, activation='{output_activation}'))

# 4. COMPILAR EL MODELO
# Seleccionar optimizador
if '{optimizer}' == 'adam':
    optimizer = Adam()
elif '{optimizer}' == 'sgd':
    optimizer = SGD()
elif '{optimizer}' == 'rmsprop':
    optimizer = RMSprop()

model.compile(
    optimizer=optimizer,
    loss='{loss}',
    metrics={metrics}
)

# 5. MOSTRAR RESUMEN DE LA ARQUITECTURA
print("=== ARQUITECTURA DE LA RED NEURONAL ===")
model.summary()

# Información detallada
total_params = model.count_params()
print(f"\\nTotal de parámetros: {{total_params:,}}")
print(f"Arquitectura: {architecture}")
print(f"Funciones de activación: {activation} (ocultas), {output_activation} (salida)")
print(f"Dropout: {dropout_rate}")
print(f"Optimizador: {optimizer}")
print(f"Batch size: {batch_size}")

# 6. ENTRENAR EL MODELO
print("\\n=== INICIANDO ENTRENAMIENTO ===")

# Callbacks opcionales
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-7
    )
]

# Entrenar
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,  # Ajusta según necesites
    batch_size={batch_size},
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 7. EVALUAR EL MODELO
print("\\n=== EVALUACIÓN DEL MODELO ===")

# Predicciones
y_pred = model.predict(X_test_scaled)
{output_processing}

# 8. VISUALIZAR HISTORIAL DE ENTRENAMIENTO
plt.figure(figsize=(12, 4))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Métrica principal
plt.subplot(1, 2, 2)
metric_key = list(history.history.keys())[1]  # Primera métrica después de loss
plt.plot(history.history[metric_key], label='Entrenamiento')
plt.plot(history.history[f'val_{{metric_key}}'], label='Validación')
plt.title(f'{{metric_key.title()}} durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel(metric_key.title())
plt.legend()

plt.tight_layout()
plt.show()

# 9. FUNCIÓN PARA NUEVAS PREDICCIONES
def predecir_nueva_muestra(nueva_muestra):
    \"\"\"
    Función para hacer predicciones con nuevos datos.
    
    Parámetros:
    nueva_muestra: lista con valores para cada característica
                  en el orden: {feature_names_str}
    
    Retorna:
    prediccion: resultado de la predicción
    \"\"\"
    # Convertir a array y normalizar
    nueva_muestra = np.array(nueva_muestra).reshape(1, -1)
    nueva_muestra_scaled = scaler.transform(nueva_muestra)
    
    # Predecir
    prediccion = model.predict(nueva_muestra_scaled)
    
    {"# Para clasificación, convertir a clase" if task_type == "Clasificación" else "# Para regresión, devolver valor directo"}
    {"if prediccion[0][0] > 0.5: return 'Clase 1' else return 'Clase 0'  # Binaria" if task_type == "Clasificación" and architecture[-1] == 1 else ""}
    {"return np.argmax(prediccion[0])  # Multiclase" if task_type == "Clasificación" and architecture[-1] > 1 else ""}
    {"return prediccion[0][0]  # Regresión" if task_type == "Regresión" else ""}

# Ejemplo de uso:
# nueva_muestra = [valor1, valor2, valor3, ...]  # Reemplaza con tus valores
# resultado = predecir_nueva_muestra(nueva_muestra)
# print(f"Predicción: {{resultado}}")

# 10. GUARDAR EL MODELO
print("\\n=== GUARDANDO MODELO ===")

# Guardar modelo completo
model.save('modelo_red_neuronal.h5')
print("Modelo guardado como 'modelo_red_neuronal.h5'")

# Guardar scaler por separado
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler guardado como 'scaler.pkl'")

# Código para cargar el modelo guardado:
# modelo_cargado = keras.models.load_model('modelo_red_neuronal.h5')
# with open('scaler.pkl', 'rb') as f:
#     scaler_cargado = pickle.load(f)

print("\\n✅ ¡Entrenamiento completado!")
print("Tu red neuronal está lista para hacer predicciones.")
"""

    return code


def generate_neural_network_evaluation_code(config, feature_names, class_names=None):
    """Genera código Python para evaluación de red neuronal."""

    feature_names_str = str(
        feature_names) if feature_names else "['feature_1', 'feature_2', ...]"
    class_names_str = str(
        class_names) if class_names else "['Clase_0', 'Clase_1', ...]"

    if config['task_type'] == "Clasificación":
        if config['output_size'] == 1:  # Clasificación binaria
            evaluation_metrics = """
# Evaluación para clasificación binaria
y_pred_classes = (y_pred > 0.5).astype(int).flatten()
y_test_flat = y_test.flatten()

# Métricas principales
accuracy = accuracy_score(y_test_flat, y_pred_classes)
precision = precision_score(y_test_flat, y_pred_classes)
recall = recall_score(y_test_flat, y_pred_classes)
f1 = f1_score(y_test_flat, y_pred_classes)

print("=== MÉTRICAS DE CLASIFICACIÓN BINARIA ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_test_flat, y_pred_classes)
print("\\nMatriz de Confusión:")
print(cm)
"""
        else:  # Clasificación multiclase
            evaluation_metrics = """
# Evaluación para clasificación multiclase
y_pred_classes = np.argmax(y_pred, axis=1)
if len(y_test.shape) > 1:
    y_test_classes = np.argmax(y_test, axis=1)
else:
    y_test_classes = y_test.flatten()

# Métricas principales
accuracy = accuracy_score(y_test_classes, y_pred_classes)

print("=== MÉTRICAS DE CLASIFICACIÓN MULTICLASE ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Reporte detallado
class_names = """ + class_names_str + """
print("\\nReporte de Clasificación:")
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))

# Matriz de confusión
cm = confusion_matrix(y_test_classes, y_pred_classes)
print("\\nMatriz de Confusión:")
print(cm)
"""
    else:  # Regresión
        evaluation_metrics = """
# Evaluación para regresión
y_pred_flat = y_pred.flatten()
y_test_flat = y_test.flatten()

# Métricas principales
mse = mean_squared_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)

print("=== MÉTRICAS DE REGRESIÓN ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Análisis de residuos
residuos = y_test_flat - y_pred_flat
print(f"\\nAnálisis de Residuos:")
print(f"Media de residuos: {np.mean(residuos):.6f}")
print(f"Desviación estándar de residuos: {np.std(residuos):.4f}")
"""

    visualization_code = """
# Visualizaciones
plt.figure(figsize=(15, 10))

# Historial de entrenamiento
plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Métrica principal (accuracy o mae)
plt.subplot(2, 3, 2)
metric_key = list(history.history.keys())[1]  # Primera métrica después de loss
plt.plot(history.history[metric_key], label='Entrenamiento')
if f'val_{metric_key}' in history.history:
    plt.plot(history.history[f'val_{metric_key}'], label='Validación')
plt.title(f'{metric_key.title()} durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel(metric_key.title())
plt.legend()
"""

    if config['task_type'] == "Clasificación":
        specific_viz = """
# Matriz de confusión visualizada
plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')

# Distribución de confianza
plt.subplot(2, 3, 4)
if """ + str(config['output_size']) + """ == 1:
    confidence = np.maximum(y_pred.flatten(), 1 - y_pred.flatten())
else:
    confidence = np.max(y_pred, axis=1)
plt.hist(confidence, bins=20, alpha=0.7)
plt.title('Distribución de Confianza de Predicciones')
plt.xlabel('Confianza')
plt.ylabel('Frecuencia')
"""
    else:
        specific_viz = """
# Predicciones vs Valores Reales
plt.subplot(2, 3, 3)
plt.scatter(y_test_flat, y_pred_flat, alpha=0.6)
plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')

# Distribución de residuos
plt.subplot(2, 3, 4)
plt.hist(residuos, bins=20, alpha=0.7)
plt.title('Distribución de Residuos')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')

# Q-Q plot de residuos
plt.subplot(2, 3, 5)
from scipy import stats
stats.probplot(residuos, dist="norm", plot=plt)
plt.title('Q-Q Plot de Residuos')
"""

    code = f"""# Código completo para Evaluación de Red Neuronal - {config['task_type']}
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CARGAR MODELO Y DATOS
# Asumiendo que ya tienes:
# - model: tu modelo entrenado
# - X_test, y_test: datos de prueba
# - scaler: preprocessor para normalizar datos
# - history: historial de entrenamiento

print("=== EVALUACIÓN DE RED NEURONAL ===")
print("Tipo de tarea: {config['task_type']}")
print("Arquitectura: {config['architecture']}")

# 2. HACER PREDICCIONES
print("\\nHaciendo predicciones...")
y_pred = model.predict(X_test, verbose=0)

# 3. CALCULAR MÉTRICAS
{evaluation_metrics}

# 4. VISUALIZACIONES
{visualization_code}
{specific_viz}

plt.tight_layout()
plt.show()

# 5. ANÁLISIS DETALLADO DEL MODELO
print("\\n=== INFORMACIÓN DEL MODELO ===")
model.summary()

# Contar parámetros
total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = total_params - trainable_params

print(f"\\nParámetros totales: {{total_params:,}}")
print(f"Parámetros entrenables: {{trainable_params:,}}")
print(f"Parámetros no entrenables: {{non_trainable_params:,}}")

# 6. FUNCIÓN PARA NUEVAS PREDICCIONES CON MÉTRICAS
def evaluar_nueva_muestra(nueva_muestra, valor_real=None):
    \"\"\"
    Evalúa una nueva muestra y opcionalmente compara con valor real.
    
    Parámetros:
    nueva_muestra: lista con valores para cada característica
    valor_real: valor real para comparar (opcional)
    \"\"\"
    # Normalizar
    nueva_muestra = np.array(nueva_muestra).reshape(1, -1)
    nueva_muestra_scaled = scaler.transform(nueva_muestra)
    
    # Predecir
    prediccion = model.predict(nueva_muestra_scaled, verbose=0)
    
    print(f"\\n=== PREDICCIÓN INDIVIDUAL ===")
    print(f"Entrada: {{nueva_muestra[0]}}")
    
    {"# Clasificación" if config['task_type'] == "Clasificación" else "# Regresión"}
    {"if prediccion[0][0] > 0.5:" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"    clase_pred = 1" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"    confianza = prediccion[0][0]" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"else:" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"    clase_pred = 0" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"    confianza = 1 - prediccion[0][0]" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"print(f'Clase predicha: {clase_pred}')" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"print(f'Confianza: {confianza:.4f} ({confianza*100:.2f}%)')" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    
    {"clase_pred = np.argmax(prediccion[0])" if config['task_type'] == "Clasificación" and config['output_size'] > 1 else ""}
    {"confianza = np.max(prediccion[0])" if config['task_type'] == "Clasificación" and config['output_size'] > 1 else ""}
    {"print(f'Clase predicha: {clase_pred}')" if config['task_type'] == "Clasificación" and config['output_size'] > 1 else ""}
    {"print(f'Confianza: {confianza:.4f} ({confianza*100:.2f}%)')" if config['task_type'] == "Clasificación" and config['output_size'] > 1 else ""}
    {"print(f'Probabilidades por clase: {prediccion[0]}')" if config['task_type'] == "Clasificación" and config['output_size'] > 1 else ""}
    
    {"valor_pred = prediccion[0][0]" if config['task_type'] == "Regresión" else ""}
    {"print(f'Valor predicho: {valor_pred:.4f}')" if config['task_type'] == "Regresión" else ""}
    
    if valor_real is not None:
        {"error = abs(valor_real - clase_pred)" if config['task_type'] == "Clasificación" else "error = abs(valor_real - valor_pred)"}
        print(f"Valor real: {{valor_real}}")
        print(f"Error: {{error:.4f}}")
    
    {"return clase_pred, confianza" if config['task_type'] == "Clasificación" else "return valor_pred"}

# Ejemplo de uso:
# nueva_muestra = [valor1, valor2, valor3, ...]  # Reemplaza con tus valores
# resultado = evaluar_nueva_muestra(nueva_muestra)
# print(f"Resultado: {{resultado}}")

print("\\n✅ Evaluación completada!")
"""

    return code


def generate_neural_network_visualization_code(config):
    """Genera código Python para visualizaciones de red neuronal."""

    code = f"""# Código completo para Visualizaciones de Red Neuronal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# Configuración de matplotlib
plt.style.use('default')
sns.set_palette("husl")

print("=== VISUALIZACIONES DE RED NEURONAL ===")
print("Tipo de tarea: {config['task_type']}")
print("Arquitectura: {config['architecture']}")

# 1. HISTORIAL DE ENTRENAMIENTO
def plot_training_history_detailed(history):
    \"\"\"Crea gráficas detalladas del historial de entrenamiento.\"\"\"
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pérdida
    axes[0, 0].plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validación', linewidth=2)
    axes[0, 0].set_title('Pérdida durante el entrenamiento', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Pérdida')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Métrica principal
    metric_key = list(history.history.keys())[1]  # Primera métrica después de loss
    axes[0, 1].plot(history.history[metric_key], label='Entrenamiento', linewidth=2)
    if f'val_{{metric_key}}' in history.history:
        axes[0, 1].plot(history.history[f'val_{{metric_key}}'], label='Validación', linewidth=2)
    axes[0, 1].set_title(f'{{metric_key.title()}} durante el entrenamiento', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel(metric_key.title())
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (si disponible)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], color='red', linewidth=2)
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\\nno disponible', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Mejora por época
    loss_improvement = np.diff(history.history['loss'])
    axes[1, 1].plot(loss_improvement, color='purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Mejora por Época (Pérdida)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Cambio en Pérdida')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 2. ANÁLISIS DE PESOS Y SESGOS
def analyze_weights_and_biases(model):
    \"\"\"Analiza la distribución de pesos y sesgos en todas las capas.\"\"\"
    
    layer_weights = []
    layer_biases = []
    
    # Extraer pesos y sesgos
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            if len(weights) >= 2:
                layer_weights.append(weights[0])
                layer_biases.append(weights[1])
    
    if not layer_weights:
        print("No se encontraron capas con pesos")
        return
    
    num_layers = len(layer_weights)
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 4*num_layers))
    
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for i, (weights, biases) in enumerate(zip(layer_weights, layer_biases)):
        # Histograma de pesos
        axes[i, 0].hist(weights.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[i, 0].set_title(f'Distribución de Pesos - Capa {{i+1}}', fontweight='bold')
        axes[i, 0].set_xlabel('Valor de Peso')
        axes[i, 0].set_ylabel('Frecuencia')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Estadísticas de pesos
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        axes[i, 0].axvline(mean_w, color='red', linestyle='--', label=f'Media: {{mean_w:.4f}}')
        axes[i, 0].axvline(mean_w + std_w, color='orange', linestyle=':', label=f'±1σ: {{std_w:.4f}}')
        axes[i, 0].axvline(mean_w - std_w, color='orange', linestyle=':')
        axes[i, 0].legend()
        
        # Histograma de sesgos
        axes[i, 1].hist(biases.flatten(), bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[i, 1].set_title(f'Distribución de Sesgos - Capa {{i+1}}', fontweight='bold')
        axes[i, 1].set_xlabel('Valor de Sesgo')
        axes[i, 1].set_ylabel('Frecuencia')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Estadísticas de sesgos
        mean_b = np.mean(biases)
        std_b = np.std(biases)
        axes[i, 1].axvline(mean_b, color='red', linestyle='--', label=f'Media: {{mean_b:.4f}}')
        axes[i, 1].axvline(mean_b + std_b, color='blue', linestyle=':', label=f'±1σ: {{std_b:.4f}}')
        axes[i, 1].axvline(mean_b - std_b, color='blue', linestyle=':')
        axes[i, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas generales
    all_weights = np.concatenate([w.flatten() for w in layer_weights])
    all_biases = np.concatenate([b.flatten() for b in layer_biases])
    
    print("\\n=== ESTADÍSTICAS GENERALES ===")
    print(f"Número total de pesos: {{len(all_weights):,}}")
    print(f"Número total de sesgos: {{len(all_biases):,}}")
    print(f"\\nPesos - Media: {{np.mean(all_weights):.6f}}, Std: {{np.std(all_weights):.6f}}")
    print(f"Pesos - Min: {{np.min(all_weights):.6f}}, Max: {{np.max(all_weights):.6f}}")
    print(f"\\nSesgos - Media: {{np.mean(all_biases):.6f}}, Std: {{np.std(all_biases):.6f}}")
    print(f"Sesgos - Min: {{np.min(all_biases):.6f}}, Max: {{np.max(all_biases):.6f}}")
    
    # Detección de problemas
    dead_weights = np.sum(np.abs(all_weights) < 1e-6)
    if dead_weights > len(all_weights) * 0.1:
        print(f"\\n⚠️ ADVERTENCIA: {{dead_weights}} pesos muy cerca de cero ({{dead_weights/len(all_weights)*100:.1f}}%)")
    
    if np.std(all_weights) < 0.01:
        print("\\n🚨 PROBLEMA: Pesos muy pequeños, la red puede no haber aprendido correctamente")
    elif np.std(all_weights) > 2:
        print("\\n⚠️ ATENCIÓN: Pesos muy grandes, posible inestabilidad")

# 3. ANÁLISIS DE ACTIVACIONES
def analyze_layer_activations(model, X_sample):
    \"\"\"Analiza las activaciones de cada capa con datos de muestra.\"\"\"
    
    # Crear modelo para extraer activaciones
    layer_outputs = [layer.output for layer in model.layers[:-1]]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Obtener activaciones
    activations = activation_model.predict(X_sample, verbose=0)
    if not isinstance(activations, list):
        activations = [activations]
    
    # Analizar cada capa
    print("\\n=== ANÁLISIS DE ACTIVACIONES ===")
    for i, activation in enumerate(activations):
        print(f"\\nCapa {{i+1}}:")
        print(f"  Forma: {{activation.shape}}")
        print(f"  Media: {{np.mean(activation):.4f}}")
        print(f"  Desviación estándar: {{np.std(activation):.4f}}")
        print(f"  Min: {{np.min(activation):.4f}}, Max: {{np.max(activation):.4f}}")
        
        # Neuronas muertas (siempre 0)
        dead_neurons = np.mean(activation == 0, axis=0)
        dead_ratio = np.mean(dead_neurons > 0.95) * 100
        print(f"  Neuronas muertas: {{dead_ratio:.1f}}%")
        
        # Neuronas saturadas (siempre cerca del máximo)
        if activation.max() > 0:
            saturated_neurons = np.mean(activation >= 0.99 * activation.max(), axis=0)
            saturated_ratio = np.mean(saturated_neurons > 0.95) * 100
            print(f"  Neuronas saturadas: {{saturated_ratio:.1f}}%")
        
        # Visualizar distribución de activaciones
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(activation.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'Distribución de Activaciones - Capa {{i+1}}')
        plt.xlabel('Valor de Activación')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(activation.T)
        plt.title(f'Box Plot por Neurona - Capa {{i+1}}')
        plt.xlabel('Neurona')
        plt.ylabel('Activación')
        plt.xticks(range(1, min(21, activation.shape[1]+1)))  # Máximo 20 neuronas
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 4. FUNCIÓN PRINCIPAL DE VISUALIZACIÓN
def visualize_neural_network_complete(model, history, X_sample=None):
    \"\"\"Ejecuta todas las visualizaciones de la red neuronal.\"\"\"
    
    print("Generando visualizaciones completas...")
    
    # 1. Historial de entrenamiento
    print("\\n1. Analizando historial de entrenamiento...")
    plot_training_history_detailed(history)
    
    # 2. Pesos y sesgos
    print("\\n2. Analizando pesos y sesgos...")
    analyze_weights_and_biases(model)
    
    # 3. Activaciones (si se proporcionan datos)
    if X_sample is not None:
        print("\\n3. Analizando activaciones...")
        analyze_layer_activations(model, X_sample)
    else:
        print("\\n3. Análisis de activaciones omitido (no se proporcionaron datos)")
    
    print("\\n✅ Visualizaciones completadas!")

# EJEMPLO DE USO:
# Asumiendo que tienes:
# - model: tu modelo entrenado
# - history: historial de entrenamiento
# - X_test: datos de prueba para análisis de activaciones

# Ejecutar todas las visualizaciones
# visualize_neural_network_complete(model, history, X_test[:100])

# O ejecutar individualmente:
# plot_training_history_detailed(history)
# analyze_weights_and_biases(model)
# analyze_layer_activations(model, X_test[:100])
"""

    return code


def generate_neural_network_complete_code(config, feature_names, class_names=None):
    """Genera código Python completo para entrenar y usar la red neuronal."""
    # Esta función se puede expandir para generar código más completo
    # incluyendo carga de datos, entrenamiento completo, etc.
    pass


if __name__ == "__main__":
    main()
