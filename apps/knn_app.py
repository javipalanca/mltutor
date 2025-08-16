import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from dataset.dataset_manager import create_dataset_selector, load_data, preprocess_data
from algorithms.model_training import train_knn_model
from algorithms.model_evaluation import evaluate_classification_model, evaluate_regression_model, show_detailed_evaluation
from ui import create_prediction_interface, create_button_panel
from dataset.dataset_tab import run_dataset_tab
from utils import create_info_box, get_image_download_link, show_code_with_download
from viz.roc import plot_roc_curve, plot_threshold_analysis
from viz.decision_boundary import plot_decision_surface, plot_decision_boundary
from viz.residual import plot_predictions, plot_residuals
from viz.features import display_feature_importance


def run_knn_app():
    """Ejecuta la aplicación específica de K-Nearest Neighbors (KNN) para clasificación y regresión."""
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
        "🔍 Características",
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

    # Estado de datos y modelo
    if 'knn_dataset' not in st.session_state:
        st.session_state.knn_dataset = None
    if 'knn_X' not in st.session_state:
        st.session_state.knn_X = None
    if 'knn_y' not in st.session_state:
        st.session_state.knn_y = None
    if 'knn_Xtrain' not in st.session_state:
        st.session_state.knn_Xtrain = None
    if 'knn_Xtest' not in st.session_state:
        st.session_state.knn_Xtest = None
    if 'knn_ytrain' not in st.session_state:
        st.session_state.knn_ytrain = None
    if 'knn_ytest' not in st.session_state:
        st.session_state.knn_ytest = None
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

    ###########################################
    # Pestaña de Datos                        #
    ###########################################
    run_dataset_tab(tab)

    ###########################################
    # Pestaña de Entrenamiento                #
    ###########################################
    if tab == 1:
        st.header("Configuración del Modelo KNN")

        # Inicializar variables de sesión necesarias
        if 'dataset_option_knn' not in st.session_state:
            st.session_state.dataset_option_knn = st.session_state.selected_dataset

        # Cargar datos para la vista previa si cambia el dataset o si no se ha cargado
        if st.session_state.selected_dataset != st.session_state.dataset_option_knn or st.session_state.knn_X is None:
            try:
                X, y, feature_names, class_names, info, task_type = load_data(
                    st.session_state.selected_dataset)

                st.session_state.dataset_option_knn = st.session_state.selected_dataset

                # Actualizar información del dataset
                st.session_state.knn_dataset = st.session_state.selected_dataset
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

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                n_neighbors = st.number_input(
                    "Vecinos (K)", min_value=1, max_value=20, value=5, step=1, key="knn_n_neighbors")
            with col2:
                weights = st.selectbox(
                    "Pesos", options=["uniform", "distance"], key="knn_weights")
            with col3:
                metric = st.selectbox("Métrica", options=[
                                      "minkowski", "euclidean", "manhattan"], key="knn_metric")

            with col4:
                test_size = st.slider(
                    "Tamaño del Conjunto de Prueba:",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.3,
                    step=0.05,
                    help="Proporción de datos que se usará para evaluar el modelo."
                )

            if st.button("🚀 Entrenar Modelo KNN", key="train_knn_button", type="primary"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        Xtrain, Xtest, ytrain, ytest = preprocess_data(
                            X=st.session_state.knn_X,
                            y=st.session_state.knn_y,
                            test_size=test_size
                        )

                        model = train_knn_model(
                            Xtrain,
                            ytrain,
                            task_type=st.session_state.knn_task_type,
                            n_neighbors=n_neighbors,
                            weights=weights,
                            metric=metric,
                        )
                        st.session_state.knn_Xtrain = Xtrain
                        st.session_state.knn_ytrain = ytrain
                        st.session_state.knn_Xtest = Xtest
                        st.session_state.knn_ytest = ytest
                        st.session_state.knn_model = model
                        st.session_state.knn_trained = True
                        st.success("¡Modelo KNN entrenado correctamente!")

                        # Sugerir ir a la pestaña de evaluación
                        st.info(
                            "👉 Ve a la pestaña '📈 Evaluación' para ver los resultados del modelo.")

                    except Exception as e:
                        st.error(f"Error al entrenar el modelo: {str(e)}")
        else:
            st.info("Primero selecciona y carga un dataset en la pestaña de Datos.")

    ###########################################
    # Pestaña de Evaluación.                  #
    ###########################################
    elif tab == 2:
        st.header("📈 Evaluación del Modelo KNN")
        if st.session_state.knn_trained:  # and st.session_state.knn_metrics is not None:
            # Obtener las predicciones del modelo
            if hasattr(st.session_state, 'knn_Xtest') and hasattr(st.session_state, 'knn_ytest'):

                # Obtener las predicciones
                y_pred = st.session_state.knn_model.predict(
                    st.session_state.knn_Xtest)

                # Mostrar evaluación detallada del modelo usando la misma función que otros algoritmos
                show_detailed_evaluation(
                    st.session_state.knn_ytest,
                    y_pred,
                    st.session_state.knn_class_names if st.session_state.knn_task_type == "Clasificación" else None,
                    st.session_state.knn_task_type
                )
            else:
                st.error(
                    "No se encontraron los datos necesarios para la evaluación.")
        else:
            st.info("Primero entrena un modelo KNN.")

    ###########################################
    # Pestaña de Visualización                #
    ###########################################
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
            if task_type == "Clasificación":
                viz_options.extend([
                    ("📊 Distribución de Probabilidades",
                     "Probabilidades", "viz_prob"),
                    ("📉 Curva ROC", "ROC", "viz_roc"),
                    ("🌈 Frontera de Decisión", "Frontera", "viz_boundary"),
                    ("🎮 Visualización Interactiva",
                     "Interactiva", "viz_interactive"),
                    ("📏 Análisis de Distancias", "Distancias", "viz_distances")
                ])
            else:
                viz_options.append(
                    ("📊 Distribución de Predicciones", "Predicciones", "viz_predictions"))
                viz_options.append(
                    ("🌐 Superficie de Predicción", "Superficie", "viz_surface"))

            viz_type = create_button_panel(viz_options)

            if viz_type == "Probabilidades":
                st.markdown("### 📊 Distribución de Probabilidades")
                y_test = st.session_state.knn_ytest
                X_test = st.session_state.knn_Xtest
                model = st.session_state.knn_model
                # Obtener probabilidades de predicción
                y_pred_proba = model.predict_proba(X_test)
                plot_threshold_analysis(
                    y_test, y_pred_proba, class_names=class_names)

            if viz_type == "ROC":
                model = st.session_state.knn_model
                X_test = st.session_state.knn_Xtest
                y_test = st.session_state.knn_ytest
                # Obtener probabilidades de predicción
                y_pred_proba = model.predict_proba(X_test)
                plot_roc_curve(y_test, y_pred_proba)

            elif viz_type == "Predicciones":
                model = st.session_state.knn_model
                X_test = st.session_state.knn_Xtest
                y_test = st.session_state.knn_ytest
                y_pred = model.predict(X_test)

                # Crear visualizaciones con mejor tamaño
                st.markdown("### 📊 Gráfico de Predicciones vs Valores Reales")
                plot_predictions(y_test, y_pred)

                # Gráfico de residuos
                st.markdown("### 📈 Análisis de Residuos")
                plot_residuals(y_test, y_pred)

            elif viz_type == "Frontera":
                if not st.session_state.get('knn_trained', False):
                    st.warning(
                        "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'.")

                else:
                    model = st.session_state.knn_model
                    model_2d = KNeighborsClassifier(
                        n_neighbors=model.n_neighbors,
                        weights=model.weights,
                        metric=model.metric
                    )

                    plot_decision_boundary(
                        model_2d, X, y, feature_names, st.session_state.knn_class_names)

            elif viz_type == "Superficie":
                if not st.session_state.get('knn_trained', False):
                    st.warning(
                        "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'.")

                else:
                    model = st.session_state.knn_model
                    model_2d = KNeighborsRegressor(
                        n_neighbors=model.n_neighbors,
                        weights=model.weights,
                        metric=model.metric
                    )
                    plot_decision_surface(model_2d, feature_names, X, y)

            elif viz_type == "Interactiva":
                st.markdown("### Visualización Interactiva de KNN")
                st.markdown(
                    "Explora cómo el algoritmo KNN toma decisiones en tiempo real")

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

            elif viz_type == "Distancias":
                st.markdown("### Análisis de Distancias entre Clases")

                # Calcular distancias promedio entre clases
                from sklearn.metrics.pairwise import pairwise_distances

                # Recrear el split para análisis
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                # Convertir a numpy arrays para evitar problemas de indexación
                if hasattr(X_train, 'values'):
                    X_train = X_train.values
                if hasattr(y_train, 'values'):
                    y_train = y_train.values

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

    ###########################################
    # Pestaña de Características              #
    ###########################################
    elif tab == 4:
        st.header("🔍 Importancia de Características")
        if st.session_state.knn_trained and st.session_state.knn_model is not None:
            # Mostrar importancia de características usando permutation importance
            display_feature_importance(
                st.session_state.knn_model,
                st.session_state.knn_feature_names,
                X_test=st.session_state.get('knn_Xtest', None),
                y_test=st.session_state.get('knn_ytest', None),
                task_type=st.session_state.get('knn_task_type', 'Clasificación')
            )
        else:
            st.info("Primero entrena un modelo KNN para ver la importancia de características.")

    ###########################################
    # Pestaña de Predicciones                 #
    ###########################################
    elif tab == 5:
        st.header("🔮 Predicciones con Nuevos Datos")
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
