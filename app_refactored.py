#!/usr/bin/env python3
"""
MLTutor: Plataforma educativa para el aprendizaje de Machine Learning.

Esta es la versión refactorizada de la aplicación MLTutor, que separa la página de inicio
de los algoritmos específicos para una mejor experiencia de usuario.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

# Importar módulos refactorizados
from dataset_manager import load_data, preprocess_data, create_dataset_selector
from model_training import train_decision_tree, predict_sample
from model_evaluation import evaluate_classification_model, evaluate_regression_model, show_detailed_evaluation
from decision_boundary import plot_decision_boundary
from ui import (
    setup_page, init_session_state, show_welcome_page,
    display_feature_importance, display_model_export_options, create_prediction_interface
)
from tree_visualizer import (
    check_visualization_modules, render_tree_visualization,
    create_tree_visualization, get_tree_text
)
from utils import (
    get_image_download_link, generate_model_code, export_model_pickle, export_model_onnx,
    create_info_box, format_number, show_code_with_download
)


def main():
    """Función principal que ejecuta la aplicación MLTutor."""
    # Configuración de la página
    setup_page()

    # Inicializar estado de la sesión
    init_session_state()

    # Verificar disponibilidad de módulos de visualización
    viz_availability = check_visualization_modules()

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

    if st.sidebar.button("📊 Regresión Logística (próximamente)",
                         key="nav_logistic",
                         use_container_width=True,
                         disabled=True):
        st.session_state.navigation = "📊 Regresión Logística (próximamente)"
        st.rerun()

    if st.sidebar.button("🔍 K-Nearest Neighbors (próximamente)",
                         key="nav_knn",
                         use_container_width=True,
                         disabled=True):
        st.session_state.navigation = "🔍 K-Nearest Neighbors (próximamente)"
        st.rerun()

    if st.sidebar.button("🧠 Redes Neuronales (próximamente)",
                         key="nav_nn",
                         use_container_width=True,
                         disabled=True):
        st.session_state.navigation = "🧠 Redes Neuronales (próximamente)"
        st.rerun()

    # Página de inicio
    if st.session_state.navigation == "🏠 Inicio":
        show_welcome_page()
        return

    # Páginas de algoritmos
    if st.session_state.navigation == "🌲 Árboles de Decisión":
        run_decision_trees_app(viz_availability)
    elif st.session_state.navigation in ["📊 Regresión Logística (próximamente)",
                                         "🔍 K-Nearest Neighbors (próximamente)",
                                         "🧠 Redes Neuronales (próximamente)"]:
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


def run_decision_trees_app(viz_availability):
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
        "⚙️ Configuración",
        "📊 Datos",
        "📈 Evaluación",
        "🌲 Visualización",
        "🔍 Características",
        "🔮 Predicciones",
        "💾 Exportar Modelo"
    ]

    # Crear contenedor para los botones de las pestañas
    tab_cols = st.columns(len(tab_options))

    # Estilo CSS para los botones de pestañas
    st.markdown("""
    <style>
    div.tab-button > button {
        border-radius: 4px 4px 0 0;
        padding: 10px;
        width: 100%;
        white-space: nowrap;
        background-color: #F0F2F6;
        border-bottom: 2px solid #E0E0E0;
    }
    div.tab-button-active > button {
        background-color: #E3F2FD;
        border-bottom: 2px solid #1E88E5;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Crear botones para las pestañas
    for i, (tab_name, col) in enumerate(zip(tab_options, tab_cols)):
        button_key = f"tab_{i}"
        button_style = "tab-button-active" if st.session_state.active_tab == i else "tab-button"

        with col:
            st.markdown(f"<div class='{button_style}'>",
                        unsafe_allow_html=True)
            if st.button(tab_name, key=button_key, use_container_width=True):
                st.session_state.active_tab = i
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Separador visual
    st.markdown("---")

    # Pestaña de Configuración
    if st.session_state.active_tab == 0:
        st.header("Configuración del Modelo")

        # Selección de dataset
        dataset_option = st.selectbox(
            "Dataset de ejemplo:",
            ("Iris (clasificación de flores)",
             "Vino (clasificación de vinos)", "Cáncer de mama (diagnóstico)"),
            key="dataset_selector_config"
        )

        # Cargar datos para la vista previa si cambia el dataset o si no se ha cargado
        if dataset_option != st.session_state.dataset_option or not dataset_loaded:
            try:
                X, y, feature_names, class_names, dataset_info, task_type = load_data(
                    dataset_option)
                st.session_state.dataset_option = dataset_option
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

                    if st.button("🏷️ Clasificación",                                key="btn_classification",
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
                                 type="primary" if is_regression else "secondary",                                use_container_width=True,
                                 help="Para predecir valores numéricos continuos"):
                        tree_type = "Regresión"
                        st.session_state.tree_type = tree_type
                        st.rerun()

                # Obtener el valor actual del tipo de árbol
                tree_type = st.session_state.tree_type if "tree_type" in st.session_state else (
                    "Clasificación" if task_type == "Clasificación" else "Regresión"
                )

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
                ["gini", "entropy"] if st.session_state.tree_type == "Clasificación" else [
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
                        dataset_option)
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

    # Pestaña de Datos
    elif st.session_state.active_tab == 1:
        st.header("Exploración de Datos")

        if not st.session_state.is_trained:
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '⚙️ Configuración'.")
        else:
            # Mostrar información del dataset
            st.markdown("### Información del Dataset")
            st.markdown(create_info_box(
                st.session_state.dataset_info), unsafe_allow_html=True)

            # Mostrar las primeras filas de los datos
            st.markdown("### Vista previa de datos")
            X_df = pd.DataFrame(np.vstack((st.session_state.X_train, st.session_state.X_test)),
                                columns=st.session_state.feature_names)
            y_df = pd.Series(np.hstack((st.session_state.y_train, st.session_state.y_test)),
                             name="target")

            if st.session_state.tree_type == "Clasificación" and st.session_state.class_names is not None:
                y_df = y_df.map(
                    {i: name for i, name in enumerate(st.session_state.class_names)})

            df = pd.concat([X_df, y_df], axis=1)
            st.dataframe(df.head(10))

            # Estadísticas descriptivas
            st.markdown("### Estadísticas Descriptivas")
            st.dataframe(df.describe())

            # Distribución de clases o valores objetivo
            st.markdown("### Distribución del Objetivo")

            fig, ax = plt.subplots(figsize=(10, 6))

            if st.session_state.tree_type == "Clasificación":
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
                sns.histplot(y_df, kde=True, ax=ax)
                ax.set_title("Distribución de Valores Objetivo")
                ax.set_xlabel("Valor")
                ax.set_ylabel("Frecuencia")

            st.pyplot(fig)

            # Análisis de correlación
            st.markdown("### Matriz de Correlación")

            # Matriz de correlación
            corr = X_df.corr()

            # Generar máscara para el triángulo superior
            mask = np.triu(np.ones_like(corr, dtype=bool))

            # Generar mapa de calor
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                        square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
            ax.set_title("Matriz de Correlación de Características")

            st.pyplot(fig)

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
            available_features = X_df.columns.tolist()

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

            # Crear el dataframe para la visualización
            plot_df = X_df[selected_features].copy()
            # Añadir la variable objetivo para colorear
            plot_df['target'] = y_df

            # Generar el pairplot
            with st.spinner("Generando matriz de dispersión..."):
                pair_fig = plt.figure(figsize=(12, 10))
                pair_plot = sns.pairplot(
                    plot_df,
                    hue='target',
                    diag_kind=diag_kind,
                    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
                    diag_kws={'alpha': 0.5},
                    height=2.5
                )
                pair_plot.fig.suptitle(
                    "Matriz de Dispersión de Características", y=1.02, fontsize=16)
                st.pyplot(pair_plot.fig)

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
mask = np.triu(np.ones_like(corr, dtype=bool))  # Máscara para triángulo superior

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
           square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Matriz de Correlación de Características")

plt.tight_layout()
plt.show()

# Matriz de dispersión (Scatterplot Matrix)
# Seleccionar características específicas para visualizar
selected_features = ['feature1', 'feature2', 'feature3']  # Reemplaza con tus características de interés
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
pair_plot.fig.suptitle("Matriz de Dispersión de Características", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
"""

            show_code_with_download(
                code, "Código para análisis exploratorio", "analisis_exploratorio.py")

    # Pestaña de Evaluación
    elif st.session_state.active_tab == 2:
        st.header("Evaluación del Modelo")

        if not st.session_state.is_trained:
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '⚙️ Configuración'.")
        else:
            # Obtener predicciones del modelo
            y_pred = st.session_state.tree_model.predict(
                st.session_state.X_test)

            # Mostrar evaluación detallada del modelo
            show_detailed_evaluation(
                st.session_state.y_test,
                y_pred,
                st.session_state.class_names if st.session_state.tree_type == "Clasificación" else None,
                st.session_state.tree_type
            )

            # Decisión boundary si es clasificación y tiene 2 características
            if st.session_state.tree_type == "Clasificación" and st.session_state.X_train.shape[1] <= 2:
                st.markdown("### Frontera de Decisión")

                if st.session_state.X_train.shape[1] == 1:
                    # Caso especial: solo una característica - agregar una segunda artificial
                    X_plot = np.column_stack(
                        [st.session_state.X_train, np.zeros_like(st.session_state.X_train)])
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
                    X_plot = st.session_state.X_train[:, feature_idx]
                    feature_names_plot = [feature1, feature2]

                # Generar y mostrar el plot
                fig = plot_decision_boundary(
                    st.session_state.tree_model,
                    X_plot,
                    st.session_state.y_train,
                    feature_names=feature_names_plot,
                    class_names=st.session_state.class_names
                )

                st.pyplot(fig)

                # Enlace para descargar
                st.markdown(
                    get_image_download_link(
                        fig, "frontera_decision", "📥 Descargar visualización de la frontera"),
                    unsafe_allow_html=True
                )

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
    fig, ax = plt.subplots(figsize=(10, 8))
    
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

        if not st.session_state.is_trained:
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '⚙️ Configuración'.")
        else:
            # Configuración de la visualización
            st.markdown("### Tipo de visualización")

            # Usar botones para seleccionar el tipo de visualización
            if "viz_type" not in st.session_state:
                st.session_state.viz_type = "Estándar"

            viz_col1, viz_col2, viz_col3 = st.columns(3)

            with viz_col1:
                if st.button("📊 Estándar",
                             key="viz_standard",
                             type="primary" if st.session_state.viz_type == "Estándar" else "secondary",
                             use_container_width=True):
                    st.session_state.viz_type = "Estándar"
                    st.rerun()

            with viz_col2:
                if st.button("🔍 Detallada",
                             key="viz_detailed",
                             type="primary" if st.session_state.viz_type == "Detallada" else "secondary",
                             use_container_width=True):
                    st.session_state.viz_type = "Detallada"
                    st.rerun()

            with viz_col3:
                if st.button("📝 Texto",
                             key="viz_text",
                             type="primary" if st.session_state.viz_type == "Texto" else "secondary",
                             use_container_width=True):
                    st.session_state.viz_type = "Texto"
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

                st.pyplot(fig)

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

            elif viz_type == "Detallada":
                # Verificar disponibilidad de visualización mejorada
                if viz_availability["graphviz"]:
                    # Visualización mejorada (requiere graphviz)
                    try:
                        # Mostrar visualización
                        viz_data = create_tree_visualization(
                            st.session_state.tree_model,
                            st.session_state.feature_names,
                            st.session_state.class_names if st.session_state.tree_type == "Clasificación" else None,
                            st.session_state.tree_type
                        )

                        render_tree_visualization(
                            viz_data, width=fig_width, height=fig_height)

                        # Mostrar código para generar esta visualización
                        code_detailed = """
from sklearn.tree import export_graphviz
import graphviz

def create_tree_visualization(model, feature_names, class_names=None, tree_type="Clasificación"):
    \"\"\"
    Crea una visualización detallada de un árbol de decisión usando graphviz.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol entrenado
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases (para clasificación)
    tree_type : str
        "Clasificación" o "Regresión"
        
    Returns:
    --------
    graphviz.Source
        Objeto de visualización
    \"\"\"
    # Exportar a DOT
    dot_data = export_graphviz(
        model,
        feature_names=feature_names,
        class_names=class_names if tree_type == "Clasificación" else None,
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=True,
        impurity=True,
        out_file=None
    )
    
    # Crear y devolver objeto de visualización
    return graphviz.Source(dot_data)

# Para usar:
# viz = create_tree_visualization(tree_model, feature_names, class_names, tree_type)
# viz.render("arbol_decision", format="png", cleanup=True)  # Guardar como imagen
# display(viz)  # Mostrar en Jupyter
"""

                        show_code_with_download(
                            code_detailed, "Código para visualización detallada", "visualizacion_detallada.py")

                    except Exception as e:
                        st.error(
                            f"Error al crear la visualización detallada: {str(e)}")
                        st.info(
                            "Vuelve a intentarlo con la visualización 'Estándar' o 'Texto'.")
                else:
                    st.warning(
                        "La visualización detallada requiere que Graphviz esté instalado.")
                    st.info(
                        "Puedes instalarlo con `pip install graphviz` y asegurarte de que el ejecutable de Graphviz esté en tu PATH.")
                    st.info(
                        "Mientras tanto, puedes usar la visualización 'Estándar' o 'Texto'.")

            else:  # Visualización de texto
                # Obtener representación de texto
                tree_text = get_tree_text(
                    st.session_state.tree_model,
                    st.session_state.feature_names,
                    show_class_name=st.session_state.tree_type == "Clasificación"
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

# Para usar:
# tree_text = get_tree_text(tree_model, feature_names)
# print(tree_text)

# Para guardar a un archivo:
# with open('arbol_texto.txt', 'w') as f:
#     f.write(tree_text)
"""

                show_code_with_download(
                    code_text, "Código para visualización de texto", "visualizacion_texto.py")

    # Pestaña de Características
    elif st.session_state.active_tab == 4:
        st.header("Importancia de Características")

        if not st.session_state.is_trained:
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '⚙️ Configuración'.")
        else:
            # Mostrar importancia de características
            display_feature_importance(
                st.session_state.tree_model,
                st.session_state.feature_names
            )

    # Pestaña de Predicciones
    elif st.session_state.active_tab == 5:
        st.header("Predicciones con Nuevos Datos")

        if not st.session_state.is_trained:
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '⚙️ Configuración'.")
        else:
            # Interfaz para hacer predicciones
            create_prediction_interface(
                st.session_state.tree_model,
                st.session_state.feature_names,
                st.session_state.class_names,
                st.session_state.tree_type
            )

    # Pestaña de Exportar Modelo
    elif st.session_state.active_tab == 6:
        st.header("Exportar Modelo")

        if not st.session_state.is_trained:
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '⚙️ Configuración'.")
        else:
            # Opciones para exportar el modelo
            display_model_export_options(
                st.session_state.tree_model,
                st.session_state.feature_names,
                st.session_state.class_names,
                st.session_state.tree_type,
                st.session_state.max_depth,
                st.session_state.min_samples_split,
                st.session_state.criterion
            )


if __name__ == "__main__":
    main()
