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
from dataset_manager import load_data, preprocess_data, create_dataset_selector, load_dataset_from_file
from model_training import train_decision_tree, predict_sample, train_linear_model
from model_evaluation import evaluate_classification_model, evaluate_regression_model, show_detailed_evaluation
from decision_boundary import plot_decision_boundary
from sklearn.model_selection import train_test_split
from ui import (
    setup_page, init_session_state, show_welcome_page,
    display_feature_importance, display_model_export_options, create_prediction_interface
)
from tree_visualizer import (
    render_tree_visualization,
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
    elif st.session_state.navigation == "📁 Cargar CSV Personalizado":
        run_csv_loader_app()
    elif st.session_state.navigation in ["🔍 K-Nearest Neighbors (próximamente)",
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

                # Generar y mostrar el plot en tamaño reducido
                try:
                    ax = plot_decision_boundary(
                        st.session_state.tree_model,
                        X_plot,
                        st.session_state.y_train,
                        feature_names=feature_names_plot,
                        class_names=st.session_state.class_names
                    )

                    # Mostrar en columnas para reducir el tamaño al 75%
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        # Obtener la figura desde los ejes y mostrarla
                        plt.tight_layout()
                        fig = ax.get_figure()  # Use get_figure() method instead
                        st.pyplot(fig, clear_figure=True,
                                  use_container_width=True)

                    # Enlace para descargar
                    fig = ax.get_figure()  # Get figure for download link
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

            viz_col1, viz_col2 = st.columns(2)

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

            else:  # Visualización de texto
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

    # Crear botones para las pestañas
    for i, (tab_name, col) in enumerate(zip(tab_options, tab_cols)):
        button_key = f"tab_lr_{i}"
        button_style = "tab-button-active" if st.session_state.active_tab_lr == i else "tab-button"

        with col:
            st.markdown(f"<div class='{button_style}'>",
                        unsafe_allow_html=True)
            if st.button(tab_name, key=button_key, use_container_width=True):
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

            # Mostrar la figura
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

            # Mostrar la figura
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
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(pair_plot.fig, use_container_width=True)

                # Enlace para descargar
                st.markdown(
                    get_image_download_link(
                        pair_plot.fig, "matriz_dispersion_lr", "📥 Descargar matriz de dispersión"),
                    unsafe_allow_html=True
                )

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

            if model_type == "Linear":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
                with col2:
                    st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                with col3:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                with col2:
                    st.metric(
                        "Precision", f"{metrics.get('precision', 0):.4f}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        else:
            st.info("Entrena un modelo primero para ver las métricas de evaluación.")

    # Pestaña de Visualización
    elif st.session_state.active_tab_lr == 3:
        st.header("Visualizaciones")

        if st.session_state.get('model_trained_lr', False):
            model_type = st.session_state.get('model_type_lr', 'Linear')
            X_test = st.session_state.get('X_test_lr')
            y_test = st.session_state.get('y_test_lr')
            model = st.session_state.get('model_lr')

            if model_type == "Linear" and X_test is not None and y_test is not None and model is not None:
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, y_pred, alpha=0.7)
                ax.plot([y_test.min(), y_test.max()], [
                        y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Valores Reales')
                ax.set_ylabel('Predicciones')
                ax.set_title('Predicciones vs Valores Reales')
                st.pyplot(fig)
        else:
            st.info("Entrena un modelo primero para ver las visualizaciones.")

    # Pestaña de Coeficientes
    elif st.session_state.active_tab_lr == 4:
        st.header("Coeficientes del Modelo")

        if st.session_state.get('model_trained_lr', False):
            model = st.session_state.get('model_lr')
            feature_names = st.session_state.get('feature_names_lr', [])

            if model is not None and hasattr(model, 'coef_'):
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
                })
                coef_df = coef_df.sort_values(
                    'Coefficient', key=abs, ascending=False)

                st.dataframe(coef_df, use_container_width=True)

                # Gráfico de coeficientes
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(coef_df['Feature'], coef_df['Coefficient'])
                ax.set_xlabel('Coefficient Value')
                ax.set_title('Feature Coefficients')
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Entrena un modelo primero para ver los coeficientes.")

    # Pestaña de Predicciones
    elif st.session_state.active_tab_lr == 5:
        st.header("Hacer Predicciones")

        if st.session_state.get('model_trained_lr', False):
            model = st.session_state.get('model_lr')
            feature_names = st.session_state.get('feature_names_lr', [])

            st.markdown("Ingresa los valores para hacer una predicción:")

            input_values = []
            cols = st.columns(min(3, len(feature_names)))

            for i, feature in enumerate(feature_names):
                with cols[i % len(cols)]:
                    value = st.number_input(
                        f"{feature}:", key=f"pred_input_{i}")
                    input_values.append(value)

            if st.button("🔮 Predecir", key="predict_lr_button"):
                try:
                    if model is not None:
                        prediction = model.predict([input_values])[0]
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

            with col1:
                if st.button("📥 Descargar Modelo (Pickle)", key="download_pickle_lr"):
                    pickle_data = export_model_pickle(model)
                    st.download_button(
                        label="Descargar modelo.pkl",
                        data=pickle_data,
                        file_name="linear_regression_model.pkl",
                        mime="application/octet-stream"
                    )

            with col2:
                if st.button("📄 Generar Código", key="generate_code_lr"):
                    # For linear models, we need a different function than generate_model_code which is for trees
                    st.code(f"""
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression

# Cargar el modelo (si lo guardaste como pickle)
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Hacer predicciones con nuevos datos
# new_data = [[valor1, valor2, valor3, ...]]  # Reemplaza con tus valores
# prediction = model.predict(new_data)
# print(f"Predicción: {{prediction[0]}}")
""", language="python")
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


if __name__ == "__main__":
    main()
