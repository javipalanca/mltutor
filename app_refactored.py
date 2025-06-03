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
from model_training import train_decision_tree, predict_sample
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
    elif st.session_state.navigation == "📁 Cargar CSV Personalizado":
        run_csv_loader_app()
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
        
        # Selector unificado
        dataset_option = st.selectbox(
            "Dataset de ejemplo:",
            ("🌸 Iris - Clasificación de flores",
             "🍷 Vino - Clasificación de vinos",
             "🔬 Cáncer - Diagnóstico binario",
             "🚢 Titanic - Supervivencia",
             "💰 Propinas - Predicción de propinas",
             "🏠 Viviendas California - Precios",
             "🐧 Pingüinos - Clasificación de especies"),
            index=("🌸 Iris - Clasificación de flores",
                   "🍷 Vino - Clasificación de vinos",
                   "🔬 Cáncer - Diagnóstico binario",
                   "🚢 Titanic - Supervivencia",
                   "💰 Propinas - Predicción de propinas",
                   "🏠 Viviendas California - Precios",
                   "🐧 Pingüinos - Clasificación de especies").index(st.session_state.selected_dataset),
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
            st.info(f"📊 **Dataset actual:** {st.session_state.selected_dataset}")
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
                ax = plot_decision_boundary(
                    st.session_state.tree_model,
                    X_plot,
                    st.session_state.y_train,
                    feature_names=feature_names_plot,
                    class_names=st.session_state.class_names
                )

                # Obtener la figura desde los ejes
                fig = ax.figure

                # Mostrar en columnas para reducir el tamaño al 75%
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(fig)                # Enlace para descargar
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
                st.session_state.get('X_train', None)  # Pasar datos de entrenamiento para rangos dinámicos
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


def run_csv_loader_app():
    """Aplicación para cargar y analizar datasets CSV personalizados."""
    st.header("📁 Cargar Dataset CSV Personalizado")
    st.markdown("""
    Esta herramienta te permite cargar y analizar tus propios datasets en formato CSV.
    Puedes explorar los datos, entrenar modelos de Machine Learning y comparar diferentes algoritmos.
    """)

    # Usar el selector de dataset mejorado
    dataset_result = create_dataset_selector()

    if dataset_result is None:
        st.info("👆 Por favor, carga un archivo CSV para continuar con el análisis.")
        return

    # Verificar si se cargó un CSV o se seleccionó un dataset predefinido
    if isinstance(dataset_result, tuple):
        # Se cargó un CSV personalizado
        file_path, target_col, task_type = dataset_result

        try:
            # Cargar el dataset personalizado
            X, y, feature_names, class_names, dataset_info, detected_task_type = load_dataset_from_file(
                file_path, target_col, task_type
            )

            st.success(f"✅ Dataset cargado exitosamente: {dataset_info}")

            # Mostrar tabs para análisis
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Exploración de Datos",
                "🏋️ Entrenamiento de Modelo",
                "📈 Evaluación",
                "🔍 Predicciones"
            ])

            with tab1:
                st.subheader("📊 Análisis Exploratorio de Datos")

                # Información general del dataset
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Muestras", len(X))
                with col2:
                    st.metric("Características", len(feature_names))
                with col3:
                    st.metric("Tipo de Tarea", detected_task_type)
                with col4:
                    if detected_task_type == "Clasificación":
                        st.metric("Clases", len(class_names)
                                  if class_names else "N/A")
                    else:
                        st.metric("Variable Objetivo", target_col)

                # Vista previa de los datos
                st.subheader("Vista Previa de los Datos")
                df_combined = pd.DataFrame(X, columns=feature_names)
                df_combined[target_col] = y
                st.dataframe(df_combined.head(10), use_container_width=True)

                # Estadísticas descriptivas
                st.subheader("Estadísticas Descriptivas")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Características numéricas:**")
                    numeric_features = X.select_dtypes(
                        include=[np.number]).columns
                    if len(numeric_features) > 0:
                        st.dataframe(X[numeric_features].describe(),
                                     use_container_width=True)
                    else:
                        st.info("No hay características numéricas")

                with col2:
                    st.write("**Variable objetivo:**")
                    if detected_task_type == "Clasificación":
                        value_counts = pd.Series(y).value_counts()
                        st.dataframe(value_counts.to_frame(
                            "Frecuencia"), use_container_width=True)
                    else:
                        target_stats = pd.Series(y).describe()
                        st.dataframe(target_stats.to_frame(
                            "Estadística"), use_container_width=True)

                # Visualizaciones
                if len(numeric_features) > 0:
                    st.subheader("Visualizaciones")

                    # Histogramas de características numéricas
                    if len(numeric_features) <= 10:  # Evitar sobrecarga con muchas características
                        fig, axes = plt.subplots(
                            nrows=(len(numeric_features) + 2) // 3,
                            ncols=3,
                            figsize=(
                                12, 4 * ((len(numeric_features) + 2) // 3))
                        )
                        if len(numeric_features) == 1:
                            axes = [axes]
                        elif (len(numeric_features) + 2) // 3 == 1:
                            axes = [axes]
                        else:
                            axes = axes.flatten()

                        for i, col in enumerate(numeric_features):
                            axes[i].hist(X[col], bins=20,
                                         alpha=0.7, edgecolor='black')
                            axes[i].set_title(f'Distribución de {col}')
                            axes[i].set_xlabel(col)
                            axes[i].set_ylabel('Frecuencia')

                        # Ocultar subplots vacíos
                        for i in range(len(numeric_features), len(axes)):
                            axes[i].set_visible(False)

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

            with tab2:
                st.subheader("🏋️ Entrenamiento de Modelo")

                # División de datos
                test_size = st.slider(
                    "Porcentaje para prueba (%)", 10, 50, 30) / 100

                # Preprocesamiento básico para datos categóricos
                X_processed = X.copy()

                # Codificar variables categóricas si existen
                categorical_features = X.select_dtypes(
                    include=['object']).columns
                if len(categorical_features) > 0:
                    st.info(
                        f"Se encontraron {len(categorical_features)} características categóricas que serán codificadas automáticamente.")

                    from sklearn.preprocessing import LabelEncoder
                    for col in categorical_features:
                        le = LabelEncoder()
                        X_processed[col] = le.fit_transform(X[col].astype(str))

                # Dividir los datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=test_size, random_state=42
                )

                st.success(
                    f"Datos divididos: {len(X_train)} muestras para entrenamiento, {len(X_test)} para prueba")

                # Selección de algoritmo
                st.subheader("Selección de Algoritmo")

                if detected_task_type == "Clasificación":
                    algorithm = st.selectbox(
                        "Algoritmo:",
                        ["Árbol de Decisión", "Random Forest", "Regresión Logística"]
                    )
                else:
                    algorithm = st.selectbox(
                        "Algoritmo:",
                        ["Árbol de Decisión", "Random Forest", "Regresión Lineal"]
                    )

                # Entrenar modelo
                if st.button("🚀 Entrenar Modelo", type="primary"):
                    with st.spinner("Entrenando modelo..."):
                        if algorithm == "Árbol de Decisión":
                            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

                            if detected_task_type == "Clasificación":
                                model = DecisionTreeClassifier(
                                    random_state=42, max_depth=5)
                            else:
                                model = DecisionTreeRegressor(
                                    random_state=42, max_depth=5)

                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                        elif algorithm == "Random Forest":
                            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

                            if detected_task_type == "Clasificación":
                                model = RandomForestClassifier(
                                    random_state=42, n_estimators=100)
                            else:
                                model = RandomForestRegressor(
                                    random_state=42, n_estimators=100)

                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                        else:  # Regresión Logística o Lineal
                            if detected_task_type == "Clasificación":
                                from sklearn.linear_model import LogisticRegression
                                model = LogisticRegression(
                                    random_state=42, max_iter=1000)
                            else:
                                from sklearn.linear_model import LinearRegression
                                model = LinearRegression()

                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                        # Guardar en session state
                        st.session_state.csv_model = model
                        st.session_state.csv_X_test = X_test
                        st.session_state.csv_y_test = y_test
                        st.session_state.csv_y_pred = y_pred
                        st.session_state.csv_feature_names = feature_names
                        st.session_state.csv_task_type = detected_task_type
                        st.session_state.csv_algorithm = algorithm

                        st.success(
                            f"✅ Modelo {algorithm} entrenado exitosamente!")

            with tab3:
                st.subheader("📈 Evaluación del Modelo")

                if 'csv_model' not in st.session_state:
                    st.warning(
                        "Primero entrena un modelo en la pestaña 'Entrenamiento'.")
                else:
                    model = st.session_state.csv_model
                    X_test = st.session_state.csv_X_test
                    y_test = st.session_state.csv_y_test
                    y_pred = st.session_state.csv_y_pred
                    task_type = st.session_state.csv_task_type
                    algorithm = st.session_state.csv_algorithm

                    st.info(f"Evaluando modelo: {algorithm} para {task_type}")

                    if task_type == "Clasificación":
                        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

                        # Métricas principales
                        accuracy = accuracy_score(y_test, y_pred)
                        st.metric("Precisión", f"{accuracy:.3f}")

                        # Matriz de confusión
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cm = confusion_matrix(y_test, y_pred)
                        sns.heatmap(cm, annot=True, fmt='d',
                                    cmap='Blues', ax=ax)
                        ax.set_title('Matriz de Confusión')
                        ax.set_xlabel('Predicción')
                        ax.set_ylabel('Valor Real')
                        st.pyplot(fig)
                        plt.close()

                        # Reporte de clasificación
                        st.subheader("Reporte Detallado")
                        report = classification_report(
                            y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)

                    else:  # Regresión
                        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

                        # Métricas principales
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R²", f"{r2:.3f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.3f}")
                        with col3:
                            st.metric("MAE", f"{mae:.3f}")
                        with col4:
                            st.metric("MSE", f"{mse:.3f}")

                        # Gráfico de predicciones vs valores reales
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(y_test, y_pred, alpha=0.6)
                        ax.plot([y_test.min(), y_test.max()], [
                                y_test.min(), y_test.max()], 'r--', lw=2)
                        ax.set_xlabel('Valores Reales')
                        ax.set_ylabel('Predicciones')
                        ax.set_title('Predicciones vs Valores Reales')
                        st.pyplot(fig)
                        plt.close()

            with tab4:
                st.subheader("🔍 Hacer Predicciones")

                if 'csv_model' not in st.session_state:
                    st.warning(
                        "Primero entrena un modelo en la pestaña 'Entrenamiento'.")
                else:
                    st.info("Introduce los valores para hacer una predicción:")

                    # Crear formulario para predicción
                    prediction_values = {}

                    # Obtener solo características numéricas para simplificar
                    numeric_features = X.select_dtypes(
                        include=[np.number]).columns

                    # Limitar a 10 características
                    for feature in numeric_features[:10]:
                        min_val = float(X[feature].min())
                        max_val = float(X[feature].max())
                        mean_val = float(X[feature].mean())

                        prediction_values[feature] = st.number_input(
                            f"{feature}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=(max_val - min_val) / 100
                        )

                    if st.button("🎯 Realizar Predicción"):
                        # Preparar datos para predicción
                        pred_data = []
                        for feature in feature_names:
                            if feature in prediction_values:
                                pred_data.append(prediction_values[feature])
                            else:
                                # Para características categóricas, usar valor más común
                                if feature in X.columns:
                                    pred_data.append(X[feature].mode()[0] if len(
                                        X[feature].mode()) > 0 else 0)
                                else:
                                    pred_data.append(0)

                        # Hacer predicción
                        model = st.session_state.csv_model
                        pred_array = np.array([pred_data])
                        prediction = model.predict(pred_array)[0]

                        # Mostrar resultado
                        task_type = st.session_state.csv_task_type
                        if task_type == "Clasificación":
                            if class_names and prediction < len(class_names):
                                result = class_names[int(prediction)]
                            else:
                                result = f"Clase {int(prediction)}"
                            st.success(f"🎯 Predicción: **{result}**")
                        else:
                            st.success(f"🎯 Predicción: **{prediction:.3f}**")

                        # Mostrar confianza si es posible
                        if hasattr(model, 'predict_proba') and task_type == "Clasificación":
                            probabilities = model.predict_proba(pred_array)[0]
                            st.subheader("Confianza por clase:")
                            for i, prob in enumerate(probabilities):
                                class_name = class_names[i] if class_names and i < len(
                                    class_names) else f"Clase {i}"
                                st.write(
                                    f"• {class_name}: {prob:.3f} ({prob*100:.1f}%)")

        except Exception as e:
            st.error(f"❌ Error al procesar el dataset: {str(e)}")

    else:
        # Se seleccionó un dataset predefinido
        st.info(f"Dataset seleccionado: {dataset_result}")
        st.markdown("### 🎯 Análisis con Dataset Predefinido")
        st.markdown("""
        Para analizar datasets predefinidos como Iris, Titanic, etc., 
        ve a la sección **🌲 Árboles de Decisión** donde encontrarás todas las herramientas de análisis.
        """)

        if st.button("🚀 Ir a Árboles de Decisión"):
            st.session_state.navigation = "🌲 Árboles de Decisión"
            st.rerun()


if __name__ == "__main__":
    main()
