from mltutor.desktop import ui as components
import plotly.express as px
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mltutor.desktop import ui as st

from mltutor.utils import (
    get_image_download_link, generate_model_code, export_model_pickle, export_model_onnx,
    create_info_box, format_number, show_code_with_download
)
from mltutor.viz.tree_visualizer import (
    render_tree_visualization,
    create_tree_visualization, get_tree_text
)
from mltutor.ui import (
    setup_page, init_session_state, show_welcome_page, show_footer,
    display_feature_importance, display_model_export_options, create_prediction_interface
)
from sklearn.model_selection import train_test_split

from mltutor.viz.decision_boundary import plot_decision_boundary
from mltutor.algorithms.model_evaluation import evaluate_classification_model, evaluate_regression_model, show_detailed_evaluation
from mltutor.algorithms.model_training import train_decision_tree, predict_sample, train_linear_model, train_knn_model
from mltutor.dataset.dataset_manager import load_data, preprocess_data, create_dataset_selector, load_dataset_from_file
from mltutor.apps.decission_tree_app import run_decision_trees_app
from mltutor.apps.linear_regression_app import run_linear_regression_app
from mltutor.apps.knn_app import run_knn_app
from mltutor.apps.neural_network_app import run_neural_networks_app


def main():
    """Función principal que ejecuta la aplicación MLTutor."""
    # Configuración de la página
    setup_page()

    st.markdown(
        """
        <style>
            .stDeployButton {display:none;}
            .stAppDeployButton {display:none;}
            .stMainMenu {display:none;}
        </style>
    """,
        unsafe_allow_html=True,
    )

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
        show_footer()
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

    show_footer()


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
