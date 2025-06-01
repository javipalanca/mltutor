"""
Este módulo contiene componentes de la interfaz de usuario para la aplicación MLTutor.
Incluye funciones para crear elementos de la interfaz, mostrar visualizaciones y gestionar la interacción del usuario.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64

from utils import get_image_download_link, generate_model_code, export_model_pickle, export_model_onnx
from model_evaluation import show_detailed_evaluation, show_prediction_path
from tree_visualization import (
    create_static_tree_visualization, get_tree_text,
    check_visualization_availability, render_visualization
)

# Funciones para la configuración de la página


def setup_page():
    """
    Configura la página principal de la aplicación Streamlit con estilos y título.
    """
    # Configuración de la página
    st.set_page_config(
        page_title="MLTutor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Estilos CSS personalizados
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #0D47A1;
        }
        .info-box {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #E8F5E9;
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 1rem;
        }
        .inference-box {
            background-color: #FFF8E1;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
            border: 1px solid #FFE082;
        }
        .code-box {
            background-color: #ECEFF1;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted #ccc;
            cursor: help;
        }
        .tooltip .tooltip-text {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e0e0e0;
            color: #666;
            font-size: 0.8rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F0F2F6;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #E3F2FD;
            border-bottom: 2px solid #1E88E5;
        }
    </style>
    """, unsafe_allow_html=True)

    # Título de la aplicación
    st.markdown("<h1 class='main-header'>MLTutor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Plataforma de aprendizaje de Machine Learning</p>",
                unsafe_allow_html=True)


def init_session_state():
    """
    Inicializa las variables de sesión necesarias para la aplicación.
    """
    # Estado global
    if 'tree_model' not in st.session_state:
        st.session_state.tree_model = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    if 'tree_type' not in st.session_state:
        st.session_state.tree_type = "Clasificación"
    if 'fig_width' not in st.session_state:
        st.session_state.fig_width = 14
    if 'fig_height' not in st.session_state:
        st.session_state.fig_height = 10
    if 'fig_size' not in st.session_state:
        st.session_state.fig_size = 3
    if 'show_values' not in st.session_state:
        st.session_state.show_values = True
    if 'show_animation' not in st.session_state:
        st.session_state.show_animation = False
    if 'show_boundary' not in st.session_state:
        st.session_state.show_boundary = False
    if 'show_math' not in st.session_state:
        st.session_state.show_math = True
    if 'interactive_tree' not in st.session_state:
        st.session_state.interactive_tree = True
    if 'animation_fps' not in st.session_state:
        st.session_state.animation_fps = 1.0
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    if 'dataset_option' not in st.session_state:
        st.session_state.dataset_option = "Iris (clasificación de flores)"
    if 'test_results' not in st.session_state:
        st.session_state.test_results = None
    if 'criterion' not in st.session_state:
        st.session_state.criterion = "gini"
    if 'max_depth' not in st.session_state:
        st.session_state.max_depth = 3
    if 'min_samples_split' not in st.session_state:
        st.session_state.min_samples_split = 2
    if 'test_size' not in st.session_state:
        st.session_state.test_size = 0.3
    if 'show_boundary' not in st.session_state:
        st.session_state.show_boundary = False
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
    if 'use_improved_viz' not in st.session_state:
        # Este valor se debe inicializar en el archivo principal con el valor real
        st.session_state.use_improved_viz = True
    if 'use_explanatory_viz' not in st.session_state:
        # Este valor se debe inicializar en el archivo principal con el valor real
        st.session_state.use_explanatory_viz = False


def show_welcome_page():
    """
    Muestra la página de bienvenida para los usuarios que visitan por primera vez.
    """
    st.markdown(f"""
    <div style="background-color: #E1F5FE; padding: 30px; border-radius: 10px; border-left: 5px solid #03A9F4; margin: 20px 0;">
        <h2 style="color: #0288D1; margin-top: 0;">Bienvenido a MLTutor</h2>
        <p style="font-size: 18px;">MLTutor es una plataforma educativa para aprender Machine Learning de forma interactiva.</p>
        <p style="font-size: 16px;">Para comenzar, selecciona un algoritmo en el menú lateral.</p>
        <h3>Algoritmos disponibles:</h3>
        <ul>
            <li><strong>Árboles de Decisión</strong> - Completamente implementado</li>
            <li>Regresión Logística (próximamente)</li>
            <li>K-Nearest Neighbors (próximamente)</li>
            <li>Redes Neuronales (próximamente)</li>
        </ul>
    </div>
    
    <div style="text-align: center; margin-top: 40px;">
        <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png" 
             alt="Comparación de algoritmos de ML" 
             style="max-width: 80%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <p style="margin-top: 10px; color: #666; font-style: italic;">Comparación visual de diferentes algoritmos de Machine Learning</p>
    </div>

    <div style="margin-top: 40px; padding: 20px; background-color: #F5F5F5; border-radius: 10px;">
        <h3>¿Qué es el Machine Learning?</h3>
        <p>El Machine Learning es una rama de la inteligencia artificial que permite a los sistemas aprender patrones a partir de datos y tomar decisiones sin ser explícitamente programados para ello.</p>
        <p>En esta plataforma, aprenderás de forma interactiva cómo funcionan diferentes algoritmos de ML, empezando por los árboles de decisión.</p>
    </div>
    """, unsafe_allow_html=True)


def show_sidebar_config():
    """
    Configura y muestra el sidebar con opciones para la configuración del modelo.
    Esta versión simplificada solo muestra la selección de algoritmo.

    Returns:
    --------
    dict
        Diccionario con los parámetros seleccionados por el usuario
    """
    with st.sidebar:
        st.header("MLTutor")

        # Selección del algoritmo
        st.subheader("Algoritmo")
        algorithm_type = st.selectbox(
            "Selecciona un algoritmo:",
            ("Árboles de Decisión", "Regresión Logística",
             "K-Nearest Neighbors", "Redes Neuronales"),
            index=0,  # Árboles de Decisión como opción por defecto
            help="Selecciona el algoritmo de machine learning que quieres explorar",
            on_change=lambda: st.session_state.update(
                {"algorithm_changed": True})
        )

        # Información acerca de la ubicación de configuración
        if algorithm_type == "Árboles de Decisión":
            st.info(
                "La configuración del modelo se encuentra en la pestaña '⚙️ Configuración'.")

            return {
                "algorithm_type": algorithm_type
            }
        else:
            # Mensaje de "Próximamente" para algoritmos no implementados
            st.markdown(f"""
            <div style="background-color: #FFF3E0; padding: 20px; border-radius: 10px; border-left: 5px solid #FF9800; margin-bottom: 20px;">
                <h3 style="color: #FF9800; margin-top: 0;">🚧 Próximamente: {algorithm_type} 🚧</h3>
                <p>Esta sección está en desarrollo y estará disponible en futuras actualizaciones de MLTutor.</p>
                <p>Mientras tanto, puedes explorar los Árboles de Decisión que ya están implementados.</p>
            </div>
            """, unsafe_allow_html=True)

            # Imagen ilustrativa para el algoritmo seleccionado
            if algorithm_type == "Regresión Logística":
                st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_logistic_001.png",
                         caption="Ilustración de Regresión Logística")
            elif algorithm_type == "K-Nearest Neighbors":
                st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png",
                         caption="Ilustración de K-Nearest Neighbors")
            elif algorithm_type == "Redes Neuronales":
                st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_mlp_001.png",
                         caption="Ilustración de Redes Neuronales")

            # Redirigir al usuario a Árboles de Decisión
            st.subheader("Mientras tanto...")
            st.markdown(
                "Puedes seleccionar 'Árboles de Decisión' en el menú desplegable para explorar la funcionalidad implementada.")

            # Valores por defecto para evitar errores
            return {
                "algorithm_type": algorithm_type
            }


def display_tree_visualization(tree_model, feature_names, class_names=None, tree_type="Clasificación"):
    """
    Muestra una visualización del árbol de decisión entrenado.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases (para clasificación)
    tree_type : str, default="Clasificación"
        Tipo de árbol ("Clasificación" o "Regresión")
    """
    st.subheader("Visualización del Árbol")

    # Configuración de la visualización
    fig_width = st.slider("Ancho de figura:", 8, 20,
                          st.session_state.fig_width)
    fig_height = st.slider("Alto de figura:", 6, 15,
                           st.session_state.fig_height)

    # Mostrar texto o visualización gráfica
    viz_type = st.radio("Tipo de visualización:",
                        ["Gráfico", "Texto"],
                        horizontal=True)

    if viz_type == "Gráfico":
        # Mostrar árbol como gráfico
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        plot_tree(tree_model,
                  feature_names=feature_names,
                  class_names=class_names if tree_type == "Clasificación" else None,
                  filled=True,
                  rounded=True,
                  ax=ax,
                  proportion=True,
                  impurity=True)
        st.pyplot(fig)

        # Enlace para descargar
        st.markdown(get_image_download_link(fig, "arbol_decision", "📥 Descargar visualización del árbol"),
                    unsafe_allow_html=True)
    else:
        # Mostrar árbol como texto
        from sklearn.tree import export_text
        tree_text = export_text(tree_model,
                                feature_names=feature_names,
                                show_weights=True)
        st.text(tree_text)

        # Enlace para descargar texto
        text_bytes = tree_text.encode()
        b64 = base64.b64encode(text_bytes).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="arbol_texto.txt">📥 Descargar texto del árbol</a>'
        st.markdown(href, unsafe_allow_html=True)


def display_feature_importance(tree_model, feature_names):
    """
    Muestra la importancia de las características del modelo.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características
    """
    st.subheader("Importancia de Características")

    # Obtener y ordenar las importancias
    importances = tree_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Crear DataFrame para mostrar
    importance_df = pd.DataFrame({
        'Característica': [feature_names[i] for i in indices],
        'Importancia': importances[indices]
    })

    # Mostrar gráfico y tabla
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(indices)), importances[indices])
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([feature_names[i]
                           for i in indices], rotation=45, ha='right')
        ax.set_title('Importancia de Características')
        ax.set_ylabel('Importancia')
        st.pyplot(fig)

    with col2:
        st.dataframe(importance_df)

        with st.expander("ℹ️ ¿Qué significa la importancia?"):
            st.markdown("""
            La **importancia de las características** mide cuánto contribuye cada característica a la predicción final.
            
            En los árboles de decisión, se calcula basándose en cuánto mejora cada característica la pureza (o reduce la impureza) cuando se usa para dividir los datos.
            
            Un valor más alto indica que la característica es más importante para el modelo.
            """)


def display_model_export_options(tree_model, feature_names, class_names, tree_type, max_depth, min_samples_split, criterion):
    """
    Muestra opciones para exportar el modelo entrenado.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases (para clasificación)
    tree_type : str
        Tipo de árbol ("Clasificación" o "Regresión")
    max_depth : int
        Profundidad máxima utilizada
    min_samples_split : int
        Número mínimo de muestras para dividir utilizado
    criterion : str
        Criterio utilizado para las divisiones
    """
    st.subheader("Exportar Modelo")

    # Generar código para el modelo
    code = generate_model_code(
        tree_model,
        tree_type,
        max_depth,
        min_samples_split,
        criterion,
        feature_names,
        class_names
    )

    # Opciones de exportación
    export_option = st.radio(
        "Formato de exportación:",
        ["Código Python", "Modelo (.pkl)", "ONNX (.onnx)"],
        horizontal=True
    )

    if export_option == "Código Python":
        st.code(code, language="python")
        st.download_button(
            label="📥 Descargar código Python",
            data=code,
            file_name="modelo_arbol_decision.py",
            mime="text/plain"
        )

    elif export_option == "Modelo (.pkl)":
        model_pickle = export_model_pickle(tree_model)
        st.download_button(
            label="📥 Descargar modelo (.pkl)",
            data=model_pickle,
            file_name="modelo_arbol_decision.pkl",
            mime="application/octet-stream"
        )

        with st.expander("ℹ️ ¿Cómo usar el modelo guardado?"):
            st.code("""
# Cargar el modelo guardado
import pickle

# Cargar el modelo desde el archivo
with open('modelo_arbol_decision.pkl', 'rb') as f:
    modelo_cargado = pickle.load(f)

# Usar el modelo para predicciones
nuevos_datos = [[5.1, 3.5, 1.4, 0.2]]  # Reemplaza con tus datos
prediccion = modelo_cargado.predict(nuevos_datos)
print(f"Predicción: {prediccion}")
            """, language="python")

    elif export_option == "ONNX (.onnx)":
        try:
            # Intenta importar skl2onnx
            import skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            # Exportar a ONNX
            model_onnx = export_model_onnx(tree_model, len(feature_names))

            if model_onnx:
                st.download_button(
                    label="📥 Descargar modelo ONNX",
                    data=model_onnx,
                    file_name="modelo_arbol_decision.onnx",
                    mime="application/octet-stream"
                )

                with st.expander("ℹ️ ¿Qué es ONNX y cómo usarlo?"):
                    st.markdown("""
                    **ONNX** (Open Neural Network Exchange) es un formato abierto para representar modelos de machine learning. Permite intercambiar modelos entre diferentes frameworks y plataformas.
                    
                    Para usar un modelo ONNX, necesitas una biblioteca compatible como `onnxruntime`:
                    """)

                    st.code("""
# Instalar onnxruntime (si no lo tienes)
# pip install onnxruntime

import onnxruntime as rt
import numpy as np

# Cargar modelo ONNX
session = rt.InferenceSession("modelo_arbol_decision.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Preparar datos
datos = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)  # Reemplaza con tus datos

# Realizar predicción
prediccion = session.run([output_name], {input_name: datos})[0]
print(f"Predicción: {prediccion}")
                    """, language="python")
            else:
                st.error(
                    "No se pudo generar el modelo ONNX. Verifica que las dependencias estén instaladas.")
        except ImportError:
            st.error(
                "La biblioteca skl2onnx no está instalada. Instálala con `pip install skl2onnx`.")
            st.code("pip install skl2onnx", language="bash")


def create_prediction_interface(tree_model, feature_names, class_names, tree_type):
    """
    Crea una interfaz para hacer predicciones con nuevos datos.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases (para clasificación)
    tree_type : str
        Tipo de árbol ("Clasificación" o "Regresión")
    """
    st.subheader("Predicciones con nuevos datos")

    # Crear sliders para cada característica
    st.markdown("### Ingresa valores para predecir")

    # Inicializar lista para almacenar valores de características
    new_data_values = []

    # Crear dos columnas para los sliders
    col1, col2 = st.columns(2)

    # Distribuir las características en las columnas
    half = len(feature_names) // 2 + len(feature_names) % 2

    # Primera columna
    with col1:
        for i, feature in enumerate(feature_names[:half]):
            value = st.slider(
                f"{feature}:",
                float(0),
                float(10),  # Valores arbitrarios, podrían ajustarse
                float(5),
                step=0.1,
                key=f"feature_{i}"
            )
            new_data_values.append(value)

    # Segunda columna
    with col2:
        for i, feature in enumerate(feature_names[half:], start=half):
            value = st.slider(
                f"{feature}:",
                float(0),
                float(10),  # Valores arbitrarios, podrían ajustarse
                float(5),
                step=0.1,
                key=f"feature_{i}"
            )
            new_data_values.append(value)

    # Botón para predecir
    predict_button = st.button("Realizar predicción", type="primary")

    # Realizar predicción cuando se presiona el botón
    if predict_button:
        # Convertir a array para la predicción
        new_data = np.array([new_data_values])

        # Hacer predicción
        prediction = tree_model.predict(new_data)[0]

        # Mostrar resultado según el tipo de árbol
        if tree_type == "Clasificación":
            prediction_label = class_names[prediction]

            st.markdown(f"""
            <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                <h3>Resultado de la predicción</h3>
                <p style="font-size: 24px; font-weight: bold; color: #2E7D32;">Clase predicha: {prediction_label}</p>
            </div>
            """, unsafe_allow_html=True)

            # Mostrar el camino de decisión
            st.markdown("### Camino de decisión")

            # Usar la función desde model_evaluation
            # Nota: No necesitamos pasar new_data (ya es un array 2D)
            # pero aseguramos el formato correcto
            show_prediction_path(tree_model, new_data,
                                 feature_names, class_names)

        else:  # Regresión
            st.markdown(f"""
            <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                <h3>Resultado de la predicción</h3>
                <p style="font-size: 24px; font-weight: bold; color: #2E7D32;">Valor predicho: {prediction:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

            # Mostrar el camino de decisión
            st.markdown("### Camino de decisión")

            # Usar la función desde model_evaluation
            show_prediction_path(tree_model, new_data, feature_names)
