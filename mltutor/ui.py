"""
Este módulo contiene componentes de la interfaz de usuario para la aplicación MLTutor.
Incluye funciones para crear elementos de la interfaz, mostrar visualizaciones y gestionar la interacción del usuario.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import os
import base64
from sklearn.tree import plot_tree

from utils import get_image_download_link, generate_model_code, export_model_pickle, export_model_onnx, show_code_with_download
from algorithms.model_evaluation import show_detailed_evaluation, show_prediction_path
from viz.tree_visualization import (
    create_static_tree_visualization, get_tree_text,
    render_visualization
)
from dataset.dataset_metadata import get_dataset_metadata

# Funciones para la configuración de la página


def setup_page():
    """
    Configura la página principal de la aplicación Streamlit con estilos y título.
    """
    # Configuración de la página
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
    st.set_page_config(
        page_title="🎓 MLTutor",
        page_icon=icon_path if os.path.exists(icon_path) else "🎓",
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
            margin-top: 5rem;
            margin-bottom: 0;
            text-align: center;
            padding: 0.4rem 1rem;
            font-size: 0.65rem;
            color: #999;
            background-color: #fafafa;
            border-top: 1px solid #f0f0f0;
            line-height: 1.2;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
            position: relative;
            bottom: 0;
        }
        .footer p {
            margin: 0;
            display: inline;
        }
        .footer p:not(:last-child)::after {
            content: " • ";
            color: #ccc;
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
    st.markdown("<h1 class='main-header'>🎓 MLTutor</h1>",
                unsafe_allow_html=True)
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
    if 'navigation' not in st.session_state:
        st.session_state.navigation = "🏠 Inicio"
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
    st.header("🌟 Bienvenido a 🎓 MLTutor")
    st.markdown("### Una plataforma interactiva para aprender Machine Learning")

    # Introducción a ML
    st.markdown("""
    <div style="background-color: #E1F5FE; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #0288D1;">¿Qué es el Machine Learning?</h3>
        <p>El Machine Learning es una rama de la inteligencia artificial que permite a los sistemas aprender patrones a partir 
        de datos y tomar decisiones sin ser explícitamente programados para ello.</p>
        <p>Con 🎓 MLTutor podrás:</p>
        <ul>
            <li>Aprender conceptos fundamentales de Machine Learning</li>
            <li>Experimentar con algoritmos en tiempo real</li>
            <li>Visualizar cómo funcionan los modelos internamente</li>
            <li>Entrenar modelos con datos reales y realizar predicciones</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Sección: Algoritmos disponibles
    st.markdown("## 📚 Algoritmos disponibles")
    # Crear una disposición de tarjetas para los algoritmos
    st.markdown("Selecciona un algoritmo para comenzar a aprender:")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="background-color: #E8F5E9; padding: 15px; border-radius: 10px; height: 200px; border-left: 5px solid #4CAF50;">
            <h3 style="color: #2E7D32;">🌲 Árboles de Decisión</h3>
            <p>Algoritmos que toman decisiones basadas en condiciones en forma de árbol.</p>
            <p><strong>Estado:</strong> <span style="color: #2E7D32;">Disponible</span></p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("✨ Explorar Árboles de Decisión", use_container_width=True):
            st.session_state.navigation = "🌲 Árboles de Decisión"
            st.rerun()

    with col2:
        st.markdown("""
        <div style="background-color: #FFF3E0; padding: 15px; border-radius: 10px; height: 200px; border-left: 5px solid #FF9800;">
            <h3 style="color: #E65100;">📊 Regresión</h3>
            <p>Modelos lineales para regresión y clasificación logística.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("✨ Explorar Regresión", use_container_width=True):
            st.session_state.navigation = "📊 Regresión"
            st.rerun()

    with col3:
        st.markdown("""
        <div style="background-color: #E0F7FA; padding: 15px; border-radius: 10px; height: 200px; border-left: 5px solid #00BCD4;">
            <h3 style="color: #006064;">🔍 K-Nearest Neighbors</h3>
            <p>Algoritmo basado en la similitud entre ejemplos.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("✨ Explorar KNN", use_container_width=True):
            st.session_state.navigation = "🔍 K-Nearest Neighbors"
            st.rerun()

    with col4:
        st.markdown("""
        <div style="background-color: #F3E5F5; padding: 15px; border-radius: 10px; height: 200px; border-left: 5px solid #9C27B0;">
            <h3 style="color: #4A148C;">🧠 Redes Neuronales</h3>
            <p>Modelos inspirados en el cerebro humano.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("✨ Explorar Redes Neuronales", use_container_width=True):
            st.session_state.navigation = "🧠 Redes Neuronales"
            st.rerun()

    # Sección: Cómo usar MLTutor
    st.markdown("## 📝 Cómo usar MLTutor")

    st.markdown("""
    <div style="background-color: #F5F5F5; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h3>Guía de uso</h3>
        <ol>
            <li><strong>Selecciona un algoritmo</strong> en el menú lateral o desde las tarjetas de arriba.</li>
            <li><strong>Explora los datos</strong> disponibles o carga tus propios conjuntos de datos.</li>
            <li><strong>Configura los parámetros</strong> del modelo según tus necesidades.</li>
            <li><strong>Entrena el modelo</strong> y observa cómo funciona internamente.</li>
            <li><strong>Analiza los resultados</strong> y realiza predicciones con nuevos datos.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Imagen ilustrativa en la parte inferior
    # st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png",
    #         caption="Comparación visual de diferentes algoritmos de Machine Learning",
    #         use_container_width=True)


def show_footer():
    """
    Muestra el pie de página con información de contacto y enlaces útiles.
    """
    st.markdown("""
    <div class="footer">
        <p>Desarrollado por Javi Palanca</p>
        <p>Departament de Sistemes Informàtics i Computació</p>
        <p>Universitat Politècnica de València</p>
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
            ("Árboles de Decisión", "Regresión",
             "K-Nearest Neighbors", "Redes Neuronales"),
            index=0,  # Árboles de Decisión como opción por defecto
            help="Selecciona el algoritmo de machine learning que quieres explorar",
            on_change=lambda: st.session_state.update(
                {"algorithm_changed": True})
        )

        # Información acerca de la ubicación de configuración
        if algorithm_type == "Árboles de Decisión":
            st.info(
                "La configuración del modelo se encuentra en la pestaña '🏋️ Entrenamiento'.")

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
            if algorithm_type == "Regresión":
                st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_logistic_001.png",
                         caption="Ilustración de Regresión")
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

        # Mostrar con tamaño reducido pero expandible
        col_inner1, col_inner2, col_inner3 = st.columns([1, 3, 1])
        with col_inner2:
            st.pyplot(fig, use_container_width=True)

        # Enlace para descargar la imagen
        st.markdown(get_image_download_link(fig, "importancia_caracteristicas", "📥 Descargar gráfico"),
                    unsafe_allow_html=True)

        # Mostrar el código que genera esta visualización
        code = """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Obtener la importancia de las características
importances = tree_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Crear DataFrame para visualizar
importance_df = pd.DataFrame({
    'Característica': [feature_names[i] for i in indices],
    'Importancia': importances[indices]
})

# Crear la visualización
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(indices)), importances[indices])
ax.set_xticks(range(len(indices)))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
ax.set_title('Importancia de Características')
ax.set_ylabel('Importancia')

# Para mostrar en Streamlit
st.pyplot(fig)
# Para usar en Jupyter/Python normal
# plt.tight_layout()
# plt.show()
"""
        show_code_with_download(
            code, "Código para generar este gráfico", "importancia_caracteristicas.py")

    with col2:
        st.dataframe(importance_df)

        # Mostrar código para generar y exportar el DataFrame
        code_df = """
import pandas as pd
import numpy as np

# Obtener la importancia de las características
importances = tree_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Crear DataFrame con las importancias ordenadas
importance_df = pd.DataFrame({
    'Característica': [feature_names[i] for i in indices],
    'Importancia': importances[indices]
})

# Para mostrar en Streamlit
st.dataframe(importance_df)

# Para exportar a CSV
# importance_df.to_csv('importancia_caracteristicas.csv', index=False)
"""
        show_code_with_download(
            code_df, "Código para generar esta tabla", "importancia_tabla.py")

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

        # Mostrar código para generar esta exportación
        code_gen = f"""
import numpy as np
from sklearn.tree import {'DecisionTreeClassifier' if tree_type == 'Clasificación' else 'DecisionTreeRegressor'}

# Definir los nombres de las características
feature_names = {feature_names}

# Definir los nombres de las clases (solo para clasificación)
{'class_names = ' + str(class_names) if tree_type == 'Clasificación' and class_names else '# No hay nombres de clases para regresión'}

# Crear el modelo
model = {'DecisionTreeClassifier' if tree_type == 'Clasificación' else 'DecisionTreeRegressor'}(
    max_depth={max_depth},
    min_samples_split={min_samples_split},
    criterion="{criterion}",
    random_state=42
)

# El modelo ya está entrenado, para recrearlo necesitarás entrenarlo con tus datos:
# model.fit(X_train, y_train)

# Ejemplo de uso para predicción:
# Nuevos datos: [característica1, característica2, ...]
nuevo_ejemplo = [[{', '.join(['0.0' for _ in feature_names])}]]
prediccion = model.predict(nuevo_ejemplo)

# Para mostrar la predicción
print(f"Predicción: {{prediccion}}")
"""

        # Add probability code based on tree type
        if tree_type == 'Clasificación':
            code_gen += """
# Para obtener probabilidades (solo para clasificación):
# probabilidades = model.predict_proba(nuevo_ejemplo)
"""

        show_code_with_download(
            code_gen, "Código para generar la exportación de código", "generar_exportacion_codigo.py")

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

        # Mostrar código para generar esta exportación
        code_pickle = """
import pickle

# Exportar el modelo a formato pickle
def export_model_pickle(model):
    # Serializar el modelo a bytes
    modelo_serializado = pickle.dumps(model)
    
    # Para guardar a un archivo
    # with open('modelo_arbol_decision.pkl', 'wb') as f:
    #    pickle.dump(model, f)
    
    return modelo_serializado

# Para usar con Streamlit:
# model_pickle = export_model_pickle(tree_model)
# st.download_button(
#     label="Descargar modelo (.pkl)",
#     data=model_pickle,
#     file_name="modelo_arbol_decision.pkl",
#     mime="application/octet-stream"
# )
"""
        show_code_with_download(
            code_pickle, "Código para generar la exportación pickle", "exportar_modelo_pickle.py")

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

                # Mostrar código para generar esta exportación
                code_onnx = f"""
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np

def export_modelo_onnx(model, num_features):
    \"\"\"
    Convierte un modelo scikit-learn a formato ONNX.
    
    Parameters:
    -----------
    model : objeto modelo scikit-learn
        El modelo a convertir
    num_features : int
        Número de características que acepta el modelo
        
    Returns:
    --------
    bytes
        Datos serializados del modelo en formato ONNX
    \"\"\"
    # Configuración inicial para la conversión
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    
    # Convertir el modelo
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Serializar a bytes
    return onnx_model.SerializeToString()

# Para guardar el modelo a un archivo:
# with open('modelo_arbol_decision.onnx', 'wb') as f:
#     f.write(export_modelo_onnx(tree_model, {len(feature_names)}))

# Para usar con Streamlit:
# model_onnx = export_modelo_onnx(tree_model, {len(feature_names)})
# st.download_button(
#     label="Descargar modelo ONNX",
#     data=model_onnx,
#     file_name="modelo_arbol_decision.onnx",
#     mime="application/octet-stream"
# )
"""
                show_code_with_download(
                    code_onnx, "Código para generar la exportación ONNX", "exportar_modelo_onnx.py")
            else:
                st.error(
                    "No se pudo generar el modelo ONNX. Verifica que las dependencias estén instaladas.")
        except ImportError as e:
            st.error(
                f"La biblioteca skl2onnx no está instalada. Instálala con `pip install skl2onnx`. {e}")
            st.code("pip install skl2onnx", language="bash")


def create_button_panel(buttons) -> str:
    # Crear columnas dinámicamente según el número de opciones
    num_options = len(buttons)
    viz_cols = st.columns(num_options)

    if "viz_type" not in st.session_state or st.session_state.viz_type not in (v[1] for v in buttons):
        st.session_state.viz_type = buttons[0][1]

    # Crear botones dinámicamente
    for i, (label, viz_type, key) in enumerate(buttons):
        with viz_cols[i]:
            if st.button(label,
                         key=key,
                         type="primary" if st.session_state.viz_type == viz_type else "secondary",
                         use_container_width=True):
                st.session_state.viz_type = viz_type
                st.rerun()
    return st.session_state.viz_type


def create_prediction_interface(model, feature_names, class_names, task_type, X_train=None, dataset_name=None):
    """
    Crea una interfaz para hacer predicciones con nuevos datos.

    Parameters:
    -----------
    model : scikit-learn model
        Modelo entrenado (DecisionTree, KNN, etc.)
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases (para clasificación)
    task_type : str
        Tipo de tarea ("Clasificación" o "Regresión")
    X_train : pd.DataFrame or np.array, optional
        Datos de entrenamiento para determinar rangos dinámicos de características
    dataset_name : str, optional
        Nombre del dataset para obtener metadata adicional
    """
    # st.subheader("Predicciones con nuevos datos")

    # Obtener metadata del dataset si está disponible
    metadata = get_dataset_metadata(dataset_name) if dataset_name else {}
    feature_descriptions = metadata.get('feature_descriptions', {})
    value_mappings = metadata.get('value_mappings', {})
    original_to_display = metadata.get('original_to_display', {})
    categorical_features = metadata.get('categorical_features', [])

    # Crear sliders para cada característica
    st.markdown("### Ingresa valores para predecir")

    # Inicializar lista para almacenar valores de características
    new_data_values = []

    # Analizar características si tenemos datos de entrenamiento
    feature_info = {}
    if X_train is not None:
        # Convertir a DataFrame si es necesario
        if isinstance(X_train, np.ndarray):
            df_train = pd.DataFrame(X_train, columns=range(len(feature_names)))
            # Mapear columnas originales si están disponibles
            if hasattr(X_train, 'columns'):
                column_names = X_train.columns
            else:
                column_names = range(len(feature_names))
        else:
            df_train = X_train
            column_names = df_train.columns

        n_cols = df_train.shape[1]
        for i, feature_display_name in enumerate(feature_names):
            # Si el índice excede las columnas disponibles (mismatch), crear info sintética
            if i >= n_cols:
                feature_info[feature_display_name] = {
                    'type': 'continuous',
                    'values': [0.0, 1.0],
                    'min': 0.0,
                    'max': 1.0,
                    'mean': 0.5,
                    'original_column': feature_display_name
                }
                continue

            # Obtener el nombre original de la columna
            original_col_name = column_names[i]

            # Acceso seguro a la columna por posición
            feature_col = df_train.iloc[:, i]

            # Determinar tipo de característica
            unique_values = feature_col.nunique()
            unique_vals = sorted(feature_col.unique())

            # Verificar si es categórica según metadata
            is_categorical_by_metadata = original_col_name in categorical_features

            if unique_values <= 2:
                # Característica binaria
                feature_type = 'binary'
            elif unique_values <= 10 and (all(isinstance(x, (int, np.integer)) for x in unique_vals) or is_categorical_by_metadata):
                # Característica categórica (pocos valores enteros o marcada en metadata)
                feature_type = 'categorical'
            else:
                # Característica numérica continua
                feature_type = 'continuous'

            # Preparar valores para mostrar
            if feature_type in ['binary', 'categorical'] and original_col_name in value_mappings:
                # Usar mapeo de valores si está disponible
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

                feature_info[feature_display_name] = {
                    'type': feature_type,
                    'values': unique_vals,
                    'display_values': display_values,
                    'value_to_original': value_to_original,
                    'original_column': original_col_name
                }
            else:
                feature_info[feature_display_name] = {
                    'type': feature_type,
                    'values': unique_vals,
                    'min': float(feature_col.min()) if feature_type == 'continuous' else min(unique_vals),
                    'max': float(feature_col.max()) if feature_type == 'continuous' else max(unique_vals),
                    'mean': float(feature_col.mean()) if feature_type == 'continuous' else None,
                    'original_column': original_col_name
                }
    else:
        # Valores por defecto si no hay datos de entrenamiento
        for feature in feature_names:
            feature_info[feature] = {
                'type': 'continuous',
                'min': 0.0,
                'max': 10.0,
                'mean': 5.0,
                'original_column': feature
            }

    # Crear dos columnas para los controles
    col1, col2 = st.columns(2)

    # Distribuir las características en las columnas
    half = len(feature_names) // 2 + len(feature_names) % 2

    # Crear controles para todas las características en orden
    feature_values = {}

    # Función auxiliar para crear etiqueta con descripción
    def create_feature_label(feature_name, info):
        original_col = info.get('original_column', feature_name)
        description = feature_descriptions.get(original_col, '')
        if description:
            return f"**{feature_name}**\n\n*{description}*"
        return feature_name

    # Primera columna
    with col1:
        for i, feature in enumerate(feature_names[:half]):
            info = feature_info[feature]
            label = create_feature_label(feature, info)

            if info['type'] == 'binary':
                # Checkbox para características binarias con valores descriptivos
                if 'display_values' in info and 'value_to_original' in info:
                    # Usar valores descriptivos
                    selected_display = st.selectbox(
                        label,
                        options=info['display_values'],
                        index=0,
                        key=f"feature_{i}"
                    )
                    # Convertir valor descriptivo de vuelta al original
                    feature_values[i] = info['value_to_original'][selected_display]
                elif len(info['values']) == 2 and 0 in info['values'] and 1 in info['values']:
                    value = st.checkbox(label, key=f"feature_{i}")
                    feature_values[i] = 1 if value else 0
                else:
                    # Selectbox para binaria no 0/1
                    value = st.selectbox(
                        label,
                        options=info['values'],
                        index=0,
                        key=f"feature_{i}"
                    )
                    feature_values[i] = value

            elif info['type'] == 'categorical':
                # Selectbox para características categóricas con valores descriptivos
                if 'display_values' in info and 'value_to_original' in info:
                    # Usar valores descriptivos
                    selected_display = st.selectbox(
                        label,
                        options=info['display_values'],
                        index=len(
                            info['display_values'])//2 if len(info['display_values']) > 1 else 0,
                        key=f"feature_{i}"
                    )
                    # Convertir valor descriptivo de vuelta al original
                    feature_values[i] = info['value_to_original'][selected_display]
                else:
                    # Usar valores originales
                    value = st.selectbox(
                        label,
                        options=info['values'],
                        index=len(info['values']
                                  )//2 if len(info['values']) > 1 else 0,
                        key=f"feature_{i}"
                    )
                    feature_values[i] = value

            else:  # continuous
                # Slider para características continuas
                step = (info['max'] - info['min']) / \
                    100 if info['max'] != info['min'] else 0.1
                default_val = info.get('mean', (info['min'] + info['max']) / 2)

                value = st.slider(
                    label,
                    min_value=info['min'],
                    max_value=info['max'],
                    value=default_val,
                    step=step,
                    key=f"feature_{i}"
                )
                feature_values[i] = value

    # Segunda columna
    with col2:
        for i, feature in enumerate(feature_names[half:], start=half):
            info = feature_info[feature]
            label = create_feature_label(feature, info)

            if info['type'] == 'binary':
                # Checkbox para características binarias con valores descriptivos
                if 'display_values' in info and 'value_to_original' in info:
                    # Usar valores descriptivos
                    selected_display = st.selectbox(
                        label,
                        options=info['display_values'],
                        index=0,
                        key=f"feature_{i}"
                    )
                    # Convertir valor descriptivo de vuelta al original
                    feature_values[i] = info['value_to_original'][selected_display]
                elif len(info['values']) == 2 and 0 in info['values'] and 1 in info['values']:
                    value = st.checkbox(label, key=f"feature_{i}")
                    feature_values[i] = 1 if value else 0
                else:
                    # Selectbox para binaria no 0/1
                    value = st.selectbox(
                        label,
                        options=info['values'],
                        index=0,
                        key=f"feature_{i}"
                    )
                    feature_values[i] = value

            elif info['type'] == 'categorical':
                # Selectbox para características categóricas con valores descriptivos
                if 'display_values' in info and 'value_to_original' in info:
                    # Usar valores descriptivos
                    selected_display = st.selectbox(
                        label,
                        options=info['display_values'],
                        index=len(
                            info['display_values'])//2 if len(info['display_values']) > 1 else 0,
                        key=f"feature_{i}"
                    )
                    # Convertir valor descriptivo de vuelta al original
                    feature_values[i] = info['value_to_original'][selected_display]
                else:
                    # Usar valores originales
                    value = st.selectbox(
                        label,
                        options=info['values'],
                        index=len(info['values']
                                  )//2 if len(info['values']) > 1 else 0,
                        key=f"feature_{i}"
                    )
                    feature_values[i] = value

            else:  # continuous
                # Slider para características continuas
                step = (info['max'] - info['min']) / \
                    100 if info['max'] != info['min'] else 0.1
                default_val = info.get('mean', (info['min'] + info['max']) / 2)

                value = st.slider(
                    label,
                    min_value=info['min'],
                    max_value=info['max'],
                    value=default_val,
                    step=step,
                    key=f"feature_{i}"
                )
                feature_values[i] = value

    # Convertir el diccionario a lista en orden correcto
    new_data_values = [feature_values[i] for i in range(len(feature_names))]

    # Botón para predecir
    predict_button = st.button("Realizar predicción", type="primary")

    # Realizar predicción cuando se presiona el botón
    if predict_button:
        # Convertir a array para la predicción (orden original)
        new_data = np.array([new_data_values], dtype=np.float32)

        # Detectar si es un modelo TensorFlow/Keras
        is_keras = False
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow.keras import Model as KerasModel
            is_keras = isinstance(model, KerasModel)
        except Exception:
            is_keras = False

        # Aplicar scaler si está disponible en sesión (para NN) y no es ya árbol/KNN
        scaler = st.session_state.get(
            'nn_scaler') if 'nn_scaler' in st.session_state else None
        if is_keras and scaler is not None:
            try:
                new_data_scaled = scaler.transform(new_data)
            except Exception:
                new_data_scaled = new_data
        else:
            new_data_scaled = new_data

        # Predicción según tipo de modelo
        try:
            if is_keras:
                raw_pred = model.predict(new_data_scaled, verbose=0)
            else:
                raw_pred = model.predict(new_data)
        except Exception as pred_err:
            st.error(f"Error durante la predicción: {pred_err}")
            return

        # --- CLASIFICACIÓN ---
        if task_type == "Clasificación":
            # Preparar nombres de clases (árbol/KNN vs NN)
            if class_names is None and 'nn_class_names' in st.session_state:
                class_names = st.session_state.nn_class_names

            # Keras binary (salida (1,) o (1,1))
            if is_keras and raw_pred is not None:
                arr = np.array(raw_pred)
                if arr.ndim == 2 and arr.shape[1] == 1:
                    prob = float(arr[0, 0])
                    pred_idx = 1 if prob >= 0.5 else 0
                    pred_label = class_names[pred_idx] if class_names and len(
                        class_names) >= pred_idx+1 else str(pred_idx)
                    st.markdown(f"""
                    <div style=\"background-color: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;\">
                        <h3>Resultado de la predicción</h3>
                        <p style=\"font-size: 22px; font-weight: bold; color: #2E7D32;\">Clase: {pred_label}</p>
                        <p style=\"margin:4px 0;\">Probabilidad clase positiva: {prob:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Multiclase softmax
                    probs = arr[0]
                    pred_idx = int(np.argmax(probs))
                    pred_label = class_names[pred_idx] if class_names and len(
                        class_names) > pred_idx else str(pred_idx)
                    st.markdown(f"""
                    <div style=\"background-color: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;\">
                        <h3>Resultado de la predicción</h3>
                        <p style=\"font-size: 22px; font-weight: bold; color: #2E7D32;\">Clase: {pred_label}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    # Tabla de probabilidades
                    if class_names:
                        prob_df = pd.DataFrame({
                            'Clase': class_names,
                            'Probabilidad': probs
                        })
                        st.dataframe(prob_df.style.format(
                            {'Probabilidad': '{:.4f}'}), use_container_width=True)
            else:  # scikit-learn modelos clásicos
                prediction = int(raw_pred[0])
                pred_label = class_names[prediction] if class_names and len(
                    class_names) > prediction else str(prediction)
                st.markdown(f"""
                <div style=\"background-color: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;\">
                    <h3>Resultado de la predicción</h3>
                    <p style=\"font-size: 24px; font-weight: bold; color: #2E7D32;\">Clase: {pred_label}</p>
                </div>
                """, unsafe_allow_html=True)

                # Camino de decisión / info modelo
                if hasattr(model, 'tree_'):
                    st.markdown("### Camino de decisión")
                    show_prediction_path(
                        model, new_data, feature_names, class_names)
                elif hasattr(model, 'n_neighbors'):
                    st.markdown("### Información del modelo KNN")
                    st.info(f"""
                        **Modelo KNN**
                        - K: {model.n_neighbors}
                        - Pesos: {model.weights}
                        - Métrica: {model.metric}
                    """)
                elif type(model).__name__ == "LogisticRegression" and hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(new_data)[0]
                    if class_names and len(probs) == len(class_names):
                        prob_df = pd.DataFrame(
                            {'Clase': class_names, 'Probabilidad': probs})
                        st.dataframe(prob_df.style.format(
                            {'Probabilidad': '{:.4f}'}), use_container_width=True)

        # --- REGRESIÓN ---
        else:
            if is_keras:
                val = float(np.array(raw_pred).reshape(-1)[0])
            else:
                val = float(raw_pred) if np.isscalar(raw_pred) else float(
                    np.array(raw_pred).reshape(-1)[0])
            st.markdown(f"""
            <div style=\"background-color: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;\">
                <h3>Resultado de la predicción</h3>
                <p style=\"font-size: 24px; font-weight: bold; color: #2E7D32;\">Valor: {val:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

            if hasattr(model, 'tree_'):
                st.markdown("### Camino de decisión")
                show_prediction_path(model, new_data, feature_names)
            elif hasattr(model, 'n_neighbors'):
                st.markdown("### Información del modelo KNN")
                st.info(f"""
                    **Modelo KNN**
                    - K: {model.n_neighbors}
                    - Pesos: {model.weights}
                    - Métrica: {model.metric}
                """)
            elif type(model).__name__ == "LinearRegression" and hasattr(model, 'coef_'):
                st.markdown("### Coeficientes")
                coef_df = pd.DataFrame({
                    'Característica': feature_names,
                    'Coeficiente': model.coef_.flatten() if len(getattr(model, 'coef_', [])) else []
                })
                if not coef_df.empty:
                    st.dataframe(coef_df)
