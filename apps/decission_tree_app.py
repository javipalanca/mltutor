import base64

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

from dataset.dataset_manager import load_data, preprocess_data
from algorithms.model_training import train_decision_tree
from algorithms.model_evaluation import evaluate_classification_model, evaluate_regression_model, show_detailed_evaluation
from utils import create_info_box, get_image_download_link, show_code_with_download
from dataset.dataset_tab import run_dataset_tab, run_select_dataset
from algorithms.code_examples import DECISION_BOUNDARY_CODE, VIZ_TREE_CODE, TEXT_TREE_CODE, generate_decision_boundary_code
from viz.tree_visualizer import get_tree_text
from viz.decision_boundary import plot_decision_boundary


def display_feature_importance(model, feature_names):
    """
    Muestra la importancia de las características usando solo librerías de terceros.
    """
    # Obtener importancias del modelo
    importances = model.feature_importances_

    # Crear DataFrame para ordenar
    import pandas as pd
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)

    # Crear visualización
    fig, ax = plt.subplots(figsize=(10, 8))

    # Barplot horizontal
    bars = ax.barh(
        feature_importance_df['feature'], feature_importance_df['importance'])

    # Colorear barras
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Importancia')
    ax.set_title('Importancia de Características')
    ax.grid(True, alpha=0.3)

    # Añadir valores en las barras
    for i, (feature, importance) in enumerate(zip(feature_importance_df['feature'], feature_importance_df['importance'])):
        ax.text(importance + 0.001, i, f'{importance:.3f}', va='center')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Mostrar tabla con los valores
    st.markdown("### Valores de Importancia")
    importance_table = feature_importance_df.sort_values(
        'importance', ascending=False)
    importance_table['importance'] = importance_table['importance'].round(4)
    st.dataframe(importance_table, use_container_width=True)


def create_prediction_interface(model, feature_names, class_names, task_type, X_train=None, dataset_name='Dataset'):
    """
    Crea una interfaz para hacer predicciones con nuevos datos usando solo librerías de terceros.
    """
    st.markdown("### Hacer Predicción")

    # Crear inputs para cada característica
    inputs = {}

    # Usar rangos dinámicos si se proporcionan datos de entrenamiento
    if X_train is not None:
        if hasattr(X_train, 'iloc'):
            # DataFrame
            mins = X_train.min()
            maxs = X_train.max()
            means = X_train.mean()
        else:
            # NumPy array
            mins = np.min(X_train, axis=0)
            maxs = np.max(X_train, axis=0)
            means = np.mean(X_train, axis=0)
    else:
        mins = [0.0] * len(feature_names)
        maxs = [100.0] * len(feature_names)
        means = [50.0] * len(feature_names)

    # Crear sliders o inputs numéricos
    cols = st.columns(2)
    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            if X_train is not None:
                min_val = float(mins[i] if hasattr(
                    mins, '__getitem__') else mins.iloc[i])
                max_val = float(maxs[i] if hasattr(
                    maxs, '__getitem__') else maxs.iloc[i])
                mean_val = float(means[i] if hasattr(
                    means, '__getitem__') else means.iloc[i])

                inputs[feature] = st.slider(
                    f"{feature}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )
            else:
                inputs[feature] = st.number_input(f"{feature}:", value=0.0)

    # Botón para hacer predicción
    if st.button("Hacer Predicción", type="primary"):
        # Preparar datos de entrada
        input_values = [inputs[feature] for feature in feature_names]
        input_array = np.array(input_values).reshape(1, -1)

        try:
            # Hacer predicción
            prediction = model.predict(input_array)[0]

            if task_type == "Clasificación":
                # Obtener probabilidades si están disponibles
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_array)[0]

                st.success(
                    f"**Predicción:** {class_names[prediction] if class_names else f'Clase {prediction}'}")

                if hasattr(model, 'predict_proba'):
                    st.markdown("### Probabilidades por Clase")
                    prob_df = pd.DataFrame({
                        'Clase': class_names if class_names else [f'Clase {i}' for i in range(len(probabilities))],
                        'Probabilidad': probabilities
                    })
                    prob_df['Probabilidad (%)'] = (
                        prob_df['Probabilidad'] * 100).round(2)
                    st.dataframe(prob_df, use_container_width=True)

                    # Gráfico de barras de probabilidades
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(prob_df['Clase'], prob_df['Probabilidad'])
                    ax.set_ylabel('Probabilidad')
                    ax.set_title('Probabilidades de Predicción')
                    ax.set_ylim(0, 1)

                    # Colorear la barra de la predicción
                    bars[prediction].set_color('red')
                    for i, bar in enumerate(bars):
                        if i != prediction:
                            bar.set_color('lightblue')

                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
            else:
                st.success(f"**Predicción:** {prediction:.4f}")

        except Exception as e:
            st.error(f"Error al hacer la predicción: {str(e)}")


def display_model_export_options(model, feature_names, class_names, task_type, max_depth, min_samples_split, criterion):
    """
    Muestra opciones para exportar el modelo usando solo librerías de terceros.
    """
    st.markdown("### Opciones de Exportación")

    # Información del modelo
    st.markdown("#### Información del Modelo")
    model_info = {
        "Tipo de Tarea": task_type,
        "Criterio": criterion,
        "Profundidad Máxima": max_depth,
        "Muestras Mínimas para División": min_samples_split,
        "Número de Características": len(feature_names),
        "Número de Hojas": model.get_n_leaves(),
        "Profundidad del Árbol": model.get_depth()
    }

    info_df = pd.DataFrame(list(model_info.items()),
                           columns=['Parámetro', 'Valor'])
    st.dataframe(info_df, use_container_width=True)

    # Opciones de exportación
    st.markdown("#### Formatos de Exportación")

    col1, col2 = st.columns(2)

    with col1:
        # Exportar como pickle
        if st.button("📦 Exportar como Pickle", use_container_width=True):
            import pickle
            import io

            # Crear objeto para serializar
            model_data = {
                'model': model,
                'feature_names': feature_names,
                'class_names': class_names,
                'task_type': task_type,
                'model_info': model_info
            }

            # Serializar
            buffer = io.BytesIO()
            pickle.dump(model_data, buffer)
            buffer.seek(0)

            st.download_button(
                label="Descargar Modelo (.pkl)",
                data=buffer.getvalue(),
                file_name=f"decision_tree_model.pkl",
                mime="application/octet-stream"
            )

    with col2:
        # Exportar reglas como texto
        if st.button("📝 Exportar Reglas como Texto", use_container_width=True):
            from sklearn.tree import export_text

            tree_rules = export_text(
                model,
                feature_names=feature_names
            )

            st.download_button(
                label="Descargar Reglas (.txt)",
                data=tree_rules,
                file_name="decision_tree_rules.txt",
                mime="text/plain"
            )

    # Código Python para usar el modelo
    st.markdown("#### Código Python para Usar el Modelo")

    python_code = f"""
# Código para usar el modelo de árbol de decisión
import pickle
import numpy as np

# Cargar el modelo
with open('decision_tree_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['feature_names']
class_names = model_data['class_names']
task_type = model_data['task_type']

# Función para hacer predicciones
def predecir(valores_caracteristicas):
    \"\"\"
    Hace una predicción con el modelo de árbol de decisión.
    
    Args:
        valores_caracteristicas (list): Lista con valores para cada característica
                                      en el orden: {feature_names}
    
    Returns:
        Predicción del modelo
    \"\"\"
    # Convertir a array numpy
    X = np.array(valores_caracteristicas).reshape(1, -1)
    
    # Hacer predicción
    prediction = model.predict(X)[0]
    
    if task_type == "Clasificación":
        if class_names:
            return class_names[prediction]
        else:
            return f"Clase {{prediction}}"
    else:
        return prediction

# Ejemplo de uso:
# resultado = predecir([valor1, valor2, valor3, ...])
# print(f"Predicción: {{resultado}}")

# Para obtener probabilidades (solo clasificación):
if task_type == "Clasificación" and hasattr(model, 'predict_proba'):
    def predecir_con_probabilidades(valores_caracteristicas):
        X = np.array(valores_caracteristicas).reshape(1, -1)
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        resultado = {{
            'prediccion': class_names[prediction] if class_names else f"Clase {{prediction}}",
            'probabilidades': {{
                (class_names[i] if class_names else f"Clase {{i}}"): prob 
                for i, prob in enumerate(probabilities)
            }}
        }}
        
        return resultado
"""

    st.code(python_code, language='python')

    # Botón para descargar el código
    st.download_button(
        label="📥 Descargar Código Python",
        data=python_code,
        file_name="usar_modelo.py",
        mime="text/plain"
    )


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

    ###########################################
    #     Pestaña de Datos                    #
    ###########################################

    run_dataset_tab(st.session_state.active_tab)

    ###########################################
    # Pestaña de Entrenamiento                #
    ###########################################

    if st.session_state.active_tab == 1:
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
                st.session_state.class_names = class_names

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
                    tree_model = train_decision_tree(
                        X_train, y_train,
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        tree_type=tree_type
                    )

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

                    st.success(
                        "¡Modelo entrenado con éxito! Ahora puedes explorar las otras pestañas.")

                    # Sugerir ir a la pestaña de visualización
                    st.info(
                        "👉 Ve a la pestaña '🌲 Visualización' para ver el árbol generado.")

                except Exception as e:
                    st.error(f"Error al entrenar el modelo: {str(e)}")

    ###########################################
    # Pestaña de Evaluación                   #
    ###########################################
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

    ###########################################
    # Pestaña de Visualización                #
    ###########################################

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
                code_tree = VIZ_TREE_CODE

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
                code_text = TEXT_TREE_CODE

                show_code_with_download(
                    code_text, "Código para obtener el texto del árbol", "texto_arbol.py")

            elif viz_type == "Frontera":
                # Verificar que es un modelo de clasificación y que está entrenado
                if not st.session_state.get('is_trained', False):
                    st.warning(
                        "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'.")
                elif st.session_state.get('tree_type', 'Clasificación') == "Clasificación":

                    # Crear un nuevo modelo entrenado solo con las 2 características seleccionadas
                    # para que sea compatible con DecisionBoundaryDisplay
                    try:
                        model_2d = DecisionTreeClassifier(
                            max_depth=getattr(
                                st.session_state.tree_model, 'max_depth', None),
                            min_samples_split=getattr(
                                st.session_state.tree_model, 'min_samples_split', 2),
                            criterion=getattr(
                                st.session_state.tree_model, 'criterion', 'gini'),
                            random_state=42
                        )

                        plot_decision_boundary(
                            model_2d,
                            st.session_state.X_train,
                            st.session_state.y_train,
                            st.session_state.feature_names,
                            st.session_state.class_names,
                            fig_width,
                            fig_height
                        )
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
                else:
                    st.warning(
                        "La visualización de frontera de decisión solo está disponible para modelos de clasificación.")

    ###########################################
    # Pestaña de Características              #
    ###########################################
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

    ###########################################
    # Pestaña de Predicciones                 #
    ###########################################
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

    ###########################################
    # Pestaña de Exportar Modelo              #
    ###########################################
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
