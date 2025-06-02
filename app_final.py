#!/usr/bin/env python3
"""
MLTutor: Plataforma educativa para el aprendizaje de Machine Learning.

Esta es la versión final refactorizada de la aplicación MLTutor, que integra
todos los módulos separados para crear una experiencia de aprendizaje coherente.
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
    setup_page, init_session_state, show_welcome_page, show_sidebar_config,
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

    # Configurar navegación principal
    navigation = st.sidebar.radio(
        "Navegación:",
        ["🏠 Inicio", "🌲 Árboles de Decisión", "📊 Regresión Logística (próximamente)",
         "🔍 K-Nearest Neighbors (próximamente)", "🧠 Redes Neuronales (próximamente)"]
    )

    # Página de inicio
    if navigation == "🏠 Inicio":
        show_welcome_page()
        return

    # Páginas de algoritmos
    if navigation == "🌲 Árboles de Decisión":
        run_decision_trees_app(viz_availability)
    elif navigation in ["📊 Regresión Logística (próximamente)",
                        "🔍 K-Nearest Neighbors (próximamente)",
                        "🧠 Redes Neuronales (próximamente)"]:
        st.header(f"{navigation.split(' ')[1]} {navigation.split(' ')[2]}")
        st.info("Esta funcionalidad estará disponible próximamente. Por ahora, puedes explorar los Árboles de Decisión.")

        # Mostrar una imagen ilustrativa según el algoritmo
        if "Regresión Logística" in navigation:
            st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_logistic_001.png",
                     caption="Ilustración de Regresión Logística")
        elif "K-Nearest Neighbors" in navigation:
            st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png",
                     caption="Ilustración de K-Nearest Neighbors")
        elif "Redes Neuronales" in navigation:
            st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_mlp_001.png",
                     caption="Ilustración de Redes Neuronales")


def run_decision_trees_app(viz_availability):
    """Ejecuta la aplicación específica de árboles de decisión."""
    st.header("Árboles de Decisión")
    st.markdown("Aprende sobre los árboles de decisión de forma interactiva")

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

    # Usar radio para emular tabs con estado persistente
    selected_tab = st.radio("", tab_options, index=st.session_state.active_tab,
                            horizontal=True, label_visibility="collapsed")
    st.session_state.active_tab = tab_options.index(selected_tab)

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
                    dataset_loaded = True
                    st.session_state.dataset_option = dataset_option

                    # Mostrar información del dataset
                    st.markdown(create_info_box(dataset_info),
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error al cargar los datos: {str(e)}")
                    dataset_loaded = False

            # Tipo de árbol
            tree_type = st.radio(
                "Tipo de árbol:",
                ("Clasificación", "Regresión"),
                help="Clasificación para predecir categorías, Regresión para valores continuos",
                key="tree_type_selector_config"
            )

            # Asegurarse de que el tipo de árbol sea consistente con los datos
            if dataset_loaded and task_type != tree_type:
                st.warning(
                    f"El conjunto de datos seleccionado es para {task_type}, pero se ha seleccionado un árbol de {tree_type}. Se utilizará {task_type}.")
                tree_type = task_type

            # Profundidad máxima
            max_depth = st.slider(
                "Profundidad máxima:",
                1, 10, st.session_state.max_depth,
                help="Controla la complejidad del árbol. Mayor profundidad puede llevar a sobreajuste."
            )

            # Muestras mínimas para dividir
            min_samples_split = st.slider(
                "Muestras mínimas para dividir:",
                2, 20, st.session_state.min_samples_split,
                help="Número mínimo de muestras requeridas para dividir un nodo."
            )

            # Criterio
            if tree_type == "Clasificación":
                criterion_options = ["gini", "entropy"]
                criterion_desc = {
                    "gini": "Mide la impureza (menos homogéneo = mayor impureza)",
                    "entropy": "Mide la ganancia de información (entropía de Shannon)"
                }
            else:
                criterion_options = ["squared_error",
                                     "friedman_mse", "absolute_error", "poisson"]
                criterion_desc = {
                    "squared_error": "Minimiza el error cuadrático medio (MSE)",
                    "friedman_mse": "Mejora la selección de características",
                    "absolute_error": "Minimiza el error absoluto medio (MAE)",
                    "poisson": "Para datos que siguen distribución de Poisson"
                }

            criterion = st.selectbox(
                "Criterio de división:",
                criterion_options,
                index=criterion_options.index(
                    st.session_state.criterion) if st.session_state.criterion in criterion_options else 0,
                help="Método para evaluar la calidad de una división del nodo.",
                key="criterion_selector_config"
            )
            st.caption(criterion_desc[criterion])

            # Porcentaje de división de datos
            test_size = st.slider(
                "Porcentaje para prueba:",
                0.1, 0.5, st.session_state.test_size, 0.05,
                help="Porcentaje de datos reservados para evaluar el modelo."
            )

            # Si se presiona el botón entrenar
            train_pressed = st.button("Entrenar Árbol", type="primary")
            if train_pressed:
                # Actualizar parámetros en el estado de la sesión
                st.session_state.max_depth = max_depth
                st.session_state.min_samples_split = min_samples_split
                st.session_state.criterion = criterion
                st.session_state.test_size = test_size
                st.session_state.tree_type = tree_type

                try:
                    # Entrenar modelo
                    with st.spinner("Entrenando el modelo..."):
                        results = train_decision_tree(
                            X, y,
                            tree_type=tree_type,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            criterion=criterion,
                            test_size=test_size
                        )

                        # Guardar resultados en el estado de la sesión
                        st.session_state.tree_model = results["model"]
                        st.session_state.X_train = results["X_train"]
                        st.session_state.X_test = results["X_test"]
                        st.session_state.y_train = results["y_train"]
                        st.session_state.y_test = results["y_test"]
                        st.session_state.feature_names = feature_names
                        st.session_state.class_names = class_names
                        st.session_state.tree_type = tree_type
                        st.session_state.test_results = results["evaluation"]
                        st.session_state.is_trained = True

                        # Mostrar mensaje de éxito
                        st.success("¡Modelo entrenado correctamente!")

                        # Avanzar a la pestaña de evaluación automáticamente
                        st.session_state.active_tab = 2
                        st.rerun()
                except Exception as e:
                    st.error(f"Error al entrenar el modelo: {str(e)}")
                    st.session_state.is_trained = False

            # Mostrar explicaciones sobre los parámetros
            with st.expander("ℹ️ ¿Qué son los parámetros?"):
                st.markdown("""
                **Profundidad máxima**: Limita cuántos niveles puede tener el árbol. Un valor bajo crea un árbol simple que puede no capturar todos los patrones. Un valor alto puede crear un árbol complejo que se sobreajusta a los datos.

                **Muestras mínimas**: Número mínimo de muestras necesarias para dividir un nodo. Valores más altos previenen divisiones con pocos ejemplos, reduciendo el sobreajuste.

                **Criterio**: Método usado para evaluar la calidad de una división. En clasificación, 'gini' mide la impureza y 'entropy' la ganancia de información. En regresión, generalmente se usa el error cuadrático.

                **Porcentaje para prueba**: Fracción de datos que se reservan para evaluar el modelo. No se usan durante el entrenamiento.
                """)

        # Pestaña de Datos
        elif st.session_state.active_tab == 1:
            st.header("Exploración de Datos")

            # Cargar los datos si no están cargados
            if not dataset_loaded:
                try:
                    X, y, feature_names, class_names, dataset_info, task_type = load_data(
                        st.session_state.dataset_option)
                    dataset_loaded = True
                except Exception as e:
                    st.error(f"Error al cargar los datos: {str(e)}")
                    dataset_loaded = False

            if dataset_loaded:
                # Crear un DataFrame con los datos
                df = pd.DataFrame(X, columns=feature_names)
                if isinstance(y, np.ndarray):
                    if st.session_state.tree_type == "Clasificación" and class_names:
                        df['target'] = [class_names[int(val)] for val in y]
                    else:
                        df['target'] = y
                else:  # Si y ya es una Serie o similar
                    df['target'] = y

                # Mostrar información del dataset
                st.markdown(create_info_box(dataset_info),
                            unsafe_allow_html=True)

                # Estadísticas básicas
                st.subheader("Estadísticas básicas")
                st.write(df.describe())

                # Mostrar los primeros registros
                st.subheader("Primeras filas del dataset")
                st.dataframe(df.head(10))

                # Visualización de datos
                st.subheader("Visualización de datos")

                # Opciones de visualización
                viz_data_options = [
                    "Scatter Plot", "Matriz de Correlación", "Matriz de Dispersión"]
                viz_data_type = st.radio(
                    "Tipo de visualización:", viz_data_options, horizontal=True, key="viz_data_type")

                if viz_data_type == "Scatter Plot":
                    # Permitir seleccionar características para visualizar
                    if len(feature_names) > 1:
                        cols = st.columns(2)
                        with cols[0]:
                            x_axis = st.selectbox(
                                "Eje X:", feature_names, index=0, key="x_axis_selector")
                        with cols[1]:
                            y_axis = st.selectbox(
                                "Eje Y:", feature_names, index=min(1, len(feature_names)-1), key="y_axis_selector")

                        # Crear scatter plot
                        fig, ax = plt.subplots(figsize=(10, 6))

                        if task_type == "Clasificación":
                            # Colorear por clase para clasificación
                            for i, class_name in enumerate(class_names):
                                # Obtener índices para esta clase
                                if isinstance(y, np.ndarray):
                                    idx = np.where(y == i)[0]
                                    # Si X es DataFrame, usamos iloc para acceder por índice
                                    if isinstance(X, pd.DataFrame):
                                        x_values = X.iloc[idx][x_axis]
                                        y_values = X.iloc[idx][y_axis]
                                    else:
                                        # Si X es array, usamos los índices de feature_names
                                        x_values = X[idx,
                                                     feature_names.index(x_axis)]
                                        y_values = X[idx,
                                                     feature_names.index(y_axis)]
                                else:
                                    # Si y es Series, usamos métodos de pandas
                                    idx = y == i
                                    x_values = X[idx][x_axis]
                                    y_values = X[idx][y_axis]

                                ax.scatter(x_values, y_values,
                                           label=class_name, alpha=0.7)
                            ax.legend(title="Clase")
                        else:
                            # Colorear por valor para regresión
                            if isinstance(X, pd.DataFrame):
                                scatter = ax.scatter(X[x_axis],
                                                     X[y_axis],
                                                     c=y, cmap='viridis', alpha=0.7)
                            else:
                                scatter = ax.scatter(X[:, feature_names.index(x_axis)],
                                                     X[:, feature_names.index(
                                                         y_axis)],
                                                     c=y, cmap='viridis', alpha=0.7)
                            plt.colorbar(scatter, ax=ax,
                                         label="Valor objetivo")

                        ax.set_xlabel(x_axis)
                        ax.set_ylabel(y_axis)
                        ax.set_title(f"Relación entre {x_axis} y {y_axis}")
                        ax.grid(True, linestyle='--', alpha=0.7)

                        st.pyplot(fig)

                        # Enlace para descargar la imagen
                        st.markdown(get_image_download_link(fig, "scatter_plot", "📥 Descargar gráfico"),
                                    unsafe_allow_html=True)

                        # Mostrar el código que genera este gráfico
                        if tree_type == "Clasificación":
                            code_scatter = f"""
import matplotlib.pyplot as plt
import numpy as np

# Crear la figura
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar cada clase con un color diferente
for i, class_name in enumerate(class_names):
    # Obtener índices para esta clase
    idx = np.where(y == i)[0]
    
    # Extraer valores para las características seleccionadas
    if isinstance(X, pd.DataFrame):
        x_values = X.iloc[idx]['{x_axis}']
        y_values = X.iloc[idx]['{y_axis}']
    else:
        x_idx = feature_names.index('{x_axis}')
        y_idx = feature_names.index('{y_axis}')
        x_values = X[idx, x_idx]
        y_values = X[idx, y_idx]
    
    # Graficar los puntos
    ax.scatter(x_values, y_values, label=class_name, alpha=0.7)

# Añadir leyenda y etiquetas
ax.legend(title="Clase")
ax.set_xlabel('{x_axis}')
ax.set_ylabel('{y_axis}')
ax.set_title('Relación entre {x_axis} y {y_axis}')
ax.grid(True, linestyle='--', alpha=0.7)

# Para mostrar en Streamlit
# st.pyplot(fig)

# Para uso normal en Python/Jupyter
# plt.tight_layout()
# plt.show()
"""
                        else:
                            code_scatter = f"""
import matplotlib.pyplot as plt
import numpy as np

# Crear la figura
fig, ax = plt.subplots(figsize=(10, 6))

# Extraer valores para las características seleccionadas
if isinstance(X, pd.DataFrame):
    x_values = X['{x_axis}']
    y_values = X['{y_axis}']
else:
    x_idx = feature_names.index('{x_axis}')
    y_idx = feature_names.index('{y_axis}')
    x_values = X[:, x_idx]
    y_values = X[:, y_idx]

# Graficar los puntos coloreados por el valor objetivo
scatter = ax.scatter(x_values, y_values, c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, ax=ax, label="Valor objetivo")

# Añadir etiquetas
ax.set_xlabel('{x_axis}')
ax.set_ylabel('{y_axis}')
ax.set_title('Relación entre {x_axis} y {y_axis}')
ax.grid(True, linestyle='--', alpha=0.7)

# Para mostrar en Streamlit
# st.pyplot(fig)

# Para uso normal en Python/Jupyter
# plt.tight_layout()
# plt.show()
"""
                        show_code_with_download(
                            code_scatter, "Código para generar este gráfico", "scatter_plot.py")

                elif viz_data_type == "Matriz de Correlación":
                    st.write("### Matriz de Correlación")
                    st.write(
                        "Muestra las correlaciones entre las características del dataset.")

                    # Calcular la matriz de correlación
                    corr_matrix = df.corr()

                    # Crear el heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    cmap = sns.diverging_palette(220, 10, as_cmap=True)

                    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                                annot=True, fmt=".2f", square=True, linewidths=.5, ax=ax)

                    plt.title('Matriz de Correlación de Características')
                    st.pyplot(fig)

                    # Enlace para descargar la imagen
                    st.markdown(get_image_download_link(fig, "matriz_correlacion", "📥 Descargar matriz de correlación"),
                                unsafe_allow_html=True)

                    # Mostrar el código que genera este gráfico
                    code_corr = """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Calcular la matriz de correlación
corr_matrix = df.corr()

# Crear el heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
           annot=True, fmt=".2f", square=True, linewidths=.5, ax=ax)

plt.title('Matriz de Correlación de Características')

# Para mostrar en Streamlit
# st.pyplot(fig)

# Para uso normal en Python/Jupyter
# plt.tight_layout()
# plt.show()
"""
                    show_code_with_download(
                        code_corr, "Código para generar esta matriz", "matriz_correlacion.py")

                    # Añadir explicación
                    st.write("""
                    **Interpretación:**
                    - Valores cercanos a 1: Fuerte correlación positiva
                    - Valores cercanos a -1: Fuerte correlación negativa
                    - Valores cercanos a 0: Poca o ninguna correlación
                    """)

                elif viz_data_type == "Matriz de Dispersión":
                    st.write("### Matriz de Dispersión (Pair Plot)")
                    st.write(
                        "Muestra relaciones entre múltiples características simultáneamente.")

                    # Opciones de configuración
                    with st.expander("Opciones de visualización"):
                        # Permitir seleccionar un subconjunto de características si hay muchas
                        if len(feature_names) > 5:
                            st.warning(
                                "Selecciona un subconjunto de características para visualizar (máximo 5 recomendado).")
                            selected_features = st.multiselect(
                                "Características a visualizar:",
                                feature_names,
                                # Por defecto, mostrar las primeras 4
                                default=feature_names[:4],
                                max_selections=5
                            )
                        else:
                            selected_features = feature_names

                        # Si no se seleccionó ninguna característica, usar todas (hasta 5)
                        if not selected_features:
                            selected_features = feature_names[:min(
                                5, len(feature_names))]

                        # Opciones de estilo
                        col1, col2 = st.columns(2)
                        with col1:
                            corner = st.checkbox("Mostrar solo mitad inferior", value=True,
                                                 help="Muestra solo la mitad inferior de la matriz para evitar duplicados")
                        with col2:
                            diag_kind = st.radio("Visualización en diagonal:",
                                                 ["hist", "kde"],
                                                 horizontal=True,
                                                 help="Histograma o gráfico de densidad en la diagonal")

                    # Crear un DataFrame con los datos seleccionados
                    plot_df = pd.DataFrame(X, columns=feature_names)[
                        selected_features].copy()

                    # Añadir la columna objetivo para colorear los puntos
                    if st.session_state.tree_type == "Clasificación" and class_names:
                        if isinstance(y, np.ndarray):
                            plot_df['target'] = [
                                class_names[int(val)] for val in y]
                        else:
                            plot_df['target'] = y
                        hue_col = 'target'
                    else:
                        # Para regresión, no usamos color
                        hue_col = None

                    # Crear la figura
                    with st.spinner("Generando matriz de dispersión..."):
                        try:
                            # Ajustar el tamaño según el número de características
                            fig_size = max(8, len(selected_features) * 2)

                            if corner:
                                # Pairplot con solo la mitad inferior
                                g = sns.PairGrid(plot_df, hue=hue_col, diag_sharey=False,
                                                 height=fig_size /
                                                 len(selected_features),
                                                 aspect=1)
                                g.map_lower(sns.scatterplot, alpha=0.7)
                                if diag_kind == "hist":
                                    g.map_diag(sns.histplot)
                                else:
                                    g.map_diag(sns.kdeplot)

                                if hue_col:
                                    g.add_legend(title="Clase", bbox_to_anchor=(
                                        1.05, 1), loc='upper left')
                            else:
                                # Usar pairplot directamente para la matriz completa
                                g = sns.pairplot(plot_df, hue=hue_col, diag_kind=diag_kind,
                                                 height=fig_size /
                                                 len(selected_features),
                                                 plot_kws={'alpha': 0.7})

                            # Ajustar espaciado y título
                            plt.tight_layout()
                            plt.suptitle('Matriz de Dispersión de Características',
                                         fontsize=16, y=1.02)

                            # Mostrar la figura
                            st.pyplot(g.fig)

                            # Enlace para descargar la imagen
                            st.markdown(get_image_download_link(g.fig, "matriz_dispersion", "📥 Descargar matriz de dispersión"),
                                        unsafe_allow_html=True)

                            # Mostrar el código que genera este gráfico
                            code_scatter = f"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Seleccionar las características para visualizar
selected_features = {selected_features}

# Crear un DataFrame con los datos seleccionados
plot_df = pd.DataFrame(X, columns=feature_names)[selected_features].copy()

# Añadir la columna objetivo para colorear los puntos
if tree_type == "Clasificación" and class_names:
    plot_df['target'] = [class_names[int(val)] for val in y]
    hue_col = 'target'
else:
    hue_col = None

# Ajustar el tamaño según el número de características
fig_size = max(8, len(selected_features) * 2)

# Crear la visualización
if {corner}:  # Mostrar solo mitad inferior
    g = sns.PairGrid(plot_df, hue=hue_col, diag_sharey=False,
                    height=fig_size / len(selected_features),
                    aspect=1)
    g.map_lower(sns.scatterplot, alpha=0.7)
    if "{diag_kind}" == "hist":
        g.map_diag(sns.histplot)
    else:
        g.map_diag(sns.kdeplot)
    
    if hue_col:
        g.add_legend(title="Clase", bbox_to_anchor=(1.05, 1), loc='upper left')
else:
    # Usar pairplot directamente para la matriz completa
    g = sns.pairplot(plot_df, hue=hue_col, diag_kind="{diag_kind}",
                    height=fig_size / len(selected_features),
                    plot_kws={{'alpha': 0.7}})

# Ajustar espaciado y título
plt.tight_layout()
plt.suptitle('Matriz de Dispersión de Características', fontsize=16, y=1.02)

# Para mostrar en Streamlit
# st.pyplot(g.fig)

# Para uso normal en Python/Jupyter
# plt.show()
"""
                            show_code_with_download(
                                code_scatter, "Código para generar esta matriz", "matriz_dispersion.py")

                            # Añadir explicación
                            st.write("""
                            **Interpretación:**
                            - Cada celda muestra la relación entre dos características
                            - La diagonal muestra la distribución de cada característica
                            - Los colores representan las diferentes clases
                            - Patrones claros de separación entre colores indican que esas características son útiles para distinguir clases
                            """)

                        except Exception as e:
                            st.error(
                                f"Error al generar la matriz de dispersión: {str(e)}")
                            st.info(
                                "Intenta seleccionar menos características o un subconjunto diferente.")
            else:
                st.info(
                    "Selecciona un dataset en la pestaña de configuración para visualizar los datos.")

        # Pestaña de Evaluación
        elif st.session_state.active_tab == 2:
            st.header("Evaluación del Modelo")

            if not st.session_state.is_trained:
                st.warning(
                    "Primero debes entrenar un modelo en la pestaña de Configuración.")
            else:
                # Mostrar métricas detalladas
                show_detailed_evaluation(
                    st.session_state.y_test,
                    st.session_state.test_results["y_pred"],
                    st.session_state.class_names if st.session_state.tree_type == "Clasificación" else None,
                    st.session_state.tree_type
                )

        # Pestaña de Visualización
        elif st.session_state.active_tab == 3:
            st.header("Visualización del Árbol")

            if not st.session_state.is_trained:
                st.warning(
                    "Primero debes entrenar un modelo en la pestaña de Configuración.")
            else:
                # Usar un tamaño fijo para las figuras
                fig_width = 12
                fig_height = 8

                # Selección de tipo de visualización
                viz_options = ["Estándar", "Texto"]

                # Añadir opción de frontera de decisión solo para clasificación y 2+ características
                if st.session_state.tree_type == "Clasificación" and len(st.session_state.feature_names) >= 2:
                    viz_options.append("Frontera de Decisión")

                # Eliminamos las opciones dinámicas y dejamos solo las estáticas
                viz_type = st.radio(
                    "Tipo de visualización:",
                    viz_options,
                    horizontal=True,
                    key="viz_type_selector"
                )

                if viz_type == "Estándar":
                    # Visualización estándar mejorada con más explicaciones
                    st.write("### Visualización del árbol de decisión")
                    st.write("""
                    **Cómo interpretar esta visualización:**
                    - Cada nodo muestra la condición de división (feature y umbral)
                    - Los nodos coloreados muestran la distribución de clases o valores
                    - Las hojas (nodos finales) muestran la predicción para ese camino
                    - La impureza indica la homogeneidad de las muestras en el nodo
                    """)

                    fig = create_tree_visualization(
                        st.session_state.tree_model,
                        st.session_state.feature_names,
                        st.session_state.class_names if st.session_state.tree_type == "Clasificación" else None,
                        figsize=(fig_width, fig_height)
                    )
                    st.pyplot(fig)

                    # Añadir explicación adicional
                    if st.session_state.tree_type == "Clasificación":
                        st.write("""
                        **Detalles adicionales:**
                        - **gini/entropy:** Medida de impureza en cada nodo (menor es mejor)
                        - **samples:** Número de muestras en el nodo
                        - **value:** Distribución de muestras por clase
                        - **class:** Clase mayoritaria (predicción final en hojas)
                        """)
                    else:
                        st.write("""
                        **Detalles adicionales:**
                        - **mse/mae:** Error cuadrático/absoluto medio en cada nodo
                        - **samples:** Número de muestras en el nodo
                        - **value:** Valor medio de la variable objetivo
                        """)

                    # Enlace para descargar
                    st.markdown(get_image_download_link(
                        fig, "arbol_decision", "📥 Descargar visualización del árbol"),
                        unsafe_allow_html=True
                    )

                elif viz_type == "Texto":
                    # Visualización como texto con explicación
                    st.write("### Representación textual del árbol")
                    st.write("""
                    **Cómo interpretar esta visualización:**
                    - Cada línea representa un nodo o una hoja
                    - La indentación indica la profundidad en el árbol
                    - Las condiciones muestran la regla de decisión en cada nodo
                    - Los valores finales muestran la predicción para cada camino
                    """)

                    tree_text = get_tree_text(
                        st.session_state.tree_model,
                        st.session_state.feature_names
                    )
                    st.text(tree_text)

                    # Enlace para descargar texto
                    text_bytes = tree_text.encode()
                    b64 = base64.b64encode(text_bytes).decode()
                    href = f'<a href="data:text/plain;base64,{b64}" download="arbol_texto.txt">📥 Descargar texto del árbol</a>'
                    st.markdown(href, unsafe_allow_html=True)

                elif viz_type == "Frontera de Decisión":
                    st.write("### Visualización de Frontera de Decisión")
                    st.write("""
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
                                key="feature1_selector"
                            )
                        with cols[1]:
                            feature2 = st.selectbox(
                                "Segunda característica:",
                                st.session_state.feature_names,
                                index=1,
                                key="feature2_selector"
                            )

                        # Obtener índices de las características seleccionadas
                        # Asegurarse de que feature_names sea una lista
                        feature_names_list = list(
                            st.session_state.feature_names)
                        f1_idx = feature_names_list.index(feature1)
                        f2_idx = feature_names_list.index(feature2)

                        # Crear DataFrame o array con solo las dos características seleccionadas para la frontera de decisión
                        if isinstance(st.session_state.X_train, pd.DataFrame):
                            # Para DataFrames, seleccionamos directamente por nombre de columna y mantenemos como DataFrame
                            X_boundary = st.session_state.X_train[[
                                feature1, feature2]]
                        else:
                            # Para arrays de NumPy, usamos índices numéricos y creamos un nuevo array
                            X_boundary = np.column_stack([
                                st.session_state.X_train[:, f1_idx],
                                st.session_state.X_train[:, f2_idx]
                            ])
                        feature_names_boundary = [feature1, feature2]
                    else:
                        # Si solo hay dos características, usarlas directamente
                        if isinstance(st.session_state.X_train, pd.DataFrame):
                            # Para DataFrames, mantenemos como DataFrame
                            X_boundary = st.session_state.X_train
                        else:
                            # Para arrays de NumPy, usamos directamente
                            X_boundary = st.session_state.X_train
                        feature_names_boundary = st.session_state.feature_names

                    # Crear figura y dibujar frontera de decisión
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plot_decision_boundary(
                        st.session_state.tree_model,
                        X_boundary,
                        st.session_state.y_train,
                        ax=ax,
                        feature_names=feature_names_boundary,
                        class_names=st.session_state.class_names
                    )

                    # Mostrar la figura
                    st.pyplot(fig)

                    # Enlace para descargar
                    st.markdown(get_image_download_link(
                        fig, "frontera_decision", "📥 Descargar visualización de frontera de decisión"),
                        unsafe_allow_html=True
                    )

                    # Explicación adicional
                    st.write("""
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

        # Pestaña de Características
        elif st.session_state.active_tab == 4:
            st.header("Importancia de Características")

            if not st.session_state.is_trained:
                st.warning(
                    "Primero debes entrenar un modelo en la pestaña de Configuración.")
            else:
                display_feature_importance(
                    st.session_state.tree_model,
                    st.session_state.feature_names
                )

        # Pestaña de Predicciones
        elif st.session_state.active_tab == 5:
            st.header("Realizar Predicciones")

            if not st.session_state.is_trained:
                st.warning(
                    "Primero debes entrenar un modelo en la pestaña de Configuración.")
            else:
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
                    "Primero debes entrenar un modelo en la pestaña de Configuración.")
            else:
                display_model_export_options(
                    st.session_state.tree_model,
                    st.session_state.feature_names,
                    st.session_state.class_names,
                    st.session_state.tree_type,
                    st.session_state.max_depth,
                    st.session_state.min_samples_split,
                    st.session_state.criterion
                )

    else:  # Si no se selecciona Árboles de Decisión
        st.info(
            "Configura los parámetros en el panel lateral y haz clic en 'Entrenar Árbol' para comenzar.")

        # Mostrar imagen ilustrativa
        st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_tree_regression_001.png",
                 caption="Ejemplo de un Árbol de Decisión")

    # Pie de página
    st.markdown("""
    <div class="footer">
        <p>MLTutor - Plataforma Educativa de Machine Learning</p>
        <p>© 2023 - Desarrollado con fines educativos</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Ejecutar la aplicación
    main()
