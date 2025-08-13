import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dataset_manager import load_data
from utils import create_info_box, get_image_download_link, show_code_with_download
from algorithms.code_examples import SCATTERPLOT_MATRIX


def run_dataset_tab(active_tab, tips=None):
    # SELECTOR UNIFICADO DE DATASET (solo mostrar en pestañas que lo necesiten)
    if active_tab in [0, 1]:  # Exploración y Entrenamiento
        st.header("📊 Selección y Preparación de Datos")
        if tips:
            # Tips educativos sobre datos
            st.info(tips)
        run_select_dataset()
        if active_tab == 0:
            run_explore_dataset_tab()
    else:
        # Para otras pestañas, mostrar qué dataset está seleccionado actualmente
        if hasattr(st.session_state, 'selected_dataset'):
            st.info(
                f"📊 **Dataset actual:** {st.session_state.selected_dataset}")
            st.markdown("---")


def run_select_dataset():
    st.markdown("### 🎯 Selección de Dataset")

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
    st.session_state.selected_dataset = dataset_option

    # Separador después del selector
    st.markdown("---")


def show_dataset_info():
    # Mostrar información del dataset seleccionado
    if 'selected_dataset' in st.session_state:
        dataset_name = st.session_state.selected_dataset
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
            st.session_state.df = df
            st.session_state.target_col = target_col
            st.session_state.feature_names = feature_names
            st.session_state.class_names = class_names
            st.session_state.task_type = task_type
            st.session_state.dataset_info = dataset_info

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
                st.metric(f"{task_icon} Tipo de Tarea", task_type,
                          help="Clasificación o Regresión")
        except Exception as e:
            st.error(f"Error cargando dataset: {str(e)}")
            st.info("Por favor, selecciona un dataset válido.")
    else:
        st.warning("⚠️ Por favor, selecciona un dataset para continuar.")


def run_explore_dataset_tab():

    show_dataset_info()
    # Pestaña de Datos
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

        # Determinar el nombre de la columna objetivo
        if class_names is not None and len(class_names) > 0:
            # Para clasificación, usar el nombre de la variable objetivo del dataset_info
            target_col = 'target'  # Nombre por defecto
            if hasattr(dataset_info, 'target_names'):
                target_col = dataset_info.target_names[0]
        else:
            # Para regresión
            target_col = 'target'

        y_df = pd.Series(y, name=target_col)

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
        # st.markdown("### Información del Dataset")
        # st.markdown(create_info_box(dataset_info), unsafe_allow_html=True)

        # Mostrar las primeras filas de los datos
        st.markdown("### 👀 Vista previa de datos")
        st.dataframe(df_display.head(10))

        # Tip sobre los datos
        with st.expander("💡 ¿Qué significan estos datos?"):
            st.markdown(f"""
            - **Filas**: Cada fila es un ejemplo que el modelo usará para aprender
            - **Columnas de características**: Las variables que el modelo analiza para hacer predicciones
            - **Variable objetivo ({target_col})**: Lo que queremos predecir
            """)

        # Estadísticas descriptivas
        st.markdown("### Estadísticas Descriptivas")
        st.dataframe(df_display.describe())

        # Distribución de clases o valores objetivo
        st.markdown("### 🎯 Distribución del Objetivo")

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

            # Explicación sobre balance de clases
            class_counts = df[target_col].value_counts()
            balance_ratio = class_counts.max() / class_counts.min()
            if balance_ratio > 3:
                st.warning(
                    f"⚠️ **Dataset desbalanceado**: La clase más frecuente tiene {balance_ratio:.1f}x más ejemplos que la menos frecuente")
                st.info(
                    "💡 Los modelos de IA funcionan mejor con clases balanceadas. Considera técnicas de balanceo si es necesario.")
            else:
                st.success(
                    "✅ **Dataset bien balanceado**: Las clases tienen cantidad similar de ejemplos")
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
        code = SCATTERPLOT_MATRIX

        show_code_with_download(
            code, "Código para análisis exploratorio", "analisis_exploratorio.py")

    except Exception as e:
        st.error(f"Error al cargar el dataset: {str(e)}")
        st.info(
            "Por favor, selecciona un dataset válido para continuar con la exploración.")
